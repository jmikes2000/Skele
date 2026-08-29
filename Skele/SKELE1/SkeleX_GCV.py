# SkeleX_GCV.py  –  Gtuner IV Computer Vision script (standalone SkeleX)
# YOLOv8 pose + ball detection + head-line trigger.
# Settings read live from settings.json (edit via main.py).
#
# Improvements over baseline:
#  1.  Kalman filter on bbox (constant-velocity model)
#  2.  Combined IoU + centre-distance track matching
#  3.  Ghost frames with Kalman prediction + confidence decay (2k_Vision style)
#  4.  Adaptive EMA smoothing based on motion magnitude
#  5.  Shooter ID locking with hysteresis
#  6.  Leaky trigger accumulator (replaces hard frame count)
#  7.  Wrist trajectory prediction — fire before crossing (latency compensation)
#  8.  Zone-aware head-line offset
#  9.  Keypoint confidence-weighted positions
# 10.  Shot phase state machine  (IDLE→LOAD→RISING→RELEASE_WINDOW→COOLDOWN)
# 11.  Contested shot detection
# 12.  Per-session zone auto-calibration
# 13.  Temporal keypoint de-ghosting

import os, json, time
import cv2
import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(_DIR, "_torch_cache"))
os.environ.setdefault("TORCH_HOME",              os.path.join(_DIR, "_torch_home"))
os.environ.setdefault("YOLO_CONFIG_DIR",         os.path.join(_DIR, "_yolo_cfg"))
_SETTINGS = os.path.join(_DIR, "settings.json")

_IMGSZ      = 832
_BALL_CLASS = 32   # COCO "sports ball"

# ── COCO keypoint indices ────────────────────────────────────────────────────
_NOSE          = 0
_L_EAR, _R_EAR = 3, 4
_L_SH,  _R_SH  = 5, 6
_L_EL,  _R_EL  = 7, 8
_L_WR,  _R_WR  = 9, 10
_L_HIP, _R_HIP = 11, 12
_L_KN,  _R_KN  = 13, 14
_L_AN,  _R_AN  = 15, 16

# ── Shot phase state machine ─────────────────────────────────────────────────
PHASE_IDLE           = 0
PHASE_LOADING        = 1
PHASE_RISING         = 2
PHASE_RELEASE_WINDOW = 3
PHASE_COOLDOWN       = 4
_PHASE_NAMES = {0:"IDLE", 1:"LOAD", 2:"RISING", 3:"RELEASE", 4:"COOL"}

# ── Court zones ──────────────────────────────────────────────────────────────
_ZONES = ["left_corner","left_wing","center","right_wing","right_corner"]

# ── Drawing ──────────────────────────────────────────────────────────────────
_BONES = [
    (_L_SH, _R_SH),
    (_L_SH, _L_EL),(_L_EL, _L_WR),
    (_R_SH, _R_EL),(_R_EL, _R_WR),
    (_L_SH, _L_HIP),(_R_SH, _R_HIP),
    (_L_HIP, _R_HIP),
    (_L_HIP, _L_KN),(_L_KN, _L_AN),
    (_R_HIP, _R_KN),(_R_KN, _R_AN),
]
_HEAD_KPS = {0,1,2,3,4}

_DEFAULTS = {
    "ball_conf":0.25, "ball_nms_iou":0.45,
    "pose_conf":0.30, "keypoint_conf":0.25, "smoothing_ema":0.35,
    "show_pose_skeleton":True,
    "trigger_mode":"head_line",
    "elbow_angle_threshold":155.0, "ball_release_threshold":0.50,
    "head_line_enabled":True, "show_head_line":False,
    "shooting_hand":"right", "head_line_offset_px":0,
    "hold_frames":0, "min_wrist_rise_px":0,
    "foot_lift_enabled":False, "foot_lift_px":12,
    "vy_smoothing_alpha":0.55, "max_shot_hold_ms":800, "apex_holdoff_ms":20,
    "wrist_release_enabled":True, "wrist_conf_min":0.40,
    "wrist_elev_frames":2, "wrist_only_above_px":15,
    "show_roi_box":False, "roi_padding_px":0,
    "rhythm_ms_phase1":0,
    # ── New ──
    "ghost_frames_max":12,
    "track_conf_decay":0.10,
    "track_conf_min":0.50,
    "shooter_lock_frames":10,
    "trigger_decay_rate":0.50,
    "prediction_latency_ms":50.0,
    "contested_extra_offset_px":10,
    "zone_calibration_enabled":True,
    # Kalman noise tuning (higher Q = follows faster; higher R = smoother)
    "kalman_process_noise":0.01,
    "kalman_measure_noise":10.0,
}


# ── Helpers ──────────────────────────────────────────────────────────────────
def _load_settings():
    for _ in range(3):
        try:
            with open(_SETTINGS,"r",encoding="utf-8") as f:
                s = f.read().strip()
            if s:
                d = json.loads(s)
                if isinstance(d,dict):
                    if "smooth"  in d and "smoothing_ema" not in d: d["smoothing_ema"]=d["smooth"]
                    if "kp_conf" in d and "keypoint_conf" not in d: d["keypoint_conf"]=d["kp_conf"]
                    for k,v in _DEFAULTS.items(): d.setdefault(k,v)
                    return d
        except Exception: pass
        time.sleep(0.02)
    return dict(_DEFAULTS)

def _calculate_angle(a,b,c):
    ba=a-b; bc=c-b
    cos=np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)+1e-6)
    return float(np.degrees(np.arccos(np.clip(cos,-1.0,1.0))))

def _iou(a,b):
    ix1=max(a[0],b[0]); iy1=max(a[1],b[1])
    ix2=min(a[2],b[2]); iy2=min(a[3],b[3])
    inter=max(0,ix2-ix1)*max(0,iy2-iy1)
    if inter==0: return 0.0
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua>0 else 0.0

def _match_score(trk_box, det_box, frame_w, frame_h):
    """Combined IoU + normalised-centre-distance score.
    Handles fast lateral movement where IoU alone drops below threshold."""
    iou  = _iou(trk_box, det_box)
    diag = (frame_w**2+frame_h**2)**0.5
    tcx=(trk_box[0]+trk_box[2])/2; tcy=(trk_box[1]+trk_box[3])/2
    dcx=(det_box[0]+det_box[2])/2; dcy=(det_box[1]+det_box[3])/2
    dist=((tcx-dcx)**2+(tcy-dcy)**2)**0.5
    norm_dist=min(1.0, dist/(diag*0.30))
    return 0.50*iou + 0.50*(1.0-norm_dist)

def _box_to_kf_meas(box):
    """[x1,y1,x2,y2] → column vector [cx,cy,w,h]."""
    cx=(box[0]+box[2])/2; cy=(box[1]+box[3])/2
    w=box[2]-box[0];      h=box[3]-box[1]
    return np.array([[cx],[cy],[w],[h]],dtype=np.float32)

def _kf_state_to_box(state):
    """Kalman state [cx,cy,w,h,vx,vy] → [x1,y1,x2,y2]."""
    cx,cy,w,h = float(state[0]),float(state[1]),float(state[2]),float(state[3])
    w=max(1.0,w); h=max(1.0,h)
    return np.array([cx-w/2, cy-h/2, cx+w/2, cy+h/2], dtype=np.float32)

def _make_kalman(box, proc_noise=0.01, meas_noise=10.0):
    """Create and initialise a 6-state / 4-measurement Kalman filter.
    State vector: [cx, cy, w, h, vx, vy]
    """
    kf = cv2.KalmanFilter(6,4)
    # Constant-velocity transition
    kf.transitionMatrix = np.array([
        [1,0,0,0,1,0],
        [0,1,0,0,0,1],
        [0,0,1,0,0,0],
        [0,0,0,1,0,0],
        [0,0,0,0,1,0],
        [0,0,0,0,0,1],
    ],dtype=np.float32)
    # Observe cx,cy,w,h
    kf.measurementMatrix = np.array([
        [1,0,0,0,0,0],
        [0,1,0,0,0,0],
        [0,0,1,0,0,0],
        [0,0,0,1,0,0],
    ],dtype=np.float32)
    kf.processNoiseCov  = np.eye(6,dtype=np.float32)*proc_noise
    kf.measurementNoiseCov = np.eye(4,dtype=np.float32)*meas_noise
    kf.errorCovPost     = np.eye(6,dtype=np.float32)
    # Seed state from first detection
    m = _box_to_kf_meas(box)
    kf.statePost = np.array(
        [[m[0,0]],[m[1,0]],[m[2,0]],[m[3,0]],[0.0],[0.0]], dtype=np.float32)
    return kf


# ── Per-person track ─────────────────────────────────────────────────────────
class _Track:
    _next_id = 0

    def __init__(self, kp, box, conf, smooth,
                 proc_noise=0.01, meas_noise=10.0):
        self.id   = _Track._next_id; _Track._next_id += 1
        self.kp   = kp.copy()
        self.conf = conf
        self.age  = 0
        self._above_count = 0
        self._wrist_y_hist: list = []

        # ── Kalman filter for bbox (item 1) ───────────────────────────────
        self._kf  = _make_kalman(box, proc_noise, meas_noise)
        self.box  = box.copy()   # always reflects latest KF output

        # Foot / ankle tracking
        self._ankle_baseline_y    = -1.0
        self._feet_on_ground      = True
        self._feet_left_ground_frame = 0
        self.frame_count          = 0

        # Jump velocity
        self._prev_hip_cy    = -1.0
        self._prev_yolo_cy   = -1.0
        self._vy_smooth      = 0.0
        self._vy_prev_smooth = 0.0

        # Shot tracking (legacy — used by elbow / ball / apex modes)
        self._shot_tracking    = False
        self._shot_track_start = 0.0
        self._wrist_above_frames = 0
        self._decel_entered    = False
        self._decel_enter_time = 0.0

        # ── Confidence decay (2k_Vision style) ───────────────────────────
        self._confidence_score = 1.0
        self._missed_frames    = 0

        # ── Shot phase state machine ──────────────────────────────────────
        self._shot_phase       = PHASE_IDLE
        self._phase_entry_time = 0.0

        # ── Leaky trigger accumulator ─────────────────────────────────────
        self._trigger_accum = 0.0

        # ── Wrist trajectory for prediction ──────────────────────────────
        self._wrist_y_traj: list = []   # (perf_counter, wrist_y)

        # ── Court zone + per-zone calibration ────────────────────────────
        self._player_zone = "center"

        # Keypoint confidence-weighted last positions
        self._wrist_y_last: float = -1.0
        self._sh_y_last:    float = -1.0

    # ── Detection matched — Kalman correct + EMA keypoints ───────────────
    def update(self, kp, box, conf, base_smooth, frame_w=1920,
               proc_noise=0.01, meas_noise=10.0):
        # Update KF noise in case settings changed
        self._kf.processNoiseCov    = np.eye(6,dtype=np.float32)*proc_noise
        self._kf.measurementNoiseCov = np.eye(4,dtype=np.float32)*meas_noise

        # Adaptive smoothing: fast motion → follow quickly; stationary → stable
        prev_cx=(self.box[0]+self.box[2])/2; prev_cy=(self.box[1]+self.box[3])/2
        new_cx=(box[0]+box[2])/2;            new_cy=(box[1]+box[3])/2
        motion=((new_cx-prev_cx)**2+(new_cy-prev_cy)**2)**0.5
        adaptive_smooth = max(0.05, base_smooth - min(0.50, motion/100.0))

        # Kalman predict → correct
        self._kf.predict()
        self._kf.correct(_box_to_kf_meas(box))
        self.box = _kf_state_to_box(self._kf.statePost)

        # EMA keypoints with temporal de-ghosting (item 13)
        a = 1.0 - adaptive_smooth
        new_kp = kp.copy()
        for i in range(min(len(kp), len(self.kp))):
            nc = float(kp[i,2]); oc = float(self.kp[i,2])
            if nc < 0.30 and oc > 0.30:
                # Brief occlusion — blend heavily toward previous position
                blend = max(0.10, nc)
                new_kp[i,0] = blend*kp[i,0]+(1.0-blend)*self.kp[i,0]
                new_kp[i,1] = blend*kp[i,1]+(1.0-blend)*self.kp[i,1]
                new_kp[i,2] = oc*0.70
            else:
                new_kp[i,0] = a*kp[i,0]+(1.0-a)*self.kp[i,0]
                new_kp[i,1] = a*kp[i,1]+(1.0-a)*self.kp[i,1]
                new_kp[i,2] = kp[i,2]
        self.kp   = new_kp
        self.conf = conf
        self.age  = 0
        self._missed_frames    = 0
        self._confidence_score = 1.0
        self._player_zone = self._compute_zone(frame_w)

    # ── No match — Kalman predict only + confidence decay ─────────────────
    def miss_frame(self, conf_decay=0.10, conf_min=0.50):
        self._missed_frames += 1
        self.age += 1
        # 2k_Vision confidence decay
        self._confidence_score = max(conf_min-0.01, self._confidence_score-conf_decay)
        # Kalman extrapolation (no correction step)
        self._kf.predict()
        self.box = _kf_state_to_box(self._kf.statePost)
        # Decay keypoint certainty — don't move them
        self.kp[:,2] *= 0.85

    def _compute_zone(self, frame_w):
        cx  = (self.box[0]+self.box[2])/2.0
        rel = cx/max(1, frame_w)
        if   rel<0.15: return "left_corner"
        elif rel<0.35: return "left_wing"
        elif rel<0.65: return "center"
        elif rel<0.85: return "right_wing"
        else:          return "right_corner"

    def update_foot_tracking(self, foot_lift_px):
        la=self.kp[15]; ra=self.kp[16]
        visible=[]
        if la[2]>0.25: visible.append(la[1])
        if ra[2]>0.25: visible.append(ra[1])
        if not visible: return
        ankle_y=max(visible)
        if self._ankle_baseline_y<0:
            self._ankle_baseline_y=ankle_y
        elif self._feet_on_ground:
            self._ankle_baseline_y+=0.002*(ankle_y-self._ankle_baseline_y)
        if self._ankle_baseline_y>0:
            lift=self._ankle_baseline_y-ankle_y
            was=self._feet_on_ground
            self._feet_on_ground=lift<float(foot_lift_px)
            if was and not self._feet_on_ground:
                self._feet_left_ground_frame=self.frame_count

    def update_jump_velocity(self, alpha):
        self.frame_count+=1
        lh=self.kp[11]; rh=self.kp[12]
        hip_cy=(lh[1]+rh[1])/2.0 if (lh[2]>0.35 and rh[2]>0.35) else -1.0
        yolo_cy=(self.box[1]+self.box[3])/2.0
        raw=0.0
        if hip_cy>0 and self._prev_hip_cy>0: raw=hip_cy-self._prev_hip_cy
        elif yolo_cy>0 and self._prev_yolo_cy>0: raw=yolo_cy-self._prev_yolo_cy
        self._vy_prev_smooth=self._vy_smooth
        self._vy_smooth=(1.0-alpha)*self._vy_smooth+alpha*raw
        self._prev_hip_cy=hip_cy; self._prev_yolo_cy=yolo_cy


# ── Skeleton draw ────────────────────────────────────────────────────────────
def _draw_skeleton(frame, track, kp_conf):
    kp=track.kp
    def pt(i): return int(kp[i,0]),int(kp[i,1]),float(kp[i,2])
    for a,b in _BONES:
        ax,ay,ac=pt(a); bx,by,bc=pt(b)
        if ac>=kp_conf and bc>=kp_conf:
            cv2.line(frame,(ax,ay),(bx,by),(255,255,255),1,cv2.LINE_AA)
    sq=3
    for i in range(17):
        if i in _HEAD_KPS: continue
        px,py,pc=pt(i)
        if pc>=kp_conf:
            cv2.rectangle(frame,(px-sq,py-sq),(px+sq,py+sq),(0,255,0),-1,cv2.LINE_AA)


# ── GCVWorker ────────────────────────────────────────────────────────────────
class GCVWorker:

    def __init__(self, width=640, height=480):
        self.gcvdata          = bytearray(32)
        self.width, self.height = width, height

        from ultralytics import YOLO
        for model_name in ("yolov8m-pose","yolov8n-pose"):
            pt  = os.path.join(_DIR, f"{model_name}.pt")
            trt = os.path.join(_DIR, f"{model_name}-{_IMGSZ}.engine")
            loaded=False
            if os.path.exists(trt):
                try:
                    print(f"[SkeleX] Loading {model_name} TensorRT engine…")
                    self._pose_model=YOLO(trt)
                    dummy=np.zeros((_IMGSZ*9//16,_IMGSZ,3),dtype=np.uint8)
                    self._pose_model(dummy,verbose=False,imgsz=_IMGSZ)
                    self._model_name=model_name
                    print(f"[SkeleX] TensorRT engine loaded and verified.")
                    loaded=True
                except Exception as e:
                    print(f"[SkeleX] Incompatible engine: {e}")
                    try: os.remove(trt)
                    except: pass
            if not loaded and os.path.exists(pt):
                try:
                    import tensorrt
                    print(f"[SkeleX] Exporting {model_name} to TensorRT (~60s)…")
                    YOLO(pt).export(format="engine",device="cuda",imgsz=_IMGSZ,half=True,simplify=True)
                    eng=os.path.join(_DIR,f"{model_name}.engine")
                    if os.path.exists(eng): os.rename(eng,trt)
                    self._pose_model=YOLO(trt)
                    dummy=np.zeros((_IMGSZ*9//16,_IMGSZ,3),dtype=np.uint8)
                    self._pose_model(dummy,verbose=False,imgsz=_IMGSZ)
                    self._model_name=model_name; loaded=True
                    print(f"[SkeleX] TensorRT ready.")
                except Exception as e:
                    print(f"[SkeleX] TRT failed ({e}), using PyTorch.")
                    try:
                        if os.path.exists(trt): os.remove(trt)
                    except: pass
                    self._pose_model=YOLO(pt)
                    dummy=np.zeros((_IMGSZ*9//16,_IMGSZ,3),dtype=np.uint8)
                    self._pose_model(dummy,verbose=False,imgsz=_IMGSZ)
                    self._model_name=model_name; loaded=True
            if loaded: break
        else:
            raise FileNotFoundError(
                f"No pose model found. Download yolov8m-pose.pt to:\n{_DIR}\n"
                "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m-pose.pt")

        self._imgsz=_IMGSZ
        print("[SkeleX] Ready.")

        self._tracks: list = []
        self._mtime  = self._get_mtime()

        # Ball
        self._prev_ball_cx=None; self._prev_ball_cy=None
        self._ball_held_frames=0; self._prev_ball_intersecting=False
        self._prev_gap=None

        # Diagnostics
        self._current_elbow_angle=0.0; self._current_ball_score=0.0

        # Shooter ID locking
        self._shooter_track_id   = None
        self._shooter_lost_frames = 0

        # Global zone calibration (survives track resets)
        self._zone_offsets: dict = {z:0.0 for z in _ZONES}
        self._zone_history: dict = {z:[]  for z in _ZONES}

        self._apply(_load_settings())

        try:
            import subprocess, sys
            creationflags=0x08000000 if sys.platform=="win32" else 0
            subprocess.Popen([sys.executable,os.path.join(_DIR,"main.py")],
                             creationflags=creationflags)
            print("[SkeleX] Settings GUI spawned.")
        except Exception as e:
            print(f"[SkeleX] GUI spawn failed: {e}")

    def _get_mtime(self):
        try: return os.stat(_SETTINGS).st_mtime_ns
        except: return 0

    def _apply(self, d):
        self.ball_conf             = max(0.05,min(0.95,float(d.get("ball_conf",0.25))))
        self.ball_nms_iou          = max(0.05,min(0.95,float(d.get("ball_nms_iou",0.45))))
        self.pose_conf             = max(0.05,min(0.99,float(d.get("pose_conf",0.30))))
        self.kp_conf               = max(0.05,min(0.99,float(d.get("keypoint_conf",0.25))))
        self.smooth                = max(0.0, min(0.99,float(d.get("smoothing_ema",0.35))))
        self.show_pose_skeleton    = bool(d.get("show_pose_skeleton",True))
        self.trigger_mode          = str(d.get("trigger_mode","head_line")).lower()
        if self.trigger_mode not in ("head_line","elbow_angle","ball_release","apex_decel"):
            self.trigger_mode="head_line"
        self.elbow_angle_threshold  = max(120.0,min(180.0,float(d.get("elbow_angle_threshold",155.0))))
        self.ball_release_threshold = max(0.30, min(0.90, float(d.get("ball_release_threshold",0.50))))
        self.head_line_enabled     = bool(d.get("head_line_enabled",True))
        self.show_head_line        = bool(d.get("show_head_line",False))
        self.shooting_hand         = str(d.get("shooting_hand","right")).lower()
        if self.shooting_hand not in ("right","left","auto"): self.shooting_hand="right"
        self.head_line_offset_px   = int(d.get("head_line_offset_px",0))
        self.hold_frames           = max(0,min(30,int(d.get("hold_frames",0))))
        self.min_wrist_rise_px     = max(0.0,min(30.0,float(d.get("min_wrist_rise_px",0))))
        self.show_roi_box          = bool(d.get("show_roi_box",False))
        self.roi_padding_px        = max(0,min(400,int(d.get("roi_padding_px",0))))
        self.rhythm_ms_phase1      = max(0,min(255,int(d.get("rhythm_ms_phase1",0))))
        self.foot_lift_enabled     = bool(d.get("foot_lift_enabled",False))
        self.foot_lift_px          = max(1,min(100,int(d.get("foot_lift_px",12))))
        self.vy_smoothing_alpha    = max(0.01,min(0.99,float(d.get("vy_smoothing_alpha",0.55))))
        self.max_shot_hold_ms      = max(100,min(3000,int(d.get("max_shot_hold_ms",800))))
        self.apex_holdoff_ms       = max(0,min(1000,int(d.get("apex_holdoff_ms",20))))
        self.wrist_release_enabled = bool(d.get("wrist_release_enabled",True))
        self.wrist_conf_min        = max(0.05,min(0.99,float(d.get("wrist_conf_min",0.40))))
        self.wrist_elev_frames     = max(1,min(30,int(d.get("wrist_elev_frames",2))))
        self.wrist_only_above_px   = max(1,min(100,int(d.get("wrist_only_above_px",15))))
        # New
        self.ghost_frames_max          = max(4, min(30, int(d.get("ghost_frames_max",12))))
        self.track_conf_decay          = max(0.01,min(0.5,float(d.get("track_conf_decay",0.10))))
        self.track_conf_min            = max(0.20,min(0.90,float(d.get("track_conf_min",0.50))))
        self.shooter_lock_frames       = max(1, min(30, int(d.get("shooter_lock_frames",10))))
        self.trigger_decay_rate        = max(0.10,min(0.99,float(d.get("trigger_decay_rate",0.50))))
        self.prediction_latency_ms     = max(0.0,min(200.0,float(d.get("prediction_latency_ms",50.0))))
        self.contested_extra_offset_px = max(0, min(50, int(d.get("contested_extra_offset_px",10))))
        self.zone_calibration_enabled  = bool(d.get("zone_calibration_enabled",True))
        self.kalman_process_noise      = max(1e-6,float(d.get("kalman_process_noise",0.01)))
        self.kalman_measure_noise      = max(0.01,float(d.get("kalman_measure_noise",10.0)))
        print(f"[SkeleX] mode={self.trigger_mode} pose={self.pose_conf:.2f} "
              f"smooth={self.smooth:.2f} hand={self.shooting_hand} "
              f"head_off={self.head_line_offset_px}px hold={self.hold_frames}f "
              f"ghost={self.ghost_frames_max}f KF(Q={self.kalman_process_noise} R={self.kalman_measure_noise})")

    def _maybe_reload(self):
        t=self._get_mtime()
        if t!=self._mtime: self._mtime=t; self._apply(_load_settings())

    # ── Shooter ID locking + hysteresis ──────────────────────────────────
    def _pick_shooter(self):
        hand=self.shooting_hand
        def _valid(t):
            if t.kp.shape[0]<=16: return False
            if float(t.kp[0,2])<0.10: return False
            if t._confidence_score<self.track_conf_min: return False
            if hand=="auto":
                return ((float(t.kp[9,2])>=0.15 and float(t.kp[5,2])>=0.15) or
                        (float(t.kp[10,2])>=0.15 and float(t.kp[6,2])>=0.15))
            wi=10 if hand=="right" else 9; si=6 if hand=="right" else 5
            return min(float(t.kp[wi,2]),float(t.kp[si,2]))>=0.15
        def _area(t): return (t.box[2]-t.box[0])*(t.box[3]-t.box[1])

        # Try locked shooter first
        if self._shooter_track_id is not None:
            locked=next((t for t in self._tracks if t.id==self._shooter_track_id),None)
            if locked is not None and _valid(locked):
                self._shooter_lost_frames=0; return locked
            self._shooter_lost_frames+=1
            if self._shooter_lost_frames<self.shooter_lock_frames: return None
            self._shooter_track_id=None   # unlock

        # Pick best candidate
        cands=[t for t in self._tracks if _valid(t)]
        if not cands: return None
        best=max(cands,key=_area)

        # Hysteresis: only switch if >40% larger
        if self._shooter_track_id is not None:
            old=next((t for t in self._tracks if t.id==self._shooter_track_id),None)
            if old is not None and _valid(old) and _area(best)<_area(old)*1.40:
                return old

        self._shooter_track_id=best.id; self._shooter_lost_frames=0
        return best

    # ── Contested detection ───────────────────────────────────────────────
    def _is_contested(self, shooter):
        if shooter is None or len(self._tracks)<2: return False
        scx=(shooter.box[0]+shooter.box[2])/2; scy=(shooter.box[1]+shooter.box[3])/2
        sh=(shooter.box[3]-shooter.box[1])
        for t in self._tracks:
            if t.id==shooter.id or t._confidence_score<self.track_conf_min: continue
            tcx=(t.box[0]+t.box[2])/2; tcy=(t.box[1]+t.box[3])/2
            if ((scx-tcx)**2+(scy-tcy)**2)**0.5 < sh*1.0: return True
        return False

    # ── Shot phase state machine ──────────────────────────────────────────
    def _update_shot_phase(self, shooter, now):
        phase=shooter._shot_phase
        elapsed=(now-shooter._phase_entry_time)*1000.0
        vy=shooter._vy_smooth

        # Use foot lift when enabled; otherwise require sustained upward velocity.
        # Threshold -0.55 (was -0.30) to avoid false triggers during running.
        if self.foot_lift_enabled:
            airborne=not shooter._feet_on_ground
        else:
            # Must have 2 consecutive frames of strong upward motion to be a real jump
            hist=getattr(shooter,'_vy_hist',[])
            hist.append(vy)
            if len(hist)>3: hist.pop(0)
            shooter._vy_hist=hist
            # Sustained negative vy (upward on screen) required
            airborne=(len(hist)>=2 and all(v<-0.55 for v in hist[-2:]))

        if phase==PHASE_IDLE:
            if airborne:
                shooter._shot_phase=PHASE_RISING; shooter._phase_entry_time=now
                # ── Set legacy fields for apex_decel mode ────────────────
                shooter._shot_tracking=True; shooter._shot_track_start=now
                shooter._wrist_above_frames=0; shooter._decel_entered=False
            elif vy>0.30:
                shooter._shot_phase=PHASE_LOADING; shooter._phase_entry_time=now
        elif phase==PHASE_LOADING:
            if airborne:
                shooter._shot_phase=PHASE_RISING; shooter._phase_entry_time=now
                shooter._shot_tracking=True; shooter._shot_track_start=now
                shooter._wrist_above_frames=0; shooter._decel_entered=False
            elif elapsed>1200: shooter._shot_phase=PHASE_IDLE
        elif phase==PHASE_RISING:
            # Land check: positive vy + elapsed>100ms means they landed without shooting
            if not airborne and vy>=0.10 and elapsed>100:
                shooter._shot_phase=PHASE_IDLE; shooter._phase_entry_time=now
                shooter._shot_tracking=False; shooter._shot_track_start=0.0
            elif elapsed>float(self.max_shot_hold_ms):
                shooter._shot_phase=PHASE_COOLDOWN; shooter._phase_entry_time=now
                shooter._shot_tracking=False
        elif phase==PHASE_COOLDOWN:
            if elapsed>700:   # 700ms cooldown prevents re-trigger on same shot
                shooter._shot_phase=PHASE_IDLE; shooter._trigger_accum=0.0
                shooter._shot_tracking=False
        return shooter._shot_phase

    # ── Head-line trigger (items 6-12) ────────────────────────────────────
    def _resolve_wrist_idx(self, shooter):
        hand=self.shooting_hand
        if hand=="auto":
            lw=shooter.kp[9]; ls=shooter.kp[5]
            rw=shooter.kp[10]; rs=shooter.kp[6]
            l_rise=ls[1]-lw[1] if (lw[2]>=0.15 and ls[2]>=0.15) else -9999.0
            r_rise=rs[1]-rw[1] if (rw[2]>=0.15 and rs[2]>=0.15) else -9999.0
            if r_rise>=l_rise: self.resolved_hand="right"; return 10
            else:              self.resolved_hand="left";  return 9
        self.resolved_hand=hand
        return 10 if hand=="right" else 9

    def _eval_head_line(self, shooter, phase, now):
        kp=shooter.kp
        wrist_idx=self._resolve_wrist_idx(shooter)
        sh_idx=6 if self.resolved_hand=="right" else 5

        # ── Keypoint confidence-weighted positions (item 9) ──────────────
        wc=float(kp[wrist_idx,2]); sc=float(kp[sh_idx,2])
        raw_wy=float(kp[wrist_idx,1]); raw_sy=float(kp[sh_idx,1])
        if shooter._wrist_y_last<0: shooter._wrist_y_last=raw_wy
        if shooter._sh_y_last<0:    shooter._sh_y_last=raw_sy
        wrist_y=wc*raw_wy+(1.0-wc)*shooter._wrist_y_last
        sh_y   =sc*raw_sy+(1.0-sc)*shooter._sh_y_last
        shooter._wrist_y_last=wrist_y; shooter._sh_y_last=sh_y

        # Zoom scale + speed offset
        sh_h=max(10.0,float(shooter.box[3]-shooter.box[1]))
        scale=sh_h/200.0
        hist=shooter._wrist_y_hist; hist.append(wrist_y)
        if len(hist)>6: hist.pop(0)
        vel=(hist[-2]-wrist_y) if len(hist)>=3 else 0.0
        speed_off=max(0.0,(vel/scale-6.0)*1.5)

        # ── Zone offset + contested adjustment (items 8 & 11) ────────────
        zone=shooter._player_zone
        zone_adj=self._zone_offsets.get(zone,0.0)
        contested=self._is_contested(shooter)
        contest_adj=self.contested_extra_offset_px if contested else 0

        head_line_y=(sh_y
                     -(float(self.head_line_offset_px)+speed_off)*scale
                     -zone_adj
                     -contest_adj)

        above_raw=wrist_y<head_line_y

        # ── Wrist trajectory prediction (item 7) ─────────────────────────
        traj=shooter._wrist_y_traj; traj.append((now,wrist_y))
        if len(traj)>8: traj.pop(0)
        above=above_raw
        if not above_raw and self.prediction_latency_ms>0 and len(traj)>=4:
            pts=traj[-4:]; t0=pts[0][0]
            times=np.array([p[0]-t0 for p in pts],dtype=np.float64)
            ys=np.array([p[1]       for p in pts],dtype=np.float64)
            if times[-1]>1e-6:
                slope=(ys[-1]-ys[0])/times[-1]
                future_y=wrist_y+slope*(self.prediction_latency_ms/1000.0)
                if future_y<head_line_y: above=True

        # ── Leaky accumulator (item 6) ────────────────────────────────────
        if above:
            shooter._trigger_accum=min(float(self.hold_frames)+2.0,
                                       shooter._trigger_accum+1.0)
        else:
            shooter._trigger_accum=max(0.0,
                                       shooter._trigger_accum-self.trigger_decay_rate)

        threshold=max(1.0,float(self.hold_frames))
        fire=shooter._trigger_accum>=threshold

        # ── Phase gate (item 10) ──────────────────────────────────────────
        if phase not in (PHASE_RISING,PHASE_RELEASE_WINDOW):
            fire=False
        elif fire:
            shooter._shot_phase=PHASE_COOLDOWN
            shooter._phase_entry_time=now
            shooter._trigger_accum=0.0
            # ── Zone auto-calibration (item 12) ──────────────────────────
            if self.zone_calibration_enabled:
                delta=wrist_y-head_line_y   # +early / -late
                zh=self._zone_history.setdefault(zone,[])
                zh.append(delta)
                if len(zh)>10: zh.pop(0)
                if len(zh)>=5:
                    avg=sum(zh[-5:])/5.0
                    self._zone_offsets[zone]=max(-60.0,min(60.0,
                        self._zone_offsets.get(zone,0.0)+avg*0.10))

        # Min wrist rise gate
        if fire and self.min_wrist_rise_px>0 and len(hist)>=3:
            if (hist[-3]-wrist_y)<(self.min_wrist_rise_px*scale):
                fire=False

        return (1,int(self.rhythm_ms_phase1)) if fire else (0,0)

    # ── Other trigger modes (unchanged logic, kept for compatibility) ─────
    def _get_shooting_wrist(self, shooter):
        hand=getattr(self,"resolved_hand",self.shooting_hand)
        lw=shooter.kp[9]; rw=shooter.kp[10]
        ls=shooter.kp[5]; rs=shooter.kp[6]
        px=float(self.wrist_only_above_px)
        if hand=="left":
            return lw if (lw[2]>=self.wrist_conf_min and ls[2]>=0.30 and (ls[1]-lw[1])>=px) else None
        return rw if (rw[2]>=self.wrist_conf_min and rs[2]>=0.30 and (rs[1]-rw[1])>=px) else None

    def _check_wrist_release(self, shooter, now):
        if not self.wrist_release_enabled: return False
        if shooter._shot_track_start<=0: return False
        if (now-shooter._shot_track_start)*1000.0<float(self.apex_holdoff_ms): return False
        if self._get_shooting_wrist(shooter) is not None:
            shooter._wrist_above_frames+=1
        else:
            shooter._wrist_above_frames=max(0,shooter._wrist_above_frames-1)
        return shooter._wrist_above_frames>=int(self.wrist_elev_frames)

    def _eval_apex_decel(self, shooter, now):
        if not shooter._shot_tracking: return 0,0
        el=(now-shooter._shot_track_start)*1000.0
        # Require at least 60ms airborne before any trigger — prevents
        # false fires from brief velocity spikes during dribbling.
        if el < 60.0: return 0,0
        if el>float(self.max_shot_hold_ms): return 1,int(self.rhythm_ms_phase1)
        if self._check_wrist_release(shooter,now): return 1,int(self.rhythm_ms_phase1)
        t_s=(self.ball_release_threshold-0.30)/0.60
        delay=60.0+t_s*340.0
        if el>delay*0.6: shooter._decel_entered=True; shooter._decel_enter_time=now
        return (1,int(self.rhythm_ms_phase1)) if el>=delay else (0,0)

    def _eval_elbow_angle(self, shooter, speed_off):
        kp=shooter.kp
        hand=getattr(self,"resolved_hand",self.shooting_hand)
        si=6 if hand=="right" else 5
        ei=8 if hand=="right" else 7
        wi=10 if hand=="right" else 9
        if kp.shape[0]<=max(si,ei,wi): return 0,0
        if min(float(kp[si,2]),float(kp[ei,2]),float(kp[wi,2]))<self.kp_conf: return 0,0
        angle=_calculate_angle(kp[si,:2],kp[ei,:2],kp[wi,:2])
        self._current_elbow_angle=angle
        return (1,int(self.rhythm_ms_phase1)) if angle>=self.elbow_angle_threshold-speed_off else (0,0)

    def _eval_ball_release(self, shooter, balls, scale, speed_off):
        if not balls:
            return self._eval_elbow_angle(shooter,speed_off)
        best=max(balls,key=lambda x:x[0])
        if best[0]<self.ball_conf:
            self._prev_ball_cx=None; self._prev_ball_cy=None
            self._prev_ball_intersecting=False
            return self._eval_elbow_angle(shooter,speed_off)
        _,bbox=best; bx1,by1,bx2,by2=bbox
        cx=(bx1+bx2)/2; cy=(by1+by2)/2
        px1,py1,px2,py2=shooter.box
        isect=(max(bx1,px1)<min(bx2,px2)) and (max(by1,py1)<min(by2,py2))
        bvy=((cy-self._prev_ball_cy)/max(0.1,scale)) if self._prev_ball_cy is not None else 0.0
        hand=getattr(self,"resolved_hand",self.shooting_hand)
        wi=10 if hand=="right" else 9
        wx,wy=shooter.kp[wi,0],shooter.kp[wi,1]
        above=cy<wy
        gap=float(((cx-wx)**2+(cy-wy)**2)**0.5)
        gv=((gap-self._prev_gap)/max(0.1,scale)) if self._prev_gap is not None else 0.0
        if isect: self._ball_held_frames+=1
        sep=self._prev_ball_intersecting and not isect
        sc=0.0
        if sep:       sc+=0.35
        if bvy<-8.0:  sc+=0.25*min(1.0,abs(bvy)/25.0)
        if above:     sc+=0.25
        if gv>3.0:    sc+=0.15*min(1.0,gv/12.0)
        self._current_ball_score=sc
        dyn=max(0.20,self.ball_release_threshold-speed_off*0.005)
        fire=sep and (self._ball_held_frames>=3) and (sc>=dyn)
        self._prev_ball_cx=cx; self._prev_ball_cy=cy
        self._prev_ball_intersecting=isect; self._prev_gap=gap
        if not isect and not sep: self._ball_held_frames=0
        return (1,int(self.rhythm_ms_phase1)) if fire else (0,0)

    # ── Trigger dispatcher ────────────────────────────────────────────────
    def _evaluate_trigger(self, shooter, balls):
        if shooter is None or not self.head_line_enabled: return 0,0
        shooter.update_foot_tracking(self.foot_lift_px)
        shooter.update_jump_velocity(self.vy_smoothing_alpha)
        now=time.perf_counter()
        phase=self._update_shot_phase(shooter,now)
        sh_h=max(10.0,float(shooter.box[3]-shooter.box[1])); scale=sh_h/200.0

        if self.trigger_mode=="head_line":
            return self._eval_head_line(shooter,phase,now)
        elif self.trigger_mode=="elbow_angle":
            return self._eval_elbow_angle(shooter,0.0)
        elif self.trigger_mode=="ball_release":
            return self._eval_ball_release(shooter,balls,scale,0.0)
        elif self.trigger_mode=="apex_decel":
            return self._eval_apex_decel(shooter,now)
        return 0,0

    # ── Overlays ──────────────────────────────────────────────────────────
    def _draw_overlays(self, frame, shooter, balls, contested=False):
        for conf,box in balls:
            x1,y1,x2,y2=box.astype(int)
            r=max(6,(x2-x1+y2-y1)//4)
            cv2.circle(frame,((x1+x2)//2,(y1+y2)//2),r,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(frame,f"ball {conf:.2f}",(x1,max(0,y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.35,(255,255,255),1,cv2.LINE_AA)
        if self.show_pose_skeleton:
            for t in self._tracks: _draw_skeleton(frame,t,self.kp_conf)
        if shooter is None: return
        kp=shooter.kp
        hand=getattr(self,"resolved_hand",self.shooting_hand)
        si=6 if hand=="right" else 5
        sx=int(kp[si,0]); sy=int(kp[si,1])
        off=int(self.head_line_offset_px); ly=max(0,sy-off)
        if self.trigger_mode=="head_line" and self.show_head_line:
            bw=max(40,int(shooter.box[2]-shooter.box[0])); half=int(bw*0.6)
            col=(0,80,255) if contested else (0,220,255)
            cv2.line(frame,(max(0,sx-half),ly),(min(frame.shape[1],sx+half),ly),col,1,cv2.LINE_AA)
            lbl="HEAD LINE [C]" if contested else "HEAD LINE"
            cv2.putText(frame,lbl,(sx+8,max(12,ly-4)),cv2.FONT_HERSHEY_SIMPLEX,0.35,col,1,cv2.LINE_AA)
        if self.show_roi_box:
            pad=int(self.roi_padding_px); x1,y1,x2,y2=shooter.box.astype(int)
            cv2.rectangle(frame,(max(0,x1-pad),max(0,y1-pad)),
                          (min(frame.shape[1],x2+pad),min(frame.shape[0],y2+pad)),(255,60,60),1,cv2.LINE_AA)
        # Debug overlay: phase + zone + accum + KF confidence
        pname=_PHASE_NAMES.get(shooter._shot_phase,"?")
        zadj=self._zone_offsets.get(shooter._player_zone,0.0)
        conf_s=f"{shooter._confidence_score:.2f}"
        px1=int(shooter.box[0]); py1=max(12,int(shooter.box[1])-30)
        cv2.putText(frame,
            f"[{pname}] acc={shooter._trigger_accum:.1f} "
            f"zone={shooter._player_zone}({zadj:+.0f}) KF={conf_s}",
            (px1,py1),cv2.FONT_HERSHEY_SIMPLEX,0.30,(200,200,60),1,cv2.LINE_AA)
        if self.trigger_mode=="elbow_angle":
            ei=8 if hand=="right" else 7
            if kp.shape[0]>ei:
                ex,ey=int(kp[ei,0]),int(kp[ei,1])
                cv2.putText(frame,f"elbow:{self._current_elbow_angle:.1f}/{self.elbow_angle_threshold:.1f}",
                            (ex+8,ey-4),cv2.FONT_HERSHEY_SIMPLEX,0.35,(0,255,255),1,cv2.LINE_AA)
        elif self.trigger_mode=="ball_release":
            cv2.putText(frame,f"ball:{self._current_ball_score:.2f}/{self.ball_release_threshold:.2f}",
                        (px1,max(12,int(shooter.box[1])-8)),cv2.FONT_HERSHEY_SIMPLEX,0.35,(0,255,120),1,cv2.LINE_AA)
        elif self.trigger_mode=="apex_decel":
            st=("DECEL" if shooter._decel_entered else
                f"JUMP(vy:{shooter._vy_smooth:.2f})" if shooter._shot_tracking else "GND")
            cv2.putText(frame,f"apex:{st}|w_elev:{shooter._wrist_above_frames}",
                        (px1,max(12,int(shooter.box[1])-8)),cv2.FONT_HERSHEY_SIMPLEX,0.35,(255,100,255),1,cv2.LINE_AA)

    # ── Main process loop ─────────────────────────────────────────────────
    def process(self, frame):
        self._maybe_reload()
        self.gcvdata[30]=0; self.gcvdata[31]=0
        if frame is None: return frame,self.gcvdata

        h,w=frame.shape[:2]
        small=cv2.resize(frame,(self._imgsz,self._imgsz*9//16))
        sx=w/self._imgsz; sy=h/(self._imgsz*9//16)

        results=self._pose_model(small,verbose=False,imgsz=self._imgsz,
                                  conf=self.ball_conf,classes=[0,_BALL_CLASS])
        people,balls=[],[]
        for r in results:
            if r.boxes is None or r.keypoints is None: continue
            for i in range(len(r.boxes)):
                cls=int(r.boxes.cls[i].item()); conf=float(r.boxes.conf[i].item())
                box=r.boxes.xyxy[i].cpu().numpy().copy()
                box[:2]*=(sx,sy); box[2:]*=(sx,sy)
                if cls==_BALL_CLASS:
                    balls.append((conf,box))
                elif cls==0:
                    kp=r.keypoints.data[i].cpu().numpy().copy()
                    kp[:,0]*=sx; kp[:,1]*=sy
                    people.append((conf,kp,box))

        # ── Combined IoU + centre-distance track matching (item 2) ────────
        matched_det,matched_trk=set(),set()
        for ti,trk in enumerate(self._tracks):
            best_s,best_di=0.0,-1
            for di,(conf,kp,box) in enumerate(people):
                if di in matched_det: continue
                s=_match_score(trk.box,box,w,h)
                if s>best_s: best_s,best_di=s,di
            if best_s>0.20 and best_di>=0:
                conf,kp,box=people[best_di]
                trk.update(kp,box,conf,self.smooth,w,
                           self.kalman_process_noise,self.kalman_measure_noise)
                matched_det.add(best_di); matched_trk.add(ti)
            else:
                # Ghost frame: Kalman predict + confidence decay (items 1,3)
                trk.miss_frame(self.track_conf_decay,self.track_conf_min)

        # New tracks for unmatched detections
        for di,(conf,kp,box) in enumerate(people):
            if di not in matched_det:
                self._tracks.append(_Track(kp,box,conf,self.smooth,
                                           self.kalman_process_noise,
                                           self.kalman_measure_noise))

        # Drop tracks below confidence floor or past ghost limit
        self._tracks=[t for t in self._tracks
                      if t._confidence_score>self.track_conf_min
                      and t.age<self.ghost_frames_max]

        shooter=self._pick_shooter()
        contested=self._is_contested(shooter)
        trigger,rhythm_ms=self._evaluate_trigger(shooter,balls)
        self.gcvdata[30]=trigger; self.gcvdata[31]=rhythm_ms
        self._draw_overlays(frame,shooter,balls,contested)
        return frame,self.gcvdata
