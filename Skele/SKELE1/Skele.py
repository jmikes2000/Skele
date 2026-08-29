"""
ball.py  –  AA Games YOLOv11 detection + pose worker

gcvdata layout (8 bytes — matches All Shots.gpc):
  [0] shot_trigger   — 1 for one frame when ball crosses trigger line
  [1] tempo_ms       — 1-100, sent every frame
  [2] shot_mode      — 0=Rhythm 1=Button 2=Stick 3=Straight
  [3] stamina_on     — 0/1 infinite stamina toggle
  [4] ball_x_norm    — 0-255 Kalman X
  [5] ball_y_norm    — 0-255 Kalman Y
  [6] stamina_count  — bars detected
  [7] fps_approx     — capped 255
"""

import os, sys, json, time, glob, shutil, subprocess
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "ultralytics", "--quiet"])
    from ultralytics import YOLO

_DIR      = os.path.dirname(os.path.abspath(__file__))
_SETTINGS = os.path.join(_DIR, "settings.json")

CLS_BALL    = 0
CLS_STAMINA = 1

MODEL_IMGSZ  = 832
POSE_IMGSZ   = 832   # engine was built at 832
NMS_IOU      = 0.45
ROI_SIZE     = 832
ROI_SMOOTH   = 0.80
POSE_SKIP    = 2     # run pose every Nth frame (2 = every other frame)

# Ball size hard limits in pixels (applied in full-frame coords)
BALL_PX_MIN = 10
BALL_PX_MAX = 45

# COCO 17-keypoint skeleton
_SKELETON = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]

COL_ROI       = (255, 255, 255)  # white  — ROI brackets
COL_BALL_BOX  = (0, 255, 0)     # green  — ball bounding box
COL_BALL_XHAIR= (255, 255, 255) # white  — crosshair inside ball box
COL_SKEL      = (255, 255, 255) # white  — skeleton lines
COL_JOINT     = (0, 255, 0)     # green  — keypoints
COL_STAM      = (0, 0, 255)     # red    — stamina bar bounding box only
# legacy alias
COL_BALL      = COL_BALL_BOX

_DEFAULTS = {
    "model_path":       "",
    "pose_path":        "",
    "ball_conf":        0.50,
    "stamina_conf":     0.40,
    "pose_conf":        0.50,
    "show_ball":        True,
    "show_stamina":     True,
    "show_pose":        True,
    "show_roi":         True,
    "show_player_box":  True,
    "show_trigger":     True,
    "stam_min_aspect":  2.0,
    "stam_max_w_frac":  0.40,
    "stam_max_h_frac":  0.08,
    "roi_expand_up":    420,
    "roi_expand_down":  60,
    "trigger_offset":   0,
    "trigger_height":   12,   # half-height of trigger box in pixels
    # GPC controls
    "tempo_ms":         45,    # gcv[1] — sent every frame
    "shot_mode":        0,     # gcv[2] — 0=Rhythm 1=Button 2=Stick 3=Straight
    "stamina_on":       0,     # gcv[3] — 0=off 1=on
}


def _load_settings() -> dict:
    for _ in range(3):
        try:
            with open(_SETTINGS) as f:
                raw = f.read().strip()
            if raw:
                d = json.loads(raw)
                for k, v in _DEFAULTS.items():
                    d.setdefault(k, v)
                return d
        except Exception:
            pass
        time.sleep(0.02)
    return dict(_DEFAULTS)


def _get_mtime() -> int:
    try:    return os.stat(_SETTINGS).st_mtime_ns
    except: return 0


def _find_model(key: str, *globs: str) -> str:
    cfg = _load_settings()
    explicit = cfg.get(key, "").strip()
    if explicit and os.path.exists(explicit):
        return explicit
    for pattern in globs:
        hits = glob.glob(pattern, recursive=True)
        if hits:
            return max(hits, key=os.path.getmtime)
    return ""


# ── Pose engine auto-download / export ───────────────────────────────────────
_POSE_MODEL_NAME = "yolo11s-pose"   # change to yolo11m-pose etc. to match GPU VRAM

def _download_pose_engine() -> str:
    """Download yolo11s-pose.pt and export it to TensorRT engine (fp16).
    Falls back to ONNX if TensorRT is unavailable.
    Returns the path to the created file, or '' on failure.
    """
    pt_path     = os.path.join(_DIR, f"{_POSE_MODEL_NAME}.pt")
    engine_path = os.path.join(_DIR, f"{_POSE_MODEL_NAME}.engine")
    onnx_path   = os.path.join(_DIR, f"{_POSE_MODEL_NAME}.onnx")

    # Already exported?
    for p in (engine_path, onnx_path):
        if os.path.exists(p):
            print(f"[AAGames] Pose file already exists: {p}")
            return p

    print(f"[AAGames] Pose engine not found — downloading {_POSE_MODEL_NAME}.pt ...")
    try:
        from ultralytics import YOLO as _YOLO
        model = _YOLO(f"{_POSE_MODEL_NAME}.pt")   # Ultralytics auto-downloads to its cache
        # copy .pt into our project dir so the export lands beside Skele.py
        import ultralytics
        cache_pt = os.path.join(os.path.dirname(ultralytics.__file__),
                                "assets", f"{_POSE_MODEL_NAME}.pt")
        # Ultralytics actually places downloaded weights in cwd or ~/.config/ultralytics
        # Try both; whichever exists wins.
        for candidate in [
            f"{_POSE_MODEL_NAME}.pt",
            os.path.join(os.path.expanduser("~"), ".config", "Ultralytics",
                         f"{_POSE_MODEL_NAME}.pt"),
        ]:
            if os.path.exists(candidate) and not os.path.exists(pt_path):
                shutil.copy2(candidate, pt_path)
                break

        print(f"[AAGames] Exporting to TensorRT engine (fp16) — this may take a few minutes ...")
        try:
            export_path = model.export(format="engine", half=True, imgsz=832,
                                       workspace=4)
            # move to project dir if exported elsewhere
            if export_path and os.path.exists(str(export_path)):
                if os.path.abspath(str(export_path)) != os.path.abspath(engine_path):
                    shutil.move(str(export_path), engine_path)
            if not os.path.exists(engine_path):
                raise RuntimeError("engine export produced no file")
            print(f"[AAGames] Pose engine saved: {engine_path}")
            _save_pose_path(engine_path)
            return engine_path
        except Exception as trt_err:
            print(f"[AAGames] TensorRT export failed ({trt_err}); falling back to ONNX ...")
            export_path = model.export(format="onnx", imgsz=832)
            if export_path and os.path.exists(str(export_path)):
                if os.path.abspath(str(export_path)) != os.path.abspath(onnx_path):
                    shutil.move(str(export_path), onnx_path)
            if not os.path.exists(onnx_path):
                raise RuntimeError("ONNX export produced no file")
            print(f"[AAGames] Pose ONNX saved: {onnx_path}")
            _save_pose_path(onnx_path)
            return onnx_path
    except Exception as e:
        print(f"[AAGames] Auto-download/export failed: {e}")
        return ""


def _save_pose_path(path: str):
    """Persist pose_path into settings.json so the GUI reflects the new file."""
    try:
        cfg = _load_settings()
        cfg["pose_path"] = path
        with open(_SETTINGS, "w") as f:
            json.dump(cfg, f, indent=2)
    except Exception as e:
        print(f"[AAGames] Could not update settings.json: {e}")


def _draw_skeleton(frame, kp_xy, kp_c, thresh):
    for a, b in _SKELETON:
        if kp_c[a] >= thresh and kp_c[b] >= thresh:
            x1, y1 = int(kp_xy[a][0]), int(kp_xy[a][1])
            x2, y2 = int(kp_xy[b][0]), int(kp_xy[b][1])
            if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                cv2.line(frame, (x1, y1), (x2, y2), COL_SKEL, 1, cv2.LINE_AA)
    for (x, y), c in zip(kp_xy, kp_c):
        if c >= thresh and x > 0 and y > 0:
            ix, iy = int(x), int(y)
            cv2.rectangle(frame, (ix-2, iy-2), (ix+2, iy+2), COL_JOINT, -1)


def _draw_ball_box(frame, x1, y1, x2, y2):
    cv2.rectangle(frame, (x1, y1), (x2, y2), COL_BALL_BOX, 1, cv2.LINE_AA)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    arm = max(4, (x2 - x1) // 3)
    cv2.line(frame, (cx - arm, cy), (cx + arm, cy), COL_BALL_XHAIR, 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - arm), (cx, cy + arm), COL_BALL_XHAIR, 1, cv2.LINE_AA)


def _draw_roi_brackets(frame, rx1, ry1, rx2, ry2, colour, brk=24, t=2):
    cv2.line(frame, (rx1, ry1), (rx1+brk, ry1), colour, t)
    cv2.line(frame, (rx1, ry1), (rx1, ry1+brk), colour, t)
    cv2.line(frame, (rx2, ry1), (rx2-brk, ry1), colour, t)
    cv2.line(frame, (rx2, ry1), (rx2, ry1+brk), colour, t)
    cv2.line(frame, (rx1, ry2), (rx1+brk, ry2), colour, t)
    cv2.line(frame, (rx1, ry2), (rx1, ry2-brk), colour, t)
    cv2.line(frame, (rx2, ry2), (rx2-brk, ry2), colour, t)
    cv2.line(frame, (rx2, ry2), (rx2, ry2-brk), colour, t)


# ── Kalman filter for ball position ──────────────────────────────────────────
class BallKalman:
    """2D constant-velocity Kalman filter for ball centre prediction."""
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        # state: [x, y, vx, vy]
        self.kf.transitionMatrix    = np.array([[1,0,1,0],
                                                 [0,1,0,1],
                                                 [0,0,1,0],
                                                 [0,0,0,1]], dtype=np.float32)
        self.kf.measurementMatrix   = np.array([[1,0,0,0],
                                                 [0,1,0,0]], dtype=np.float32)
        self.kf.processNoiseCov     = np.eye(4, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        self.kf.errorCovPost        = np.eye(4, dtype=np.float32)
        self._init = False

    def update(self, cx: float, cy: float):
        meas = np.array([[cx], [cy]], dtype=np.float32)
        if not self._init:
            self.kf.statePre = np.array([[cx],[cy],[0],[0]], dtype=np.float32)
            self.kf.statePost = self.kf.statePre.copy()
            self._init = True
        self.kf.correct(meas)

    def predict(self):
        """Returns (px, py) predicted centre."""
        pred = self.kf.predict()
        return float(pred[0][0]), float(pred[1][0])

    @property
    def initialised(self):
        return self._init


class GCVWorker:

    def __init__(self, width: int = 1920, height: int = 1080):
        self.gcvdata  = bytearray(8)
        self.width    = width
        self.height   = height

        self._mtime       = _get_mtime()
        self._frame_count = 0
        self._t_last      = time.perf_counter()
        self._fps         = 0.0

        # ROI smoothing (drawing only)
        self._last_roi   = None
        self._roi_cx_s   = None
        self._roi_cy_s   = None
        self._last_pose  = None
        self._ball_kf    = BallKalman()
        self._prev_ball_y   = None
        self._trigger_fired = False
        self._ball_was_in_box = False
        self._player_boxes   = []        # list of (x1,y1,x2,y2) per detected person
        self._player_box_top = None      # kept for fallback when no pose available
        self._player_box_x1  = None
        self._player_box_x2  = None

        cfg = _load_settings()
        self._apply(cfg)

        # ── Detection model ───────────────────────────────────────────────────
        det_path = _find_model(
            "model_path",
            os.path.join(_DIR, "runs", "**", "best.engine"),
            os.path.join(_DIR, "runs", "**", "best.onnx"),
            os.path.join(_DIR, "runs", "**", "best.pt"),
        )
        if not det_path:
            raise FileNotFoundError("[AAGames] No detection model. Run train.py + export.py.")
        print(f"[AAGames] Detection : {det_path}")
        self._det = YOLO(det_path, task="detect")

        # ── Pose model ────────────────────────────────────────────────────────
        pose_path = _find_model(
            "pose_path",
            os.path.join(_DIR, "yolo11s-pose.engine"),
            os.path.join(_DIR, "*pose*.engine"),
            os.path.join(_DIR, "*pose*.onnx"),
        )
        if not pose_path:
            pose_path = _download_pose_engine()
        if pose_path:
            print(f"[AAGames] Pose      : {pose_path}")
            self._pose = YOLO(pose_path, task="pose")
        else:
            print("[AAGames] No pose model — skeleton disabled")
            self._pose = None

        # warm-up
        dummy_full = np.zeros((height, width, 3), dtype=np.uint8)
        dummy_roi  = np.zeros((ROI_SIZE, ROI_SIZE, 3), dtype=np.uint8)
        self._det(dummy_full, verbose=False)
        if self._pose:
            self._pose(dummy_roi, verbose=False)

        print(f"[AAGames] Ready  —  ball={self._ball_conf:.2f}  "
              f"stamina={self._stam_conf:.2f}  pose={self._pose_conf:.2f}  "
              f"ball_px={BALL_PX_MIN}-{BALL_PX_MAX}")

    # ── Settings ──────────────────────────────────────────────────────────────
    def _apply(self, d: dict):
        self._ball_conf       = float(d.get("ball_conf",       0.50))
        self._stam_conf       = float(d.get("stamina_conf",    0.40))
        self._pose_conf       = float(d.get("pose_conf",       0.50))
        self._show_ball       = bool(d.get("show_ball",        True))
        self._show_stamina    = bool(d.get("show_stamina",     True))
        self._show_pose       = bool(d.get("show_pose",        True))
        self._show_roi        = bool(d.get("show_roi",         True))
        self._show_player_box = bool(d.get("show_player_box",  True))
        self._show_trigger    = bool(d.get("show_trigger",     True))
        self._stam_min_asp    = float(d.get("stam_min_aspect", 2.0))
        self._stam_max_w      = float(d.get("stam_max_w_frac", 0.40))
        self._stam_max_h      = float(d.get("stam_max_h_frac", 0.08))
        self._roi_up          = int(d.get("roi_expand_up",     420))
        self._roi_down        = int(d.get("roi_expand_down",   60))
        self._trigger_offset  = int(d.get("trigger_offset",   0))
        self._trigger_height  = max(1, int(d.get("trigger_height",  12)))
        self._tempo_ms        = max(1, min(100, int(d.get("tempo_ms",   45))))
        self._shot_mode       = int(d.get("shot_mode",  0))
        self._stamina_on      = int(d.get("stamina_on", 0))

    def _maybe_reload(self):
        self._frame_count += 1
        t = _get_mtime()
        if t != self._mtime or self._frame_count % 60 == 0:
            self._mtime = t
            self._apply(_load_settings())

    def _build_roi(self, cx: int, cy: int, fh: int, fw: int):
        """Build exact ROI from real stamina position (no smoothing)."""
        ry1 = cy - self._roi_up
        rx1 = cx - ROI_SIZE // 2
        rx2 = rx1 + ROI_SIZE
        ry2 = ry1 + ROI_SIZE
        if rx1 < 0:  rx2 -= rx1; rx1 = 0
        if ry1 < 0:  ry2 -= ry1; ry1 = 0
        if rx2 > fw: rx1 -= (rx2-fw); rx2 = fw
        if ry2 > fh: ry1 -= (ry2-fh); ry2 = fh
        return max(0,rx1), max(0,ry1), min(fw,rx2), min(fh,ry2)

    def _smooth_roi_for_draw(self, cx: int, cy: int, fh: int, fw: int):
        """EMA-smoothed ROI used only for drawing the bracket."""
        if self._roi_cx_s is None:
            self._roi_cx_s = float(cx)
            self._roi_cy_s = float(cy)
        else:
            self._roi_cx_s = ROI_SMOOTH * self._roi_cx_s + (1-ROI_SMOOTH) * cx
            self._roi_cy_s = ROI_SMOOTH * self._roi_cy_s + (1-ROI_SMOOTH) * cy
        return self._build_roi(int(self._roi_cx_s), int(self._roi_cy_s), fh, fw)

    # ── Main processing ───────────────────────────────────────────────────────
    def process(self, frame: np.ndarray):
        self._maybe_reload()

        self.gcvdata[0] = 0                          # shot_trigger
        self.gcvdata[1] = self._tempo_ms             # tempo_ms
        self.gcvdata[2] = self._shot_mode            # shot_mode
        self.gcvdata[3] = self._stamina_on           # stamina_on
        self.gcvdata[4] = 128                        # ball_x_norm
        self.gcvdata[5] = 128                        # ball_y_norm
        self.gcvdata[6] = 0                          # stamina_count
        self.gcvdata[7] = min(255, int(self._fps))   # fps

        if frame is None:
            return frame, self.gcvdata

        fh, fw = frame.shape[:2]

        # ── Step 1: ONE full-frame detection pass — gets stamina + ball ─────────
        det_full = self._det(
            frame,
            conf    = min(self._ball_conf, self._stam_conf),
            iou     = NMS_IOU,
            imgsz   = MODEL_IMGSZ,
            verbose = False,
        )

        stam_cx = stam_cy = None
        stamina_count = 0
        stam_box = None
        raw_balls = []   # (fx1,fy1,fx2,fy2,conf) — filtered by ROI in step 2

        if det_full and det_full[0].boxes is not None:
            boxes = det_full[0].boxes.xyxy.cpu().numpy().astype(int)
            clses = det_full[0].boxes.cls.cpu().numpy().astype(int)
            confs = det_full[0].boxes.conf.cpu().numpy()
            best_sc = 0.0

            for (x1, y1, x2, y2), cls, conf in zip(boxes, clses, confs):
                bw, bh = x2-x1, y2-y1
                if bw < 1 or bh < 1:
                    continue

                if cls == CLS_STAMINA and conf >= self._stam_conf:
                    if (bw/bh) < self._stam_min_asp: continue
                    if bw > fw * self._stam_max_w:   continue
                    if bh > fh * self._stam_max_h:   continue
                    stamina_count += 1
                    if conf > best_sc:
                        best_sc  = float(conf)
                        stam_cx  = (x1+x2)//2
                        stam_cy  = (y1+y2)//2
                        stam_box = (x1, y1, x2, y2)

                elif cls == CLS_BALL and conf >= self._ball_conf:
                    if bw < BALL_PX_MIN or bw > BALL_PX_MAX: continue
                    if bh < BALL_PX_MIN or bh > BALL_PX_MAX: continue
                    aspect = bw / bh
                    if aspect > 2.5 or aspect < 1/2.5: continue
                    raw_balls.append((x1, y1, x2, y2, float(conf)))

        self.gcvdata[6] = min(255, stamina_count)

        if stam_cx is not None and self._show_stamina and stam_box:
            x1, y1, x2, y2 = stam_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), COL_STAM, 2, cv2.LINE_AA)

        # ── Step 2: build ROI ─────────────────────────────────────────────────
        if stam_cx is not None:
            real_roi = self._build_roi(stam_cx, stam_cy, fh, fw)
            draw_roi = self._smooth_roi_for_draw(stam_cx, stam_cy, fh, fw)
            self._last_roi = (real_roi, draw_roi, stam_cy)
        else:
            # no stamina — centre ROI on screen
            cx = fw // 2
            cy = fh // 2
            real_roi = self._build_roi(cx, cy, fh, fw)
            if self._last_roi is not None:
                _, _, stam_cy = self._last_roi
            draw_roi = self._smooth_roi_for_draw(cx, cy, fh, fw)
            self._last_roi = (real_roi, draw_roi, stam_cy)

        rx1, ry1, rx2, ry2 = real_roi
        drx1, dry1, drx2, dry2 = draw_roi

        if self._show_roi:
            _draw_roi_brackets(frame, drx1, dry1, drx2, dry2, COL_ROI)

        # ── Step 3: filter raw_balls to those inside ROI, update Kalman ─────────
        ball_detected = False

        for (fx1, fy1, fx2, fy2, conf) in raw_balls:
            # must be inside the real ROI
            fcx = (fx1+fx2)//2
            fcy = (fy1+fy2)//2
            if not (rx1 <= fcx <= rx2 and ry1 <= fcy <= ry2):
                continue
            self._ball_kf.update(float(fcx), float(fcy))
            ball_detected = True
            if self._show_ball:
                _draw_ball_box(frame, fx1, fy1, fx2, fy2)
            break  # only use highest-conf ball (raw_balls order from model)

        # use Kalman prediction for gcvdata regardless of detection this frame
        if self._ball_kf.initialised:
            px, py = self._ball_kf.predict()
            self.gcvdata[4] = int(max(0, min(255, px / fw * 255)))
            self.gcvdata[5] = int(max(0, min(255, py / fh * 255)))

        # ── Step 4: pose inside REAL ROI — skeleton + collect all player boxes ─
        person_count = 0
        self._player_boxes = []   # reset each frame regardless of pose availability
        if self._pose and (self._show_pose or self._show_player_box):
            run_pose = (self._frame_count % POSE_SKIP == 0)
            if run_pose:
                crop = frame[ry1:ry2, rx1:rx2].copy()
                if crop.shape[0] >= 32 and crop.shape[1] >= 32:
                    pose_res = self._pose(
                        crop,
                        conf    = self._pose_conf,
                        iou     = NMS_IOU,
                        imgsz   = POSE_IMGSZ,
                        verbose = False,
                    )
                    if pose_res and pose_res[0].keypoints is not None:
                        kp_xy   = pose_res[0].keypoints.xy.cpu().numpy()
                        kp_c    = pose_res[0].keypoints.conf.cpu().numpy()
                        p_boxes = pose_res[0].boxes
                        person_count = len(kp_xy)
                        self._last_pose = (kp_xy, kp_c, p_boxes)
            elif hasattr(self, '_last_pose') and self._last_pose is not None:
                kp_xy, kp_c, p_boxes = self._last_pose
                person_count = len(kp_xy)

            # collect player boxes in full-frame coords, draw skeletons
            if person_count > 0:
                for i in range(person_count):
                    pboxes_np = (p_boxes.xyxy.cpu().numpy().astype(int)
                                 if p_boxes is not None and len(p_boxes) > i
                                 else None)
                    if pboxes_np is not None:
                        bx1, by1, bx2, by2 = pboxes_np[i]
                        fx1 = bx1 + rx1; fy1 = by1 + ry1
                        fx2 = bx2 + rx1; fy2 = by2 + ry1
                        self._player_boxes.append((fx1, fy1, fx2, fy2))
                        if self._show_player_box:
                            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2),
                                          COL_ROI, 1, cv2.LINE_AA)
                    if self._show_pose and len(kp_xy) > i:
                        offset_xy = kp_xy[i] + np.array([rx1, ry1], dtype=np.float32)
                        _draw_skeleton(frame, offset_xy, kp_c[i], self._pose_conf)
        # ── Trigger boxes — one per detected player ───────────────────────────
        # Each box spans the player's full width, centred at their box-top
        # + trigger_offset, half-height = trigger_height.
        # Falls back to a single ROI-wide box when no pose is available.
        if not self._player_boxes:
            fallback_y = (self._player_box_top + self._trigger_offset
                          if self._player_box_top is not None
                          else ry1 + self._trigger_offset)
            self._player_boxes = [(rx1, fallback_y - self._trigger_height,
                                   rx2, fallback_y + self._trigger_height)]
            use_raw = True   # already converted to trig coords
        else:
            use_raw = False

        ball_in_any = False
        if self._ball_kf.initialised:
            px, py = self._ball_kf.predict()
            bpx, bpy = int(px), int(py)

            for (fx1, fy1, fx2, fy2) in self._player_boxes:
                if use_raw:
                    t_top, t_bot = fy1, fy2
                    t_x1,  t_x2  = fx1, fx2
                else:
                    t_cy  = fy1 + self._trigger_offset   # top of player box + offset
                    t_top = t_cy - self._trigger_height
                    t_bot = t_cy + self._trigger_height
                    t_x1, t_x2 = fx1, fx2

                if t_x1 <= bpx <= t_x2 and t_top <= bpy <= t_bot:
                    ball_in_any = True

                if self._show_trigger:
                    fired = ball_in_any
                    col_trig = (0, 255, 255) if fired else (0, 0, 255)
                    if fired:
                        ov = frame.copy()
                        cv2.rectangle(ov, (t_x1, t_top), (t_x2, t_bot),
                                      col_trig, -1)
                        cv2.addWeighted(ov, 0.35, frame, 0.65, 0, frame)
                    cv2.rectangle(frame, (t_x1, t_top), (t_x2, t_bot),
                                  col_trig, 1, cv2.LINE_AA)

        was_in_any = getattr(self, '_ball_was_in_box', False)
        self.gcvdata[0] = 1 if (ball_in_any and not was_in_any) else 0
        self._ball_was_in_box = ball_in_any

        # gcvdata[1][2][3] are already set at top of process() each frame
        # just keep fps updated
        now = time.perf_counter()
        dt  = now - self._t_last; self._t_last = now
        if dt > 0: self._fps = 0.1*(1/dt) + 0.9*self._fps
        self.gcvdata[7] = min(255, int(self._fps))

        return frame, self.gcvdata


if __name__ == "__main__":
    print("[AAGames] Standalone test — press Q to quit")
    worker = GCVWorker(1920, 1080)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        dummy = np.zeros((1080, 1920, 3), dtype=np.uint8)
        out, data = worker.process(dummy)
        print(f"gcvdata = {list(data)}")
        sys.exit(0)
    while True:
        ret, frame = cap.read()
        if not ret: break
        out, data = worker.process(frame)
        cv2.imshow("AAGames", cv2.resize(out, (1280, 720)))
        if cv2.waitKey(1) & 0xFF == ord("q"): break
    cap.release()
    cv2.destroyAllWindows()
