# -*- coding: utf-8 -*-
"""
SkeleX – Settings GUI
Run:  python main.py
Writes settings.json — picked up live by SkeleX_GCV.py in Gtuner IV.

Mirrors the CVbeta/CV/skelex_pose.py defaults exactly so the same
skele_config.json schema can be used in both deployment contexts.
"""

import json
import os
import sys
import tkinter as tk
from tkinter import ttk

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
try:
    from SkeleX_GCV import GCVWorker
except Exception:
    pass

SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")

DEFAULTS = {
    # Ball detection
    "ball_conf": 0.25,
    "ball_nms_iou": 0.45,

    # Pose / skeleton
    "pose_conf": 0.30,
    "keypoint_conf": 0.25,
    "smoothing_ema": 0.35,
    "show_pose_skeleton": True,

    # Trigger Mode Options
    "trigger_mode": "head_line",
    "elbow_angle_threshold": 155.0,
    "ball_release_threshold": 0.50,

    # Head-line trigger
    "head_line_enabled": True,
    "show_head_line": False,
    "shooting_hand": "right",
    "head_line_offset_px": 0,
    "hold_frames": 0,
    "min_wrist_rise_px": 0.0,

    # Foot lift-off verification
    "foot_lift_enabled": False,
    "foot_lift_px": 12,

    # Apex Deceleration
    "vy_smoothing_alpha": 0.55,
    "max_shot_hold_ms": 800,
    "apex_holdoff_ms": 20,

    # Wrist Elevation Release
    "wrist_release_enabled": True,
    "wrist_conf_min": 0.40,
    "wrist_elev_frames": 2,
    "wrist_only_above_px": 15,

    # Player ROI
    "show_roi_box": False,
    "roi_padding_px": 0,

    # GPC timing
    "rhythm_ms_phase1": 0,
}


def load_settings() -> dict:
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            d = json.loads(f.read().strip())
        if isinstance(d, dict):
            # Migrate legacy keys
            if "smooth"  in d and "smoothing_ema" not in d: d["smoothing_ema"]  = d["smooth"]
            if "kp_conf" in d and "keypoint_conf" not in d: d["keypoint_conf"] = d["kp_conf"]
            for k, v in DEFAULTS.items():
                d.setdefault(k, v)
            return d
    except Exception:
        pass
    return dict(DEFAULTS)


def save_settings(data: dict) -> None:
    merged = dict(DEFAULTS)
    merged.update(data or {})
    tmp = SETTINGS_FILE + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, SETTINGS_FILE)
    except Exception:
        # Fallback to non-atomic write
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2)


def _make_slider_row(parent, key, label, lo, hi, val, tip):
    container = ttk.Frame(parent)
    container.pack(fill="x", pady=(4, 4))
    
    v = tk.DoubleVar(value=float(val))
    row = ttk.Frame(container)
    row.pack(fill="x", pady=(6, 0))
    tk.Label(row, text=label, bg="#111122", fg="#cccccc",
             font=("Consolas", 10, "bold"), anchor="w").pack(side="left")
    val_lbl = tk.Label(row, text=f"{float(val):.2f}",
                       bg="#111122", fg="#00e7ff",
                       font=("Consolas", 10), width=6, anchor="e")
    val_lbl.pack(side="right")

    slider = ttk.Scale(container, from_=lo, to=hi, variable=v, orient="horizontal", length=340,
                       command=lambda _, vv=v, lbl=val_lbl, key=key:
                           (lbl.config(text=f"{vv.get():.2f}"),
                            _save_after()))
    slider.pack(fill="x", pady=(2, 0))
    tk.Label(container, text=tip, bg="#111122", fg="#555577",
             font=("Consolas", 8), justify="left", wraplength=340, anchor="w").pack(fill="x", pady=(1, 4))
    return v, container


def _make_int_row(parent, key, label, lo, hi, val, tip):
    container = ttk.Frame(parent)
    container.pack(fill="x", pady=(4, 4))
    
    v = tk.IntVar(value=int(val))
    row = ttk.Frame(container)
    row.pack(fill="x", pady=(6, 0))
    tk.Label(row, text=label, bg="#111122", fg="#cccccc",
             font=("Consolas", 10, "bold"), anchor="w").pack(side="left")
    val_lbl = tk.Label(row, text=f"{int(val)}",
                       bg="#111122", fg="#00e7ff",
                       font=("Consolas", 10), width=6, anchor="e")
    val_lbl.pack(side="right")

    slider = ttk.Scale(container, from_=lo, to=hi, variable=v, orient="horizontal", length=340,
                       command=lambda _, vv=v, lbl=val_lbl, key=key:
                           (lbl.config(text=f"{int(vv.get())}"),
                            _save_after()))
    slider.pack(fill="x", pady=(2, 0))
    tk.Label(container, text=tip, bg="#111122", fg="#555577",
             font=("Consolas", 8), justify="left", wraplength=340, anchor="w").pack(fill="x", pady=(1, 4))
    return v, container


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SkeleX")
        self.resizable(False, False)
        self.configure(bg="#111122")

        st = ttk.Style(self)
        st.theme_use("clam")
        st.configure(".",  background="#111122", foreground="#dddddd", font=("Consolas", 10))
        st.configure("TScale",    background="#111122")
        st.configure("TLabel",    background="#111122", foreground="#aaaaaa")
        st.configure("TFrame",    background="#111122")
        st.configure("TButton",   background="#1e1e3a", foreground="#dddddd", padding=5)
        st.configure("TCheckbutton", background="#111122", foreground="#dddddd")
        st.map("TButton", background=[("active", "#2a2a50")])

        cfg = load_settings()

        tk.Label(self, text="SkeleX", bg="#111122", fg="#00e7ff",
                 font=("Consolas", 16, "bold")).pack(pady=(18, 2))
        tk.Label(self, text="live — changes apply immediately",
                 bg="#111122", fg="#444466",
                 font=("Consolas", 8)).pack(pady=(0, 14))

        self._vars: dict = {}
        self._checks: dict = {}

        # ── Ball Detection ─────────────────────────────────────────────────
        tk.Label(self, text="BALL DETECTION", bg="#111122", fg="#00e7ff",
                 font=("Consolas", 11, "bold")).pack(pady=(8, 0), anchor="w", padx=24)
        ball_frame = ttk.Frame(self); ball_frame.pack(padx=24, pady=4, fill="x")
        self._vars["ball_conf"], _ = _make_slider_row(ball_frame, "ball_conf",
            "Ball confidence", 0.05, 0.95, cfg["ball_conf"],
            "Min score to draw the ball circle.")
        self._vars["ball_nms_iou"], _ = _make_slider_row(ball_frame, "ball_nms_iou",
            "NMS IoU", 0.05, 0.95, cfg["ball_nms_iou"],
            "Non-max suppression — lower = fewer overlapping circles.")

        # ── Pose / Skeleton ────────────────────────────────────────────────
        tk.Label(self, text="POSE / SKELETON", bg="#111122", fg="#00e7ff",
                 font=("Consolas", 11, "bold")).pack(pady=(12, 0), anchor="w", padx=24)
        pose_frame = ttk.Frame(self); pose_frame.pack(padx=24, pady=4, fill="x")
        self._vars["pose_conf"], _ = _make_slider_row(pose_frame, "pose_conf",
            "Pose confidence", 0.05, 0.99, cfg["pose_conf"],
            "Min score to show a person skeleton.")
        self._vars["keypoint_conf"], _ = _make_slider_row(pose_frame, "keypoint_conf",
            "Keypoint confidence", 0.05, 0.99, cfg["keypoint_conf"],
            "Min per-joint score to draw a bone or dot.")
        self._vars["smoothing_ema"], _ = _make_slider_row(pose_frame, "smoothing_ema",
            "Smoothing (EMA)", 0.0, 0.99, cfg["smoothing_ema"],
            "Keypoint smoothing across frames.")
        self._checks["show_pose_skeleton"] = tk.BooleanVar(value=bool(cfg["show_pose_skeleton"]))
        tk.Checkbutton(pose_frame, text="Show pose skeleton", variable=self._checks["show_pose_skeleton"],
                       bg="#111122", fg="#dddddd", selectcolor="#1e1e3a", activebackground="#111122",
                       command=_save_after).pack(anchor="w", pady=(4, 0))

        # ── Trigger Configuration ──────────────────────────────────────────
        tk.Label(self, text="TRIGGER MODE & CONFIG", bg="#111122", fg="#00e7ff",
                 font=("Consolas", 11, "bold")).pack(pady=(12, 0), anchor="w", padx=24)
        trig_frame = ttk.Frame(self); trig_frame.pack(padx=24, pady=4, fill="x")
        
        # Trigger Enablement
        self._checks["head_line_enabled"] = tk.BooleanVar(value=bool(cfg["head_line_enabled"]))
        tk.Checkbutton(trig_frame, text="Enable auto-shoot trigger",
                       variable=self._checks["head_line_enabled"],
                       bg="#111122", fg="#00ff88", selectcolor="#1e1e3a", activebackground="#111122",
                       command=_save_after, font=("Consolas", 10, "bold")).pack(anchor="w", pady=(2, 0))

        # Mode Selection Combobox
        mode_row = ttk.Frame(trig_frame)
        mode_row.pack(fill="x", pady=(8, 0))
        tk.Label(mode_row, text="Trigger Mode", bg="#111122", fg="#cccccc",
                 font=("Consolas", 10, "bold"), anchor="w").pack(side="left")
                 
        self._trigger_mode = tk.StringVar(value=str(cfg["trigger_mode"]))
        mode_cb = ttk.Combobox(mode_row, textvariable=self._trigger_mode, 
                               values=("head_line", "elbow_angle", "ball_release", "apex_decel"),
                               state="readonly", width=15)
        mode_cb.pack(side="right")
        mode_cb.bind("<<ComboboxSelected>>", lambda _: (self._update_ui_state(), _save_after()))

        # Hand Selection
        tk.Label(trig_frame, text="Shooting hand", bg="#111122", fg="#cccccc",
                 font=("Consolas", 10, "bold")).pack(anchor="w", pady=(8, 2))
        self._shooting_hand = tk.StringVar(value=str(cfg["shooting_hand"]))
        hand_row = ttk.Frame(trig_frame); hand_row.pack(anchor="w")
        for h in ("right", "left", "auto"):
            tk.Radiobutton(hand_row, text=h.title(), variable=self._shooting_hand, value=h,
                           bg="#111122", fg="#dddddd", selectcolor="#1e1e3a",
                           activebackground="#111122", command=_save_after).pack(side="left", padx=(0, 12))

        # ── Mode-Specific Sliders ──
        self._headline_containers = []

        # Head Line Visibility
        self._checks["show_head_line"] = tk.BooleanVar(value=bool(cfg["show_head_line"]))
        shl_btn = tk.Checkbutton(trig_frame, text="Show head line (overlay)",
                                 variable=self._checks["show_head_line"],
                                 bg="#111122", fg="#dddddd", selectcolor="#1e1e3a", activebackground="#111122",
                                 command=_save_after)
        shl_btn.pack(anchor="w", pady=(4, 0))
        self._headline_containers.append(shl_btn)

        # Head-line offset, hold, rise
        var, container = _make_int_row(trig_frame, "head_line_offset_px",
            "Head line offset (px)", -200, 200, cfg["head_line_offset_px"],
            "Shift line up (-) or down (+) from the shoulder level.")
        self._vars["head_line_offset_px"] = var
        self._headline_containers.append(container)

        var, container = _make_int_row(trig_frame, "hold_frames",
            "Hold frames", 0, 30, cfg["hold_frames"],
            "Frames ball/wrist must stay above the line to fire.")
        self._vars["hold_frames"] = var
        self._headline_containers.append(container)

        var, container = _make_slider_row(trig_frame, "min_wrist_rise_px",
            "Min wrist rise (px/frame)", 0.0, 30.0, cfg["min_wrist_rise_px"],
            "Wrist must be moving upward at least this fast to trigger.")
        self._vars["min_wrist_rise_px"] = var
        self._headline_containers.append(container)

        # Elbow Angle Slider
        var, container = _make_slider_row(trig_frame, "elbow_angle_threshold",
            "Elbow angle threshold (°)", 120.0, 180.0, cfg["elbow_angle_threshold"],
            "Release shot when shooting elbow extension reaches this angle.")
        self._vars["elbow_angle_threshold"] = var
        self._elbow_container = container

        # Ball Release Slider
        var, container = _make_slider_row(trig_frame, "ball_release_threshold",
            "Ball release threshold", 0.30, 0.90, cfg["ball_release_threshold"],
            "Release shot when composite ball separation score reaches this score.")
        self._vars["ball_release_threshold"] = var
        self._ball_container = container

        # Apex Deceleration Sliders
        self._apex_containers = []
        
        self._checks["wrist_release_enabled"] = tk.BooleanVar(value=bool(cfg.get("wrist_release_enabled", True)))
        wre_btn = tk.Checkbutton(trig_frame, text="Enable wrist elevation release",
                                 variable=self._checks["wrist_release_enabled"],
                                 bg="#111122", fg="#dddddd", selectcolor="#1e1e3a", activebackground="#111122",
                                 command=_save_after)
        self._apex_containers.append(wre_btn)
        
        var, container = _make_slider_row(trig_frame, "wrist_conf_min",
            "Wrist confidence min", 0.05, 0.99, cfg.get("wrist_conf_min", 0.40),
            "Minimum keypoint confidence to track shooting wrist.")
        self._vars["wrist_conf_min"] = var
        self._apex_containers.append(container)
        
        var, container = _make_int_row(trig_frame, "wrist_elev_frames",
            "Wrist elevation frames", 1, 10, cfg.get("wrist_elev_frames", 2),
            "Consecutive frames wrist must stay elevated to trigger release.")
        self._vars["wrist_elev_frames"] = var
        self._apex_containers.append(container)
        
        var, container = _make_int_row(trig_frame, "wrist_only_above_px",
            "Wrist elevation threshold (px)", 1, 50, cfg.get("wrist_only_above_px", 15),
            "Wrist must rise this many pixels above shoulder to count as elevated.")
        self._vars["wrist_only_above_px"] = var
        self._apex_containers.append(container)
        
        var, container = _make_slider_row(trig_frame, "vy_smoothing_alpha",
            "Velocity smoothing alpha", 0.05, 0.99, cfg.get("vy_smoothing_alpha", 0.55),
            "EMA smoothing factor for jump velocity estimation.")
        self._vars["vy_smoothing_alpha"] = var
        self._apex_containers.append(container)
        
        var, container = _make_int_row(trig_frame, "max_shot_hold_ms",
            "Max shot hold (ms)", 100, 2000, cfg.get("max_shot_hold_ms", 800),
            "Safety timer cutoff — forces release if shot is held this long.")
        self._vars["max_shot_hold_ms"] = var
        self._apex_containers.append(container)
        
        var, container = _make_int_row(trig_frame, "apex_holdoff_ms",
            "Apex holdoff delay (ms)", 0, 500, cfg.get("apex_holdoff_ms", 20),
            "Hold off checking wrist elevation for this duration after jump starts.")
        self._vars["apex_holdoff_ms"] = var
        self._apex_containers.append(container)

        # ── Foot Lift-Off Verification ──────────────────────────────────────
        tk.Label(self, text="FOOT LIFT-OFF VERIFICATION", bg="#111122", fg="#00e7ff",
                 font=("Consolas", 11, "bold")).pack(pady=(12, 0), anchor="w", padx=24)
        foot_frame = ttk.Frame(self); foot_frame.pack(padx=24, pady=4, fill="x")
        self._checks["foot_lift_enabled"] = tk.BooleanVar(value=bool(cfg.get("foot_lift_enabled", False)))
        tk.Checkbutton(foot_frame, text="Enable foot lift-off verification", variable=self._checks["foot_lift_enabled"],
                       bg="#111122", fg="#00ff88", selectcolor="#1e1e3a", activebackground="#111122",
                       command=_save_after, font=("Consolas", 10, "bold")).pack(anchor="w")
        self._vars["foot_lift_px"], _ = _make_int_row(foot_frame, "foot_lift_px",
            "Foot lift threshold (px)", 1, 100, cfg.get("foot_lift_px", 12),
            "Ankles must rise at least this many pixels above baseline to qualify jump.")

        # ── Player ROI ──────────────────────────────────────────────────────
        tk.Label(self, text="PLAYER ROI", bg="#111122", fg="#00e7ff",
                 font=("Consolas", 11, "bold")).pack(pady=(12, 0), anchor="w", padx=24)
        roi_frame = ttk.Frame(self); roi_frame.pack(padx=24, pady=4, fill="x")
        self._checks["show_roi_box"] = tk.BooleanVar(value=bool(cfg["show_roi_box"]))
        tk.Checkbutton(roi_frame, text="Show ROI box", variable=self._checks["show_roi_box"],
                       bg="#111122", fg="#dddddd", selectcolor="#1e1e3a", activebackground="#111122",
                       command=_save_after).pack(anchor="w")
        self._vars["roi_padding_px"], _ = _make_int_row(roi_frame, "roi_padding_px",
            "ROI padding (px)", 0, 400, cfg["roi_padding_px"],
            "Expand player box each zoom — balls outside are ignored.")

        # ── Track Robustness ────────────────────────────────────────────────────
        tk.Label(self, text="TRACK ROBUSTNESS", bg="#111122", fg="#00e7ff",
                 font=("Consolas", 11, "bold")).pack(pady=(12, 0), anchor="w", padx=24)
        trk_frame = ttk.Frame(self); trk_frame.pack(padx=24, pady=4, fill="x")
        self._vars["ghost_frames_max"], _ = _make_int_row(trk_frame, "ghost_frames_max",
            "Ghost frames", 4, 30, cfg.get("ghost_frames_max", 12),
            "Frames to hold a lost track via Kalman prediction before dropping it.")
        self._vars["track_conf_decay"], _ = _make_slider_row(trk_frame, "track_conf_decay",
            "Confidence decay", 0.01, 0.50, cfg.get("track_conf_decay", 0.10),
            "Confidence lost per missed frame. Lower = slower decay, holds longer.")
        self._vars["shooter_lock_frames"], _ = _make_int_row(trk_frame, "shooter_lock_frames",
            "Shooter lock frames", 1, 30, cfg.get("shooter_lock_frames", 10),
            "Grace frames before unlocking shooter ID when briefly lost.")

        # ── Kalman Filter ──────────────────────────────────────────────────────
        tk.Label(self, text="KALMAN FILTER", bg="#111122", fg="#00e7ff",
                 font=("Consolas", 11, "bold")).pack(pady=(12, 0), anchor="w", padx=24)
        kf_frame = ttk.Frame(self); kf_frame.pack(padx=24, pady=4, fill="x")
        self._vars["kalman_process_noise"], _ = _make_slider_row(kf_frame, "kalman_process_noise",
            "Process noise (Q)", 0.001, 1.0, cfg.get("kalman_process_noise", 0.01),
            "Higher = follows detections faster. Lower = more inertia / smoother.")
        self._vars["kalman_measure_noise"], _ = _make_slider_row(kf_frame, "kalman_measure_noise",
            "Measurement noise (R)", 0.1, 50.0, cfg.get("kalman_measure_noise", 10.0),
            "Higher = trust predictions over raw detections (smoother, slower to react).")

        # ── Shooting Logic ─────────────────────────────────────────────────────
        tk.Label(self, text="SHOOTING LOGIC", bg="#111122", fg="#00e7ff",
                 font=("Consolas", 11, "bold")).pack(pady=(12, 0), anchor="w", padx=24)
        sl_frame = ttk.Frame(self); sl_frame.pack(padx=24, pady=4, fill="x")
        self._vars["trigger_decay_rate"], _ = _make_slider_row(sl_frame, "trigger_decay_rate",
            "Trigger decay rate", 0.10, 0.99, cfg.get("trigger_decay_rate", 0.50),
            "How fast the accumulator drains when wrist drops. Lower = more forgiving.")
        self._vars["prediction_latency_ms"], _ = _make_slider_row(sl_frame, "prediction_latency_ms",
            "Prediction latency (ms)", 0.0, 200.0, cfg.get("prediction_latency_ms", 50.0),
            "Fire this many ms before predicted wrist crossing to compensate input lag.")
        self._vars["contested_extra_offset_px"], _ = _make_int_row(sl_frame, "contested_extra_offset_px",
            "Contested offset (px)", 0, 50, cfg.get("contested_extra_offset_px", 10),
            "Extra px lowered on head line when a defender is within 1 player height.")
        self._checks["zone_calibration_enabled"] = tk.BooleanVar(
            value=bool(cfg.get("zone_calibration_enabled", True)))
        tk.Checkbutton(sl_frame, text="Enable per-zone auto-calibration",
                       variable=self._checks["zone_calibration_enabled"],
                       bg="#111122", fg="#00ff88", selectcolor="#1e1e3a",
                       activebackground="#111122", command=_save_after,
                       font=("Consolas", 10, "bold")).pack(anchor="w", pady=(4, 0))

        # ── GPC Timing ─────────────────────────────────────────────────────────
        tk.Label(self, text="GPC TIMING — SKELE.GPC", bg="#111122", fg="#a855f7",
                 font=("Consolas", 11, "bold")).pack(pady=(12, 0), anchor="w", padx=24)
        gpc_frame = ttk.Frame(self); gpc_frame.pack(padx=24, pady=4, fill="x")
        self._vars["rhythm_ms_phase1"], _ = _make_int_row(gpc_frame, "rhythm_ms_phase1",
            "Rhythm ms  (phase 1)", 0, 255, cfg["rhythm_ms_phase1"],
            "Writes to gcvdata[31] every frame.\n"
            "GPC reads it after the initial stick deflection — edit in Skele.gpc.")

        # Status + buttons
        self._status = tk.StringVar(value="")
        tk.Label(self, textvariable=self._status,
                 bg="#111122", fg="#00ff88",
                 font=("Consolas", 9)).pack(pady=(8, 0))
        bf = ttk.Frame(self); bf.pack(pady=14)
        ttk.Button(bf, text="Reset defaults", command=self._reset).pack(side="left", padx=8)
        ttk.Button(bf, text="Quit", command=self.destroy).pack(side="left", padx=8)

        # Force window to the foreground and request focus
        self.lift()
        self.attributes("-topmost", True)
        self.after_idle(self.attributes, "-topmost", False)
        self.focus_force()

        # Load initial conditional visibility state
        self._update_ui_state()

    def _update_ui_state(self):
        mode = self._trigger_mode.get()
        
        # Hide all conditional components
        for container in self._headline_containers:
            container.pack_forget()
        self._elbow_container.pack_forget()
        self._ball_container.pack_forget()
        for container in self._apex_containers:
            container.pack_forget()
        
        # Show relevant components
        if mode == "head_line":
            self._headline_containers[0].pack(anchor="w", pady=(4, 0)) # Checkbutton
            for container in self._headline_containers[1:]:
                container.pack(fill="x", pady=(4, 4))
        elif mode == "elbow_angle":
            self._elbow_container.pack(fill="x", pady=(4, 4))
        elif mode == "ball_release":
            self._ball_container.pack(fill="x", pady=(4, 4))
        elif mode == "apex_decel":
            self._apex_containers[0].pack(anchor="w", pady=(4, 0)) # Checkbutton
            for container in self._apex_containers[1:]:
                container.pack(fill="x", pady=(4, 4))

    def _autosave(self):
        try:
            data = {k: round(v.get(), 3) for k, v in self._vars.items()}
            for k, v in self._checks.items():
                data[k] = bool(v.get())
            data["shooting_hand"] = self._shooting_hand.get()
            data["trigger_mode"] = self._trigger_mode.get()
            save_settings(data)
            self._status.set("✔ saved")
            self.after(1500, lambda: self._status.set(""))
        except Exception as e:
            self._status.set(f"error: {e}")

    def _reset(self):
        for k, v in self._vars.items():
            v.set(DEFAULTS[k])
        for k, v in self._checks.items():
            v.set(DEFAULTS[k])
        self._shooting_hand.set(DEFAULTS["shooting_hand"])
        self._trigger_mode.set(DEFAULTS["trigger_mode"])
        self._update_ui_state()
        self._autosave()


# ── Module-level save helper (slider/checkbox commands) ───────────────────────
_app: "App | None" = None


def _save_after():
    if _app is not None:
        _app._autosave()


if __name__ == "__main__":
    import socket
    try:
        # Bind to a local port to prevent multiple GUI instances from running at the same time
        _lock_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        _lock_socket.bind(("127.0.0.1", 18332))
    except OSError:
        # Port already bound, exit silently
        sys.exit(0)

    try:
        _app = App()
        _app.mainloop()
    except BaseException as e:
        import traceback
        _dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(_dir, "crash_log.txt"), "w", encoding="utf-8") as f:
            traceback.print_exc(file=f)
        raise e
