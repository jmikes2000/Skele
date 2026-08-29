# -*- coding: utf-8 -*-
"""
gui.py  –  AA Games Settings GUI
python gui.py

Two-column wide layout. Auto-saves to settings.json — ball.py hot-reloads every 60 frames.
"""

import json, os, glob as _glob, threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

_DIR          = os.path.dirname(os.path.abspath(__file__))
SETTINGS_FILE = os.path.join(_DIR, "settings.json")

DEFAULTS = {
    "model_path":      "",
    "pose_path":       "",
    "ball_conf":       0.50,
    "stamina_conf":    0.40,
    "pose_conf":       0.50,
    "show_ball":       True,
    "show_stamina":    True,
    "show_pose":       True,
    "show_roi":        True,
    "show_player_box": True,
    "show_trigger":    True,
    "trigger_offset":  0,
    "trigger_height":  12,
    "tempo_ms":        45,
    "shot_mode":       0,
    "stamina_on":      0,
}

BG      = "#0e0e1a"
BG2     = "#1a1a2e"
BG3     = "#12122a"
ACCENT  = "#00e7ff"
DIM     = "#44446a"
FG      = "#ccccdd"
GREEN   = "#00ff88"
ORANGE  = "#ff9900"
PINK    = "#ff50c8"
CYAN    = "#00e7ff"
RED     = "#ff3355"
YELLOW  = "#ffdd00"
DIVIDER = "#1e1e3a"


def load_settings() -> dict:
    try:
        with open(SETTINGS_FILE) as f:
            d = json.loads(f.read().strip())
        for k, v in DEFAULTS.items():
            d.setdefault(k, v)
        return d
    except Exception:
        return dict(DEFAULTS)


def save_settings(data: dict):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def _section(parent, text, fg=DIM):
    tk.Frame(parent, bg=DIVIDER, height=1).pack(fill="x", pady=(14, 3))
    tk.Label(parent, text=text, bg=BG, fg=fg,
             font=("Consolas", 7, "bold")).pack(anchor="w")


def _slider(parent, key, label, lo, hi, val, vars_dict, save_fn, accent=ACCENT):
    v = tk.DoubleVar(value=round(float(val), 3))
    vars_dict[key] = v
    row = tk.Frame(parent, bg=BG)
    row.pack(fill="x", pady=(4, 0))
    hdr = tk.Frame(row, bg=BG)
    hdr.pack(fill="x")
    tk.Label(hdr, text=label, bg=BG, fg=FG,
             font=("Consolas", 9), anchor="w").pack(side="left")
    lbl = tk.Label(hdr, text=f"{val:.2f}", bg=BG, fg=accent,
                   font=("Consolas", 9, "bold"), width=6, anchor="e")
    lbl.pack(side="right")
    def _upd(_e=None, _l=lbl, _v=v):
        _l.config(text=f"{_v.get():.2f}")
        save_fn()
    ttk.Scale(row, from_=lo, to=hi, variable=v,
              orient="horizontal", command=_upd).pack(fill="x", pady=(2, 0))


class App(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("AA Games  –  YOLOv11 Control Panel")
        self.resizable(True, True)
        self.configure(bg=BG)
        self.geometry("920x720")

        st = ttk.Style(self)
        st.theme_use("clam")
        st.configure(".",            background=BG, foreground=FG, font=("Consolas", 9))
        st.configure("TFrame",       background=BG)
        st.configure("TScale",       background=BG, troughcolor=BG2, sliderlength=16)
        st.configure("TButton",      background="#1e1e3a", foreground=FG,
                     padding=(6, 4), relief="flat")
        st.map("TButton",            background=[("active", "#2a2a50")])
        st.configure("TCheckbutton", background=BG, foreground=FG)
        st.map("TCheckbutton",       background=[("active", BG)])
        st.configure("TRadiobutton", background=BG, foreground=FG)
        st.map("TRadiobutton",       background=[("active", BG)])

        cfg        = load_settings()
        self._vars = {}
        s          = self._autosave

        # ── Header ────────────────────────────────────────────────────────────
        hdr = tk.Frame(self, bg=BG)
        hdr.pack(fill="x", padx=24, pady=(14, 4))
        tk.Label(hdr, text="AA Games", bg=BG, fg=ACCENT,
                 font=("Consolas", 20, "bold")).pack(side="left")
        tk.Label(hdr, text="YOLOv11  ·  TensorRT  ·  Ball + Stamina + Pose",
                 bg=BG, fg=DIM, font=("Consolas", 9)).pack(side="left", padx=16)

        self._status = tk.StringVar(value="")
        tk.Label(hdr, textvariable=self._status, bg=BG, fg=GREEN,
                 font=("Consolas", 9, "bold")).pack(side="right")

        tk.Frame(self, bg=DIVIDER, height=1).pack(fill="x")

        # ── Two-column body ───────────────────────────────────────────────────
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=0)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)

        left  = tk.Frame(body, bg=BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(24, 12), pady=8)
        right = tk.Frame(body, bg=BG)
        right.grid(row=0, column=1, sticky="nsew", padx=(12, 24), pady=8)

        # ══════════════════════════════════════════════
        #  LEFT COLUMN
        # ══════════════════════════════════════════════

        # ── MODELS ────────────────────────────────────
        _section(left, "MODELS")
        self._det_var  = tk.StringVar(value=cfg.get("model_path", ""))
        self._pose_var = tk.StringVar(value=cfg.get("pose_path",  ""))
        self._vars["model_path"] = self._det_var
        self._vars["pose_path"]  = self._pose_var

        # Detection row — Browse + Auto
        r = tk.Frame(left, bg=BG)
        r.pack(fill="x", pady=(5, 0))
        tk.Label(r, text="Detection engine", bg=BG, fg=FG,
                 font=("Consolas", 9), width=18, anchor="w").pack(side="left")
        ttk.Button(r, text="Browse", command=self._browse_det).pack(side="right", padx=(4, 0))
        ttk.Button(r, text="Auto",   command=self._auto_det).pack(side="right")
        tk.Label(left, textvariable=self._det_var, bg=BG2, fg=GREEN,
                 font=("Consolas", 7), anchor="w",
                 wraplength=340, justify="left",
                 padx=3, pady=2).pack(fill="x", pady=(2, 0))

        # Pose row — Browse + Auto + Download
        r2 = tk.Frame(left, bg=BG)
        r2.pack(fill="x", pady=(5, 0))
        tk.Label(r2, text="Pose engine", bg=BG, fg=FG,
                 font=("Consolas", 9), width=18, anchor="w").pack(side="left")
        ttk.Button(r2, text="Browse",   command=self._browse_pose).pack(side="right", padx=(4, 0))
        ttk.Button(r2, text="Auto",     command=self._auto_pose).pack(side="right")
        ttk.Button(r2, text="Download", command=self._download_pose).pack(side="right", padx=(0, 4))
        tk.Label(left, textvariable=self._pose_var, bg=BG2, fg=GREEN,
                 font=("Consolas", 7), anchor="w",
                 wraplength=340, justify="left",
                 padx=3, pady=2).pack(fill="x", pady=(2, 0))

        # ── CONFIDENCE ────────────────────────────────
        _section(left, "CONFIDENCE")
        _slider(left, "ball_conf",    "Ball",     0.05, 0.95, cfg["ball_conf"],    self._vars, s, ORANGE)
        _slider(left, "stamina_conf", "Stamina",  0.05, 0.95, cfg["stamina_conf"], self._vars, s, PINK)
        _slider(left, "pose_conf",    "Skeleton", 0.05, 0.95, cfg["pose_conf"],    self._vars, s, CYAN)

        # ── DRAW OVERLAYS ─────────────────────────────
        _section(left, "DRAW OVERLAYS")
        checks = [
            ("show_ball",       "Draw ball"),
            ("show_stamina",    "Draw stamina bar"),
            ("show_pose",       "Draw skeleton"),
            ("show_roi",        "Draw ROI border"),
            ("show_player_box", "Draw player box"),
            ("show_trigger",    "Draw trigger box"),
        ]
        # two columns of checkboxes
        cbox_frame = tk.Frame(left, bg=BG)
        cbox_frame.pack(fill="x", pady=(4,0))
        for i, (key, label) in enumerate(checks):
            v = tk.BooleanVar(value=bool(cfg.get(key, True)))
            self._vars[key] = v
            col = i % 2
            row = i // 2
            ttk.Checkbutton(cbox_frame, text=label, variable=v,
                            command=s).grid(row=row, column=col,
                                            sticky="w", padx=(0,12), pady=2)

        # ══════════════════════════════════════════════
        #  RIGHT COLUMN
        # ══════════════════════════════════════════════

        # ── SHOT MODE ─────────────────────────────────
        _section(right, "SHOT MODE", fg=ORANGE)
        mode_var = tk.IntVar(value=int(cfg.get("shot_mode", 0)))
        self._vars["shot_mode"] = mode_var
        modes = [
            (0, "Rhythm  (EV stick timing)"),
            (1, "Button  (Square press)"),
            (2, "Stick   (Centre stick)"),
            (3, "Straight (R2 + EV stick)"),
        ]
        mode_frame = tk.Frame(right, bg=BG)
        mode_frame.pack(fill="x", pady=(6,0))
        for val, label in modes:
            ttk.Radiobutton(mode_frame, text=label, variable=mode_var,
                            value=val, command=s).pack(anchor="w", pady=3)

        # ── TEMPO ─────────────────────────────────────
        _section(right, "TEMPO  (gcv[1]  →  wait before full stick)", fg=YELLOW)
        tk.Label(right,
                 text="Lower = earlier release  ·  range 1-100 ms",
                 bg=BG, fg=DIM, font=("Consolas", 7)).pack(anchor="w")

        tempo_row = tk.Frame(right, bg=BG)
        tempo_row.pack(fill="x", pady=(6,0))
        self._tempo_var = tk.IntVar(value=int(cfg.get("tempo_ms", 45)))
        self._vars["tempo_ms"] = self._tempo_var
        tk.Label(tempo_row, text="Tempo (ms)", bg=BG, fg=FG,
                 font=("Consolas", 9)).pack(side="left")
        self._tempo_lbl = tk.Label(tempo_row,
                                   text=str(self._tempo_var.get()),
                                   bg=BG, fg=YELLOW,
                                   font=("Consolas", 13, "bold"), width=5)
        self._tempo_lbl.pack(side="right")

        ttk.Scale(right, from_=1, to=100, variable=self._tempo_var,
                  orient="horizontal",
                  command=self._tempo_update).pack(fill="x", pady=(4,0))

        btn_row = tk.Frame(right, bg=BG)
        btn_row.pack(pady=(4,0))
        for delta, lbl in [(-5,"−5"),(-1,"−1"),(1,"+1"),(5,"+5")]:
            ttk.Button(btn_row, text=lbl, width=4,
                       command=lambda d=delta: self._tempo_nudge(d)
                       ).pack(side="left", padx=2)

        # ── INFINITE STAMINA ──────────────────────────
        _section(right, "INFINITE STAMINA  (gcv[3])", fg=GREEN)
        stam_var = tk.BooleanVar(value=bool(int(cfg.get("stamina_on", 0))))
        self._vars["stamina_on"] = stam_var
        ttk.Checkbutton(right,
                        text="Enable infinite stamina (scales LX/LY to 70%)",
                        variable=stam_var, command=s).pack(anchor="w", pady=(6,0))

        # ── SHOT TRIGGER BOX ─────────────────────────────────────────────────
        _section(right, "SHOT TRIGGER BOX", fg=RED)
        tk.Label(right,
                 text="Offset: moves box UP (−) or DOWN (+)\n"
                      "Height: half-height of the box in pixels",
                 bg=BG, fg=DIM, font=("Consolas", 7), justify="left").pack(anchor="w")

        # — Offset —
        trig_row = tk.Frame(right, bg=BG)
        trig_row.pack(fill="x", pady=(6,0))
        self._trig_var = tk.IntVar(value=int(cfg.get("trigger_offset", 0)))
        self._vars["trigger_offset"] = self._trig_var
        tk.Label(trig_row, text="Offset (px)", bg=BG, fg=FG,
                 font=("Consolas", 9)).pack(side="left")
        self._trig_lbl = tk.Label(trig_row, text=str(self._trig_var.get()),
                                  bg=BG, fg=RED,
                                  font=("Consolas", 13, "bold"), width=6)
        self._trig_lbl.pack(side="right")

        ttk.Scale(right, from_=-200, to=400, variable=self._trig_var,
                  orient="horizontal",
                  command=self._trig_update).pack(fill="x", pady=(4,0))

        tbtn_row = tk.Frame(right, bg=BG)
        tbtn_row.pack(pady=(4,0))
        for delta, lbl in [(-5,"−5"),(-1,"−1"),(1,"+1"),(5,"+5")]:
            ttk.Button(tbtn_row, text=lbl, width=4,
                       command=lambda d=delta: self._trig_nudge(d)
                       ).pack(side="left", padx=2)
        tk.Label(right,
                 text="← earlier (up)                    later (down) →",
                 bg=BG, fg=DIM, font=("Consolas", 7)).pack(anchor="w", pady=(2,0))

        # — Height —
        th_row = tk.Frame(right, bg=BG)
        th_row.pack(fill="x", pady=(8,0))
        self._th_var = tk.IntVar(value=int(cfg.get("trigger_height", 12)))
        self._vars["trigger_height"] = self._th_var
        tk.Label(th_row, text="Height (px)", bg=BG, fg=FG,
                 font=("Consolas", 9)).pack(side="left")
        self._th_lbl = tk.Label(th_row, text=str(self._th_var.get()),
                                bg=BG, fg=ORANGE,
                                font=("Consolas", 13, "bold"), width=5)
        self._th_lbl.pack(side="right")

        ttk.Scale(right, from_=1, to=100, variable=self._th_var,
                  orient="horizontal",
                  command=self._th_update).pack(fill="x", pady=(4,0))

        thbtn_row = tk.Frame(right, bg=BG)
        thbtn_row.pack(pady=(4,0))
        for delta, lbl in [(-5,"−5"),(-1,"−1"),(1,"+1"),(5,"+5")]:
            ttk.Button(thbtn_row, text=lbl, width=4,
                       command=lambda d=delta: self._th_nudge(d)
                       ).pack(side="left", padx=2)

        # ── BOTTOM BUTTONS ────────────────────────────
        tk.Frame(self, bg=DIVIDER, height=1).pack(fill="x", padx=16)
        bf = tk.Frame(self, bg=BG)
        bf.pack(pady=(6, 12))
        ttk.Button(bf, text="Reset defaults",
                   command=self._reset).pack(side="left", padx=8)
        ttk.Button(bf, text="Quit",
                   command=self.destroy).pack(side="left", padx=8)

    # ── Model browse / auto-detect ────────────────────────────────────────────
    def _browse(self, var):
        p = filedialog.askopenfilename(
            title="Select model file", initialdir=_DIR,
            filetypes=[("Engine/ONNX","*.engine *.onnx"),
                       ("PyTorch","*.pt"),("All","*.*")])
        if p:
            var.set(p); self._autosave()

    def _browse_det(self):  self._browse(self._det_var)
    def _browse_pose(self): self._browse(self._pose_var)

    def _auto_det(self):
        for ext in ("engine","onnx","pt"):
            hits = _glob.glob(os.path.join(_DIR,"runs","**",f"best.{ext}"), recursive=True)
            if hits:
                self._det_var.set(max(hits, key=os.path.getmtime))
                self._autosave()
                self._status.set("✔ detection found")
                self.after(2000, lambda: self._status.set(""))
                return
        messagebox.showinfo("Not found","No detection model in runs/.")

    def _auto_pose(self):
        hits = (_glob.glob(os.path.join(_DIR,"*pose*.engine")) +
                _glob.glob(os.path.join(_DIR,"*pose*.onnx")))
        if hits:
            self._pose_var.set(max(hits, key=os.path.getmtime))
            self._autosave()
            self._status.set("✔ pose found")
            self.after(2000, lambda: self._status.set(""))
        else:
            messagebox.showinfo("Not found","No pose engine found.")

    def _download_pose(self):
        """Run the Skele.py auto-download in a background thread so the GUI stays responsive."""
        self._status.set("⬇ downloading pose …")

        def _worker():
            try:
                import sys as _sys
                _sys.path.insert(0, _DIR)
                from Skele import _download_pose_engine
                result = _download_pose_engine()
                if result:
                    # update GUI on main thread
                    self.after(0, lambda: self._pose_var.set(result))
                    self.after(0, self._autosave)
                    self.after(0, lambda: self._status.set("✔ pose engine ready"))
                    self.after(3000, lambda: self._status.set(""))
                else:
                    self.after(0, lambda: self._status.set("✖ download failed — check console"))
                    self.after(4000, lambda: self._status.set(""))
            except Exception as exc:
                self.after(0, lambda: self._status.set(f"✖ {exc}"))
                self.after(4000, lambda: self._status.set(""))

        threading.Thread(target=_worker, daemon=True).start()

    # ── Tempo helpers ─────────────────────────────────────────────────────────
    def _tempo_update(self, *_):
        v = max(1, min(100, int(self._tempo_var.get())))
        self._tempo_var.set(v)
        self._tempo_lbl.config(text=str(v))
        self._autosave()

    def _tempo_nudge(self, delta):
        v = max(1, min(100, int(self._tempo_var.get()) + delta))
        self._tempo_var.set(v)
        self._tempo_lbl.config(text=str(v))
        self._autosave()

    # ── Trigger offset helpers ────────────────────────────────────────────────
    def _trig_update(self, *_):
        v = int(self._trig_var.get())
        self._trig_lbl.config(text=str(v))
        self._autosave()

    def _trig_nudge(self, delta):
        v = max(-200, min(400, int(self._trig_var.get()) + delta))
        self._trig_var.set(v)
        self._trig_lbl.config(text=str(v))
        self._autosave()

    # ── Trigger height helpers ────────────────────────────────────────────────
    def _th_update(self, *_):
        v = max(1, min(100, int(self._th_var.get())))
        self._th_var.set(v)
        self._th_lbl.config(text=str(v))
        self._autosave()

    def _th_nudge(self, delta):
        v = max(1, min(100, int(self._th_var.get()) + delta))
        self._th_var.set(v)
        self._th_lbl.config(text=str(v))
        self._autosave()

    # ── Persistence ───────────────────────────────────────────────────────────
    def _get_data(self) -> dict:
        existing = load_settings()
        for k, v in self._vars.items():
            raw = v.get()
            if isinstance(v, tk.BooleanVar):
                existing[k] = 1 if raw else 0
            elif k in ("model_path", "pose_path"):
                existing[k] = str(raw)
            elif k in ("trigger_offset", "trigger_height", "tempo_ms", "shot_mode", "stamina_on"):
                existing[k] = int(raw)
            else:
                existing[k] = round(float(raw), 3)
        return existing

    def _autosave(self, *_):
        try:
            save_settings(self._get_data())
            self._status.set("✔ saved")
            self.after(1200, lambda: self._status.set(""))
        except Exception as e:
            self._status.set(f"error: {e}")

    def _reset(self):
        for k, v in self._vars.items():
            if k in DEFAULTS:
                v.set(DEFAULTS[k])
        self._autosave()


if __name__ == "__main__":
    App().mainloop()
