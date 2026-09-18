"""
Tactical Radar HUD Overlay (Multi-Source Acoustic Display)
- 360-degree continuous circular acoustic heat-ring (16 independent sectors)
- Simultaneous detection and display of multiple sound sources (e.g. Left footsteps + Right gunfire)
- Independent phosphor decay per direction
- Draggable, Always-on-top, Adjustable Opacity
"""

import math
import time
import tkinter as tk
from tkinter import ttk


class TacticalRadarApp:
    def __init__(self, root, on_mode_change=None, on_filter_change=None):
        self.root = root
        self.root.title("三角洲多方向听障雷达 (MVP)")
        self.root.geometry("340x510")
        self.root.minsize(320, 480)
        self.root.configure(bg="#0B0E14")
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.92)

        # Callbacks
        self.on_mode_change = on_mode_change
        self.on_filter_change = on_filter_change

        # Display state for multi-source visualization
        self.NUM_SECTORS = 16
        self.sector_decays = [0.0] * self.NUM_SECTORS
        self.active_peaks = []
        self.db_level = -80.0
        self.status_text = "等待系统声音..."

        # Settings: Default to 'live' so user hears real game/pc audio immediately
        self.mode_var = tk.StringVar(value="live")       # "live" or "mock"
        self.filter_var = tk.BooleanVar(value=True)     # Footstep bandpass filter
        self.gain_var = tk.DoubleVar(value=1.5)         # Sensitivity multiplier
        self.threshold_var = tk.DoubleVar(value=-50.0)  # Noise gate (dB)
        self.opacity_var = tk.DoubleVar(value=0.92)

        # UI Setup
        self._setup_styles()
        self._setup_titlebar()
        self._setup_canvas()
        self._setup_controls()

        # Canvas Dimensions
        self.cx = 150
        self.cy = 150
        self.radius = 120

        # Mouse Dragging Support
        self._drag_start_x = 0
        self._drag_start_y = 0
        self.title_frame.bind("<Button-1>", self._start_drag)
        self.title_frame.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<Button-1>", self._start_drag)
        self.canvas.bind("<B1-Motion>", self._on_drag)

        # Periodic Render Loop (~60 FPS)
        self._render_loop()

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TLabel", background="#0B0E14", foreground="#00FFAA", font=("Consolas", 9))
        style.configure("TScale", background="#0B0E14")
        style.configure("TCheckbutton", background="#0B0E14", foreground="#A0AEC0", font=("Consolas", 9))
        style.map("TCheckbutton", foreground=[("active", "#00FFAA")])

    def _setup_titlebar(self):
        self.title_frame = tk.Frame(self.root, bg="#131924", height=32)
        self.title_frame.pack(fill=tk.X, side=tk.TOP)

        title_lbl = tk.Label(
            self.title_frame,
            text="DELTA FORCE · 360° MULTI-SOUND RADAR",
            bg="#131924",
            fg="#00E5FF",
            font=("Segoe UI", 8, "bold")
        )
        title_lbl.pack(side=tk.LEFT, padx=10, pady=4)

        self.pin_lbl = tk.Label(
            self.title_frame,
            text="[TOP]",
            bg="#131924",
            fg="#00FFAA",
            font=("Consolas", 8, "bold")
        )
        self.pin_lbl.pack(side=tk.RIGHT, padx=8)

    def _setup_canvas(self):
        self.canvas = tk.Canvas(
            self.root,
            width=300,
            height=300,
            bg="#080C10",
            highlightthickness=1,
            highlightbackground="#1E2A38"
        )
        self.canvas.pack(pady=6, padx=10)

    def _setup_controls(self):
        ctrl_frame = tk.Frame(self.root, bg="#0B0E14")
        ctrl_frame.pack(fill=tk.X, padx=14, pady=2)

        # Row 1: Mode Switch & Filter Toggle
        row1 = tk.Frame(ctrl_frame, bg="#0B0E14")
        row1.pack(fill=tk.X, pady=2)

        btn_live = tk.Radiobutton(
            row1, text="实时监听(游戏实战)", variable=self.mode_var, value="live",
            command=self._on_mode_toggled, bg="#0B0E14", fg="#00E5FF",
            selectcolor="#131924", activebackground="#0B0E14", activeforeground="#00E5FF",
            font=("Microsoft YaHei", 8, "bold")
        )
        btn_live.pack(side=tk.LEFT)

        btn_mock = tk.Radiobutton(
            row1, text="离线双声源演示", variable=self.mode_var, value="mock",
            command=self._on_mode_toggled, bg="#0B0E14", fg="#64748B",
            selectcolor="#131924", activebackground="#0B0E14", activeforeground="#00FFAA",
            font=("Microsoft YaHei", 8)
        )
        btn_mock.pack(side=tk.LEFT, padx=6)

        chk_filter = tk.Checkbutton(
            row1, text="脚步频段增强", variable=self.filter_var,
            command=self._on_filter_toggled, bg="#0B0E14", fg="#E2E8F0",
            selectcolor="#131924", activebackground="#0B0E14", activeforeground="#00FFAA",
            font=("Microsoft YaHei", 8)
        )
        chk_filter.pack(side=tk.RIGHT)

        # Row 2: Sensitivity (Gain) Slider
        row2 = tk.Frame(ctrl_frame, bg="#0B0E14")
        row2.pack(fill=tk.X, pady=3)
        lbl_gain = tk.Label(row2, text="灵敏度:", bg="#0B0E14", fg="#94A3B8", font=("Microsoft YaHei", 8))
        lbl_gain.pack(side=tk.LEFT)
        scale_gain = ttk.Scale(row2, from_=0.5, to=4.0, variable=self.gain_var, orient=tk.HORIZONTAL)
        scale_gain.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)

        # Row 3: Noise Gate (Squelch threshold)
        row3 = tk.Frame(ctrl_frame, bg="#0B0E14")
        row3.pack(fill=tk.X, pady=3)
        lbl_gate = tk.Label(row3, text="降噪门限:", bg="#0B0E14", fg="#94A3B8", font=("Microsoft YaHei", 8))
        lbl_gate.pack(side=tk.LEFT)
        scale_gate = ttk.Scale(row3, from_=-65.0, to=-20.0, variable=self.threshold_var, orient=tk.HORIZONTAL)
        scale_gate.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)

        # Usage Hint Label
        self.hint_lbl = tk.Label(
            ctrl_frame,
            text="💡 支持多方向同时指示：左边脚步与右边枪声将同时亮起！",
            bg="#0B0E14",
            fg="#38BDF8",
            font=("Microsoft YaHei", 7)
        )
        self.hint_lbl.pack(fill=tk.X, pady=2)

        # Status Bar at bottom
        self.status_lbl = tk.Label(
            self.root,
            text="状态: 准备就绪",
            bg="#0B0E14",
            fg="#64748B",
            font=("Microsoft YaHei", 8),
            anchor=tk.W
        )
        self.status_lbl.pack(fill=tk.X, side=tk.BOTTOM, padx=14, pady=4)

    def _start_drag(self, event):
        self._drag_start_x = event.x
        self._drag_start_y = event.y

    def _on_drag(self, event):
        x = self.root.winfo_x() + (event.x - self._drag_start_x)
        y = self.root.winfo_y() + (event.y - self._drag_start_y)
        self.root.geometry(f"+{x}+{y}")

    def _on_mode_toggled(self):
        mode = self.mode_var.get()
        if self.on_mode_change:
            self.on_mode_change(mode)

    def _on_filter_toggled(self):
        enabled = self.filter_var.get()
        if self.on_filter_change:
            self.on_filter_change(enabled)

    def update_multi_data(self, sector_energies, peaks, db_level, status=None):
        """Thread-safe update entry for multi-source acoustic data."""
        self.db_level = db_level
        if status:
            self.status_text = status

        # Check noise gate threshold
        if db_level < self.threshold_var.get():
            self.active_peaks = []
            return

        gain = self.gain_var.get()
        for i, val in enumerate(sector_energies):
            scaled_val = min(1.0, val * gain)
            if scaled_val > self.sector_decays[i]:
                self.sector_decays[i] = scaled_val

        self.active_peaks = peaks

    def _render_loop(self):
        """Periodic redraw loop running at ~60 FPS."""
        # Smooth decay across all 16 sectors
        for i in range(self.NUM_SECTORS):
            if self.sector_decays[i] > 0.0:
                self.sector_decays[i] = max(0.0, self.sector_decays[i] - 0.045)

        self._draw_radar()
        self.root.after(16, self._render_loop)

    def _draw_radar(self):
        c = self.canvas
        c.delete("all")
        cx, cy, r = self.cx, self.cy, self.radius

        # 1. Background Grid & Range Rings
        c.create_oval(cx - r, cy - r, cx + r, cy + r, outline="#162536", width=2)
        c.create_oval(cx - r * 0.75, cy - r * 0.75, cx + r * 0.75, cy + r * 0.75, outline="#121D2B", width=1, dash=(3, 3))
        c.create_oval(cx - r * 0.5, cy - r * 0.5, cx + r * 0.5, cy + r * 0.5, outline="#162536", width=1)
        c.create_oval(cx - r * 0.25, cy - r * 0.25, cx + r * 0.25, cy + r * 0.25, outline="#121D2B", width=1, dash=(3, 3))

        # Radial Crosshairs & 45-degree angle lines
        c.create_line(cx - r, cy, cx + r, cy, fill="#162536", width=1)
        c.create_line(cx, cy - r, cx, cy + r, fill="#162536", width=1)

        d_rad = r * 0.707
        c.create_line(cx - d_rad, cy - d_rad, cx + d_rad, cy + d_rad, fill="#101824", width=1)
        c.create_line(cx - d_rad, cy + d_rad, cx + d_rad, cy - d_rad, fill="#101824", width=1)

        # Cardinal Compass Points
        c.create_text(cx, cy - r + 12, text="N (前)", fill="#00E5FF", font=("Consolas", 8, "bold"))
        c.create_text(cx + r - 14, cy, text="E (右)", fill="#64748B", font=("Consolas", 8))
        c.create_text(cx, cy + r - 12, text="S (后)", fill="#64748B", font=("Consolas", 8))
        c.create_text(cx - r + 14, cy, text="W (左)", fill="#64748B", font=("Consolas", 8))

        # Center Player Chevron (Facing North)
        c.create_polygon(cx, cy - 8, cx - 6, cy + 6, cx, cy + 3, cx + 6, cy + 6, fill="#00FFAA", outline="")

        # 2. Outer 360° Multi-Directional Acoustic Energy Arc Segments (16 Sectors)
        for i in range(self.NUM_SECTORS):
            decay = self.sector_decays[i]
            if decay <= 0.03:
                continue

            # Color by intensity
            if decay < 0.4:
                glow_color = "#00FFAA"
            elif decay < 0.7:
                glow_color = "#FFD600"
            else:
                glow_color = "#FF1744"

            # Dynamic outer radius expansion
            arc_r = r - 5 + int(12 * decay)
            # Center of sector i (0 is North, 90 is East)
            sector_deg = i * 22.5
            # Tkinter: 0 is 3 o'clock counter-clockwise
            tk_center = (90.0 - sector_deg) % 360.0
            arc_start = tk_center - 10.0
            arc_width = max(2, int(7 * decay))

            c.create_arc(
                cx - arc_r, cy - arc_r, cx + arc_r, cy + arc_r,
                start=arc_start, extent=20.0,
                outline=glow_color, width=arc_width, style=tk.ARC
            )

        # 3. Draw Threat Markers & Radial Beams for Each Detected Peak Source
        if self.active_peaks:
            for p_idx, peak in enumerate(self.active_peaks):
                deg = peak["angle"]
                mag = peak["magnitude"]
                rad = math.radians(deg)

                dir_x = math.sin(rad)
                dir_y = -math.cos(rad)

                # Threat color
                if mag < 0.4:
                    t_color = "#00FFAA"
                elif mag < 0.7:
                    t_color = "#FFD600"
                else:
                    t_color = "#FF1744"

                # Radial vector ray
                dist = r * (0.35 + 0.55 * mag)
                tx = cx + dir_x * dist
                ty = cy + dir_y * dist

                c.create_line(
                    cx + dir_x * 12, cy + dir_y * 12,
                    tx, ty,
                    fill=t_color, width=2
                )

                # Blip dot
                blip_r = 4 + int(mag * 5)
                c.create_oval(
                    tx - blip_r, ty - blip_r, tx + blip_r, ty + blip_r,
                    fill=t_color, outline="#FFFFFF", width=1
                )

                # Outer tactical direction arrow / pip on the ring
                arrow_x = cx + dir_x * (r - 2)
                arrow_y = cy + dir_y * (r - 2)
                c.create_oval(
                    arrow_x - 3, arrow_y - 3, arrow_x + 3, arrow_y + 3,
                    fill="#FFFFFF", outline=t_color, width=1
                )

                # Degree text tag
                tag_x = cx + dir_x * (r - 18)
                tag_y = cy + dir_y * (r - 18)
                c.create_text(
                    tag_x, tag_y,
                    text=f"{int(deg)}°",
                    fill=t_color, font=("Consolas", 7, "bold")
                )

        # 4. HUD Digital Readout
        if self.active_peaks:
            if len(self.active_peaks) == 1:
                p0 = self.active_peaks[0]
                dir_text = f"🎯 单声源: {int(p0['angle']):03d}° ({p0['compass']})"
            else:
                source_strs = [f"{int(p['angle'])}°({p['compass']})" for p in self.active_peaks[:2]]
                dir_text = f"🎯 多声源: {' | '.join(source_strs)}"
            dir_color = "#00FFAA"
            pwr_text = f"PWR: {self.db_level:5.1f} dB"
            pwr_color = "#00E5FF"
        else:
            dir_text = "🎯 待命 (耳机监听中)"
            pwr_text = "PWR: 静音待命"
            dir_color = "#64748B"
            pwr_color = "#64748B"

        c.create_text(
            15, 15,
            text=dir_text,
            anchor=tk.W, fill=dir_color, font=("Consolas", 8, "bold")
        )

        c.create_text(
            285, 15,
            text=pwr_text,
            anchor=tk.E, fill=pwr_color, font=("Consolas", 8, "bold")
        )

        # Update Status Text
        mode_name = "离线多声源演示" if self.mode_var.get() == "mock" else "实时监听耳机"
        self.status_lbl.config(text=f"模式: {mode_name} | {self.status_text}")
