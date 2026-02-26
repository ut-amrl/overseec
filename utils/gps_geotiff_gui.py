#!/usr/bin/env python3
"""
gps_geotiff_gui.py  —  Tkinter GUI for multi-track GPS-on-GeoTIFF plotting.

Workflow
--------
1.  Select a GeoTIFF → loaded once and locked.
2.  Select a ROS2 bag → choose style options → Load Bag.
3.  Batch mode:   click "Add Track to Plot" to render all fixes.
    Interactive:  step through fixes (Space / Next Point),
                  optionally set a start marker (Set Start button),
                  then "Commit Track [E]" to add the start→current segment.
4.  Repeat steps 2-3 for as many bags as desired (tracks accumulate).
5.  Click "Save Plot [R]" (or press R) at any time to write the PNG.

New features
------------
  • Display Size panel (unlocked after GeoTIFF loads) — set the canvas area
    to the GeoTIFF's native pixel dimensions, or type in a custom W × H and
    click Apply.  Window remains freely resizable by mouse drag after apply.
  • Start / End marker colour can be set to "None" to skip drawing that marker.
  • "Edit Labels…" button lets you rename any committed track's legend entry.

Keyboard shortcuts (canvas must be clicked once to receive focus)
-----------------------------------------------------------------
  Space  →  Interactive: advance one GPS fix
  E      →  Interactive: commit current segment to the composite plot
  R      →  Save the composite plot to the output file (all modes)

Dependencies
------------
    pip install rosbags rasterio matplotlib numpy pyproj
"""

import threading
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import matplotlib.patches as mpatches

# ── Colour palette ────────────────────────────────────────────────────────────

COLOR_NAMES = ["Red", "Green", "Blue", "Orange", "Yellow", "Violet"]

# "None" is offered only for start/end markers (means "don't draw that marker")
MARKER_COLOR_NAMES = COLOR_NAMES + ["None"]

COLOR_RGB: dict[str, tuple] = {
    "Red":    (1.0, 0.00, 0.00),
    "Green":  (0.0, 1.00, 0.00),
    "Blue":   (0.0, 0.00, 1.00),
    "Orange": (1.0, 0.50, 0.00),
    "Yellow": (1.0, 1.00, 0.00),
    "Violet": (0.5, 0.00, 1.00),
}
COLOR_HEX: dict[str, str] = {
    "Red":    "#ff0000",
    "Green":  "#00ff00",
    "Blue":   "#0000ff",
    "Orange": "#ff8000",
    "Yellow": "#ffff00",
    "Violet": "#8000ff",
    "None":   "#cccccc",
}

_TRACK_COLOR_CYCLE = ["Green", "Blue", "Orange", "Violet", "Yellow", "Red"]
COLORBY_OPTIONS    = ["time", "altitude", "speed", "solid color"]


# ── Back-end helpers ──────────────────────────────────────────────────────────

def read_navsatfix(bag_path: str, topic: str, downsample: int = 1,
                   status_cb=None):
    try:
        from rosbags.rosbag2 import Reader
        from rosbags.typesys import Stores, get_typestore
    except ImportError:
        raise ImportError("rosbags is not installed.\nRun:  pip install rosbags")
    try:
        import rasterio
        from pyproj import Transformer  # noqa: F401
    except ImportError:
        raise ImportError(
            "rasterio or pyproj not installed.\n"
            "Run:  pip install rasterio pyproj")

    typestore = get_typestore(Stores.ROS2_JAZZY)
    timestamps, lats, lons, alts, statuses = [], [], [], [], []

    with Reader(bag_path) as reader:
        available = [conn.topic for conn in reader.connections]
        if topic not in available:
            raise ValueError(
                f"Topic '{topic}' not found in bag.\n"
                f"Available topics:\n  " + "\n  ".join(available))
        connections = [c for c in reader.connections if c.topic == topic]
        for i, (conn, timestamp, rawdata) in enumerate(
                reader.messages(connections=connections)):
            if i % downsample != 0:
                continue
            msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
            if msg.status.status < 0:
                continue
            timestamps.append(timestamp * 1e-9)
            lats.append(msg.latitude)
            lons.append(msg.longitude)
            alts.append(msg.altitude)
            statuses.append(int(msg.status.status))
            if status_cb and i % 200 == 0:
                status_cb(f"Reading… {len(lats)} valid fixes (msg {i})")

    return (np.array(timestamps), np.array(lats), np.array(lons),
            np.array(alts), np.array(statuses))


def _load_raster(geotiff_path: str):
    """
    Returns (raster, img_extent, tiff_crs, is_rgb, px_width, px_height).
    """
    import rasterio
    with rasterio.open(geotiff_path) as ds:
        tiff_crs   = ds.crs.to_string()
        img_extent = [ds.bounds.left, ds.bounds.right,
                      ds.bounds.bottom, ds.bounds.top]
        px_w, px_h = ds.width, ds.height

        def _norm(a):
            a = a.astype(np.float32)
            lo, hi = a.min(), a.max()
            return (a - lo) / (hi - lo + 1e-9)

        if ds.count >= 3:
            rgb = np.dstack([_norm(ds.read(1)),
                             _norm(ds.read(2)),
                             _norm(ds.read(3))])
            return rgb, img_extent, tiff_crs, True, px_w, px_h
        return _norm(ds.read(1)), img_extent, tiff_crs, False, px_w, px_h


def _project_coords(lats, lons, tiff_crs):
    from pyproj import Transformer
    t = Transformer.from_crs("EPSG:4326", tiff_crs, always_xy=True)
    return t.transform(lons, lats)


def _draw_background(ax, raster, img_extent, is_rgb):
    kwargs = dict(extent=img_extent, origin="upper",
                  aspect="equal", interpolation="bilinear")
    ax.imshow(raster, cmap=(None if is_rgb else "gray"), **kwargs)


def _compute_color_values(timestamps, lats, lons, alts, colorby):
    if colorby == "altitude":
        return alts.copy(), "Altitude (m)"
    if colorby == "speed":
        R  = 6_371_000.0
        dt = np.diff(timestamps)
        phi1, phi2 = np.radians(lats[:-1]), np.radians(lats[1:])
        dphi, dlam = np.radians(np.diff(lats)), np.radians(np.diff(lons))
        a = (np.sin(dphi / 2) ** 2
             + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2)
        dist  = 2 * R * np.arcsin(np.sqrt(a))
        speed = np.where(dt > 0, dist / dt, 0.0)
        speed = np.append(speed, speed[-1] if len(speed) else 0.0)
        return speed, "Speed (m/s)"
    if colorby == "time":
        return timestamps - timestamps[0], "Elapsed time (s)"
    return None, None


def _color_or_none(name: str):
    """Return RGB tuple or None when name == 'None'."""
    return None if name == "None" else COLOR_RGB[name]


# ── Track drawing helper ──────────────────────────────────────────────────────

def _draw_track_on_ax(ax, xs, ys, colorby, color_values,
                      norm, cmap_, track_color,
                      start_color, end_color, label, zorder_base=2):
    """
    Draws one committed track onto ax.
    start_color / end_color may be None (that marker is simply omitted).
    Returns list of all artists created.
    """
    artists = []

    if colorby == "solid color":
        line, = ax.plot(xs, ys, color=track_color,
                        linewidth=1.5, alpha=0.85, zorder=zorder_base)
        sc = ax.scatter(xs, ys, color=track_color,
                        s=8, zorder=zorder_base + 1, linewidths=0)
        artists += [line, sc]
    else:
        line, = ax.plot(xs, ys, color="white",
                        linewidth=1.0, alpha=0.5, zorder=zorder_base)
        sc = ax.scatter(xs, ys, c=color_values, cmap=cmap_, norm=norm,
                        s=8, zorder=zorder_base + 1, linewidths=0)
        artists += [line, sc]

    if start_color is not None:
        artists.append(ax.scatter(
            [xs[0]], [ys[0]], color=start_color, s=90, marker="^",
            zorder=zorder_base + 3, label=f"▲ {label} start"))

    if end_color is not None:
        artists.append(ax.scatter(
            [xs[-1]], [ys[-1]], color=end_color, s=90, marker="s",
            zorder=zorder_base + 3, label=f"■ {label} end"))

    return artists


# ── Main GUI class ────────────────────────────────────────────────────────────

class GPSPlotterGUI:

    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("GPS on GeoTIFF Plotter")
        root.minsize(1200, 740)

        self._raster_data    = None   # (raster, extent, crs, is_rgb, pw, ph)
        self._geotiff_locked = False
        self._geotiff_px_w   = 0
        self._geotiff_px_h   = 0

        self._gps_data  = None
        self._projected = None

        self._committed_tracks: list[dict] = []
        self._track_count = 0
        self._istate: dict = {}
        self._colorbar = None

        self._build_ui()

    # ═══ Top-level layout ════════════════════════════════════════════════════

    def _build_ui(self):
        pane = tk.PanedWindow(self.root, orient=tk.HORIZONTAL,
                              sashwidth=6, sashrelief=tk.RIDGE, bg="#bbb")
        pane.pack(fill=tk.BOTH, expand=True)

        self._ctrl_outer   = ttk.Frame(pane, padding=(6, 6))
        self._canvas_frame = ttk.Frame(pane)
        pane.add(self._ctrl_outer,   minsize=300)
        pane.add(self._canvas_frame, minsize=680)

        self._build_controls(self._ctrl_outer)
        self._build_canvas(self._canvas_frame)

    # ── Scrollable control panel ──────────────────────────────────────────────

    def _build_controls(self, outer):
        cv = tk.Canvas(outer, borderwidth=0, highlightthickness=0)
        sb = ttk.Scrollbar(outer, orient="vertical", command=cv.yview)
        cv.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        cv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        inner  = ttk.Frame(cv)
        win_id = cv.create_window((0, 0), window=inner, anchor="nw")

        def _resize(e=None):
            cv.configure(scrollregion=cv.bbox("all"))
            cv.itemconfig(win_id, width=cv.winfo_width())

        inner.bind("<Configure>", _resize)
        cv.bind("<Configure>", _resize)
        cv.bind_all("<MouseWheel>",
                    lambda e: cv.yview_scroll(int(-1*(e.delta/120)), "units"))

        self._build_geotiff_section(inner)
        self._build_display_size_section(inner)
        self._build_bag_section(inner)
        self._build_track_style_section(inner)
        self._build_mode_section(inner)
        self._build_load_bag_section(inner)
        self._build_batch_section(inner)
        self._build_interactive_section(inner)
        self._build_time_range_section(inner)
        self._build_track_list_section(inner)
        self._build_save_section(inner)
        self._build_status_section(inner)

    def _section(self, parent, title):
        lf = ttk.LabelFrame(parent, text=f" {title} ", padding=6)
        lf.pack(fill=tk.X, pady=(0, 5))
        return lf

    # ── ① GeoTIFF ─────────────────────────────────────────────────────────────

    def _build_geotiff_section(self, parent):
        sec = self._section(parent, "① GeoTIFF  (select once, then locked)")

        self._geotiff_var = tk.StringVar()

        row = ttk.Frame(sec)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="File:", width=6, anchor="w").pack(side=tk.LEFT)
        self._geotiff_entry = ttk.Entry(row, textvariable=self._geotiff_var)
        self._geotiff_entry.pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        self._geotiff_btn = ttk.Button(row, text="…", width=3,
                                       command=self._browse_geotiff)
        self._geotiff_btn.pack(side=tk.LEFT)

        self._load_geotiff_btn = ttk.Button(
            sec, text="Load GeoTIFF", command=self._on_load_geotiff)
        self._load_geotiff_btn.pack(fill=tk.X, pady=(4, 0))

        self._geotiff_status_var = tk.StringVar(value="No GeoTIFF loaded.")
        ttk.Label(sec, textvariable=self._geotiff_status_var,
                  foreground="gray", wraplength=270,
                  justify=tk.LEFT).pack(anchor="w", pady=(4, 0))

    def _browse_geotiff(self):
        if self._geotiff_locked:
            return
        path = filedialog.askopenfilename(
            title="Select GeoTIFF",
            filetypes=[("GeoTIFF", "*.tif *.tiff"), ("All files", "*.*")])
        if path:
            self._geotiff_var.set(path)

    def _on_load_geotiff(self):
        path = self._geotiff_var.get().strip()
        if not path:
            messagebox.showwarning("No file", "Select a GeoTIFF file first.")
            return
        if not Path(path).exists():
            messagebox.showerror("Not found", f"File not found:\n{path}")
            return
        try:
            self._set_status("Loading GeoTIFF…")
            self._raster_data = _load_raster(path)
            raster, img_extent, tiff_crs, is_rgb, px_w, px_h = self._raster_data
            self._geotiff_px_w = px_w
            self._geotiff_px_h = px_h
            self._geotiff_locked = True

            # Lock GeoTIFF widgets
            self._geotiff_entry.configure(state="disabled")
            self._geotiff_btn.configure(state="disabled")
            self._load_geotiff_btn.configure(state="disabled")
            self._geotiff_status_var.set(
                f"✓ Loaded  |  {px_w}×{px_h} px  |  CRS: {tiff_crs[:35]}")

            # Populate and enable the Display Size panel
            self._disp_native_lbl.configure(
                text=f"GeoTIFF native: {px_w} × {px_h} px")
            self._disp_w_var.set(str(px_w))
            self._disp_h_var.set(str(px_h))
            for w in self._disp_size_widgets:
                w.configure(state="normal")

            self._draw_geotiff_background()
            self._set_status(
                "GeoTIFF loaded and locked. Now select a bag to add tracks.")
        except Exception as exc:
            messagebox.showerror("GeoTIFF Error", str(exc))
            self._set_status(f"GeoTIFF error: {exc}")

    def _draw_geotiff_background(self):
        raster, img_extent, tiff_crs, is_rgb, *_ = self._raster_data
        self._fig.clear()
        self._colorbar = None
        self._ax = self._fig.add_subplot(111)
        _draw_background(self._ax, raster, img_extent, is_rgb)
        self._ax.set_xlabel(f"Easting / X  [{tiff_crs}]")
        self._ax.set_ylabel(f"Northing / Y  [{tiff_crs}]")
        self._ax.tick_params(axis="both", labelsize=8)
        self._ax.set_title("GPS Track Composite", fontsize=12)
        self._fig.tight_layout()
        self._canvas.draw()

    # ── ② Display Size ────────────────────────────────────────────────────────

    def _build_display_size_section(self, parent):
        sec = self._section(parent, "② Display Size")

        # Collect widgets that start disabled and get enabled after GeoTIFF loads
        self._disp_size_widgets = []

        self._disp_native_lbl = ttk.Label(
            sec, text="Load a GeoTIFF to see native dimensions.",
            foreground="gray", wraplength=268)
        self._disp_native_lbl.pack(anchor="w", pady=(0, 4))

        # "Reset to native" shortcut
        btn_native = ttk.Button(
            sec, text="↩ Reset to native GeoTIFF size",
            command=self._use_native_size, state="disabled")
        btn_native.pack(fill=tk.X, pady=(0, 4))
        self._disp_size_widgets.append(btn_native)

        # W × H entry row
        row = ttk.Frame(sec)
        row.pack(fill=tk.X, pady=2)

        ttk.Label(row, text="W:", width=3, anchor="w").pack(side=tk.LEFT)
        self._disp_w_var = tk.StringVar()
        ew = ttk.Entry(row, textvariable=self._disp_w_var, width=7,
                       state="disabled")
        ew.pack(side=tk.LEFT, padx=(0, 8))
        self._disp_size_widgets.append(ew)

        ttk.Label(row, text="H:", width=3, anchor="w").pack(side=tk.LEFT)
        self._disp_h_var = tk.StringVar()
        eh = ttk.Entry(row, textvariable=self._disp_h_var, width=7,
                       state="disabled")
        eh.pack(side=tk.LEFT, padx=(0, 6))
        self._disp_size_widgets.append(eh)

        ttk.Label(row, text="px", foreground="gray").pack(side=tk.LEFT)

        btn_apply = ttk.Button(
            sec, text="Apply Display Size",
            command=self._apply_display_size, state="disabled")
        btn_apply.pack(fill=tk.X, pady=(4, 0))
        self._disp_size_widgets.append(btn_apply)

        ttk.Label(
            sec,
            text="Resizes the window so the plot area matches the requested "
                 "pixel dimensions. You can still resize the window freely "
                 "by dragging afterwards.",
            foreground="gray", wraplength=268,
            justify=tk.LEFT).pack(anchor="w", pady=(4, 0))

    def _use_native_size(self):
        self._disp_w_var.set(str(self._geotiff_px_w))
        self._disp_h_var.set(str(self._geotiff_px_h))
        self._apply_display_size()

    def _apply_display_size(self):
        """
        Resize the Tk window so the matplotlib canvas area is approximately
        desired_w × desired_h pixels.  Because we only call geometry() once,
        the window remains freely resizable by mouse drag afterwards.
        """
        try:
            desired_w = int(self._disp_w_var.get())
            desired_h = int(self._disp_h_var.get())
        except ValueError:
            messagebox.showwarning(
                "Invalid size",
                "Width and height must be whole-number pixel values.")
            return
        if desired_w < 100 or desired_h < 100:
            messagebox.showwarning(
                "Too small", "Width and height must each be at least 100 px.")
            return

        self.root.update_idletasks()

        # Measure the current control-panel width and toolbar height
        ctrl_w  = self._ctrl_outer.winfo_width()
        sash_w  = 6
        tb_h    = self._tb_frame.winfo_height()
        if tb_h < 2:
            tb_h = 30   # fallback if not yet rendered

        # Estimate window decoration overhead
        outer_w = self.root.winfo_width()
        outer_h = self.root.winfo_height()
        pane_w  = (self._ctrl_outer.winfo_width()
                   + sash_w
                   + self._canvas_frame.winfo_width())
        pane_h  = self._canvas_frame.winfo_height() + tb_h
        deco_w  = max(outer_w - pane_w, 0)
        deco_h  = max(outer_h - pane_h, 0)

        new_w = max(ctrl_w + sash_w + desired_w + deco_w,
                    self.root.winfo_reqwidth())
        new_h = max(desired_h + tb_h + deco_h, 400)

        self.root.geometry(f"{new_w}x{new_h}")
        self._set_status(
            f"Window resized for ~{desired_w}×{desired_h} px canvas. "
            "Drag the window border to resize freely.")

    # ── ③ Bag File ─────────────────────────────────────────────────────────────

    def _build_bag_section(self, parent):
        sec = self._section(parent, "③ Bag File")

        self._bag_var        = tk.StringVar()
        self._topic_var      = tk.StringVar(value="/hostname/sensors/ublox/fix")
        self._downsample_var = tk.IntVar(value=1)

        r1 = ttk.Frame(sec)
        r1.pack(fill=tk.X, pady=2)
        ttk.Label(r1, text="Dir:", width=10, anchor="w").pack(side=tk.LEFT)
        ttk.Entry(r1, textvariable=self._bag_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        ttk.Button(r1, text="…", width=3,
                   command=self._browse_bag).pack(side=tk.LEFT)

        r2 = ttk.Frame(sec)
        r2.pack(fill=tk.X, pady=2)
        ttk.Label(r2, text="Topic:", width=10, anchor="w").pack(side=tk.LEFT)
        ttk.Entry(r2, textvariable=self._topic_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True)

        r3 = ttk.Frame(sec)
        r3.pack(fill=tk.X, pady=2)
        ttk.Label(r3, text="Downsample:", width=10, anchor="w").pack(side=tk.LEFT)
        ttk.Spinbox(r3, from_=1, to=10000,
                    textvariable=self._downsample_var, width=6).pack(side=tk.LEFT)
        ttk.Label(r3, text=" every Nth fix",
                  foreground="gray").pack(side=tk.LEFT)

        self._bag_status_var = tk.StringVar(value="No bag loaded.")
        ttk.Label(sec, textvariable=self._bag_status_var,
                  foreground="gray", wraplength=270,
                  justify=tk.LEFT).pack(anchor="w", pady=(4, 0))

    def _browse_bag(self):
        path = filedialog.askdirectory(title="Select ROS2 Bag Directory")
        if path:
            self._bag_var.set(path)
            self._bag_status_var.set(
                "Bag selected — click 'Load Bag' to read it.")
            self._gps_data  = None
            self._projected = None
            self._istate    = {}

    # ── ④ Track Style ─────────────────────────────────────────────────────────

    def _build_track_style_section(self, parent):
        sec = self._section(parent, "④ Track Style  (for next track added)")

        # Colour-by
        r1 = ttk.Frame(sec)
        r1.pack(fill=tk.X, pady=2)
        ttk.Label(r1, text="Colour by:", width=12, anchor="w").pack(side=tk.LEFT)
        self._colorby_var = tk.StringVar(value="time")
        self._colorby_cb  = ttk.Combobox(r1, textvariable=self._colorby_var,
                                         values=COLORBY_OPTIONS,
                                         state="readonly", width=13)
        self._colorby_cb.pack(side=tk.LEFT)
        self._colorby_cb.bind("<<ComboboxSelected>>", self._on_colorby_change)

        # Track solid colour (enabled only when colorby = "solid color")
        self._track_row = ttk.Frame(sec)
        self._track_row.pack(fill=tk.X, pady=2)
        ttk.Label(self._track_row, text="Track:", width=12,
                  anchor="w").pack(side=tk.LEFT)
        default_tc = _TRACK_COLOR_CYCLE[0]
        self._track_swatch = tk.Label(self._track_row, width=2, relief="solid",
                                      bg=COLOR_HEX[default_tc])
        self._track_swatch.pack(side=tk.LEFT, padx=(0, 4))
        self._track_color_var = tk.StringVar(value=default_tc)
        self._track_color_cb  = ttk.Combobox(
            self._track_row, textvariable=self._track_color_var,
            values=COLOR_NAMES, state="disabled", width=9)
        self._track_color_cb.pack(side=tk.LEFT)
        self._track_color_cb.bind(
            "<<ComboboxSelected>>",
            lambda e: self._update_swatch(self._track_color_var,
                                          self._track_swatch))

        # Start / End marker colours — include "None" option
        self._start_color_var = tk.StringVar(value="Green")
        self._end_color_var   = tk.StringVar(value="Red")
        self._start_swatch = self._make_color_row(
            sec, "Start (▲):", self._start_color_var, "Green",
            MARKER_COLOR_NAMES)
        self._end_swatch = self._make_color_row(
            sec, "End (■):",   self._end_color_var,   "Red",
            MARKER_COLOR_NAMES)

        self._on_colorby_change()

    def _make_color_row(self, parent, label, var, default, names):
        """Build a label + swatch + combobox row. Returns the swatch Label."""
        f = ttk.Frame(parent)
        f.pack(fill=tk.X, pady=2)
        ttk.Label(f, text=label, width=12, anchor="w").pack(side=tk.LEFT)
        sw = tk.Label(f, width=2, relief="solid", bg=COLOR_HEX[default])
        sw.pack(side=tk.LEFT, padx=(0, 4))
        cb = ttk.Combobox(f, textvariable=var, values=names,
                          state="readonly", width=9)
        cb.pack(side=tk.LEFT)
        cb.bind("<<ComboboxSelected>>",
                lambda e, v=var, s=sw: self._update_swatch(v, s))
        return sw

    def _update_swatch(self, var, swatch):
        name = var.get()
        if name == "None":
            swatch.configure(bg="#cccccc", text="—", fg="#666666")
        else:
            swatch.configure(bg=COLOR_HEX.get(name, "#ffffff"),
                             text="", fg="black")

    def _on_colorby_change(self, *_):
        is_solid = self._colorby_var.get() == "solid color"
        self._track_color_cb.configure(
            state="readonly" if is_solid else "disabled")

    # ── ⑤ Mode ────────────────────────────────────────────────────────────────

    def _build_mode_section(self, parent):
        sec = self._section(parent, "⑤ Mode")
        self._mode_var = tk.StringVar(value="batch")
        ttk.Radiobutton(sec, text="Batch — add all fixes at once",
                        variable=self._mode_var, value="batch").pack(anchor="w")
        ttk.Radiobutton(sec, text="Interactive — step through fixes",
                        variable=self._mode_var,
                        value="interactive").pack(anchor="w")
        ttk.Radiobutton(sec, text="Time Range — select by ROS timestamp",
                        variable=self._mode_var,
                        value="timerange").pack(anchor="w")

    # ── Load Bag button ───────────────────────────────────────────────────────

    def _build_load_bag_section(self, parent):
        f = ttk.Frame(parent)
        f.pack(fill=tk.X, pady=(0, 4))
        self._load_bag_btn = ttk.Button(
            f, text="⏳  Load Bag", command=self._on_load_bag)
        self._load_bag_btn.pack(fill=tk.X)
        self._progress = ttk.Progressbar(f, mode="indeterminate")
        self._progress.pack(fill=tk.X, pady=(4, 0))

    # ── Batch controls ────────────────────────────────────────────────────────

    def _build_batch_section(self, parent):
        sec = self._section(parent, "Batch Controls")
        self._add_track_btn = ttk.Button(
            sec, text="➕  Add Track to Plot",
            command=self._batch_add_track, state=tk.DISABLED)
        self._add_track_btn.pack(fill=tk.X)

    # ── Interactive controls ──────────────────────────────────────────────────

    def _build_interactive_section(self, parent):
        sec = self._section(parent, "Interactive Controls")

        self._next_btn = ttk.Button(
            sec, text="▶  Next Point           [Space]",
            command=self._next_point)
        self._next_btn.pack(fill=tk.X, pady=2)

        self._set_start_btn = ttk.Button(
            sec, text="▲  Set Start", command=self._set_start)
        self._set_start_btn.pack(fill=tk.X, pady=2)

        self._commit_btn = ttk.Button(
            sec, text="✔  Commit Track to Plot   [E]",
            command=self._commit_segment)
        self._commit_btn.pack(fill=tk.X, pady=2)

        self._reset_step_btn = ttk.Button(
            sec, text="↺  Reset Stepping", command=self._reset_stepping)
        self._reset_step_btn.pack(fill=tk.X, pady=(2, 0))

        self._int_info_var = tk.StringVar(
            value="Load a bag in Interactive mode to enable.")
        ttk.Label(sec, textvariable=self._int_info_var,
                  foreground="gray", wraplength=268,
                  justify=tk.LEFT).pack(anchor="w", pady=(6, 0))

        self._set_interactive_btns(tk.DISABLED)

    # ── Time Range controls ───────────────────────────────────────────────────

    def _build_time_range_section(self, parent):
        sec = self._section(parent, "Time Range Controls")

        # Info row – shows the bag's full timestamp span once a bag is loaded
        self._tr_range_var = tk.StringVar(
            value="Load a bag in Time Range mode to see timestamps.")
        ttk.Label(sec, textvariable=self._tr_range_var,
                  foreground="gray", wraplength=268,
                  justify=tk.LEFT).pack(anchor="w", pady=(0, 6))

        # ── Start time ────────────────────────────────────────────────────────
        ttk.Label(sec, text="Start time (ROS seconds):",
                  anchor="w").pack(fill=tk.X)
        start_row = ttk.Frame(sec)
        start_row.pack(fill=tk.X, pady=(2, 0))
        self._tr_start_var = tk.StringVar()
        self._tr_start_entry = ttk.Entry(
            start_row, textvariable=self._tr_start_var, width=18)
        self._tr_start_entry.pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(start_row, text="◀ First",
                   command=self._tr_goto_start).pack(side=tk.LEFT)

        self._tr_start_dt_var = tk.StringVar(value="")
        ttk.Label(sec, textvariable=self._tr_start_dt_var,
                  foreground="gray", font=("TkDefaultFont", 8)).pack(
                      anchor="w", pady=(1, 4))

        # ── End time ──────────────────────────────────────────────────────────
        ttk.Label(sec, text="End time (ROS seconds):",
                  anchor="w").pack(fill=tk.X)
        end_row = ttk.Frame(sec)
        end_row.pack(fill=tk.X, pady=(2, 0))
        self._tr_end_var = tk.StringVar()
        self._tr_end_entry = ttk.Entry(
            end_row, textvariable=self._tr_end_var, width=18)
        self._tr_end_entry.pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(end_row, text="Last ▶",
                   command=self._tr_goto_end).pack(side=tk.LEFT)

        self._tr_end_dt_var = tk.StringVar(value="")
        ttk.Label(sec, textvariable=self._tr_end_dt_var,
                  foreground="gray", font=("TkDefaultFont", 8)).pack(
                      anchor="w", pady=(1, 4))

        # Live-update the human-readable datetime label as the user types
        self._tr_start_var.trace_add("write",
            lambda *_: self._tr_update_dt_label(
                self._tr_start_var, self._tr_start_dt_var))
        self._tr_end_var.trace_add("write",
            lambda *_: self._tr_update_dt_label(
                self._tr_end_var, self._tr_end_dt_var))

        # ── Add button ────────────────────────────────────────────────────────
        self._tr_add_btn = ttk.Button(
            sec, text="➕  Add Time-Range Track",
            command=self._tr_add_track, state=tk.DISABLED)
        self._tr_add_btn.pack(fill=tk.X, pady=(4, 0))

        # Start disabled; entries enabled only once a bag is loaded
        self._tr_start_entry.configure(state="disabled")
        self._tr_end_entry.configure(state="disabled")

    def _tr_update_dt_label(self, time_var, dt_var):
        """Convert a raw seconds StringVar to a UTC datetime label."""
        try:
            t = float(time_var.get())
            dt_str = datetime.fromtimestamp(t, tz=timezone.utc).strftime(
                "%Y-%m-%d  %H:%M:%S.%f")[:-3] + " UTC"
            dt_var.set(dt_str)
        except (ValueError, OSError):
            dt_var.set("")

    def _tr_goto_start(self):
        """Reset start entry to the bag's first timestamp."""
        if hasattr(self, "_tr_t_min"):
            self._tr_start_var.set(f"{self._tr_t_min:.3f}")

    def _tr_goto_end(self):
        """Reset end entry to the bag's last timestamp."""
        if hasattr(self, "_tr_t_max"):
            self._tr_end_var.set(f"{self._tr_t_max:.3f}")

    # ── Committed track list ──────────────────────────────────────────────────

    def _build_track_list_section(self, parent):
        sec = self._section(parent, "Committed Tracks")

        self._track_listbox = tk.Listbox(
            sec, height=5, selectmode=tk.SINGLE,
            font=("Courier", 9), activestyle="none",
            exportselection=False)
        self._track_listbox.pack(fill=tk.X, pady=(0, 4))

        btn_row = ttk.Frame(sec)
        btn_row.pack(fill=tk.X)
        ttk.Button(btn_row, text="Remove Selected",
                   command=self._remove_selected_track).pack(
                       side=tk.LEFT, padx=(0, 4))
        ttk.Button(btn_row, text="Clear All",
                   command=self._clear_all_tracks).pack(
                       side=tk.LEFT, padx=(0, 4))
        ttk.Button(btn_row, text="Edit Labels…",
                   command=self._open_edit_labels_dialog).pack(side=tk.LEFT)

    # ── Output / Save ─────────────────────────────────────────────────────────

    def _build_save_section(self, parent):
        sec = self._section(parent, "Output")

        self._output_var = tk.StringVar(value="gps_composite.png")
        row = ttk.Frame(sec)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="File:", width=6, anchor="w").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self._output_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        ttk.Button(row, text="…", width=3,
                   command=self._browse_output).pack(side=tk.LEFT)

        ttk.Button(sec, text="💾  Save Plot  [R]",
                   command=self._save_plot).pack(fill=tk.X, pady=(6, 0))

    def _browse_output(self):
        path = filedialog.asksaveasfilename(
            title="Save composite plot as…",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("All files", "*.*")])
        if path:
            self._output_var.set(path)

    # ── Status bar ────────────────────────────────────────────────────────────

    def _build_status_section(self, parent):
        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)
        self._status_var = tk.StringVar(
            value="Ready — load a GeoTIFF first, then add bag tracks.")
        ttk.Label(parent, textvariable=self._status_var,
                  foreground="navy", wraplength=272,
                  justify=tk.LEFT).pack(anchor="w")

    # ── Plot canvas ───────────────────────────────────────────────────────────

    def _build_canvas(self, parent):
        self._fig = Figure(figsize=(10, 8), dpi=100)
        self._fig.patch.set_facecolor("#1e1e2e")
        self._ax = self._fig.add_subplot(111)
        self._ax.set_facecolor("#1e1e2e")
        self._ax.text(0.5, 0.5, "Load a GeoTIFF to begin.",
                      ha="center", va="center", color="#a0a8c0",
                      fontsize=16, transform=self._ax.transAxes)
        self._ax.set_xticks([])
        self._ax.set_yticks([])

        self._canvas = FigureCanvasTkAgg(self._fig, master=parent)
        widget = self._canvas.get_tk_widget()
        widget.pack(fill=tk.BOTH, expand=True)

        # Store reference for height measurement used by _apply_display_size
        self._tb_frame = ttk.Frame(parent)
        self._tb_frame.pack(fill=tk.X)
        NavigationToolbar2Tk(self._canvas, self._tb_frame).update()

        widget.bind("<Button-1>", lambda e: widget.focus_set())
        widget.bind("<space>",    lambda e: self._next_point())
        widget.bind("<e>",        lambda e: self._commit_segment())
        widget.bind("<E>",        lambda e: self._commit_segment())
        widget.bind("<r>",        lambda e: self._save_plot())
        widget.bind("<R>",        lambda e: self._save_plot())

    # ═══ Helper utilities ════════════════════════════════════════════════════

    def _set_status(self, msg: str):
        self.root.after(0, lambda: self._status_var.set(msg))

    def _set_int_info(self, msg: str):
        self.root.after(0, lambda: self._int_info_var.set(msg))

    def _set_interactive_btns(self, state):
        for btn in (self._next_btn, self._set_start_btn,
                    self._commit_btn, self._reset_step_btn):
            btn.configure(state=state)

    def _start_spinner(self):
        self._progress.start(12)

    def _stop_spinner(self):
        self._progress.stop()

    def _auto_track_color(self):
        return _TRACK_COLOR_CYCLE[self._track_count % len(_TRACK_COLOR_CYCLE)]

    def _resolved_colors(self):
        """Return (track_rgb, start_rgb_or_None, end_rgb_or_None)."""
        return (COLOR_RGB[self._track_color_var.get()],
                _color_or_none(self._start_color_var.get()),
                _color_or_none(self._end_color_var.get()))

    def _update_legend(self):
        handles = []
        for t in self._committed_tracks:
            col = COLOR_RGB.get(t["track_color_name"], COLOR_RGB["Green"])
            handles.append(mpatches.Patch(color=col, label=t["label"]))
        if handles:
            self._ax.legend(handles=handles, loc="upper left", fontsize=8)
        else:
            leg = self._ax.get_legend()
            if leg:
                leg.remove()
        self._canvas.draw_idle()

    def _update_track_listbox(self):
        self._track_listbox.delete(0, tk.END)
        for i, t in enumerate(self._committed_tracks):
            self._track_listbox.insert(
                tk.END,
                f"{i+1:>2}. {t['label'][:28]}  ({t['n_fixes']} fixes)")

    def _guard_geotiff(self) -> bool:
        if not self._geotiff_locked or self._raster_data is None:
            messagebox.showwarning(
                "No GeoTIFF", "Please load a GeoTIFF first (Section ①).")
            return False
        return True

    def _guard_bag(self) -> bool:
        if self._gps_data is None or self._projected is None:
            messagebox.showwarning("No bag loaded",
                                   "Please load a bag file first.")
            return False
        return True

    def _update_title(self):
        if not self._geotiff_locked:
            return
        n = len(self._committed_tracks)
        if n == 0:
            self._ax.set_title(
                "GPS Track Composite — no tracks yet", fontsize=12)
        elif not self._istate:
            self._ax.set_title(
                f"GPS Track Composite — "
                f"{n} track{'s' if n != 1 else ''} plotted",
                fontsize=12)

    # ═══ Loading ═════════════════════════════════════════════════════════════

    def _on_load_bag(self):
        bag_path = self._bag_var.get().strip()
        if not bag_path:
            messagebox.showwarning("No bag", "Select a bag directory first.")
            return
        if not Path(bag_path).exists():
            messagebox.showerror("Not found",
                                 f"Bag directory not found:\n{bag_path}")
            return
        if not self._guard_geotiff():
            return

        self._istate = {}
        self._set_interactive_btns(tk.DISABLED)
        self._add_track_btn.configure(state=tk.DISABLED)
        self._commit_btn.configure(state=tk.DISABLED)
        self._load_bag_btn.configure(state=tk.DISABLED)
        self._start_spinner()
        self._set_status("Reading bag…")

        threading.Thread(target=self._load_bag_worker,
                         args=(bag_path,), daemon=True).start()

    def _load_bag_worker(self, bag_path: str):
        try:
            data = read_navsatfix(
                bag_path,
                self._topic_var.get().strip(),
                self._downsample_var.get(),
                status_cb=self._set_status)
            if len(data[1]) == 0:
                raise ValueError(
                    "No valid GPS fixes found — "
                    "check topic name and fix status.")

            lats, lons = data[1], data[2]
            _, _, tiff_crs, *_ = self._raster_data
            projected = _project_coords(lats, lons, tiff_crs)

            self._gps_data  = data
            self._projected = projected

            n        = len(data[1])
            bag_name = Path(bag_path).name
            self._set_status(f"Loaded {n} fixes from '{bag_name}'.")
            self.root.after(0, lambda: self._bag_status_var.set(
                f"✓ {n} fixes  |  {bag_name}"))
            self.root.after(0, self._on_bag_ready)

        except Exception as exc:
            self.root.after(0,
                            lambda: messagebox.showerror("Load Error", str(exc)))
            self._set_status(f"Error: {exc}")
        finally:
            self.root.after(0, self._stop_spinner)
            self.root.after(0, lambda: self._load_bag_btn.configure(
                state=tk.NORMAL))

    def _on_bag_ready(self):
        mode = self._mode_var.get()
        if mode == "batch":
            self._add_track_btn.configure(state=tk.NORMAL)
            self._set_interactive_btns(tk.DISABLED)
            self._tr_add_btn.configure(state=tk.DISABLED)
            self._set_int_info(
                "Switch to Interactive mode to use these controls.")
        elif mode == "interactive":
            self._add_track_btn.configure(state=tk.DISABLED)
            self._tr_add_btn.configure(state=tk.DISABLED)
            self._init_interactive()
        else:   # timerange
            self._add_track_btn.configure(state=tk.DISABLED)
            self._set_interactive_btns(tk.DISABLED)
            self._set_int_info(
                "Switch to Interactive mode to use these controls.")
            self._on_bag_ready_timerange()

    # ═══ Batch mode ══════════════════════════════════════════════════════════

    def _batch_add_track(self):
        if not self._guard_geotiff() or not self._guard_bag():
            return

        timestamps, lats, lons, alts, _ = self._gps_data
        xs, ys = self._projected
        colorby = self._colorby_var.get()
        track_color, start_color, end_color = self._resolved_colors()

        n        = len(xs)
        bag_name = Path(self._bag_var.get()).name
        label    = f"T{self._track_count + 1} {bag_name[:20]}"

        norm, cmap_, color_values = None, None, None
        if colorby != "solid color":
            color_values, _ = _compute_color_values(
                timestamps, lats, lons, alts, colorby)
            norm  = Normalize(vmin=color_values.min(), vmax=color_values.max())
            cmap_ = cm.plasma

        zbase   = 2 + self._track_count * 5
        artists = _draw_track_on_ax(
            self._ax, xs, ys,
            colorby, color_values, norm, cmap_,
            track_color, start_color, end_color,
            label, zorder_base=zbase)

        self._committed_tracks.append({
            "label":            label,
            "n_fixes":          n,
            "artists":          artists,
            "track_color_name": self._track_color_var.get(),
            "start_color_name": self._start_color_var.get(),
            "end_color_name":   self._end_color_var.get(),
            "colorby":          colorby,
        })
        self._track_count += 1
        self._track_color_var.set(self._auto_track_color())
        self._update_swatch(self._track_color_var, self._track_swatch)

        self._update_legend()
        self._update_track_listbox()
        self._update_title()
        self._canvas.draw()

        self._add_track_btn.configure(state=tk.DISABLED)
        self._gps_data  = None
        self._projected = None
        self._set_status(
            f"Track '{label}' added ({n} fixes). "
            "Load another bag or save the plot.")

    # ═══ Interactive mode ════════════════════════════════════════════════════

    def _init_interactive(self):
        timestamps, lats, lons, alts, _ = self._gps_data
        xs, ys = self._projected
        colorby     = self._colorby_var.get()
        use_flat    = (colorby == "solid color")
        track_color, start_color, end_color = self._resolved_colors()
        n_total     = len(xs)

        norm, cmap_, color_values = None, None, None
        if not use_flat:
            color_values, _ = _compute_color_values(
                timestamps, lats, lons, alts, colorby)
            norm  = Normalize(vmin=color_values.min(), vmax=color_values.max())
            cmap_ = cm.plasma

        zbase = 2 + self._track_count * 5

        # Dimmed all-points preview
        line_all, = self._ax.plot([], [], color="white",
                                  linewidth=0.8, alpha=0.25, zorder=zbase)
        if use_flat:
            scatter_all = self._ax.scatter(
                [], [], color=(*track_color, 0.25),
                s=6, zorder=zbase + 1, linewidths=0)
        else:
            scatter_all = self._ax.scatter(
                [], [], c=[], cmap=cmap_, norm=norm,
                s=6, zorder=zbase + 1, linewidths=0, alpha=0.25)

        # Selected-segment preview
        line_sel, = self._ax.plot([], [], color="white",
                                  linewidth=1.2, alpha=0.80, zorder=zbase + 2)
        if use_flat:
            scatter_sel = self._ax.scatter(
                [], [], color=track_color, s=9, zorder=zbase + 3, linewidths=0)
        else:
            scatter_sel = self._ax.scatter(
                [], [], c=[], cmap=cmap_, norm=norm,
                s=9, zorder=zbase + 3, linewidths=0)

        # Start / end preview markers (only if colour != None)
        start_artist = (
            self._ax.scatter([], [], color=start_color, s=120, marker="^",
                             zorder=zbase + 4)
            if start_color is not None else None)
        end_preview = (
            self._ax.scatter([], [], color=end_color, s=120, marker="s",
                             zorder=zbase + 4)
            if end_color is not None else None)

        preview_artists = [a for a in
                           [line_all, scatter_all, line_sel, scatter_sel,
                            start_artist, end_preview]
                           if a is not None]

        self._istate = {
            "xs": xs, "ys": ys,
            "lats": lats, "lons": lons,
            "alts": alts, "timestamps": timestamps,
            "colorby": colorby,
            "use_flat": use_flat,
            "color_values": color_values,
            "track_color": track_color,
            "start_color": start_color,
            "end_color":   end_color,
            "norm": norm, "cmap": cmap_,
            "n_total": n_total,
            "line_all": line_all, "scatter_all": scatter_all,
            "line_sel": line_sel, "scatter_sel": scatter_sel,
            "start_artist": start_artist, "end_preview": end_preview,
            "preview_artists": preview_artists,
            "idx": -1, "start_idx": -1, "first_global_idx": 0,
            "xs_all": [], "ys_all": [], "cv_all": [],
        }

        self._set_interactive_btns(tk.NORMAL)
        self._commit_btn.configure(state=tk.DISABLED)
        self._set_int_info(
            f"{n_total} fixes ready.\n"
            "Space/Next → step  |  Set Start → mark start\n"
            "E/Commit → add segment to plot")
        self._update_title()
        self._canvas.draw()

    def _irefresh(self):
        s = self._istate
        if not s or s["idx"] < 0:
            self._canvas.draw_idle()
            return

        xv_all = np.array(s["xs_all"])
        yv_all = np.array(s["ys_all"])

        s["line_all"].set_data(xv_all, yv_all)
        s["scatter_all"].set_offsets(np.column_stack([xv_all, yv_all]))
        if not s["use_flat"] and s["cv_all"]:
            s["scatter_all"].set_array(np.array(s["cv_all"]))

        xv_sel, yv_sel, cv_sel = self._selected_slice()
        if len(xv_sel) > 0:
            xa, ya = np.array(xv_sel), np.array(yv_sel)
            s["line_sel"].set_data(xa, ya)
            s["scatter_sel"].set_offsets(np.column_stack([xa, ya]))
            if not s["use_flat"] and cv_sel:
                s["scatter_sel"].set_array(np.array(cv_sel))
        else:
            s["line_sel"].set_data([], [])
            s["scatter_sel"].set_offsets(np.empty((0, 2)))

        if s["start_idx"] >= 0 and s["start_artist"] is not None:
            off = s["start_idx"] - s["first_global_idx"]
            s["start_artist"].set_offsets(
                [[s["xs_all"][off], s["ys_all"][off]]])

        if s["end_preview"] is not None:
            s["end_preview"].set_offsets(
                [[s["xs_all"][-1], s["ys_all"][-1]]])

        idx     = s["idx"]
        elapsed = s["timestamps"][idx] - s["timestamps"][0]
        start_note = (
            f"  |  Start @ fix {s['start_idx'] + 1}"
            if s["start_idx"] >= 0
            else "  |  click 'Set Start' to mark"
        )
        self._ax.set_title(
            f"Interactive — fix {idx + 1} / {s['n_total']}{start_note}\n"
            f"Lat {s['lats'][idx]:.6f}   Lon {s['lons'][idx]:.6f}   "
            f"Alt {s['alts'][idx]:.2f} m   t+{elapsed:.1f} s",
            fontsize=10)
        self._canvas.draw_idle()

    def _selected_slice(self):
        s     = self._istate
        si    = s["start_idx"]
        first = s["first_global_idx"]
        if si < 0:
            return [], [], []
        lo = si - first
        return (s["xs_all"][lo:],
                s["ys_all"][lo:],
                s["cv_all"][lo:] if s["cv_all"] else [])

    # ── Interactive actions ───────────────────────────────────────────────────

    def _next_point(self):
        if not self._istate or self._mode_var.get() != "interactive":
            return
        s = self._istate
        next_idx = s["idx"] + 1
        if next_idx >= s["n_total"]:
            self._set_status(
                f"All {s['n_total']} fixes stepped through. "
                "Click 'Commit Track' to add the segment.")
            return
        if s["idx"] < 0:
            s["first_global_idx"] = next_idx
        s["idx"] = next_idx
        s["xs_all"].append(float(s["xs"][next_idx]))
        s["ys_all"].append(float(s["ys"][next_idx]))
        if not s["use_flat"] and s["color_values"] is not None:
            s["cv_all"].append(float(s["color_values"][next_idx]))
        remaining  = s["n_total"] - next_idx - 1
        start_hint = (f"Start@{s['start_idx'] + 1}  "
                      if s["start_idx"] >= 0 else "no start set  ")
        self._set_status(
            f"Fix {next_idx + 1}/{s['n_total']}  |  {remaining} left  |  "
            + start_hint + "|  E = commit")
        if s["idx"] > 0:
            self._commit_btn.configure(state=tk.NORMAL)
        self._irefresh()

    def _set_start(self):
        if not self._istate or self._mode_var.get() != "interactive":
            return
        s = self._istate
        if s["idx"] < 0:
            self._set_status("No fixes plotted yet — press Next Point first.")
            return
        s["start_idx"] = s["idx"]
        remaining = s["n_total"] - s["idx"] - 1
        self._set_status(
            f"Start set at fix {s['idx'] + 1}/{s['n_total']}  "
            f"|  {remaining} left  |  E = commit")
        self._irefresh()

    def _commit_segment(self):
        if not self._istate or self._mode_var.get() != "interactive":
            return
        s = self._istate
        if s["idx"] < 0:
            self._set_status("No fixes stepped through yet.")
            return
        if s["start_idx"] < 0:
            s["start_idx"] = s["first_global_idx"]
        if s["start_idx"] == s["idx"]:
            self._set_status(
                "Start and End are the same fix — "
                "step further before committing.")
            return

        xv_sel, yv_sel, cv_sel = self._selected_slice()
        if not len(xv_sel):
            self._set_status("Empty segment — nothing to commit.")
            return

        xs_a  = np.array(xv_sel)
        ys_a  = np.array(yv_sel)
        n_seg = len(xs_a)

        bag_name = Path(self._bag_var.get()).name
        label    = f"T{self._track_count + 1} {bag_name[:20]}"
        cv_arr   = (np.array(cv_sel)
                    if (not s["use_flat"] and cv_sel) else None)

        for art in s.get("preview_artists", []):
            try:
                art.remove()
            except Exception:
                pass

        zbase   = 2 + self._track_count * 5
        artists = _draw_track_on_ax(
            self._ax, xs_a, ys_a,
            s["colorby"], cv_arr, s["norm"], s["cmap"],
            s["track_color"], s["start_color"], s["end_color"],
            label, zorder_base=zbase)

        self._committed_tracks.append({
            "label":            label,
            "n_fixes":          n_seg,
            "artists":          artists,
            "track_color_name": self._track_color_var.get(),
            "start_color_name": self._start_color_var.get(),
            "end_color_name":   self._end_color_var.get(),
            "colorby":          s["colorby"],
        })
        self._track_count += 1
        self._track_color_var.set(self._auto_track_color())
        self._update_swatch(self._track_color_var, self._track_swatch)

        self._istate = {}
        self._set_interactive_btns(tk.DISABLED)
        self._update_legend()
        self._update_track_listbox()
        self._update_title()
        self._canvas.draw()

        self._gps_data  = None
        self._projected = None
        self._set_status(
            f"Track '{label}' committed ({n_seg} fixes). "
            "Load another bag or save the plot.")
        self._set_int_info(
            f"Track committed! {n_seg} fixes added.\n"
            "Load a new bag to continue.")

    def _reset_stepping(self):
        if self._istate:
            for art in self._istate.get("preview_artists", []):
                try:
                    art.remove()
                except Exception:
                    pass
            self._istate = {}
        self._set_interactive_btns(tk.DISABLED)
        self._update_title()
        self._canvas.draw_idle()
        if self._gps_data is not None:
            self._init_interactive()
            self._set_status("Stepping reset — re-starting from the first fix.")
        else:
            self._set_status("Stepping reset. Load a bag to begin a new track.")

    # ═══ Time Range mode ══════════════════════════════════════════════════════

    def _on_bag_ready_timerange(self):
        """Populate the Time Range panel from the loaded bag's timestamps."""
        timestamps = self._gps_data[0]          # seconds (float64 array)
        t_min = float(timestamps.min())
        t_max = float(timestamps.max())
        self._tr_t_min = t_min
        self._tr_t_max = t_max

        # Enable entry fields and fill with full range
        self._tr_start_entry.configure(state="normal")
        self._tr_end_entry.configure(state="normal")
        self._tr_start_var.set(f"{t_min:.3f}")
        self._tr_end_var.set(f"{t_max:.3f}")
        self._tr_add_btn.configure(state=tk.NORMAL)

        # Summary line
        def _fmt(t):
            return datetime.fromtimestamp(t, tz=timezone.utc).strftime(
                "%H:%M:%S.%f")[:-3]

        n = len(timestamps)
        span = t_max - t_min
        self._tr_range_var.set(
            f"Bag span:  {_fmt(t_min)} → {_fmt(t_max)} UTC\n"
            f"({span:.1f} s total  |  {n} valid fixes)")

        self._set_status(
            f"Time Range ready — {n} fixes spanning {span:.1f} s. "
            "Adjust start/end times then click Add.")

    def _tr_add_track(self):
        """Filter the loaded GPS data to [start, end] and add as a track."""
        if not self._guard_geotiff() or not self._guard_bag():
            return

        # Validate and clamp the entered times
        try:
            t_start = float(self._tr_start_var.get())
            t_end   = float(self._tr_end_var.get())
        except ValueError:
            messagebox.showwarning(
                "Invalid time",
                "Start and End times must be numeric (ROS seconds).")
            return

        t_min = getattr(self, "_tr_t_min", None)
        t_max = getattr(self, "_tr_t_max", None)

        if t_min is None:
            messagebox.showwarning("No bag", "Load a bag first.")
            return

        # Clamp to valid range and catch impossible selections
        t_start = max(t_start, t_min)
        t_end   = min(t_end,   t_max)

        if t_start >= t_end:
            messagebox.showwarning(
                "Invalid range",
                f"Start time must be before end time.\n"
                f"Valid range: {t_min:.3f} – {t_max:.3f} s")
            return

        # Slice the data
        timestamps, lats, lons, alts, statuses = self._gps_data
        mask = (timestamps >= t_start) & (timestamps <= t_end)
        if mask.sum() < 2:
            messagebox.showwarning(
                "Too few fixes",
                "Fewer than 2 GPS fixes in the selected time range. "
                "Widen the range.")
            return

        ts_s  = timestamps[mask]
        lats_s = lats[mask]
        lons_s = lons[mask]
        alts_s = alts[mask]
        xs_all, ys_all = self._projected
        xs_s = xs_all[mask]
        ys_s = ys_all[mask]

        colorby     = self._colorby_var.get()
        track_color, start_color, end_color = self._resolved_colors()

        norm, cmap_, color_values = None, None, None
        if colorby != "solid color":
            color_values, _ = _compute_color_values(
                ts_s, lats_s, lons_s, alts_s, colorby)
            norm  = Normalize(vmin=color_values.min(), vmax=color_values.max())
            cmap_ = cm.plasma

        n_seg    = int(mask.sum())
        bag_name = Path(self._bag_var.get()).name
        label    = f"T{self._track_count + 1} {bag_name[:20]}"

        zbase   = 2 + self._track_count * 5
        artists = _draw_track_on_ax(
            self._ax, xs_s, ys_s,
            colorby, color_values, norm, cmap_,
            track_color, start_color, end_color,
            label, zorder_base=zbase)

        self._committed_tracks.append({
            "label":            label,
            "n_fixes":          n_seg,
            "artists":          artists,
            "track_color_name": self._track_color_var.get(),
            "start_color_name": self._start_color_var.get(),
            "end_color_name":   self._end_color_var.get(),
            "colorby":          colorby,
        })
        self._track_count += 1
        self._track_color_var.set(self._auto_track_color())
        self._update_swatch(self._track_color_var, self._track_swatch)

        self._update_legend()
        self._update_track_listbox()
        self._update_title()
        self._canvas.draw()

        self._tr_add_btn.configure(state=tk.DISABLED)
        self._tr_start_entry.configure(state="disabled")
        self._tr_end_entry.configure(state="disabled")
        self._gps_data  = None
        self._projected = None
        self._set_status(
            f"Track '{label}' added ({n_seg} fixes, "
            f"{t_start:.3f}–{t_end:.3f} s). "
            "Load another bag or save the plot.")

    # ═══ Track list management ════════════════════════════════════════════════

    def _remove_selected_track(self):
        sel = self._track_listbox.curselection()
        if not sel:
            messagebox.showinfo("Nothing selected",
                                "Click a track in the list first.")
            return
        idx   = sel[0]
        track = self._committed_tracks[idx]
        for art in track["artists"]:
            try:
                art.remove()
            except Exception:
                pass
        del self._committed_tracks[idx]
        self._update_legend()
        self._update_track_listbox()
        self._update_title()
        self._canvas.draw()
        self._set_status(f"Removed track: {track['label']}")

    def _clear_all_tracks(self):
        if not self._committed_tracks:
            return
        if not messagebox.askyesno(
                "Clear all tracks",
                "Remove all committed tracks from the plot?"):
            return
        for track in self._committed_tracks:
            for art in track["artists"]:
                try:
                    art.remove()
                except Exception:
                    pass
        self._committed_tracks.clear()
        self._track_count = 0
        self._track_color_var.set(_TRACK_COLOR_CYCLE[0])
        self._update_swatch(self._track_color_var, self._track_swatch)
        self._update_legend()
        self._update_track_listbox()
        self._update_title()
        self._canvas.draw()
        self._set_status("All tracks removed.")

    # ── Edit Labels dialog ────────────────────────────────────────────────────

    def _open_edit_labels_dialog(self):
        if not self._committed_tracks:
            messagebox.showinfo(
                "No tracks",
                "Add at least one track before editing labels.")
            return

        dlg = tk.Toplevel(self.root)
        dlg.title("Edit Track Legend Labels")
        dlg.resizable(True, False)
        dlg.transient(self.root)
        dlg.grab_set()

        ttk.Label(
            dlg,
            text="Edit the legend label for each committed track.\n"
                 "Changes take effect when you click Apply.",
            justify=tk.LEFT, padding=(10, 8, 10, 4)).pack(anchor="w")

        ttk.Separator(dlg, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8)

        frame = ttk.Frame(dlg, padding=(10, 6))
        frame.pack(fill=tk.BOTH, expand=True)

        label_vars: list[tk.StringVar] = []
        for i, t in enumerate(self._committed_tracks):
            row = ttk.Frame(frame)
            row.pack(fill=tk.X, pady=3)

            # Colour swatch for quick identification
            tc_hex = COLOR_HEX.get(t["track_color_name"], "#888888")
            tk.Label(row, width=2, relief="solid",
                     bg=tc_hex).pack(side=tk.LEFT, padx=(0, 6))

            ttk.Label(row, text=f"T{i+1}:", width=4,
                      anchor="w").pack(side=tk.LEFT)
            var = tk.StringVar(value=t["label"])
            ttk.Entry(row, textvariable=var, width=30).pack(
                side=tk.LEFT, fill=tk.X, expand=True)
            label_vars.append(var)

        ttk.Separator(dlg, orient=tk.HORIZONTAL).pack(
            fill=tk.X, padx=8, pady=(6, 0))

        btn_row = ttk.Frame(dlg, padding=(10, 6))
        btn_row.pack(fill=tk.X)

        def _apply():
            for i, var in enumerate(label_vars):
                new_label = var.get().strip()
                if not new_label:
                    messagebox.showwarning(
                        "Empty label",
                        f"Track {i+1} label cannot be blank.",
                        parent=dlg)
                    return
                self._committed_tracks[i]["label"] = new_label
            self._update_legend()
            self._update_track_listbox()
            self._canvas.draw_idle()
            self._set_status("Legend labels updated.")
            dlg.destroy()

        ttk.Button(btn_row, text="Apply",
                   command=_apply).pack(side=tk.RIGHT, padx=(4, 0))
        ttk.Button(btn_row, text="Cancel",
                   command=dlg.destroy).pack(side=tk.RIGHT)

        # Centre over the main window
        self.root.update_idletasks()
        mx = self.root.winfo_x() + self.root.winfo_width()  // 2
        my = self.root.winfo_y() + self.root.winfo_height() // 2
        dlg.update_idletasks()
        dlg.geometry(f"+{mx - dlg.winfo_width()//2}+{my - dlg.winfo_height()//2}")

    # ═══ Save ════════════════════════════════════════════════════════════════

    def _save_plot(self):
        if not self._geotiff_locked:
            messagebox.showwarning("No GeoTIFF",
                                   "Load a GeoTIFF before saving.")
            return
        output = self._output_var.get().strip()
        if not output:
            messagebox.showwarning("No output path",
                                   "Set an output file path first.")
            return
        try:
            self._fig.savefig(output, dpi=200, bbox_inches="tight")
            n = len(self._committed_tracks)
            self._set_status(
                f"✓ Saved composite plot ({n} track(s)) → {output}")
        except Exception as exc:
            messagebox.showerror("Save Error", str(exc))
            self._set_status(f"Save failed: {exc}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    try:
        root.tk.call("tk", "scaling", 1.25)
    except Exception:
        pass
    GPSPlotterGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
