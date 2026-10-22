import time
import matplotlib

try:
    matplotlib.use("Qt5Agg")
except ImportError:
    pass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Circle
from matplotlib.transforms import Affine2D
from matplotlib.widgets import Button, Slider, RadioButtons
from astropy.visualization import (
    ImageNormalize, LinearStretch, LogStretch, SqrtStretch
)

_STRETCHES = {'lin': LinearStretch, 'log': LogStretch, 'sqrt': SqrtStretch}


def _clip_limits(image, lo=1.0, hi=99.5, target=250_000):
    """Fast (vmin, vmax) from percentiles, computed on a strided subsample.

    Percentiles on a full multi-megapixel frame are slow (~60 ms); a subsample of a few
    hundred-thousand pixels gives effectively identical clip limits in ~3 ms.
    """
    arr = image
    if arr.size > target:
        step = int(np.sqrt(arr.size / target)) + 1
        arr = arr[::step, ::step]
    vmin, vmax = np.nanpercentile(arr, [lo, hi])
    vmin, vmax = float(vmin), float(vmax)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1e-6
    return vmin, vmax

class ApertureSelectorTwoLayer:
    """
    Two-layer approach:
      - ax_img: Bottom axis, shows only the static image
      - ax_dyn: Top axis (transparent), for dynamic patches & text
      - info_ax: Right-side axis for help annotation
      - We do partial blitting *only* on ax_dyn, so that the bottom image is never re-drawn.
    """

    def __init__(self, image, title="Select Satellite Trail Aperture",
                 img_norm='lin', cmap='gray_r'):
        self.image = image
        self.title = title
        self.img_norm = img_norm
        self.cmap = cmap

        # Image display (cuts) state, adjustable live via the control widgets.
        # Absolute data values (vmin/vmax); slider bounds come from the image's robust
        # data range so the useful range is not swamped by outliers.
        self.im = None
        self.stretch = img_norm if img_norm in _STRETCHES else 'sqrt'
        self._dlo, self._dhi = _clip_limits(image, 0.1, 99.9)   # slider bounds
        if self._dhi <= self._dlo:
            self._dhi = self._dlo + 1.0
        self.vmin, self.vmax = _clip_limits(image, 1.0, 99.5)   # initial cuts
        # Defaults remembered for the Reset button
        self._vmin0, self._vmax0, self._stretch0 = self.vmin, self.vmax, self.stretch

        # Initial geometry
        self.ny, self.nx = image.shape
        self.cx, self.cy = self.nx//2, self.ny//2
        self.width, self.height = self.nx//10, self.ny//20
        self.theta = 0.0

        # Interaction state
        self.dragging_center = False
        self.dragging_corner = False
        self.dragging_edge = False
        self.rotating = False
        self.active_corner = None
        self.active_edge = None
        self.last_x = 0.0
        self.last_y = 0.0
        self.last_motion_event = None
        self.start_theta = 0.0
        self.start_angle = 0.0

        # Rate-limit
        self.last_draw_time = 0.0
        self.redraw_interval = 0.02  # 50 FPS

        # We'll store a background only for the top axis (ax_dyn)
        self.dyn_background = None
        # ...and a separate clean background for the info panel, so the changing
        # help/stats text can be redrawn each frame without smearing over old text.
        self.info_background = None

        self.first_draw_complete = False

        # Visual items
        self.rect_patch = None
        self.dash_lines = []
        self.corner_handles = []
        self.edge_lines = []
        self.hover_text = None
        self.help_annotation = None

        self._setup_figure()

    # ------------------------------------------------------------------------
    # Figure Setup
    # ------------------------------------------------------------------------
    # Layout constants (inches): panel width, gap, (left,right,bottom,top) margins, caps.
    # The bottom margin reserves room for the image-cuts control strip (sliders + radio).
    _PANEL_W = 2.2
    _GAP = 0.25
    _MARGINS = (0.4, 0.2, 1.2, 0.6)
    _MAX_W, _MAX_H = 6.5, 6.0

    def _initial_figsize(self):
        """Figure size (inches) so the image box matches the image aspect ratio."""
        aspect = self.nx / self.ny
        m_l, m_r, m_b, m_t = self._MARGINS
        if self._MAX_W / self._MAX_H >= aspect:   # height-limited
            img_h = self._MAX_H
            img_w = img_h * aspect
        else:                                     # width-limited
            img_w = self._MAX_W
            img_h = img_w / aspect
        return (m_l + img_w + self._GAP + self._PANEL_W + m_r, m_b + img_h + m_t)

    def _control_rects(self):
        """[l,b,w,h] rects for the low/high cut sliders, the stretch radio, and the reset
        button, placed in the reserved bottom strip and spanning under the image."""
        fig_w, fig_h = self.fig.get_size_inches()
        img_rect, _ = self._layout_rects()
        ix0, iw = img_rect[0], img_rect[2]
        # Start the sliders well right of the stretch radio (~1.14 in wide) so the slider
        # labels ('vmin'/'vmax', drawn to the left of the track) do not crowd it.
        s_l = max(ix0, 1.85 / fig_w)
        s_r = ix0 + iw
        reset_w = 0.75 / fig_w
        # Sliders draw their current value just to the right of the track; leave room for
        # that readout AND the reset button so they do not overlap.
        reset_gap = 0.85 / fig_w
        s_w = max(0.1, (s_r - reset_w - reset_gap) - s_l)
        radio = [0.02, 0.10 / fig_h, 0.95 / fig_w, 0.92 / fig_h]
        hi = [s_l, 0.74 / fig_h, s_w, 0.13 / fig_h]
        lo = [s_l, 0.44 / fig_h, s_w, 0.13 / fig_h]
        reset = [s_r - reset_w, 0.50 / fig_h, reset_w, 0.31 / fig_h]
        return lo, hi, radio, reset

    def _on_reset_cuts(self, _ev):
        """Restore the default cut values and stretch."""
        self.s_lo.set_val(self._vmin0)            # each triggers _apply_norm
        self.s_hi.set_val(self._vmax0)
        self.radio_stretch.set_active(('lin', 'log', 'sqrt').index(self._stretch0))

    def _make_norm(self):
        """Build an ImageNormalize from the current absolute cut values + stretch."""
        vmin, vmax = self.vmin, self.vmax
        if vmax <= vmin:
            vmax = vmin + 1e-6
        stretch_cls = _STRETCHES.get(self.stretch, LinearStretch)
        return ImageNormalize(vmin=vmin, vmax=vmax, stretch=stretch_cls())

    def _apply_norm(self):
        """Re-apply the display norm to the image and refresh (invalidating blit caches)."""
        if self.im is None:
            return
        self.im.set_norm(self._make_norm())
        # ax_img changed, so the cached blit backgrounds are stale. Do a synchronous full
        # draw (on_draw recaptures the new backgrounds), then re-blit the overlay on top so
        # the aperture/help stay visible (the animated artists are not in the full draw).
        self.dyn_background = None
        self.info_background = None
        self.fig.canvas.draw()
        self._blit_draw()

    def _on_cuts_changed(self, _val):
        self.vmin = float(self.s_lo.val)
        self.vmax = float(self.s_hi.val)
        self._apply_norm()

    def _on_stretch_changed(self, label):
        self.stretch = label
        self._apply_norm()

    def _layout_rects(self):
        """[l,b,w,h] rects for the image axes and the info panel that keep the image at
        its data aspect and top-align the panel, for the CURRENT figure size."""
        fig_w, fig_h = self.fig.get_size_inches()
        aspect = self.nx / self.ny
        m_l, m_r, m_b, m_t = self._MARGINS
        avail_w = max(1e-3, fig_w - m_l - m_r - self._GAP - self._PANEL_W)
        avail_h = max(1e-3, fig_h - m_b - m_t)
        if avail_w / avail_h >= aspect:           # height-limited
            img_h = avail_h
            img_w = img_h * aspect
        else:                                     # width-limited
            img_w = avail_w
            img_h = img_w / aspect
        l = m_l / fig_w
        b = m_b / fig_h
        w = img_w / fig_w
        h = img_h / fig_h
        panel_l = (m_l + img_w + self._GAP) / fig_w
        panel_frac = self._PANEL_W / fig_w
        return [l, b, w, h], [panel_l, b, panel_frac, h]

    def _setup_figure(self):
        # --- Managed, aspect-adaptive layout -----------------------------------------
        # The window opens sized to the image aspect; `on_resize` re-letterboxes the image
        # box into the available area on every resize, so the equal-aspect constraint never
        # shrinks/re-centres it and the help panel stays aligned with the image.
        fig_w, fig_h = self._initial_figsize()
        self.fig = plt.figure(figsize=(fig_w, fig_h))
        self.fig.suptitle(self.title, fontsize=11)

        img_rect, info_rect = self._layout_rects()

        # 1) Bottom axis for the image only
        self.ax_img = self.fig.add_axes(img_rect)
        self.ax_img.set_anchor('N')

        # 2) Top axis (transparent) for dynamic patches
        self.ax_dyn = self.fig.add_axes(img_rect,
                                        sharex=self.ax_img, sharey=self.ax_img)
        self.ax_dyn.set_anchor('N')
        self.ax_dyn.set_facecolor("none")
        self.ax_dyn.set_zorder(2)  # ensure it's above self.ax_img

        # Turn off ticks/spines in top axis if you like
        self.ax_dyn.set_xticks([])
        self.ax_dyn.set_yticks([])
        for spine in self.ax_dyn.spines.values():
            spine.set_visible(False)

        # 3) Right axis for help annotation (same vertical extent as the image box)
        self.info_ax = self.fig.add_axes(info_rect)
        self.info_ax.axis('off')

        # Show the image in ax_img (norm from the current cut percentiles + stretch)
        self.im = self.ax_img.imshow(self.image, norm=self._make_norm(), cmap=self.cmap,
                                     origin='lower', extent=(0, self.nx, 0, self.ny))
        self.ax_img.set_xlim(0, self.nx)
        self.ax_img.set_ylim(0, self.ny)
        self.ax_img.set_aspect('equal', 'box')  # 1:1 aspect ratio

        # Image-cuts controls in the reserved bottom strip: low/high percentile sliders
        # and a lin/log/sqrt stretch selector. These live in their own axes (not blitted).
        lo_rect, hi_rect, radio_rect, reset_rect = self._control_rects()
        self.ax_lo = self.fig.add_axes(lo_rect)
        self.ax_hi = self.fig.add_axes(hi_rect)
        self.ax_stretch = self.fig.add_axes(radio_rect)
        self.ax_reset = self.fig.add_axes(reset_rect)
        # Slider.__init__ calls set_val(), which triggers a full-figure draw_idle and
        # re-renders the (large) image once per slider. Suppress those draws during
        # construction - the figure is drawn once by show().
        _orig_draw_idle = self.fig.canvas.draw_idle
        self.fig.canvas.draw_idle = lambda *a, **k: None
        try:
            self.s_lo = Slider(self.ax_lo, 'vmin', self._dlo, self._dhi,
                               valinit=self.vmin, valfmt='%.4g')
            self.s_hi = Slider(self.ax_hi, 'vmax', self._dlo, self._dhi,
                               valinit=self.vmax, valfmt='%.4g')
            _stretch_opts = ('lin', 'log', 'sqrt')
            self.radio_stretch = RadioButtons(self.ax_stretch, _stretch_opts,
                                              active=_stretch_opts.index(self.stretch))
            self.btn_reset = Button(self.ax_reset, 'Reset')
            self.btn_reset.label.set_fontsize(8)
        finally:
            self.fig.canvas.draw_idle = _orig_draw_idle
        self.s_lo.on_changed(self._on_cuts_changed)
        self.s_hi.on_changed(self._on_cuts_changed)
        self.radio_stretch.on_clicked(self._on_stretch_changed)
        self.btn_reset.on_clicked(self._on_reset_cuts)

        # Also force no margins and 1:1 aspect here
        self.ax_dyn.set_xlim(0, self.nx)
        self.ax_dyn.set_ylim(0, self.ny)
        self.ax_dyn.set_aspect('equal', 'box')

        # Single annotation in info_ax. Marked 'animated' so it is excluded from normal
        # canvas draws; this lets on_draw capture a clean (text-free) panel background,
        # which _blit_draw restores before re-drawing the annotation (prevents ghosting).
        self.help_annotation = self.info_ax.annotate(
            "",
            xy=(0, 1),
            xycoords='axes fraction',
            ha='left', va='top',
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.7)
        )
        self.help_annotation.set_animated(True)

        # Patches in ax_dyn. All dynamic artists are 'animated' so they are excluded from
        # normal canvas draws (the blit background stays clean) and are rendered only via
        # draw_artist in _blit_draw. This replaces the old on_draw cla() hack, which broke
        # when full redraws became frequent (e.g. every cut-slider change).
        # Rectangle
        self.rect_patch = Rectangle(
            xy=(-self.width/2, -self.height/2),
            width=self.width, height=self.height,
            angle=0,
            fill=False,
            edgecolor='lime',
            linewidth=2,
            alpha=0.8,
            animated=True
        )
        self.ax_dyn.add_patch(self.rect_patch)

        # Dashed lines
        for _ in range(2):
            line = Line2D([], [], linestyle='--', color='black', linewidth=1., animated=True)
            self.ax_dyn.add_line(line)
            self.dash_lines.append(line)

        # Corner handles
        for _ in range(4):
            c = Circle((0, 0), radius=5, facecolor='red', edgecolor='white', lw=1, animated=True)
            self.ax_dyn.add_patch(c)
            self.corner_handles.append(c)

        # Edge handles
        for _ in range(4):
            e = Line2D([], [], color='red', lw=1.5, animated=True)
            self.ax_dyn.add_line(e)
            self.edge_lines.append(e)

        # Hover text
        self.hover_text = self.ax_dyn.text(
            0, 0, "",
            color='yellow', fontsize=8,
            bbox=dict(boxstyle='round', fc='black', alpha=0.7),
            animated=True
        )

        # Dynamic artists render via blitting (draw_artist), not the normal draw.
        self.rect_patch.set_visible(True)
        for patch in self.corner_handles + self.edge_lines + self.dash_lines:
            patch.set_visible(True)
        self.hover_text.set_visible(False)

        # Connect events
        self.fig.canvas.mpl_connect('draw_event', self.on_draw)
        self.fig.canvas.mpl_connect('button_press_event', self.on_button_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_button_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('resize_event', self.on_resize)

    # ------------------------------------------------------------------------
    # on_resize: re-letterbox the image box so alignment/aspect survive resizing
    # ------------------------------------------------------------------------
    def on_resize(self, event):
        if not hasattr(self, 'ax_img'):
            return
        img_rect, info_rect = self._layout_rects()
        self.ax_img.set_position(img_rect)
        self.ax_dyn.set_position(img_rect)
        self.info_ax.set_position(info_rect)
        # Reposition the cut controls in the bottom strip for the new size
        if hasattr(self, 'ax_lo'):
            lo_rect, hi_rect, radio_rect, reset_rect = self._control_rects()
            self.ax_lo.set_position(lo_rect)
            self.ax_hi.set_position(hi_rect)
            self.ax_stretch.set_position(radio_rect)
            self.ax_reset.set_position(reset_rect)
        # Backgrounds are stale at the new size; on_draw (fired by the resize) recaptures.
        self.dyn_background = None
        self.info_background = None

    # ------------------------------------------------------------------------
    # on_draw: recapture clean blit backgrounds after any full canvas draw
    # ------------------------------------------------------------------------
    def on_draw(self, event):
        if event.canvas != self.fig.canvas:
            return
        # All overlay artists (and the info annotation) are 'animated', so this full draw
        # excludes them and the captured backgrounds are clean. No cla() needed — that old
        # hack broke under frequent redraws (e.g. cut-slider changes). Wrapped so a draw-
        # callback exception can never disable further draws in the Qt event loop.
        try:
            self.dyn_background = self.fig.canvas.copy_from_bbox(self.ax_dyn.bbox)
            self.info_background = self.fig.canvas.copy_from_bbox(self.info_ax.bbox)
            self.first_draw_complete = True
        except Exception:
            self.dyn_background = None
            self.info_background = None

    # ------------------------------------------------------------------------
    # Event Handlers
    # ------------------------------------------------------------------------
    def on_button_press(self, event):
        # If zoom or pan tool is active, invalidate the blit background
        if self.fig.canvas.toolbar:
            if self.fig.canvas.toolbar.mode in ('pan', 'zoom'):
                self.dyn_background = None

        if event.inaxes != self.ax_dyn:
            return
        if event.xdata is None or event.ydata is None:
            return

        self.last_x, self.last_y = event.xdata, event.ydata

        # SHIFT => rotate
        if event.key == 'shift':
            self.rotating = True
            self.start_theta = self.theta
            self.start_angle = np.degrees(np.arctan2(event.ydata - self.cy,
                                                     event.xdata - self.cx))
            return

        # Corner vs Edge vs Inside
        c_idx = self.get_corner_index(event.xdata, event.ydata)
        if c_idx is not None:
            self.dragging_corner = True
            self.active_corner = c_idx
            return

        e_idx = self.get_edge_index(event.xdata, event.ydata)
        if e_idx is not None:
            self.dragging_edge = True
            self.active_edge = e_idx
            return

        if self.inside_rectangle(event.xdata, event.ydata):
            self.dragging_center = True

    def on_button_release(self, event):
        self.dragging_center = False
        self.dragging_corner = False
        self.dragging_edge = False
        self.rotating = False
        self.active_corner = None
        self.active_edge = None

    def on_motion(self, event):
        if event.inaxes != self.ax_dyn:
            return
        if event.xdata is None or event.ydata is None:
            self._blit_draw()
            return

        self.last_motion_event = event
        dx = event.xdata - self.last_x
        dy = event.ydata - self.last_y

        # Move center
        if self.dragging_center and not self.rotating:
            self.cx += dx
            self.cy += dy
        # Rotate
        elif self.rotating:
            cur_angle = np.degrees(np.arctan2(event.ydata - self.cy,
                                              event.xdata - self.cx))
            angle_diff = cur_angle - self.start_angle
            self.theta = (self.start_theta + angle_diff) % 360
        # Resize corner
        elif self.dragging_corner and (self.active_corner is not None):
            lx, ly = self.to_local_coords(event.xdata, event.ydata)
            if self.active_corner in [0, 1]:
                self.width = max(10, 2*abs(lx))
            else:
                self.width = max(10, 2*abs(lx))
            if self.active_corner in [0, 3]:
                self.height = max(10, 2*abs(ly))
            else:
                self.height = max(10, 2*abs(ly))
        # Resize edge
        elif self.dragging_edge and (self.active_edge is not None):
            lx, ly = self.to_local_coords(event.xdata, event.ydata)
            if self.active_edge in [0,2]:
                self.height = max(10, 2*abs(ly))
            else:
                self.width = max(10, 2*abs(lx))

        self.last_x = event.xdata
        self.last_y = event.ydata

        # Rate limit
        now = time.time()
        if (now - self.last_draw_time) < self.redraw_interval:
            return
        self.last_draw_time = now

        self._blit_draw()

    def on_key_press(self, event):
        if event.key == 'enter':
            plt.close(self.fig)

    # ------------------------------------------------------------------------
    # Partial Blit Routine
    # ------------------------------------------------------------------------
    def _blit_draw(self):
        if self.dyn_background is None:
            self.fig.canvas.draw()
            return

        # Ensure figure is still valid before blitting
        if not plt.fignum_exists(self.fig.number):
            return

        # Wrapped so a transient rendering error never propagates into the event loop.
        try:
            self._update_artists()

            # Restore background + draw the overlay artists
            self.fig.canvas.restore_region(self.dyn_background)
            self.ax_dyn.draw_artist(self.rect_patch)
            for ln in self.dash_lines:
                self.ax_dyn.draw_artist(ln)
            for corner in self.corner_handles:
                self.ax_dyn.draw_artist(corner)
            for ed in self.edge_lines:
                self.ax_dyn.draw_artist(ed)
            if self.hover_text.get_visible():
                self.ax_dyn.draw_artist(self.hover_text)

            # Re-draw the annotation on a clean panel background (no text smearing)
            if self.info_background is not None:
                self.fig.canvas.restore_region(self.info_background)
            self.info_ax.draw_artist(self.help_annotation)

            self.fig.canvas.blit(self.ax_dyn.bbox)
            self.fig.canvas.blit(self.info_ax.bbox)
        except Exception:
            pass

    # ------------------------------------------------------------------------
    # Update geometry and annotation text
    # ------------------------------------------------------------------------
    def _update_artists(self):
        # Transform
        transform = Affine2D().rotate_deg_around(0,0,self.theta).translate(self.cx, self.cy)

        # Rectangle
        self.rect_patch.set_x(-self.width/2)
        self.rect_patch.set_y(-self.height/2)
        self.rect_patch.set_width(self.width)
        self.rect_patch.set_height(self.height)
        self.rect_patch.set_transform(transform + self.ax_dyn.transData)

        # Dashed lines
        w2, h2 = self.width/2, self.height/2
        self.dash_lines[0].set_data([-w2, w2], [0, 0])
        self.dash_lines[0].set_transform(transform + self.ax_dyn.transData)
        self.dash_lines[1].set_data([0, 0], [-h2, h2])
        self.dash_lines[1].set_transform(transform + self.ax_dyn.transData)

        # Corners (radius set in data units to a fixed on-screen size)
        corners_local = [
            ( w2,  h2),
            ( w2, -h2),
            (-w2, -h2),
            (-w2,  h2)
        ]
        corner_radius = self._pick_tol_data(6.0)
        for i, corner in enumerate(self.corner_handles):
            gx, gy = self.to_global_coords(*corners_local[i])
            corner.center = (gx, gy)
            corner.set_radius(corner_radius)

        # Edges
        edge_coords = [
            [(-w2/2,  h2), ( w2/2,  h2)],   # top
            [( w2,  h2/2), ( w2,  -h2/2)],  # right
            [( w2/2, -h2), (-w2/2, -h2)],   # bottom
            [(-w2,   -h2/2), (-w2,   h2/2)] # left
        ]
        for i, ed in enumerate(self.edge_lines):
            (lx1, ly1), (lx2, ly2) = edge_coords[i]
            gx1, gy1 = self.to_global_coords(lx1, ly1)
            gx2, gy2 = self.to_global_coords(lx2, ly2)
            ed.set_data([gx1, gx2], [gy1, gy2])

        # Build text
        help_text = (
            "Controls:\n"
            " • Drag center: move\n"
            " • Drag corner: resize\n"
            " • Drag edge: resize one dimension\n"
            " • Shift+Drag center: rotate\n"
            " • Press Enter: confirm"
        )
        # Insert a horizontal line of dashes or underscores
        separator = "\n" + "-" * 30 + "\n"
        stats_text = (
            f"Aperture Status:\n"
            f"  Center = ({self.cx:.1f}, {self.cy:.1f})\n"
            f"  Size   = {self.width:.1f} × {self.height:.1f}\n"
            f"  Angle  = {self.theta:.1f}°\n"
            f"  Mode   = {self._active_mode_str()}"
        )
        combined = help_text + separator + stats_text
        self.help_annotation.set_text(combined)

        # Hover text
        if self.last_motion_event is not None:
            x_now, y_now = self.last_motion_event.xdata, self.last_motion_event.ydata
            if x_now is not None and y_now is not None:
                self.hover_text.set_position((x_now + 10, y_now + 10))
                self.hover_text.set_text(self._hover_text_str())
                self.hover_text.set_visible(True)
            else:
                self.hover_text.set_visible(False)
        else:
            self.hover_text.set_visible(False)

    # ------------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------------
    def _active_mode_str(self):
        if self.rotating:
            return "Rotating"
        elif self.dragging_center:
            return "Moving"
        elif self.dragging_corner:
            return f"Resizing corner {self.active_corner}"
        elif self.dragging_edge:
            return f"Resizing edge {self.active_edge}"
        return "Idle"

    def _hover_text_str(self):
        if self.rotating:
            return "Rotating…"
        if self.dragging_corner:
            return f"Corner {self.active_corner} resize"
        if self.dragging_edge:
            return f"Edge {self.active_edge} resize"
        if self.dragging_center:
            return "Moving aperture"
        return ""

    def _px_per_data(self):
        """Display pixels per data unit on ax_dyn (equal aspect => same in x and y)."""
        try:
            p0, p1 = self.ax_dyn.transData.transform([(0.0, 0.0), (1.0, 0.0)])
            s = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
            return s if s > 1e-9 else 1.0
        except Exception:
            return 1.0

    def _pick_tol_data(self, px=10.0):
        """Convert a screen-pixel pick tolerance to data units, so handles are equally
        easy to grab regardless of image size or zoom level."""
        return px / self._px_per_data()

    def get_corner_index(self, gx, gy):
        w2, h2 = self.width/2, self.height/2
        corners = [
            ( w2,  h2),
            ( w2, -h2),
            (-w2, -h2),
            (-w2,  h2),
        ]
        lx, ly = self.to_local_coords(gx, gy)
        dists = [np.hypot(lx - c[0], ly - c[1]) for c in corners]
        min_d = min(dists)
        idx = dists.index(min_d)
        if min_d < self._pick_tol_data(10.0):
            return idx
        return None

    def get_edge_index(self, gx, gy):
        w2, h2 = self.width/2, self.height/2
        edges = [
            [(-w2/2, h2), ( w2/2, h2)],   # top
            [( w2,   h2/2), ( w2,  -h2/2)],  # right
            [( w2/2, -h2), (-w2/2, -h2)],    # bottom
            [(-w2,   -h2/2), (-w2,   h2/2)]  # left
        ]
        lx, ly = self.to_local_coords(gx, gy)

        def dist_pt_to_line(px,py, x1,y1, x2,y2):
            seg_len2 = (x2-x1)**2 + (y2-y1)**2
            if seg_len2<1e-12:
                return np.hypot(px - x1, py - y1)
            t = ((px-x1)*(x2-x1)+(py-y1)*(y2-y1))/seg_len2
            t = max(0, min(1,t))
            projx = x1 + t*(x2 - x1)
            projy = y1 + t*(y2 - y1)
            return np.hypot(px-projx, py-projy)

        dists = []
        for (ex1, ey1), (ex2, ey2) in edges:
            d = dist_pt_to_line(lx,ly, ex1,ey1, ex2,ey2)
            dists.append(d)
        min_d = min(dists)
        idx = dists.index(min_d)
        if min_d < self._pick_tol_data(10.0):
            return idx
        return None

    def inside_rectangle(self, gx, gy):
        lx, ly = self.to_local_coords(gx, gy)
        return (abs(lx)<=self.width/2) and (abs(ly)<=self.height/2)

    # ------------------------------------------------------------------------
    # 3) Coordinate transforms
    # ------------------------------------------------------------------------
    def to_local_coords(self, gx, gy):
        dx, dy = gx - self.cx, gy - self.cy
        th = np.radians(self.theta)
        cos_t, sin_t = np.cos(-th), np.sin(-th)
        lx = dx * cos_t - dy * sin_t
        ly = dx * sin_t + dy * cos_t
        return lx, ly

    def to_global_coords(self, lx, ly):
        th = np.radians(self.theta)
        cos_t, sin_t = np.cos(th), np.sin(th)
        gx = lx * cos_t - ly * sin_t + self.cx
        gy = lx * sin_t + ly * cos_t + self.cy
        return gx, gy

    def show(self):
        """Do an initial draw (so on_draw stores the backgrounds), render the overlay
        once (the animated artists are not part of a normal draw), then block."""
        self.fig.canvas.draw()
        self._blit_draw()
        plt.show()

        return {
            'position': (self.cx, self.cy),
            'width': self.width,
            'height': self.height,
            'theta': self.theta
        }

def select_aperture_interactive(image):
    sel = ApertureSelectorTwoLayer(image, title="Interactive Aperture Selector")
    return sel.show()

# --------------------------------------------------------------
def confirm_trail_gui(image, norm_type='sqrt', cmap='gray_r'):
    # Single, explicit layout: the figure is sized to the image aspect and all axes are
    # placed with add_axes. (No subplots_adjust / tight_layout - those conflict with
    # manually-placed axes and misplaced the title/buttons.)
    ny, nx = image.shape
    aspect = nx / ny
    img_h_in = 3.2
    img_w_in = min(7.0, max(2.5, img_h_in * aspect))
    fig = plt.figure(figsize=(img_w_in + 0.4, img_h_in + 1.3))

    # Hide the toolbar for this dialog if the backend provides one
    manager = plt.get_current_fig_manager()
    toolbar = getattr(manager, 'toolbar', None)
    if toolbar is not None:
        try:
            toolbar.hide()
        except Exception:
            pass

    fig.suptitle("Mark satellite trail in this image?", fontsize=10)

    # Image axis (with margins; imshow keeps the image at 1:1 aspect within the box)
    ax_img = fig.add_axes([0.06, 0.26, 0.88, 0.62])
    ax_img.set_xticks([])
    ax_img.set_yticks([])

    # Normalize like in the selector (fast subsampled percentiles)
    vmin, vmax = _clip_limits(image, 1.0, 99.5)
    stretch_cls = _STRETCHES.get(norm_type, SqrtStretch)
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=stretch_cls())

    ax_img.imshow(image, cmap=cmap, norm=norm, origin='lower')

    # Button axes (below image, centered)
    yes_ax = fig.add_axes([0.15, 0.06, 0.30, 0.11])
    no_ax = fig.add_axes([0.55, 0.06, 0.30, 0.11])

    btn_yes = Button(yes_ax, "Yes")
    btn_no = Button(no_ax, "No")

    btn_yes.label.set_fontsize(9)
    btn_no.label.set_fontsize(9)

    result = {'answer': None}

    def on_yes(ev):
        result['answer'] = True
        plt.close(fig)

    def on_no(ev):
        result['answer'] = False
        plt.close(fig)

    btn_yes.on_clicked(on_yes)
    btn_no.on_clicked(on_no)
    plt.show()
    return result['answer'] if result['answer'] is not None else False

def confirm_trail_and_select_aperture_gui(image):
    has_trail = confirm_trail_gui(image)
    if has_trail:
        return select_aperture_interactive(image), True
    else:
        return None, False

