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
from matplotlib.widgets import Button
from astropy.visualization import (
    ImageNormalize, LinearStretch, LogStretch, SqrtStretch
)

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
    def _setup_figure(self):
        self.fig = plt.figure(figsize=(10, 8))
        self.fig.suptitle(self.title, fontsize=11)

        # 1) Bottom axis for the image only
        self.ax_img = self.fig.add_axes([0.05, 0.05, 0.65, 0.9])

        # 2) Top axis (transparent) for dynamic patches
        self.ax_dyn = self.fig.add_axes([0.05, 0.05, 0.65, 0.9],
                                        sharex=self.ax_img, sharey=self.ax_img)
        self.ax_dyn.set_facecolor("none")
        self.ax_dyn.set_zorder(2)  # ensure it's above self.ax_img

        # Turn off ticks/spines in top axis if you like
        self.ax_dyn.set_xticks([])
        self.ax_dyn.set_yticks([])
        for spine in self.ax_dyn.spines.values():
            spine.set_visible(False)

        # 3) Right axis for help annotation
        self.info_ax = self.fig.add_axes([0.72, 0.05, 0.25, 0.9])
        self.info_ax.axis('off')

        # Show the image in ax_img
        vmin = np.nanpercentile(self.image, 1.)
        vmax = np.nanpercentile(self.image, 99.5)
        if self.img_norm == 'lin':
            norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LinearStretch())
        elif self.img_norm == 'log':
            norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LogStretch())
        else:
            norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SqrtStretch())

        self.ax_img.imshow(self.image, norm=norm, cmap=self.cmap, origin='lower',
            extent=(0, self.nx, 0, self.ny))
        self.ax_img.set_xlim(0, self.nx)
        self.ax_img.set_ylim(0, self.ny)
        self.ax_img.set_aspect('equal', 'box')  # 1:1 aspect ratio
        # self.ax_img.set_autoscale_on(False)

        # Also force no margins and 1:1 aspect here
        self.ax_dyn.set_xlim(0, self.nx)
        self.ax_dyn.set_ylim(0, self.ny)
        self.ax_dyn.set_aspect('equal', 'box')

        # Single annotation in info_ax
        self.help_annotation = self.info_ax.annotate(
            "",
            xy=(0, 1),
            xycoords='axes fraction',
            ha='left', va='top',
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.7)
        )

        # Patches in ax_dyn
        # Rectangle
        self.rect_patch = Rectangle(
            xy=(-self.width/2, -self.height/2),
            width=self.width, height=self.height,
            angle=0,
            fill=False,
            edgecolor='lime',
            linewidth=2,
            alpha=0.8,
            visible=False
        )
        self.ax_dyn.add_patch(self.rect_patch)

        # Dashed lines
        for _ in range(2):
            line = Line2D([], [], linestyle='--', color='black', linewidth=1., visible=False)
            self.ax_dyn.add_line(line)
            self.dash_lines.append(line)

        # Corner handles
        for _ in range(4):
            c = Circle((0, 0), radius=5, facecolor='red', edgecolor='white', lw=1, visible=False)
            self.ax_dyn.add_patch(c)
            self.corner_handles.append(c)

        # Edge handles
        for _ in range(4):
            e = Line2D([], [], color='red', lw=1.5, visible=False)
            self.ax_dyn.add_line(e)
            self.edge_lines.append(e)

        # Hover text
        self.hover_text = self.ax_dyn.text(
            0, 0, "",
            color='yellow', fontsize=8,
            bbox=dict(boxstyle='round', fc='black', alpha=0.7),
            visible=False
        )

        # Hide everything initially
        self.rect_patch.set_visible(False)
        for patch in self.corner_handles + self.edge_lines + self.dash_lines:
            patch.set_visible(False)
        self.hover_text.set_visible(False)

        # Connect events
        self.fig.canvas.mpl_connect('draw_event', self.on_draw)
        self.fig.canvas.mpl_connect('button_press_event', self.on_button_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_button_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

    # ------------------------------------------------------------------------
    # on_draw: We capture the background of the top axis only
    # ------------------------------------------------------------------------
    def on_draw(self, event):
        if event.canvas != self.fig.canvas:
            return

        # ❗️ Clear dynamic axis before capturing the background (removes duplicates)
        self.ax_dyn.cla()
        self.ax_dyn.set_facecolor("none")
        self.ax_dyn.set_xticks([])
        self.ax_dyn.set_yticks([])
        for spine in self.ax_dyn.spines.values():
            spine.set_visible(False)

        # Keep the forced limits and aspect
        self.ax_dyn.set_xlim(0, self.nx)
        self.ax_dyn.set_ylim(0, self.ny)
        self.ax_dyn.set_aspect('equal', 'box')

        # Re-capture clean background
        self.dyn_background = self.fig.canvas.copy_from_bbox(self.ax_dyn.bbox)

        ## Ensure all patches are now visible
        self.rect_patch.set_visible(True)
        for patch in self.corner_handles + self.edge_lines + self.dash_lines:
            patch.set_visible(True)
        self.hover_text.set_visible(False)

        self.first_draw_complete = True

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

        self._update_artists()

        # Restore background
        self.fig.canvas.restore_region(self.dyn_background)

        # Draw dynamic artists
        self.ax_dyn.draw_artist(self.rect_patch)
        for ln in self.dash_lines:
            self.ax_dyn.draw_artist(ln)
        for corner in self.corner_handles:
            self.ax_dyn.draw_artist(corner)
        for ed in self.edge_lines:
            self.ax_dyn.draw_artist(ed)

        # Ensure hover text is drawn only if figure is valid
        if self.hover_text.get_visible() and self.fig is not None:
            try:
                self.ax_dyn.draw_artist(self.hover_text)
            except AttributeError:
                pass  # If the figure was closed, don't try to render text

        # Also re-draw the annotation in the info axis
        self.info_ax.draw_artist(self.help_annotation)

        # Blit only if figure is still valid
        if plt.fignum_exists(self.fig.number):
            self.fig.canvas.blit(self.ax_dyn.bbox)
            self.fig.canvas.blit(self.info_ax.bbox)

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

        # Corners
        corners_local = [
            ( w2,  h2),
            ( w2, -h2),
            (-w2, -h2),
            (-w2,  h2)
        ]
        for i, corner in enumerate(self.corner_handles):
            gx, gy = self.to_global_coords(*corners_local[i])
            corner.center = (gx, gy)

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
        if min_d < 10:
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
        if min_d<10:
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
        """Do an initial draw (so on_draw will store the top axis background), then block."""
        self.fig.canvas.draw()
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
    fig = plt.figure(figsize=(3.5, 3.5))
    manager = plt.get_current_fig_manager()
    if hasattr(manager, 'toolbar'):
        manager.toolbar.hide()  # ✅ hide toolbar just for this figure

    fig.subplots_adjust(top=0.8, bottom=0.125, left=0.1, right=0.9)

    # Title
    fig.suptitle("Mark satellite trail in this image?", fontsize=10)

    # Image axis
    ax_img = fig.add_axes([0, 0.3, 1, 0.6])
    ax_img.set_xticks([])
    ax_img.set_yticks([])

    # Normalize like in the selector
    vmin = float(np.nanpercentile(image, 1.0))
    vmax = float(np.nanpercentile(image, 99.5))
    if norm_type == 'lin':
        norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LinearStretch())
    elif norm_type == 'log':
        norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LogStretch())
    else:
        norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SqrtStretch())

    ax_img.imshow(image, cmap=cmap, norm=norm, origin='lower')

    # Button axes (below image, centered)
    yes_ax = fig.add_axes([0.15, 0.1, 0.3, 0.1])
    no_ax = fig.add_axes([0.55, 0.1, 0.3, 0.1])

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
    plt.tight_layout()
    plt.show()
    return result['answer'] if result['answer'] is not None else False

def confirm_trail_and_select_aperture_gui(image):
    has_trail = confirm_trail_gui(image)
    if has_trail:
        return select_aperture_interactive(image), True
    else:
        return None, False

