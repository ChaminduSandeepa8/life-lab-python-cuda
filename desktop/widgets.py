"""Bounded-memory grid renderer and scrollable native window content."""
import math
import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import Image, ImageDraw, ImageTk

BG = "#0b1424"
PANEL = "#111e32"
TEXT = "#e4ecf8"
MUTED = "#9bb0cc"


class ScrollableFrame(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, background=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.content = ttk.Frame(self.canvas)
        self.window = self.canvas.create_window(0, 0, anchor="nw", window=self.content)
        self.content.bind("<Configure>", lambda _: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self.window, width=e.width))
        self.top = self.winfo_toplevel()
        self.wheel_bindings = {sequence: self.top.bind(sequence, self._wheel, add="+")
                               for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>")}

    def _wheel(self, event):
        widget = event.widget
        # A grid has its own scrolling; do not move both it and the surrounding page.
        while widget is not self:
            if widget is None or isinstance(widget, (GridCanvas, ttk.Combobox, ttk.Scale)):
                return
            widget = getattr(widget, "master", None)
        direction = -1 if event.num == 4 or getattr(event, "delta", 0) > 0 else 1
        self.canvas.yview_scroll(direction * 3, "units")
        return "break"

    def destroy(self):
        for sequence, binding in self.wheel_bindings.items():
            self.top.unbind(sequence, binding)
        super().destroy()


class GridCanvas(ttk.Frame):
    """Render just visible cells, even when a 1024² grid is zoomed to 12 px."""
    def __init__(self, parent, color="#14b8a6", height=230):
        super().__init__(parent, style="Panel.TFrame")
        self.color = color
        self.grid_data = None
        self.cell_size = 5
        self.photo = None
        self.pending = None
        self.center_pending = False
        self.canvas = tk.Canvas(self, bg="#0a1221", highlightthickness=0, height=height, width=300)
        self.xbar = ttk.Scrollbar(self, orient="horizontal", command=self._xview)
        self.ybar = ttk.Scrollbar(self, orient="vertical", command=self._yview)
        self.canvas.configure(xscrollcommand=self.xbar.set, yscrollcommand=self.ybar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.ybar.grid(row=0, column=1, sticky="ns")
        self.xbar.grid(row=1, column=0, sticky="ew")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.image_id = self.canvas.create_image(0, 0, anchor="nw")
        self.canvas.bind("<Configure>", lambda _: self.schedule())
        self.canvas.bind("<Button-4>", lambda _: self._yview("scroll", -3, "units"))
        self.canvas.bind("<Button-5>", lambda _: self._yview("scroll", 3, "units"))
        self.canvas.bind("<MouseWheel>", lambda e: self._yview("scroll", -1 if e.delta > 0 else 1, "units"))

    def _xview(self, *args):
        self.canvas.xview(*args)
        self.schedule()

    def _yview(self, *args):
        self.canvas.yview(*args)
        self.schedule()

    def set_grid(self, grid, cell_size=None, reset=False):
        self.grid_data = grid
        if grid is None:
            self.canvas.itemconfigure(self.image_id, image="")
        if cell_size is not None:
            self.cell_size = cell_size
        if reset:
            self.center_pending = True
        self.schedule()

    def schedule(self):
        if self.pending is None:
            self.pending = self.after_idle(self.render)

    def render(self):
        self.pending = None
        if self.grid_data is None:
            return
        a, px = self.grid_data, self.cell_size
        h, w = a.shape
        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        ox, oy = max(0, (cw - w * px) // 2), max(0, (ch - h * px) // 2)
        self.canvas.configure(scrollregion=(0, 0, max(cw, w*px), max(ch, h*px)))
        if self.center_pending:
            self.canvas.xview_moveto(max(0, (w*px - cw) / 2) / max(cw, w*px))
            self.canvas.yview_moveto(max(0, (h*px - ch) / 2) / max(ch, h*px))
            self.center_pending = False
        x = max(0, min(w - 1, math.floor((self.canvas.canvasx(0) - ox) / px)))
        y = max(0, min(h - 1, math.floor((self.canvas.canvasy(0) - oy) / px)))
        right, bottom = min(w, x + math.ceil(cw / px) + 2), min(h, y + math.ceil(ch / px) + 2)
        rgb = tuple(int(self.color[i:i+2], 16) for i in (1, 3, 5))
        palette = np.asarray([[10, 18, 33], rgb], dtype=np.uint8)
        image = Image.fromarray(palette[a[y:bottom, x:right]])
        image = image.resize(((right - x)*px, (bottom - y)*px), Image.Resampling.NEAREST)
        if px >= 6:
            pen = ImageDraw.Draw(image)
            for line in range(0, image.width, px):
                pen.line((line, 0, line, image.height), fill="#17263b")
            for line in range(0, image.height, px):
                pen.line((0, line, image.width, line), fill="#17263b")
        self.photo = ImageTk.PhotoImage(image, master=self.canvas)
        self.canvas.itemconfigure(self.image_id, image=self.photo)
        self.canvas.coords(self.image_id, x*px + ox, y*px + oy)

    def destroy(self):
        if self.pending is not None:
            self.after_cancel(self.pending)
        super().destroy()
