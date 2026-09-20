from __future__ import annotations

import tkinter as tk
from tkinter import ttk


def fit_window_to_screen(
    window: tk.Misc,
    *,
    preferred_width: int,
    preferred_height: int,
    min_width: int = 640,
    min_height: int = 480,
    margin_x: int = 48,
    margin_y: int = 96,
) -> None:
    """Size a toplevel so controls remain inside the usable screen area.

    Tk reports the physical screen rather than the desktop work area, so a modest
    vertical margin is reserved for title bars/panels. Content that still exceeds
    the resulting height should live inside a ScrollableFrame with its action bar
    outside that frame.
    """
    window.update_idletasks()
    screen_w = max(640, int(window.winfo_screenwidth()))
    screen_h = max(480, int(window.winfo_screenheight()))
    width = max(min_width, min(preferred_width, screen_w - margin_x))
    height = max(min_height, min(preferred_height, screen_h - margin_y))
    width = min(width, screen_w)
    height = min(height, screen_h)
    window.geometry(f"{width}x{height}")
    window.minsize(min(min_width, width), min(min_height, height))


class ScrollableFrame(ttk.Frame):
    """A vertically scrollable ttk frame with a width-tracking interior."""

    def __init__(self, parent, *, padding: int = 0, **kwargs):
        super().__init__(parent, **kwargs)
        try:
            canvas_bg = parent.winfo_toplevel().cget("background")
        except tk.TclError:
            canvas_bg = None
        kwargs_canvas = {"highlightthickness": 0, "borderwidth": 0}
        if canvas_bg:
            kwargs_canvas["background"] = canvas_bg
        self.canvas = tk.Canvas(self, **kwargs_canvas)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.body = ttk.Frame(self.canvas, padding=padding)
        self._window_id = self.canvas.create_window((0, 0), window=self.body, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.body.bind("<Configure>", self._on_body_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind("<Enter>", self._bind_mousewheel)
        self.canvas.bind("<Leave>", self._unbind_mousewheel)

    def _on_body_configure(self, _event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.canvas.itemconfigure(self._window_id, width=event.width)

    def _bind_mousewheel(self, _event=None):
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel_linux)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel_linux)

    def _unbind_mousewheel(self, _event=None):
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _on_mousewheel(self, event):
        delta = -1 if event.delta > 0 else 1
        self.canvas.yview_scroll(delta, "units")

    def _on_mousewheel_linux(self, event):
        self.canvas.yview_scroll(-1 if event.num == 4 else 1, "units")
