"""Fit the pygame window to the browser page when running under pygbag.

pygbag draws the pygame surface into an HTML canvas and stretches that canvas
to a fixed aspect ratio, which leaves borders and blurs it on high-DPI screens.
Here the surface is instead made as large as the page in device pixels and the
canvas is shown at exactly that size, so one surface pixel is one screen pixel.
On the desktop every function is a no-op.
"""

from __future__ import annotations

import sys

WEB = sys.platform == "emscripten"
POLL_SECONDS = 0.25        # how often to check whether the page was resized

_last: tuple[int, int] | None = None
_since_poll = 0.0


def viewport() -> tuple[int, int] | None:
    """Size of the browser page in device pixels, or None on the desktop."""
    if not WEB:
        return None
    import platform  # pygbag's module, with access to the browser
    win = platform.window
    dpr = float(win.devicePixelRatio or 1)
    return round(win.innerWidth * dpr), round(win.innerHeight * dpr)


def fit(size: tuple[int, int], background: tuple[int, int, int]) -> None:
    """Show a surface of ``size`` pixels at one pixel per device pixel.

    Call after every ``set_mode``. The canvas is only scaled (down) when the
    page is smaller than the surface.
    """
    global _last
    if not WEB:
        return
    import platform
    win = platform.window
    win.config.user_canvas_managed = 1     # stop pygbag's own resizing
    dpr = float(win.devicePixelRatio or 1)
    page_w, page_h = win.innerWidth, win.innerHeight
    w, h = size
    scale = min(1.0, page_w * dpr / w, page_h * dpr / h)
    style = win.canvas.style
    style.width = f"{w * scale / dpr}px"
    style.height = f"{h * scale / dpr}px"
    style.position = "absolute"
    style.left = style.top = style.right = style.bottom = "0"
    style.margin = "auto"
    body = platform.document.body.style
    body.background = "rgb({}, {}, {})".format(*background)
    body.overflow = "hidden"
    _last = viewport()


def poll(dt: float) -> tuple[int, int] | None:
    """The new page size in device pixels if the page was resized, else None."""
    global _since_poll
    if not WEB:
        return None
    _since_poll += dt
    if _since_poll < POLL_SECONDS:
        return None
    _since_poll = 0.0
    size = viewport()
    return size if size != _last else None
