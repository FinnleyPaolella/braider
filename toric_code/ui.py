"""Theme, fonts and rounded-rectangle widgets."""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from typing import Callable

import pygame

# ----------------------------------------------------------------------
# palette
# ----------------------------------------------------------------------
BG = (250, 250, 250)
PANEL = (255, 255, 255)
GRID = (216, 216, 216)
GRID_GHOST = (236, 236, 236)
INK = (32, 34, 38)
INK_SOFT = (122, 128, 138)
INK_FAINT = (176, 181, 190)
HAIRLINE = (234, 234, 238)

X_COL = (224, 74, 74)
Y_COL = (232, 185, 59)
Z_COL = (59, 125, 232)
GATE_COL = {"X": X_COL, "Y": Y_COL, "Z": Z_COL}

E_COL = (46, 168, 128)
M_COL = (150, 92, 214)
ACCENT = (24, 26, 30)
GLOW = (255, 196, 60)
WHITE = (255, 255, 255)

FONT_STACK = "segoeui,segoe ui,inter,helveticaneue,helvetica neue,arial"

WEB = sys.platform == "emscripten"
_FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")


def sysfont(stack: str, size: int, bold: bool = False, italic: bool = False) -> pygame.font.Font:
    """SysFont on the desktop; the bundled DejaVu fonts in the browser."""
    if not WEB:
        return pygame.font.SysFont(stack, size, bold=bold, italic=italic)
    serif = "cambria" in stack
    bold = bold or "semibold" in stack or "black" in stack
    if serif:
        name = "DejaVuSerif-Italic" if italic else "DejaVuSerif"
    else:
        name = "DejaVuSans-Bold" if bold else "DejaVuSans"
    return pygame.font.Font(os.path.join(_FONT_DIR, name + ".ttf"), size)


class Fonts:
    """The font sizes used across the app."""

    def __init__(self) -> None:
        def f(size: int, bold: bool = False) -> pygame.font.Font:
            return sysfont(FONT_STACK, size, bold=bold)

        self.title = f(21, True)
        self.label = f(15)
        self.label_bold = f(15, True)
        self.small = f(13)
        self.tiny = f(11)
        self.gate = f(14, True)
        self.gate_small = f(12, True)
        self.anyon = f(14, True)
        self.state = f(19)
        self.state_bold = f(19, True)
        self.caption = f(14)
        # subscripts for the fonts that set formulas, see math_at
        self._sub = {
            id(self.state): f(13),
            id(self.state_bold): f(13, True),
            id(self.caption): f(11),
            id(self.label): f(11),
            id(self.label_bold): f(11, True),
            id(self.small): f(10),
        }

    def sub(self, font: pygame.font.Font) -> pygame.font.Font:
        """The subscript font that goes with font."""
        return self._sub[id(font)]


def rounded(
    surf: pygame.Surface,
    color,
    rect: pygame.Rect,
    radius: int,
    width: int = 0,
) -> None:
    pygame.draw.rect(surf, color, rect, width, border_radius=radius)


def text_at(
    surf: pygame.Surface,
    font: pygame.font.Font,
    s: str,
    color,
    pos: tuple[float, float],
    anchor: str = "topleft",
) -> pygame.Rect:
    img = font.render(s, True, color)
    rect = img.get_rect(**{anchor: (round(pos[0]), round(pos[1]))})
    surf.blit(img, rect)
    return rect


def blend(a, b, t: float):
    t = max(0.0, min(1.0, t))
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


# ----------------------------------------------------------------------
# formulas: X_12 and (X_1)_L get real subscripts, |00_L> a proper angle bracket
# ----------------------------------------------------------------------
_MATH = re.compile(r"_\{([^}]*)\}|_([A-Za-z0-9]+)|(\|)|(>)|([^_|>]+|_)")


def _math_runs(s: str) -> list[tuple[str, str]]:
    """Split ``s`` into ``(kind, text)`` runs: 'text', 'sub' or 'rangle'.

    A ``>`` only closes a ket when a ``|`` opened one, so arrows stay arrows.
    """
    runs: list[tuple[str, str]] = []
    in_ket = False
    for m in _MATH.finditer(s):
        braced, bare, bar, gt, plain = m.groups()
        if braced is not None or bare is not None:
            runs.append(("sub", braced if braced is not None else bare))
        elif bar is not None:
            in_ket = True
            runs.append(("text", "|"))
        elif gt is not None:
            runs.append(("rangle", "") if in_ket else ("text", ">"))
            in_ket = False
        else:
            runs.append(("text", plain))
    return runs


def _rangle_box(font: pygame.font.Font) -> tuple[int, int, int]:
    """Width, and top/bottom offsets from the text top, of a ket's closing bracket."""
    _, _, miny, maxy, _ = font.metrics("|")[0]
    top = font.get_ascent() - maxy
    bottom = font.get_ascent() - miny
    return max(5, round((bottom - top) * 0.3)), top, bottom


def math_size(font: pygame.font.Font, sub: pygame.font.Font, s: str) -> tuple[int, int]:
    w = 0
    for kind, text in _math_runs(s):
        if kind == "sub":
            w += sub.size(text)[0]
        elif kind == "rangle":
            w += _rangle_box(font)[0] + 2
        else:
            w += font.size(text)[0]
    return w, font.get_height()


_rangle_cache: dict = {}


def _rangle(surf, color, x: float, y: float, font: pygame.font.Font) -> None:
    """Draw a thin anti-aliased ``⟩`` (few UI fonts have the glyph)."""
    w, top, bottom = _rangle_box(font)
    h = bottom - top
    stroke = max(1.4, h / 12) * (1.3 if font.get_bold() else 1.0)
    key = (w, h, round(stroke * 4), tuple(color))
    img = _rangle_cache.get(key)
    if img is None:
        ss = 4
        pad = 2
        big = pygame.Surface(((w + 2 * pad) * ss, (h + 2 * pad) * ss), pygame.SRCALPHA)
        pts = [(pad * ss, pad * ss), ((pad + w) * ss, (pad + h / 2) * ss),
               (pad * ss, (pad + h) * ss)]
        pygame.draw.lines(big, color, False, pts, max(1, round(stroke * ss)))
        for p in pts:  # round the stroke ends and the joint
            pygame.draw.circle(big, color, p, stroke * ss / 2)
        img = pygame.transform.smoothscale(big, (w + 2 * pad, h + 2 * pad))
        _rangle_cache[key] = img
    surf.blit(img, (round(x) - 1, round(y + top) - 2))


def math_at(
    surf: pygame.Surface,
    font: pygame.font.Font,
    sub: pygame.font.Font,
    s: str,
    color,
    pos: tuple[float, float],
    anchor: str = "topleft",
) -> pygame.Rect:
    """Like ``text_at``, but typesets subscripts and ket brackets."""
    w, h = math_size(font, sub, s)
    rect = pygame.Rect(0, 0, w, h)
    setattr(rect, anchor, (round(pos[0]), round(pos[1])))
    x, y = rect.x, rect.y
    # subscript baseline sits a little below the main one
    sub_y = y + font.get_ascent() - sub.get_ascent() + round(font.get_height() * 0.2)
    for kind, text in _math_runs(s):
        if kind == "sub":
            surf.blit(sub.render(text, True, color), (x, sub_y))
            x += sub.size(text)[0]
        elif kind == "rangle":
            _rangle(surf, color, x, y, font)
            x += _rangle_box(font)[0] + 2
        else:
            surf.blit(font.render(text, True, color), (x, y))
            x += font.size(text)[0]
    return rect


# ----------------------------------------------------------------------
# buttons
# ----------------------------------------------------------------------
@dataclass
class Button:
    """A pill-shaped button. ``icon`` draws a glyph instead of a label."""

    key: str
    label: str
    rect: pygame.Rect
    fill: tuple = field(default=(242, 243, 245))
    fill_on: tuple = field(default=ACCENT)
    text_col: tuple = field(default=INK)
    text_on: tuple = field(default=WHITE)
    icon: Callable[[pygame.Surface, pygame.Rect, tuple], None] | None = None
    selected: bool = False
    enabled: bool = True
    hover: bool = False

    def draw(self, surf: pygame.Surface, fonts: Fonts) -> None:
        radius = self.rect.height // 2
        if not self.enabled:
            bg, fg = (246, 246, 248), INK_FAINT
        elif self.selected:
            bg, fg = self.fill_on, self.text_on
        elif self.hover:
            bg, fg = blend(self.fill, (226, 228, 232), 0.7), self.text_col
        else:
            bg, fg = self.fill, self.text_col
        rounded(surf, bg, self.rect, radius)
        if self.icon is not None:
            self.icon(surf, self.rect, fg)
        else:
            text_at(surf, fonts.label_bold, self.label, fg, self.rect.center, "center")

    def hit(self, pos) -> bool:
        return self.enabled and self.rect.collidepoint(pos)


# icon painters -------------------------------------------------------
def icon_play(surf, rect, col) -> None:
    c = rect.center
    s = rect.height * 0.26
    pygame.draw.polygon(
        surf, col,
        [(c[0] - s * 0.55, c[1] - s), (c[0] - s * 0.55, c[1] + s), (c[0] + s * 0.95, c[1])],
    )


def icon_pause(surf, rect, col) -> None:
    c = rect.center
    s = rect.height * 0.26
    w = max(3, int(s * 0.45))
    for dx in (-s * 0.55, s * 0.1):
        pygame.draw.rect(
            surf, col,
            pygame.Rect(round(c[0] + dx), round(c[1] - s), w, round(2 * s)),
            border_radius=2,
        )


def _triangle(surf, col, c, s, facing: int) -> None:
    pygame.draw.polygon(
        surf, col,
        [(c[0] - facing * s * 0.5, c[1] - s), (c[0] - facing * s * 0.5, c[1] + s),
         (c[0] + facing * s * 0.9, c[1])],
    )


def icon_next(surf, rect, col) -> None:
    c = rect.center
    s = rect.height * 0.24
    _triangle(surf, col, (c[0] - s * 0.35, c[1]), s, 1)
    pygame.draw.rect(
        surf, col,
        pygame.Rect(round(c[0] + s * 0.65), round(c[1] - s), max(3, int(s * 0.4)), round(2 * s)),
        border_radius=2,
    )


def icon_prev(surf, rect, col) -> None:
    c = rect.center
    s = rect.height * 0.24
    _triangle(surf, col, (c[0] + s * 0.35, c[1]), s, -1)
    pygame.draw.rect(
        surf, col,
        pygame.Rect(round(c[0] - s * 1.05), round(c[1] - s), max(3, int(s * 0.4)), round(2 * s)),
        border_radius=2,
    )


def icon_check(surf, rect, col) -> None:
    c = rect.center
    s = rect.height * 0.22
    pygame.draw.lines(
        surf, col, False,
        [(c[0] - s, c[1]), (c[0] - s * 0.25, c[1] + s * 0.7), (c[0] + s, c[1] - s * 0.7)],
        3,
    )


def icon_close(surf, rect, col) -> None:
    c = rect.center
    s = rect.height * 0.2
    pygame.draw.line(surf, col, (c[0] - s, c[1] - s), (c[0] + s, c[1] + s), 3)
    pygame.draw.line(surf, col, (c[0] - s, c[1] + s), (c[0] + s, c[1] - s), 3)


# ----------------------------------------------------------------------
# anti-aliased primitives
# ----------------------------------------------------------------------
def aa_circle(surf: pygame.Surface, color, center, radius: float, width: int = 0) -> None:
    """Circle drawn on a 4x supersampled scratch surface, for smooth edges."""
    r = max(1, int(radius))
    ss = 4
    size = (r + 2) * 2 * ss
    scratch = pygame.Surface((size, size), pygame.SRCALPHA)
    pygame.draw.circle(
        scratch, color, (size // 2, size // 2), r * ss, width * ss if width else 0
    )
    small = pygame.transform.smoothscale(scratch, ((r + 2) * 2, (r + 2) * 2))
    surf.blit(small, small.get_rect(center=(round(center[0]), round(center[1]))))
