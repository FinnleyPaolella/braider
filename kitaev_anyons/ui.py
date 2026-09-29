"""Palette, fonts and small drawing helpers shared by the panels."""

from __future__ import annotations

import math
import os
import sys
from functools import lru_cache

import pygame

# ----------------------------------------------------------------------
# palette (the x / y / z colours match the toric-code viewer's X / Y / Z)
# ----------------------------------------------------------------------
BG = (246, 247, 249)
CARD = (255, 255, 255)
HAIRLINE = (228, 230, 235)
INK = (28, 30, 35)
INK_SOFT = (112, 118, 128)
INK_FAINT = (170, 175, 184)
WHITE = (255, 255, 255)
ACCENT = (24, 26, 30)

X_COL = (224, 74, 74)
Y_COL = (222, 158, 24)
Z_COL = (59, 125, 232)
BOND_COL = {"x": X_COL, "y": Y_COL, "z": Z_COL}
C_COL = (40, 44, 52)           # the matter Majorana c

UI_STACK = "segoeui,segoe ui,inter,helveticaneue,helvetica neue,arial"

WEB = sys.platform == "emscripten"
_FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")


def sysfont(stack: str, size: int, bold: bool = False, italic: bool = False) -> pygame.font.Font:
    """SysFont on the desktop; the bundled DejaVu fonts in the browser."""
    if not WEB:
        return pygame.font.SysFont(stack, size, bold=bold, italic=italic)
    serif = "cambria" in stack
    bold = bold or "semibold" in stack
    if serif:
        name = "DejaVuSerif-Italic" if italic else "DejaVuSerif"
    else:
        name = "DejaVuSans-Bold" if bold else "DejaVuSans"
    return pygame.font.Font(os.path.join(_FONT_DIR, name + ".ttf"), size)


class Fonts:
    def __init__(self) -> None:
        self.title = sysfont("segoeuisemibold," + UI_STACK, 20)
        self.caps = sysfont("segoeuisemibold," + UI_STACK, 13)
        self.ui = sysfont(UI_STACK, 17)
        self.ui_bold = sysfont("segoeuisemibold," + UI_STACK, 17)
        self.small = sysfont(UI_STACK, 15)
        self.tiny = sysfont(UI_STACK, 14)


def blend(a, b, t: float):
    t = max(0.0, min(1.0, t))
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b))


def text_at(surf, font, s: str, color, pos, anchor: str = "topleft") -> pygame.Rect:
    img = font.render(s, True, color)
    rect = img.get_rect(**{anchor: (round(pos[0]), round(pos[1]))})
    surf.blit(img, rect)
    return rect


def card(surf: pygame.Surface, rect: pygame.Rect) -> None:
    """A white rounded panel with a hairline border."""
    pygame.draw.rect(surf, CARD, rect, border_radius=14)
    pygame.draw.rect(surf, HAIRLINE, rect, 1, border_radius=14)


def caps(surf, fonts: Fonts, s: str, pos) -> pygame.Rect:
    """A small letter-spaced heading in capitals."""
    x, y = pos
    for ch in s.upper():
        img = fonts.caps.render(ch, True, INK_SOFT)
        surf.blit(img, (x, y))
        x += img.get_width() + 1.2
    return pygame.Rect(pos[0], y, x - pos[0], fonts.caps.get_height())


# ----------------------------------------------------------------------
# anti-aliased line art (drawn supersampled on a scratch surface)
# ----------------------------------------------------------------------
SS = 4


def stroke(points, width: float, color, closed: bool = False) -> tuple[pygame.Surface, tuple[int, int]]:
    """A smooth polyline as (image, top-left) in the points' coordinates."""
    pad = width + 2
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0, y0 = math.floor(min(xs) - pad), math.floor(min(ys) - pad)
    w = math.ceil(max(xs) + pad) - x0
    h = math.ceil(max(ys) + pad) - y0
    big = pygame.Surface((w * SS, h * SS), pygame.SRCALPHA)
    big.fill((*color[:3], 0))
    pts = [((x - x0) * SS, (y - y0) * SS) for x, y in points]
    lw = max(1, round(width * SS))
    pygame.draw.lines(big, color, closed, pts, lw)
    for p in (pts if closed else pts[:: max(1, len(pts) - 1)]):
        pygame.draw.circle(big, color, p, lw / 2)
    for p in pts[1:-1]:  # round joints
        pygame.draw.circle(big, color, p, lw / 2)
    return pygame.transform.smoothscale(big, (w, h)), (x0, y0)


@lru_cache(maxsize=512)
def _cached_stroke(points, width, color, closed):
    return stroke(points, width, color, closed)


def draw_stroke(surf, points, width: float, color, closed: bool = False) -> None:
    pts = tuple((round(x, 1), round(y, 1)) for x, y in points)
    img, pos = _cached_stroke(pts, round(width, 2), tuple(color), closed)
    surf.blit(img, pos)


def aa_circle(surf, color, center, r: float, width: float = 0) -> None:
    size = int(2 * r + 4)
    big = pygame.Surface((size * SS, size * SS), pygame.SRCALPHA)
    big.fill((*color[:3], 0))
    pygame.draw.circle(big, color, (size * SS / 2, size * SS / 2), r * SS,
                       round(width * SS) if width else 0)
    small = pygame.transform.smoothscale(big, (size, size))
    surf.blit(small, small.get_rect(center=(round(center[0]), round(center[1]))))


def arrow(surf, a, b, color, width: float = 1.6, head: float = 8.0) -> None:
    """An anti-aliased arrow from a to b."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    n = math.hypot(dx, dy) or 1.0
    ux, uy = dx / n, dy / n
    tip = (b[0] - ux * 1.0, b[1] - uy * 1.0)
    base = (b[0] - ux * head, b[1] - uy * head)
    draw_stroke(surf, [a, base], width, color)
    wing = head * 0.5
    poly = [tip, (base[0] - uy * wing, base[1] + ux * wing),
            (base[0] + uy * wing, base[1] - ux * wing)]
    smooth_polygon(surf, color, poly)


def smooth_polygon(surf, color, pts) -> None:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, y0 = math.floor(min(xs)) - 2, math.floor(min(ys)) - 2
    w = math.ceil(max(xs)) - x0 + 3
    h = math.ceil(max(ys)) - y0 + 3
    big = pygame.Surface((w * SS, h * SS), pygame.SRCALPHA)
    big.fill((*color[:3], 0))
    pygame.draw.polygon(big, color, [((x - x0) * SS, (y - y0) * SS) for x, y in pts])
    surf.blit(pygame.transform.smoothscale(big, (w, h)), (x0, y0))


# ----------------------------------------------------------------------
# colour map for energies (viridis, sampled)
# ----------------------------------------------------------------------
_VIRIDIS = [
    (68, 1, 84), (72, 35, 116), (64, 67, 135), (52, 94, 141), (41, 120, 142),
    (32, 144, 140), (34, 167, 132), (68, 190, 112), (121, 209, 81),
    (189, 222, 38), (253, 231, 37),
]


def _colormap(t: float):
    t = max(0.0, min(1.0, t)) * (len(_VIRIDIS) - 1)
    i = min(int(t), len(_VIRIDIS) - 2)
    return blend(_VIRIDIS[i], _VIRIDIS[i + 1], t - i)


LUT = [_colormap(i / 255) for i in range(256)]


def cmap(t: float):
    return LUT[max(0, min(255, int(t * 255)))]
