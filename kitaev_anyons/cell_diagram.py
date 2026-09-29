"""The couplings of one unit cell: what the Bloch matrix needs.

The home cell holds an A site (even sublattice) and a B site joined by its
z-link. Its A site is linked to B sites in the cells 0, n1, n2 (weights
2 J_alpha u_jk), and each sublattice has three independent second-neighbour
terms, towards the cells n1, n2 and n1 - n2 (weight 2 kappa); every other
coupling follows by translation and A_kj = -A_jk. The arrows point along
A_jk = +2 kappa in the gauge u = +1; their directions are read from the
Bloch terms in majorana.py, so they match the matrix actually used.
"""

from __future__ import annotations

import math

import pygame

import majorana as mj
import mathtext as mt
from ui import (
    BOND_COL, C_COL, INK, INK_SOFT, WHITE, aa_circle, arrow, blend,
    draw_stroke,
)

KAPPA_COL = (150, 96, 206)
SQ3 = math.sqrt(3.0)
N1 = (SQ3 / 2, 1.5)
N2 = (-SQ3 / 2, 1.5)


def cell_pos(i: int, j: int, s: int) -> tuple[float, float]:
    """Position of sublattice s (0 = A, 1 = B) in cell i n1 + j n2 (y up)."""
    x = i * N1[0] + j * N2[0]
    y = i * N1[1] + j * N2[1] - (1.0 if s else 0.0)
    return x, y


# the cells that appear, with their labels
CELLS = {(0, 0): "0", (1, 0): "n_1", (0, 1): "n_2", (1, -1): "n_1 − n_2"}
NNN = ((1, 0), (0, 1), (1, -1))


def nnn_arrows() -> list[tuple[int, tuple[int, int], bool]]:
    """(sublattice, cell offset, outgoing): outgoing when A_{home, neighbour} = +2 kappa."""
    out = []
    for s, t, di, dj, amp in mj.bloch_terms((1.0, 1.0, 1.0), 1.0):
        if s == t and (di, dj) in NNN:
            a = (amp / 1j).real          # amp = i A_jk
            out.append((s, (di, dj), a > 0))
    return out


class CellDiagram:
    def __init__(self) -> None:
        self._cache: pygame.Surface | None = None
        self._key = None

    def draw(self, surf: pygame.Surface, rect: pygame.Rect, J, kappa: float, kappa_max: float) -> None:
        """Link widths and opacities follow J_alpha, the arrows' follow |kappa| / kappa_max."""
        key = (rect.size, rect.topleft, tuple(round(j, 3) for j in J), round(kappa / kappa_max, 3))
        if key != self._key:
            self._key = key
            self._cache = self._render(rect.size, J, abs(kappa) / kappa_max)
        surf.blit(self._cache, rect)

    def _render(self, size, J, k: float) -> pygame.Surface:
        W, H = size
        surf = pygame.Surface(size, pygame.SRCALPHA)
        # world window around the sites that appear
        x0, x1 = -SQ3 / 2 - 0.75, SQ3 + 0.95
        y0, y1 = -1.0 - 0.75, 1.5 + 0.75
        scale = min(W / (x1 - x0), H / (y1 - y0))
        ox = (W - (x1 - x0) * scale) / 2 - x0 * scale
        oy = (H - (y1 - y0) * scale) / 2 + y1 * scale

        def P(p):
            return ox + p[0] * scale, oy - p[1] * scale

        # faint honeycomb behind everything
        for i in range(-2, 3):
            for j in range(-2, 3):
                a = cell_pos(i, j, 0)
                for bi, bj in ((i, j), (i + 1, j), (i, j + 1)):
                    b = cell_pos(bi, bj, 1)
                    pa, pb = P(a), P(b)
                    if all(-20 < v[0] < W + 20 and -20 < v[1] < H + 20 for v in (pa, pb)):
                        draw_stroke(surf, [pa, pb], 1.2, (232, 234, 239))

        # unit cells as pills around their z-link, labelled above or below
        r = 0.36 * scale
        for (i, j), name in CELLS.items():
            a, b = P(cell_pos(i, j, 0)), P(cell_pos(i, j, 1))
            home = (i, j) == (0, 0)
            _pill(surf, a, b, r, (226, 229, 236) if home else (241, 242, 246))
            lab = mt.formula(name, 16 if home else 15, INK if home else INK_SOFT)
            above = (i, j) in ((1, 0), (0, 1))
            y = min(a[1], b[1]) - r - 6 if above else max(a[1], b[1]) + r + 4 + lab.h
            lab.blit(surf, a[0] - lab.w / 2, y - lab.h + lab.base)

        # the three links of the home A site: to B in the cells 0, n1, n2
        # (each coupling is drawn opaque on its own layer, which is then faded,
        # so overlapping pieces such as an arrow's shaft and head do not darken)
        home_a = P(cell_pos(0, 0, 0))
        Jd = dict(zip("xyz", J))
        for kind, cell in (("z", (0, 0)), ("x", (1, 0)), ("y", (0, 1))):
            b = P(cell_pos(*cell, 1))
            j = Jd[kind]
            layer = pygame.Surface(size, pygame.SRCALPHA)
            draw_stroke(layer, [home_a, b], max(1.2, scale * (0.015 + 0.17 * j)), BOND_COL[kind])
            _fade(layer, 0.22 + 0.78 * min(1.0, j / 0.45))
            surf.blit(layer, (0, 0))

        # the second-neighbour kappa terms, three per sublattice
        rs = max(4.0, scale * 0.09)
        for s, cell, outgoing in nnn_arrows():
            p, q = P(cell_pos(0, 0, s)), P(cell_pos(*cell, s))
            if not outgoing:
                p, q = q, p
            L = math.dist(p, q)
            ux, uy = (q[0] - p[0]) / L, (q[1] - p[1]) / L
            a = (p[0] + ux * rs * 1.8, p[1] + uy * rs * 1.8)
            b = (q[0] - ux * rs * 1.8, q[1] - uy * rs * 1.8)
            col = KAPPA_COL if s == 0 else blend(KAPPA_COL, INK, 0.25)
            layer = pygame.Surface(size, pygame.SRCALPHA)
            arrow(layer, a, b, col, 1.0 + 3.0 * k, 8 + 7 * k)
            _fade(layer, 0.14 + 0.86 * min(1.0, k / 0.5))
            surf.blit(layer, (0, 0))

        # sites: A filled, B hollow; home sites a bit larger
        for (i, j) in CELLS:
            for s in (0, 1):
                p = P(cell_pos(i, j, s))
                rr = rs * (1.25 if (i, j) == (0, 0) else 1.0)
                aa_circle(surf, C_COL, p, rr)
                if s:
                    aa_circle(surf, WHITE, p, rr - max(1.5, rr * 0.35))
        return surf


def _fade(layer: pygame.Surface, opacity: float) -> None:
    layer.fill((255, 255, 255, round(255 * opacity)), special_flags=pygame.BLEND_RGBA_MULT)


def _pill(surf, a, b, r: float, colour) -> None:
    """A rounded band of radius r around the segment ab (vertical here)."""
    top, bot = (a, b) if a[1] < b[1] else (b, a)
    rect = pygame.Rect(0, 0, round(2 * r), round(bot[1] - top[1] + 2 * r))
    rect.midtop = (round(top[0]), round(top[1] - r))
    big = pygame.Surface((rect.w * 4, rect.h * 4), pygame.SRCALPHA)
    pygame.draw.rect(big, colour, big.get_rect(), border_radius=round(r * 4))
    surf.blit(pygame.transform.smoothscale(big, rect.size), rect)
