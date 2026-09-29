"""The finite honeycomb cluster: c Majoranas on the sites, u on the links,
w_p in the hexagons. Links that cross a seam of the torus are drawn as two
half-links (stubs), one leaving each end; either stub flips that link.
"""

from __future__ import annotations

import math

import pygame

import majorana as mj
from ui import (
    BOND_COL, C_COL, CARD, INK, WHITE, aa_circle, blend, draw_stroke,
    smooth_polygon, text_at,
)

FLUX_FILL = (220, 212, 240)       # hexagon with w_p = -1
FLUX_INK = (86, 58, 150)
PLUS_INK = (196, 199, 207)
MODE_COL = (18, 150, 124)       # the selected eigenmode
MODE_R = 0.48                   # its largest circle, in bond lengths
SEAM_COL = {"periodic": (175, 180, 190), "antiperiodic": (150, 96, 206)}


class LatticeView:
    def __init__(self, fonts) -> None:
        self.fonts = fonts
        self.rect = pygame.Rect(0, 0, 1, 1)
        self.lat: mj.Lattice | None = None
        self.scale = 1.0
        self.origin = (0.0, 0.0)
        self.segments: list[tuple[int, tuple, tuple]] = []   # (bond, p, q) on screen
        self.hover: int | None = None
        self._cache = None
        self._key = None
        self._mode_cache = None
        self._mode_key = None

    # ------------------------------------------------------------------
    def layout(self, rect: pygame.Rect, lat: mj.Lattice) -> None:
        self.rect = rect
        self.lat = lat
        xs = [p[0] for p in lat.pos]
        ys = [p[1] for p in lat.pos]
        pad = 0.9
        x0, x1 = min(xs) - pad, max(xs) + pad
        y0, y1 = min(ys) - pad, max(ys) + 1.0 + pad     # top hexagons stick out by 1
        self.scale = min(rect.w / (x1 - x0), rect.h / (y1 - y0))
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        self.origin = (rect.centerx - cx * self.scale, rect.centery + cy * self.scale)
        self._key = None

    def to_screen(self, p) -> tuple[float, float]:
        return self.origin[0] + p[0] * self.scale, self.origin[1] - p[1] * self.scale

    def _segments(self, bc) -> list[tuple[int, tuple, tuple]]:
        """Screen segments per bond: one full link, or two stubs across a seam."""
        lat = self.lat
        out = []
        for i, bond in enumerate(lat.bonds):
            pa, pb = lat.pos[bond.a], lat.pos[bond.b]
            if not bond.seam:
                out.append((i, self.to_screen(pa), self.to_screen(pb)))
            elif lat.active(bond, bc):
                d = mj.DELTA[bond.kind]
                half = (d[0] / 2, d[1] / 2)
                out.append((i, self.to_screen(pa), self.to_screen((pa[0] + half[0], pa[1] + half[1]))))
                out.append((i, self.to_screen(pb), self.to_screen((pb[0] - half[0], pb[1] - half[1]))))
        return out

    def bond_at(self, pos) -> int | None:
        best, best_d = None, 0.28 * self.scale
        for i, p, q in self.segments:
            d = _seg_dist(pos, p, q)
            if d < best_d:
                best, best_d = i, d
        return best

    # ------------------------------------------------------------------
    def draw(self, surf, u, bc, fluxes, mouse, mode=None) -> None:
        """``mode``: (key, amplitudes |psi_j|) of a mode to overlay, or None."""
        key = (tuple(u), tuple(bc), self.rect.topleft, self.rect.size, id(self.lat))
        if key != self._key:
            self._key = key
            self.segments = self._segments(bc)
            self._cache = self._render(u, bc, fluxes)
            self._mode_key = None
        surf.blit(self._cache, self.rect.topleft)
        if mode is not None:
            if mode[0] != self._mode_key:
                self._mode_key = mode[0]
                self._mode_cache = self._render_mode(mode[1])
            surf.blit(self._mode_cache, self.rect.topleft)

        self.hover = self.bond_at(mouse) if self.rect.collidepoint(mouse) else None
        if self.hover is not None:
            bond = self.lat.bonds[self.hover]
            for i, p, q in self.segments:
                if i == self.hover:
                    draw_stroke(surf, [p, q], max(7.0, self.scale * 0.2), (255, 214, 120))
                    draw_stroke(surf, [p, q], self._width(u[i]), BOND_COL[bond.kind])
            for s in (bond.a, bond.b):
                self._site(surf, s)

    def _render_mode(self, amps) -> pygame.Surface:
        """A circle on every site with radius proportional to |psi_j| (so its
        area is proportional to the weight |psi_j|^2), largest = MODE_R bonds."""
        surf = pygame.Surface(self.rect.size, pygame.SRCALPHA)
        ox, oy = self.rect.topleft
        top = max(amps) or 1.0
        for s, a in enumerate(amps):
            r = MODE_R * self.scale * a / top
            if r < 0.8:
                continue
            x, y = self.to_screen(self.lat.pos[s])
            aa_circle(surf, (*MODE_COL, 90), (x - ox, y - oy), r)
            aa_circle(surf, (*MODE_COL, 220), (x - ox, y - oy), r, width=1.3)
        return surf

    def _width(self, ub: int) -> float:
        return max(2.5, self.scale * (0.1 if ub < 0 else 0.07))

    def _site(self, surf, s: int) -> None:
        p = self.to_screen(self.lat.pos[s])
        r = max(3.0, self.scale * 0.1)
        if s % 2 == 0:
            aa_circle(surf, C_COL, p, r)
        else:
            aa_circle(surf, C_COL, p, r)
            aa_circle(surf, WHITE, p, r - max(1.4, r * 0.35))

    def _render(self, u, bc, fluxes) -> pygame.Surface:
        lat = self.lat
        ox, oy = self.rect.topleft
        surf = pygame.Surface(self.rect.size, pygame.SRCALPHA)
        surf.fill((*CARD, 0))
        sub = surf.subsurface(surf.get_rect())
        shift = lambda p: (p[0] - ox, p[1] - oy)  # noqa: E731

        # hexagons with w_p = -1
        for p, w in zip(lat.plaquettes, fluxes):
            if w == -1:
                c = p.center
                pts = [shift(self.to_screen((c[0] + math.cos(math.pi / 6 + k * math.pi / 3),
                                             c[1] + math.sin(math.pi / 6 + k * math.pi / 3))))
                       for k in range(6)]
                smooth_polygon(sub, FLUX_FILL, pts)

        # seams: dashed lines through the middles of the crossing links
        for seam in (1, 2):
            mode = bc[seam - 1]
            if mode == "open":
                continue
            ends = self._seam_lines(seam)
            for a, b in ends:
                _dashed(sub, shift(a), shift(b), SEAM_COL[mode], 1.6 if mode == "periodic" else 2.4,
                        max(4.0, self.scale * 0.12))

        # links: a flipped link u = -1 gets a dark casing
        for i, p, q in self.segments:
            bond = lat.bonds[i]
            if u[i] < 0:
                draw_stroke(sub, [shift(p), shift(q)], max(6.0, self.scale * 0.2), blend(INK, CARD, 0.1))
            draw_stroke(sub, [shift(p), shift(q)], self._width(u[i]), BOND_COL[bond.kind])

        for s in range(lat.N):
            p = self.to_screen(lat.pos[s])
            r = max(3.0, self.scale * 0.1)
            aa_circle(sub, C_COL, shift(p), r)
            if s % 2:
                aa_circle(sub, WHITE, shift(p), r - max(1.4, r * 0.35))

        # the plaquette signs
        font = self.fonts.plaquette
        for p, w in zip(lat.plaquettes, fluxes):
            if w is None:
                continue
            c = shift(self.to_screen(p.center))
            if w == 1:
                text_at(sub, font, "+1", PLUS_INK, c, "center")
            else:
                text_at(sub, self.fonts.plaquette_bold, "−1", FLUX_INK, c, "center")
        return surf

    def _seam_lines(self, seam: int) -> list[tuple[tuple, tuple]]:
        """Both copies of a seam: through the crossing links' midpoints on each side."""
        lat = self.lat
        near, far = [], []
        for bond in lat.bonds:
            if bond.seam != seam:
                continue
            d = mj.DELTA[bond.kind]
            pa, pb = lat.pos[bond.a], lat.pos[bond.b]
            near.append(self.to_screen((pa[0] + d[0] / 2, pa[1] + d[1] / 2)))
            far.append(self.to_screen((pb[0] - d[0] / 2, pb[1] - d[1] / 2)))
        out = []
        for pts in (near, far):
            pts.sort(key=lambda p: (p[1], p[0]))
            ext = 0.6 * self.scale
            a, b = pts[0], pts[-1]
            L = math.dist(a, b) or 1.0
            ux, uy = (b[0] - a[0]) / L, (b[1] - a[1]) / L
            out.append(((a[0] - ux * ext, a[1] - uy * ext), (b[0] + ux * ext, b[1] + uy * ext)))
        return out


def _seg_dist(p, a, b) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    L2 = dx * dx + dy * dy or 1.0
    t = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L2))
    return math.dist(p, (a[0] + t * dx, a[1] + t * dy))


def _dashed(surf, a, b, colour, width: float, dash: float) -> None:
    L = math.dist(a, b)
    n = max(1, int(L / (2 * dash)))
    for k in range(n):
        t0, t1 = (2 * k) / (2 * n), (2 * k + 1) / (2 * n)
        p = (a[0] + (b[0] - a[0]) * t0, a[1] + (b[1] - a[1]) * t0)
        q = (a[0] + (b[0] - a[0]) * t1, a[1] + (b[1] - a[1]) * t1)
        draw_stroke(surf, [p, q], width, colour)
