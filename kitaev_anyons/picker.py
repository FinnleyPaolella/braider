"""The coupling triangle: drag a point in the plane J_x + J_y + J_z = 1.

A point's barycentric coordinates are (J_x, J_y, J_z). The shading is
Kitaev's phase diagram: gapped abelian corners A_x, A_y, A_z (Chern number 0)
and the B phase in the middle. The kappa term does not move the boundaries
(the gap it opens vanishes exactly where the Dirac points merge): B is
gapless at kappa = 0 and a gapped phase with Chern number sign(kappa) otherwise.
"""

from __future__ import annotations

import math

import pygame

import mathtext as mt
from ui import BOND_COL, CARD, INK, INK_SOFT, WHITE, aa_circle, blend, draw_stroke, smooth_polygon

ISOTROPIC = (1 / 3, 1 / 3, 1 / 3)
J_MIN = 0.004


class Picker:
    def __init__(self) -> None:
        self.J = ISOTROPIC
        self.kappa = 0.0          # only its sign matters here: the phase labels
        self.dragging = False
        self.rect = pygame.Rect(0, 0, 1, 1)
        self.verts: dict[str, tuple[float, float]] = {}
        self._bg: pygame.Surface | None = None
        self._bg_key = None

    def layout(self, rect: pygame.Rect) -> None:
        """Fit the triangle (and its corner labels) into ``rect``."""
        self.rect = rect
        side = min(rect.w - 90, (rect.h - 44) * 2 / math.sqrt(3))
        h = side * math.sqrt(3) / 2
        cx = rect.centerx
        top = rect.y + 22 + (rect.h - 44 - h) / 2
        self.verts = {"z": (cx, top), "x": (cx - side / 2, top + h), "y": (cx + side / 2, top + h)}

    def point(self) -> tuple[float, float]:
        return (sum(j * self.verts[k][0] for j, k in zip(self.J, "xyz")),
                sum(j * self.verts[k][1] for j, k in zip(self.J, "xyz")))

    def barycentric(self, p) -> tuple[float, float, float]:
        (x1, y1), (x2, y2), (x3, y3) = self.verts["x"], self.verts["y"], self.verts["z"]
        det = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
        a = ((y2 - y3) * (p[0] - x3) + (x3 - x2) * (p[1] - y3)) / det
        b = ((y3 - y1) * (p[0] - x3) + (x1 - x3) * (p[1] - y3)) / det
        return a, b, 1 - a - b

    def set_from(self, p) -> None:
        lam = [max(J_MIN, v) for v in self.barycentric(p)]
        s = sum(lam)
        self.J = tuple(v / s for v in lam)

    def inside(self, p, slack: float = 0.06) -> bool:
        return min(self.barycentric(p)) > -slack

    def handle(self, event) -> bool:
        """Returns True when J changed."""
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.inside(event.pos) or math.dist(event.pos, self.point()) < 14:
                self.dragging = True
                self.set_from(event.pos)
                return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self.set_from(event.pos)
            return True
        return False

    # ------------------------------------------------------------------
    def draw(self, surf: pygame.Surface, mouse) -> None:
        key = (self.rect.topleft, self.rect.size, (self.kappa > 0) - (self.kappa < 0))
        if key != self._bg_key:
            self._bg_key = key
            self._bg = self._background()
        surf.blit(self._bg, self.rect.topleft)
        V = self.verts
        p = self.point()
        for k in "xyz":            # the couplings as distances to the opposite edges
            a, b = (V[o] for o in "xyz" if o != k)
            foot = _foot(p, a, b)
            draw_stroke(surf, [p, foot], 2.0, blend(BOND_COL[k], CARD, 0.15))
            aa_circle(surf, BOND_COL[k], foot, 3)
        if self.dragging or math.dist(mouse, p) < 14:
            aa_circle(surf, (255, 226, 150), p, 13)
        aa_circle(surf, WHITE, p, 9)
        aa_circle(surf, INK, p, 7)

    def _background(self) -> pygame.Surface:
        ox, oy = self.rect.topleft
        surf = pygame.Surface(self.rect.size, pygame.SRCALPHA)
        V = {k: (x - ox, y - oy) for k, (x, y) in self.verts.items()}
        mid = {"x": _mid(V["y"], V["z"]), "y": _mid(V["x"], V["z"]), "z": _mid(V["x"], V["y"])}
        smooth_polygon(surf, (236, 238, 243), [mid["x"], mid["y"], mid["z"]])
        for k in "xyz":
            o1, o2 = (o for o in "xyz" if o != k)
            smooth_polygon(surf, blend(BOND_COL[k], CARD, 0.84), [V[k], mid[o1], mid[o2]])
        draw_stroke(surf, [mid["x"], mid["y"], mid["z"]], 1.0, (205, 208, 216), closed=True)
        draw_stroke(surf, [V["x"], V["y"], V["z"]], 1.4, (190, 194, 202), closed=True)
        cen = _centroid(V["x"], V["y"], V["z"])
        # under the top edge of the B region (the centre is where the point sits
        # by default); main.py writes out the phase and its Chern number
        lab = mt.formula("B", 12, INK_SOFT)
        lab.blit(surf, cen[0] - lab.w / 2, mid["x"][1] + 3 + lab.base)
        for k in "xyz":            # the abelian corners, Chern number 0
            o1, o2 = (o for o in "xyz" if o != k)
            cx, cy = _centroid(V[k], mid[o1], mid[o2])
            lab = mt.formula(f"A_{k}", 13, blend(BOND_COL[k], INK, 0.3))
            lab.blit(surf, cx - lab.w / 2, cy + lab.base - lab.h / 2)
        for k, (dx, dy) in (("z", (0, -12)), ("x", (-6, 13)), ("y", (6, 13))):
            lab = mt.formula(f"\\c{{{k}}}{{J_{k}}} = 1", 14)
            x, y = V[k][0] + dx, V[k][1] + dy
            if k == "x":
                x -= lab.w
            elif k == "z":
                x -= lab.w / 2
            lab.blit(surf, x, y + lab.base - lab.h / 2)
        return surf


def phase_name(J) -> str:
    """'B' inside the triangle inequalities (boundaries included), else 'A_x', ..."""
    for k, a, b, c in (("x", *J), ("y", J[1], J[2], J[0]), ("z", J[2], J[0], J[1])):
        if a > b + c:
            return f"A_{k}"
    return "B"


def _mid(a, b):
    return (a[0] + b[0]) / 2, (a[1] + b[1]) / 2


def _centroid(a, b, c):
    return (a[0] + b[0] + c[0]) / 3, (a[1] + b[1] + c[1]) / 3


def _foot(p, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy)
    return a[0] + t * dx, a[1] + t * dy
