"""Both Majorana bands +-eps(q) of the vortex-free sector as 3D surfaces.

The bands include the kappa term: eps(q) = sqrt(|f(q)|^2 + Delta(q)^2), computed from the Bloch
matrix in majorana.py. The floor is the cell spanned by q1, q2 (Kitaev 2006,
Eq. 34); drag to orbit, wheel to zoom, right-click to reset.
"""

from __future__ import annotations

import cmath
import math

import pygame

import majorana as mj
import mathtext as mt
from majorana import np
from ui import CARD, INK, INK_FAINT, INK_SOFT, LUT, WHITE, text_at

SQ3 = math.sqrt(3.0)
Q1 = (2 * math.pi / SQ3, 2 * math.pi / 3)
Q2 = (-2 * math.pi / SQ3, 2 * math.pi / 3)
CELL_HALF_W = 2 * math.pi / SQ3
CELL_H = 4 * math.pi / 3

MESH_N = 26          # subdivisions of each side of the cell
Z_SCALE = 0.42       # world height per unit of energy
SS3 = 2
BAND_ALPHA = 178
LIGHT = (0.35, -0.45, 0.82)


def from_reduced(s: float, t: float) -> tuple[float, float]:
    return s * Q1[0] + t * Q2[0], s * Q1[1] + t * Q2[1]



def world_xy(s: float, t: float) -> tuple[float, float]:
    """Reduced momentum -> 3D floor coordinates: the cell centred, half-width 1."""
    qx, qy = from_reduced(s, t)
    return qx / CELL_HALF_W, (qy - CELL_H / 2) / CELL_HALF_W


def touching_points(J, kappa: float) -> list[tuple[float, float]]:
    """Reduced momenta where the bands touch: the zeros of f(q) when kappa = 0
    (two Dirac points in phase B, one merged point on a boundary); on a boundary
    with kappa != 0 the merged point only."""
    if not mj.bands_touch(J, kappa):
        return []
    jx, jy, jz = J
    if min(J) <= 1e-9:
        return []
    cos_a = max(-1.0, min(1.0, (jy * jy - jx * jx - jz * jz) / (2 * jx * jz)))
    out = []
    for sign in (1, -1):
        a = sign * math.acos(cos_a)
        b = cmath.phase(-(jz + jx * cmath.exp(1j * a)) / jy)
        p = ((a / (2 * math.pi)) % 1.0, (b / (2 * math.pi)) % 1.0)
        if all(math.dist(p, o) > 1e-6 for o in out):
            out.append(p)
    return out


def _images(p, eps: float = 1e-6):
    """Copies of a reduced momentum on the closed cell [0, 1]^2."""
    s, t = p
    ss = [s] + ([s + 1] if s < eps else []) + ([s - 1] if s > 1 - eps else [])
    ts = [t] + ([t + 1] if t < eps else []) + ([t - 1] if t > 1 - eps else [])
    return [(a, b) for a in ss for b in ts]


class Band3D:
    def __init__(self, fonts) -> None:
        self.fonts = fonts
        self.reset_view()
        self.dragging = False
        self.last_mouse = (0, 0)
        self.focal = 900.0
        self._cache: pygame.Surface | None = None
        self._key = None
        self._ss = 0
        n = MESH_N
        g = np.arange(n + 1) / n
        self.S1, self.S2 = np.meshgrid(g, g, indexing="ij")
        self.verts = [world_xy(i / n, j / n) for i in range(n + 1) for j in range(n + 1)]
        self.tris = []
        for i in range(n):
            for j in range(n):
                v = i * (n + 1) + j
                self.tris.append((v, v + n + 1, v + n + 2))
                self.tris.append((v, v + n + 2, v + 1))

    def reset_view(self) -> None:
        self.az, self.el, self.dist = -0.3, 0.42, 2.9

    def handle(self, event, rect: pygame.Rect) -> bool:
        """Orbit / zoom; returns True when the event was used."""
        if event.type == pygame.MOUSEBUTTONDOWN and rect.collidepoint(event.pos):
            if event.button == 1:
                self.dragging = True
                self.last_mouse = event.pos
                return True
            if event.button == 3:
                self.reset_view()
                return True
            if event.button in (4, 5):
                return True     # handled by MOUSEWHEEL
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.dragging:
            self.dragging = False
            return True
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            dx = event.pos[0] - self.last_mouse[0]
            dy = event.pos[1] - self.last_mouse[1]
            self.last_mouse = event.pos
            self.az -= dx * 0.009
            self.el = max(-1.5, min(1.5, self.el + dy * 0.009))
            return True
        elif event.type == pygame.MOUSEWHEEL and rect.collidepoint(pygame.mouse.get_pos()):
            self.dist = max(1.6, min(12.0, self.dist * (0.9 ** event.y)))
            return True
        return False

    def project(self, p, size) -> tuple[float, float, float]:
        x, y, z = p
        ca, sa = math.cos(self.az), math.sin(self.az)
        xr = x * ca - y * sa
        yr = x * sa + y * ca
        ce, se = math.cos(self.el), math.sin(self.el)
        yt = yr * ce - z * se
        zt = yr * se + z * ce
        depth = max(0.3, yt + self.dist)
        f = self.focal * min(size) / 900 / depth
        return size[0] / 2 + xr * f, size[1] * 0.5 - zt * f, depth

    def draw(self, surf: pygame.Surface, rect: pygame.Rect, J, kappa: float, busy: bool = False) -> None:
        """``busy``: something is being dragged, so render quickly at 1x and
        redo it supersampled once things settle."""
        key = (tuple(round(j, 5) for j in J), round(kappa, 5), round(self.az, 4),
               round(self.el, 4), round(self.dist, 4), rect.size)
        ss = 1 if (busy or self.dragging) else SS3
        if key != self._key or self._cache is None or ss > self._ss:
            self._key, self._ss = key, ss
            self._cache = self._render(rect.size, J, kappa, ss)
        surf.blit(self._cache, rect)

    def _render(self, size, J, kappa: float, SS3: int) -> pygame.Surface:
        W, H = size[0] * SS3, size[1] * SS3
        big = pygame.Surface((W, H))
        big.fill(CARD)
        energies = [float(e) for e in mj.band_energy(J, kappa, self.S1, self.S2).ravel()]
        e_max = max(2.0, max(energies))
        labels: list[tuple[str, tuple[float, float]]] = []

        def proj(p):
            return self.project(p, (W, H))

        # energy axis, behind everything, beside the cell on the far side
        ang = math.radians(210) if math.cos(self.az) > 0 else math.radians(30)
        axis_xy = (1.25 * math.cos(ang - self.az), 1.25 * math.sin(ang - self.az))
        top = proj((*axis_xy, e_max * Z_SCALE))
        bot = proj((*axis_xy, -e_max * Z_SCALE))
        pygame.draw.line(big, INK_FAINT, top[:2], bot[:2], SS3)
        for v in range(-int(e_max), int(e_max) + 1):
            p = proj((*axis_xy, v * Z_SCALE))
            pygame.draw.line(big, INK_FAINT, (p[0] - 5 * SS3, p[1]), (p[0], p[1]), SS3)
            labels.append((f"{v:+d}" if v else "0", (p[0] - 8 * SS3, p[1])))
        labels.append(("ε", (top[0], top[1] - 12 * SS3)))

        upper = self._band_layer(big.get_size(), energies, e_max, +1, proj)
        lower = self._band_layer(big.get_size(), energies, e_max, -1, proj)

        # the cell on the floor E = 0, with its diagonal and the touching points
        plane = pygame.Surface(big.get_size(), pygame.SRCALPHA)
        o, a, c, b = (proj((*world_xy(*st), 0))[:2] for st in ((0, 0), (1, 0), (1, 1), (0, 1)))
        pygame.draw.lines(plane, (120, 126, 138, 200), True, [o, a, c, b], SS3)
        pygame.draw.line(plane, (120, 126, 138, 130), o, c, SS3)
        labels.append(("q_1", a))
        labels.append(("q_2", b))
        for pt in touching_points(J, kappa):
            for st in _images(pt):
                p = proj((*world_xy(*st), 0))
                pygame.draw.circle(plane, INK, p[:2], 5.5 * SS3)
                pygame.draw.circle(plane, WHITE, p[:2], 3.5 * SS3)

        layers = [lower, plane, upper] if self.el >= 0 else [upper, plane, lower]
        for layer in layers:
            big.blit(layer, (0, 0))

        img = pygame.transform.smoothscale(big, size)
        f = self.fonts.small
        for s, (x, y) in labels:
            if s == "ε":
                mt.formula("ε", 16, INK_SOFT).blit(img, x / SS3 - 4, y / SS3 + 6)
            elif s.startswith("q"):
                lab = mt.formula(s, 15, INK_SOFT)
                side = -1 if x < W / 2 else 1
                lab.blit(img, x / SS3 + side * (lab.w / 2 + 12) - lab.w / 2,
                         y / SS3 + lab.base / 2)
            else:
                text_at(img, f, s, INK_FAINT, (x / SS3, y / SS3), "midright")
        return img

    def _band_layer(self, size, energies, e_max: float, sign: int, proj) -> pygame.Surface:
        world = [(x, y, sign * e * Z_SCALE) for (x, y), e in zip(self.verts, energies)]
        pts = [proj(w) for w in world]
        lx, ly, lz = LIGHT
        ln = math.sqrt(lx * lx + ly * ly + lz * lz)
        lx, ly, lz = lx / ln, ly / ln, lz / ln
        faces = []
        for a, b, c in self.tris:
            pa, pb, pc = pts[a], pts[b], pts[c]
            depth = pa[2] + pb[2] + pc[2]
            wa, wb, wc = world[a], world[b], world[c]
            ux, uy, uz = wb[0] - wa[0], wb[1] - wa[1], wb[2] - wa[2]
            vx, vy, vz = wc[0] - wa[0], wc[1] - wa[1], wc[2] - wa[2]
            nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
            nn = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
            shade = 0.58 + 0.42 * abs(nx * lx + ny * ly + nz * lz) / nn
            e = (energies[a] + energies[b] + energies[c]) / 3
            col = LUT[min(255, int(e / e_max * 255))]
            col = (int(col[0] * shade), int(col[1] * shade), int(col[2] * shade))
            faces.append((depth, (pa[:2], pb[:2], pc[:2]), col))
        faces.sort(key=lambda t: -t[0])
        solid = pygame.Surface(size, pygame.SRCALPHA)
        for _, poly, col in faces:
            pygame.draw.polygon(solid, col, poly)
        solid.fill((255, 255, 255, BAND_ALPHA), special_flags=pygame.BLEND_RGBA_MULT)
        return solid
