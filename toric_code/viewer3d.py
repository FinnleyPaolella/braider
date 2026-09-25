"""The 3D worldline view, drawn as a panel inside the main window.

The lattice lies in the horizontal plane and time runs upward, so a particle
sitting still is a vertical line and a hop is a kink. A pair creation and a
pair annihilation each draw a hairpin turn joining the two partners, so a pair
that is created, moved and annihilated closes into a single loop.

Rendering is a plain painter's algorithm: every primitive is projected, tagged
with its camera depth, sorted back to front and drawn. The result is cached and
repainted only when the scene or the camera actually changes.

This lives in the main window rather than a second one so that input is routed
by cursor position: the grid stays fully interactive while the panel is open.
A second OS window would have to route clicks by window id, and pygame does not
reliably attribute mouse events to a window, so the grid could become
unreachable whenever the other window held focus.
"""

from __future__ import annotations

import math

import pygame

from ui import (
    BG, E_COL, GRID, HAIRLINE, INK, INK_FAINT, INK_SOFT, M_COL, WHITE,
    aa_circle, blend, text_at,
)

# a hop is drawn as a short slanted piece at the end of the waiting line,
# so the kink reads as a move rather than a long diagonal drift
HOP_FRACTION = 0.55
# how far the turn at a pair creation / annihilation bulges past the event,
# in time units, so the two worldlines join into one closed curve
LINK_BULGE = 0.4


class WorldlinePanel:
    """Draws the worldlines into a rectangle of a surface."""

    def __init__(self, lat, tracker, fonts) -> None:
        self.lat = lat
        self.tracker = tracker
        self.fonts = fonts

        self.az = -0.65      # azimuth, radians
        self.el = 0.42       # elevation, radians
        self.dist = 26.0
        self.focal = 860.0
        self.z_off = 0.0      # keeps the growing time tower centred
        self.auto_fit = True  # frame the scene until the user zooms themselves
        self.dragging = False
        self.last_mouse = (0, 0)
        self.show_floor = True

        self._cache: pygame.Surface | None = None
        self._last_key = None
        self._size = (0, 0)

    # ------------------------------------------------------------------
    # camera
    # ------------------------------------------------------------------
    def reset_view(self) -> None:
        self.az, self.el, self.dist = -0.65, 0.42, 26.0
        self.auto_fit = True

    def _fit_distance(self, tower: float) -> float:
        """Camera distance that frames the whole lattice-and-time box."""
        L = self.lat.L
        radius = math.sqrt((L / 2) ** 2 + (L / 2) ** 2 + (tower / 2) ** 2)
        # the bounding sphere overstates the visible extent, so fill generously
        span = 0.46 * min(self._size) if min(self._size) else 300.0
        return max(6.0, min(120.0, radius * self.focal / max(1.0, span)))

    def project(self, p) -> tuple[float, float, float]:
        """World (x, y, z) -> (panel x, panel y, depth)."""
        x, y, z = p
        ca, sa = math.cos(self.az), math.sin(self.az)
        xr = x * ca - y * sa
        yr = x * sa + y * ca
        ce, se = math.cos(self.el), math.sin(self.el)
        yt = yr * ce - z * se
        zt = yr * se + z * ce
        depth = yt + self.dist
        if depth < 0.6:
            depth = 0.6
        f = self.focal / depth
        w, h = self._size
        return w / 2 + xr * f, h / 2 - zt * f, depth

    # ------------------------------------------------------------------
    # events (positions are in window coordinates)
    # ------------------------------------------------------------------
    def handle(self, event) -> None:
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button in (4, 5):
                self.auto_fit = False
                self.dist = max(6.0, min(
                    120.0, self.dist * (0.9 if event.button == 4 else 1.1)))
            elif event.button == 1:
                self.dragging = True
                self.last_mouse = event.pos
            elif event.button == 3:
                self.reset_view()
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            dx = event.pos[0] - self.last_mouse[0]
            dy = event.pos[1] - self.last_mouse[1]
            self.last_mouse = event.pos
            self.az -= dx * 0.008
            self.el = max(-1.45, min(1.45, self.el + dy * 0.008))
        elif event.type == pygame.MOUSEWHEEL:
            self.auto_fit = False
            self.dist = max(6.0, min(120.0, self.dist * (0.9 ** event.y)))

    # ------------------------------------------------------------------
    # geometry
    # ------------------------------------------------------------------
    def _z_scale(self, t_max: int) -> float:
        return min(0.95, 13.0 / max(1, t_max))

    def _world(self, xy, t, zs) -> tuple[float, float, float]:
        L = self.lat.L
        return xy[0] - L / 2, xy[1] - L / 2, t * zs - self.z_off

    def _hop_pieces(self, kind, site_a, site_b):
        """The hop a->b, split in two when it crosses the periodic seam.

        A wrapped hop leaves the box on one side and re-enters on the other.
        """
        L = self.lat.L
        ax, ay = self.tracker.site_xy(kind, site_a)
        bx, by = self.tracker.site_xy(kind, site_b)
        dx, dy = bx - ax, by - ay
        if dx > L / 2:
            dx -= L
        elif dx < -L / 2:
            dx += L
        if dy > L / 2:
            dy -= L
        elif dy < -L / 2:
            dy += L
        out = (ax + dx, ay + dy)
        if abs(out[0] - bx) < 1e-9 and abs(out[1] - by) < 1e-9:
            return [((ax, ay), (bx, by))]
        return [((ax, ay), out), ((bx - dx, by - dy), (bx, by))]

    def link_curves(self, link, zs: float):
        """The turn joining a pair, as one polyline (two when it wraps)."""
        sign = -1.0 if link.creation else 1.0
        curves = []
        for a_xy, b_xy in self._hop_pieces(link.kind, link.a, link.b):
            pts = []
            steps = 12
            for i in range(steps + 1):
                u = i / steps
                x = a_xy[0] + (b_xy[0] - a_xy[0]) * u
                y = a_xy[1] + (b_xy[1] - a_xy[1]) * u
                z = link.t + sign * LINK_BULGE * math.sin(math.pi * u)
                pts.append(self._world((x, y), z, zs))
            curves.append(pts)
        return curves

    def polyline(self, part, t_now: int, zs: float):
        """A particle's worldline as a list of 3D segments."""
        segs = []
        path = part.path
        for i, (t, site) in enumerate(path):
            nxt = path[i + 1] if i + 1 < len(path) else None
            end_t = nxt[0] if nxt else (part.death if part.death is not None else t_now)
            rise_to = end_t - HOP_FRACTION if nxt is not None else end_t
            xy = self.tracker.site_xy(part.kind, site)
            if rise_to > t + 1e-9:
                segs.append((self._world(xy, t, zs), self._world(xy, rise_to, zs)))
            if nxt is not None:
                for a_xy, b_xy in self._hop_pieces(part.kind, site, nxt[1]):
                    segs.append((
                        self._world(a_xy, max(t, rise_to), zs),
                        self._world(b_xy, end_t, zs),
                    ))
        return segs

    # ------------------------------------------------------------------
    # drawing
    # ------------------------------------------------------------------
    def _state_key(self):
        return (
            getattr(self.tracker, "version", 0), self.tracker.t,
            round(self.az, 4), round(self.el, 4), round(self.dist, 3),
            self._size, self.show_floor, self.auto_fit,
        )

    def draw(self, target: pygame.Surface, rect: pygame.Rect,
             force: bool = False) -> None:
        """Render into ``rect`` of ``target``, reusing the cached image."""
        size = (rect.width, rect.height)
        if size != self._size or self._cache is None:
            self._size = size
            self._cache = pygame.Surface(size)
            force = True
        key = self._state_key()
        if force or key != self._last_key:
            self._last_key = key
            self._render(self._cache)
        target.blit(self._cache, rect.topleft)

    def _render(self, surf: pygame.Surface) -> None:
        surf.fill(BG)
        tr = self.tracker
        t_now = tr.t
        zs = self._z_scale(t_now)
        # look at the middle of the tower, not at its base
        self.z_off = t_now * zs / 2
        if self.auto_fit:
            # ease toward the framing distance so growth does not jump
            target = self._fit_distance(t_now * zs)
            self.dist += (target - self.dist) * 0.25
        L = self.lat.L
        prims: list[tuple[float, object]] = []

        def line(p, q, col, width=2, aa=True):
            sx1, sy1, d1 = self.project(p)
            sx2, sy2, d2 = self.project(q)
            if d1 <= 0.6 and d2 <= 0.6:
                return
            prims.append(((d1 + d2) / 2,
                          ("line", (sx1, sy1), (sx2, sy2), col, width, aa)))

        def dot(p, col, r):
            sx, sy, d = self.project(p)
            scale = self.focal / d / 60.0
            prims.append((d, ("dot", (sx, sy), col, max(2.5, r * scale))))

        # --- floor: the lattice plane at t = 0 ---
        if self.show_floor:
            for i in range(L + 1):
                edge = i in (0, L)
                col = GRID if edge else blend(GRID, BG, 0.45)
                line(self._world((i, 0), 0, zs), self._world((i, L), 0, zs),
                     col, 2 if edge else 1)
                line(self._world((0, i), 0, zs), self._world((L, i), 0, zs),
                     col, 2 if edge else 1)

        # --- the "now" cap ---
        if t_now > 0:
            cap = [self._world((0, 0), t_now, zs), self._world((L, 0), t_now, zs),
                   self._world((L, L), t_now, zs), self._world((0, L), t_now, zs)]
            for i in range(4):
                line(cap[i], cap[(i + 1) % 4], blend(INK_FAINT, BG, 0.45), 1)

        # --- time axis with tick marks ---
        axis_xy = (-0.55, -0.55)
        line(self._world(axis_xy, 0, zs), self._world(axis_xy, max(t_now, 1), zs),
             blend(INK_FAINT, BG, 0.3), 1)
        tick_every = max(1, int(math.ceil(max(t_now, 1) / 8)))
        for t in range(0, t_now + 1, tick_every):
            p = self._world(axis_xy, t, zs)
            q = self._world((axis_xy[0] - 0.25, axis_xy[1] - 0.25), t, zs)
            line(p, q, blend(INK_FAINT, BG, 0.3), 1)
            sx, sy, d = self.project(q)
            prims.append((d, ("label", (sx - 6, sy), str(t))))

        # --- worldlines ---
        for part in tr.particles.values():
            col = E_COL if part.kind == "e" else M_COL
            faded = col if part.alive else blend(col, BG, 0.42)
            for a, b in self.polyline(part, t_now, zs):
                line(a, b, faded, 4 if part.alive else 3)
            if part.alive:
                lxy = tr.site_xy(part.kind, part.path[-1][1])
                dot(self._world(lxy, t_now, zs), col, 6.0)

        # --- the turns closing a pair off at birth and at annihilation ---
        for link in tr.links:
            col = E_COL if link.kind == "e" else M_COL
            # match the worldlines the turn joins: an annihilation always ends
            # two of them, a creation stays vivid while either partner lives
            live = any(
                pid in tr.particles and tr.particles[pid].alive
                for pid in (link.pa, link.pb)
            )
            vivid = link.creation and live
            shade = col if vivid else blend(col, BG, 0.42)
            for pts in self.link_curves(link, zs):
                for i in range(len(pts) - 1):
                    line(pts[i], pts[i + 1], shade, 4 if vivid else 3)

        prims.sort(key=lambda it: -it[0])
        for _, prim in prims:
            what = prim[0]
            if what == "line":
                _, a, b, col, width, aa = prim
                self._line(surf, a, b, col, width, aa)
            elif what == "dot":
                _, c, col, r = prim
                aa_circle(surf, col, c, r)
            else:
                _, pos, s = prim
                text_at(surf, self.fonts.tiny, s, INK_FAINT, pos, "midright")

        self._hud(surf)

    @staticmethod
    def _line(surf, a, b, col, width, aa) -> None:
        w, h = surf.get_size()
        # cheap reject for anything far outside the panel
        if max(a[0], b[0]) < -2000 or min(a[0], b[0]) > w + 2000:
            return
        if max(a[1], b[1]) < -2000 or min(a[1], b[1]) > h + 2000:
            return
        try:
            if width <= 1:
                pygame.draw.aaline(surf, col, a, b)
            else:
                pygame.draw.line(surf, col, a, b, int(width))
                if aa:
                    pygame.draw.aaline(surf, col, a, b)
        except (TypeError, ValueError):
            pass

    def _hud(self, surf: pygame.Surface) -> None:
        w, h = self._size
        tr = self.tracker
        n_e, n_m, total = tr.counts()

        pygame.draw.line(surf, HAIRLINE, (0, 0), (0, h), 1)
        text_at(surf, self.fonts.tiny, "WORLDLINES", INK_FAINT, (24, 22))
        text_at(surf, self.fonts.label_bold, f"{tr.t} moves", INK, (24, 40))
        text_at(surf, self.fonts.small, f"{total} particles", INK_SOFT, (24, 62))

        x = 152
        for name, live, col in (("e", n_e, E_COL), ("m", n_m, M_COL)):
            aa_circle(surf, col, (x, 49), 9)
            text_at(surf, self.fonts.gate_small, name, WHITE, (x, 48), "center")
            text_at(surf, self.fonts.label_bold, str(live), INK, (x + 16, 40))
            x += 54

        if tr.is_empty():
            text_at(surf, self.fonts.label, "No moves yet.", INK_FAINT,
                    (w / 2, h / 2 - 10), "center")
            text_at(surf, self.fonts.small, "Add a gate or drag an anyon.",
                    INK_FAINT, (w / 2, h / 2 + 12), "center")

        hint = "drag orbit  .  wheel zoom  .  right-click reset  .  f floor"
        text_at(surf, self.fonts.small, hint, INK_FAINT, (w / 2, h - 22), "center")
