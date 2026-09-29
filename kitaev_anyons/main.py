# /// script
# dependencies = [
#     "numpy",
# ]
# ///

"""Majorana fermions in a fixed Z2 gauge field on a finite honeycomb cluster.

Run with:  python main.py

Left: the c Majoranas on an L1 x L2 cluster. Click a link to flip its u_jk;
the hexagons show w_p = prod u_jk. For every configuration the matrix A of
H = (i/4) sum A_jk c_j c_k (Kitaev 2006, Eq. 48: nearest and next-nearest
neighbours) is built and diagonalised.
Right: the Hamiltonian with the couplings of a unit cell and the J triangle;
the lattice size and boundary conditions; the bands of the vortex-free sector
with the Chern number of the lower band; the single-particle energies (click
one to see its mode on the lattice). The "?" buttons explain the details.

Keys: 0 all u = +1, i isotropic couplings, Esc close a popup / quit.
"""

from __future__ import annotations

import asyncio
import os
import sys

# pygame imports numpy, so MKL's threading layer must be chosen before pygame
# loads (see majorana.py).
os.environ.setdefault("MKL_THREADING_LAYER", "TBB")

import pygame  # noqa: E402

import help_texts  # noqa: E402
import majorana as mj  # noqa: E402
import mathtext as mt  # noqa: E402
from bands import Band3D  # noqa: E402
from cell_diagram import KAPPA_COL, CellDiagram  # noqa: E402
from help import HelpButton  # noqa: E402
from lattice_view import FLUX_FILL, FLUX_INK, MODE_COL, LatticeView  # noqa: E402
import webcanvas  # noqa: E402
from picker import ISOTROPIC, Picker, phase_name  # noqa: E402
from ui import (  # noqa: E402
    BG, BOND_COL, HAIRLINE, INK, INK_SOFT, UI_STACK, Fonts, aa_circle, arrow,
    caps, card, sysfont, text_at,
)
from widgets import Button, Segmented, Slider, Stepper  # noqa: E402

M = 16              # margin between cards
PAD = 20            # inside a card
MIN_W, MIN_H = 1500, 900
KAPPA_MAX = 0.2
SEAM_NAMES = ("seam 1 (along n₁)", "seam 2 (along n₂)")
TOUCH_COL = (150, 96, 206)
BC_CARD_H = 222
TRI_H = 214
HOVER_R = 10        # px: how close the mouse must be to a dot in the spectrum


def _enable_dpi_awareness() -> None:
    try:
        import ctypes

        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except (AttributeError, OSError):
            ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


class App:
    def __init__(self) -> None:
        _enable_dpi_awareness()
        pygame.init()
        pygame.display.set_caption("Kitaev model: Majoranas in a gauge field")
        info = pygame.display.Info()
        self.W = min(1880, max(MIN_W, info.current_w - 60))
        self.H = min(1060, max(MIN_H, info.current_h - 110))
        page = webcanvas.viewport()
        if page:
            self.W, self.H = max(MIN_W, page[0]), max(MIN_H, page[1])
        self.flags = 0 if page else pygame.RESIZABLE
        self.screen = pygame.display.set_mode((self.W, self.H), self.flags)
        webcanvas.fit((self.W, self.H), BG)
        self.clock = pygame.time.Clock()
        self.fonts = Fonts()
        self.mouse = (0, 0)

        self.picker = Picker()
        self.kappa = Slider(-KAPPA_MAX, KAPPA_MAX, 0.05, snap=0.004)
        self.iso_button = Button("isotropic")
        self.kappa0_button = Button("κ = 0")
        self.L1 = Stepper(8, 2, 12)
        self.L2 = Stepper(8, 2, 12)
        self.seams = [Segmented(list(mj.BCS), "periodic") for _ in range(2)]
        self.reset_button = Button("all u = +1")
        self.view = LatticeView(self.fonts)
        self.diagram = CellDiagram()
        self.band3d = Band3D(self.fonts)
        self.helps = {
            "ham": HelpButton("The Hamiltonian", help_texts.HAMILTONIAN),
            "bands": HelpButton("Bands and Chern number", help_texts.BANDS),
            "bc": HelpButton("Boundary conditions", help_texts.BOUNDARY),
        }
        self._band_key = None
        self.hover_dot: int | None = None

        self.new_lattice()
        self.layout()

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------
    @property
    def bc(self) -> tuple[str, str]:
        return self.seams[0].value, self.seams[1].value

    def new_lattice(self) -> None:
        self.lat = mj.Lattice(self.L1.value, self.L2.value)
        self.u = [1] * len(self.lat.bonds)
        self.dirty = True
        self.selected: int | None = None     # the mode shown on the lattice
        self.spec_dots: list[tuple[float, float]] = []

    def recompute(self) -> None:
        J, kappa = self.picker.J, self.kappa.value
        if self.dirty:
            self.result = mj.solve(self.lat, self.u, J, kappa, self.bc)
            self.solve_count = getattr(self, "solve_count", 0) + 1
            self.fluxes = self.lat.fluxes(self.u, self.bc)
            self.dirty = False
        if (J, kappa) != self._band_key:
            self._band_key = (J, kappa)
            self.touch = mj.bands_touch(J, kappa)
            self.chern = None if self.touch else mj.chern_lower(J, kappa)

    def changed(self, lattice: bool = False) -> None:
        self.dirty = True
        if lattice:
            self.layout()

    # ------------------------------------------------------------------
    # layout
    # ------------------------------------------------------------------
    def layout(self) -> None:
        W, H = self.W, self.H
        LW = round(W * 0.37)
        self.lattice_card = pygame.Rect(M, M, LW, H - 2 * M)
        inner = self.lattice_card.inflate(-2 * PAD, -2 * PAD)
        inner.y += 66
        inner.h -= 66 + 30
        self.view.layout(inner, self.lat)
        size = max(10, min(19, round(self.view.scale * 0.36)))
        self.fonts.plaquette = sysfont(UI_STACK, size)
        self.fonts.plaquette_bold = sysfont("segoeuisemibold," + UI_STACK, size)

        rx = M + LW + M
        cw = (W - rx - M - M) // 2
        x1, x2 = rx, rx + cw + M

        # column 1: Hamiltonian (with the unit cell and the triangle), lattice
        self.ham_card = c = pygame.Rect(x1, M, cw, H - 3 * M - BC_CARD_H)
        self.helps["ham"].place((c.right - PAD + 4, c.y + PAD - 6))
        self.formula_y = c.y + PAD + 34
        kappa_top = c.bottom - PAD - 64
        tri = pygame.Rect(c.x + PAD - 12, kappa_top - TRI_H - 6, 290, TRI_H)
        self.picker.layout(tri)
        self.j_bars_x = tri.right + 14
        self.iso_button.fit(self.fonts, self.j_bars_x, tri.y + 118)
        diag_top = self.formula_y + 84
        diag_h = tri.y - 10 - diag_top
        self.diagram_rect = pygame.Rect(c.x + PAD, diag_top, round((cw - 2 * PAD) * 0.52), diag_h)
        self.legend_x = self.diagram_rect.right + 14
        self.kappa_label_y = kappa_top
        self.kappa0_button.fit(self.fonts, c.right - PAD - 70, kappa_top + 12)
        self.kappa.fit(c.x + PAD + 150, kappa_top + 19, self.kappa0_button.rect.x - 24 - (c.x + PAD + 150))

        self.bc_card = c = pygame.Rect(x1, self.ham_card.bottom + M, cw, BC_CARD_H)
        self.helps["bc"].place((c.right - PAD + 4, c.y + PAD - 6))
        row = c.y + PAD + 36
        self.L1.fit(c.x + PAD + 36, row)
        self.L2.fit(c.x + PAD + 36 + 150, row)
        self.reset_button.fit(self.fonts, c.right - PAD - 118, row - 1)
        self.seam_rows = []
        for k, seg in enumerate(self.seams):
            ry = row + 50 + 42 * k
            self.seam_rows.append(ry)
            seg.fit(self.fonts, c.right - PAD - 262, ry)

        # column 2: bands, spectrum
        bh = round((H - 3 * M) * 0.52)
        self.bands_card = c = pygame.Rect(x2, M, cw, bh)
        self.helps["bands"].place((c.right - PAD + 4, c.y + PAD - 6))
        self.band_rect = pygame.Rect(c.x + 8, c.y + PAD + 70, cw - 16, c.h - PAD - 70 - 34)
        self.spec_card = pygame.Rect(x2, c.bottom + M, cw, H - 3 * M - bh)

    def fit_page(self, size) -> None:
        self.W, self.H = max(MIN_W, size[0]), max(MIN_H, size[1])
        self.screen = pygame.display.set_mode((self.W, self.H), self.flags)
        webcanvas.fit((self.W, self.H), BG)
        self.layout()

    # ------------------------------------------------------------------
    # events
    # ------------------------------------------------------------------
    def dispatch(self, event) -> bool:
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.MOUSEMOTION:
            self.mouse = event.pos
        # an open popup takes every click (and Esc)
        for h in self.helps.values():
            if h.open and h.handle(event):
                return True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE and not webcanvas.WEB:
                return False
            if event.key == pygame.K_0:
                self.u = [1] * len(self.lat.bonds)
                self.changed()
            if event.key == pygame.K_i:
                self.picker.J = ISOTROPIC
                self.changed()
        if event.type == pygame.VIDEORESIZE and not webcanvas.WEB:
            self.fit_page(event.size)
        for h in self.helps.values():
            if h.handle(event):
                return True

        if self.band3d.handle(event, self.band_rect):
            return True
        if self.picker.handle(event) or self.kappa.handle(event):
            self.changed()
        if self.kappa0_button.clicked(event):
            self.kappa.value = 0.0
            self.changed()
        if self.iso_button.clicked(event):
            self.picker.J = ISOTROPIC
            self.changed()
        if (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1
                and self.hover_dot is not None and self.spec_card.collidepoint(event.pos)):
            # click a dot in the spectrum to show its mode (again to hide it)
            m = self.hover_dot
            self.selected = None if m == self.selected else m
        if self.L1.handle(event) or self.L2.handle(event):
            self.new_lattice()
            self.changed(lattice=True)
        for seg in self.seams:
            if seg.handle(event):
                self.changed(lattice=True)
        if self.reset_button.clicked(event):
            self.u = [1] * len(self.lat.bonds)
            self.changed()
        if (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1
                and self.view.rect.collidepoint(event.pos)):
            b = self.view.bond_at(event.pos)
            if b is not None:
                self.u[b] = -self.u[b]
                self.changed()
        return True

    # ------------------------------------------------------------------
    # drawing
    # ------------------------------------------------------------------
    def draw(self) -> None:
        s = self.screen
        s.fill(BG)
        self.recompute()
        self.picker.kappa = self.kappa.value
        self.hover_dot = self._nearest_dot(self.mouse)
        self._draw_lattice_card(s)
        self._draw_ham_card(s)
        self._draw_bc_card(s)
        self._draw_bands_card(s)
        self._draw_spec_card(s)
        for h in self.helps.values():
            h.draw_popup(s, self.fonts)
        pygame.display.flip()

    def _header(self, s, c: pygame.Rect, title: str, help_key: str | None = None) -> None:
        card(s, c)
        caps(s, self.fonts, title, (c.x + PAD, c.y + PAD - 2))
        if help_key:
            self.helps[help_key].draw_button(s, self.fonts, self.mouse)

    def _draw_lattice_card(self, s) -> None:
        c = self.lattice_card
        self._header(s, c, "Majoranas c in a static gauge field")
        n_v = sum(1 for w in self.fluxes if w == -1)
        text_at(s, self.fonts.ui, f"{self.lat.L1} × {self.lat.L2} cells · {self.lat.N} Majoranas · "
                f"{n_v} vortex{'es' if n_v != 1 else ''}", INK_SOFT, (c.x + PAD, c.y + PAD + 20))
        mode = None
        if self.selected is not None:
            m = self.selected
            amps = [float(a) for a in abs(self.result.modes[:, m])]
            mode = ((self.solve_count, m), amps)
            text_at(s, self.fonts.ui, f"mode m = {m + 1},  ε = {self.result.eps[m]:.5f}  ·  "
                    "circle area proportional to |ψ_j|²", MODE_COL, (c.x + PAD, c.y + PAD + 44))
        self.view.draw(s, self.u, self.bc, self.fluxes, self.mouse, mode)

        # legend along the bottom
        y = c.bottom - PAD - 6
        x = c.x + PAD
        for k in "xyz":
            pygame.draw.line(s, BOND_COL[k], (x, y), (x + 22, y), 3)
            x = text_at(s, self.fonts.small, k, INK_SOFT, (x + 28, y), "midleft").right + 14
        pygame.draw.line(s, INK, (x, y), (x + 22, y), 7)
        pygame.draw.line(s, INK_SOFT, (x, y), (x + 22, y), 3)
        x = text_at(s, self.fonts.small, "u = −1", INK_SOFT, (x + 28, y), "midleft").right + 14
        pygame.draw.rect(s, FLUX_FILL, (x, y - 10, 30, 20), border_radius=4)
        text_at(s, self.fonts.small, "−1", FLUX_INK, (x + 15, y), "center")
        x = text_at(s, self.fonts.small, "vortex", INK_SOFT, (x + 36, y), "midleft").right + 14
        text_at(s, self.fonts.small, "click a link to flip u", INK_SOFT, (c.right - PAD, y), "midright")

    def _draw_ham_card(self, s) -> None:
        c = self.ham_card
        self._header(s, c, "Hamiltonian and couplings", "ham")
        y = self.formula_y
        f = _formula(r"H = \frac{i}{4}\sum{j,k} A_{jk} c_j c_k", 20)
        f.blit(s, c.x + PAD, y + f.base - 6)
        y += f.h + 4
        f = _formula(r"A_{jk} = 2J_{α} u_{jk}~\rm{(links)},~~2κ ε_{jlk} u_{jl} u_{lk}"
                     r"~\rm{(next-nearest)}", 17)
        f.blit(s, c.x + PAD, y + f.base)

        # the unit cell and its legend
        self.diagram.draw(s, self.diagram_rect, self.picker.J, self.kappa.value, KAPPA_MAX)
        x, y = self.legend_x, self.diagram_rect.y + 14
        caps(s, self.fonts, "one unit cell", (x, y))
        y += 30
        for k in "xyz":
            pygame.draw.line(s, BOND_COL[k], (x, y), (x + 26, y), 4)
            f = _formula(f"\\c{{{k}}}{{2J_{k}}} u_{{jk}}", 17)
            f.blit(s, x + 36, y + f.base - f.h / 2)
            y += 30
        arrow(s, (x, y), (x + 26, y), KAPPA_COL, 2.0, 10)
        f = _formula("2κ", 17)
        f.blit(s, x + 36, y + f.base - f.h / 2)
        y += 22
        for line in ("arrow: A_jk = +2κ (u = +1)", "●  A site   ○  B site"):
            text_at(s, self.fonts.small, line, INK_SOFT, (x, y))
            y += 20

        self.picker.draw(s, self.mouse)
        # coupling bars to the right of the triangle
        x, y = self.j_bars_x, self.picker.rect.y + 22
        bar_w = c.right - PAD - x - 66
        for k, j in zip("xyz", self.picker.J):
            lab = _formula(f"\\c{{{k}}}{{J_{k}}}", 17)
            lab.blit(s, x, y + 13)
            track = pygame.Rect(x + 30, y + 4, bar_w, 9)
            pygame.draw.rect(s, (236, 238, 243), track, border_radius=4)
            fill = track.copy()
            fill.w = max(9, round(track.w * j))
            pygame.draw.rect(s, BOND_COL[k], fill, border_radius=4)
            text_at(s, self.fonts.small, f"{j:.3f}", INK, (track.right + 8, y + 8), "midleft")
            y += 30
        self.iso_button.draw(s, self.fonts, self.mouse)
        # the phase: the kappa term gaps B but leaves the boundaries
        name = phase_name(self.picker.J)
        if self.touch:
            desc = "gapless" if name == "B" else "boundary"
        else:
            desc = f"ν = {self.chern:+d}" if self.chern else "ν = 0"
        f = _formula(f"\\rm{{phase}}~{name}", 17)
        y = self.iso_button.rect.bottom + 12
        f.blit(s, self.j_bars_x, y + f.base)
        text_at(s, self.fonts.ui, f"·  {desc}", TOUCH_COL if self.touch else INK_SOFT,
                (self.j_bars_x + f.w + 8, y + f.h / 2), "midleft")

        # kappa
        k = self.kappa.value
        f = _formula(f"κ = {k:.3f}", 18)
        f.blit(s, c.x + PAD, self.kappa_label_y + 34)
        self.kappa.draw(s, self.mouse)
        self.kappa0_button.draw(s, self.fonts, self.mouse)

    def _draw_bc_card(self, s) -> None:
        c = self.bc_card
        self._header(s, c, "Lattice and boundary conditions", "bc")
        for name, st in (("L₁", self.L1), ("L₂", self.L2)):
            text_at(s, self.fonts.ui, name, INK_SOFT, (st.minus.x - 8, st.minus.centery), "midright")
            st.draw(s, self.fonts, self.mouse)
        self.reset_button.draw(s, self.fonts, self.mouse)
        for name, ry, seg in zip(SEAM_NAMES, self.seam_rows, self.seams):
            text_at(s, self.fonts.ui, name, INK, (c.x + PAD, ry + 16), "midleft")
            seg.draw(s, self.fonts, self.mouse)

    def _draw_bands_card(self, s) -> None:
        c = self.bands_card
        self._header(s, c, "Bands of the vortex-free sector", "bands")
        x, y = c.x + PAD, c.y + PAD + 32
        if self.touch:
            text_at(s, self.fonts.ui_bold, "The bands touch: the Chern number is not defined.",
                    TOUCH_COL, (x, y))
        else:
            r = text_at(s, self.fonts.ui, "Chern number of the lower band:", INK, (x, y))
            f = _formula(f"ν = {self.chern:+d}" if self.chern else "ν = 0", 20)
            f.blit(s, r.right + 12, r.centery + f.base - f.h / 2)
        busy = self.picker.dragging or self.kappa.dragging
        self.band3d.draw(s, self.band_rect, self.picker.J, self.kappa.value, busy)
        text_at(s, self.fonts.small, "drag to orbit · wheel to zoom · right-click to reset",
                INK_SOFT, (c.centerx, c.bottom - PAD + 4), "midbottom")

    # ------------------------------------------------------------------
    def _spec_plot(self) -> pygame.Rect:
        c = self.spec_card
        return pygame.Rect(c.x + PAD + 40, c.y + PAD + 44, c.w - 2 * PAD - 48, c.h - 2 * PAD - 84)

    def _nearest_dot(self, pos) -> int | None:
        if not self.spec_dots or not self.spec_card.collidepoint(pos):
            return None
        m = min(range(len(self.spec_dots)), key=lambda k: _dist2(self.spec_dots[k], pos))
        return m if _dist2(self.spec_dots[m], pos) < HOVER_R ** 2 else None

    def _draw_spec_card(self, s) -> None:
        c = self.spec_card
        self._header(s, c, "Single-particle energies")
        eps = self.result.eps
        f = _formula(f"ε_1 = {eps[0]:.5f}", 17)
        f.blit(s, c.right - PAD - f.w, c.y + PAD - 6 + f.base)
        plot = self._spec_plot()
        if plot.h < 40:
            return
        top = max(2.0, float(eps[-1]) * 1.05)
        pygame.draw.line(s, HAIRLINE, plot.bottomleft, plot.bottomright)
        t = 0.0
        while t <= top:
            yy = plot.bottom - t / top * plot.h
            pygame.draw.line(s, (240, 241, 245), (plot.x, yy), (plot.right, yy))
            text_at(s, self.fonts.small, f"{t:.1f}", INK_SOFT, (plot.x - 8, yy), "midright")
            t += 0.5
        lab = _formula("ε_m", 17, INK_SOFT)
        lab.blit(s, c.x + PAD, plot.y - 30 + lab.base)
        n = len(eps)
        self.spec_dots = []
        for m, e in enumerate(eps):
            px = plot.x + (m + 0.5) / n * plot.w
            py = plot.bottom - e / top * plot.h
            self.spec_dots.append((px, py))
        if self.hover_dot is not None and self.hover_dot < n:
            aa_circle(s, (255, 226, 150), self.spec_dots[self.hover_dot], 7)
        for p in self.spec_dots:
            aa_circle(s, INK, p, 2.6)
        if self.selected is not None:
            aa_circle(s, MODE_COL, self.spec_dots[self.selected], 8.5, width=2.0)
        text_at(s, self.fonts.small, f"m = 1 … {n}", INK_SOFT, (plot.centerx, plot.bottom + 18), "center")
        text_at(s, self.fonts.small, "click a dot to show its mode", INK_SOFT,
                (plot.right, c.bottom - PAD + 2), "bottomright")

    # ------------------------------------------------------------------
    async def run(self) -> None:
        running = True
        while running:
            dt = self.clock.tick(60) / 1000.0
            page = webcanvas.poll(dt)
            if page:
                self.fit_page(page)
            for event in pygame.event.get():
                if not self.dispatch(event):
                    running = False
            self.draw()
            await asyncio.sleep(0)  # lets the browser run between frames
        pygame.quit()


def _dist2(a, b) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


_formula_cache: dict = {}


def _formula(src: str, size: int, colour=INK) -> mt.Box:
    key = (src, size, colour)
    if key not in _formula_cache:
        _formula_cache[key] = mt.formula(src, size, colour)
    return _formula_cache[key]


async def main() -> None:
    await App().run()


if __name__ == "__main__":
    asyncio.run(main())
    if sys.platform != "emscripten":
        sys.exit(0)
