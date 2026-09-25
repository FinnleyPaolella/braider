"""Drawing of fusion-tree kets, the state expression and the F-matrix panel.

Kets are drawn as in the usual fusion-diagram notation: a row of dots, with a
labelled box around every group of anyons that is fused together.
Everything here is assembled from small surfaces that are laid out side by
side, centred on a common horizontal axis.
"""

from __future__ import annotations

import math
import os
import sys
from functools import lru_cache

import pygame

from anyons import LABEL, PHASE_TEXT, TAU, VAC, State, basis, internal_spans

INK = (22, 24, 28)
WHITE = (255, 255, 255)
BOX = (96, 104, 120)
# highlight roles (see steps.py): the three parts of an F-move, the braided pair
# the fusion channels of a braided pair, and the phases they pick up
SECTOR_COLOURS = {VAC: (224, 138, 16), TAU: (214, 56, 140)}
ROLE_COLOURS = {
    "a": (226, 88, 72),
    "b": (30, 160, 110),
    "c": (50, 116, 230),
    "pair": (146, 78, 214),
    "vac": SECTOR_COLOURS[VAC],
    "tau": SECTOR_COLOURS[TAU],
}
LABEL_CHARGE = {text: q for q, text in LABEL.items()}


def role_colour(role: str, label: str = ""):
    """``sector`` takes the colour of the box's own charge."""
    if role == "sector":
        return SECTOR_COLOURS[LABEL_CHARGE[label]]
    return ROLE_COLOURS[role]
FADED_ALPHA = 80           # alpha of terms with zero amplitude
PANEL_BG = (241, 242, 245)

SS = 4                     # supersampling for line art

DOT_SPACING = 24           # minimum distance between dots
DOT_GAP = 7                # between a label or dot and the next box or dot
DOT_R = 8.5
BOX_PAD = 6                # between the dots and their box
BOX_STEP = 5               # between a box (or its label) and the box around it
BOX_RADIUS = 6
LABEL_GAP = 3              # between a box's right edge and its label
LABEL_DROP = 5             # label baseline below the box's bottom edge
LINE_W = 1.6

UI_STACK = "segoeui,segoe ui,inter,helveticaneue,helvetica neue,arial"
MATH_STACK = "cambria,cambriamath,georgia,timesnewroman"

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

# (first anyon, last anyon, charge label) of one box
Group = tuple[int, int, str]


class Fonts:
    def __init__(self) -> None:
        self.heading = sysfont("segoeuiblack,arialblack," + UI_STACK, 25)
        self.math = sysfont(MATH_STACK, 22)
        self.math_it = sysfont(MATH_STACK, 22, italic=True)
        self.sup = sysfont(MATH_STACK, 14)
        self.label = sysfont(MATH_STACK, 17, italic=True)
        self.dot_tau = sysfont(MATH_STACK, 15, italic=True)
        self.ui = sysfont(UI_STACK, 15)
        self.ui_bold = sysfont("segoeuisemibold," + UI_STACK, 15)
        self.ui_title = sysfont("segoeuisemibold," + UI_STACK, 19)
        self.caps = sysfont("segoeuisemibold," + UI_STACK, 12)


_fonts: Fonts | None = None


def fonts() -> Fonts:
    global _fonts
    if _fonts is None:
        _fonts = Fonts()
    return _fonts


# ----------------------------------------------------------------------
# fusion trees
# ----------------------------------------------------------------------
def tree_groups(state: State, labels: tuple[int, ...]) -> tuple[Group, ...]:
    """The boxes of one basis state of ``state``'s fusion tree."""
    return tuple(
        (lo, hi, LABEL[x]) for (lo, hi), x in zip(internal_spans(state.tree), labels)
    )


def tau_glyph(size: int = 0) -> pygame.Surface:
    """A white tau, cropped to its ink, for drawing inside an anyon."""
    f = fonts().dot_tau if not size else _tau_font(size)
    img = f.render("τ", True, WHITE)
    return img.subsurface(img.get_bounding_rect()).copy()


@lru_cache(maxsize=4)
def _tau_font(size: int) -> pygame.font.Font:
    return sysfont(MATH_STACK, size, italic=True)


def left_tree(labels: tuple[int, ...]) -> tuple[Group, ...]:
    """((t, t)_x1, t)_x2 ...: every box starts at the first anyon."""
    return tuple((0, k + 1, LABEL[x]) for k, x in enumerate(labels))


def right_tree3(inner: int, total: int) -> tuple[Group, ...]:
    """(t, (t, t)_inner)_total."""
    return ((1, 2, LABEL[inner]), (0, 2, LABEL[total]))


Highlight = tuple[tuple[tuple[int, int], str], ...]


@lru_cache(maxsize=512)
def ket(n: int, groups: tuple[Group, ...], hl: Highlight = ()) -> pygame.Surface:
    """A ket |...> of n anyons with a labelled box around each fused group.

    Boxes are laid out from the inside out. Each label sits as a subscript
    just right of its box's lower right corner, and every enclosing box grows
    to clear the boxes and labels inside it. ``hl`` colours boxes or single
    anyons, given by their span, with a highlight role.
    """
    f = fonts()
    label_of = {(g[0], g[1]): g[2] for g in groups}
    role = {sp: role_colour(r, label_of.get(sp, "")) for sp, r in hl}
    ascent = f.label.get_ascent()

    # per group: [left, right, half-height, label image, label left, glyph bottom]
    boxes: dict[Group, list] = {}
    xs: list[float] = []

    def close(g: Group) -> None:
        inner = [boxes[h] for h in boxes if g[0] <= h[0] and h[1] <= g[1]]
        left = min([xs[g[0]] - DOT_R - BOX_PAD] + [b[0] - BOX_STEP for b in inner])
        right = max(
            [xs[g[1]] + DOT_R + BOX_PAD]
            + [b[4] + b[3].get_width() + BOX_STEP for b in inner]
        )
        half_h = max(
            [DOT_R + BOX_PAD] + [max(b[2], b[5]) + BOX_STEP for b in inner]
        )
        img = f.label.render(g[2], True, role.get((g[0], g[1]), INK))
        br = img.get_bounding_rect()
        img = img.subsurface((br.x, 0, br.w, img.get_height())).copy()
        # the glyph's baseline sits LABEL_DROP below the box's bottom edge
        glyph_bottom = half_h + LABEL_DROP + (br.bottom - ascent)
        boxes[g] = [left, right, half_h, img, right + LABEL_GAP, glyph_bottom]

    # place the dots left to right, each clear of the boxes and labels that
    # closed before it and with room for the boxes that open at it
    for i in range(n):
        if i == 0:
            xs.append(0.0)
        else:
            opening = sum(1 for g in groups if g[0] == i)
            room = DOT_R + (BOX_PAD + BOX_STEP * (opening - 1) if opening else 0)
            done = [b[4] + b[3].get_width() for g, b in boxes.items() if g[1] < i]
            xs.append(max([xs[-1] + DOT_SPACING, xs[-1] + DOT_R + DOT_GAP + room]
                          + [x + DOT_GAP + room for x in done]))
        for g in sorted((g for g in groups if g[1] == i), key=lambda g: g[1] - g[0]):
            close(g)

    min_x = min([-DOT_R] + [b[0] for b in boxes.values()])
    max_x = max(
        [xs[-1] + DOT_R]
        + [b[4] + b[3].get_width() for b in boxes.values()]
    )
    half_h = max([DOT_R] + [max(b[2], b[5]) for b in boxes.values()]) + 6
    bracket_gap, angle_w = 8, 9
    left = min_x - bracket_gap - 2
    right = max_x + bracket_gap + angle_w + 2
    w, h = math.ceil(right - left), math.ceil(2 * half_h + 2)
    ox, oy = -left, h / 2  # local -> surface coordinates

    big = pygame.Surface((w * SS, h * SS), pygame.SRCALPHA)
    big.fill((*BOX, 0))  # so scaled edges blend towards the line colour

    def P(x: float, y: float) -> tuple[float, float]:
        return ((x + ox) * SS, (y + oy) * SS)

    lw = round(LINE_W * SS)

    def box_rect(b) -> pygame.Rect:
        x0, y0 = P(b[0], -b[2])
        return pygame.Rect(round(x0), round(y0), round((b[1] - b[0]) * SS), round(2 * b[2] * SS))

    # tints of highlighted boxes first, outermost at the bottom
    for g, b in sorted(boxes.items(), key=lambda gb: gb[0][0] - gb[0][1]):
        col = role.get((g[0], g[1]))
        if col:
            tint = pygame.Surface(box_rect(b).size, pygame.SRCALPHA)
            pygame.draw.rect(tint, (*col, 34), tint.get_rect(),
                             border_radius=round(BOX_RADIUS * SS))
            big.blit(tint, box_rect(b))
    for g, b in boxes.items():
        col = role.get((g[0], g[1]))
        pygame.draw.rect(big, col or BOX, box_rect(b), round(lw * 1.7) if col else lw,
                         border_radius=round(BOX_RADIUS * SS))
    for i in range(n):
        pygame.draw.circle(big, role.get((i, i), INK), P(xs[i], 0), DOT_R * SS)
    # |  and  >
    top, bot = -half_h + 1, half_h - 1
    bx = min_x - bracket_gap
    pygame.draw.line(big, INK, P(bx, top), P(bx, bot), lw)
    ax = max_x + bracket_gap
    pygame.draw.lines(big, INK, False, [P(ax, top), P(ax + angle_w, 0), P(ax, bot)], lw)
    for pt in (P(ax, top), P(ax + angle_w, 0), P(ax, bot)):
        pygame.draw.circle(big, INK, pt, lw / 2)

    surf = pygame.transform.smoothscale(big, (w, h))
    for _, _, bh, img, lx, _ in boxes.values():
        top_y = bh + LABEL_DROP - ascent
        surf.blit(img, (round(lx + ox), round(top_y + oy)))
    tau = tau_glyph()
    for x in xs:
        surf.blit(tau, tau.get_rect(center=(round(x + ox), round(oy))))
    return surf


def ribbon(pts: list[tuple[float, float]], width: float) -> list[tuple[float, float]]:
    """Outline of a polyline thickened to ``width``."""
    h = width / 2
    left, right = [], []
    for i, (x, y) in enumerate(pts):
        ax, ay = pts[max(i - 1, 0)]
        bx, by = pts[min(i + 1, len(pts) - 1)]
        dx, dy = bx - ax, by - ay
        n = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / n * h, dx / n * h
        left.append((x + nx, y + ny))
        right.append((x - nx, y - ny))
    return left + right[::-1]


# ----------------------------------------------------------------------
# formula layout
# ----------------------------------------------------------------------
def text(s: str) -> pygame.Surface:
    return fonts().math.render(s, True, INK)


def power(base: str, exp: str) -> pygame.Surface:
    """``base`` with a raised exponent, e.g. phi^-1."""
    f = fonts()
    b = f.math_it.render(base, True, INK)
    e = f.sup.render(exp, True, INK)
    rise = round(b.get_height() * 0.28)
    w = b.get_width() + e.get_width() + 1
    h = b.get_height() + 2 * rise   # symmetric, so the centre stays on the axis
    out = pygame.Surface((w, h), pygame.SRCALPHA)
    out.blit(b, (0, rise))
    out.blit(e, (b.get_width() + 1, rise - 2))
    return out


def sqrt(inner: pygame.Surface) -> pygame.Surface:
    """A radical sign with an overline over ``inner``."""
    iw, ih = inner.get_size()
    hook, pad_top, pad_r = 13, 5, 2
    w, h = iw + hook + pad_r + 2, ih + pad_top + 2
    big = pygame.Surface((w * SS, h * SS), pygame.SRCALPHA)
    top = 2
    pts = [
        (1, h * 0.62), (4, h * 0.55), (8, h - 3), (hook - 1, top), (w - 1, top),
    ]
    pygame.draw.lines(big, INK, False, [(x * SS, y * SS) for x, y in pts], round(1.5 * SS))
    out = pygame.transform.smoothscale(big, (w, h))
    out.blit(inner, (hook + 1, pad_top + 1))
    return out


def hcat(parts: list[pygame.Surface], gap: int = 8) -> pygame.Surface:
    """Parts side by side, vertically centred."""
    w = sum(p.get_width() for p in parts) + gap * max(len(parts) - 1, 0)
    h = max((p.get_height() for p in parts), default=1)
    out = pygame.Surface((max(w, 1), h), pygame.SRCALPHA)
    x = 0
    for p in parts:
        out.blit(p, (x, (h - p.get_height()) // 2))
        x += p.get_width() + gap
    return out


def vstack(rows: list[pygame.Surface], gap: int) -> pygame.Surface:
    w = max(r.get_width() for r in rows)
    h = sum(r.get_height() for r in rows) + gap * (len(rows) - 1)
    out = pygame.Surface((w, h), pygame.SRCALPHA)
    y = 0
    for r in rows:
        out.blit(r, (0, y))
        y += r.get_height() + gap
    return out


def faded(s: pygame.Surface, alpha: int) -> pygame.Surface:
    out = s.copy()
    out.fill((255, 255, 255, alpha), special_flags=pygame.BLEND_RGBA_MULT)
    return out


# ----------------------------------------------------------------------
# the state
# ----------------------------------------------------------------------
def fmt_real(x: float) -> str:
    return f"{abs(x):.3f}".rstrip("0").rstrip(".")


def fmt_amp(z: complex) -> tuple[str, str]:
    """(sign, magnitude text) of an amplitude, the sign being '+' or '−'."""
    re, im = z.real, z.imag
    if abs(re) < 5e-4 and abs(im) < 5e-4:
        return "+", "0"
    if abs(im) < 5e-4:
        return ("−" if re < 0 else "+"), fmt_real(re)
    if abs(re) < 5e-4:
        return ("−" if im < 0 else "+"), fmt_real(im) + "i"
    op = "−" if im < 0 else "+"
    return "+", f"({'−' if re < 0 else ''}{fmt_real(re)} {op} {fmt_real(im)}i)"


def phase_image(text: str, colour) -> pygame.Surface:
    return fonts().math.render(text, True, colour)


def term_phase(state: State, labels: tuple[int, ...], phases) -> tuple[str, tuple]:
    """The braid phase written in front of one term, and its colour."""
    gap, ccw = phases
    charge = state.charges(labels)[(gap, gap + 1)]
    return PHASE_TEXT[(charge, ccw)], SECTOR_COLOURS[charge]


def state_expression(
    n: int, state: State, max_w: float, hl: Highlight = (),
    phases: tuple[int, bool] | None = None, blank_phases: bool = False,
) -> tuple[pygame.Surface, list[dict]]:
    """alpha_1 |s1> + alpha_2 |s2> + ..., wrapped to ``max_w`` pixels.

    With ``phases`` = (gap, ccw) every term is preceded by its braid phase,
    coloured by the pair's charge (left as empty space if ``blank_phases``).
    Also returns per term the rectangles of its ``phase`` and ``amp``, for
    animating the phases into the coefficients.
    """
    term_gap = 10
    # per term: [(surface, tag, gap after it)], and whether its amplitude is 0
    terms = []
    for i, labels in enumerate(basis(state.tree)):
        sign, mag = fmt_amp(state.amp(labels))
        parts = []
        if i > 0 or sign == "−":
            parts.append((text(sign), "sign", 10))
        if phases is not None:
            ptext, pcol = term_phase(state, labels, phases)
            img = phase_image(ptext, pcol)
            if blank_phases:
                img = pygame.Surface(img.get_size(), pygame.SRCALPHA)
            parts.append((img, "phase", 4))
            parts.append((text("·"), "dot", 4))
        parts.append((text(mag), "amp", 4))
        parts.append((ket(n, tree_groups(state, labels), hl), "ket", 0))
        terms.append((parts, mag == "0"))

    # flow the terms into rows
    rows: list[list] = [[]]
    row_w = 0
    for idx, (parts, zero) in enumerate(terms):
        w = sum(p.get_width() + g for p, _, g in parts) + term_gap
        if rows[-1] and row_w + w > max_w:
            rows.append([])
            row_w = 0
        rows[-1].append((idx, parts, zero))
        row_w += w

    row_gap = 14
    heights = [max(p.get_height() for _, parts, _ in r for p, _, _ in parts) for r in rows]
    widths = [sum(p.get_width() + g for _, parts, _ in r for p, _, g in parts)
              + term_gap * (len(r) - 1) for r in rows]
    out = pygame.Surface((max(1, max(widths)), sum(heights) + row_gap * (len(rows) - 1)),
                         pygame.SRCALPHA)
    anchors: list[dict] = [{} for _ in terms]
    y = 0
    for r, h in zip(rows, heights):
        x = 0
        for idx, parts, zero in r:
            for img, tag, g in parts:
                if zero:
                    img = faded(img, FADED_ALPHA)
                rect = img.get_rect(topleft=(x, y + (h - img.get_height()) // 2))
                out.blit(img, rect)
                if tag in ("phase", "amp"):
                    anchors[idx][tag] = rect
                x += img.get_width() + g
            x += term_gap
        y += h + row_gap
    return out, anchors


# ----------------------------------------------------------------------
# the F matrix
# ----------------------------------------------------------------------
@lru_cache(maxsize=8)
def f_matrix_panel(width: int = 0, coloured: bool = False) -> pygame.Surface:
    """The F-moves of three taus, with the equals signs aligned.

    ``coloured`` paints the three anyons like the parts of the F-move being
    explained.
    """
    hl = (((0, 0), "a"), ((1, 1), "b"), ((2, 2), "c")) if coloured else ()
    phi_inv = lambda: power("φ", "−1")  # noqa: E731
    sqrt_phi_inv = lambda: sqrt(power("φ", "−1"))  # noqa: E731
    L = lambda a, c: ket(3, left_tree((a, c)), hl)  # noqa: E731
    Rt = lambda a, c: ket(3, right_tree3(a, c), hl)  # noqa: E731

    eqs = [
        (L(TAU, VAC), [Rt(TAU, VAC)]),
        (L(VAC, TAU), [phi_inv(), Rt(VAC, TAU), text("+"), sqrt_phi_inv(), Rt(TAU, TAU)]),
        (L(TAU, TAU), [sqrt_phi_inv(), Rt(VAC, TAU), text("−"), phi_inv(), Rt(TAU, TAU)]),
    ]
    lhs_w = max(lhs.get_width() for lhs, _ in eqs)
    eq_sign = text("=")
    rows = []
    for lhs, rhs in eqs:
        r = hcat(rhs, gap=8)
        h = max(lhs.get_height(), r.get_height())
        row = pygame.Surface(
            (lhs_w + 14 + eq_sign.get_width() + 14 + r.get_width(), h), pygame.SRCALPHA
        )
        row.blit(lhs, (lhs_w - lhs.get_width(), (h - lhs.get_height()) // 2))
        x = lhs_w + 14
        row.blit(eq_sign, (x, (h - eq_sign.get_height()) // 2))
        row.blit(r, (x + eq_sign.get_width() + 14, (h - r.get_height()) // 2))
        rows.append(row)
    return panel("F Matrix", vstack(rows, gap=12), width)


def panel(title: str, body: pygame.Surface, width: int = 0) -> pygame.Surface:
    """A grey rounded panel with a bold title; ``body`` is centred in it."""
    title_img = fonts().heading.render(title, True, INK)
    pad = 28
    w = max(width, body.get_width() + 2 * pad, title_img.get_width() + 2 * pad)
    h = pad + title_img.get_height() + 16 + body.get_height() + pad
    out = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.rect(out, PANEL_BG, out.get_rect(), border_radius=18)
    out.blit(title_img, (pad, pad - 6))
    out.blit(body, ((w - body.get_width()) // 2, pad + title_img.get_height() + 16))
    return out


# ----------------------------------------------------------------------
# the braid (R) matrix
# ----------------------------------------------------------------------
CROSS_W = 34               # distance between the two strands
CROSS_H = 70
CROSS_STRAND_W = 3.5
CROSS_HALO = 4
CROSS_PAD = 11             # between the strands and their box


def crossing(a_over: bool, charge: str, colour=None) -> pygame.Surface:
    """Two strands exchanging once inside a box labelled with their charge.

    Strand A runs from bottom left to top right; ``a_over`` puts it in front.
    The shape matches the crossings of the braid diagram. ``colour`` paints
    the box and its label.
    """
    f = fonts()
    ascent = f.label.get_ascent()
    img = f.label.render(charge, True, colour or INK)
    br = img.get_bounding_rect()
    img = img.subsurface((br.x, 0, br.w, img.get_height())).copy()

    bw = CROSS_W + 2 * CROSS_PAD
    bh = CROSS_H + 2 * CROSS_PAD
    w = bw + LABEL_GAP + img.get_width() + 2
    h = bh + LABEL_DROP + (br.bottom - ascent) + 2
    big = pygame.Surface((w * SS, h * SS), pygame.SRCALPHA)
    # transparent pixels carry the panel colour, so scaled edges don't darken
    big.fill((*PANEL_BG, 0))
    if colour:
        tint = pygame.Surface((bw * SS, bh * SS), pygame.SRCALPHA)
        pygame.draw.rect(tint, (*colour, 34), tint.get_rect(),
                         border_radius=round(BOX_RADIUS * SS))
        big.blit(tint, (0, 0))
    pygame.draw.rect(
        big, colour or BOX, (0, 0, bw * SS, bh * SS),
        round(LINE_W * SS * (1.7 if colour else 1)),
        border_radius=round(BOX_RADIUS * SS),
    )

    cx, r = bw / 2, CROSS_W / 2

    def strand(sign: int, lo: float = 0.0, hi: float = 1.0):
        vs = [lo + (hi - lo) * k / 48 for k in range(49)]
        return [
            ((cx - sign * r * math.cos(math.pi * v)) * SS,
             (CROSS_PAD + (1 - v) * CROSS_H) * SS)
            for v in vs
        ]

    over, under = (+1, -1) if a_over else (-1, +1)
    pygame.draw.polygon(big, INK, ribbon(strand(under), CROSS_STRAND_W * SS))
    pygame.draw.polygon(
        big, PANEL_BG, ribbon(strand(over, 0.2, 0.8), (CROSS_STRAND_W + 2 * CROSS_HALO) * SS)
    )
    pygame.draw.polygon(big, INK, ribbon(strand(over), CROSS_STRAND_W * SS))
    for x in (cx - r, cx + r):
        for y in (CROSS_PAD, CROSS_PAD + CROSS_H):
            pygame.draw.circle(big, INK, (x * SS, y * SS), CROSS_STRAND_W / 2 * SS)

    out = pygame.transform.smoothscale(big, (w, h))
    out.blit(img, (bw + LABEL_GAP, bh + LABEL_DROP - ascent))
    return out


@lru_cache(maxsize=8)
def r_matrix_panel(width: int = 0, active: bool | None = None) -> pygame.Surface:
    """The phase of one exchange, per fusion channel and direction.

    Counterclockwise (strand A in front) gives R_1 = exp(-4 pi i/5) and
    R_tau = exp(3 pi i/5); clockwise gives their inverses. ``active`` (True
    for counterclockwise) colours that direction by fusion channel and fades
    the other one.
    """
    cells = []
    for ccw in (True, False):
        for charge in (VAC, TAU):
            lit = ccw == active
            colour = SECTOR_COLOURS[charge] if lit else None
            diagram = crossing(ccw, LABEL[charge], colour)
            label = fonts().math.render(PHASE_TEXT[(charge, ccw)], True, colour or INK)
            cell = pygame.Surface(
                (max(diagram.get_width(), label.get_width()),
                 diagram.get_height() + 12 + label.get_height()),
                pygame.SRCALPHA,
            )
            # centre the box itself (not box + subscript) over the phase
            box_w = CROSS_W + 2 * CROSS_PAD
            cell.blit(diagram, ((cell.get_width() - box_w) // 2, 0))
            cell.blit(label, ((cell.get_width() - label.get_width()) // 2,
                              diagram.get_height() + 12))
            if active is not None and not lit:
                cell = faded(cell, 90)
            cells.append(cell)

    gap = 36
    divider_gap = 2 * gap
    w = sum(c.get_width() for c in cells) + 2 * gap + divider_gap
    h = max(c.get_height() for c in cells)
    body = pygame.Surface((w, h), pygame.SRCALPHA)
    x = 0
    for i, c in enumerate(cells):
        body.blit(c, (x, 0))
        x += c.get_width() + (divider_gap if i == 1 else gap)
        if i == 1:
            # a hairline between the two directions
            mx = x - divider_gap // 2
            pygame.draw.line(body, (214, 217, 223), (mx, 6), (mx, h - 6), 1)
    return panel("Braid Matrix", body, width)
