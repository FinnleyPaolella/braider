"""A small TeX-like formula typesetter on top of pygame fonts.

``formula(src, size)`` lays out a string such as ``H = -J_x \\sum{x} σ^x_j σ^x_k``
and returns a ``Box``: a surface plus the height of its baseline, so boxes can
be set side by side on a common baseline (``hbox``) or stacked (``vstack``).

Syntax
    a_b  a^b  a_{...}^{...}   subscript / superscript (either order)
    {...}                     grouping
    \\sum{below} \\prod{below} large operator with a limit underneath
    \\frac{num}{den}           fraction
    \\hat{...} \\tilde{...}     accents
    \\paren{...} \\abs{...}     brackets that grow with their contents
    \\langle \\rangle           angle brackets
    \\rm{...}                  upright text
    \\c{x}{...}                colour (a key of ``COLOURS``, x / y / z)
    ' ' thin space, '~' wide space

Latin and lower-case Greek letters are set in italics, everything else upright.
Operators get spacing on both sides unless they start a group (unary minus).
"""

from __future__ import annotations

import math
from functools import lru_cache

import pygame

from ui import BOND_COL, INK, stroke, sysfont

MATH_STACK = "cambria,cambriamath,georgia,timesnewroman"
COLOURS = dict(BOND_COL)
OPS = set("+−=±∈→≡")
TIGHT_OPS = set("·")
UPRIGHT = set("0123456789()[]|,.;:!*'/ΣΔΓΨΦΩ")


@lru_cache(maxsize=64)
def _font(size: int, italic: bool) -> pygame.font.Font:
    return sysfont(MATH_STACK, size, italic=italic)


def script_size(size: int) -> int:
    return max(9, round(size * 0.7))


class Box:
    __slots__ = ("surf", "base")

    def __init__(self, surf: pygame.Surface, base: float) -> None:
        self.surf = surf
        self.base = base

    @property
    def w(self) -> int:
        return self.surf.get_width()

    @property
    def h(self) -> int:
        return self.surf.get_height()

    def blit(self, target: pygame.Surface, x: float, baseline_y: float) -> None:
        target.blit(self.surf, (round(x), round(baseline_y - self.base)))


def space(w: float, h: float = 1, base: float = 0) -> Box:
    return Box(pygame.Surface((max(0, round(w)), max(1, round(h))), pygame.SRCALPHA), base)


def place(items: list[tuple[Box, float, float]]) -> Box:
    """Compose boxes at (x, baseline offset upward) into one box."""
    if not items:
        return space(0)
    top = max(b.base + dy for b, x, dy in items)
    bottom = max(b.h - b.base - dy for b, x, dy in items)
    left = min(0, min(x for b, x, dy in items))
    right = max(x + b.w for b, x, dy in items)
    surf = pygame.Surface((max(1, math.ceil(right - left)), max(1, math.ceil(top + bottom))),
                          pygame.SRCALPHA)
    for b, x, dy in items:
        surf.blit(b.surf, (round(x - left), round(top - dy - b.base)))
    return Box(surf, top)


def hbox(boxes: list[Box], gap: float = 0) -> Box:
    items, x = [], 0.0
    for b in boxes:
        items.append((b, x, 0))
        x += b.w + gap
    return place(items)


def vstack(rows: list[Box], gap: float, align: str = "left", base_row: int = 0) -> Box:
    """Rows top to bottom; the result's baseline is that of ``base_row``."""
    width = max(r.w for r in rows)
    y = 0.0
    items = []
    base_y = 0.0
    for i, r in enumerate(rows):
        x = {"left": 0, "center": (width - r.w) / 2, "right": width - r.w}[align]
        items.append((r, x, y))
        if i == base_row:
            base_y = y + r.base
        y += r.h + gap
    surf = pygame.Surface((max(1, width), max(1, round(y - gap))), pygame.SRCALPHA)
    for r, x, yy in items:
        surf.blit(r.surf, (round(x), round(yy)))
    return Box(surf, base_y)


def glyph(s: str, size: int, italic: bool, colour) -> Box:
    f = _font(size, italic)
    return Box(f.render(s, True, colour), f.get_ascent())


# ----------------------------------------------------------------------
# parsing
# ----------------------------------------------------------------------
def parse(src: str) -> list:
    """Source string -> list of nodes (see the module docstring)."""
    pos = 0

    def group() -> list:
        nodes: list = []
        while pos < len(src) and src[pos] != "}":
            nodes.append(item())
        return nodes

    def arg():
        nonlocal pos
        if pos < len(src) and src[pos] == "{":
            pos += 1
            g = group()
            pos += 1  # the closing brace
            return ("grp", g)
        return atom()

    def atom():
        nonlocal pos
        ch = src[pos]
        if ch == "{":
            return arg()
        if ch == "\\":
            end = pos + 1
            while end < len(src) and src[end].isalpha():
                end += 1
            name = src[pos + 1:end]
            pos = end
            nargs = {"sum": 1, "prod": 1, "frac": 2, "hat": 1, "tilde": 1, "paren": 1,
                     "abs": 1, "rm": 1, "c": 2, "langle": 0, "rangle": 0}[name]
            if name in ("rm", "c"):
                # the first argument of \rm and \c is literal text
                args = [_raw()]
                if name == "c":
                    args.append(arg())
                return ("cmd", name, args)
            return ("cmd", name, [arg() for _ in range(nargs)])
        pos += 1
        return ("ch", ch)

    def _raw() -> str:
        nonlocal pos
        assert src[pos] == "{"
        end = src.index("}", pos)
        s = src[pos + 1:end]
        pos = end + 1
        return s

    def item():
        nonlocal pos
        base = atom()
        sub = sup = None
        while pos < len(src) and src[pos] in "_^":
            mark = src[pos]
            pos += 1
            a = arg()
            if mark == "_":
                sub = a
            else:
                sup = a
        if sub is None and sup is None:
            return base
        return ("scr", base, sub, sup)

    nodes = []
    while pos < len(src):
        nodes.append(item())
    return nodes


# ----------------------------------------------------------------------
# layout
# ----------------------------------------------------------------------
def _is_italic(ch: str) -> bool:
    if ch in UPRIGHT:
        return False
    return ch.isalpha() and (ch.isascii() or ch.islower())


def layout(nodes: list, size: int, colour=INK) -> Box:
    items: list[Box] = []
    prev = None     # the last node that was not a space
    for node in nodes:
        kind = node[0]
        unary = prev is None or (prev[0] == "ch" and prev[1] in OPS | set("(,"))
        if not (kind == "ch" and node[1] in " ~"):
            prev = node
        if kind == "ch":
            ch = node[1]
            if ch == " ":
                items.append(space(size * 0.17))
            elif ch == "~":
                items.append(space(size * 0.9))
            elif ch in OPS:
                g = glyph(ch, size, False, colour)
                pad = size * (0.2 if size >= 14 else 0.08)
                if unary:
                    items.append(hbox([g, space(size * 0.04)]))
                else:
                    items.append(hbox([space(pad), g, space(pad)]))
            elif ch in TIGHT_OPS:
                g = glyph(ch, size, False, colour)
                items.append(hbox([space(size * 0.08), g, space(size * 0.08)]))
            elif ch == ",":
                items.append(hbox([glyph(",", size, False, colour), space(size * 0.15)]))
            else:
                items.append(glyph(ch, size, _is_italic(ch), colour))
        else:
            items.append(_layout_node(node, size, colour))
    return hbox(items)


def _layout_node(node, size: int, colour) -> Box:
    kind = node[0]
    if kind == "ch":
        return layout([node], size, colour)
    if kind == "grp":
        return layout(node[1], size, colour)
    if kind == "scr":
        _, base, sub, sup = node
        b = _layout_node(base, size, colour)
        ss = script_size(size)
        items = [(b, 0, 0)]
        x = b.w + 1
        tall = b.base > _font(size, False).get_ascent() * 1.15   # a big operator
        if sup is not None:
            p = _layout_node(sup, ss, colour)
            up = (b.base - ss * 0.55) if tall else size * 0.40
            items.append((p, x, up))
        if sub is not None:
            s = _layout_node(sub, ss, colour)
            down = (b.h - b.base) * 0.8 if tall else size * (0.26 if sup is not None else 0.2)
            items.append((s, x - (1 if base[0] == "ch" and _is_italic(base[1]) else 0), -down))
        return place(items)
    # commands
    _, name, args = node
    if name == "c":
        return _layout_node(args[1], size, COLOURS.get(args[0], colour))
    if name == "rm":
        return glyph(args[0], size, False, colour)
    if name in ("sum", "prod"):
        sig = glyph("Σ" if name == "sum" else "Π", round(size * 1.6), False, colour)
        ink = sig.surf.get_bounding_rect()
        # crop to the ink so the limit sits snugly underneath
        sig = Box(sig.surf.subsurface(ink).copy(), sig.base - ink.top - size * 0.14)
        below = _layout_node(args[0], script_size(size), colour)
        return hbox([space(size * 0.1),
                     vstack([sig, below], size * 0.08 - _ink_top(below), "center", 0),
                     space(size * 0.12)])
    if name == "frac":
        fs = round(size * 0.8)
        num = _layout_node(args[0], fs, colour)
        den = _layout_node(args[1], fs, colour)
        w = max(num.w, den.w) + round(size * 0.25)
        thick = max(1.0, size / 16)
        gap = size * 0.06
        h = num.h + den.h + 2 * gap + thick
        surf = pygame.Surface((w + 2, math.ceil(h)), pygame.SRCALPHA)
        surf.blit(num.surf, ((w - num.w) / 2 + 1, 0))
        bar_y = num.h + gap
        pygame.draw.rect(surf, colour, pygame.Rect(1, round(bar_y), w, round(thick)))
        surf.blit(den.surf, ((w - den.w) / 2 + 1, round(bar_y + thick + gap)))
        axis = size * 0.27
        return hbox([space(size * 0.08), Box(surf, bar_y + thick / 2 + axis), space(size * 0.08)])
    if name in ("hat", "tilde"):
        inner = _layout_node(args[0], size, colour)
        italic = args[0][0] == "ch" and _is_italic(args[0][1])
        top_ink = _ink_top(inner)
        acc_h = size * 0.2
        cx = inner.w / 2 + (size * 0.08 if italic else 0)
        aw = min(size * 0.5, max(inner.w * 0.8, size * 0.35))
        lw = max(1.0, size / 18)
        y1 = acc_h + 2
        if name == "hat":
            pts = [(cx - aw / 2, y1), (cx, 2), (cx + aw / 2, y1)]
        else:
            pts = [(cx - aw / 2 + aw * t / 12,
                    2 + acc_h / 2 - acc_h / 2 * math.sin(2 * math.pi * t / 12))
                   for t in range(13)]
        img, (ox, oy) = stroke(pts, lw, colour)
        lift = max(0.0, acc_h + 4 - top_ink)
        surf = pygame.Surface((inner.w, math.ceil(inner.h + lift)), pygame.SRCALPHA)
        surf.blit(inner.surf, (0, lift))
        surf.blit(img, (ox, oy + top_ink + lift - acc_h - 4))
        return Box(surf, inner.base + lift)
    if name in ("paren", "abs"):
        inner = _layout_node(args[0], size, colour)
        return delimit(inner, size, colour, "(" if name == "paren" else "|")
    if name in ("langle", "rangle"):
        f = _font(size, False)
        h = f.get_ascent() * 0.95
        w = size * 0.3
        top = f.get_ascent() - h + size * 0.12
        if name == "langle":
            pts = [(w, top), (1.5, top + h / 2), (w, top + h)]
        else:
            pts = [(1.5, top), (w, top + h / 2), (1.5, top + h)]
        img, (ox, oy) = stroke(pts, max(1.0, size / 20), colour)
        surf = pygame.Surface((round(w + 3), f.get_height()), pygame.SRCALPHA)
        surf.blit(img, (ox, oy))
        return Box(surf, f.get_ascent())
    raise ValueError(name)


def _ink_top(box: Box) -> float:
    """Distance from the top of the box to its first inked row."""
    r = box.surf.get_bounding_rect()
    return r.top if r.height else box.base


def delimit(inner: Box, size: int, colour, kind: str = "(", right: str | None = None) -> Box:
    """Brackets around ``inner``, tall enough to cover its ink."""
    ink = inner.surf.get_bounding_rect()
    font_top = inner.base - _font(size, False).get_ascent() * 0.78
    font_bot = inner.base + size * 0.24
    top = min(ink.top - 1, font_top)
    bot = max(ink.bottom + 1, font_bot)
    h = bot - top
    lw = max(1.1, size / 17)
    w = max(size * 0.28, min(size * 0.5, h * 0.18))
    left = _bracket(kind, h, w, lw, colour, False)
    rgt = _bracket(right or kind, h, w, lw, colour, True)
    pad = size * 0.08
    items = [(left, 0, inner.base - top - left.base),
             (inner, left.w + pad, 0),
             (rgt, left.w + 2 * pad + inner.w, inner.base - top - rgt.base)]
    return place(items)


def _bracket(kind: str, h: float, w: float, lw: float, colour, flip: bool) -> Box:
    if kind == "(":
        pts = []
        for t in range(21):
            u = t / 20
            y = u * h
            x = w * 0.9 - w * 0.75 * math.sin(math.pi * u)
            pts.append((x, y))
    elif kind == "|":
        w = lw + 4
        pts = [(w / 2, 0), (w / 2, h)]
    else:
        raise ValueError(kind)
    if flip:
        pts = [(w - x, y) for x, y in pts]
    img, (ox, oy) = stroke(pts, lw, colour)
    surf = pygame.Surface((math.ceil(w) + 1, math.ceil(h) + 1), pygame.SRCALPHA)
    surf.blit(img, (ox, oy))
    return Box(surf, 0)


def matrix(rows: list[list[Box]], size: int, colour=INK) -> Box:
    """A matrix of boxes in round brackets, centred on the maths axis."""
    ncol = max(len(r) for r in rows)
    col_w = [max(r[c].w for r in rows if c < len(r)) for c in range(ncol)]
    asc = [max(b.base for b in r) for r in rows]
    desc = [max(b.h - b.base for b in r) for r in rows]
    cgap = size * 0.9
    rgap = size * 0.2
    width = sum(col_w) + cgap * (ncol - 1)
    height = sum(asc) + sum(desc) + rgap * (len(rows) - 1)
    surf = pygame.Surface((math.ceil(width) + 2, math.ceil(height) + 2), pygame.SRCALPHA)
    y = 0.0
    for r, a, d in zip(rows, asc, desc):
        x = 0.0
        for c, b in enumerate(r):
            b.blit(surf, x + (col_w[c] - b.w) / 2, y + a)
            x += col_w[c] + cgap
        y += a + d + rgap
    inner = Box(surf, height / 2 + size * 0.27)
    return delimit(inner, size, colour)


@lru_cache(maxsize=1024)
def formula(src: str, size: int, colour=INK) -> Box:
    """Typeset ``src``; the result is cached, so treat it as read-only."""
    return layout(parse(src), size, tuple(colour))
