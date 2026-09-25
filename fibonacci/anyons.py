"""Fibonacci anyon fusion spaces, F-moves and braiding.

Charges are ``VAC`` (the vacuum, 1) and ``TAU`` (tau), with tau x tau = 1 + tau.

A basis of n tau anyons is a binary fusion tree whose leaves are the anyon
positions 0..n-1 in order, e.g. ``((0, 1), 2)`` or ``(0, (1, 2))``. Every
internal node is identified by the span ``(first, last)`` of anyons below it,
and a basis state assigns a charge to each internal node; the root carries the
total charge. The standard basis is the left comb ``(((0, 1), 2), ...)``.

Conventions: the F-move ((a, b)_e, c)_d = sum_f [F^{abc}_d]_{ef} (a, (b, c)_f)_d,
and a counterclockwise exchange of two taus fused to c multiplies by R_c.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from typing import Union

VAC, TAU = 0, 1
LABEL = {VAC: "1", TAU: "τ"}

PHI = (1 + math.sqrt(5)) / 2

# F^{tau tau tau}_tau, rows e and columns f indexed by (VAC, TAU); it is real,
# symmetric and its own inverse
F = (
    (1 / PHI, 1 / math.sqrt(PHI)),
    (1 / math.sqrt(PHI), -1 / PHI),
)

# counterclockwise exchange of two taus with total charge c
R = {VAC: cmath.exp(-4j * math.pi / 5), TAU: cmath.exp(3j * math.pi / 5)}

# the same phases written out, keyed by (charge, counterclockwise)
PHASE_TEXT = {
    (VAC, True): "exp(−4πi/5)", (TAU, True): "exp(3πi/5)",
    (VAC, False): "exp(4πi/5)", (TAU, False): "exp(−3πi/5)",
}

Tree = Union[int, tuple]
Span = tuple[int, int]
Labels = tuple[int, ...]
Move = tuple[Span, str]   # (node, "right" | "left"), see ``rotate``


# ----------------------------------------------------------------------
# trees
# ----------------------------------------------------------------------
def comb(n: int) -> Tree:
    """The standard basis tree (((0, 1), 2), ...)."""
    t: Tree = 0
    for i in range(1, n):
        t = (t, i)
    return t


def span(t: Tree) -> Span:
    if isinstance(t, int):
        return (t, t)
    return (span(t[0])[0], span(t[1])[1])


def internal_spans(t: Tree) -> list[Span]:
    """Spans of the internal nodes, children before parents (postorder)."""
    if isinstance(t, int):
        return []
    return internal_spans(t[0]) + internal_spans(t[1]) + [span(t)]


def subtree(t: Tree, sp: Span) -> Tree:
    while span(t) != sp:
        t = t[0] if span(t[0])[1] >= sp[1] else t[1]
    return t


def replace(t: Tree, sp: Span, new: Tree) -> Tree:
    if span(t) == sp:
        return new
    if span(t[0])[1] >= sp[1]:
        return (replace(t[0], sp, new), t[1])
    return (t[0], replace(t[1], sp, new))


def rotate(t: Tree, sp: Span, direction: str) -> Tree:
    """"right": ((A, B), C) -> (A, (B, C)) at node ``sp``; "left" undoes it."""
    node = subtree(t, sp)
    if direction == "right":
        (a, b), c = node
        return replace(t, sp, (a, (b, c)))
    a, (b, c) = node
    return replace(t, sp, ((a, b), c))


def move_parts(t: Tree, move: Move) -> tuple[Span, Span, Span]:
    """The spans of the three parts A, B, C that ``move`` regroups."""
    sp, direction = move
    node = subtree(t, sp)
    if direction == "right":
        (a, b), c = node
    else:
        a, (b, c) = node
    return span(a), span(b), span(c)


def invert(move: Move) -> Move:
    """The move that undoes ``move`` (the node keeps its span)."""
    return (move[0], "left" if move[1] == "right" else "right")


def moves_to_pair(t: Tree, gap: int) -> list[Move]:
    """F-moves after which anyons ``gap`` and ``gap + 1`` fuse directly.

    Take the lowest node containing both. While its left child is not the
    single anyon ``gap``, rotate right there, which moves everything left of
    ``gap`` out of the way; then, while its right child is not the single
    anyon ``gap + 1``, rotate left. Every move shrinks that lowest common node,
    so from the standard basis a single move suffices.
    """
    moves: list[Move] = []
    while True:
        node = t
        while True:
            left, right = node
            if span(left)[1] >= gap + 1:
                node = left
            elif span(right)[0] <= gap:
                node = right
            else:
                break
        left, right = node
        if left != gap:
            move = (span(node), "right")
        elif right != gap + 1:
            move = (span(node), "left")
        else:
            return moves
        moves.append(move)
        t = rotate(t, *move)


def moves_to_comb(t: Tree) -> list[Move]:
    """F-moves back to the standard basis, rotating left from the top down."""
    moves: list[Move] = []
    while True:
        target = None
        # the topmost node whose right child is not a single anyon
        for sp in reversed(internal_spans(t)):
            if not isinstance(subtree(t, sp)[1], int):
                target = sp
                break
        if target is None:
            return moves
        moves.append((target, "left"))
        t = rotate(t, target, "left")


# ----------------------------------------------------------------------
# states
# ----------------------------------------------------------------------
def fuse(a: int, b: int) -> tuple[int, ...]:
    if a == VAC:
        return (b,)
    if b == VAC:
        return (a,)
    return (VAC, TAU)


def f_symbol(a: int, b: int, c: int, d: int, e: int, f: int) -> float:
    """[F^{abc}_d]_{ef}; only the all-tau case is a nontrivial matrix."""
    ok_e = e in fuse(a, b) and d in fuse(e, c)
    ok_f = f in fuse(b, c) and d in fuse(a, f)
    if not (ok_e and ok_f):
        return 0.0
    if a == b == c == d == TAU:
        return F[e][f]
    return 1.0


def basis(t: Tree) -> list[Labels]:
    """All charge assignments of tree ``t`` (1 before tau, lexicographic)."""

    def grow(t: Tree) -> list[tuple[Labels, int]]:
        if isinstance(t, int):
            return [((), TAU)]
        out = []
        for lt, a in grow(t[0]):
            for rt, b in grow(t[1]):
                for c in fuse(a, b):
                    out.append((lt + rt + (c,), c))
        return out

    return sorted(labels for labels, _ in grow(t))


@dataclass(frozen=True)
class State:
    """Amplitudes on the basis of fusion tree ``tree``."""

    tree: Tree
    amps: tuple[tuple[Labels, complex], ...]

    @staticmethod
    def make(tree: Tree, amps: dict[Labels, complex]) -> "State":
        kept = {k: v for k, v in amps.items() if abs(v) > 1e-12}
        return State(tree, tuple(sorted(kept.items())))

    def amp(self, labels: Labels) -> complex:
        return dict(self.amps).get(labels, 0j)

    def charges(self, labels: Labels) -> dict[Span, int]:
        return dict(zip(internal_spans(self.tree), labels))

    def is_standard(self) -> bool:
        n = span(self.tree)[1] + 1
        return self.tree == comb(n)


def initial_state(n: int) -> State:
    """((t, t)_1, t)_tau for three anyons; generally 1 first, then tau."""
    labels = tuple(VAC if k == 0 and n >= 3 else TAU for k in range(n - 1))
    return State.make(comb(n), {labels: 1.0 + 0j})


def f_move(state: State, move: Move) -> State:
    """Rewrite ``state`` in the basis obtained by applying ``move``."""
    sp, direction = move
    sa, sb, sc = move_parts(state.tree, move)
    old_inner = (sa[0], sb[1]) if direction == "right" else (sb[0], sc[1])
    new_inner = (sb[0], sc[1]) if direction == "right" else (sa[0], sb[1])
    new_tree = rotate(state.tree, sp, direction)
    new_spans = internal_spans(new_tree)

    out: dict[Labels, complex] = {}
    for labels, amp in state.amps:
        ch = state.charges(labels)

        def q(s: Span) -> int:
            return TAU if s[0] == s[1] else ch[s]

        a, b, c, d, old = q(sa), q(sb), q(sc), ch[sp], ch[old_inner]
        for new in (VAC, TAU):
            # F^{-1} = F^T, so the inverse move reads the same entry transposed
            if direction == "right":
                coeff = f_symbol(a, b, c, d, old, new)
            else:
                coeff = f_symbol(a, b, c, d, new, old)
            if coeff:
                nch = dict(ch)
                del nch[old_inner]
                nch[new_inner] = new
                key = tuple(nch[s] for s in new_spans)
                out[key] = out.get(key, 0j) + amp * coeff
    return State.make(new_tree, out)


def braid(state: State, gap: int, ccw: bool) -> State:
    """Exchange anyons ``gap`` and ``gap + 1``, which must fuse directly."""
    sp = (gap, gap + 1)
    out: dict[Labels, complex] = {}
    for labels, amp in state.amps:
        r = R[state.charges(labels)[sp]]
        out[labels] = amp * (r if ccw else r.conjugate())
    return State.make(state.tree, out)
