"""Find braids of three Fibonacci anyons that approximate single-qubit gates.

Run with:  python compile_gates.py        (writes gates.py next to this file)

The qubit is the total-charge-tau sector of three anyons, |0> = ((t t)_1 t)_tau
and |1> = ((t t)_tau t)_tau. On it the exchanges act as

    s1 = diag(R_1, R_tau)     (anyons 1 and 2, counterclockwise)
    s2 = F s1 F               (anyons 2 and 3, counterclockwise)

Only Z (= s1^5) is exact; X, Y, H, S and T can only be approximated. The
search is meet-in-the-middle: all braids of up to HALF syllables s_i^k are
stored as points on the unit-quaternion sphere (phase removed), and for each
braid A the grid is searched for a B with A B close to the target. Among the
braids within TOLERANCE the one with the fewest exchanges wins.

Pure Python on purpose: it needs no numpy. It takes about ten minutes.
"""

from __future__ import annotations

import cmath
import math
import os
import time
from collections import defaultdict

from anyons import F, R, TAU, VAC

HALF = 5                   # syllables per half
TOLERANCE = 5e-3           # accept braids with at most this error
EXPONENTS = [k for k in range(-4, 6) if k != 0]   # s^10 is a pure phase
CELL = 0.06                # grid spacing on the quaternion sphere

Mat = tuple[complex, complex, complex, complex]   # row-major 2x2
Word = list[tuple[int, int]]                      # [(generator, exponent)]


def mul(a: Mat, b: Mat) -> Mat:
    return (a[0] * b[0] + a[1] * b[2], a[0] * b[1] + a[1] * b[3],
            a[2] * b[0] + a[3] * b[2], a[2] * b[1] + a[3] * b[3])


def dag(a: Mat) -> Mat:
    return (a[0].conjugate(), a[2].conjugate(), a[1].conjugate(), a[3].conjugate())


def power(m: Mat, k: int) -> Mat:
    out: Mat = (1, 0, 0, 1)
    for _ in range(abs(k)):
        out = mul(out, m if k > 0 else dag(m))
    return out


FM: Mat = (F[0][0], F[0][1], F[1][0], F[1][1])
S1: Mat = (R[VAC], 0, 0, R[TAU])
S2: Mat = mul(mul(FM, S1), FM)
PIECES = {(g, k): power(S1 if g == 1 else S2, k) for g in (1, 2) for k in EXPONENTS}

R2 = 1 / math.sqrt(2)
TARGETS: dict[str, Mat] = {
    "X": (0, 1, 1, 0),
    "Y": (0, -1j, 1j, 0),
    "Z": (1, 0, 0, -1),
    "H": (R2, R2, R2, -R2),
    "S": (1, 0, 0, 1j),
    "T": (1, 0, 0, cmath.exp(1j * math.pi / 4)),
}


def quaternion(u: Mat) -> tuple[float, float, float, float]:
    """``u`` up to phase as a unit quaternion, sign fixed (q ~ -q)."""
    ph = cmath.sqrt(u[0] * u[3] - u[1] * u[2])
    a, b = u[0] / ph, u[1] / ph
    q = (a.real, a.imag, b.real, b.imag)
    for x in q:
        if abs(x) > 1e-12:
            return q if x > 0 else tuple(-y for y in q)
    return q


def error(target: Mat, u: Mat) -> float:
    """Phase-invariant distance sqrt(1 - |tr(T^dag U)| / 2)."""
    td = dag(target)
    tr = td[0] * u[0] + td[1] * u[2] + td[2] * u[1] + td[3] * u[3]
    return math.sqrt(max(0.0, 1 - abs(tr) / 2))


def exchanges(word: Word) -> int:
    return sum(abs(k) for _, k in word)


def all_words(max_syllables: int) -> list[tuple[Mat, Word]]:
    out: list[tuple[Mat, Word]] = [((1, 0, 0, 1), [])]

    def grow(u: Mat, word: Word, last: int) -> None:
        if len(word) == max_syllables:
            return
        for g in (1, 2):
            if g == last:
                continue
            for k in EXPONENTS:
                v = mul(u, PIECES[(g, k)])
                w = word + [(g, k)]
                out.append((v, w))
                grow(v, w, g)

    grow((1, 0, 0, 1), [], 0)
    return out


def cell(q) -> tuple[int, ...]:
    return tuple(math.floor(x / CELL) for x in q)


def neighbours(c):
    for d0 in (-1, 0, 1):
        for d1 in (-1, 0, 1):
            for d2 in (-1, 0, 1):
                for d3 in (-1, 0, 1):
                    yield (c[0] + d0, c[1] + d1, c[2] + d2, c[3] + d3)


def simplify(word: Word) -> Word:
    """Merge neighbouring powers of the same generator, exponents mod 10."""
    out: Word = []
    for g, k in word:
        if out and out[-1][0] == g:
            k += out.pop()[1]
        k = (k + 4) % 10 - 4          # into -4..5
        if k:
            out.append((g, k))
    return out


def compile_all() -> dict[str, tuple[Word, float]]:
    t0 = time.time()
    words = all_words(HALF)
    grid: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for i, (u, _) in enumerate(words):
        grid[cell(quaternion(u))].append(i)
    print(f"{len(words)} half-braids in {time.time() - t0:.0f}s")

    found: dict[str, tuple[Word, float]] = {}
    for name, target in TARGETS.items():
        best: tuple[int, float, Word] | None = None
        for a, wa in words:
            # want a b = target, so b ~ a^dag target
            want = quaternion(mul(dag(a), target))
            for q in (want, tuple(-x for x in want)):
                for c in neighbours(cell(q)):
                    for j in grid.get(c, ()):
                        b, wb = words[j]
                        n = exchanges(wa) + exchanges(wb)
                        if best is not None and n > best[0]:
                            continue
                        e = error(target, mul(a, b))
                        if e > TOLERANCE:
                            continue
                        w = simplify(wa + wb)
                        n = exchanges(w)
                        if best is None or (n, e) < (best[0], best[1]):
                            best = (n, e, w)
        assert best is not None, f"nothing within {TOLERANCE} for {name}"
        n, e, w = best
        # check the merged word independently
        u: Mat = (1, 0, 0, 1)
        for piece in w:
            u = mul(u, PIECES[piece])
        e = error(target, u)
        found[name] = (w, e)
        print(f"{name}: {n} exchanges, error {e:.1e}  ({time.time() - t0:.0f}s)")
    return found


def write(found: dict[str, tuple[Word, float]]) -> None:
    lines = [
        '"""Braids approximating single-qubit gates on three Fibonacci anyons.',
        "",
        "Generated by compile_gates.py; see there for the conventions. Each word is",
        "a list of (generator, exponent) read as the matrix product s_g1^k1 s_g2^k2 ...,",
        "so the last entry is performed first. Generator 1 exchanges anyons 1 and 2,",
        "generator 2 anyons 2 and 3; positive exponents are counterclockwise.",
        "The error is sqrt(1 - |tr(G^dag U)| / 2), ignoring the global phase.",
        '"""',
        "",
        "GATES = {",
    ]
    for name, (w, e) in found.items():
        lines.append(f'    "{name}": ({w!r}, {e:.3e}),')
    lines.append("}")
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gates.py")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    found = compile_all()
    # Z is exact: s1^5 (a shorter word than any the search needs)
    z = [(1, 5)]
    found["Z"] = min(found["Z"], (z, error(TARGETS["Z"], PIECES[(1, 5)])),
                     key=lambda we: (exchanges(we[0]), we[1]))
    write(found)


