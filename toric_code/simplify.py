"""Finding contractible loops and scripting the animation that removes them.

An X-type stabilizer is a product of stars, ``prod_{v in C} A_v``, whose support
is the *edge boundary* of the vertex set ``C`` (edges with exactly one endpoint
in ``C``). So a subset of the current X-support is a stabilizer exactly when it
is such a boundary. Delete the supported edges from the lattice graph: every
remaining edge then has both endpoints in one component, so any ``C`` whose
boundary lies inside the support must be a union of those components. Reading
off component boundaries therefore enumerates candidate loops directly, and it
can only ever produce genuine products of generators -- a non-contractible
logical loop is rejected automatically, because cutting a torus along one
homologically nontrivial cycle leaves it connected.

Z-type loops are the same computation on the dual graph (faces as nodes).

Non-contractible loops -- the logical operators -- are found separately, as the
shortest cycle in the support whose homology class is nonzero. The class is read
off by counting, mod 2, how often the cycle crosses a fixed representative of
each conjugate logical operator. The logical qubits are

    (Z_1)_L = Z on the horizontal edges of row 0      (a primal loop along x)
    (X_1)_L = X on the horizontal edges of column 0   (a dual loop along y)
    (Z_2)_L = Z on the vertical edges of column 0     (a primal loop along y)
    (X_2)_L = X on the vertical edges of row 0        (a dual loop along x)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from lattice import TorusLattice
from pauli import Factor, PauliWord, _reduce_letters, _snap, anticommute, ket_label


# ----------------------------------------------------------------------
# loop detection
# ----------------------------------------------------------------------
def _components(n_nodes: int, adjacency: list[list[int]]) -> list[list[int]]:
    """Connected components given node -> [neighbour, ...]."""
    seen = [False] * n_nodes
    comps: list[list[int]] = []
    for start in range(n_nodes):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        comp = []
        while stack:
            node = stack.pop()
            comp.append(node)
            for nbr in adjacency[node]:
                if not seen[nbr]:
                    seen[nbr] = True
                    stack.append(nbr)
        comps.append(comp)
    return comps


def _smallest_boundary(
    n_nodes: int,
    incidence: list[tuple[int, int]],
    support: set[int],
    node_edges: list[list[int]],
) -> set[int] | None:
    """Smallest nonempty component boundary contained in ``support``.

    ``incidence[q]`` gives the two nodes qubit ``q`` joins; ``node_edges[n]``
    lists the qubits at node ``n``.
    """
    adjacency: list[list[int]] = [[] for _ in range(n_nodes)]
    for node in range(n_nodes):
        for q in node_edges[node]:
            if q in support:
                continue  # edge removed: it is a candidate loop edge
            a, b = incidence[q]
            adjacency[node].append(b if a == node else a)

    best: set[int] | None = None
    for comp in _components(n_nodes, adjacency):
        members = set(comp)
        boundary = {
            q
            for node in comp
            for q in node_edges[node]
            if (incidence[q][0] in members) != (incidence[q][1] in members)
        }
        if boundary and (best is None or len(boundary) < len(best)):
            best = boundary
    return best


def find_x_loop(lat: TorusLattice, net: dict[int, str]) -> set[int] | None:
    """A contractible loop of X (a product of star stabilizers), if any."""
    support = {q for q, letter in net.items() if letter in ("X", "Y")}
    if not support:
        return None
    return _smallest_boundary(
        lat.n_vertices, lat.edge_vertices, support, lat.vertex_incident
    )


def find_z_loop(lat: TorusLattice, net: dict[int, str]) -> set[int] | None:
    """A contractible loop of Z (a product of plaquette stabilizers), if any."""
    support = {q for q, letter in net.items() if letter in ("Z", "Y")}
    if not support:
        return None
    return _smallest_boundary(lat.n_faces, lat.edge_faces, support, lat.face_incident)


def _shortest_logical_cycle(
    incidence: list[tuple[int, int]],
    support: set[int],
    node_edges: list[list[int]],
    cuts: tuple[set[int], set[int]],
) -> tuple[set[int], tuple[int, int]] | None:
    """Shortest homologically nontrivial cycle inside ``support``.

    Breadth-first search on the four-sheeted cover of the support graph: the
    sheet records the parity of crossings with each of the two ``cuts``, and a
    walk from ``(s, 0)`` back to ``s`` on another sheet is a nontrivial cycle.
    The shortest such walk is a simple cycle. Returns ``(qubits, (c1, c2))``.
    """
    def sheet(q: int) -> int:
        return (q in cuts[0]) | ((q in cuts[1]) << 1)

    best: set[int] | None = None
    for start in sorted({n for q in support for n in incidence[q]}):
        prev: dict[tuple[int, int], tuple[int, int, int] | None] = {(start, 0): None}
        queue = deque([(start, 0)])
        end = None
        while queue and end is None:
            node, c = queue.popleft()
            for q in node_edges[node]:
                if q not in support:
                    continue
                a, b = incidence[q]
                nxt = (b if a == node else a, c ^ sheet(q))
                if nxt in prev:
                    continue
                prev[nxt] = (node, c, q)
                if nxt[0] == start and nxt[1]:
                    end = nxt
                    break
                queue.append(nxt)
        if end is None:
            continue
        cycle: set[int] = set()
        state = end
        while prev[state] is not None:
            node, c, q = prev[state]
            cycle ^= {q}
            state = (node, c)
        if best is None or len(cycle) < len(best):
            best = cycle
    if best is None:
        return None
    return best, (len(best & cuts[0]) % 2, len(best & cuts[1]) % 2)


def find_logical_x_loop(
    lat: TorusLattice, net: dict[int, str]
) -> tuple[set[int], tuple[int, int]] | None:
    """A non-contractible X-loop (on the dual lattice) and its logical class."""
    support = {q for q, letter in net.items() if letter in ("X", "Y")}
    if not support:
        return None
    L = lat.L
    # crossings with (Z_1)_L and (Z_2)_L
    cuts = ({lat.h(x, 0) for x in range(L)}, {lat.v(0, y) for y in range(L)})
    return _shortest_logical_cycle(lat.edge_faces, support, lat.face_incident, cuts)


def find_logical_z_loop(
    lat: TorusLattice, net: dict[int, str]
) -> tuple[set[int], tuple[int, int]] | None:
    """A non-contractible Z-loop (on the primal lattice) and its logical class."""
    support = {q for q, letter in net.items() if letter in ("Z", "Y")}
    if not support:
        return None
    L = lat.L
    # crossings with (X_1)_L and (X_2)_L
    cuts = ({lat.h(0, y) for y in range(L)}, {lat.v(x, 0) for x in range(L)})
    return _shortest_logical_cycle(lat.edge_vertices, support, lat.vertex_incident, cuts)


def find_pair(word: PauliWord) -> tuple[Factor, Factor] | None:
    """Two factors with the same qubit and letter, i.e. a plain cancellation."""
    seen: dict[tuple[int, str], Factor] = {}
    for f in word.factors:
        key = (f.qubit, f.letter)
        if key in seen:
            return seen[key], f
        seen[key] = f
    return None


# ----------------------------------------------------------------------
# animation script
# ----------------------------------------------------------------------
@dataclass
class Segment:
    """A further ``= ...`` link of the equation, drawn after the main one."""

    phase: complex
    factors: list[Factor]
    ket: tuple[int, int]
    ops: list[str] = field(default_factory=list)  # logical operators, e.g. "(X_1)_L"


@dataclass
class Snapshot:
    """The complete rendered state at one step of the animation."""

    factors: list[Factor]
    phase: complex
    caption: str
    highlight: set[int] = field(default_factory=set)   # uids of the loop factors
    flash: set[int] = field(default_factory=set)       # uids that just anticommuted
    moving: int | None = None                          # uid gliding on this step
    bracket: tuple[int, int] | None = None             # inclusive index range
    loop_qubits: set[int] = field(default_factory=set)
    ket: tuple[int, int] = (0, 0)
    chain: list[Segment] = field(default_factory=list)  # later links, if any


@dataclass
class Script:
    snapshots: list[Snapshot]
    loop_qubits: set[int]
    kind: str  # 'X', 'Z', 'XL', 'ZL' (logical loops), 'pair' or 'reduce'


NOTHING = "Nothing to simplify: no closed loop or repeated factor in the word."


def build_script(lat: TorusLattice, word: PauliWord) -> tuple["Script | None", str]:
    """Look for something to simplify and script its removal.

    Returns ``(script, message)``; ``script`` is None when nothing was found.
    """
    net = word.net()

    loop = find_x_loop(lat, net)
    letter, kind = "X", "X"
    if loop is None:
        loop = find_z_loop(lat, net)
        letter, kind = "Z", "Z"
    logical: list[int] = []  # which logical qubits a non-contractible loop hits
    if loop is None:
        for letter, finder in (("X", find_logical_x_loop), ("Z", find_logical_z_loop)):
            hit = finder(lat, net)
            if hit is not None:
                loop, cls = hit
                logical = [i + 1 for i in (0, 1) if cls[i]]
                kind = letter + "L"
                break

    if loop is None:
        pair = find_pair(word)
        if pair is None:
            return None, NOTHING
        chosen = [pair[0], pair[1]]
        loop_qubits = {pair[0].qubit}
        kind = "pair"
        letter = pair[0].letter
        opening = (
            f"Repeated factor: {pair[0].label()} appears twice, and "
            f"{letter}{letter} = I."
        )
    else:
        loop_qubits = set(loop)
        if logical:
            opening = (
                f"Non-contractible {letter}-loop on {len(loop)} qubits: it winds "
                f"around the torus, so it is a logical operator."
            )
        else:
            gens = "star" if kind == "X" else "plaquette"
            opening = (
                f"Contractible {letter}-loop on {len(loop)} qubits "
                f"-- a product of {gens} stabilizers."
            )

    factors = list(word.factors)
    phase = word.phase
    snaps: list[Snapshot] = []

    if kind != "pair":
        # For each loop qubit take the rightmost factor carrying that letter --
        # it has the shortest distance to travel. If the qubit only supplies its
        # letter through a Y, split that Y first: Y = i X Z.
        chosen = []
        splits: list[tuple[int, Factor, Factor, Factor]] = []
        for qubit in sorted(loop):
            direct = [f for f in factors if f.qubit == qubit and f.letter == letter]
            if direct:
                chosen.append(direct[-1])
                continue
            ys = [f for f in factors if f.qubit == qubit and f.letter == "Y"]
            if not ys:
                return None, NOTHING
            y = ys[-1]
            xf, zf = Factor(qubit, "X"), Factor(qubit, "Z")
            splits.append((_index_of(factors, y.uid), y, xf, zf))
            chosen.append(xf if letter == "X" else zf)

        snaps.append(
            Snapshot(
                factors=list(factors),
                phase=phase,
                caption=opening,
                highlight={f.uid for f in chosen if f.uid in {g.uid for g in factors}}
                | {y.uid for _, y, _, _ in splits},
                loop_qubits=set(loop_qubits),
            )
        )
        # apply the splits, newest index first so earlier indices stay valid
        for idx, y, xf, zf in sorted(splits, key=lambda s: s[0], reverse=True):
            factors[idx:idx + 1] = [xf, zf]
            phase = _snap(phase * 1j)
            snaps.append(
                Snapshot(
                    factors=list(factors),
                    phase=phase,
                    caption=(
                        f"{y.label()} = i {xf.label()} {zf.label()}: split it so the "
                        f"loop can take its {letter} part."
                    ),
                    highlight={f.uid for f in chosen},
                    flash={xf.uid, zf.uid},
                    loop_qubits=set(loop_qubits),
                )
            )

    uids = {f.uid for f in chosen}
    if not snaps:
        snaps.append(
            Snapshot(
                factors=list(factors),
                phase=phase,
                caption=opening,
                highlight=set(uids),
                loop_qubits=set(loop_qubits),
            )
        )

    # slide the loop factors to the right end, rightmost one first
    order = sorted(chosen, key=lambda f: _index_of(factors, f.uid), reverse=True)
    for k, f in enumerate(order):
        old = _index_of(factors, f.uid)
        target = len(factors) - 1 - k
        factors.pop(old)
        passed = factors[old:target]
        flips = [p for p in passed if anticommute(p, f)]
        factors.insert(target, f)
        phase = _snap(phase * (-1) ** len(flips))
        if not flips:
            # only passes factors on other qubits: shown together with the
            # next step that matters
            continue

        plural = "s" if len(passed) != 1 else ""
        names = ", ".join(sorted({p.label() for p in flips}))
        caption = (
            f"Commute {f.label()} right past {len(passed)} factor{plural}: "
            f"it anticommutes with {names}, giving {len(flips)} sign flip"
            f"{'s' if len(flips) != 1 else ''} -- phase now {_phase_name(phase)}."
        )
        snaps.append(
            Snapshot(
                factors=list(factors),
                phase=phase,
                caption=caption,
                highlight=set(uids),
                flash={p.uid for p in flips},
                moving=f.uid,
                loop_qubits=set(loop_qubits),
            )
        )

    lo = len(factors) - len(order)
    if logical:
        snaps += _logical_ending(
            factors, phase, word.ket, uids, letter, logical, lo, set(loop_qubits)
        )
        for snap in snaps:
            snap.ket = word.ket
        return Script(snaps, set(loop_qubits), kind), opening
    if kind == "pair":
        gather = f"The two {letter}_{chosen[0].qubit + 1} factors now sit together."
        closing = f"{letter}{letter} = I, so the pair vanishes."
    else:
        gens = "A" if kind == "X" else "B"
        gather = (
            f"All {len(order)} factors now sit in front of the ket and form "
            f"the stabilizer."
        )
        closing = (
            f"{gens}{ket_label(word.ket)} = {ket_label(word.ket)}: the stabilizer "
            f"acts trivially, so it drops out."
        )
    snaps.append(
        Snapshot(
            factors=list(factors),
            phase=phase,
            caption=gather,
            highlight=set(uids),
            bracket=(lo, len(factors) - 1),
            loop_qubits=set(loop_qubits),
        )
    )
    snaps.append(
        Snapshot(
            factors=[f for f in factors if f.uid not in uids],
            phase=phase,
            caption=closing,
            loop_qubits=set(loop_qubits),
        )
    )
    for snap in snaps:
        snap.ket = word.ket
    return Script(snaps, set(loop_qubits), kind), opening


def _logical_ending(
    factors: list[Factor],
    phase: complex,
    ket: tuple[int, int],
    uids: set[int],
    letter: str,
    logical: list[int],
    lo: int,
    loop_qubits: set[int],
) -> list[Snapshot]:
    """The last three steps once a logical loop sits right in front of the ket.

    The gathered loop is bracketed; then it is rewritten as the standard logical
    operator it equals up to stabilizers; then that operator acts on the ket.
    """
    rest = [f for f in factors if f.uid not in uids]
    ops = [f"({letter}_{i})_L" for i in logical]
    ops_text = " ".join(ops)
    n = len(factors) - lo
    bracket = (lo, len(factors) - 1)

    as_op = Segment(phase, rest, ket, ops)
    if letter == "X":
        new_ket = tuple(b ^ (i + 1 in logical) for i, b in enumerate(ket))
        new_phase = phase
        effect = (
            f"{ops_text} flips logical qubit{'s' if len(ops) > 1 else ''} "
            f"{' and '.join(map(str, logical))}: "
            f"{ket_label(ket)} → {ket_label(new_ket)}."
        )
    else:
        new_ket = ket
        sign = (-1) ** sum(ket[i - 1] for i in logical)
        new_phase = _snap(phase * sign)
        bits = " and ".join(f"qubit {i} is {ket[i - 1]}" for i in logical)
        effect = (
            f"{ops_text} {ket_label(ket)} = {'-' if sign < 0 else ''}{ket_label(ket)}"
            f": logical {bits}, so {'a sign' if sign < 0 else 'no sign'}."
        )
    acted = Segment(new_phase, rest, new_ket)

    common = dict(highlight=set(uids), bracket=bracket, loop_qubits=loop_qubits)
    return [
        Snapshot(
            factors=list(factors), phase=phase,
            caption=f"All {n} factors of the loop now sit in front of the ket.",
            **common,
        ),
        Snapshot(
            factors=list(factors), phase=phase,
            caption=(
                f"Up to stabilizers the loop equals {ops_text}, "
                f"which winds the same way round the torus."
            ),
            chain=[as_op], **common,
        ),
        Snapshot(
            factors=list(factors), phase=phase, caption=effect,
            chain=[as_op, acted], **common,
        ),
    ]


REDUCE_NOTHING = "Nothing to reduce: every qubit already carries a single operator."


def build_reduce_script(word: PauliWord) -> tuple["Script | None", str]:
    """Script the same-qubit reduction of every qubit carrying several factors.

    Gathering a qubit's factors costs no signs: on the way they only pass
    factors on *other* qubits, which commute freely. The phase appears when the
    gathered factors are finally multiplied out -- ``X_q Z_q = -i Y_q``,
    ``X_q X_q = I``, and so on.
    """
    factors = list(word.factors)
    phase = word.phase

    counts: dict[int, int] = {}
    first: dict[int, int] = {}
    for i, f in enumerate(factors):
        counts[f.qubit] = counts.get(f.qubit, 0) + 1
        first.setdefault(f.qubit, i)
    targets = sorted((q for q, c in counts.items() if c >= 2), key=lambda q: first[q])
    if not targets:
        return None, REDUCE_NOTHING

    snaps: list[Snapshot] = []
    for q in targets:
        idxs = [i for i, f in enumerate(factors) if f.qubit == q]
        group = [factors[i] for i in idxs]
        uids = {f.uid for f in group}
        shown = " ".join(g.label() for g in group)
        snaps.append(
            Snapshot(
                factors=list(factors),
                phase=phase,
                caption=f"Qubit {q + 1} carries {len(group)} operators: {shown}.",
                highlight=set(uids),
                loop_qubits={q},
            )
        )

        # slide them together at the leftmost one; everything crossed on the way
        # sits on another qubit, so nothing picks up a sign and the slide gets
        # no step of its own
        dest = idxs[0]
        moved_any = idxs != list(range(dest, dest + len(group)))
        for k, g in enumerate(group[1:], start=1):
            old = _index_of(factors, g.uid)
            factors.insert(dest + k, factors.pop(old))

        local, letter = _reduce_letters(g.letter for g in group)
        gathered = "Gathered (other qubits commute, no sign): " if moved_any else ""
        snaps.append(
            Snapshot(
                factors=list(factors),
                phase=phase,
                caption=f"{gathered}{shown} = {_term(local, letter, q)}.",
                highlight=set(uids),
                bracket=(dest, dest + len(group) - 1),
                loop_qubits={q},
            )
        )

        phase = _snap(phase * local)
        merged = Factor(q, letter) if letter != "I" else None
        factors = (
            factors[:dest]
            + ([merged] if merged else [])
            + factors[dest + len(group):]
        )
        snaps.append(
            Snapshot(
                factors=list(factors),
                phase=phase,
                caption=(
                    f"Qubit {q + 1} is now a single {letter}."
                    if merged
                    else f"The operators on qubit {q + 1} cancel."
                ),
                highlight={merged.uid} if merged else set(),
                loop_qubits={q},
            )
        )

    n = len(targets)
    return (
        Script(snaps, set(targets), "reduce"),
        f"Reducing {n} qubit{'s' if n != 1 else ''} that carry more than one operator.",
    )


def _term(phase: complex, letter: str, qubit: int) -> str:
    """Render a phase times a single-qubit Pauli, e.g. ``-i Y_12`` or ``I``."""
    sym = "I" if letter == "I" else f"{letter}_{qubit + 1}"
    prefix = {1: "", -1: "-", 1j: "i ", -1j: "-i "}.get(phase, f"({phase}) ")
    return prefix + sym


def _index_of(factors: list[Factor], uid: int) -> int:
    for i, f in enumerate(factors):
        if f.uid == uid:
            return i
    raise KeyError(uid)


def _phase_name(phase: complex) -> str:
    return {1: "+1", -1: "-1", 1j: "+i", -1j: "-i"}.get(phase, str(phase))
