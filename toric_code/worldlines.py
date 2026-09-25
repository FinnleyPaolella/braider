"""Tracking anyon worldlines through the moves made so far.

The syndrome alone only says *which* sites are excited, not which particle is
which, so identity has to be derived from the operations themselves. Every edit
is local: applying a Pauli to qubit ``q`` flips the e-syndrome at its two
endpoint vertices (if the gate has a Z component) and the m-syndrome at its two
adjacent faces (if it has an X component). For each flipped pair of sites there
are exactly three possibilities:

    neither occupied  ->  a pair is created
    one occupied      ->  that particle hops to the other site
    both occupied     ->  the two annihilate

which is all that is needed to follow particles over time. Simplify and Reduce
never change the syndrome, so they contribute no worldline events.

Time is measured in moves: one unit per operation that changes the syndrome.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Particle:
    pid: int
    kind: str                       # 'e' or 'm'
    birth: int                      # move index it appeared at
    path: list[tuple[int, int]]     # (move index, site) waypoints
    death: int | None = None

    @property
    def alive(self) -> bool:
        return self.death is None


@dataclass
class Link:
    """The turn joining two worldlines where a pair is born or annihilates."""

    kind: str
    t: int
    a: int          # site of the first partner
    b: int          # site of the second partner
    creation: bool  # True = pair creation, False = annihilation
    pa: int = 0     # the two particles the turn joins
    pb: int = 0


@dataclass
class _Step:
    """One undoable edit: the sub-events it produced, newest last."""

    subs: list = field(default_factory=list)
    advanced: bool = False
    snapshot: dict | None = None    # only for a full reset (Clear)


class WorldlineTracker:
    def __init__(self, lat) -> None:
        self.lat = lat
        self.reset()

    def reset(self) -> None:
        self.version = getattr(self, "version", 0) + 1
        self.t = 0
        self.occ: dict[str, dict[int, int]] = {"e": {}, "m": {}}
        self.particles: dict[int, Particle] = {}
        self.links: list[Link] = []
        self.steps: list[_Step] = []
        self._next_pid = 1

    # ------------------------------------------------------------------
    # recording
    # ------------------------------------------------------------------
    def begin_step(self) -> None:
        self.steps.append(_Step())

    def begin_reset_step(self) -> None:
        """Record a step that wipes the history but can still be undone."""
        step = _Step(snapshot=self._snapshot())
        self.reset_keep_steps()
        self.steps.append(step)

    def _snapshot(self) -> dict:
        return {
            "t": self.t,
            "occ": {k: dict(v) for k, v in self.occ.items()},
            "particles": {
                pid: Particle(p.pid, p.kind, p.birth, list(p.path), p.death)
                for pid, p in self.particles.items()
            },
            "links": list(self.links),
            "next_pid": self._next_pid,
        }

    def _restore(self, snap: dict) -> None:
        self.t = snap["t"]
        self.occ = {k: dict(v) for k, v in snap["occ"].items()}
        self.particles = {
            pid: Particle(p.pid, p.kind, p.birth, list(p.path), p.death)
            for pid, p in snap["particles"].items()
        }
        self.links = list(snap["links"])
        self._next_pid = snap["next_pid"]

    def reset_keep_steps(self) -> None:
        steps = self.steps
        self.reset()
        self.steps = steps

    def flip(self, qubit: int, do_e: bool, do_m: bool) -> None:
        """Record the syndrome flips caused by touching a qubit."""
        if not (do_e or do_m):
            return
        self.version += 1
        if not self.steps:
            self.begin_step()
        step = self.steps[-1]
        self.t += 1
        step.advanced = True
        if do_e:
            a, b = self.lat.edge_vertices[qubit]
            self._pair(step, "e", a, b)
        if do_m:
            a, b = self.lat.edge_faces[qubit]
            self._pair(step, "m", a, b)

    def _pair(self, step: _Step, kind: str, a: int, b: int) -> None:
        occ = self.occ[kind]
        pa, pb = occ.get(a), occ.get(b)
        if pa is None and pb is None:
            pids = []
            for site in (a, b):
                pid = self._next_pid
                self._next_pid += 1
                self.particles[pid] = Particle(pid, kind, self.t, [(self.t, site)])
                occ[site] = pid
                pids.append(pid)
            self.links.append(Link(kind, self.t, a, b, True, pids[0], pids[1]))
            step.subs.append(("create", kind, a, b, pids[0], pids[1]))
        elif pa is not None and pb is None:
            self._hop(step, kind, pa, a, b)
        elif pb is not None and pa is None:
            self._hop(step, kind, pb, b, a)
        else:
            for site, pid in ((a, pa), (b, pb)):
                self.particles[pid].path.append((self.t, site))
                self.particles[pid].death = self.t
                del occ[site]
            self.links.append(Link(kind, self.t, a, b, False, pa, pb))
            step.subs.append(("annihilate", kind, a, b, pa, pb))

    def _hop(self, step: _Step, kind: str, pid: int, src: int, dst: int) -> None:
        occ = self.occ[kind]
        del occ[src]
        occ[dst] = pid
        self.particles[pid].path.append((self.t, dst))
        step.subs.append(("hop", kind, src, dst, pid, None))

    # ------------------------------------------------------------------
    # undo
    # ------------------------------------------------------------------
    def undo_step(self) -> None:
        if not self.steps:
            return
        self.version += 1
        step = self.steps.pop()
        if step.snapshot is not None:
            self._restore(step.snapshot)
            return
        for sub in reversed(step.subs):
            what, kind, a, b, p1, p2 = sub
            occ = self.occ[kind]
            if what == "create":
                if self.links:
                    self.links.pop()
                for site, pid in ((a, p1), (b, p2)):
                    occ.pop(site, None)
                    self.particles.pop(pid, None)
            elif what == "hop":
                occ.pop(b, None)
                occ[a] = p1
                self.particles[p1].path.pop()
            else:  # annihilate
                if self.links:
                    self.links.pop()
                for site, pid in ((a, p1), (b, p2)):
                    part = self.particles[pid]
                    part.death = None
                    part.path.pop()
                    occ[site] = pid
        if step.advanced:
            self.t -= 1

    # ------------------------------------------------------------------
    # queries
    # ------------------------------------------------------------------
    def site_xy(self, kind: str, site: int) -> tuple[float, float]:
        """Lattice coordinates of a site: vertices on the grid, faces centred."""
        if kind == "e":
            x, y = self.lat.vertex_xy(site)
            return float(x), float(y)
        x, y = self.lat.face_xy(site)
        return x + 0.5, y + 0.5

    def counts(self) -> tuple[int, int, int]:
        """(live e, live m, total particles ever created)."""
        return len(self.occ["e"]), len(self.occ["m"]), len(self.particles)

    def is_empty(self) -> bool:
        return not self.particles
