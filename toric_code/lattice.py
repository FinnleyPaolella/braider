"""Geometry and indexing for a periodic (torus) square lattice toric code.

Conventions (Kitaev standard):

    A_v = prod X  over the 4 edges touching vertex v   (star)
    B_p = prod Z  over the 4 edges bounding face p     (plaquette)

so Z-strings create ``e`` charges on vertices and X-strings create ``m`` fluxes
on faces.

Qubit indexing for an L x L torus (2 L^2 qubits total)::

    h(x, y) = y * L + x            joins (x, y) -> ((x+1) % L, y)
    v(x, y) = L*L + y * L + x      joins (x, y) -> (x, (y+1) % L)

Screen layout uses an (L+1) x (L+1) block of vertex slots: column L and row L
are *ghost* copies of column 0 and row 0. Every edge, wrap-around ones included,
is then an ordinary straight segment drawn exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EdgeGeom:
    """Where a qubit lives on screen, in lattice-slot coordinates."""

    qubit: int
    horizontal: bool
    # endpoints as (col, row) slot indices in the (L+1) x (L+1) drawing grid
    a: tuple[int, int]
    b: tuple[int, int]


class TorusLattice:
    """Indexing, incidence and screen placement for an L x L torus."""

    def __init__(self, size: int = 8) -> None:
        self.L = size
        self.n_vertices = size * size
        self.n_faces = size * size
        self.n_qubits = 2 * size * size
        self._build_incidence()
        self._build_geometry()

    # ------------------------------------------------------------------
    # indexing
    # ------------------------------------------------------------------
    def h(self, x: int, y: int) -> int:
        L = self.L
        return (y % L) * L + (x % L)

    def v(self, x: int, y: int) -> int:
        L = self.L
        return L * L + (y % L) * L + (x % L)

    def vertex(self, x: int, y: int) -> int:
        L = self.L
        return (y % L) * L + (x % L)

    def face(self, x: int, y: int) -> int:
        L = self.L
        return (y % L) * L + (x % L)

    def vertex_xy(self, idx: int) -> tuple[int, int]:
        return idx % self.L, idx // self.L

    def face_xy(self, idx: int) -> tuple[int, int]:
        return idx % self.L, idx // self.L

    def is_horizontal(self, qubit: int) -> bool:
        return qubit < self.L * self.L

    # ------------------------------------------------------------------
    # incidence
    # ------------------------------------------------------------------
    def vertex_edges(self, x: int, y: int) -> list[int]:
        """The four qubits touching vertex (x, y): E, W, N, S."""
        return [
            self.h(x, y),
            self.h(x - 1, y),
            self.v(x, y),
            self.v(x, y - 1),
        ]

    def face_edges(self, x: int, y: int) -> list[int]:
        """The four qubits bounding the face whose lower-left corner is (x, y)."""
        return [
            self.h(x, y),
            self.h(x, y + 1),
            self.v(x, y),
            self.v(x + 1, y),
        ]

    def _build_incidence(self) -> None:
        L = self.L
        # qubit -> the two vertices it joins, and the two faces it separates
        self.edge_vertices: list[tuple[int, int]] = [(0, 0)] * self.n_qubits
        self.edge_faces: list[tuple[int, int]] = [(0, 0)] * self.n_qubits
        for y in range(L):
            for x in range(L):
                q = self.h(x, y)
                self.edge_vertices[q] = (self.vertex(x, y), self.vertex(x + 1, y))
                # a horizontal edge separates the face below it from the face above
                self.edge_faces[q] = (self.face(x, y - 1), self.face(x, y))

                q = self.v(x, y)
                self.edge_vertices[q] = (self.vertex(x, y), self.vertex(x, y + 1))
                # a vertical edge separates the face to its left from the one right
                self.edge_faces[q] = (self.face(x - 1, y), self.face(x, y))

        self.vertex_incident: list[list[int]] = [
            self.vertex_edges(*self.vertex_xy(i)) for i in range(self.n_vertices)
        ]
        self.face_incident: list[list[int]] = [
            self.face_edges(*self.face_xy(i)) for i in range(self.n_faces)
        ]

    def edge_between_vertices(self, va: int, vb: int) -> int | None:
        """The qubit joining two adjacent vertices, or None if not adjacent."""
        for q in self.vertex_incident[va]:
            a, b = self.edge_vertices[q]
            if (a, b) == (va, vb) or (a, b) == (vb, va):
                return q
        return None

    def edge_between_faces(self, fa: int, fb: int) -> int | None:
        """The qubit shared by two adjacent faces, or None if not adjacent."""
        for q in self.face_incident[fa]:
            a, b = self.edge_faces[q]
            if (a, b) == (fa, fb) or (a, b) == (fb, fa):
                return q
        return None

    def vertex_faces(self, idx: int) -> list[int]:
        """The four faces having this vertex as a corner."""
        x, y = self.vertex_xy(idx)
        return [
            self.face(x, y),
            self.face(x - 1, y),
            self.face(x, y - 1),
            self.face(x - 1, y - 1),
        ]

    def face_vertices(self, idx: int) -> list[int]:
        """The four corners of this face."""
        x, y = self.face_xy(idx)
        return [
            self.vertex(x, y),
            self.vertex(x + 1, y),
            self.vertex(x, y + 1),
            self.vertex(x + 1, y + 1),
        ]

    def epsilon_moves(self, vertex: int, face: int) -> list[int]:
        """Qubits whose Y moves an ``e`` at ``vertex`` and an ``m`` at ``face``.

        Y = i X Z: the Z part hops the charge between the edge's two vertices
        and the X part hops the flux between its two faces, so a single Y moves
        both at once -- provided the edge touches the vertex *and* bounds the
        face. When the two sit on a bound pair there are exactly two such edges.
        """
        on_face = set(self.face_incident[face])
        return [q for q in self.vertex_incident[vertex] if q in on_face]

    def epsilon_step(self, qubit: int, vertex: int, face: int) -> tuple[int, int]:
        """Where the bound pair lands after Y on ``qubit``."""
        va, vb = self.edge_vertices[qubit]
        fa, fb = self.edge_faces[qubit]
        return (vb if va == vertex else va), (fb if fa == face else fa)

    def vertex_neighbours(self, idx: int) -> list[int]:
        x, y = self.vertex_xy(idx)
        return [
            self.vertex(x + 1, y),
            self.vertex(x - 1, y),
            self.vertex(x, y + 1),
            self.vertex(x, y - 1),
        ]

    def face_neighbours(self, idx: int) -> list[int]:
        x, y = self.face_xy(idx)
        return [
            self.face(x + 1, y),
            self.face(x - 1, y),
            self.face(x, y + 1),
            self.face(x, y - 1),
        ]

    # ------------------------------------------------------------------
    # geometry (slot coordinates; pixels are computed by the view)
    # ------------------------------------------------------------------
    def _build_geometry(self) -> None:
        L = self.L
        self.edge_geom: list[EdgeGeom] = [None] * self.n_qubits  # type: ignore[list-item]
        for y in range(L):
            for x in range(L):
                q = self.h(x, y)
                self.edge_geom[q] = EdgeGeom(q, True, (x, y), (x + 1, y))
                q = self.v(x, y)
                self.edge_geom[q] = EdgeGeom(q, False, (x, y), (x, y + 1))

    def vertex_slots(self, idx: int) -> list[tuple[int, int]]:
        """Every drawing slot showing this vertex (1, 2 or 4 with ghosts)."""
        L = self.L
        x, y = self.vertex_xy(idx)
        xs = [x] + ([L] if x == 0 else [])
        ys = [y] + ([L] if y == 0 else [])
        return [(sx, sy) for sx in xs for sy in ys]

    def face_slots(self, idx: int) -> list[tuple[int, int]]:
        """Lower-left slot(s) of every drawn copy of this face."""
        L = self.L
        x, y = self.face_xy(idx)
        xs = [x] + ([L] if x == 0 else [])
        ys = [y] + ([L] if y == 0 else [])
        # a face drawn with lower-left at slot (L, *) would fall outside the grid
        return [(sx, sy) for sx in xs for sy in ys if sx < L and sy < L]

    # ------------------------------------------------------------------
    # syndrome
    # ------------------------------------------------------------------
    def syndrome(self, net: dict[int, str]) -> tuple[set[int], set[int]]:
        """(e charges on vertices, m fluxes on faces) for a net Pauli assignment.

        A vertex carries ``e`` when an odd number of its edges anticommute with
        ``A_v = prod X``, i.e. carry a Z component. A face carries ``m`` when an
        odd number of its edges anticommute with ``B_p = prod Z``, i.e. carry an
        X component.
        """
        e_sites: set[int] = set()
        m_sites: set[int] = set()
        for i, edges in enumerate(self.vertex_incident):
            if sum(net.get(q, "I") in ("Z", "Y") for q in edges) % 2:
                e_sites.add(i)
        for i, edges in enumerate(self.face_incident):
            if sum(net.get(q, "I") in ("X", "Y") for q in edges) % 2:
                m_sites.add(i)
        return e_sites, m_sites
