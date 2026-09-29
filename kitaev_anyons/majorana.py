"""Majorana fermions c hopping in a static Z2 gauge field u (no pygame).

This is Kitaev, Ann. Phys. 321, 2 (2006), Eq. (47)/(48): the honeycomb model
with the magnetic field replaced by its third-order term, written with
sigma^alpha_j = i b^alpha_j c_j and u_jk = i b^alpha_j b^alpha_k as

    H = (i/4) sum_{j,k} A_jk c_j c_k,

    A_jk = 2 J_alpha u_jk                         (j, k linked by an alpha-link)
    A_jk = 2 kappa sum_l eps_jlk u_jl u_lk        (j, k next-nearest neighbours)

with eps_jlk = +1 when the path j -> l -> k turns left, -1 when it turns right.
Both terms are gauge covariant and A is real antisymmetric. u_jk = -u_kj; a
link stores u = u_jk with j on the even sublattice A, k on the odd one B.

iA is Hermitian with eigenvalues +-eps_m, and H = sum_m eps_m (n_m - 1/2).
So the fermionic ground state energy in a fixed u sector is

    E_0 = -1/2 sum_m eps_m = 1/2 * (sum of the negative eigenvalues of iA).

Physical states obey D_j = b^x_j b^y_j b^z_j c_j = 1 on every site. On a
closed torus the product of all D_j reduces to (sign) * prod(u) * P_c, so in
each u sector only one c-fermion parity P_c is allowed. If the fermionic ground
state has the other parity, the lowest physical state has the lowest mode
filled. With an open boundary the dangling b's absorb the parity and there is
no constraint.

Geometry: bond length 1, y up, z-links vertical;
from an A site the x-link points up-right, the y-link up-left, the z-link down.
The cluster is L1 x L2 unit cells R = i n1 + j n2. x-links from the last column
(i = L1 - 1) cross seam 1, y-links from the last row (j = L2 - 1) cross seam 2.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

# MKL's Intel-OpenMP threading layer crashes this environment (0xC06D007F) on
# any BLAS/LAPACK call; its TBB layer works. Must be set before numpy loads
# (and pygame loads numpy, so main.py sets it too).
os.environ.setdefault("MKL_THREADING_LAYER", "TBB")

import numpy as np  # noqa: E402

SQ3 = math.sqrt(3.0)
DELTA = {"x": (SQ3 / 2, 0.5), "y": (-SQ3 / 2, 0.5), "z": (0.0, -1.0)}   # A -> B
N1 = (SQ3 / 2, 1.5)
N2 = (-SQ3 / 2, 1.5)
KINDS = "xyz"

BCS = ("periodic", "antiperiodic", "open")
ZERO_TOL = 1e-9      # |eps| below this counts as an exact zero mode


@dataclass(frozen=True)
class Bond:
    a: int            # site on sublattice A
    b: int            # site on sublattice B
    kind: str         # "x", "y" or "z"
    seam: int         # 0: inside the cluster, 1 or 2: crosses that seam


@dataclass(frozen=True)
class Plaquette:
    cell: tuple[int, int]
    sites: tuple[int, ...]       # six sites, counter-clockwise from the bottom
    bonds: tuple[int, ...]       # the six links between consecutive sites
    center: tuple[float, float]
    seams: frozenset             # seams this hexagon straddles


class Lattice:
    """An L1 x L2 honeycomb cluster; which links exist depends on the bc."""

    def __init__(self, L1: int, L2: int) -> None:
        self.L1, self.L2 = L1, L2
        self.N = 2 * L1 * L2
        self.pos: list[tuple[float, float]] = [None] * self.N
        for i in range(L1):
            for j in range(L2):
                R = (i * N1[0] + j * N2[0], i * N1[1] + j * N2[1])
                self.pos[self.A(i, j)] = R
                self.pos[self.B(i, j)] = (R[0], R[1] - 1.0)

        self.bonds: list[Bond] = []
        for i in range(L1):
            for j in range(L2):
                a = self.A(i, j)
                self.bonds.append(Bond(a, self.B((i + 1) % L1, j), "x", 1 if i == L1 - 1 else 0))
                self.bonds.append(Bond(a, self.B(i, (j + 1) % L2), "y", 2 if j == L2 - 1 else 0))
                self.bonds.append(Bond(a, self.B(i, j), "z", 0))

        self.plaquettes: list[Plaquette] = []
        for i in range(L1):
            for j in range(L2):
                i1, j1 = (i + 1) % L1, (j + 1) % L2
                sites = (self.A(i, j), self.B(i1, j), self.A(i1, j),
                         self.B(i1, j1), self.A(i, j1), self.B(i, j1))
                bonds = (self.bond(i, j, "x"), self.bond(i1, j, "z"), self.bond(i1, j, "y"),
                         self.bond(i, j1, "x"), self.bond(i, j1, "z"), self.bond(i, j, "y"))
                ax, ay = self.pos[self.A(i, j)]
                seams = frozenset(s for s, hit in ((1, i == L1 - 1), (2, j == L2 - 1)) if hit)
                self.plaquettes.append(Plaquette((i, j), sites, bonds, (ax, ay + 1.0), seams))

    def A(self, i: int, j: int) -> int:
        return 2 * (i * self.L2 + j)

    def B(self, i: int, j: int) -> int:
        return 2 * (i * self.L2 + j) + 1

    def bond(self, i: int, j: int, kind: str) -> int:
        return 3 * (i * self.L2 + j) + KINDS.index(kind)

    # ------------------------------------------------------------------
    def active(self, bond: Bond, bc) -> bool:
        return not (bond.seam and bc[bond.seam - 1] == "open")

    def u_eff(self, u, bc) -> list[int]:
        """The link variables the fermions see: antiperiodic seams add a sign,
        open seams remove the link (0)."""
        out = []
        for bond, ub in zip(self.bonds, u):
            if not bond.seam:
                out.append(ub)
            elif bc[bond.seam - 1] == "open":
                out.append(0)
            else:
                out.append(-ub if bc[bond.seam - 1] == "antiperiodic" else ub)
        return out

    def plaquette_closed(self, p: Plaquette, bc) -> bool:
        return all(bc[s - 1] != "open" for s in p.seams)

    def fluxes(self, u, bc) -> list[int | None]:
        """w_p = prod of u_jk (j in A) around each hexagon; None if cut open."""
        ue = self.u_eff(u, bc)
        out = []
        for p in self.plaquettes:
            if not self.plaquette_closed(p, bc):
                out.append(None)
                continue
            w = 1
            for b in p.bonds:
                w *= ue[b]
            out.append(w)
        return out

    def loops(self, u, bc) -> tuple[int | None, int | None]:
        """Wilson loops around the torus along n1 (x, z links) and n2 (y, z)."""
        ue = self.u_eff(u, bc)
        w1 = w2 = 1
        for i in range(self.L1):
            w1 *= ue[self.bond(i, 0, "x")] * ue[self.bond((i + 1) % self.L1, 0, "z")]
        for j in range(self.L2):
            w2 *= ue[self.bond(0, j, "y")] * ue[self.bond(0, (j + 1) % self.L2, "z")]
        return (w1 if bc[0] != "open" else None, w2 if bc[1] != "open" else None)


# ----------------------------------------------------------------------
# the Majorana matrix
# ----------------------------------------------------------------------
def matrix(lat: Lattice, u, J, kappa: float, bc) -> np.ndarray:
    """The real antisymmetric N x N matrix A_jk."""
    ue = lat.u_eff(u, bc)
    Jd = dict(zip(KINDS, J))
    A = np.zeros((lat.N, lat.N))
    # neighbours of each site: (other site, u_{site,other}, vector site -> other)
    nbrs: list[list] = [[] for _ in range(lat.N)]
    for bond, ub in zip(lat.bonds, ue):
        if ub == 0:
            continue
        A[bond.a, bond.b] += 2 * Jd[bond.kind] * ub
        A[bond.b, bond.a] -= 2 * Jd[bond.kind] * ub
        d = DELTA[bond.kind]
        nbrs[bond.a].append((bond.b, ub, d))
        nbrs[bond.b].append((bond.a, -ub, (-d[0], -d[1])))
    if kappa:
        for l in range(lat.N):
            for j, u_lj, d_lj in nbrs[l]:
                for k, u_lk, d_lk in nbrs[l]:
                    if k == j:
                        continue
                    # j -> l is -d_lj, l -> k is d_lk; eps = sign of their cross product
                    cross = -d_lj[0] * d_lk[1] + d_lj[1] * d_lk[0]
                    eps = 1.0 if cross > 0 else -1.0
                    A[j, k] += 2 * kappa * eps * (-u_lj) * u_lk
    return A


def pfaffian(M: np.ndarray) -> float:
    """Pfaffian of a real antisymmetric matrix (Parlett-Reid elimination)."""
    M = np.array(M, dtype=float)
    n = M.shape[0]
    if n % 2:
        return 0.0
    pf = 1.0
    for k in range(0, n - 1, 2):
        kp = k + 1 + int(np.argmax(np.abs(M[k + 1:, k])))
        if kp != k + 1:
            M[[k + 1, kp], :] = M[[kp, k + 1], :]
            M[:, [k + 1, kp]] = M[:, [kp, k + 1]]
            pf = -pf
        if M[k + 1, k] == 0.0:
            return 0.0
        pf *= M[k, k + 1]
        if k + 2 < n:
            tau = M[k, k + 2:] / M[k, k + 1]
            col = M[k + 2:, k + 1]
            M[k + 2:, k + 2:] += np.outer(tau, col) - np.outer(col, tau)
    return pf


def _perm_sign(seq) -> int:
    """Sign of the permutation that sorts ``seq`` (a permutation of 0..n-1)."""
    seen = [False] * len(seq)
    sign = 1
    for start in range(len(seq)):
        if seen[start]:
            continue
        length = 0
        k = start
        while not seen[k]:
            seen[k] = True
            k = seq[k]
            length += 1
        if length % 2 == 0:
            sign = -sign
    return sign


def required_parity(lat: Lattice, u, bc) -> int | None:
    """The c-fermion parity P_c = prod_m (-i c_2m c_2m+1) that prod_j D_j = 1
    allows, or None when a seam is open (no constraint)."""
    if "open" in bc:
        return None
    ue = lat.u_eff(u, bc)
    # prod_j D_j = prod_j (b^x_j b^y_j b^z_j c_j); Majorana 4j + {0,1,2,3}.
    # Reorder into (b_a b_b) for every link, then c_0 c_1 ... c_{N-1}.
    target = []
    for bond in lat.bonds:
        k = KINDS.index(bond.kind)
        target += [4 * bond.a + k, 4 * bond.b + k]
    target += [4 * j + 3 for j in range(lat.N)]
    factor = complex(_perm_sign(target))
    for ub in ue:                               # b_a b_b = -i u_ab
        factor *= -1j * ub
    factor *= 1j ** (lat.N // 2)                # c_0 ... c_{N-1} = i^{N/2} P_c
    # prod D = factor * P_c must be 1, so P_c = 1 / factor = conj(factor)
    p = factor.conjugate()
    assert abs(p.imag) < 1e-9 and abs(abs(p.real) - 1) < 1e-9, p
    return round(p.real)


# ----------------------------------------------------------------------
@dataclass
class Result:
    eps: np.ndarray            # single-particle energies eps_m >= 0, ascending
    E0: float                  # fermionic ground state energy -1/2 sum eps
    parity_gs: int             # P_c of the fermionic ground state (0: zero mode)
    parity_req: int | None     # P_c demanded by the projection (None: open)
    E_phys: float              # lowest energy of a physical state in this sector
    modes: np.ndarray          # column m: eigenvector of iA with eigenvalue +eps_m

    @property
    def parity_ok(self) -> bool:
        return self.parity_req is None or self.parity_gs in (0, self.parity_req)


def solve(lat: Lattice, u, J, kappa: float, bc) -> Result:
    A = matrix(lat, u, J, kappa, bc)
    w, V = np.linalg.eigh(1j * A)
    eps = w[lat.N // 2:].copy()                  # the +eps half, ascending
    E0 = -0.5 * float(eps.sum())
    if eps[0] < ZERO_TOL:
        parity_gs = 0
    else:
        # ground-state correlations <-i c_j c_k> = -i sign(iA)_jk; P_c = Pf of that
        sgn = (V * np.sign(w)) @ V.conj().T
        G = np.real(-1j * sgn)
        parity_gs = 1 if pfaffian(G) > 0 else -1
    parity_req = required_parity(lat, u, bc)
    E_phys = E0
    if parity_req is not None and parity_gs not in (0, parity_req):
        E_phys = E0 + float(eps[0])
    return Result(eps, E0, parity_gs, parity_req, E_phys, V[:, lat.N // 2:])


# ----------------------------------------------------------------------
# the translation-invariant (vortex-free, u = +1) sector in momentum space
# ----------------------------------------------------------------------
# A Bloch wave c_(R, s) = e^{i q.R} psi_s, with q = s1 q1 + s2 q2 and q_i the
# basis dual to (n1, n2), so that q.(i n1 + j n2) = 2 pi (s1 i + s2 j). The
# terms of the 2x2 Bloch matrix are read off the real-space matrix itself, so
# the bands use exactly the conventions of ``matrix``.

def bloch_terms(J, kappa: float) -> list[tuple[int, int, int, int, complex]]:
    """(s, s', di, dj, amplitude): H_ss'(q) = sum amplitude e^{2 pi i (s1 di + s2 dj)}."""
    L = 4
    lat = Lattice(L, L)
    A = matrix(lat, [1] * len(lat.bonds), J, kappa, ("periodic", "periodic"))
    terms = []
    for s in (0, 1):
        row = s                       # the site of sublattice s in cell (0, 0)
        for k in np.nonzero(A[row])[0]:
            cell, s2 = divmod(int(k), 2)
            i, j = divmod(cell, L)
            di = i if i <= L // 2 else i - L
            dj = j if j <= L // 2 else j - L
            terms.append((s, s2, di, dj, 1j * A[row, k]))
    return terms


def bloch(J, kappa: float, s1, s2) -> np.ndarray:
    """The Bloch matrices iA(q) for arrays of reduced momenta, shape (..., 2, 2)."""
    s1, s2 = np.asarray(s1, float), np.asarray(s2, float)
    H = np.zeros(s1.shape + (2, 2), dtype=complex)
    for s, t, di, dj, amp in bloch_terms(J, kappa):
        H[..., s, t] += amp * np.exp(2j * np.pi * (s1 * di + s2 * dj))
    return H


def band_energy(J, kappa: float, s1, s2) -> np.ndarray:
    """The upper band eps(q) >= 0 (the lower one is -eps(q))."""
    H = bloch(J, kappa, s1, s2)
    d0 = (H[..., 0, 0] + H[..., 1, 1]).real / 2
    dz = (H[..., 0, 0] - H[..., 1, 1]).real / 2
    return np.sqrt(dz ** 2 + np.abs(H[..., 0, 1]) ** 2) + d0


def bands_touch(J, kappa: float, tol: float = 1e-9) -> bool:
    """Whether the gap closes somewhere in the Brillouin zone.

    The kappa term is a mass Delta(q) sigma_z with Delta = 0 only on the lines
    q.n1 = 0, q.n2 = 0, q.n1 = q.n2 (mod 2 pi); f(q) vanishes there only on the
    phase boundaries J_a = J_b + J_c. So: kappa = 0 in (the closure of) phase B,
    or any kappa on a phase boundary. check_majorana.py confirms this form.
    """
    js = sorted(J)
    excess = js[2] - js[0] - js[1]          # > 0 in an A phase
    return abs(excess) < tol or (kappa == 0 and excess < 0)


def chern_lower(J, kappa: float, n: int = 48) -> int:
    """Chern number of the lower band (Fukui-Hatsugai-Suzuki on an n x n grid)."""
    g = np.arange(n) / n
    S1, S2 = np.meshgrid(g, g, indexing="ij")
    _, V = np.linalg.eigh(bloch(J, kappa, S1, S2))
    u = V[..., :, 0]                                  # lower band, shape (n, n, 2)
    def link(a, b):
        z = np.sum(a.conj() * b, axis=-1)
        return z / np.abs(z)
    u1 = np.roll(u, -1, axis=0)
    u2 = np.roll(u, -1, axis=1)
    u12 = np.roll(u1, -1, axis=1)
    F = np.angle(link(u, u1) * link(u1, u12) * link(u12, u2) * link(u2, u))
    return round(float(F.sum()) / (2 * np.pi))


def gap_min(J, kappa: float, n: int = 90) -> float:
    """min over q of the band gap 2 eps(q), on an n x n grid."""
    g = np.arange(n) / n
    S1, S2 = np.meshgrid(g, g, indexing="ij")
    return 2 * float(band_energy(J, kappa, S1, S2).min())
