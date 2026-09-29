"""Check majorana.py against exact diagonalisation of the spin model.

On a small periodic torus the full spin spectrum must equal the union, over
all gauge classes (fluxes w_p and the two Wilson loops), of the Majorana
many-body spectra restricted to the parity allowed by prod_j D_j = 1.

The spin Hamiltonian is built from Pauli matrices only:
    -J_alpha sigma^alpha_j sigma^alpha_k            on every alpha-link
    kappa term: u_jl u_lk c_j c_k = -(sigma^a_j sigma^a_l)(sigma^b_l sigma^b_k)

Run with:  python check_majorana.py
"""

from __future__ import annotations

import itertools
import random

import majorana as mj
from majorana import np


def pauli_string(ops, n: int):
    """prod(sigma^kind_site) for ops = [(site, kind), ...] as (rows, amplitudes):
    column c of the matrix has its one entry amplitudes[c] in row rows[c]."""
    idx = np.arange(2 ** n)
    amp = np.ones(2 ** n, dtype=complex)
    for site, kind in reversed(ops):
        bit = (idx >> site) & 1
        sgn = 1 - 2 * bit
        if kind == "z":
            amp = amp * sgn
        else:
            if kind == "y":
                amp = amp * 1j * sgn
            idx = idx ^ (1 << site)
    return idx, amp


def spin_hamiltonian(lat: mj.Lattice, J, kappa: float) -> np.ndarray:
    n = lat.N
    Jd = dict(zip("xyz", J))
    H = np.zeros((2 ** n, 2 ** n), dtype=complex)
    cols = np.arange(2 ** n)

    def add(coef, ops):
        rows, amp = pauli_string(ops, n)
        H[rows, cols] += coef * amp
    nbrs: list[list] = [[] for _ in range(n)]
    for bond in lat.bonds:
        add(-Jd[bond.kind], [(bond.a, bond.kind), (bond.b, bond.kind)])
        d = mj.DELTA[bond.kind]
        nbrs[bond.a].append((bond.b, bond.kind, d))
        nbrs[bond.b].append((bond.a, bond.kind, (-d[0], -d[1])))
    for l in range(n):
        for j, a, d_lj in nbrs[l]:
            for k, b, d_lk in nbrs[l]:
                if k == j:
                    continue
                cross = -d_lj[0] * d_lk[1] + d_lj[1] * d_lk[0]
                eps = 1.0 if cross > 0 else -1.0
                add(-(1j / 4) * 2 * kappa * eps, [(j, a), (l, a), (l, b), (k, b)])
    assert np.allclose(H, H.conj().T)
    return H


def majorana_spectrum(lat: mj.Lattice, J, kappa: float, flip: bool = False,
                      seed: int = 1) -> list[float]:
    """All physical energies; flip=True keeps the forbidden parity instead."""
    bc = ("periodic", "periodic")
    n_sectors = 2 ** (lat.L1 * lat.L2 + 1)
    rng = random.Random(seed)
    seen = {}
    while len(seen) < n_sectors:
        u = [rng.choice((1, -1)) for _ in lat.bonds]
        key = (tuple(lat.fluxes(u, bc)), lat.loops(u, bc))
        if key not in seen:
            seen[key] = u
    energies = []
    for u in seen.values():
        r = mj.solve(lat, u, J, kappa, bc)
        eps = list(r.eps)
        if r.parity_gs == 0:           # exact zero mode: keep it empty, no constraint
            for occ in itertools.product((0, 1), repeat=len(eps) - 1):
                energies.append(r.E0 + sum(e for e, o in zip(eps[1:], occ) if o))
            continue
        for occ in itertools.product((0, 1), repeat=len(eps)):
            parity = r.parity_gs * (-1) ** sum(occ)
            if parity == (-r.parity_req if flip else r.parity_req):
                energies.append(r.E0 + sum(e for e, o in zip(eps, occ) if o))
    return sorted(energies)


def check(L1: int, L2: int, J, kappa: float) -> None:
    lat = mj.Lattice(L1, L2)
    exact = np.linalg.eigvalsh(spin_hamiltonian(lat, J, kappa))
    fermions = np.array(majorana_spectrum(lat, J, kappa))
    assert len(exact) == len(fermions), (len(exact), len(fermions))
    err = np.max(np.abs(exact - fermions))
    # control: keeping the wrong parity in every sector must not match
    wrong = np.array(majorana_spectrum(lat, J, kappa, flip=True))
    err_wrong = np.max(np.abs(exact - wrong))
    ok = err < 1e-8 and err_wrong > 1e-3
    print(f"{L1}x{L2}  J={J}  kappa={kappa}:  {len(exact)} states,  "
          f"max |E_spin - E_majorana| = {err:.2e}  (wrong parity: {err_wrong:.2f})"
          f"  ->  {'OK' if ok else 'FAIL'}")


def check_bulk() -> None:
    """Isotropic, vortex-free, large torus: energy per spin and the gap 6 sqrt(3) kappa."""
    lat = mj.Lattice(30, 30)
    u = [1] * len(lat.bonds)
    r = mj.solve(lat, u, (1, 1, 1), 0.0, ("periodic", "periodic"))
    print(f"J = 1, kappa = 0, 30x30: E0 per spin = {r.E0 / lat.N:.5f}  (known: -0.78730)")
    kappa = 0.02
    r = mj.solve(lat, u, (1 / 3, 1 / 3, 1 / 3), kappa, ("periodic", "periodic"))
    print(f"J = 1/3, kappa = {kappa}: lowest eps = {r.eps[0]:.5f}, "
          f"6 sqrt(3) kappa = {6 * 3 ** 0.5 * kappa:.5f}")


def check_bands() -> None:
    """The Bloch matrix against the finite torus, its form, and the Chern numbers."""
    rng = np.random.default_rng(3)
    ok = True
    for _ in range(5):
        J = rng.random(3) + 0.05
        J = tuple(J / J.sum())
        kappa = float(rng.uniform(-0.2, 0.2))
        # the periodic L x L torus has exactly the momenta s_i = m_i / L
        L = 6
        lat = mj.Lattice(L, L)
        r = mj.solve(lat, [1] * len(lat.bonds), J, kappa, ("periodic", "periodic"))
        g = np.arange(L) / L
        S1, S2 = np.meshgrid(g, g, indexing="ij")
        e = mj.band_energy(J, kappa, S1, S2)
        ok &= np.allclose(np.sort(e.ravel()), r.eps)
        # form: no sigma_0 part, sigma_z mass = 4 kappa (sin a - sin b + sin(b - a)) up to sign
        s1, s2 = rng.random(50), rng.random(50)
        H = mj.bloch(J, kappa, s1, s2)
        a, b = 2 * np.pi * s1, 2 * np.pi * s2
        mass = 4 * kappa * (np.sin(a) - np.sin(b) + np.sin(b - a))
        dz = (H[:, 0, 0] - H[:, 1, 1]).real / 2
        ok &= np.allclose(H[:, 0, 0] + H[:, 1, 1], 0)
        ok &= np.allclose(dz, mass) or np.allclose(dz, -mass)
        f = 2 * (J[0] * np.exp(1j * a) + J[1] * np.exp(1j * b) + J[2])
        ok &= np.allclose(np.abs(H[:, 0, 1]), np.abs(f))
    print(f"Bloch matrix vs finite torus and Kitaev's form (no sigma_0, mass Delta(q)): "
          f"{'OK' if ok else 'FAIL'}")
    for J, name in (((1 / 3,) * 3, "B"), ((0.25, 0.3, 0.45), "B"), ((0.1, 0.2, 0.7), "A_z"),
                    ((0.6, 0.25, 0.15), "A_x")):
        cs = {k: mj.chern_lower(J, k) for k in (0.05, -0.05, 0.3)}
        print(f"  {name:3s} J = {tuple(round(j, 3) for j in J)}:  C(kappa = 0.05, -0.05, 0.3) = "
              f"{cs[0.05]}, {cs[-0.05]}, {cs[0.3]};  touch at kappa = 0: "
              f"{mj.bands_touch(J, 0.0)} (grid gap {mj.gap_min(J, 0.0, 240):.4f})")


if __name__ == "__main__":
    check_bands()
    check(2, 2, (0.5, 0.3, 0.2), 0.0)
    check(2, 2, (0.37, 0.41, 0.22), 0.13)
    check(3, 2, (0.45, 0.35, 0.2), 0.0)
    check(2, 3, (0.3, 0.3, 0.4), 0.07)
    check_bulk()
