"""Contents of the "?" popups. Equation, section and figure numbers refer to
A. Kitaev, Anyons in an exactly solved model and beyond, Ann. Phys. 321, 2
(2006), arXiv:cond-mat/0506438 (checked against the arXiv version)."""

from __future__ import annotations

import mathtext as mt

REFERENCE = ("A. Kitaev, Anyons in an exactly solved model and beyond, "
             "Ann. Phys. 321, 2 (2006); arXiv:cond-mat/0506438.")


def _bloch_matrix(size: int) -> mt.Box:
    """iA(q) = [[Delta, i f], [-i f*, -Delta]]"""
    lhs = mt.formula("i\\tilde{A}(q) = ", size)
    rows = [[mt.formula("Δ(q)", size), mt.formula("i f(q)", size)],
            [mt.formula("−i f(q)^*", size), mt.formula("−Δ(q)", size)]]
    return mt.hbox([lhs, mt.matrix(rows, size)])


HAMILTONIAN = [
    ("p", REFERENCE),
    ("h", "Spin model"),
    ("f", r"H = −\sum{α} J_α \sum{α\rm{-links}} σ^α_j σ^α_k~~~~\rm{Eq. (4)}"),
    ("p", "A magnetic field breaks time reversal (Eq. 45):"),
    ("f", r"V = −\sum{j} \paren{h_x σ^x_j + h_y σ^y_j + h_z σ^z_j}"),
    ("p", "At third order it produces a three-spin term (Eqs. 46–47, Sec. 6.2):"),
    ("f", r"−κ \sum{j,k,l} σ^x_j σ^y_k σ^z_l,~~κ ∼ \frac{h_x h_y h_z}{J^2}"),
    ("h", "Majorana form"),
    ("p", "Each spin becomes four Majoranas, and physical states obey D_j = 1 (Eq. 11):"),
    ("f", r"σ^α_j = i b^α_j c_j,~~D_j = b^x_j b^y_j b^z_j c_j,~~u_{jk} = i b^{α}_j b^{α}_k"),
    ("p", "The b's pair into the static gauge field u_jk = ±1 and the c's become free "
          "fermions (Eq. 13, Fig. 4). With the κ term this is Eq. (48):"),
    ("f", r"H = \frac{i}{4}\sum{j,k} A_{jk} c_j c_k,~~A_{jk} = 2J_{α} u_{jk},~~"
          r"2κ ε_{jlk} u_{jl} u_{lk}"),
    ("p", "Kitaev writes Eq. (48) for J_x = J_y = J_z = J and draws the κ term as arrows "
          "(the figure in Eq. 48); here the three J's are kept separate."),
    ("h", "The diagram"),
    ("p", "One unit cell (A: even site, B: odd site, joined by its z-link) and the "
          "neighbouring cells it couples to. The A site's three links reach B sites in "
          "the cells 0, n₁, n₂. Each sublattice has three independent second-neighbour "
          "terms, towards n₁, n₂ and n₁ − n₂. Arrows point along A_jk = +2κ in the gauge "
          "u = +1. All other couplings follow by translation and A_kj = −A_jk. This is "
          "the input of the Bloch matrix, Eqs. (29)–(32)."),
]

BANDS = [
    ("p", REFERENCE),
    ("h", "Spectrum of the vortex-free sector"),
    ("p", "With all u = +1 the problem is translation invariant, and for each momentum q "
          "it reduces to a 2 × 2 matrix, Eq. (49):"),
    ("m", _bloch_matrix),
    ("f", r"ε(q) = ±\paren{\abs{f(q)}^2 + Δ(q)^2}^{1/2}"),
    ("f", r"f(q) = 2\paren{\c{x}{J_x} e^{i q·n_1} + \c{y}{J_y} e^{i q·n_2} + \c{z}{J_z}}"),
    ("f", r"Δ(q) = 4κ\paren{\rm{sin}(q·n_1) − \rm{sin}(q·n_2) + \rm{sin}(q·(n_2 − n_1))}"),
    ("p", "f(q) is Eq. (32), with the lattice vectors n₁ and n₂; Δ(q) is Eq. (49). "
          "Eq. (49) is stated for equal J's; Δ(q) does not depend on them."),
    ("h", "Chern number"),
    ("p", "The gap closes only where f and Δ vanish together. Δ vanishes only on the lines "
          "q·n₁ = 0, q·n₂ = 0 and q·n₁ = q·n₂, and there f vanishes only on the phase "
          "boundaries J_α = J_β + J_γ (Eq. 33, Fig. 5). So the κ term leaves the phase "
          "diagram unchanged. It gaps phase B, whose lower band gets ν = sgn Δ = ±1 "
          "(Eqs. 50 and 55; Kitaev calls it B_ν), while ν = 0 in A_x, A_y, A_z. At the "
          "isotropic point the gap is Δ = 6√3 κ (Eq. 50)."),
    ("p", "Here ν is computed numerically from the Bloch eigenvectors on a 48 × 48 grid "
          "(Fukui–Hatsugai–Suzuki method). When the bands touch it is not defined."),
]

BOUNDARY = [
    ("p", "The cluster is small, so finite-size effects are strong: changing a boundary "
          "condition (periodic, antiperiodic, open) can visibly change the modes and "
          "their energies."),
    ("p", "In the topological phase (B with κ ≠ 0) an open seam reveals the chiral "
          "Majorana edge modes (Sec. 7 and Appendix B). In an infinite system there is a "
          "zero-energy edge mode. Here its energy is small but finite, because the modes "
          "on the two opposite edges overlap slightly, which splits them. Whether an edge "
          "mode near zero exists also depends on the other seam being periodic or "
          "antiperiodic, since that fixes the allowed momenta along the edge."),
]
