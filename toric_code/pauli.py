"""The operator word applied to |00_L>.

A factor is a single-qubit ``X``, ``Y`` or ``Z``. Ordering: ``word[0]`` is the
leftmost factor and is applied *last*; the rightmost factor acts first on
``|00_L>``. Applying a gate therefore prepends.

The global phase starts at +1 and only moves when the simplifier does something
to earn it: a sign from commuting two factors that anticommute, or the ``i`` in
``Y = i X Z`` when a Y has to be split so a loop can take its X part. Keeping
phase out of plain editing is what makes "erase this qubit" well defined.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count

# how the letters multiply: LEFT * RIGHT -> (phase, letter)
_MUL: dict[tuple[str, str], tuple[complex, str]] = {
    ("I", "I"): (1, "I"), ("I", "X"): (1, "X"), ("I", "Y"): (1, "Y"), ("I", "Z"): (1, "Z"),
    ("X", "I"): (1, "X"), ("X", "X"): (1, "I"), ("X", "Y"): (1j, "Z"), ("X", "Z"): (-1j, "Y"),
    ("Y", "I"): (1, "Y"), ("Y", "X"): (-1j, "Z"), ("Y", "Y"): (1, "I"), ("Y", "Z"): (1j, "X"),
    ("Z", "I"): (1, "Z"), ("Z", "X"): (1j, "Y"), ("Z", "Y"): (-1j, "X"), ("Z", "Z"): (1, "I"),
}

_ids = count(1)


@dataclass(frozen=True)
class Factor:
    """One single-qubit Pauli in the word.

    ``uid`` is a stable identity so the view can animate a particular factor
    while its index in the word changes.
    """

    qubit: int
    letter: str  # 'X', 'Y' or 'Z'
    uid: int = field(default_factory=lambda: next(_ids), compare=False)

    def label(self) -> str:
        return f"{self.letter}_{self.qubit + 1}"


def anticommute(a: Factor, b: Factor) -> bool:
    """Two single-qubit Paulis anticommute iff same qubit, different letters."""
    return a.qubit == b.qubit and a.letter != b.letter


def phase_prefix(phase: complex) -> str:
    """Render a Pauli-group phase, empty for +1."""
    if phase == 1:
        return ""
    if phase == -1:
        return "-"
    if phase == 1j:
        return "i"
    if phase == -1j:
        return "-i"
    return f"({phase})"


class PauliWord:
    """An ordered product of single-qubit X/Z factors times a global phase."""

    def __init__(self) -> None:
        self.factors: list[Factor] = []
        self.phase: complex = 1

    # ------------------------------------------------------------------
    # editing
    # ------------------------------------------------------------------
    def clone(self) -> "PauliWord":
        other = PauliWord()
        other.factors = list(self.factors)
        other.phase = self.phase
        return other

    def apply_gate(self, qubit: int, gate: str) -> None:
        """Apply X, Y or Z to a qubit (prepended: it acts last)."""
        if gate not in ("X", "Y", "Z"):
            raise ValueError(f"unknown gate {gate!r}")
        self.factors.insert(0, Factor(qubit, gate))

    def erase_qubit(self, qubit: int) -> None:
        """Drop every factor on a qubit.

        No phase correction is needed: editing never puts a phase on a qubit.
        """
        self.factors = [f for f in self.factors if f.qubit != qubit]

    def clear(self) -> None:
        self.factors.clear()
        self.phase = 1

    # ------------------------------------------------------------------
    # evaluation
    # ------------------------------------------------------------------
    def net(self) -> dict[int, str]:
        """Net non-identity Pauli letter per qubit."""
        per: dict[int, list[str]] = {}
        for f in self.factors:
            per.setdefault(f.qubit, []).append(f.letter)
        out: dict[int, str] = {}
        for qubit, letters in per.items():
            _, letter = _reduce_letters(letters)
            if letter != "I":
                out[qubit] = letter
        return out

    def net_phase(self) -> complex:
        """Total phase of the word once each qubit's factors are multiplied out."""
        per: dict[int, list[str]] = {}
        for f in self.factors:
            per.setdefault(f.qubit, []).append(f.letter)
        ph = self.phase
        for letters in per.values():
            local, _ = _reduce_letters(letters)
            ph *= local
        return _snap(ph)

    # ------------------------------------------------------------------
    # display
    # ------------------------------------------------------------------
    def tokens(self) -> list[str]:
        """The literal word, left to right, without the ket."""
        return [f.label() for f in self.factors]

    def text(self, ket: str = "|00_L>") -> str:
        parts = [phase_prefix(self.phase)] if phase_prefix(self.phase) else []
        parts += self.tokens()
        parts.append(ket)
        return " ".join(parts)


def _reduce_letters(letters) -> tuple[complex, str]:
    """Multiply a left-to-right sequence of Pauli letters."""
    phase: complex = 1
    acc = "I"
    for letter in letters:
        ph, acc = _MUL[(acc, letter)]
        phase *= ph
    return _snap(phase), acc


def _snap(z: complex) -> complex:
    """Round a Pauli phase back onto {1, -1, i, -i} after float arithmetic."""
    best = min((1, -1, 1j, -1j), key=lambda c: abs(c - z))
    return best
