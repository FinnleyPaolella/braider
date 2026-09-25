"""Turning an exchange (or a return to the standard basis) into explained steps.

Each step shows one or more frames of the state display, joined by
transitions, and names the rule it uses so the reference panels can light up.
How finely the work is split depends on the detail level:

    INSTANT   no steps, just the final state (back in the starting basis)
    EXPLAIN   one step for the whole change of basis, one for the braid, whose
              phases are written out and then absorbed on their own
    DETAILED  per F-move one step marking the three parts it regroups and one
              applying it; the braid phases are absorbed in a step of their own
"""

from __future__ import annotations

from dataclasses import dataclass, field

from anyons import (
    LABEL, PHASE_TEXT, TAU, VAC, Move, Span, State, braid, f_move, invert,
    move_parts, moves_to_comb, moves_to_pair,
)

INSTANT, EXPLAIN, DETAILED = 0, 1, 2

# highlight roles; render.py maps them to colours. "sector" colours a box by
# its own charge, matching the phase that charge picks up in a braid.
A, B, C, PAIR, SECTOR, VAC_ROLE, TAU_ROLE = "a", "b", "c", "pair", "sector", "vac", "tau"
Highlight = tuple[tuple[Span, str], ...]

# how long a frame is held before its transition starts (main.py sets seconds)
NOW, AFTER_SWAP, AFTER_PHASES = "now", "swap", "phases"

SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def anyon(i: int) -> str:
    return "τ" + str(i + 1).translate(SUB)


def part(sp: Span) -> str:
    """A leaf or a fused group, e.g. τ₃ or (τ₁…τ₃)."""
    lo, hi = sp
    if lo == hi:
        return anyon(lo)
    if hi == lo + 1:
        return f"({anyon(lo)} {anyon(hi)})"
    return f"({anyon(lo)}…{anyon(hi)})"


@dataclass(frozen=True)
class Display:
    """One frame of the state display. ``phases`` = (gap, ccw) writes the
    braid phase of the pair at ``gap`` in front of every term."""

    state: State
    hl: Highlight = ()
    phases: tuple[int, bool] | None = None


@dataclass
class Step:
    title: str                          # short name for the step list
    headline: list[tuple[str, str]]     # (text, highlight role or "")
    detail: str
    frames: list[Display]
    # between consecutive frames: ("fade" | "absorb", hold before it starts)
    transitions: list[tuple[str, str]] = field(default_factory=list)
    extra: list[tuple[str, str]] = field(default_factory=list)  # a coloured line
    swap: tuple[int, int] | None = None  # (gap, side) if the anyons move
    rule: tuple | None = None            # ("F",) or ("R", ccw): panel to light up

    @property
    def state(self) -> State:
        return self.frames[-1].state


@dataclass
class Task:
    """A requested exchange (``gap``, ``side``) or, with gap None, a return
    to the standard basis."""

    gap: int | None = None
    side: int = 0

    @property
    def ccw(self) -> bool:
        # the lower arrow sends the left anyon counterclockwise around the right
        return self.side < 0


def pair_highlight(gap: int) -> Highlight:
    return (((gap, gap + 1), SECTOR), ((gap, gap), PAIR), ((gap + 1, gap + 1), PAIR))


def amplitudes(state: State) -> list[tuple[float, float]]:
    return sorted((round(v.real, 9), round(v.imag, 9)) for _, v in state.amps)


def move_headline(sa: Span, sb: Span, sc: Span, direction: str) -> list[tuple[str, str]]:
    grouped_left = [("(", ""), (part(sa), A), (" ", ""), (part(sb), B), (") ", ""),
                    (part(sc), C)]
    grouped_right = [(part(sa), A), (" (", ""), (part(sb), B), (" ", ""),
                     (part(sc), C), (")", "")]
    arrow = [("   →   ", "")]
    if direction == "right":
        return grouped_left + arrow + grouped_right
    return grouped_right + arrow + grouped_left


def f_steps(state: State, moves: list[Move], back: bool) -> tuple[list[Step], State]:
    """Two steps per F-move: mark the three parts, then regroup them."""
    steps = []
    for k, move in enumerate(moves):
        sa, sb, sc = move_parts(state.tree, move)
        after = f_move(state, move)
        hl = ((sa, A), (sb, B), (sc, C))
        headline = move_headline(sa, sb, sc, move[1])
        count = f" {k + 1}/{len(moves)}" if len(moves) > 1 else ""
        name = "F-move back" if back else "F-move"
        steps.append(Step(
            f"Mark {name}{count}", headline,
            "The coloured parts are what the next F-move regroups. Their charges "
            "and the charge around them select the entries of the F matrix.",
            [Display(state, hl)], rule=("F",),
        ))
        mixes = amplitudes(state) != amplitudes(after)
        steps.append(Step(
            f"Apply {name}{count}", headline,
            "Regrouped with the F matrix: terms where all four charges are τ mix."
            if mixes else
            "Regrouped; with a vacuum charge involved the F-move only relabels "
            "the terms.",
            [Display(state, hl), Display(after, hl)], [("fade", NOW)], rule=("F",),
        ))
        state = after
    return steps, state


def combined_step(state: State, moves: list[Move], title: str, detail: str,
                  after_hl: Highlight = ()) -> tuple[list[Step], State]:
    """All ``moves`` as a single step."""
    if not moves:
        return [], state
    after = state
    for m in moves:
        after = f_move(after, m)
    return [Step(title, [(title, "")], detail,
                 [Display(state), Display(after, after_hl)], [("fade", NOW)])], after


def plural(k: int, word: str) -> str:
    return f"{k} {word}" + ("" if k == 1 else "s")


def braid_steps(state: State, task: Task, level: int) -> tuple[list[Step], State]:
    """Move the anyons, write out the phases, absorb them."""
    g, ccw = task.gap, task.ccw
    hl = pair_highlight(g)
    after = braid(state, g, ccw)
    who = f"{anyon(g)} and {anyon(g + 1)}"
    way = "counterclockwise" if ccw else "clockwise"
    rule = ("R", ccw)
    extra = [(f"{LABEL[VAC]}  →  {PHASE_TEXT[(VAC, ccw)]}", VAC_ROLE), ("        ", ""),
             (f"{LABEL[TAU]}  →  {PHASE_TEXT[(TAU, ccw)]}", TAU_ROLE)]
    plain = Display(state, hl)
    written = Display(state, hl, (g, ccw))
    done = Display(after, hl)
    exchange = [(f"Exchange {who} {way}", "")]
    if level == DETAILED:
        return [
            Step("Braid", exchange,
                 "Each term picks up the phase for the charge of the exchanged pair.",
                 [plain, written], [("fade", AFTER_SWAP)], extra,
                 swap=(g, task.side), rule=rule),
            Step("Absorb phases", [("Absorb the phases", "")],
                 "Every coefficient is multiplied by the phase in front of it.",
                 [written, done], [("absorb", NOW)], extra, rule=rule),
        ], after
    return [
        Step("Braid", exchange,
             "Each term picks up the phase for the charge of the exchanged pair, "
             "which is then absorbed into its coefficient.",
             [plain, written, done], [("fade", AFTER_SWAP), ("absorb", AFTER_PHASES)],
             extra, swap=(g, task.side), rule=rule),
    ], after


def plan(state: State, task: Task, level: int,
         always_standard: bool) -> tuple[list[Step], State]:
    """The steps for ``task`` and the final state."""
    steps: list[Step] = []

    if task.gap is not None:
        g = task.gap
        moves = moves_to_pair(state.tree, g)
        if level == DETAILED:
            s, state = f_steps(state, moves, back=False)
        else:
            s, state = combined_step(
                state, moves, "Change basis",
                f"{plural(len(moves), 'F-move')} so that {anyon(g)} and {anyon(g + 1)} "
                "fuse directly, giving their pair a definite charge.",
                pair_highlight(g),
            )
        steps += s

        s, state = braid_steps(state, task, level)
        steps += s

        if level == INSTANT and not always_standard:
            # quietly back to the basis we started in
            for m in reversed(moves):
                state = f_move(state, invert(m))
            return [], state

    if always_standard or task.gap is None:
        back = moves_to_comb(state.tree)
        if level == DETAILED:
            s, state = f_steps(state, back, back=True)
        else:
            s, state = combined_step(
                state, back, "Back to standard basis",
                f"{plural(len(back), 'F-move')} regroup everything as "
                "(((τ₁ τ₂) τ₃) …) again.",
            )
        steps += s

    if level == INSTANT:
        return [], state
    return steps, state
