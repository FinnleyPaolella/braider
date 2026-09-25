"""Interactive braiding of Fibonacci anyons.

Run with:  python main.py

Click an arrow above or below the gap between two neighbouring anyons to
exchange them along that semicircle. The braid diagram underneath records the
history, with time running from bottom to top. On the right, the state is
written in a fusion-tree basis and updated by every exchange: first F-moves
until the two anyons fuse directly, then the braid phase R.

The detail slider sets how much of that is shown: instantly, as a few
explained steps, or in detail: every F-move first marks the three parts it
regroups (coloured alike in the F-matrix panel), then applies them, and the
braid phases are written in front of the terms, coloured by fusion channel
as in the braid-matrix panel, before being absorbed into the coefficients.

Over/under convention: the particle plane is seen at a slant, so a particle
passing along the *upper* arc moves behind its partner and its strand goes
*under* in the braid diagram. The lower arrow is a counterclockwise exchange.

With three anyons the total-charge-tau sector is a qubit, |0> = ((t t)_1 t)_tau
and |1> = ((t t)_tau t)_tau, and the gate buttons play braids that implement
X, Y, Z, H, S and T: Z exactly, the others approximately (see gates.py).

Keys:  space  play / pause   right  next step   R  reset   Esc  quit
"""

from __future__ import annotations

import asyncio
import math
import sys
from collections import deque
from dataclasses import dataclass, field

import pygame
import pygame.gfxdraw

import render
import ui
import webcanvas
from anyons import basis, initial_state
from gates import GATES
from steps import (
    AFTER_PHASES, AFTER_SWAP, EXPLAIN, INSTANT, Display, Step, Task,
    anyon, plan,
)

N = 3  # number of particles

# ----------------------------------------------------------------------
# layout
# ----------------------------------------------------------------------
SPACING = 140          # distance between neighbouring particles
PARTICLE_R = 16
ROW_Y = 180            # y of the particle row
LEFT_MARGIN = 80       # empty space left of the time axis
PANEL_PAD = 100        # horizontal padding around the row inside the panel
PANEL_MIN_W = 460
MID_W = 720            # preferred width of the state / explanation column
WINDOW_H = 1060
MARGIN = 40            # around the window's right and bottom edges
PANEL_GAP = 20         # between the panels on the right
CARD_H = 260           # the step-by-step explanation card
CONTROLS_H = 262
GATES_H = 112          # the gate buttons, added to the controls for N = 3

ARROW_R = SPACING / 2 - PARTICLE_R - 10   # radius of the arrow arcs
ARROW_W = 12                             # band width of an arrow
ARROW_HEAD_W = 15                        # half-width of the arrow head
ARROW_HEAD_LEN = 20
SUPERSAMPLE = 4                          # arrows are rendered this much larger
ARROW_TAIL_GAP = math.radians(12)        # keeps the ends off the row line
ARROW_TIP_GAP = math.radians(8)

BRAID_GAP = 64         # between the lower arrows and the braid's top
LAYER_H = 78           # height of one exchange in the braid diagram
STRAND_W = 5
STRAND_HALO = 6        # background margin around an over-crossing strand

# ----------------------------------------------------------------------
# timing
# ----------------------------------------------------------------------
SWAP_SECONDS = 0.8     # the anyons moving
FADE_SECONDS = 0.35    # the state crossfading to its new form
ABSORB_SECONDS = 1.0   # braid phases flying into the coefficients
QUEUE_MAX = 4
SLOW_STEP, FAST_STEP = 3.6, 0.9   # autoplay seconds per step at the speed ends

# ----------------------------------------------------------------------
# palette
# ----------------------------------------------------------------------
BG = (255, 255, 255)
INK = (22, 24, 28)
STRAND = (38, 41, 48)
INK_SOFT = (120, 126, 136)
INK_FAINT = (200, 204, 211)
PANEL_BG = render.PANEL_BG
ARROW = (160, 204, 244)
ARROW_HOVER = (38, 110, 204)
CARD_LINE = (226, 229, 234)

FONT_STACK = "segoeui,segoe ui,inter,helveticaneue,helvetica neue,arial"


def _enable_dpi_awareness() -> None:
    """Ask Windows for real pixels instead of a stretched, blurry window."""
    try:
        import ctypes

        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except (AttributeError, OSError):
            ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


def ease(t: float) -> float:
    """Cubic ease-in-out on [0, 1]."""
    t = min(max(t, 0.0), 1.0)
    return 4 * t**3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def aa_polygon(surf: pygame.Surface, color, pts) -> None:
    ipts = [(round(x), round(y)) for x, y in pts]
    pygame.gfxdraw.filled_polygon(surf, ipts, color)
    pygame.gfxdraw.aapolygon(surf, ipts, color)


def aa_circle(surf: pygame.Surface, color, center, r: float) -> None:
    x, y, r = round(center[0]), round(center[1]), round(r)
    pygame.gfxdraw.filled_circle(surf, x, y, r, color)
    pygame.gfxdraw.aacircle(surf, x, y, r, color)


SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def sci(x: float) -> str:
    """2.2e-03 as 2.2·10⁻³."""
    mantissa, exponent = f"{x:.1e}".split("e")
    return f"{mantissa}·10{str(int(exponent)).translate(SUPERSCRIPT)}"


def wrap(font: pygame.font.Font, text: str, width: int) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if line and font.size(trial)[0] > width:
            lines.append(line)
            line = word
        else:
            line = trial
    return lines + ([line] if line else [])


class Arrow:
    """A fat semicircular arrow over (side=+1) or under (side=-1) a gap.

    Both arrows run from left to right; clicking one moves the left particle
    along that side of the gap.
    """

    def __init__(self, gap: int, side: int, cx: float, cy: float) -> None:
        self.gap, self.side = gap, side
        self.cx, self.cy = cx, cy
        self.hover = 0.0  # animated 0..1 hover amount
        self.sprites = [self._sprite(c) for c in (ARROW, ARROW_HOVER)]

    def _at(self, theta: float, r: float) -> tuple[float, float]:
        return (self.cx + r * math.cos(theta), self.cy - self.side * r * math.sin(theta))

    def _outline(self) -> list[tuple[float, float]]:
        head_span = ARROW_HEAD_LEN / ARROW_R
        th_head = ARROW_TIP_GAP + head_span
        th_tail = math.pi - ARROW_TAIL_GAP
        steps = 40
        thetas = [th_tail + (th_head - th_tail) * k / steps for k in range(steps + 1)]
        ro, ri = ARROW_R + ARROW_W / 2, ARROW_R - ARROW_W / 2
        outer = [self._at(t, ro) for t in thetas]
        inner = [self._at(t, ri) for t in thetas]
        head = [
            self._at(th_head, ARROW_R + ARROW_HEAD_W),
            self._at(ARROW_TIP_GAP, ARROW_R),
            self._at(th_head, ARROW_R - ARROW_HEAD_W),
        ]
        return outer + head + inner[::-1]

    def _sprite(self, color) -> tuple[pygame.Surface, tuple[int, int]]:
        """The arrow drawn supersampled, then scaled down for smooth edges."""
        pad = ARROW_W
        poly = self._outline()
        x0 = math.floor(min(x for x, _ in poly)) - pad
        y0 = math.floor(min(y for _, y in poly)) - pad
        w = math.ceil(max(x for x, _ in poly)) + pad - x0
        h = math.ceil(max(y for _, y in poly)) + pad - y0
        big = pygame.Surface((w * SUPERSAMPLE, h * SUPERSAMPLE), pygame.SRCALPHA)
        big.fill((*color, 0))  # so scaled edges don't darken
        pts = [((x - x0) * SUPERSAMPLE, (y - y0) * SUPERSAMPLE) for x, y in poly]
        pygame.draw.polygon(big, color, pts)
        tx, ty = self._at(math.pi - ARROW_TAIL_GAP, ARROW_R)
        pygame.draw.circle(
            big, color,
            ((tx - x0) * SUPERSAMPLE, (ty - y0) * SUPERSAMPLE),
            ARROW_W / 2 * SUPERSAMPLE,
        )
        return pygame.transform.smoothscale(big, (w, h)), (x0, y0)

    def hit(self, pos: tuple[int, int]) -> bool:
        dx, dy = pos[0] - self.cx, pos[1] - self.cy
        if abs(math.hypot(dx, dy) - ARROW_R) > ARROW_HEAD_W + 4:
            return False
        theta = math.atan2(-self.side * dy, dx)
        return ARROW_TIP_GAP - 0.1 <= theta <= math.pi - ARROW_TAIL_GAP + 0.25

    def draw(self, surf: pygame.Surface) -> None:
        (light, pos), (dark, _) = self.sprites
        if self.hover < 0.99:
            surf.blit(light, pos)
        if self.hover > 0.01:
            dark.set_alpha(round(255 * self.hover))
            surf.blit(dark, pos)


class Swap:
    """An exchange of the particles in slots ``gap`` and ``gap + 1``.

    ``side`` is +1 if the left particle travels along the upper arc.
    """

    def __init__(self, gap: int, side: int) -> None:
        self.gap, self.side = gap, side


@dataclass
class View:
    """What the state display shows: a sequence of frames. Each frame is held
    for its transition's hold time, then fades (or has its phases absorbed)
    into the next."""

    frames: list[Display]
    transitions: list[tuple[str, float]] = field(default_factory=list)  # (kind, hold)
    t: float = 0.0

    @staticmethod
    def still(display: Display) -> "View":
        return View([display])

    def locate(self) -> tuple[int, str | None, float]:
        """(frame index, transition kind or None, progress 0..1) at time t."""
        t = self.t
        for i, (kind, hold) in enumerate(self.transitions):
            if t < hold:
                return i, None, 0.0
            t -= hold
            dur = ABSORB_SECONDS if kind == "absorb" else FADE_SECONDS
            if t < dur:
                return i, kind, t / dur
            t -= dur
        return len(self.frames) - 1, None, 0.0

    def duration(self) -> float:
        return sum(hold + (ABSORB_SECONDS if kind == "absorb" else FADE_SECONDS)
                   for kind, hold in self.transitions)

    def skip_holds_until(self, seconds: float) -> None:
        self.t = max(self.t, seconds)


@dataclass
class GateRun:
    """A gate being applied: its exchanges, played one after the other."""

    name: str
    tasks: deque
    total: int
    done: int = 0


@dataclass
class Plan:
    """A task being explained step by step; ``index`` -1 is the intro."""

    task: Task
    steps: list[Step]
    playing: bool
    index: int = -1
    t: float = 0.0          # time since entering the current step


class App:
    def __init__(self) -> None:
        _enable_dpi_awareness()
        pygame.init()
        pygame.display.set_caption("Fibonacci anyons")
        self.panel_w = LEFT_MARGIN + max(
            PANEL_MIN_W, round((N - 1) * SPACING + 2 * PANEL_PAD)
        )
        info = pygame.display.Info()
        ref_w = self.ref_width()
        want_w = self.panel_w + MID_W + 2 * MARGIN + ref_w
        w = min(want_w, max(1200, info.current_w - 60)) if info.current_w > 0 else want_w
        h = min(WINDOW_H, max(800, info.current_h - 110)) if info.current_h > 0 else WINDOW_H
        page = webcanvas.viewport()
        if page:  # in the browser: fill the page
            w, h = max(1200, page[0]), max(800, page[1])
        self.set_mode((w, h))
        self.clock = pygame.time.Clock()
        self.font_small = render.sysfont(FONT_STACK, 13)
        f = render.fonts()
        self.particle_tau = render.tau_glyph(26)

        x0 = (LEFT_MARGIN + self.panel_w) / 2 - (N - 1) * SPACING / 2
        self.xs = [x0 + i * SPACING for i in range(N)]
        self.arrows = [
            Arrow(g, side, (self.xs[g] + self.xs[g + 1]) / 2, ROW_Y)
            for g in range(N - 1)
            for side in (+1, -1)
        ]
        self.braid_top = ROW_Y + ARROW_R + ARROW_HEAD_W + BRAID_GAP
        self.hand_cursor = False

        # controls
        self.detail = ui.Slider(f.ui, f.ui_bold, EXPLAIN, stops=["Instant", "Explain", "Detailed"])
        self.speed = ui.Slider(f.ui, f.ui_bold, 0.5, ends=("slow", "fast"))
        self.autoplay = ui.Checkbox("Autoplay", True)
        self.always_std = ui.Checkbox("Always standard basis", False)
        self.to_std = ui.Button("To Standard Basis", primary=True)
        self.next_btn = ui.Button(icon=ui.icon_next)
        self.play_btn = ui.Button(icon=lambda s, r, c: ui.icon_play_pause(s, r, c, self.play_anim))
        self.play_anim = 0.0     # 0 = play icon, 1 = pause icon
        # single-qubit gates, for the qubit of three anyons
        self.gate_buttons = {name: ui.Button(name) for name in GATES} if N == 3 else {}
        self.swap_seconds = SWAP_SECONDS
        self.clock_t = 0.0

        self._images: dict[tuple, pygame.Surface] = {}
        self.reset()

    def reset(self) -> None:
        self.order = list(range(N))       # particle id in each slot
        self.state = initial_state(N)
        self.view = View.still(Display(self.state))
        self.history: list[Swap] = []      # completed exchanges, oldest first
        self.active: Swap | None = None
        self.t = 0.0                       # raw progress of the active swap
        self.queue: deque[Task] = deque()
        self.plan: Plan | None = None
        self.gate_run: GateRun | None = None

    # ------------------------------------------------------------------
    # layout
    # ------------------------------------------------------------------
    def set_mode(self, size: tuple[int, int]) -> None:
        # the browser page is resized by webcanvas.poll, not by SDL
        flags = 0 if webcanvas.WEB else pygame.RESIZABLE
        self.screen = pygame.display.set_mode(size, flags)
        webcanvas.fit(size, BG)

    def ref_width(self) -> int:
        return max(render.f_matrix_panel().get_width(), render.r_matrix_panel().get_width())

    def rects(self) -> dict[str, pygame.Rect]:
        w, h = self.screen.get_size()
        ref_w = self.ref_width()
        r_panel = render.r_matrix_panel(ref_w)
        f_panel = render.f_matrix_panel(ref_w)
        # one column on the right: controls, braid matrix, F matrix
        controls_h = CONTROLS_H + (GATES_H if self.gate_buttons else 0)
        controls = pygame.Rect(w - MARGIN - ref_w, MARGIN, ref_w, controls_h)
        r_rect = r_panel.get_rect(topleft=(controls.x, controls.bottom + PANEL_GAP))
        f_rect = f_panel.get_rect(topleft=(controls.x, r_rect.bottom + PANEL_GAP))
        mid_x = self.panel_w
        mid_w = max(240, f_rect.x - MARGIN - mid_x)
        card = pygame.Rect(mid_x, h - MARGIN - CARD_H, mid_w, CARD_H)
        return {"controls": controls, "r": r_rect, "f": f_rect, "card": card,
                "mid": pygame.Rect(mid_x, 0, mid_w, h)}

    def layout_widgets(self) -> None:
        c = self.rects()["controls"]
        pad = 28
        col_w = (c.w - 2 * pad - 48) // 2
        x0, x1 = c.x + pad, c.x + pad + col_w + 48
        top = c.y + 84
        self.detail.rect = pygame.Rect(x0 + 12, top + 26, col_w - 24, 6)
        self.speed.rect = pygame.Rect(x0 + 12, top + 118, col_w - 24, 6)
        self.autoplay.rect = pygame.Rect(x1, top + 4, col_w, 30)
        self.always_std.rect = pygame.Rect(x1, top + 44, col_w, 30)
        self.to_std.rect = pygame.Rect(x1, top + 100, col_w, 42)
        if self.gate_buttons:
            gap = 10
            bw = (c.w - 2 * pad - gap * (len(self.gate_buttons) - 1)) // len(self.gate_buttons)
            y = c.y + CONTROLS_H + 16
            for i, b in enumerate(self.gate_buttons.values()):
                b.rect = pygame.Rect(x0 + i * (bw + gap), y, bw, 40)

        card = self.rects()["card"]
        size = 44
        self.play_btn.rect = pygame.Rect(card.right - 26 - size, card.bottom - 26 - size, size, size)
        self.next_btn.rect = self.play_btn.rect.move(-(size + 10), 0)

    # ------------------------------------------------------------------
    # tasks and steps
    # ------------------------------------------------------------------
    @property
    def level(self) -> int:
        return int(self.detail.value)

    def step_seconds(self) -> float:
        return SLOW_STEP * (FAST_STEP / SLOW_STEP) ** self.speed.value

    def request(self, task: Task) -> None:
        if len(self.queue) < QUEUE_MAX:
            self.queue.append(task)

    def busy(self) -> bool:
        return self.plan is not None or self.active is not None or self.gate_run is not None

    def start_task(self, task: Task, instant: bool = False) -> None:
        level = INSTANT if instant else self.level
        steps, final = plan(self.state, task, level, self.always_std.checked)
        if level == INSTANT:
            # the new state appears once the anyons have landed; in a gate the
            # next exchange starts right then, so the state simply switches
            hold = self.swap_seconds if task.gap is not None else 0.0
            self.view = View([Display(self.state), Display(final)], [("fade", hold)])
            self.state = final
            if task.gap is not None:
                self.start_swap(Swap(task.gap, task.side))
            return
        if not steps:
            return
        self.plan = Plan(task, steps, playing=self.autoplay.checked)

    def start_gate(self, name: str) -> None:
        """Queue the exchanges of gate ``name``; the last factor acts first."""
        word, _ = GATES[name]
        tasks = deque(
            # generator g exchanges anyons g and g + 1; k > 0 is counterclockwise,
            # which is the lower arrow (side -1)
            Task(g - 1, -1 if k > 0 else 1)
            for g, k in reversed(word) for _ in range(abs(k))
        )
        self.gate_run = GateRun(name, tasks, len(tasks))
        # a quick pace, so even 30 exchanges take only a few seconds
        self.swap_seconds = 0.15 + 0.1 * self.step_seconds()

    def next_gate_exchange(self) -> None:
        run = self.gate_run
        if run.tasks:
            run.done += 1
            self.start_task(run.tasks.popleft(), instant=True)
        elif self.view.locate()[1] is None:
            # the last exchange has landed and the state has settled
            self.gate_run = None
            self.swap_seconds = SWAP_SECONDS

    def start_swap(self, swap: Swap) -> None:
        self.active, self.t = swap, 0.0

    def finish_swap(self) -> None:
        g = self.active.gap
        self.order[g], self.order[g + 1] = self.order[g + 1], self.order[g]
        self.history.append(self.active)
        self.active = None
        self.t = 0.0

    def hold_seconds(self, tag: str) -> float:
        if tag == AFTER_SWAP:
            return SWAP_SECONDS
        if tag == AFTER_PHASES:
            # the written-out phases stay a while before being absorbed
            return max(1.0, 0.7 * self.step_seconds())
        return 0.0

    def dwell(self) -> float:
        """How long autoplay stays on the current step."""
        p = self.plan
        if p.index < 0:
            return 0.6 * self.step_seconds()
        return max(self.step_seconds(), self.view.duration() + 0.6)

    def show(self, frames: list[Display], transitions: list[tuple[str, float]]) -> None:
        """Show ``frames``, fading in from whatever is on display now."""
        current = self.view.frames[self.view.locate()[0]]
        if current != frames[0]:
            frames = [current] + frames
            transitions = [("fade", 0.0)] + transitions
        self.view = View(frames, transitions)

    def advance(self) -> None:
        """Next step, or finish the plan after the last one."""
        p = self.plan
        if p is None:
            return
        if self.active is not None:
            # the anyons are still moving: jump to the end of that first
            self.finish_swap()
            self.view.skip_holds_until(SWAP_SECONDS)
            return
        p.index += 1
        p.t = 0.0
        if p.index >= len(p.steps):
            self.plan = None
            self.show([Display(self.state)], [])
            return
        step = p.steps[p.index]
        self.show(step.frames,
                  [(kind, self.hold_seconds(tag)) for kind, tag in step.transitions])
        self.state = step.state
        if step.swap is not None:
            self.start_swap(Swap(*step.swap))

    def current_step(self) -> Step | None:
        p = self.plan
        if p is None or p.index < 0:
            return None
        return p.steps[p.index]

    def toggle_play(self) -> None:
        if self.plan is not None:
            self.plan.playing = not self.plan.playing

    def update(self, dt: float) -> None:
        self.clock_t += dt
        if self.gate_run is not None and self.active is None:
            self.next_gate_exchange()
        elif not self.busy() and self.queue:
            self.start_task(self.queue.popleft())

        if self.active is not None:
            self.t += dt / self.swap_seconds
            if self.t >= 1.0:
                self.finish_swap()
        self.view.t += dt

        p = self.plan
        if p is not None:
            p.t += dt
            if p.playing and p.t >= self.dwell():
                self.advance()

        mouse = pygame.mouse.get_pos()
        self.layout_widgets()
        self.to_std.enabled = not (self.state.is_standard() and not self.busy())
        playing = self.plan is not None and self.plan.playing
        self.next_btn.enabled = self.play_btn.enabled = self.plan is not None
        self.play_anim = ui.approach(self.play_anim, 1.0 if playing else 0.0, dt, 10)
        for b in self.gate_buttons.values():
            b.enabled = not self.busy() and not self.queue
        for w in (self.detail, self.speed, self.autoplay, self.always_std,
                  self.to_std, self.next_btn, self.play_btn, *self.gate_buttons.values()):
            w.update(mouse, dt)

        hovered = self.arrow_at(mouse)
        pulsing = self.pulsing_arrow()
        for a in self.arrows:
            if a is pulsing:
                a.hover = 0.55 + 0.45 * math.sin(self.clock_t * 5.0)
            else:
                a.hover = ui.approach(a.hover, 1.0 if a is hovered else 0.0, dt)
        over_button = any(
            b.enabled and b.rect.collidepoint(mouse)
            for b in (self.to_std, self.next_btn, self.play_btn, *self.gate_buttons.values())
        ) or any(c.rect.collidepoint(mouse) for c in (self.autoplay, self.always_std))
        want_hand = hovered is not None or over_button
        if want_hand != self.hand_cursor:
            self.hand_cursor = want_hand
            try:
                pygame.mouse.set_cursor(
                    pygame.SYSTEM_CURSOR_HAND if want_hand else pygame.SYSTEM_CURSOR_ARROW
                )
            except pygame.error:
                pass  # no system cursors (e.g. headless)

    def pulsing_arrow(self) -> Arrow | None:
        """The arrow of an exchange that is being explained but not yet run."""
        p = self.plan
        if p is None or p.task.gap is None:
            return None
        braid_index = next(i for i, s in enumerate(p.steps) if s.swap is not None)
        if p.index >= braid_index:
            return None
        return next(a for a in self.arrows
                    if a.gap == p.task.gap and a.side == p.task.side)

    def arrow_at(self, pos) -> Arrow | None:
        return next((a for a in self.arrows if a.hit(pos)), None)

    def particle_positions(self) -> list[tuple[float, float]]:
        pos = [(x, float(ROW_Y)) for x in self.xs]
        if self.active is not None:
            g, s = self.active.gap, self.active.side
            cx, r = (self.xs[g] + self.xs[g + 1]) / 2, SPACING / 2
            phi = math.pi * (1 - ease(self.t))
            pos[g] = (cx + r * math.cos(phi), ROW_Y - s * r * math.sin(phi))
            pos[g + 1] = (cx - r * math.cos(phi), ROW_Y + s * r * math.sin(phi))
        return pos

    # ------------------------------------------------------------------
    # input
    # ------------------------------------------------------------------
    def click(self, pos) -> None:
        if self.autoplay.hit(pos):
            self.autoplay.checked = not self.autoplay.checked
        elif self.always_std.hit(pos):
            self.always_std.checked = not self.always_std.checked
            if self.always_std.checked and not self.state.is_standard():
                self.request(Task())
        elif self.to_std.hit(pos):
            self.request(Task())
        elif self.next_btn.hit(pos):
            self.advance()
        elif self.play_btn.hit(pos):
            self.toggle_play()
        elif any(b.hit(pos) for b in self.gate_buttons.values()):
            self.start_gate(next(n for n, b in self.gate_buttons.items() if b.hit(pos)))
        else:
            arrow = self.arrow_at(pos)
            if arrow is not None:
                self.request(Task(arrow.gap, arrow.side))

    # ------------------------------------------------------------------
    # drawing
    # ------------------------------------------------------------------
    def draw(self) -> None:
        h = self.screen.get_height()
        self.screen.fill(BG)
        rects = self.rects()

        self.draw_braid(h)
        for a in self.arrows:
            a.draw(self.screen)
        for x, y in self.particle_positions():
            aa_circle(self.screen, INK, (x, y), PARTICLE_R)
            self.screen.blit(self.particle_tau,
                             self.particle_tau.get_rect(center=(round(x), round(y) + 1)))
        self.draw_state(rects)
        self.draw_card(rects["card"])
        self.draw_controls(rects["controls"])
        # the rule of the current step lights up in its reference panel
        step = self.current_step()
        rule = step.rule if step else None
        r_active = rule[1] if rule and rule[0] == "R" else None
        ref_w = self.ref_width()
        self.screen.blit(render.r_matrix_panel(ref_w, r_active), rects["r"])
        self.screen.blit(render.f_matrix_panel(ref_w, rule == ("F",)), rects["f"])

    def state_image(self, d: Display, max_w: int, max_h: int,
                    blank_phases: bool = False) -> tuple[pygame.Surface, list[dict], float]:
        """The state expression for ``d``, its term anchors and its scale."""
        key = (d, max_w, max_h, blank_phases)
        hit = self._images.get(key)
        if hit is None:
            # many particles: shrink the expression (re-wrapping it to the wider
            # line that allows) until it fits above the explanation card
            scale = 1.0
            while True:
                img, anchors = render.state_expression(
                    N, d.state, max_w / scale, d.hl, d.phases, blank_phases)
                if img.get_height() * scale <= max_h or scale <= 0.3:
                    break
                scale -= 0.05
            scale = min(scale, max_w / img.get_width())
            if scale < 1.0:
                size = (round(img.get_width() * scale), round(img.get_height() * scale))
                img = pygame.transform.smoothscale(img, size)
                anchors = [
                    {k: pygame.Rect(round(r.x * scale), round(r.y * scale),
                                    round(r.w * scale), round(r.h * scale))
                     for k, r in a.items()}
                    for a in anchors
                ]
            if len(self._images) > 32:
                self._images.clear()
            hit = self._images[key] = (img, anchors, min(scale, 1.0))
        return hit

    def draw_state(self, rects) -> None:
        f = render.fonts()
        mid = rects["mid"]
        x = mid.x
        heading = f.heading.render("State", True, INK)
        self.screen.blit(heading, (x, ROW_Y - 88))

        # which basis the kets are written in
        std = self.view.frames[-1].state.is_standard()
        text = "standard basis" if std else "rotated basis"
        img = f.ui.render(text, True, INK_SOFT if std else (150, 96, 20))
        pill = img.get_rect().inflate(20, 8)
        pill.midleft = (x + heading.get_width() + 16, ROW_Y - 88 + heading.get_height() // 2 + 1)
        pygame.draw.rect(self.screen, PANEL_BG if std else (252, 240, 214), pill,
                         border_radius=pill.h // 2)
        self.screen.blit(img, img.get_rect(center=pill.center))

        top = ROW_Y - 36
        size = (mid.w, rects["card"].top - 36 - top)
        v = self.view
        i, kind, u = v.locate()
        if kind is None:
            self.screen.blit(self.state_image(v.frames[i], *size)[0], (x, top))
        elif kind == "fade":
            for d, alpha in ((v.frames[i], 1 - ease(u)), (v.frames[i + 1], ease(u))):
                img = self.state_image(d, *size)[0].copy()
                img.set_alpha(round(255 * alpha))
                self.screen.blit(img, (x, top))
        else:
            self.draw_absorb(v.frames[i], v.frames[i + 1], u, (x, top), size)

    def draw_absorb(self, src: Display, dst: Display, u: float, origin, size) -> None:
        """Each written-out phase flies into the coefficient it multiplies,
        then the expression fades over to the new coefficients."""
        def window(lo: float, hi: float) -> float:
            return ease((u - lo) / (hi - lo))

        ox, oy = origin
        blank, anchors, scale = self.state_image(src, *size, blank_phases=True)
        img = blank.copy()
        img.set_alpha(round(255 * (1 - window(0.5, 0.9))))
        self.screen.blit(img, origin)
        new = self.state_image(dst, *size)[0].copy()
        new.set_alpha(round(255 * window(0.55, 1.0)))
        self.screen.blit(new, origin)

        fly = window(0.0, 0.5)
        for labels, a in zip(basis(src.state.tree), anchors):
            text, colour = render.term_phase(src.state, labels, src.phases)
            sprite = render.phase_image(text, colour)
            k = scale * (1 - 0.6 * fly)
            sprite = pygame.transform.smoothscale(
                sprite, (max(1, round(sprite.get_width() * k)),
                         max(1, round(sprite.get_height() * k))))
            p0, p1 = a["phase"].center, a["amp"].center
            pos = (ox + p0[0] + (p1[0] - p0[0]) * fly, oy + p0[1] + (p1[1] - p0[1]) * fly)
            alpha = 1 - window(0.35, 0.55)
            if src.state.amp(labels) == 0:
                alpha *= render.FADED_ALPHA / 255
            sprite.set_alpha(round(255 * alpha))
            self.screen.blit(sprite, sprite.get_rect(center=(round(pos[0]), round(pos[1]))))

    def draw_card(self, card: pygame.Rect) -> None:
        f = render.fonts()
        surf = self.screen
        if self.gate_run is not None:
            self.draw_gate_card(card)
            return
        p = self.plan
        if p is None:
            pygame.draw.rect(surf, CARD_LINE, card, 2, border_radius=18)
            if self.level == INSTANT:
                msg = "Instant mode: exchanges are applied directly."
            else:
                msg = "Click an arrow to exchange two anyons; the steps appear here."
            img = f.ui.render(msg, True, INK_SOFT)
            surf.blit(img, img.get_rect(center=card.center))
            return

        pygame.draw.rect(surf, PANEL_BG, card, border_radius=18)
        pad = 26
        x, y = card.x + pad, card.y + pad - 4
        list_w = 210
        text_w = card.w - 2 * pad - list_w - 24

        task = p.task
        if task.gap is None:
            caps = "BACK TO THE STANDARD BASIS"
        else:
            way = "COUNTERCLOCKWISE" if task.ccw else "CLOCKWISE"
            caps = f"EXCHANGE {anyon(task.gap)} ↔ {anyon(task.gap + 1)} · {way}"
        surf.blit(f.caps.render(caps, True, INK_SOFT), (x, y))
        y += 28

        extra = []
        if p.index < 0:
            headline = [("Ready", "")]
            detail = ("Press play or next step to go through the "
                      f"{len(p.steps)} step{'s' if len(p.steps) != 1 else ''}.")
        else:
            step = p.steps[p.index]
            headline, detail = step.headline, step.detail
            extra = step.extra
        hx = x
        for text, role in headline:
            col = render.ROLE_COLOURS.get(role, INK)
            img = f.ui_title.render(text, True, col)
            surf.blit(img, (hx, y))
            hx += img.get_width()
        y += 36
        if extra:
            # the rule in use, coloured like the terms it acts on
            hx = x
            for text, role in extra:
                img = f.math.render(text, True, render.ROLE_COLOURS.get(role, INK))
                surf.blit(img, (hx, y - 2))
                hx += img.get_width()
            y += 32
        for line in wrap(f.ui, detail, text_w):
            surf.blit(f.ui.render(line, True, INK_SOFT), (x, y))
            y += 21

        # the list of steps, on the right
        lx = card.right - pad - list_w
        pygame.draw.line(surf, CARD_LINE, (lx - 18, card.y + pad), (lx - 18, card.bottom - 90), 1)
        ly = card.y + pad - 2
        max_rows = 7
        first = max(0, min(p.index - 2, len(p.steps) - max_rows))
        for i in range(first, min(len(p.steps), first + max_rows)):
            step = p.steps[i]
            c = (lx + 10, ly + 11)
            if i < p.index:
                ui.smooth_circle(surf, INK, c, 8)
                pygame.draw.lines(surf, (255, 255, 255), False,
                                  [(c[0] - 4, c[1]), (c[0] - 1, c[1] + 3), (c[0] + 4, c[1] - 3)], 2)
                col = INK_SOFT
            elif i == p.index:
                ui.smooth_circle(surf, render.ROLE_COLOURS["pair"] if step.swap else INK, c, 8)
                col = INK
            else:
                ui.smooth_circle(surf, INK_FAINT, c, 8, 2)
                col = INK_SOFT
            font = f.ui_bold if i == p.index else f.ui
            surf.blit(font.render(f"{i + 1}. {step.title}", True, col), (lx + 28, ly + 1))
            ly += 28

        # progress and transport
        bar = pygame.Rect(x, card.bottom - pad - 26, self.next_btn.rect.x - 30 - x, 6)
        pygame.draw.rect(surf, (226, 228, 233), bar, border_radius=3)
        frac = (p.index + 1) / len(p.steps)
        pygame.draw.rect(surf, INK, (bar.x, bar.y, max(6, round(bar.w * frac)), 6), border_radius=3)
        label = "ready" if p.index < 0 else f"step {p.index + 1} of {len(p.steps)}"
        surf.blit(f.ui.render(label, True, INK_SOFT), (bar.x, bar.bottom + 8))
        self.next_btn.draw(surf, f.ui_bold)
        self.play_btn.draw(surf, f.ui_bold)

    def draw_gate_card(self, card: pygame.Rect) -> None:
        f = render.fonts()
        surf = self.screen
        run = self.gate_run
        word, err = GATES[run.name]
        pygame.draw.rect(surf, PANEL_BG, card, border_radius=18)
        pad = 26
        x, y = card.x + pad, card.y + pad - 4
        surf.blit(f.caps.render(f"APPLYING THE {run.name} GATE BY BRAIDING", True, INK_SOFT), (x, y))
        y += 28
        exact = err == 0
        head = f"{run.name}  {'=' if exact else '≈'}  {run.total} exchanges"
        surf.blit(f.ui_title.render(head, True, INK), (x, y))
        y += 36
        if exact:
            detail = ("Z is exact: five counterclockwise exchanges of τ₁ and τ₂ give "
                      "the two qubit states the phases 1 and −1, up to a global phase.")
        else:
            detail = (f"No braid gives {run.name} exactly; this is the shortest one found "
                      f"within 5·10⁻³ of it (error {sci(err)}, ignoring the global phase). "
                      "The total charge τ is never changed, so nothing leaks out of the qubit.")
        for line in wrap(f.ui, detail, card.w - 2 * pad):
            surf.blit(f.ui.render(line, True, INK_SOFT), (x, y))
            y += 21

        bar = pygame.Rect(x, card.bottom - pad - 26, card.w - 2 * pad, 6)
        pygame.draw.rect(surf, (226, 228, 233), bar, border_radius=3)
        done = run.done - (1 - self.t if self.active is not None else 0)
        frac = max(0.0, done) / run.total
        pygame.draw.rect(surf, INK, (bar.x, bar.y, max(6, round(bar.w * frac)), 6),
                         border_radius=3)
        label = f"exchange {run.done} of {run.total}"
        surf.blit(f.ui.render(label, True, INK_SOFT), (bar.x, bar.bottom + 8))

    def draw_controls(self, c: pygame.Rect) -> None:
        f = render.fonts()
        surf = self.screen
        pygame.draw.rect(surf, PANEL_BG, c, border_radius=18)
        surf.blit(f.heading.render("Controls", True, INK), (c.x + 28, c.y + 22))
        caps = [("DETAIL", self.detail), ("STEP SPEED", self.speed)]
        for text, slider in caps:
            surf.blit(f.caps.render(text, True, INK_SOFT),
                      (slider.rect.x - 12, slider.rect.y - 30))
            slider.draw(surf)
        self.autoplay.draw(surf, f.ui)
        self.always_std.draw(surf, f.ui)
        self.to_std.draw(surf, f.ui_bold)
        if not self.gate_buttons:
            return

        # the gates, with their braid length and error on hover
        top = c.y + CONTROLS_H - 12
        surf.blit(f.caps.render("QUBIT GATES  (BRAIDS OF THREE ANYONS)", True, INK_SOFT),
                  (c.x + 28, top))
        for b in self.gate_buttons.values():
            b.draw(surf, f.ui_title)
        hovered = next((n for n, b in self.gate_buttons.items()
                        if b.rect.collidepoint(pygame.mouse.get_pos())), None)
        if hovered is None:
            caption = "The qubit: the two states of total charge τ.  Hover a gate for its braid."
        else:
            word, err = GATES[hovered]
            n = sum(abs(k) for _, k in word)
            if err == 0:
                caption = f"{hovered}  =  σ₁⁵ exactly, {n} exchanges"
            else:
                caption = f"{hovered}  ≈  {n} exchanges, error {sci(err)} (up to a global phase)"
        b0 = next(iter(self.gate_buttons.values())).rect
        surf.blit(f.ui.render(caption, True, INK_SOFT), (c.x + 28, b0.bottom + 12))

    def draw_braid(self, h: int) -> None:
        surf = self.screen
        top = self.braid_top
        surf.set_clip(pygame.Rect(0, round(top) - STRAND_W, self.panel_w, h))

        # the time axis
        ax = LEFT_MARGIN + 34
        pygame.draw.line(surf, INK_FAINT, (ax, h - 24), (ax, top + 6), 2)
        aa_polygon(surf, INK_FAINT, [(ax - 6, top + 12), (ax, top), (ax + 6, top + 12)])
        label = pygame.transform.rotate(self.font_small.render("time", True, INK_SOFT), 90)
        surf.blit(label, label.get_rect(midright=(ax - 6, top + 44)))

        # (y at v=0, progress drawn, swap) per layer, newest first
        e = ease(self.t) if self.active is not None else 0.0
        layers: list[tuple[float, float, Swap]] = []
        if self.active is not None:
            layers.append((top + e * LAYER_H, e, self.active))
        y = top + e * LAYER_H
        for swap in reversed(self.history):
            if y > h:
                break
            y += LAYER_H
            layers.append((y, 1.0, swap))

        for y0, vmax, swap in layers:
            self.draw_layer(y0, vmax, swap)
        # strands continue straight down below the oldest layer
        for x in self.xs:
            pygame.draw.rect(surf, STRAND, (round(x - STRAND_W / 2), round(y), STRAND_W, h - y + 1))

        # the braid's open ends are the particles of the current moment
        surf.set_clip(None)
        cur = self.particle_positions()
        g = self.active.gap if self.active is not None else -1
        for i, x in enumerate(self.xs):
            if self.active is not None and i in (g, g + 1):
                x = cur[i][0]
            aa_circle(surf, STRAND, (x, top), STRAND_W / 2 + 2)

    def draw_layer(self, y0: float, vmax: float, swap: Swap) -> None:
        """One exchange, from v=0 at ``y0`` up to v=``vmax``."""
        surf = self.screen
        g = swap.gap
        y1 = y0 - vmax * LAYER_H
        for i, x in enumerate(self.xs):
            if i not in (g, g + 1):
                pygame.draw.rect(
                    surf, STRAND,
                    (round(x - STRAND_W / 2), round(y1), STRAND_W, round(y0 - y1) + 1),
                )
        if vmax < 1e-3:
            return

        cx, r = (self.xs[g] + self.xs[g + 1]) / 2, SPACING / 2
        steps = 48

        def strand(sign: int, lo: float, hi: float):
            vs = [lo + (hi - lo) * k / steps for k in range(steps + 1)]
            return [(cx - sign * r * math.cos(math.pi * v), y0 - v * LAYER_H) for v in vs]

        # sign +1: left particle moving right; it passes behind on the upper arc
        under, over = (+1, -1) if swap.side > 0 else (-1, +1)
        aa_polygon(surf, STRAND, render.ribbon(strand(under, 0.0, vmax), STRAND_W))
        lo, hi = 0.18, min(vmax, 0.82)
        if hi > lo:
            aa_polygon(surf, BG, render.ribbon(strand(over, lo, hi), STRAND_W + 2 * STRAND_HALO))
        aa_polygon(surf, STRAND, render.ribbon(strand(over, 0.0, vmax), STRAND_W))

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    async def run(self) -> None:
        while True:
            dt = self.clock.tick(120) / 1000.0
            page = webcanvas.poll(dt)
            if page:
                self.set_mode((max(1200, page[0]), max(800, page[1])))
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    return
                if self.detail.handle(ev) or self.speed.handle(ev):
                    continue
                if ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        return
                    if ev.key == pygame.K_r:
                        self.reset()
                    elif ev.key == pygame.K_SPACE:
                        self.toggle_play()
                    elif ev.key in (pygame.K_RIGHT, pygame.K_n):
                        self.advance()
                if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    self.click(ev.pos)
            self.update(dt)
            self.draw()
            pygame.display.flip()
            await asyncio.sleep(0)  # lets the browser run between frames


async def main() -> None:
    await App().run()


if __name__ == "__main__":
    asyncio.run(main())
    pygame.quit()
    if sys.platform != "emscripten":
        sys.exit()
