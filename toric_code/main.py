"""Interactive toric-code visualiser.

Run with:  python main.py

Kitaev convention: A_v = prod X over the four edges at a vertex (star),
B_p = prod Z around a face (plaquette). So Z-strings create ``e`` charges on
vertices and X-strings create ``m`` fluxes on faces.
"""

from __future__ import annotations

import asyncio
import math
import sys

import pygame

from lattice import TorusLattice
from pauli import PauliWord, phase_prefix
from simplify import build_reduce_script, build_script
from viewer3d import WorldlinePanel
from worldlines import WorldlineTracker
from ui import (
    ACCENT, BG, E_COL, GATE_COL, GLOW, GRID, GRID_GHOST, HAIRLINE, INK,
    INK_FAINT, INK_SOFT, M_COL, PANEL, WHITE, X_COL, Y_COL, Z_COL,
    Button, Fonts, aa_circle, blend, icon_close, icon_next, icon_pause,
    icon_play, icon_prev, icon_check, rounded, text_at,
)

L = 8
PANEL_W = 258
BOTTOM_H = 176
WORLD_W = 540       # preferred width of the worldline panel
WORLD_W_MIN = 300   # below this the panel is not worth showing
# canvas the grid keeps for itself: 2*40 padding plus 8 cells of >= 48 px
GRID_CANVAS_MIN = 464
STATE_LINES = 3
LINE_H = 30
STEP_SECONDS = 0.95
TWEEN_SECONDS = 0.42
# an auto-played reduction fits its whole script into roughly this long
AUTO_TOTAL_SECONDS = 3.5
AUTO_STEP_MIN, AUTO_STEP_MAX = 0.09, 0.30


def _enable_dpi_awareness() -> None:
    """Ask Windows for real pixels instead of a stretched, blurry window."""
    try:
        import ctypes

        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor v1
        except (AttributeError, OSError):
            ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass  # not Windows, or the call is unavailable: harmless


class App:
    def __init__(self) -> None:
        _enable_dpi_awareness()
        pygame.init()
        pygame.display.set_caption("Toric Code")
        info = pygame.display.Info()
        self.W = min(1440, max(1100, info.current_w - 80))
        self.H = min(940, max(780, info.current_h - 120))
        self.screen = pygame.display.set_mode((self.W, self.H))
        self.clock = pygame.time.Clock()
        self.fonts = Fonts()

        self.lat = TorusLattice(L)
        self.word = PauliWord()
        self.tracker = WorldlineTracker(self.lat)
        self.split = False
        self.undo_stack: list[PauliWord] = []
        self.tool = "Z"
        self.message = "Pick a gate and click a qubit, or drag an anyon."

        # animation state
        self.script = None
        self.step = 0
        self.playing = False
        self.tween = 1.0
        self.step_timer = 0.0
        self.scroll = 0
        self.auto = False           # play through and commit without clicks
        self.finish_timer = 0.0
        self.step_secs = STEP_SECONDS
        self.tween_secs = TWEEN_SECONDS

        # drag state
        self.drag_kind: str | None = None
        self.drag_src: int | None = None
        self.drag_pos: tuple[int, int] | None = None
        self.drag_target: int | None = None
        self.drag_pair: tuple[int, int] | None = None    # bound (vertex, face)
        self.drag_pair_target: tuple[int, int, int] | None = None  # (v, f, qubit)
        self.mouse_pos = (0, 0)

        self.time = 0.0
        self.base_W = self.W
        self.max_W = max(self.W, min(2200, info.current_w - 60))
        self.world = WorldlinePanel(self.lat, self.tracker, self.fonts)
        self._layout_grid()
        self._build_buttons()

    # ------------------------------------------------------------------
    # layout
    # ------------------------------------------------------------------
    @property
    def world_w(self) -> int:
        """Panel width, shrunk if the screen could not spare room to widen.

        The grid keeps ``GRID_CANVAS_MIN`` px whatever happens, so opening the
        panel never squeezes the lattice down to something unusable.
        """
        if not self.split:
            return 0
        room = self.W - PANEL_W - GRID_CANVAS_MIN
        return max(WORLD_W_MIN, min(WORLD_W, room))

    @property
    def canvas_right(self) -> int:
        """Left edge of the worldline panel, or of the side panel when closed."""
        return self.W - PANEL_W - self.world_w

    def world_rect(self) -> pygame.Rect | None:
        if not self.split:
            return None
        return pygame.Rect(self.canvas_right, 0, self.world_w, self.H)

    def _layout_grid(self) -> None:
        pad = 40
        avail_w = self.canvas_right - 2 * pad
        avail_h = self.H - BOTTOM_H - pad - 28
        self.spacing = min(avail_w / L, avail_h / L)
        span = self.spacing * L
        self.ox = pad + (avail_w - span) / 2
        self.oy = 30 + (avail_h - span) / 2
        self.qr = max(9, min(16, self.spacing * 0.17))

    def slot_px(self, col: float, row: float) -> tuple[float, float]:
        return self.ox + col * self.spacing, self.oy + row * self.spacing

    def qubit_center(self, q: int) -> tuple[float, float]:
        g = self.lat.edge_geom[q]
        ax, ay = self.slot_px(*g.a)
        bx, by = self.slot_px(*g.b)
        return (ax + bx) / 2, (ay + by) / 2

    def _build_buttons(self) -> None:
        x = self.W - PANEL_W + 26
        w = PANEL_W - 52
        self.buttons: dict[str, Button] = {}

        y = 104
        gap = 8
        gw = (w - 2 * gap) // 3
        for i, gate in enumerate(("X", "Y", "Z")):
            col = GATE_COL[gate]
            self.buttons[gate] = Button(
                gate, gate, pygame.Rect(x + i * (gw + gap), y, gw, 44),
                fill=(246, 247, 249), fill_on=col,
                text_col=col, text_on=WHITE if gate != "Y" else INK,
            )
        y += 52
        self.buttons["Drag"] = Button(
            "Drag", "Drag anyon", pygame.Rect(x, y, w, 42),
            fill=(246, 247, 249), fill_on=ACCENT,
        )
        y += 70
        actions = (
            ("Simplify", "Simplify"), ("Reduce", "Reduce qubits"),
            ("World", "Worldlines 3D"), ("Undo", "Undo"), ("Clear", "Clear"),
        )
        for key, label in actions:
            fill = ACCENT if key == "Simplify" else (238, 240, 243)
            tcol = WHITE if key == "Simplify" else INK
            self.buttons[key] = Button(
                key, label, pygame.Rect(x, y, w, 42), fill=fill, text_col=tcol,
                fill_on=ACCENT, text_on=WHITE,
            )
            y += 46
        self.panel_info_y = y + 10

        # transport, bottom right
        bx = self.W - 40
        by = self.H - 58
        size = 36
        specs = [
            ("Close", icon_close), ("Next", icon_next),
            ("Play", icon_play), ("Prev", icon_prev),
        ]
        for key, icon in specs:
            bx -= size + 10
            self.buttons[key] = Button(
                key, "", pygame.Rect(bx, by, size, size),
                fill=(240, 241, 244), icon=icon,
            )

    # ------------------------------------------------------------------
    # state helpers
    # ------------------------------------------------------------------
    @property
    def animating(self) -> bool:
        return self.script is not None

    def current_snapshot(self):
        return self.script.snapshots[self.step] if self.script else None

    def display_word(self) -> tuple[list, complex]:
        if self.script:
            s = self.current_snapshot()
            return s.factors, s.phase
        return self.word.factors, self.word.phase

    def display_net(self) -> dict[int, str]:
        if self.script:
            tmp = PauliWord()
            tmp.factors = list(self.current_snapshot().factors)
            return tmp.net()
        return self.word.net()

    def push_undo(self) -> None:
        self.undo_stack.append(self.word.clone())
        if len(self.undo_stack) > 200:
            self.undo_stack.pop(0)

    # ------------------------------------------------------------------
    # actions
    # ------------------------------------------------------------------
    def apply_gate(self, qubit: int, gate: str) -> None:
        self.push_undo()
        self.tracker.begin_step()
        self.word.apply_gate(qubit, gate)
        # a Z component moves e charges, an X component moves m fluxes
        self.tracker.flip(qubit, gate in ("Z", "Y"), gate in ("X", "Y"))
        self.message = f"Applied {gate} to qubit {qubit + 1}."

    def erase(self, qubit: int) -> None:
        if not any(f.qubit == qubit for f in self.word.factors):
            return
        self.push_undo()
        self.tracker.begin_step()
        before = self.word.net().get(qubit, "I")
        self.word.erase_qubit(qubit)
        self.tracker.flip(qubit, before in ("Z", "Y"), before in ("X", "Y"))
        self.message = f"Cleared qubit {qubit + 1}."

    def do_simplify(self) -> None:
        script, msg = build_script(self.lat, self.word)
        self._start_script(script, msg, auto=False)

    def do_reduce(self) -> None:
        """Gather and multiply out each qubit's operators, played automatically."""
        script, msg = build_reduce_script(self.word)
        self._start_script(script, msg, auto=True)

    def _start_script(self, script, msg: str, auto: bool) -> None:
        self.message = msg
        if script is None:
            return
        self.script = script
        self.step = 0
        self.tween = 1.0
        self.step_timer = 0.0
        self.finish_timer = 0.0
        self.scroll = 0
        self.auto = auto
        self.playing = auto
        if auto:
            # keep the whole run to about a fixed length, whatever its size
            self.step_secs = max(
                AUTO_STEP_MIN,
                min(AUTO_STEP_MAX, AUTO_TOTAL_SECONDS / max(1, len(script.snapshots))),
            )
            self.tween_secs = self.step_secs * 0.7
        else:
            self.step_secs, self.tween_secs = STEP_SECONDS, TWEEN_SECONDS

    def toggle_split(self) -> None:
        """Show or hide the worldline panel beside the grid."""
        self.split = not self.split
        # widen the window to make room, so the grid keeps its size; if the
        # screen cannot spare the pixels the grid shrinks instead
        want = self.base_W + WORLD_W if self.split else self.base_W
        self.W = min(self.max_W, want)
        self.screen = pygame.display.set_mode((self.W, self.H))
        self._layout_grid()
        self._build_buttons()
        if self.split:
            self.world.reset_view()
            self.message = "Worldlines shown - the grid stays live."
        else:
            self.message = "Worldline panel hidden."

    def close_script(self) -> None:
        if self.script and self.step == len(self.script.snapshots) - 1:
            last = self.script.snapshots[-1]
            self.push_undo()
            self.tracker.begin_step()
            self.word.factors = list(last.factors)
            self.word.phase = last.phase
            self.message = "Simplified."
        else:
            self.message = "Simplification cancelled."
        self.script = None
        self.playing = False
        self.auto = False
        self.scroll = 0
        self.step_secs, self.tween_secs = STEP_SECONDS, TWEEN_SECONDS

    def goto_step(self, delta: int) -> None:
        if not self.script:
            return
        new = self.step + delta
        if 0 <= new < len(self.script.snapshots):
            self.step = new
            self.tween = 0.0
            self.step_timer = 0.0
            if new == len(self.script.snapshots) - 1:
                self.playing = False

    # ------------------------------------------------------------------
    # hit testing
    # ------------------------------------------------------------------
    def qubit_at(self, pos) -> int | None:
        best, bestd = None, (self.qr + 7) ** 2
        for q in range(self.lat.n_qubits):
            cx, cy = self.qubit_center(q)
            d = (cx - pos[0]) ** 2 + (cy - pos[1]) ** 2
            if d < bestd:
                best, bestd = q, d
        return best

    def vertex_at(self, pos, tol: float = 0.34) -> int | None:
        """Vertex under the cursor. ``tol`` is a fraction of the grid spacing;
        pass 0.5 while dragging so the drop always snaps to the nearest slot."""
        col = round((pos[0] - self.ox) / self.spacing)
        row = round((pos[1] - self.oy) / self.spacing)
        if not (0 <= col <= L and 0 <= row <= L):
            return None
        px, py = self.slot_px(col, row)
        if (px - pos[0]) ** 2 + (py - pos[1]) ** 2 > (self.spacing * tol) ** 2:
            return None
        return self.lat.vertex(col % L, row % L)

    def face_at(self, pos) -> int | None:
        col = math.floor((pos[0] - self.ox) / self.spacing)
        row = math.floor((pos[1] - self.oy) / self.spacing)
        if not (0 <= col < L and 0 <= row < L):
            return None
        return self.lat.face(col, row)

    # ------------------------------------------------------------------
    # events
    # ------------------------------------------------------------------
    def eps_center(self, vertex: int, face: int, near=None) -> tuple[float, float]:
        """Screen point for a bound pair: midway between the charge and flux.

        Both sites can be drawn in several ghost copies, so pick the copies
        that lie closest together (optionally near a reference point).
        """
        v_pts = [self.slot_px(*sl) for sl in self.lat.vertex_slots(vertex)]
        f_pts = [self.slot_px(sl[0] + 0.5, sl[1] + 0.5)
                 for sl in self.lat.face_slots(face)]
        best, bestd = None, None
        for vp in v_pts:
            for fp in f_pts:
                d = (vp[0] - fp[0]) ** 2 + (vp[1] - fp[1]) ** 2
                if near is not None:
                    mid = ((vp[0] + fp[0]) / 2, (vp[1] + fp[1]) / 2)
                    d += 0.15 * ((mid[0] - near[0]) ** 2 + (mid[1] - near[1]) ** 2)
                if bestd is None or d < bestd:
                    best, bestd = ((vp[0] + fp[0]) / 2, (vp[1] + fp[1]) / 2), d
        return best

    def pick_bound_pair(self, pos, e_sites, m_sites):
        """Find an e and an adjacent m to grab together, nearest the cursor."""
        cands: list[tuple[float, tuple[int, int]]] = []
        v = self.vertex_at(pos)
        if v is not None and v in e_sites:
            for f in self.lat.vertex_faces(v):
                if f in m_sites:
                    c = self.eps_center(v, f, near=pos)
                    cands.append(((c[0] - pos[0]) ** 2 + (c[1] - pos[1]) ** 2, (v, f)))
        f = self.face_at(pos)
        if f is not None and f in m_sites:
            for vv in self.lat.face_vertices(f):
                if vv in e_sites:
                    c = self.eps_center(vv, f, near=pos)
                    cands.append(((c[0] - pos[0]) ** 2 + (c[1] - pos[1]) ** 2, (vv, f)))
        if not cands:
            return None
        return min(cands)[1]

    def eps_best_move(self, pos):
        """The Y move whose landing spot is nearest the cursor."""
        if self.drag_pair is None:
            return None
        v, f = self.drag_pair
        best, bestd = None, None
        for q in self.lat.epsilon_moves(v, f):
            nv, nf = self.lat.epsilon_step(q, v, f)
            c = self.eps_center(nv, nf, near=pos)
            d = (c[0] - pos[0]) ** 2 + (c[1] - pos[1]) ** 2
            if bestd is None or d < bestd:
                best, bestd = (nv, nf, q), d
        # only commit once the cursor has actually left the current spot
        here = self.eps_center(v, f, near=pos)
        if bestd is not None and bestd >= (here[0] - pos[0]) ** 2 + (here[1] - pos[1]) ** 2:
            return None
        return best

    def on_mouse_down(self, event) -> None:
        pos = event.pos
        for key, b in self.buttons.items():
            if self._button_visible(key) and b.hit(pos):
                self.on_button(key)
                return
        if self.animating:
            return
        if pos[0] > self.canvas_right or pos[1] > self.H - BOTTOM_H:
            return

        if event.button == 3:
            q = self.qubit_at(pos)
            if q is not None:
                self.erase(q)
            return
        if event.button != 1:
            return

        e_sites, m_sites = self.lat.syndrome(self.word.net())
        if self.tool == "Drag":
            # real mouse events carry no modifier state, so read the keyboard;
            # synthetic events may set .mod directly
            mods = getattr(event, "mod", None)
            if mods is None:
                mods = pygame.key.get_mods()
            if mods & pygame.KMOD_SHIFT:
                pair = self.pick_bound_pair(pos, e_sites, m_sites)
                if pair is not None:
                    self.drag_kind, self.drag_pair = "eps", pair
                    self.drag_pos, self.drag_pair_target = pos, None
                    self.message = (
                        f"Dragging a bound e-m pair (epsilon): "
                        f"e on a vertex, m in face {pair[1] + 1}."
                    )
                    return
                self.message = (
                    "No e next to an m here - shift-drag needs both. "
                    "A Y gate makes a bound pair."
                )
            v = self.vertex_at(pos)
            if v is not None and v in e_sites:
                self.drag_kind, self.drag_src, self.drag_pos = "e", v, pos
                return
            f = self.face_at(pos)
            if f is not None and f in m_sites:
                self.drag_kind, self.drag_src, self.drag_pos = "m", f, pos
                return
            self.message = "Press on an anyon (e on a vertex, m in a face) to drag it."
            return

        q = self.qubit_at(pos)
        if q is not None:
            self.apply_gate(q, self.tool)

    def on_mouse_up(self, event) -> None:
        if self.drag_kind is None:
            return
        pos = event.pos
        kind, src = self.drag_kind, self.drag_src
        pair, pair_target = self.drag_pair, self.drag_pair_target
        self.drag_kind = self.drag_src = self.drag_pos = None
        self.drag_target = None
        self.drag_pair = self.drag_pair_target = None

        if kind == "eps":
            move = pair_target or self.eps_best_move(pos)
            if move is None or pair is None:
                return
            _, _, q = move
            self.apply_gate(q, "Y")
            self.message = (
                f"Moved the bound e-m pair across qubit {q + 1} "
                f"(Y = i X Z moves both)."
            )
            return
        if kind == "e":
            dst = self.vertex_at(pos, tol=0.5)
            if dst is None or dst == src:
                return
            q = self.lat.edge_between_vertices(src, dst)
            if q is None:
                self.message = "An e charge can only hop to a neighbouring vertex."
                return
            self.apply_gate(q, "Z")
            self.message = f"Moved e along qubit {q + 1} (Z applied)."
        else:
            dst = self.face_at(pos)
            if dst is None or dst == src:
                return
            q = self.lat.edge_between_faces(src, dst)
            if q is None:
                self.message = "An m flux can only hop to a neighbouring face."
                return
            self.apply_gate(q, "X")
            self.message = f"Moved m across qubit {q + 1} (X applied)."

    def on_mouse_motion(self, event) -> None:
        for key, b in self.buttons.items():
            b.hover = self._button_visible(key) and b.hit(event.pos)
        if self.drag_kind is None:
            return
        self.drag_pos = event.pos
        if self.drag_kind == "eps":
            self.drag_pair_target = self.eps_best_move(event.pos)
            return
        if self.drag_kind == "e":
            dst = self.vertex_at(event.pos, tol=0.5)
            ok = dst is not None and dst != self.drag_src and \
                self.lat.edge_between_vertices(self.drag_src, dst) is not None
        else:
            dst = self.face_at(event.pos)
            ok = dst is not None and dst != self.drag_src and \
                self.lat.edge_between_faces(self.drag_src, dst) is not None
        self.drag_target = dst if ok else None

    def on_button(self, key: str) -> None:
        if key in ("X", "Y", "Z", "Drag"):
            self.tool = key
            self.message = (
                "Drag an e charge between vertices, or an m flux between faces."
                if key == "Drag" else f"{key} tool selected."
            )
        elif key == "Simplify":
            self.do_simplify()
        elif key == "Reduce":
            self.do_reduce()
        elif key == "World":
            self.toggle_split()
        elif key == "Undo":
            if self.undo_stack:
                self.word = self.undo_stack.pop()
                self.tracker.undo_step()
                self.message = "Undone."
            else:
                self.message = "Nothing to undo."
        elif key == "Clear":
            if self.word.factors or not self.tracker.is_empty():
                self.push_undo()
                self.tracker.begin_reset_step()
            self.word.clear()
            self.message = "Cleared board and worldline history."
        elif key == "Play":
            self.auto = False
            self.playing = not self.playing
            if self.playing and self.step == len(self.script.snapshots) - 1:
                self.step = 0
                self.tween = 0.0
        elif key == "Next":
            self.auto = False
            self.goto_step(1)
        elif key == "Prev":
            self.auto = False
            self.goto_step(-1)
        elif key == "Close":
            self.close_script()

    def on_key(self, event) -> None:
        k = event.key
        if k == pygame.K_ESCAPE:
            if self.script:
                self.close_script()
            else:
                pygame.event.post(pygame.event.Event(pygame.QUIT))
        elif self.script:
            if k == pygame.K_SPACE:
                self.on_button("Play")
            elif k == pygame.K_RIGHT:
                self.goto_step(1)
            elif k == pygame.K_LEFT:
                self.goto_step(-1)
        else:
            mapping = {
                pygame.K_x: "X", pygame.K_y: "Y", pygame.K_z: "Z",
                pygame.K_d: "Drag", pygame.K_s: "Simplify",
                pygame.K_r: "Reduce", pygame.K_u: "Undo", pygame.K_c: "Clear",
                pygame.K_w: "World",
            }
            if k == pygame.K_f and self.split:
                self.world.show_floor = not self.world.show_floor
            elif k in mapping:
                self.on_button(mapping[k])

    def _button_visible(self, key: str) -> bool:
        transport = key in ("Play", "Next", "Prev", "Close")
        if transport:
            return self.animating
        if self.animating:
            return False
        return True

    # ------------------------------------------------------------------
    # update
    # ------------------------------------------------------------------
    def update(self, dt: float) -> None:
        self.time += dt
        if self.tween < 1.0:
            self.tween = min(1.0, self.tween + dt / self.tween_secs)
        if self.script and self.playing:
            self.step_timer += dt
            if self.step_timer >= self.step_secs:
                self.step_timer = 0.0
                if self.step < len(self.script.snapshots) - 1:
                    self.goto_step(1)
                else:
                    self.playing = False
        # an auto-played run commits itself once it has settled on the last step
        if self.auto and self.script and self.step == len(self.script.snapshots) - 1:
            self.finish_timer += dt
            if self.finish_timer >= 0.7:
                self.close_script()
        for key, b in self.buttons.items():
            if key in ("X", "Y", "Z", "Drag"):
                b.selected = self.tool == key
            elif key == "World":
                b.selected = self.split
            b.enabled = self._button_visible(key)
        if self.script:
            last = self.step == len(self.script.snapshots) - 1
            self.buttons["Play"].icon = icon_pause if self.playing else icon_play
            self.buttons["Close"].icon = icon_check if last else icon_close
            self.buttons["Close"].fill = (222, 240, 230) if last else (240, 241, 244)
            self.buttons["Close"].text_col = (28, 122, 86) if last else INK
            self.buttons["Prev"].enabled = self.step > 0
            self.buttons["Next"].enabled = self.step < len(self.script.snapshots) - 1

    # ------------------------------------------------------------------
    # drawing
    # ------------------------------------------------------------------
    def draw(self) -> None:
        self.screen.fill(BG)
        self.draw_grid()
        if self.split:
            self.world.draw(self.screen, self.world_rect())
        self.draw_panel()
        self.draw_bottom()
        pygame.display.flip()

    def draw_grid(self) -> None:
        surf = self.screen
        net = self.display_net()
        snap = self.current_snapshot()
        loop_qubits = snap.loop_qubits if snap else set()
        pulse = 0.5 + 0.5 * math.sin(self.time * 4.0)

        # lattice edges
        for q in range(self.lat.n_qubits):
            g = self.lat.edge_geom[q]
            a = self.slot_px(*g.a)
            b = self.slot_px(*g.b)
            ghost = max(g.a[0], g.b[0]) == L or max(g.a[1], g.b[1]) == L
            letter = net.get(q)
            if letter:
                col = GATE_COL[letter]
                pygame.draw.line(surf, blend(col, BG, 0.55), a, b, 5)
            else:
                pygame.draw.line(surf, GRID_GHOST if ghost else GRID, a, b, 2)

        # vertices
        for i in range(self.lat.n_vertices):
            for slot in self.lat.vertex_slots(i):
                p = self.slot_px(*slot)
                ghost = slot[0] == L or slot[1] == L
                aa_circle(surf, GRID_GHOST if ghost else GRID, p, 3.2)

        # loop halos
        for q in loop_qubits:
            cx, cy = self.qubit_center(q)
            r = self.qr + 6 + 2.5 * pulse
            aa_circle(surf, blend(GLOW, BG, 0.25), (cx, cy), r, 3)

        # qubit circles
        for q in range(self.lat.n_qubits):
            cx, cy = self.qubit_center(q)
            letter = net.get(q)
            if letter:
                col = GATE_COL[letter]
                aa_circle(surf, col, (cx, cy), self.qr + 2.5)
                text_at(
                    surf, self.fonts.gate if self.qr > 11 else self.fonts.gate_small,
                    letter, INK if letter == "Y" else WHITE, (cx, cy - 1), "center",
                )
            else:
                aa_circle(surf, (240, 240, 243), (cx, cy), self.qr)
                aa_circle(surf, GRID, (cx, cy), self.qr, 2)

        # anyons
        e_sites, m_sites = self.lat.syndrome(net)
        held_v, held_f = self.drag_pair if (
            self.drag_kind == "eps" and self.drag_pos) else (None, None)
        for v in e_sites:
            if self.drag_kind == "e" and v == self.drag_src and self.drag_pos:
                continue
            if v == held_v:
                continue
            for slot in self.lat.vertex_slots(v):
                self.draw_e(self.slot_px(*slot))
        for f in m_sites:
            if self.drag_kind == "m" and f == self.drag_src and self.drag_pos:
                continue
            if f == held_f:
                continue
            for slot in self.lat.face_slots(f):
                px, py = self.slot_px(slot[0] + 0.5, slot[1] + 0.5)
                self.draw_m((px, py))

        self.draw_drag()

    def draw_e(self, pos, alpha: float = 1.0) -> None:
        r = self.qr + 2
        col = E_COL if alpha >= 1.0 else blend(E_COL, BG, 1 - alpha)
        aa_circle(self.screen, col, pos, r)
        text_at(self.screen, self.fonts.anyon, "e", WHITE, (pos[0], pos[1] - 1), "center")

    def draw_m(self, pos, alpha: float = 1.0) -> None:
        r = self.qr + 2
        col = M_COL if alpha >= 1.0 else blend(M_COL, BG, 1 - alpha)
        rect = pygame.Rect(0, 0, round(r * 2), round(r * 2))
        rect.center = (round(pos[0]), round(pos[1]))
        rounded(self.screen, col, rect, max(5, round(r * 0.55)))
        text_at(self.screen, self.fonts.anyon, "m", WHITE, (pos[0], pos[1] - 1), "center")

    def draw_eps(self, pos, alpha: float = 1.0) -> None:
        """A bound e-m pair: the charge and the flux overlapping."""
        r = max(7.5, self.qr * 0.78)
        off = r * 0.74
        e_col = E_COL if alpha >= 1 else blend(E_COL, BG, 1 - alpha)
        m_col = M_COL if alpha >= 1 else blend(M_COL, BG, 1 - alpha)
        rect = pygame.Rect(0, 0, round(r * 2), round(r * 2))
        rect.center = (round(pos[0] + off), round(pos[1] - off))
        rounded(self.screen, BG, rect.inflate(5, 5), max(5, round(r * 0.6)))
        rounded(self.screen, m_col, rect, max(4, round(r * 0.55)))
        text_at(self.screen, self.fonts.gate_small, "m", WHITE,
                (rect.centerx, rect.centery - 1), "center")
        c = (pos[0] - off, pos[1] + off)
        aa_circle(self.screen, BG, c, r + 2.5)
        aa_circle(self.screen, e_col, c, r)
        text_at(self.screen, self.fonts.gate_small, "e", WHITE,
                (c[0], c[1] - 1), "center")

    def draw_drag(self) -> None:
        if self.drag_kind is None or self.drag_pos is None:
            return
        if self.drag_kind == "eps":
            src = self.eps_center(*self.drag_pair, near=self.drag_pos)                 if self.drag_pair is not None else None
            if self.drag_pair_target is not None:
                nv, nf, q = self.drag_pair_target
                tgt = self.eps_center(nv, nf, near=self.drag_pos)
                if src is not None:
                    pygame.draw.line(self.screen, blend(Y_COL, BG, 0.5), src, tgt, 4)
                # the qubit that will take the Y
                qx, qy = self.qubit_center(q)
                aa_circle(self.screen, blend(Y_COL, BG, 0.3), (qx, qy), self.qr + 6, 3)
                self.draw_eps(tgt, alpha=0.38)
            self.draw_eps(self.drag_pos)
            return
        src_pos = None
        if self.drag_kind == "e":
            slots = self.lat.vertex_slots(self.drag_src)
            src_pos = min(
                (self.slot_px(*s) for s in slots),
                key=lambda p: (p[0] - self.drag_pos[0]) ** 2 + (p[1] - self.drag_pos[1]) ** 2,
            )
        else:
            slots = self.lat.face_slots(self.drag_src)
            src_pos = min(
                (self.slot_px(s[0] + 0.5, s[1] + 0.5) for s in slots),
                key=lambda p: (p[0] - self.drag_pos[0]) ** 2 + (p[1] - self.drag_pos[1]) ** 2,
            )
        col = E_COL if self.drag_kind == "e" else M_COL
        if self.drag_target is not None:
            if self.drag_kind == "e":
                cands = [self.slot_px(*s) for s in self.lat.vertex_slots(self.drag_target)]
            else:
                cands = [
                    self.slot_px(s[0] + 0.5, s[1] + 0.5)
                    for s in self.lat.face_slots(self.drag_target)
                ]
            tgt = min(cands, key=lambda p: (p[0] - src_pos[0]) ** 2 + (p[1] - src_pos[1]) ** 2)
            pygame.draw.line(self.screen, blend(col, BG, 0.55), src_pos, tgt, 4)
            aa_circle(self.screen, blend(col, BG, 0.4), tgt, self.qr + 4, 3)
        if self.drag_kind == "e":
            self.draw_e(self.drag_pos)
        else:
            self.draw_m(self.drag_pos)

    # ------------------------------------------------------------------
    def draw_panel(self) -> None:
        surf = self.screen
        x0 = self.W - PANEL_W
        rounded(surf, PANEL, pygame.Rect(x0, 0, PANEL_W, self.H), 0)
        pygame.draw.line(surf, HAIRLINE, (x0, 0), (x0, self.H), 1)

        text_at(surf, self.fonts.title, "Toric code", INK, (x0 + 26, 38))
        text_at(surf, self.fonts.small, f"{L}x{L} torus  .  {self.lat.n_qubits} qubits",
                INK_SOFT, (x0 + 26, 66))
        if not self.animating:
            text_at(surf, self.fonts.tiny, "TOOL", INK_FAINT, (x0 + 26, 88))

        for key, b in self.buttons.items():
            if self._button_visible(key) and key not in ("Play", "Next", "Prev", "Close"):
                b.draw(surf, self.fonts)

        if self.animating:
            snap = self.current_snapshot()
            kind = self.script.kind
            y = 104
            text_at(surf, self.fonts.tiny, "SIMPLIFYING", INK_FAINT, (x0 + 26, 88))
            if kind == "reduce":
                head, sub = "Same-qubit reduction", "gather, then multiply out"
            elif kind == "pair":
                head, sub = "Repeated factor", "same qubit, same letter"
            else:
                head = f"Contractible {kind}-loop"
                sub = "product of " + ("stars A_v" if kind == "X" else "plaquettes B_p")
            text_at(surf, self.fonts.label_bold, head, INK, (x0 + 26, y))
            text_at(surf, self.fonts.small, sub, INK_SOFT, (x0 + 26, y + 22))
            y += 54

            labels = sorted(q + 1 for q in self.script.loop_qubits)
            text_at(surf, self.fonts.tiny,
                    "QUBITS" if kind != "reduce" else "QUBITS TOUCHED",
                    INK_FAINT, (x0 + 26, y))
            y += 20
            line = ""
            for lab in labels:
                trial = (line + ", " + str(lab)) if line else str(lab)
                if self.fonts.small.size(trial)[0] > PANEL_W - 56:
                    text_at(surf, self.fonts.small, line, INK_SOFT, (x0 + 26, y))
                    y += 19
                    line = str(lab)
                else:
                    line = trial
            if line:
                text_at(surf, self.fonts.small, line, INK_SOFT, (x0 + 26, y))
                y += 19

            y += 16
            text_at(surf, self.fonts.tiny, "PHASE", INK_FAINT, (x0 + 26, y))
            y += 20
            name = {1: "+1", -1: "-1", 1j: "+i", -1j: "-i"}.get(snap.phase, str(snap.phase))
            col = X_COL if snap.phase in (-1, -1j) else INK
            text_at(surf, self.fonts.title, name, col, (x0 + 26, y))
            y += 40

            pygame.draw.line(surf, HAIRLINE, (x0 + 26, y), (self.W - 26, y), 1)
            y += 16
            total = len(self.script.snapshots)
            bar = pygame.Rect(x0 + 26, y, PANEL_W - 52, 6)
            rounded(surf, (238, 239, 242), bar, 3)
            done = pygame.Rect(bar.x, bar.y, round(bar.w * (self.step + 1) / total), 6)
            rounded(surf, ACCENT, done, 3)
            y += 18
            text_at(surf, self.fonts.small, f"step {self.step + 1} of {total}",
                    INK_SOFT, (x0 + 26, y))
            y += 30
            hints = ["space  play / pause", "arrows  step", "esc  close"]
            if self.auto:
                hints.insert(0, "playing automatically")
            for ln in hints:
                text_at(surf, self.fonts.small, ln, INK_FAINT, (x0 + 26, y))
                y += 19
            return

        y = self.panel_info_y
        pygame.draw.line(surf, HAIRLINE, (x0 + 26, y - 14), (self.W - 26, y - 14), 1)
        text_at(surf, self.fonts.tiny, "STABILIZERS", INK_FAINT, (x0 + 26, y))
        y += 22
        for label, formula, col in (
            ("A_v", "X X X X   star", X_COL),
            ("B_p", "Z Z Z Z   plaquette", Z_COL),
        ):
            text_at(surf, self.fonts.label_bold, label, col, (x0 + 26, y))
            text_at(surf, self.fonts.small, formula, INK_SOFT, (x0 + 66, y + 2))
            y += 23

        y += 8
        net = self.display_net()
        e_sites, m_sites = self.lat.syndrome(net)
        text_at(surf, self.fonts.tiny, "EXCITATIONS", INK_FAINT, (x0 + 26, y))
        y += 22
        for name, count, col, note in (
            ("e", len(e_sites), E_COL, "vertices, from Z"),
            ("m", len(m_sites), M_COL, "faces, from X"),
        ):
            aa_circle(surf, col, (x0 + 34, y + 8), 9)
            text_at(surf, self.fonts.gate_small, name, WHITE, (x0 + 34, y + 7), "center")
            text_at(surf, self.fonts.label_bold, str(count), INK, (x0 + 52, y))
            text_at(surf, self.fonts.small, note, INK_SOFT, (x0 + 76, y + 2))
            y += 24

        y += 8
        text_at(surf, self.fonts.tiny, "KEYS", INK_FAINT, (x0 + 26, y))
        y += 20
        for line in ("x y z  tools     d  drag",
                     "s simplify     r reduce qubits",
                     "w worldlines     u undo     c clear",
                     "shift-drag moves a bound e+m",
                     "right-click a qubit to clear it"):
            if y + 18 > self.H - 10:
                break
            text_at(surf, self.fonts.small, line, INK_SOFT, (x0 + 26, y))
            y += 18

    # ------------------------------------------------------------------
    def draw_bottom(self) -> None:
        surf = self.screen
        top = self.H - BOTTOM_H
        pygame.draw.line(surf, HAIRLINE, (0, top), (self.canvas_right, top), 1)
        text_at(surf, self.fonts.tiny, "STATE", INK_FAINT, (40, top + 18))

        area = pygame.Rect(40, top + 38, self.canvas_right - 80, STATE_LINES * LINE_H)
        self.draw_state_line(area)

        cap_y = self.H - 40
        caption = self.message
        if self.script:
            snap = self.current_snapshot()
            caption = snap.caption
            text_at(
                surf, self.fonts.tiny,
                f"STEP {self.step + 1} / {len(self.script.snapshots)}",
                INK_FAINT, (40, cap_y - 20),
            )
        text_at(surf, self.fonts.caption, caption, INK_SOFT, (40, cap_y))

        for key in ("Prev", "Play", "Next", "Close"):
            if self._button_visible(key):
                self.buttons[key].draw(surf, self.fonts)

    def token_layout(self, factors, phase, area: pygame.Rect):
        """Lay the word out as wrapped tokens -> {key: (x, y, w, h, text, bold)}."""
        font = self.fonts.state
        out: dict = {}
        x, y = area.x, area.y
        gap = 9

        def place(key, s, bold=False):
            nonlocal x, y
            f = self.fonts.state_bold if bold else font
            w, h = f.size(s)
            if x > area.x and x + w > area.right:
                x = area.x
                y += LINE_H
            out[key] = (x, y, w, h, s, bold)
            x += w + gap

        prefix = phase_prefix(phase)
        if prefix:
            place("__phase", prefix, True)
        for f in factors:
            place(f.uid, f.label())
        place("__ket", "|00_L>", True)
        n_lines = (y - area.y) // LINE_H + 1
        return out, n_lines

    def draw_state_line(self, area: pygame.Rect) -> None:
        surf = self.screen
        factors, phase = self.display_word()
        layout, n_lines = self.token_layout(factors, phase, area)

        prev_layout = {}
        snap = self.current_snapshot()
        if self.script and self.step > 0 and self.tween < 1.0:
            p = self.script.snapshots[self.step - 1]
            prev_layout = self.token_layout(p.factors, p.phase, area)[0]

        # keep the interesting token on screen
        focus_line = None
        if snap is not None:
            focus = snap.moving
            if focus is not None and focus in layout:
                focus_line = (layout[focus][1] - area.y) // LINE_H
            elif snap.bracket is not None and factors:
                key = factors[snap.bracket[0]].uid
                if key in layout:
                    focus_line = (layout[key][1] - area.y) // LINE_H
        if focus_line is None:
            focus_line = max(0, n_lines - 1)
        max_scroll = max(0, n_lines - STATE_LINES)
        self.scroll = max(0, min(max_scroll, focus_line - STATE_LINES + 1))
        dy = -self.scroll * LINE_H

        clip = surf.get_clip()
        surf.set_clip(area.inflate(24, 10))

        t = _ease(self.tween)
        highlight = snap.highlight if snap else set()
        flash = snap.flash if snap else set()
        moving = snap.moving if snap else None

        # bracket behind the gathered stabilizer
        if snap is not None and snap.bracket is not None and factors:
            lo, hi = snap.bracket
            rows: dict[int, list[float]] = {}
            for f in factors[lo:hi + 1]:
                if f.uid not in layout:
                    continue
                x, y, w, h, _, _ = layout[f.uid]
                row = rows.setdefault(y, [x, x + w])
                row[0] = min(row[0], x)
                row[1] = max(row[1], x + w)
            for y, (a, b) in rows.items():
                rect = pygame.Rect(round(a - 8), round(y + dy - 4), round(b - a + 16), LINE_H - 2)
                rounded(surf, blend(GLOW, BG, 0.62), rect, 12)

        for key, (x, y, w, h, s, bold) in layout.items():
            alpha = 1.0
            px, py = x, y
            if key in prev_layout:
                qx, qy = prev_layout[key][0], prev_layout[key][1]
                px = qx + (x - qx) * t
                py = qy + (y - qy) * t
            elif prev_layout:
                alpha = t
            py += dy

            col = INK
            pill = None
            if key == moving:
                pill, col = ACCENT, WHITE
            elif key in flash:
                pill, col = X_COL, WHITE
            elif key in highlight:
                pill, col = blend(GLOW, BG, 0.35), INK
            elif key == "__ket":
                col = INK
            elif isinstance(key, int):
                col = INK

            if pill is not None:
                rect = pygame.Rect(round(px - 7), round(py - 3), round(w + 14), round(h + 6))
                rounded(surf, pill, rect, (h + 6) // 2)
            if alpha < 1.0:
                col = blend(BG, col, alpha)
            font = self.fonts.state_bold if bold else self.fonts.state
            text_at(surf, font, s, col, (px, py))

        surf.set_clip(clip)

        if max_scroll > 0:
            text_at(
                surf, self.fonts.tiny,
                f"line {self.scroll + 1}-{min(n_lines, self.scroll + STATE_LINES)} of {n_lines}",
                INK_FAINT, (area.right - 4, area.y - 16), "topright",
            )

    # ------------------------------------------------------------------
    _PANEL_MOUSE = (
        pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP,
        pygame.MOUSEMOTION, pygame.MOUSEWHEEL,
    )

    def dispatch(self, event) -> bool:
        """Route one event. Returns False when the app should quit."""
        if event.type == pygame.QUIT:
            return False

        # the worldline panel owns the mouse while the cursor is over it, and
        # keeps it for the rest of an orbit drag even if the cursor leaves
        if self.split and event.type in self._PANEL_MOUSE:
            rect = self.world_rect()
            pos = getattr(event, "pos", None)
            if pos is not None:
                self.mouse_pos = pos
            elif event.type == pygame.MOUSEWHEEL:
                pos = self.mouse_pos
            if self.world.dragging or (pos is not None and rect.collidepoint(pos)):
                self.world.handle(event)
                return True

        if getattr(event, "pos", None) is not None:
            self.mouse_pos = event.pos

        if event.type == pygame.MOUSEBUTTONDOWN:
            self.on_mouse_down(event)
        elif event.type == pygame.MOUSEBUTTONUP:
            self.on_mouse_up(event)
        elif event.type == pygame.MOUSEMOTION:
            self.on_mouse_motion(event)
        elif event.type == pygame.KEYDOWN:
            self.on_key(event)
        return True

    async def run(self) -> None:
        running = True
        while running:
            dt = self.clock.tick(60) / 1000.0
            for event in pygame.event.get():
                if not self.dispatch(event):
                    running = False
            self.update(dt)
            self.draw()
            await asyncio.sleep(0)  # lets the browser run between frames
        pygame.quit()


def _ease(t: float) -> float:
    return 1 - (1 - t) ** 3


async def main() -> None:
    await App().run()


if __name__ == "__main__":
    asyncio.run(main())
    if sys.platform != "emscripten":
        sys.exit(0)
