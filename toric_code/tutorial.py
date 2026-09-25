"""The tutorial: a card drawn over the app, closed with its X, Esc or a click outside."""

from __future__ import annotations

import pygame

from ui import (
    HAIRLINE, INK, INK_SOFT, PANEL, Fonts, blend, icon_close, math_at, math_size,
    rounded, text_at,
)

# (kind, text): 'h' heading, 'p' paragraph, 'n' numbered item.
# Words may carry subscripts (A_v); see ui.math_at.
TUTORIAL: list[tuple[str, str]] = [
    ("h", "Apply errors"),
    ("p", "Choose a gate – X, Y or Z – on the right, then click an edge of the "
          "lattice to apply it to that qubit."),
    ("h", "Anyons"),
    ("p", "Every star operator A_v that now measures −1 instead of +1 is marked "
          "with an e on its vertex: think of it as an anyon carrying electric "
          "charge. Likewise every violated plaquette B_p is marked with an m, an "
          "anyon carrying magnetic flux."),
    ("h", "Move anyons"),
    ("p", "Pick “Drag anyon” and drag an e or an m around. The Z or X operator on "
          "each edge you cross is added to the state automatically."),
    ("h", "Simplify"),
    ("p", "Press “Simplify” to cancel stabilizers and to turn loops around the "
          "torus into logical operators. The animation walks through every step."),
    ("h", "Things to try"),
    ("n", "Create a pair of anyons with an X or a Z error, drag them around and "
          "let them annihilate again, then press Simplify. If their path closed "
          "into a contractible loop, the leftover operators form a stabilizer and "
          "the state is unchanged. If the loop winds around the torus, you have "
          "made a logical error: the state has moved to another state within the "
          "code space!"),
    ("n", "Charges and fluxes have non-trivial exchange statistics: braiding an "
          "e around an m gives a minus sign. To see it, create a pair of e's and a "
          "pair of m's and pull each pair apart. Move one e around one m, then "
          "annihilate the e's with each other and the m's with each other. This "
          "leaves two stabilizer loops – but a stabilizer can only be cancelled "
          "once it has been commuted to the right, where it acts directly on the "
          "state, and commuting one loop past the other produces the minus sign. "
          "Press Simplify to watch it happen (it may take a few presses to "
          "simplify everything). Click on the worldline view to see the braiding!"),
]

TITLE = "How to play with the toric code"


class TutorialOverlay:
    """Word-wrapped tutorial card, scrollable if the window is too short."""

    PAD = 34
    PARA_GAP = 8
    HEAD_GAP = 12

    def __init__(self, fonts: Fonts) -> None:
        self.fonts = fonts
        self.scroll = 0.0
        self.card = pygame.Rect(0, 0, 0, 0)
        self.close_rect = pygame.Rect(0, 0, 0, 0)
        self.close_hover = False
        self._max_scroll = 0

    # ------------------------------------------------------------------
    def _lines(self, width: int) -> list[tuple[str, str | None, int, int]]:
        """Laid-out lines as (text, number label or None, indent, y)."""
        body = self.fonts.label
        sub = self.fonts.sub(body)
        line_h = body.get_height() + 3
        out = []
        y = 0
        number = 0
        for k, (kind, text) in enumerate(TUTORIAL):
            if kind == "h":
                if k:
                    y += self.HEAD_GAP
                out.append((text, "h", 0, y))
                y += self.fonts.label_bold.get_height() + 6
                continue
            number = number + 1 if kind == "n" else 0
            indent = 26 if kind == "n" else 0
            label = f"{number}." if kind == "n" else None
            line = ""
            for word in text.split():
                trial = f"{line} {word}" if line else word
                if line and math_size(body, sub, trial)[0] > width - indent:
                    out.append((line, label, indent, y))
                    label = None
                    y += line_h
                    line = word
                else:
                    line = trial
            out.append((line, label, indent, y))
            y += line_h + self.PARA_GAP
        return out

    def layout(self, W: int, H: int) -> list:
        width = min(760, W - 120)
        lines = self._lines(width - 2 * self.PAD)
        content_h = (lines[-1][3] + self.fonts.label.get_height()) if lines else 0
        header = 64
        card_h = min(H - 40, header + content_h + self.PAD)
        self.card = pygame.Rect(0, 0, width, card_h)
        self.card.center = (W // 2, H // 2)
        self.close_rect = pygame.Rect(self.card.right - 20 - 34, self.card.y + 18, 34, 34)
        self.view = pygame.Rect(self.card.x, self.card.y + header,
                                width, card_h - header - 14)
        self._max_scroll = max(0, content_h + self.PAD - 14 - self.view.h)
        self.scroll = max(0.0, min(self._max_scroll, self.scroll))
        return lines

    # ------------------------------------------------------------------
    def handle(self, event) -> bool:
        """Consume one event; returns True when the card should close."""
        if event.type == pygame.KEYDOWN:
            return event.key in (pygame.K_ESCAPE, pygame.K_RETURN, pygame.K_SPACE)
        if event.type == pygame.MOUSEWHEEL:
            self.scroll = max(0.0, min(self._max_scroll, self.scroll - event.y * 40))
        elif event.type == pygame.MOUSEMOTION:
            self.close_hover = self.close_rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            return self.close_rect.collidepoint(event.pos) or \
                not self.card.collidepoint(event.pos)
        return False

    def draw(self, surf: pygame.Surface) -> None:
        W, H = surf.get_size()
        lines = self.layout(W, H)

        shade = pygame.Surface((W, H), pygame.SRCALPHA)
        shade.fill((20, 22, 26, 110))
        surf.blit(shade, (0, 0))
        rounded(surf, blend(INK, (250, 250, 250), 0.86), self.card.move(0, 4), 18)
        rounded(surf, PANEL, self.card, 18)

        f = self.fonts
        text_at(surf, f.title, TITLE, INK, (self.card.x + self.PAD, self.card.y + 22))
        rounded(surf, (226, 228, 232) if self.close_hover else (240, 241, 244),
                self.close_rect, self.close_rect.h // 2)
        icon_close(surf, self.close_rect, INK)
        pygame.draw.line(surf, HAIRLINE, (self.card.x + self.PAD, self.view.y - 8),
                         (self.card.right - self.PAD, self.view.y - 8), 1)

        clip = surf.get_clip()
        surf.set_clip(self.view)
        x0 = self.card.x + self.PAD
        top = self.view.y + 6 - round(self.scroll)
        for text, label, indent, y in lines:
            if label == "h":
                text_at(surf, f.label_bold, text, INK, (x0, top + y))
                continue
            if label:
                text_at(surf, f.label_bold, label, INK_SOFT, (x0, top + y))
            math_at(surf, f.label, f.sub(f.label), text, INK,
                    (x0 + indent, top + y))
        surf.set_clip(clip)

        if self.scroll < self._max_scroll:
            text_at(surf, f.tiny, "scroll for more", INK_SOFT,
                    (self.card.right - self.PAD, self.card.bottom - 12), "midright")
