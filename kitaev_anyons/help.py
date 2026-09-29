"""A small round "?" button that opens a popup with an explanation.

Popup content is a list of items:
    ("h", text)      a heading
    ("p", text)      a word-wrapped paragraph
    ("f", source)    a mathtext formula
    ("m", builder)   builder(size) -> mathtext Box, for things like matrices
While a popup is open it takes every click: a click anywhere closes it.
"""

from __future__ import annotations

import pygame

import mathtext as mt
from ui import CARD, HAIRLINE, INK, INK_SOFT, aa_circle, text_at

POPUP_W = 620
PAD = 20
FORMULA_SIZE = 18


class HelpButton:
    def __init__(self, title: str, content: list) -> None:
        self.title = title
        self.content = content
        self.rect = pygame.Rect(0, 0, 26, 26)
        self.open = False
        self._popup: pygame.Surface | None = None
        self._popup_key = None

    def place(self, top_right: tuple[int, int]) -> None:
        self.rect = pygame.Rect(0, 0, 26, 26)
        self.rect.topright = top_right

    def handle(self, event) -> bool:
        """True when the event was used (always, while the popup is open)."""
        if self.open:
            if event.type == pygame.MOUSEBUTTONDOWN:
                self.open = False
                return True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.open = False
                return True
            return event.type in (pygame.MOUSEBUTTONUP, pygame.MOUSEWHEEL)
        if (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1
                and self.rect.collidepoint(event.pos)):
            self.open = True
            return True
        return False

    def draw_button(self, surf, fonts, mouse) -> None:
        hot = self.open or self.rect.collidepoint(mouse)
        c = self.rect.center
        aa_circle(surf, (230, 232, 238) if hot else (240, 241, 245), c, 13)
        aa_circle(surf, (190, 194, 202), c, 13, width=1.2)
        text_at(surf, fonts.ui_bold, "?", INK, (c[0], c[1] - 1), "center")

    def draw_popup(self, surf, fonts) -> None:
        if not self.open:
            return
        key = id(fonts)
        if key != self._popup_key:
            self._popup = self._render(fonts)
            self._popup_key = key
        r = self._popup.get_rect()
        r.topright = (self.rect.right + 6, self.rect.bottom + 8)
        r.clamp_ip(surf.get_rect().inflate(-16, -16))
        shadow = pygame.Surface((r.w + 16, r.h + 16), pygame.SRCALPHA)
        pygame.draw.rect(shadow, (20, 24, 32, 38), shadow.get_rect(), border_radius=18)
        surf.blit(shadow, (r.x - 8, r.y - 2))
        surf.blit(self._popup, r)

    def _render(self, fonts) -> pygame.Surface:
        w = POPUP_W
        inner = w - 2 * PAD
        pieces: list[tuple[str, object, int]] = []    # (kind, image, gap above)
        pieces.append(("img", fonts.title.render(self.title, True, INK), 0))
        for kind, data in self.content:
            if kind == "h":
                pieces.append(("img", fonts.ui_bold.render(data, True, INK), 16))
            elif kind == "p":
                for k, line in enumerate(_wrap(fonts.ui, data, inner)):
                    pieces.append(("img", fonts.ui.render(line, True, INK_SOFT), 8 if k == 0 else 1))
            elif kind == "f":
                pieces.append(("box", mt.formula(data, FORMULA_SIZE), 10))
            elif kind == "m":
                pieces.append(("box", data(FORMULA_SIZE), 10))
        h = PAD + sum(g + (p.get_height() if k == "img" else p.h) for k, p, g in pieces) + PAD + 22
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(surf, CARD, surf.get_rect(), border_radius=14)
        pygame.draw.rect(surf, HAIRLINE, surf.get_rect(), 1, border_radius=14)
        y = PAD
        for kind, p, gap in pieces:
            y += gap
            if kind == "img":
                surf.blit(p, (PAD, y))
                y += p.get_height()
            else:
                p.blit(surf, PAD + 6, y + p.base)
                y += p.h
        text_at(surf, fonts.tiny, "click anywhere to close", INK_SOFT, (w - PAD, h - PAD + 4), "bottomright")
        return surf


def _wrap(font, text: str, width: int) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        trial = (line + " " + word).strip()
        if font.size(trial)[0] > width and line:
            lines.append(line)
            line = word
        else:
            line = trial
    if line:
        lines.append(line)
    return lines
