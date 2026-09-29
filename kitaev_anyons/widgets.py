"""Buttons, a segmented switch, a stepper and a slider."""

from __future__ import annotations

import pygame

from ui import CARD, HAIRLINE, INK, INK_SOFT, WHITE, aa_circle, blend, text_at

ACTIVE = (46, 50, 60)
HOVER = (238, 240, 245)


class Button:
    def __init__(self, label: str) -> None:
        self.label = label
        self.rect = pygame.Rect(0, 0, 1, 1)

    def fit(self, fonts, x: int, y: int, h: int = 34) -> pygame.Rect:
        self.rect = pygame.Rect(x, y, fonts.ui.size(self.label)[0] + 24, h)
        return self.rect

    def clicked(self, event) -> bool:
        return (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1
                and self.rect.collidepoint(event.pos))

    def draw(self, surf, fonts, mouse) -> None:
        hot = self.rect.collidepoint(mouse)
        pygame.draw.rect(surf, HOVER if hot else CARD, self.rect, border_radius=8)
        pygame.draw.rect(surf, (205, 208, 216), self.rect, 1, border_radius=8)
        text_at(surf, fonts.ui, self.label, INK, self.rect.center, "center")


class Segmented:
    """One of several options, e.g. periodic | antiperiodic | open."""

    def __init__(self, options: list[str], value: str) -> None:
        self.options = options
        self.value = value
        self.rects: list[pygame.Rect] = []

    def fit(self, fonts, x: int, y: int, h: int = 32) -> pygame.Rect:
        self.rects = []
        for opt in self.options:
            w = fonts.small.size(opt)[0] + 24
            self.rects.append(pygame.Rect(x, y, w, h))
            x += w
        return self.rects[0].unionall(self.rects[1:])

    def handle(self, event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for opt, r in zip(self.options, self.rects):
                if r.collidepoint(event.pos) and opt != self.value:
                    self.value = opt
                    return True
        return False

    def draw(self, surf, fonts, mouse) -> None:
        outer = self.rects[0].unionall(self.rects[1:])
        pygame.draw.rect(surf, CARD, outer, border_radius=7)
        for i, (opt, r) in enumerate(zip(self.options, self.rects)):
            on = opt == self.value
            if on:
                pygame.draw.rect(surf, ACTIVE, r.inflate(-4, -4), border_radius=5)
            elif r.collidepoint(mouse):
                pygame.draw.rect(surf, HOVER, r.inflate(-4, -4), border_radius=5)
            if i:
                pygame.draw.line(surf, HAIRLINE, (r.x, r.y + 5), (r.x, r.bottom - 6))
            text_at(surf, fonts.small, opt, WHITE if on else INK, r.center, "center")
        pygame.draw.rect(surf, (205, 208, 216), outer, 1, border_radius=7)


class Stepper:
    """An integer with - and + buttons."""

    def __init__(self, value: int, lo: int, hi: int) -> None:
        self.value, self.lo, self.hi = value, lo, hi
        self.minus = pygame.Rect(0, 0, 1, 1)
        self.plus = pygame.Rect(0, 0, 1, 1)
        self.label_pos = (0, 0)

    def fit(self, x: int, y: int, h: int = 32) -> pygame.Rect:
        self.minus = pygame.Rect(x, y, h, h)
        self.label_pos = (x + h + 24, y + h / 2)
        self.plus = pygame.Rect(x + h + 48, y, h, h)
        return self.minus.union(self.plus)

    def handle(self, event) -> bool:
        if event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
            return False
        step = (-1 if self.minus.collidepoint(event.pos)
                else 1 if self.plus.collidepoint(event.pos) else 0)
        new = min(self.hi, max(self.lo, self.value + step))
        if new != self.value:
            self.value = new
            return True
        return False

    def draw(self, surf, fonts, mouse) -> None:
        for r, s, ok in ((self.minus, "−", self.value > self.lo),
                         (self.plus, "+", self.value < self.hi)):
            pygame.draw.rect(surf, HOVER if r.collidepoint(mouse) and ok else CARD, r,
                             border_radius=7)
            pygame.draw.rect(surf, (205, 208, 216), r, 1, border_radius=7)
            text_at(surf, fonts.ui_bold, s, INK if ok else INK_SOFT, r.center, "center")
        text_at(surf, fonts.ui_bold, str(self.value), INK, self.label_pos, "center")


class Slider:
    def __init__(self, lo: float, hi: float, value: float, snap: float = 0.0) -> None:
        self.lo, self.hi, self.value = lo, hi, value
        self.snap = snap            # values this close to 0 become exactly 0
        self.rect = pygame.Rect(0, 0, 1, 1)
        self.dragging = False

    def fit(self, x: int, y: int, w: int) -> None:
        self.rect = pygame.Rect(x, y, w, 20)

    def _knob_x(self) -> float:
        t = (self.value - self.lo) / (self.hi - self.lo)
        return self.rect.x + t * self.rect.w

    def _set(self, mx: float) -> None:
        t = min(1.0, max(0.0, (mx - self.rect.x) / self.rect.w))
        v = self.lo + t * (self.hi - self.lo)
        self.value = 0.0 if abs(v) < self.snap else v

    def handle(self, event) -> bool:
        if (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1
                and self.rect.inflate(16, 10).collidepoint(event.pos)):
            self.dragging = True
            self._set(event.pos[0])
            return True
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        if event.type == pygame.MOUSEMOTION and self.dragging:
            self._set(event.pos[0])
            return True
        return False

    def draw(self, surf, mouse) -> None:
        cy = self.rect.centery
        track = pygame.Rect(self.rect.x, cy - 3, self.rect.w, 6)
        pygame.draw.rect(surf, (230, 232, 238), track, border_radius=3)
        if self.lo < 0 < self.hi:        # tick at zero
            zx = self.rect.x + (-self.lo) / (self.hi - self.lo) * self.rect.w
            pygame.draw.line(surf, (180, 184, 192), (zx, cy - 8), (zx, cy + 8))
        kx = self._knob_x()
        hot = self.dragging or self.rect.inflate(16, 10).collidepoint(mouse)
        if hot:
            aa_circle(surf, (255, 226, 150), (kx, cy), 12)
        aa_circle(surf, WHITE, (kx, cy), 9)
        aa_circle(surf, blend(INK, WHITE, 0.1), (kx, cy), 7)
