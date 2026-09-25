"""Small widgets: pill buttons, checkboxes, sliders and a morphing play icon.

Styled after the toric-code viewer: pill-shaped buttons, a near-black accent,
soft greys. Line art is drawn supersampled for smooth edges.
"""

from __future__ import annotations

from typing import Callable

import pygame

INK = (22, 24, 28)
INK_SOFT = (120, 126, 136)
INK_FAINT = (182, 187, 196)
ACCENT = (22, 24, 28)
WHITE = (255, 255, 255)
TRACK = (222, 225, 230)
BUTTON = (255, 255, 255)
BUTTON_HOVER = (230, 232, 236)
DISABLED = (236, 237, 240)

SS = 4


def blend(a, b, t: float):
    t = max(0.0, min(1.0, t))
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b))


def approach(value: float, target: float, dt: float, rate: float = 14.0) -> float:
    return value + (target - value) * min(1.0, dt * rate)


def smooth_polygons(surf: pygame.Surface, col, polys, box: pygame.Rect) -> None:
    """Fill polygons (in screen coordinates) inside ``box``, supersampled."""
    big = pygame.Surface((box.w * SS, box.h * SS), pygame.SRCALPHA)
    big.fill((*col, 0))
    for pts in polys:
        pygame.draw.polygon(
            big, col, [((x - box.x) * SS, (y - box.y) * SS) for x, y in pts]
        )
    surf.blit(pygame.transform.smoothscale(big, box.size), box)


def smooth_circle(surf: pygame.Surface, col, center, r: float, width: float = 0) -> None:
    size = int(2 * r + 4)
    big = pygame.Surface((size * SS, size * SS), pygame.SRCALPHA)
    big.fill((*col, 0))
    pygame.draw.circle(big, col, (size * SS / 2, size * SS / 2), r * SS,
                       round(width * SS) if width else 0)
    small = pygame.transform.smoothscale(big, (size, size))
    surf.blit(small, small.get_rect(center=(round(center[0]), round(center[1]))))


# ----------------------------------------------------------------------
# icons
# ----------------------------------------------------------------------
def _lerp_pts(a, b, t):
    return [(ax + (bx - ax) * t, ay + (by - ay) * t) for (ax, ay), (bx, by) in zip(a, b)]


def icon_play_pause(surf, rect: pygame.Rect, col, t: float) -> None:
    """Play triangle at t=0 morphing into pause bars at t=1.

    The triangle is cut into a left and a right piece, each a quad, which
    slide and straighten into the two bars.
    """
    cx, cy = rect.center
    s = rect.height * 0.25
    l, r = cx - s * 0.7, cx + s * 1.0
    m = (l + r) / 2
    play = [
        [(l, cy - s), (m, cy - s / 2), (m, cy + s / 2), (l, cy + s)],
        [(m, cy - s / 2), (r, cy), (r, cy), (m, cy + s / 2)],
    ]
    bw, gap = s * 0.62, s * 0.5
    pause = [
        [(cx - gap / 2 - bw, cy - s), (cx - gap / 2, cy - s),
         (cx - gap / 2, cy + s), (cx - gap / 2 - bw, cy + s)],
        [(cx + gap / 2, cy - s), (cx + gap / 2 + bw, cy - s),
         (cx + gap / 2 + bw, cy + s), (cx + gap / 2, cy + s)],
    ]
    t = t * t * (3 - 2 * t)
    smooth_polygons(surf, col, [_lerp_pts(p, q, t) for p, q in zip(play, pause)], rect)


def icon_next(surf, rect: pygame.Rect, col) -> None:
    cx, cy = rect.center
    s = rect.height * 0.23
    tri = [(cx - s * 0.9, cy - s), (cx - s * 0.9, cy + s), (cx + s * 0.55, cy)]
    bar_x = cx + s * 0.65
    bar = [(bar_x, cy - s), (bar_x + s * 0.42, cy - s),
           (bar_x + s * 0.42, cy + s), (bar_x, cy + s)]
    smooth_polygons(surf, col, [tri, bar], rect)


# ----------------------------------------------------------------------
# widgets
# ----------------------------------------------------------------------
class Button:
    """A pill button with a text label or an icon painter."""

    def __init__(self, label: str = "", icon: Callable | None = None,
                 primary: bool = False) -> None:
        self.label, self.icon, self.primary = label, icon, primary
        self.rect = pygame.Rect(0, 0, 0, 0)
        self.enabled = True
        self.hover = 0.0

    def update(self, mouse, dt: float) -> None:
        over = self.enabled and self.rect.collidepoint(mouse)
        self.hover = approach(self.hover, 1.0 if over else 0.0, dt)

    def hit(self, pos) -> bool:
        return self.enabled and self.rect.collidepoint(pos)

    def colours(self):
        if not self.enabled:
            return DISABLED, INK_FAINT
        if self.primary:
            return blend(ACCENT, (70, 74, 84), self.hover), WHITE
        return blend(BUTTON, BUTTON_HOVER, self.hover), INK

    def draw(self, surf: pygame.Surface, font: pygame.font.Font) -> None:
        bg, fg = self.colours()
        pygame.draw.rect(surf, bg, self.rect, border_radius=self.rect.height // 2)
        if self.icon is not None:
            self.icon(surf, self.rect, fg)
        else:
            img = font.render(self.label, True, fg)
            surf.blit(img, img.get_rect(center=self.rect.center))


class Checkbox:
    def __init__(self, label: str, checked: bool = False) -> None:
        self.label, self.checked = label, checked
        self.rect = pygame.Rect(0, 0, 0, 0)   # the whole clickable row
        self.hover = 0.0
        self.anim = 1.0 if checked else 0.0

    def update(self, mouse, dt: float) -> None:
        self.hover = approach(self.hover, 1.0 if self.rect.collidepoint(mouse) else 0.0, dt)
        self.anim = approach(self.anim, 1.0 if self.checked else 0.0, dt, 18)

    def hit(self, pos) -> bool:
        return self.rect.collidepoint(pos)

    def draw(self, surf: pygame.Surface, font: pygame.font.Font) -> None:
        box = pygame.Rect(self.rect.x, self.rect.centery - 11, 22, 22)
        border = blend(blend(INK_FAINT, INK_SOFT, self.hover), ACCENT, self.anim)
        pygame.draw.rect(surf, blend(WHITE, ACCENT, self.anim), box, border_radius=6)
        pygame.draw.rect(surf, border, box, 2, border_radius=6)
        if self.anim > 0.05:
            # the check mark grows in with the fill
            cx, cy = box.center
            k = self.anim
            pts = [(cx - 5.5, cy + 0.5), (cx - 1.5, cy + 4.5), (cx + 6, cy - 4)]
            big = pygame.Surface((box.w * SS, box.h * SS), pygame.SRCALPHA)
            big.fill((*WHITE, 0))
            pygame.draw.lines(
                big, WHITE, False,
                [((x - box.x) * SS, (y - box.y) * SS) for x, y in pts], round(2.4 * SS),
            )
            small = pygame.transform.smoothscale(big, box.size)
            small.set_alpha(round(255 * k))
            surf.blit(small, box)
        img = font.render(self.label, True, INK)
        surf.blit(img, img.get_rect(midleft=(box.right + 10, self.rect.centery)))


class Slider:
    """A horizontal slider; with ``stops`` it snaps to one of them."""

    def __init__(self, font: pygame.font.Font, bold: pygame.font.Font, value: float,
                 stops: list[str] | None = None, ends: tuple[str, str] = ("", "")) -> None:
        self.font, self.bold = font, bold
        self.stops, self.ends = stops, ends
        self.value = value          # 0..1, or a stop index
        self.shown = self.fraction()
        self.rect = pygame.Rect(0, 0, 0, 0)   # the track
        self.dragging = False
        self.hover = 0.0

    def fraction(self) -> float:
        if self.stops:
            return self.value / (len(self.stops) - 1)
        return self.value

    def _set_from_x(self, x: float) -> bool:
        f = max(0.0, min(1.0, (x - self.rect.x) / max(1, self.rect.w)))
        old = self.value
        if self.stops:
            self.value = round(f * (len(self.stops) - 1))
        else:
            self.value = f
        return self.value != old

    def hit_area(self) -> pygame.Rect:
        return self.rect.inflate(24, 30)

    def handle(self, ev) -> bool:
        """Mouse handling; returns True if the event was used."""
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            if self.hit_area().collidepoint(ev.pos):
                self.dragging = True
                self._set_from_x(ev.pos[0])
                return True
            if self.stops:
                # clicking a stop's label selects it
                for i, lab in enumerate(self._label_rects()):
                    if lab.inflate(10, 8).collidepoint(ev.pos):
                        self.value = i
                        return True
        elif ev.type == pygame.MOUSEMOTION and self.dragging:
            self._set_from_x(ev.pos[0])
            return True
        elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1 and self.dragging:
            self.dragging = False
            return True
        return False

    def update(self, mouse, dt: float) -> None:
        self.shown = approach(self.shown, self.fraction(), dt, 16)
        over = self.dragging or self.hit_area().collidepoint(mouse)
        self.hover = approach(self.hover, 1.0 if over else 0.0, dt)

    def _label_rects(self) -> list[pygame.Rect]:
        out = []
        for i, lab in enumerate(self.stops or []):
            x = self.rect.x + self.rect.w * i / (len(self.stops) - 1)
            w, h = self.font.size(lab)
            out.append(pygame.Rect(round(x - w / 2), self.rect.bottom + 12, w, h))
        return out

    def draw(self, surf: pygame.Surface) -> None:
        font, bold = self.font, self.bold
        tr = self.rect
        pygame.draw.rect(surf, TRACK, tr, border_radius=tr.h // 2)
        kx = tr.x + tr.w * self.shown
        filled = pygame.Rect(tr.x, tr.y, max(tr.h, round(kx - tr.x)), tr.h)
        pygame.draw.rect(surf, ACCENT, filled, border_radius=tr.h // 2)
        if self.stops:
            for i, lab_rect in enumerate(self._label_rects()):
                x = tr.x + tr.w * i / (len(self.stops) - 1)
                done = x <= kx + 0.5
                smooth_circle(surf, ACCENT if done else TRACK, (x, tr.centery), 5)
                sel = i == self.value
                img = (bold if sel else font).render(self.stops[i], True,
                                                     INK if sel else INK_SOFT)
                surf.blit(img, img.get_rect(midtop=(lab_rect.centerx, lab_rect.y)))
        else:
            for text, anchor, x in ((self.ends[0], "midtop", tr.x),
                                    (self.ends[1], "midtop", tr.right)):
                img = font.render(text, True, INK_SOFT)
                surf.blit(img, img.get_rect(**{anchor: (x, tr.bottom + 12)}))
        r = 10 + 1.5 * self.hover
        smooth_circle(surf, (205, 208, 214), (kx, tr.centery + 1), r + 1)
        smooth_circle(surf, WHITE, (kx, tr.centery), r)
        smooth_circle(surf, ACCENT, (kx, tr.centery), r, 2)
