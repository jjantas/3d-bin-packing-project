# src/binpack3d/fitness.py
from __future__ import annotations
from typing import Literal
from models import Solution, Container

FitnessMode = Literal["strict", "partial", "penalized"]

def container_volume(c: Container) -> int:
    return int(c.l * c.w * c.h)

def _is_supported(c: Container, accepted: list[Container]) -> bool:
    if c.z == 0:
        return True
    for b in accepted:
        if (b.z + b.dz) == c.z and c.overlaps_xy(b):
            return True
    return False

def _out_of_bounds_penalty(c: Container) -> int:
    # kara za przekroczenia w osiach (w jednostkach długości)
    # (proste i szybkie; nie wymaga liczenia "volume outside")
    Wx, Wy, Wz = c.Wx, c.Wy, c.Wz
    if None in (Wx, Wy, Wz, c.x, c.y, c.z, c.dx, c.dy, c.dz):
        return 0
    px = max(0, c.x + c.dx - Wx) + max(0, -c.x)
    py = max(0, c.y + c.dy - Wy) + max(0, -c.y)
    pz = max(0, c.z + c.dz - Wz) + max(0, -c.z)
    return px + py + pz

def fitness(solution: Solution, mode: FitnessMode = "strict") -> int:
    if mode == "strict":
        if not solution.is_feasible():
            return 0
        return sum(container_volume(c) for c in solution.containers if c.inserted)

    if mode == "partial":
        accepted: list[Container] = []
        total = 0
        for c in solution.containers:
            if not c.inserted:
                continue
            if not c.fits_in_magazine():
                continue

            ok = True
            for a in accepted:
                if not c.doesnt_overlap(a):
                    ok = False
                    break
            if not ok:
                continue

            if not _is_supported(c, accepted):
                continue

            accepted.append(c)
            total += container_volume(c)
        return total

    # ---- penalized ----
    # baza: "ile chcę zapakować" (nawet jeśli nielegalne)
    base = sum(container_volume(c) for c in solution.containers if c.inserted)

    # kary: w tej instancji objętości są rzędu ~2000,
    # więc kary muszą być porównywalne / większe.
    P_OUT = 300     # za jednostkę przekroczenia
    P_OVER = 2000   # za każdą kolizję pary
    P_FLOAT = 1500  # za każdy "wiszący" kontener

    penalty = 0

    inserted = [c for c in solution.containers if c.inserted]

    # (1) out of bounds
    for c in inserted:
        if not c.fits_in_magazine():
            penalty += P_OUT * _out_of_bounds_penalty(c)

    # (2) overlaps (para)
    for i in range(len(inserted)):
        for j in range(i + 1, len(inserted)):
            if not inserted[i].doesnt_overlap(inserted[j]):
                penalty += P_OVER

    # (3) floating (brak podparcia)
    for c in inserted:
        if c.z == 0:
            continue
        supported = False
        for b in inserted:
            if b is c:
                continue
            if (b.z + b.dz) == c.z and c.overlaps_xy(b):
                supported = True
                break
        if not supported:
            penalty += P_FLOAT

    score = base - penalty
    return max(0, int(score))
