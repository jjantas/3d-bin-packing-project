# src/binpack3d/fitness.py
from __future__ import annotations
from typing import Literal
from models import Solution, Container

FitnessMode = Literal["strict", "partial"]


def container_volume(c: Container) -> int:
    return int(c.l * c.w * c.h)


def _is_supported(c: Container, accepted: list[Container]) -> bool:
    if c.z == 0:
        return True
    for b in accepted:
        if (b.z + b.dz) == c.z and c.overlaps_xy(b):
            return True
    return False


def fitness(solution: Solution, mode: FitnessMode = "strict") -> int:
    """
    strict: jeśli NIEDOPUSZCZALNE => 0, inaczej suma objętości wszystkich inserted=True
    partial: greedy: dodajemy kontenery jeśli: inserted, fits, nie nachodzą, i są PODPARTE
    """
    if mode == "strict":
        if not solution.is_feasible():
            return 0
        return sum(container_volume(c) for c in solution.containers if c.inserted)

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
