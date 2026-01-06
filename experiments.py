# src/binpack3d/experiments.py
from __future__ import annotations
import random
from typing import List, Dict, Any
from models import Solution, Dims, Container
from io_utils import make_containers
from fitness import fitness, FitnessMode


def _rand_int_clamped(lo: int, hi: int) -> int:
    if hi < lo:
        return lo
    return random.randint(lo, hi)


def _fits_no_overlap(c: Container, placed: list[Container]) -> bool:
    """Sprawdza: mieści się i nie nachodzi na ŻADEN inserted z placed."""
    if not c.fits_in_magazine():
        return False
    for p in placed:
        if p.inserted and (not c.doesnt_overlap(p)):
            return False
    return True


def _try_place_on_floor(
    c: Container,
    placed: list[Container],
    bias_inside: bool,
    trials: int = 60,
) -> bool:
    """
    Próbuje umieścić kontener na podłodze (z=0) tak, żeby nie nachodził.
    Zwraca True jeśli się udało.
    """
    if c.Wx is None or c.Wy is None or c.Wz is None:
        raise ValueError("Warehouse not set.")
    if None in (c.dx, c.dy, c.dz):
        raise ValueError("Rotation not set.")

    Wx, Wy, Wz = c.Wx, c.Wy, c.Wz
    if c.dz > Wz:
        return False  # i tak nie zmieści się w pionie

    # losujemy x,y w [0, W-d] (jeśli możliwe)
    x_hi = max(0, Wx - c.dx)
    y_hi = max(0, Wy - c.dy)

    for _ in range(trials):
        c.z = 0
        c.x = _rand_int_clamped(0, x_hi)
        c.y = _rand_int_clamped(0, y_hi)
        if _fits_no_overlap(c, placed):
            return True

    return False


def _try_place_on_supporters(
    c: Container,
    placed: list[Container],
    bias_inside: bool,
    trials: int = 60,
) -> bool:
    """
    Próbuje postawić kontener na wierzchu istniejących (podparcie) i bez kolizji.
    Zwraca True jeśli się udało.
    """
    if c.Wx is None or c.Wy is None or c.Wz is None:
        raise ValueError("Warehouse not set.")
    if None in (c.dx, c.dy, c.dz):
        raise ValueError("Rotation not set.")

    Wx, Wy, Wz = c.Wx, c.Wy, c.Wz
    supporters = [p for p in placed if p.inserted]
    if not supporters:
        return False

    # wielokrotnie próbujemy: losowy supporter -> losowe x,y dające overlap_xy -> test kolizji
    for _ in range(trials):
        supporter = random.choice(supporters)
        z = supporter.z + supporter.dz
        if z + c.dz > Wz:
            continue  # za wysoko

        # zakres x,y dający dodatni overlap w XY
        x_lo = supporter.x - c.dx + 1
        x_hi = supporter.x + supporter.dx - 1
        y_lo = supporter.y - c.dy + 1
        y_hi = supporter.y + supporter.dy - 1

        # klamruj do [0, W-d]
        x_lo = max(x_lo, 0)
        y_lo = max(y_lo, 0)
        x_hi = min(x_hi, max(0, Wx - c.dx))
        y_hi = min(y_hi, max(0, Wy - c.dy))

        if x_hi < x_lo or y_hi < y_lo:
            continue

        c.x = _rand_int_clamped(x_lo, x_hi)
        c.y = _rand_int_clamped(y_lo, y_hi)
        c.z = z

        # (1) musi być podparty
        if not c.overlaps_xy(supporter):
            continue

        # (2) i nie może nachodzić na pozostałe
        if _fits_no_overlap(c, placed):
            return True

    return False


def _place_supported_floor_first(
    c: Container,
    placed: list[Container],
    bias_inside: bool = True,
    floor_trials: int = 80,
    support_trials: int = 120,
) -> None:
    """
    NOWA STRATEGIA:
    1) Spróbuj podłogi (z=0) bez kolizji.
    2) Jeśli się nie da, spróbuj postawić na innych (podparcie + bez kolizji).
    3) Fallback: losowo (może być invalid i odpadnie w fitness).
    """
    if _try_place_on_floor(c, placed, bias_inside=bias_inside, trials=floor_trials):
        return
    if _try_place_on_supporters(c, placed, bias_inside=bias_inside, trials=support_trials):
        return

    # fallback — cokolwiek, żeby nie było None
    c.place_randomly(bias_inside=bias_inside)
    if bias_inside:
        c.z = 0  # preferuj podłogę w fallback


def random_solution(
    boxes: List[Dims],
    warehouse: Dims,
    prob_presence: float,
    bias_inside: bool = True,
) -> Solution:
    containers = make_containers(boxes, warehouse)

    placed: list[Container] = []
    for c in containers:
        c.choose_presence_randomly(prob_presence)
        c.choose_rotation_randomly()

        if not c.inserted:
            c.place_randomly(bias_inside=bias_inside)
        else:
            _place_supported_floor_first(c, placed, bias_inside=bias_inside)

        placed.append(c)

    return Solution(containers=containers)


def random_search(
    boxes: List[Dims],
    warehouse: Dims,
    trials: int,
    prob_presence: float,
    seed: int,
    fitness_mode: FitnessMode = "strict",
) -> Dict[str, Any]:
    random.seed(seed)
    best = None
    best_f = -1

    for _ in range(trials):
        sol = random_solution(boxes, warehouse, prob_presence=prob_presence, bias_inside=True)
        f = fitness(sol, mode=fitness_mode)
        if f > best_f:
            best_f = f
            best = sol

    return {"best_fitness": best_f, "best_solution": best}
