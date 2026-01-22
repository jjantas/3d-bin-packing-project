# src/binpack3d/ga.py
from __future__ import annotations
import random
from dataclasses import dataclass
from typing import List, Literal, Dict, Any

from models import Solution, Dims, Container
from experiments import random_solution, _place_supported_floor_first, InitStrategy
from fitness import fitness, FitnessMode

SelectionMode = Literal["threshold", "tournament"]
GAInitMode = Literal["constructive", "random", "mixed"]

# raportujemy tylko "prawdziwy" wynik: strict albo partial
ReportMode = Literal["strict", "partial"]


@dataclass
class GAConfig:
    pop_size: int = 200
    generations: int = 300
    prob_presence_init: float = 0.7

    init_strategy: GAInitMode = "mixed"
    init_constructive_ratio: float = 0.2

    selection: SelectionMode = "tournament"
    threshold_keep_ratio: float = 0.25
    tournament_k: int = 5

    p_crossover: float = 0.8
    elitism: int = 2

    p_mut_move: float = 0.25
    p_mut_rot: float = 0.10
    p_mut_presence: float = 0.05

    # dodatkowa "naprawcza" mutacja wsparcia / repozycjonowania (po bulk repack)
    p_mut_resupport: float = 0.20

    mutation_strength: float = 0.15

    ratio_to_remove: float = 0.20  # ratio of placed boxes to remove during "ruin and recreate"

    # KLUCZ: na czym GA selekcjonuje (gradient)
    fitness_mode: FitnessMode = "penalized"

    # KLUCZ: co raportujemy jako "best_fitness" (porównywalne z baseline)
    report_mode: ReportMode = "strict"


def _pick_init_mode(cfg: GAConfig) -> InitStrategy:
    if cfg.init_strategy == "constructive":
        return "constructive"
    if cfg.init_strategy == "random":
        return "pure_random"
    return "constructive" if random.random() < cfg.init_constructive_ratio else "pure_random"


def evaluate_population(pop: List[Solution], mode: FitnessMode) -> List[int]:
    return [fitness(ind, mode=mode) for ind in pop]


def select_threshold(pop: List[Solution], scores: List[int], keep_ratio: float) -> List[Solution]:
    idx = list(range(len(pop)))
    idx.sort(key=lambda i: scores[i], reverse=True)
    k = max(2, int(round(len(pop) * keep_ratio)))
    return [pop[i] for i in idx[:k]]


def uniform_crossover(a: Solution, b: Solution, p_cross: float) -> Solution:
    if random.random() > p_cross:
        return (a if random.random() < 0.5 else b).copy()

    child = a.copy()
    for i in range(len(child.containers)):
        if random.random() < 0.5:
            child.containers[i] = b.containers[i].copy()
    return child


def try_insert_box(c: Container, obstacles: list[Container]) -> bool:
    if None in (c.dx, c.dy, c.dz):
        c.choose_rotation_randomly()

    c.inserted = True
    _place_supported_floor_first(c, obstacles, bias_inside=True)

    # 1) musi mieć współrzędne i mieścić się
    if None in (c.x, c.y, c.z) or (not c.fits_in_magazine()):
        c.inserted = False
        c.x, c.y, c.z = None, None, None
        return False

    # 2) nie może nachodzić na przeszkody
    for o in obstacles:
        if o.inserted and (not c.doesnt_overlap(o)):
            c.inserted = False
            c.x, c.y, c.z = None, None, None
            return False

    # 3) podparcie: jak z>0, musi overlap_xy z kimś kto ma top==z
    if c.z > 0:
        supported = any((o.z + o.dz) == c.z and c.overlaps_xy(o) for o in obstacles if o.inserted)
        if not supported:
            c.inserted = False
            c.x, c.y, c.z = None, None, None
            return False

    return True


def _resupport_pass(sol: Solution, cfg: GAConfig) -> None:
    """
    Lekka "naprawa": próbujemy repozycjonować część aktualnie włożonych pudełek,
    żeby były podparte i bez kolizji.
    """
    if cfg.p_mut_resupport <= 0:
        return

    inserted = [c for c in sol.containers if c.inserted]
    if len(inserted) <= 1:
        return

    # iterujemy w losowej kolejności, żeby nie faworyzować indeksów
    random.shuffle(inserted)

    for c in inserted:
        if random.random() > cfg.p_mut_resupport:
            continue

        # wyjmij na chwilę
        c.inserted = False
        c.x, c.y, c.z = None, None, None

        obstacles = [o for o in sol.containers if o.inserted]
        ok = try_insert_box(c, obstacles)

        if not ok:
            # jeśli nie da się włożyć poprawnie, zostawiamy wyjęte (to też jest sensowna mutacja)
            pass


def mutate_solution(sol: Solution, cfg: GAConfig) -> None:
    """
    Mutacja typu "Ruin and Recreate" + drobne próby wstawiania brakujących boxów
    + opcjonalny resupport-pass.
    """

    # 1) Drobne mutacje rotacji i obecności dla boxów NIEwłożonych
    for c in sol.containers:
        if not c.inserted and random.random() < cfg.p_mut_rot:
            c.choose_rotation_randomly()

        if not c.inserted and random.random() < cfg.p_mut_presence:
            obstacles = [x for x in sol.containers if x.inserted]
            try_insert_box(c, obstacles)

    # 2) GŁÓWNA MUTACJA: Bulk Repack (Przepakowanie grupowe)
    if random.random() < cfg.p_mut_move:
        placed = [c for c in sol.containers if c.inserted]
        if not placed:
            return

        ratio_to_remove = max(0.0, min(1.0, cfg.ratio_to_remove))
        n_remove = max(1, int(len(placed) * ratio_to_remove))

        to_remove = random.sample(placed, k=min(n_remove, len(placed)))

        for c in to_remove:
            c.inserted = False
            c.x, c.y, c.z = None, None, None

        obstacles = [c for c in sol.containers if c.inserted]

        random.shuffle(to_remove)

        for c in to_remove:
            if random.random() < 0.5:
                c.choose_rotation_randomly()

            ok = try_insert_box(c, obstacles)
            if ok:
                obstacles.append(c)

    # 3) Opcjonalny resupport pass (naprawczy / eksploracyjny)
    _resupport_pass(sol, cfg)


def run_ga(
    boxes: List[Dims],
    warehouse: Dims,
    cfg: GAConfig,
    seed: int,
    patience: int = 100,
    log_every: int = 10,
) -> Dict[str, Any]:
    random.seed(seed)

    population: List[Solution] = [
        random_solution(
            boxes=boxes,
            warehouse=warehouse,
            prob_presence=cfg.prob_presence_init,
            bias_inside=True,
            init=_pick_init_mode(cfg),
        )
        for _ in range(cfg.pop_size)
    ]

    # najlepszy wg fitness_mode (do prowadzenia selekcji)
    best_eval: Solution | None = None
    best_eval_score = -1
    best_eval_gen = 0
    no_improve = 0

    # najlepszy wg report_mode (do raportu / porównania z baseline)
    best_report: Solution | None = None
    best_report_score = -1
    best_report_gen = 0

    history = []

    for gen in range(cfg.generations):
        eval_scores = evaluate_population(population, mode=cfg.fitness_mode)
        report_scores = evaluate_population(population, mode=cfg.report_mode)

        gen_best_i = max(range(len(population)), key=lambda i: eval_scores[i])
        gen_best_eval = eval_scores[gen_best_i]

        if gen_best_eval > best_eval_score:
            best_eval_score = gen_best_eval
            best_eval = population[gen_best_i].copy()
            best_eval_gen = gen
            no_improve = 0
        else:
            no_improve += 1

        gen_best_report_i = max(range(len(population)), key=lambda i: report_scores[i])
        gen_best_report = report_scores[gen_best_report_i]
        if gen_best_report > best_report_score:
            best_report_score = gen_best_report
            best_report = population[gen_best_report_i].copy()
            best_report_gen = gen

        if gen % log_every == 0 or gen == cfg.generations - 1:
            avg_eval = sum(eval_scores) / max(1, len(eval_scores))
            avg_report = sum(report_scores) / max(1, len(report_scores))

            feasible_cnt = sum(1 for ind in population if ind.is_feasible())
            feasible_rate = feasible_cnt / max(1, len(population))

            history.append({
                "gen": gen,

                # fitness_mode (gradient)
                "best_eval": best_eval_score,
                "gen_best_eval": gen_best_eval,
                "avg_eval": avg_eval,

                # report_mode (real quality)
                "best_report": best_report_score,
                "gen_best_report": gen_best_report,
                "avg_report": avg_report,

                "feasible_rate": feasible_rate,
            })

        if no_improve >= patience:
            break

        elite_idx = list(range(len(population)))
        elite_idx.sort(key=lambda i: eval_scores[i], reverse=True)
        elites = [population[i].copy() for i in elite_idx[: cfg.elitism]]

        if cfg.selection == "threshold":
            parents_pool = select_threshold(population, eval_scores, cfg.threshold_keep_ratio)

            def pick_parent():
                return random.choice(parents_pool)
        else:
            def pick_parent():
                contenders = random.sample(range(len(population)), k=min(cfg.tournament_k, len(population)))
                best_i = max(contenders, key=lambda i: eval_scores[i])
                return population[best_i]

        new_pop: List[Solution] = []
        new_pop.extend(elites)

        while len(new_pop) < cfg.pop_size:
            p1 = pick_parent()
            p2 = pick_parent()
            child = uniform_crossover(p1, p2, cfg.p_crossover)
            mutate_solution(child, cfg)
            new_pop.append(child)

        population = new_pop

    return {
        "best_fitness": best_report_score,
        "best_solution": best_report,
        "best_generation": best_report_gen,

        "best_eval_fitness": best_eval_score,
        "best_eval_generation": best_eval_gen,

        "history": history,
        "generations_ran": history[-1]["gen"] if history else 0,
    }
