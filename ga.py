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
    mutation_strength: float = 0.15

    p_mut_resupport: float = 0.20

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


def mutate_solution(sol: Solution, cfg: GAConfig) -> None:
    placed: list[Container] = []
    for c in sol.containers:
        c.mutation(
            prob_of_moving=cfg.p_mut_move,
            prob_of_rotation=cfg.p_mut_rot,
            prob_of_presence=cfg.p_mut_presence,
            mutation_strength=cfg.mutation_strength,
        )

        # naprawa podparcia / próbuj stawiać sensownie
        if c.inserted and (random.random() < cfg.p_mut_resupport):
            _place_supported_floor_first(c, placed, bias_inside=True)

        placed.append(c)


def run_ga(
    boxes: List[Dims],
    warehouse: Dims,
    cfg: GAConfig,
    seed: int,
    patience: int = 60,
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

    # najlepszy wg penalized (do prowadzenia selekcji)
    best_eval: Solution | None = None
    best_eval_score = -1
    best_eval_gen = 0
    no_improve = 0

    # najlepszy FEASIBLE wg report_mode (do raportu / porównania z baseline)
    best_report: Solution | None = None
    best_report_score = -1
    best_report_gen = 0

    history = []

    for gen in range(cfg.generations):
        # 1) główna ocena (gradient)
        eval_scores = evaluate_population(population, mode=cfg.fitness_mode)

        # 2) raport (strict/partial) – tylko do logów i best_report
        report_scores = evaluate_population(population, mode=cfg.report_mode)

        gen_best_i = max(range(len(population)), key=lambda i: eval_scores[i])
        gen_best_eval = eval_scores[gen_best_i]

        # aktualizacja best_eval (na penalized)
        if gen_best_eval > best_eval_score:
            best_eval_score = gen_best_eval
            best_eval = population[gen_best_i].copy()
            best_eval_gen = gen
            no_improve = 0
        else:
            no_improve += 1

        # aktualizacja best_report (na strict/partial) – szukamy najlepszej wartości raportowej
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

                # penalized (czym GA się uczy)
                "best_eval": best_eval_score,
                "gen_best_eval": gen_best_eval,
                "avg_eval": avg_eval,

                # strict/partial (co porównujesz z baseline)
                "best_report": best_report_score,
                "gen_best_report": gen_best_report,
                "avg_report": avg_report,

                "feasible_rate": feasible_rate,
            })

        if no_improve >= patience:
            break

        # selekcja rodziców wg eval_scores (penalized)
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

    # ZWRACAMY wynik raportowy jako "best_fitness" (dla benchmarku i porównań)
    return {
        "best_fitness": best_report_score,
        "best_solution": best_report,
        "best_generation": best_report_gen,

        # dodatkowo: jak szło po penalized
        "best_eval_fitness": best_eval_score,
        "best_eval_generation": best_eval_gen,

        "history": history,
        "generations_ran": history[-1]["gen"] if history else 0,
    }
