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

@dataclass
class GAConfig:
    pop_size: int = 200
    generations: int = 300
    prob_presence_init: float = 0.7

    # NEW: jak inicjalizować populację startową
    init_strategy: GAInitMode = "mixed"
    init_constructive_ratio: float = 0.2  # tylko dla mixed

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
    fitness_mode: FitnessMode = "strict"

def _pick_init_mode(cfg: GAConfig) -> InitStrategy:
    if cfg.init_strategy == "constructive":
        return "constructive"
    if cfg.init_strategy == "random":
        return "pure_random"
    # mixed:
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
        # zwykła mutacja
        c.mutation(
            prob_of_moving=cfg.p_mut_move,
            prob_of_rotation=cfg.p_mut_rot,
            prob_of_presence=cfg.p_mut_presence,
            mutation_strength=cfg.mutation_strength,
        )

        # jeśli kontener jest inserted, próbuj czasem "naprawić" podparcie
        if c.inserted and (random.random() < cfg.p_mut_resupport):
            _place_supported_floor_first(c, placed, bias_inside=True)

        placed.append(c)


# reszta run_ga bez zmian
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

    best: Solution | None = None
    best_score = -1
    best_gen = 0
    no_improve = 0
    history = []

    for gen in range(cfg.generations):
        scores = evaluate_population(population, mode=cfg.fitness_mode)

        gen_best_i = max(range(len(population)), key=lambda i: scores[i])
        gen_best_score = scores[gen_best_i]
        if gen_best_score > best_score:
            best_score = gen_best_score
            best = population[gen_best_i].copy()
            best_gen = gen
            no_improve = 0
        else:
            no_improve += 1

        if gen % log_every == 0 or gen == cfg.generations - 1:
            avg = sum(scores) / max(1, len(scores))
            feasible_cnt = sum(1 for ind in population if ind.is_feasible())
            feasible_rate = feasible_cnt / max(1, len(population))
            history.append({
                "gen": gen,
                "best": best_score,
                "gen_best": gen_best_score,
                "avg": avg,
                "feasible_rate": feasible_rate,
            })

        if no_improve >= patience:
            break

        elite_idx = list(range(len(population)))
        elite_idx.sort(key=lambda i: scores[i], reverse=True)
        elites = [population[i].copy() for i in elite_idx[: cfg.elitism]]

        if cfg.selection == "threshold":
            parents_pool = select_threshold(population, scores, cfg.threshold_keep_ratio)
            def pick_parent():
                return random.choice(parents_pool)
        else:
            def pick_parent():
                contenders = random.sample(range(len(population)), k=min(cfg.tournament_k, len(population)))
                best_i = max(contenders, key=lambda i: scores[i])
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
        "best_fitness": best_score,
        "best_solution": best,
        "best_generation": best_gen,
        "history": history,
        "generations_ran": history[-1]["gen"] if history else 0,
    }
