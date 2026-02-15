# src/binpack3d/benchmark.py
from __future__ import annotations
import os
import csv
import time
from dataclasses import asdict
from typing import List, Dict, Any, Tuple

from io_utils import load_boxes_from_csv
from experiments import random_search
from ga import GAConfig, run_ga


def warehouse_volume(warehouse: Tuple[int, int, int]) -> int:
    Wx, Wy, Wz = warehouse
    return Wx * Wy * Wz


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(path))
    if not rows:
        return
    fieldnames = sorted(set().union(*(r.keys() for r in rows)))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def run_one_random(boxes, warehouse, seed: int, trials: int, presence: float, mode: str) -> Dict[str, Any]:
    t0 = time.perf_counter()
    res = random_search(
        boxes=boxes,
        warehouse=warehouse,
        trials=trials,
        prob_presence=presence,
        seed=seed,
        fitness_mode=mode,
    )
    t1 = time.perf_counter()
    best = res["best_fitness"]
    util = best / warehouse_volume(warehouse)
    return {
        "algo": "random_search",
        "seed": seed,
        "mode": mode,
        "trials": trials,
        "presence_init": presence,
        "best_fitness": best,
        "utilization": util,
        "seconds": (t1 - t0),
    }


def run_one_ga(boxes, warehouse, seed: int, cfg: GAConfig, patience: int, log_every: int, run_id: str) -> Dict[str, Any]:
    t0 = time.perf_counter()
    res = run_ga(
        boxes=boxes,
        warehouse=warehouse,
        cfg=cfg,
        seed=seed,
        patience=patience,
        log_every=log_every,
    )
    t1 = time.perf_counter()

    best = res["best_fitness"]
    util = best / warehouse_volume(warehouse)

    # zapis przebiegu (convergence)
    conv_dir = "runs/convergence"
    ensure_dir(conv_dir)
    conv_path = os.path.join(conv_dir, f"ga_{run_id}_seed{seed}.csv")
    hist = []
    for row in res["history"]:
        out = dict(row)
        out.update({
            "algo": "ga",
            "run_id": run_id,
            "seed": seed,
            "eval_mode": cfg.fitness_mode,
            "report_mode": cfg.report_mode,
            "pop": cfg.pop_size,
            "generations": cfg.generations,
            "ratio_to_remove": cfg.ratio_to_remove
        })
        hist.append(out)
    write_csv(conv_path, hist)

    return {
        "algo": "ga",
        "run_id": run_id,
        "seed": seed,
        "mode": cfg.fitness_mode,
        "pop": cfg.pop_size,
        "generations": cfg.generations,
        "patience": patience,
        "selection": cfg.selection,
        "pcross": cfg.p_crossover,
        "elitism": cfg.elitism,
        "pmove": cfg.p_mut_move,
        "prot": cfg.p_mut_rot,
        "ppres": cfg.p_mut_presence,
        "presupport": cfg.p_mut_resupport,
        "strength": cfg.mutation_strength,
        "presence_init": cfg.prob_presence_init,
        "best_fitness": best,
        "utilization": util,
        "best_generation": res["best_generation"],
        "seconds": (t1 - t0),
        "ratio_to_remove": cfg.ratio_to_remove
    }


def benchmark_basic(
    boxes_csv: str,
    warehouse: Tuple[int, int, int],
    mode: str = "strict",
    seeds: List[int] = [0, 1, 2, 3, 4],
) -> None:
    boxes, wh_from_csv = load_boxes_from_csv(boxes_csv)
    if wh_from_csv is not None:
        warehouse = wh_from_csv

    ensure_dir("runs")

    summary_rows: List[Dict[str, Any]] = []

    # 1) baseline: random search
    for s in seeds:
        summary_rows.append(run_one_random(
            boxes=boxes,
            warehouse=warehouse,
            seed=s,
            trials=100,
            presence=0.7,
            mode=mode
        ))

    # 2) GA — kilka wariantów parametrów (mini grid)

    # POMARANCZOWA LINIA w TOP 10 configurations (best_report)
    configs = [
    GAConfig(
    pop_size=300, generations=300,
    report_mode=mode,
    p_crossover=0.70,
    p_mut_move=0.20,
    ratio_to_remove=0.1,
    p_mut_resupport=0,
    init_strategy="constructive", init_constructive_ratio=1.0,
    ),

  
    # CIEMNO-NIEBIESKA LINIA w TOP 10 configurations (best_report)
    GAConfig(
    pop_size=300, generations=300,
    report_mode=mode,
    p_crossover=0.70,
    p_mut_move=0.35,
    ratio_to_remove=0.35,
    p_mut_resupport=0,
    init_strategy="constructive", init_constructive_ratio=1.0,
    ),

  
    # FIOLETOWA LINIA w TOP 10 configurations (best_report)
    GAConfig(
    pop_size=300, generations=300,
    report_mode=mode,
    p_crossover=0.70,
    p_mut_move=0.10,
    ratio_to_remove=0.10,
    p_mut_resupport=0,
    init_strategy="constructive", init_constructive_ratio=1.0,
    ),
]

    for i, cfg in enumerate(configs):
        run_id = f"cfg{i}"
        for s in seeds:
            summary_rows.append(run_one_ga(
                boxes=boxes,
                warehouse=warehouse,
                seed=s,
                cfg=cfg,
                patience=60,
                log_every=5,
                run_id=run_id
            ))

    # zapis zbiorczy
    # dopisujemy info o instancji
    for r in summary_rows:
        r["warehouse"] = str(warehouse)
        r["warehouse_volume"] = warehouse_volume(warehouse)
        r["n_boxes"] = len(boxes)

    write_csv("runs/summary.csv", summary_rows)
    print("Saved:", "runs/summary.csv")
    print("Saved convergence:", "runs/convergence/*.csv")
