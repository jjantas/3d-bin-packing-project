from __future__ import annotations

import os
import csv
import time
import json
import hashlib
from dataclasses import asdict
from itertools import product
from typing import List, Dict, Any, Tuple

from io_utils import load_boxes_from_csv
from experiments import random_search
from ga import GAConfig, run_ga

def warehouse_volume(warehouse: Tuple[int, int, int]) -> int:
    Wx, Wy, Wz = warehouse
    return Wx * Wy * Wz


def ensure_dir(path: str) -> None:
    if path:
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


def cfg_fingerprint(cfg_dict: Dict[str, Any]) -> str:
    s = json.dumps(cfg_dict, sort_keys=True)
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:8]


def run_one_random(
    boxes,
    warehouse,
    seed: int,
    trials: int,
    presence: float,
    report_mode: str,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    res = random_search(
        boxes=boxes,
        warehouse=warehouse,
        trials=trials,
        prob_presence=presence,
        seed=seed,
        fitness_mode=report_mode,
    )
    t1 = time.perf_counter()

    best = res["best_fitness"]
    util = best / warehouse_volume(warehouse)

    return {
        "algo": "random_search",
        "seed": seed,
        "report_mode": report_mode,
        "trials": trials,
        "presence_init": presence,
        "best_fitness": best,
        "utilization": util,
        "seconds": (t1 - t0),
    }


def run_one_ga(
    boxes,
    warehouse,
    seed: int,
    cfg: GAConfig,
    patience: int,
    log_every: int,
    run_id: str,
    conv_subdir: str = "A",
) -> Dict[str, Any]:
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

    cfg_dict = asdict(cfg)
    cfg_id = cfg_fingerprint(cfg_dict)

    conv_dir = os.path.join("runs", "convergence", conv_subdir)
    ensure_dir(conv_dir)
    conv_path = os.path.join(conv_dir, f"ga_{run_id}_{cfg_id}_seed{seed}.csv")

    hist = []
    for row in res["history"]:
        out = dict(row)
        out.update(
            {
                "algo": "ga",
                "run_id": run_id,
                "cfg_id": cfg_id,
                "seed": seed,
                **cfg_dict,
            }
        )
        hist.append(out)
    write_csv(conv_path, hist)

    return {
        "algo": "ga",
        "run_id": run_id,
        "cfg_id": cfg_id,
        "seed": seed,
        "best_fitness": best,
        "utilization": util,
        "best_generation": res["best_generation"],
        "best_eval_fitness": res["best_eval_fitness"],
        "best_eval_generation": res["best_eval_generation"],
        "seconds": (t1 - t0),
        **cfg_dict,
        "patience": patience,
        "log_every": log_every,
    }



def make_grid_A(report_mode: str) -> List[GAConfig]:
    ratio_to_remove_grid = [0.10, 0.20, 0.35, 0.50]  # 4
    pmove_grid = [0.10, 0.20, 0.35]  # 3
    presupport_grid = [0.0, 0.20, 0.40]  # 3
    init_ratio_grid = [0.2, 1.0]  # 2
    pcross_grid = [0.7, 0.9]  # 2

    POP_SIZE = 300
    GENERATIONS = 300

    SELECTION = "tournament"
    TOURNAMENT_K = 5
    ELITISM = 2
    P_ROT = 0.10
    P_PRES = 0.05
    MUT_STRENGTH = 0.15
    PRESENCE_INIT = 0.70

    cfgs: List[GAConfig] = []
    for rtr, pmove, pres, init_ratio, pcross in product(
        ratio_to_remove_grid,
        pmove_grid,
        presupport_grid,
        init_ratio_grid,
        pcross_grid,
    ):
        cfgs.append(
            GAConfig(
                pop_size=POP_SIZE,
                generations=GENERATIONS,
                fitness_mode="penalized",
                report_mode=report_mode,
                init_strategy="mixed",
                init_constructive_ratio=init_ratio,
                selection=SELECTION,
                tournament_k=TOURNAMENT_K,
                elitism=ELITISM,
                p_crossover=pcross,
                p_mut_move=pmove,
                ratio_to_remove=rtr,
                p_mut_resupport=pres,
                p_mut_rot=P_ROT,
                p_mut_presence=P_PRES,
                mutation_strength=MUT_STRENGTH,
                prob_presence_init=PRESENCE_INIT,
            )
        )

    return cfgs


def benchmark_basic(
    boxes_csv: str,
    warehouse: Tuple[int, int, int],
    mode: str = "strict",
    seeds: List[int] = [0, 1, 2],
    save_baseline: bool = True,
    tuning: bool = False,
) -> None:
    """
    Runs only Phase A (wide operator grid) and saves:
      - runs/convergence/A/*.csv
      - runs/summary.csv (baseline + GA grid)
      - runs/summary_A.csv (same as above, kept for convenience)
    """
    boxes, wh_from_csv = load_boxes_from_csv(boxes_csv)
    if wh_from_csv is not None:
        warehouse = wh_from_csv

    ensure_dir("runs")
    ensure_dir(os.path.join("runs", "convergence", "A"))

    summary_rows: List[Dict[str, Any]] = []

    if save_baseline:
        for s in seeds:
            summary_rows.append(
                run_one_random(
                    boxes=boxes,
                    warehouse=warehouse,
                    seed=s,
                    trials=100,
                    presence=0.7,
                    report_mode=mode,
                )
            )

    if tuning:
        cfgs = make_grid_A(report_mode=mode)
    else:
        cfgs = [
            GAConfig(
                pop_size=300,
                generations=300,
                fitness_mode="penalized",
                report_mode=mode,
                p_mut_resupport=0.00,
                p_mut_move=0.20,
                p_crossover=0.7,
                init_strategy="constructive",
                init_constructive_ratio=1.0,
                ratio_to_remove=0.1,
            ),
            GAConfig(
                pop_size=300,
                generations=300,
                fitness_mode="penalized",
                report_mode=mode,
                p_mut_resupport=0.00,
                p_mut_move=0.10,
                p_crossover=0.7,
                init_strategy="constructive",
                init_constructive_ratio=1,
                ratio_to_remove=0.1,
            ),
            GAConfig(
                pop_size=300,
                generations=300,
                fitness_mode="penalized",
                report_mode=mode,
                p_mut_resupport=0.0,
                p_mut_move=0.35,
                p_crossover=0.7,
                init_strategy="constructive",
                init_constructive_ratio=1,
                ratio_to_remove=0.35,
            ),
        ]

    for i, cfg in enumerate(cfgs):
        run_id = (
            f"A{i:03d}"
            f"_rtr{cfg.ratio_to_remove:.2f}"
            f"_pm{cfg.p_mut_move:.2f}"
            f"_prs{cfg.p_mut_resupport:.2f}"
            f"_init{cfg.init_constructive_ratio:.1f}"
            f"_pc{cfg.p_crossover:.1f}"
        )
        for s in seeds:
            row = run_one_ga(
                boxes=boxes,
                warehouse=warehouse,
                seed=s,
                cfg=cfg,
                patience=60,
                log_every=5,
                run_id=run_id,
                conv_subdir={"A" if tuning else "baseline"},
            )
            summary_rows.append(row)

    for r in summary_rows:
        r["warehouse"] = str(warehouse)
        r["warehouse_volume"] = warehouse_volume(warehouse)
        r["n_boxes"] = len(boxes)
        r["boxes_csv"] = boxes_csv

    write_csv("runs/summary.csv", summary_rows)
    if tuning:
        write_csv("runs/summary_A.csv", summary_rows)

    print("Saved:")
    print(" - runs/summary.csv")
    if tuning:
        print(" - runs/summary_A.csv")
    print(f" - runs/convergence/{'A' if tuning else 'baseline'}/*.csv")
