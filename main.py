# src/binpack3d/main.py
from __future__ import annotations
import argparse
import csv
import os
from ga import GAConfig, run_ga
from viz import plot_solution
from io_utils import load_boxes_from_csv
from benchmark import benchmark_basic

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--mode", type=str, default="strict", choices=["strict", "partial"])
    p.add_argument("--ga_eval_mode", type=str, default="strict", choices=["strict", "partial", "penalized"])


    p.add_argument("--pop", type=int, default=200)
    p.add_argument("--gen", type=int, default=300)
    p.add_argument("--patience", type=int, default=60)

    p.add_argument("--selection", type=str, default="tournament", choices=["threshold", "tournament"])
    p.add_argument("--keep_ratio", type=float, default=0.25)
    p.add_argument("--tournament_k", type=int, default=5)

    p.add_argument("--pcross", type=float, default=0.8)
    p.add_argument("--elitism", type=int, default=2)

    p.add_argument("--pmove", type=float, default=0.25)
    p.add_argument("--prot", type=float, default=0.10)
    p.add_argument("--ppres", type=float, default=0.05)
    p.add_argument("--strength", type=float, default=0.15)
    p.add_argument("--presence_init", type=float, default=0.7)

    # NOWE:
    p.add_argument("--presupport", type=float, default=0.20, help="probability of resupport after mutation")
    p.add_argument("--boxes_csv", type=str, default=None, help="path to csv with boxes (l,w,h) and optional Wx,Wy,Wz")
    p.add_argument("--warehouse", nargs=3, type=int, default=None, metavar=("Wx", "Wy", "Wz"))

    p.add_argument("--csv", type=str, default="runs/history.csv")
    p.add_argument("--plot", action="store_true")

    p.add_argument("--benchmark", action="store_true", help="run benchmark suite and save runs/summary.csv")


    return p.parse_args()


def write_history_csv(path: str, history: list[dict], meta: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = sorted(set().union(*(h.keys() for h in history)) | set(meta.keys()))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in history:
            out = dict(meta)
            out.update(row)
            w.writerow(out)


def main():
    args = parse_args()

    # defaults
    warehouse = (20, 20, 20)
    boxes = [
        (6, 5, 4),
        (7, 3, 4),
        (4, 4, 4),
        (8, 2, 3),
        (3, 3, 6),
        (5, 5, 2),
        (2, 9, 2),
        (4, 7, 2),
    ]

    # CSV override
    if args.boxes_csv is not None:
        boxes_loaded, wh_loaded = load_boxes_from_csv(args.boxes_csv)
        boxes = boxes_loaded
        if wh_loaded is not None:
            warehouse = wh_loaded

    # CLI warehouse override
    if args.warehouse is not None:
        warehouse = tuple(args.warehouse)

    if args.benchmark:
        benchmark_basic(
            boxes_csv=args.boxes_csv if args.boxes_csv else "data/boxes.csv",
            warehouse=warehouse,
            mode=args.mode,
            seeds=[0, 1, 2],
        )
        return


    cfg = GAConfig(
        pop_size=args.pop,
        generations=args.gen,
        prob_presence_init=args.presence_init,
        selection=args.selection,
        threshold_keep_ratio=args.keep_ratio,
        tournament_k=args.tournament_k,
        p_crossover=args.pcross,
        elitism=args.elitism,
        p_mut_move=args.pmove,
        p_mut_rot=args.prot,
        p_mut_presence=args.ppres,
        mutation_strength=args.strength,
        p_mut_resupport=args.presupport,
        fitness_mode=args.mode,

    )

    result = run_ga(
        boxes=boxes,
        warehouse=warehouse,
        cfg=cfg,
        seed=args.seed,
        patience=args.patience,
        log_every=5,
    )

    print("Best fitness:", result["best_fitness"], "best gen:", result["best_generation"])
    meta = {
        "seed": args.seed,
        "mode": args.mode,
        "pop": args.pop,
        "gen": args.gen,
        "selection": args.selection,
        "pcross": args.pcross,
        "elitism": args.elitism,
        "pmove": args.pmove,
        "prot": args.prot,
        "ppres": args.ppres,
        "strength": args.strength,
        "presupport": args.presupport,
        "warehouse": str(warehouse),
        "n_boxes": len(boxes),
    }
    write_history_csv(args.csv, result["history"], meta)

    if args.plot and result["best_solution"] is not None:
        plot_solution(result["best_solution"], warehouse, title=f"3D Bin Packing, Warehouse={warehouse}, Fitness={result['best_fitness']:.2f}")


if __name__ == "__main__":
    main()
