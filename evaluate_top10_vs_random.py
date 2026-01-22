from __future__ import annotations

import argparse
import os
import time
from typing import Dict, Any, List, Tuple

import pandas as pd

from io_utils import load_boxes_from_csv
from experiments import random_search
from ga import GAConfig, run_ga


# -------------------------
# Helpers
# -------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def warehouse_volume(wh: Tuple[int, int, int]) -> int:
    return int(wh[0] * wh[1] * wh[2])


def to_int(x, default: int = 0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


def to_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def pick_top_cfgs(summary_csv: str, top_n: int = 10) -> pd.DataFrame:
    """
    Wybiera TOP-N konfiguracji (cfg_id) po score_from_conv.
    Agregacja: mean(score_from_conv) po seedach.
    """
    df = pd.read_csv(summary_csv)

    # tylko GA
    df = df[df["algo"].astype(str) == "ga"].copy()

    if "score_from_conv" not in df.columns:
        raise ValueError("CSV nie ma kolumny score_from_conv")

    df["score_from_conv"] = pd.to_numeric(df["score_from_conv"], errors="coerce")
    df = df.dropna(subset=["score_from_conv"])

    # ranking po cfg_id: mean po seedach
    rank = (
        df.groupby("cfg_id")["score_from_conv"]
        .agg(["mean", "count", "std", "max"])
        .sort_values("mean", ascending=False)
        .reset_index()
    )

    top = rank.head(top_n).copy()

    # dołącz parametry z pierwszego wystąpienia cfg_id
    params_cols = [
        "elitism",
        "fitness_mode",
        "generations",
        "init_constructive_ratio",
        "init_strategy",
        "mutation_strength",
        "p_crossover",
        "p_mut_move",
        "p_mut_presence",
        "p_mut_resupport",
        "p_mut_rot",
        "pop_size",
        "prob_presence_init",
        "ratio_to_remove",
        "report_mode",
        "selection",
        "threshold_keep_ratio",
        "tournament_k",
    ]

    first_rows = df.sort_values(["cfg_id"]).drop_duplicates("cfg_id")
    top = top.merge(
        first_rows[["cfg_id"] + [c for c in params_cols if c in first_rows.columns]],
        on="cfg_id",
        how="left",
    )

    # porządek
    top = top.rename(columns={"mean": "rank_score_mean", "count": "n_seeds"})
    return top


def cfg_from_row(row: pd.Series) -> GAConfig:
    """
    Buduje GAConfig wyłącznie z pól, które masz w CSV.
    """
    return GAConfig(
        pop_size=to_int(row.get("pop_size", 200), 200),
        generations=to_int(row.get("generations", 300), 300),
        prob_presence_init=to_float(row.get("prob_presence_init", 0.7), 0.7),
        init_strategy=str(row.get("init_strategy", "mixed")),
        init_constructive_ratio=to_float(row.get("init_constructive_ratio", 0.2), 0.2),
        selection=str(row.get("selection", "tournament")),
        threshold_keep_ratio=to_float(row.get("threshold_keep_ratio", 0.25), 0.25),
        tournament_k=to_int(row.get("tournament_k", 5), 5),
        p_crossover=to_float(row.get("p_crossover", 0.8), 0.8),
        elitism=to_int(row.get("elitism", 2), 2),
        p_mut_move=to_float(row.get("p_mut_move", 0.25), 0.25),
        p_mut_rot=to_float(row.get("p_mut_rot", 0.10), 0.10),
        p_mut_presence=to_float(row.get("p_mut_presence", 0.05), 0.05),
        p_mut_resupport=to_float(row.get("p_mut_resupport", 0.20), 0.20),
        mutation_strength=to_float(row.get("mutation_strength", 0.15), 0.15),
        ratio_to_remove=to_float(row.get("ratio_to_remove", 0.20), 0.20),
        # GA selekcjonuje po fitness_mode, raportuje report_mode
        fitness_mode=str(row.get("fitness_mode", "penalized")),
        report_mode=str(row.get("report_mode", "strict")),
    )


# -------------------------
# Main evaluation
# -------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Evaluate TOP-10 GA configs vs random_search on 3 warehouse sizes."
    )
    ap.add_argument(
        "--summary_csv",
        type=str,
        required=True,
        help="CSV z podsumowaniem eksperymentów (z score_from_conv).",
    )
    ap.add_argument(
        "--boxes_csv",
        type=str,
        required=True,
        help="CSV z pudełkami (l,w,h) – użyte w ewaluacji.",
    )
    ap.add_argument(
        "--out_csv",
        type=str,
        default="runs/eval_top10_vs_random/results.csv",
        help="Gdzie zapisać wyniki.",
    )
    ap.add_argument(
        "--top_n", type=int, default=10, help="Ile najlepszych cfg_id wybrać."
    )
    ap.add_argument(
        "--seeds", type=str, default="0,1,2", help="Seedy ewaluacji (np. '0,1,2')."
    )
    ap.add_argument("--patience", type=int, default=60, help="Patience dla GA.")
    ap.add_argument(
        "--log_every",
        type=int,
        default=10,
        help="Log co ile generacji (wpływa na koszt).",
    )
    ap.add_argument(
        "--rs_trials",
        type=int,
        default=200,
        help="Ile prób random_search na każdy seed/przypadek.",
    )
    ap.add_argument(
        "--rs_presence",
        type=float,
        default=0.7,
        help="prob_presence dla random_search.",
    )
    ap.add_argument(
        "--easy_wh", type=str, default="50,50,50", help="Easy warehouse Wx,Wy,Wz."
    )
    ap.add_argument(
        "--mid_wh", type=str, default="35,35,35", help="Medium warehouse Wx,Wy,Wz."
    )
    ap.add_argument(
        "--hard_wh", type=str, default="20,20,20", help="Hard warehouse Wx,Wy,Wz."
    )
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip() != ""]

    def parse_wh(s: str) -> Tuple[int, int, int]:
        parts = [p.strip() for p in s.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Bad warehouse format: {s} (expected Wx,Wy,Wz)")
        return int(parts[0]), int(parts[1]), int(parts[2])

    warehouses = [
        ("easy", parse_wh(args.easy_wh)),
        ("medium", parse_wh(args.mid_wh)),
        ("hard", parse_wh(args.hard_wh)),
    ]

    # load boxes once
    boxes, _wh_from_csv = load_boxes_from_csv(args.boxes_csv)

    # pick top configs from summary
    top_cfgs = pick_top_cfgs(args.summary_csv, top_n=args.top_n)

    out_dir = os.path.dirname(args.out_csv)
    ensure_dir(out_dir)

    # save ranking for reference
    top_cfgs.to_csv(os.path.join(out_dir, "top_cfgs.csv"), index=False)

    rows: List[Dict[str, Any]] = []

    for rank_i, cfg_row in top_cfgs.reset_index(drop=True).iterrows():
        rank = rank_i + 1
        cfg_id = str(cfg_row["cfg_id"])
        cfg = cfg_from_row(cfg_row)

        for wh_name, wh in warehouses:
            for seed in seeds:
                # ---------- GA ----------
                t0 = time.perf_counter()
                ga_res = run_ga(
                    boxes=boxes,
                    warehouse=wh,
                    cfg=cfg,
                    seed=seed,
                    patience=args.patience,
                    log_every=args.log_every,
                )
                t1 = time.perf_counter()

                ga_score = ga_res["best_fitness"]
                ga_util = (
                    ga_score / warehouse_volume(wh) if warehouse_volume(wh) > 0 else 0.0
                )

                # ---------- Random Search ----------
                t2 = time.perf_counter()
                rs_res = random_search(
                    boxes=boxes,
                    warehouse=wh,
                    trials=args.rs_trials,
                    prob_presence=args.rs_presence,
                    seed=seed,
                    fitness_mode=cfg.report_mode,  # porównanie w tym samym trybie raportowania
                )
                t3 = time.perf_counter()

                rs_score = rs_res["best_fitness"]
                rs_util = (
                    rs_score / warehouse_volume(wh) if warehouse_volume(wh) > 0 else 0.0
                )

                rows.append(
                    {
                        "rank": rank,
                        "cfg_id": cfg_id,
                        "warehouse_case": wh_name,
                        "warehouse": str(wh),
                        "warehouse_volume": warehouse_volume(wh),
                        "seed": seed,
                        "ga_best_fitness": ga_score,
                        "ga_utilization": ga_util,
                        "ga_seconds": (t1 - t0),
                        "ga_best_generation": ga_res.get("best_generation", None),
                        "ga_best_eval_fitness": ga_res.get("best_eval_fitness", None),
                        "rs_best_fitness": rs_score,
                        "rs_utilization": rs_util,
                        "rs_seconds": (t3 - t2),
                        "ga_minus_rs": ga_score - rs_score,
                        # meta (z rankingu)
                        "rank_score_mean": to_float(
                            cfg_row.get("rank_score_mean", 0.0)
                        ),
                        "n_seeds_in_ranking": to_int(cfg_row.get("n_seeds", 0)),
                        "rank_score_std": to_float(cfg_row.get("std", 0.0)),
                    }
                )

                # zapis przyrostowy
                pd.DataFrame(rows).to_csv(args.out_csv, index=False)

    # final save
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)

    # dodatkowy agregat: mean po seedach
    df_res = pd.DataFrame(rows)
    agg = (
        df_res.groupby(["rank", "cfg_id", "warehouse_case"])
        .agg(
            {
                "ga_best_fitness": "mean",
                "rs_best_fitness": "mean",
                "ga_minus_rs": "mean",
                "ga_seconds": "mean",
                "rs_seconds": "mean",
                "ga_utilization": "mean",
                "rs_utilization": "mean",
            }
        )
        .reset_index()
        .sort_values(["warehouse_case", "rank"])
    )
    agg.to_csv(os.path.join(out_dir, "results_aggregated.csv"), index=False)

    print("Saved:")
    print(f" - {args.out_csv}")
    print(f" - {os.path.join(out_dir, 'results_aggregated.csv')}")
    print(f" - {os.path.join(out_dir, 'top_cfgs.csv')}")


if __name__ == "__main__":
    main()
