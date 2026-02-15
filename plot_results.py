# src/binpack3d/plot_results.py
from __future__ import annotations
import argparse
import csv
import glob
import os
import pandas as pd
from collections import defaultdict
from typing import Dict, Any, List, Tuple

import matplotlib.pyplot as plt


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_csv(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return [dict(row) for row in r]


def to_float(x: Any, default: float = 0.0) -> float:
    if x is None:
        return default
    s = str(x).strip()
    if s == "":
        return default
    try:
        return float(s)
    except ValueError:
        return default


def to_int(x: Any, default: int = 0) -> int:
    if x is None:
        return default
    s = str(x).strip()
    if s == "":
        return default
    try:
        return int(float(s))
    except ValueError:
        return default


def group_by(rows: List[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[str(row.get(key, ""))].append(row)
    return out


def plot_summary_scatter(summary_rows: List[Dict[str, Any]], out_dir: str) -> None:
    """
    Wykresy porównawcze GA vs random:
    - best_fitness vs seed (scatter)
    - utilization vs seed (scatter)
    - seconds vs seed (scatter)
    """
    ensure_dir(out_dir)

    # przygotuj dane
    for r in summary_rows:
        r["best_fitness"] = to_float(r.get("best_fitness", 0))
        r["utilization"] = to_float(r.get("utilization", 0))
        r["seconds"] = to_float(r.get("seconds", 0))
        r["seed"] = to_int(r.get("seed", 0))

    by_algo = group_by(summary_rows, "algo")

    # ---- best_fitness
    plt.figure()
    for algo, rows in by_algo.items():
        xs = [r["seed"] for r in rows]
        ys = [r["best_fitness"] for r in rows]
        plt.scatter(xs, ys, label=algo)
        if ys:
            mean_y = sum(ys) / len(ys)
            plt.axhline(mean_y, linestyle="--")
    plt.xlabel("seed")
    plt.ylabel("best_fitness")
    plt.title("Best fitness: GA vs Random Search")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "summary_best_fitness.png"))
    plt.close()

    # ---- utilization
    plt.figure()
    for algo, rows in by_algo.items():
        xs = [r["seed"] for r in rows]
        ys = [r["utilization"] for r in rows]
        plt.scatter(xs, ys, label=algo)
        if ys:
            mean_y = sum(ys) / len(ys)
            plt.axhline(mean_y, linestyle="--")
    plt.xlabel("seed")
    plt.ylabel("utilization (packed_volume / warehouse_volume)")
    plt.title("Utilization: GA vs Random Search")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "summary_utilization.png"))
    plt.close()

    # ---- seconds
    plt.figure()
    for algo, rows in by_algo.items():
        xs = [r["seed"] for r in rows]
        ys = [r["seconds"] for r in rows]
        plt.scatter(xs, ys, label=algo)
        if ys:
            mean_y = sum(ys) / len(ys)
            plt.axhline(mean_y, linestyle="--")
    plt.xlabel("seed")
    plt.ylabel("seconds")
    plt.title("Runtime: GA vs Random Search")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "summary_seconds.png"))
    plt.close()


def plot_ga_configs_bar(summary_rows: List[Dict[str, Any]], out_dir: str) -> None:
    """
    Dodatkowy wykres: porównanie konfiguracji GA (run_id) w uśrednieniu po seedach.
    (Random search ignorujemy)
    """
    ensure_dir(out_dir)

    ga_rows = [r for r in summary_rows if str(r.get("algo", "")) == "ga"]
    if not ga_rows:
        return

    for r in ga_rows:
        r["best_fitness"] = to_float(r.get("best_fitness", 0))
        r["utilization"] = to_float(r.get("utilization", 0))
        r["seconds"] = to_float(r.get("seconds", 0))

    by_run = group_by(ga_rows, "run_id")
    run_ids = sorted([k for k in by_run.keys() if k.strip() != ""])

    avg_fit = []
    avg_util = []
    avg_sec = []

    for rid in run_ids:
        rows = by_run[rid]
        avg_fit.append(sum(r["best_fitness"] for r in rows) / max(1, len(rows)))
        avg_util.append(sum(r["utilization"] for r in rows) / max(1, len(rows)))
        avg_sec.append(sum(r["seconds"] for r in rows) / max(1, len(rows)))

    # avg best_fitness per config
    plt.figure()
    plt.bar(run_ids, avg_fit)
    plt.xlabel("GA config (run_id)")
    plt.ylabel("avg best_fitness (over seeds)")
    plt.title("GA configs: average best fitness")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ga_configs_avg_best_fitness.png"))
    plt.close()

    # avg utilization per config
    plt.figure()
    plt.bar(run_ids, avg_util)
    plt.xlabel("GA config (run_id)")
    plt.ylabel("avg utilization (over seeds)")
    plt.title("GA configs: average utilization")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ga_configs_avg_utilization.png"))
    plt.close()

    # avg seconds per config
    plt.figure()
    plt.bar(run_ids, avg_sec)
    plt.xlabel("GA config (run_id)")
    plt.ylabel("avg seconds (over seeds)")
    plt.title("GA configs: average runtime")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ga_configs_avg_seconds.png"))
    plt.close()


def read_convergence_files(pattern: str) -> List[Tuple[str, List[Dict[str, Any]]]]:
    files = sorted(glob.glob(pattern))
    out = []
    for fp in files:
        out.append((fp, read_csv(fp)))
    return out


def plot_convergence(convergence_rows: List[Dict[str, Any]], title: str, out_path: str) -> None:
    # parse
    data = []
    for r in convergence_rows:
        gen = to_int(r.get("gen", 0))

        # preferuj report (strict/partial), ale jak nie ma, weź eval (penalized)
        best = to_float(r.get("best_report", r.get("best_eval", r.get("best", 0))))
        avg  = to_float(r.get("avg_report",  r.get("avg_eval",  r.get("avg", 0))))
        fr   = to_float(r.get("feasible_rate", 0))

        data.append((gen, best, avg, fr))

    data.sort(key=lambda t: t[0])
    gens = [t[0] for t in data]
    bests = [t[1] for t in data]
    avgs = [t[2] for t in data]
    frs = [t[3] for t in data]

    base, _ = os.path.splitext(out_path)

    # best & avg
    plt.figure()
    plt.plot(gens, bests, label="best")
    plt.plot(gens, avgs, label="avg")
    plt.xlabel("generation")
    plt.ylabel("fitness")
    plt.title(title + " (report fitness)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(base + "_fitness.png")
    plt.close()

    # feasible_rate
    plt.figure()
    plt.plot(gens, frs, label="feasible_rate")
    plt.xlabel("generation")
    plt.ylabel("feasible_rate")
    plt.title(title + " (feasible rate)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(base + "_feasible_rate.png")
    plt.close()


def plot_convergence_by_config(conv_glob: str, out_dir: str) -> None:
    """
    Zamiast osobnych wykresów dla (run_id, seed), robimy 3 wykresy zbiorcze:
      1) gen_best_report vs gen (średnia po seedach) – osobna linia per run_id
      2) avg_report vs gen (średnia po seedach) – osobna linia per run_id
      3) feasible_rate vs gen (średnia po seedach) – osobna linia per run_id

    Wymaga, żeby w convergence CSV były kolumny:
      - run_id, seed, gen
      - gen_best_report, avg_report, feasible_rate
    (albo analogiczne *_eval jeśli chcesz liczyć penalized)
    """
    ensure_dir(out_dir)

    files = sorted(glob.glob(conv_glob))
    if not files:
        print("No convergence files found with glob:", conv_glob)
        return

    dfs = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
            df["__file__"] = os.path.basename(fp)
            dfs.append(df)
        except Exception as e:
            print("Failed reading:", fp, "error:", e)

    if not dfs:
        print("No convergence data loaded.")
        return

    df = pd.concat(dfs, ignore_index=True)

    # Upewnij się, że mamy kluczowe kolumny
    required = ["run_id", "seed", "gen", "feasible_rate"]
    for col in required:
        if col not in df.columns:
            print("Missing required column in convergence CSVs:", col)
            print("Available columns:", list(df.columns))
            return

    # wybieramy metryki "report"
    # jeśli nie ma report, można fallbackować na eval
    if "gen_best_report" in df.columns:
        best_col = "gen_best_report"
    elif "gen_best_eval" in df.columns:
        best_col = "gen_best_eval"
    else:
        # ostatecznie: best_report (global) bywa stałe w generacji, więc nie jest idealne,
        # ale lepiej niż nic
        best_col = "best_report" if "best_report" in df.columns else None

    if "avg_report" in df.columns:
        avg_col = "avg_report"
    elif "avg_eval" in df.columns:
        avg_col = "avg_eval"
    else:
        avg_col = None

    if best_col is None or avg_col is None:
        print("Missing columns for best/avg. Need one of:")
        print(" - gen_best_report or gen_best_eval (for best)")
        print(" - avg_report or avg_eval (for avg)")
        print("Available columns:", list(df.columns))
        return

    # konwersje numeryczne
    for col in ["gen", "seed", "feasible_rate", best_col, avg_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # agregacja: średnia po seedach per (run_id, gen)
    g = df.groupby(["run_id", "gen"], as_index=False).agg(
        best_mean=(best_col, "mean"),
        avg_mean=(avg_col, "mean"),
        feasible_mean=("feasible_rate", "mean"),
    )

    # sortowanie dla ładnych linii
    g = g.sort_values(["run_id", "gen"])

    run_ids = sorted([rid for rid in g["run_id"].dropna().unique() if str(rid).strip() != ""])
    if not run_ids:
        print("No run_id values found after aggregation.")
        return

    # ---- 1) BEST vs generation (średnia po seedach)
    plt.figure()
    for rid in run_ids:
        sub = g[g["run_id"] == rid]
        plt.plot(sub["gen"], sub["best_mean"], label=str(rid))
    plt.xlabel("generation")
    plt.ylabel("fitness (best, mean over seeds)")
    plt.title("Convergence: Best fitness vs generation (by config)")
    plt.legend(title="config (run_id)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "convergence_best_by_config.png"))
    plt.close()

    # ---- 2) AVG vs generation (średnia po seedach)
    plt.figure()
    for rid in run_ids:
        sub = g[g["run_id"] == rid]
        plt.plot(sub["gen"], sub["avg_mean"], label=str(rid))
    plt.xlabel("generation")
    plt.ylabel("fitness (avg, mean over seeds)")
    plt.title("Convergence: Avg fitness vs generation (by config)")
    plt.legend(title="config (run_id)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "convergence_avg_by_config.png"))
    plt.close()

    # ---- 3) feasible_rate vs generation (średnia po seedach)
    plt.figure()
    for rid in run_ids:
        sub = g[g["run_id"] == rid]
        plt.plot(sub["gen"], sub["feasible_mean"], label=str(rid))
    plt.xlabel("generation")
    plt.ylabel("feasible_rate (mean over seeds)")
    plt.title("Convergence: Feasible rate vs generation (by config)")
    plt.legend(title="config (run_id)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "convergence_feasible_rate_by_config.png"))
    plt.close()

    print("Saved aggregated convergence plots to:", out_dir)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=str, default="runs/summary.csv")
    ap.add_argument("--conv_glob", type=str, default="runs/convergence/*.csv")
    ap.add_argument("--out_dir", type=str, default="runs/plots")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    if os.path.exists(args.summary):
        summary_rows = read_csv(args.summary)
        plot_summary_scatter(summary_rows, args.out_dir)
        plot_ga_configs_bar(summary_rows, args.out_dir)
        print("Saved summary plots to:", args.out_dir)
    else:
        print("No summary.csv found at:", args.summary)

    conv_files = read_convergence_files(args.conv_glob)
    if not conv_files:
        print("No convergence files found with glob:", args.conv_glob)
        return

    plot_convergence_by_config(args.conv_glob, args.out_dir)

    print("Saved convergence plots to:", args.out_dir)


if __name__ == "__main__":
    main()
