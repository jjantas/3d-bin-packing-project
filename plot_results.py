# src/binpack3d/plot_results.py
from __future__ import annotations
import argparse
import csv
import glob
import os
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
    """
    Rysuje 3 osobne wykresy (osobne PNG) albo 1 wykres na 3 serie?
    Zrobimy 2 PNG:
      - best & avg
      - feasible_rate
    """
    # parse
    data = []
    for r in convergence_rows:
        gen = to_int(r.get("gen", 0))
        best = to_float(r.get("best", 0))
        avg = to_float(r.get("avg", 0))
        fr = to_float(r.get("feasible_rate", 0))
        data.append((gen, best, avg, fr))
    data.sort(key=lambda t: t[0])

    gens = [t[0] for t in data]
    bests = [t[1] for t in data]
    avgs = [t[2] for t in data]
    frs = [t[3] for t in data]

    base, ext = os.path.splitext(out_path)

    # best & avg
    plt.figure()
    plt.plot(gens, bests, label="best")
    plt.plot(gens, avgs, label="avg")
    plt.xlabel("generation")
    plt.ylabel("fitness")
    plt.title(title + " (fitness)")
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

    # grupuj po (run_id, seed) wg nazwy pliku albo kolumn
    for fp, rows in conv_files:
        # tytuł z nazwy pliku
        name = os.path.basename(fp).replace(".csv", "")
        plot_convergence(rows, title=name, out_path=os.path.join(args.out_dir, name + ".png"))

    print("Saved convergence plots to:", args.out_dir)


if __name__ == "__main__":
    main()
