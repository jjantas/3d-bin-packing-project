import pandas as pd
import glob
import matplotlib.pyplot as plt
import os
import argparse


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def safe_get(row: dict, key: str, default="-"):
    if key not in row:
        return default
    v = row.get(key)
    if pd.isna(v):
        return default
    return v


def label_from_row(row: dict) -> str:
    """Creates a readable label from row parameters."""
    rtr = safe_get(row, "ratio_to_remove", "-")
    pm = safe_get(row, "p_mut_move", "-")
    prs = safe_get(row, "p_mut_resupport", "-")
    initr = safe_get(row, "init_constructive_ratio", "-")
    pc = safe_get(row, "p_crossover", "-")

    def fmt(x):
        try:
            xf = float(x)
            if abs(xf - round(xf)) < 1e-9:
                return str(int(round(xf)))
            return f"{xf:.2f}"
        except Exception:
            return str(x)

    parts = [
        f"rtr:{fmt(rtr)}",
        f"pm:{fmt(pm)}",
        f"prs:{fmt(prs)}",
        f"init:{fmt(initr)}",
        f"pc:{fmt(pc)}",
    ]
    return " | ".join(parts)


def load_convergence_data(files_glob: str, metric_col: str):
    files = glob.glob(files_glob)
    if not files:
        print(f"[Hyperparams] No files found for pattern: {files_glob}")
        return None

    print(f"[Hyperparams] Loading {len(files)} files...")
    frames = []
    needed_cols = [
        "gen",
        metric_col,
        "cfg_id",
        "seed",
        "ratio_to_remove",
        "p_mut_move",
        "p_mut_resupport",
        "init_constructive_ratio",
        "p_crossover",
    ]

    for fp in files:
        try:
            df = pd.read_csv(fp)
            if "gen" not in df.columns or metric_col not in df.columns:
                continue

            if "cfg_id" not in df.columns:
                df["cfg_id"] = df.get("run_id", "unknown")

            for c in needed_cols:
                if c not in df.columns:
                    df[c] = pd.NA

            frames.append(df[needed_cols].copy())
        except Exception as e:
            print(f"Error reading {fp}: {e}")

    if not frames:
        return None

    df_all = pd.concat(frames, ignore_index=True)
    df_all["gen"] = pd.to_numeric(df_all["gen"], errors="coerce").fillna(0).astype(int)
    df_all[metric_col] = pd.to_numeric(df_all[metric_col], errors="coerce").fillna(0.0)

    params = [
        "ratio_to_remove",
        "p_mut_move",
        "p_mut_resupport",
        "init_constructive_ratio",
        "p_crossover",
    ]
    for p in params:
        df_all[p] = pd.to_numeric(df_all[p], errors="coerce")

    return df_all


def plot_convergence_lines(df_all, cfg_ids, metric_col, title, out_path, max_gen=80):
    df_subset = df_all[df_all["cfg_id"].isin(cfg_ids)].copy()
    if df_subset.empty:
        print("No data after filtering - nothing to plot.")
        return

    df_agg = df_subset.groupby(["cfg_id", "gen"])[metric_col].mean().reset_index()

    if max_gen is not None:
        df_agg = df_agg[df_agg["gen"] <= max_gen]

    plt.figure(figsize=(14, 8))

    for cfg_id in cfg_ids:
        line = df_agg[df_agg["cfg_id"] == cfg_id]
        if line.empty:
            continue
        row0 = df_subset[df_subset["cfg_id"] == cfg_id].iloc[0].to_dict()
        label = label_from_row(row0)
        plt.plot(line["gen"], line[metric_col], label=label, linewidth=1.5)

    plt.title(title)
    plt.xlabel("Generation")
    plt.ylabel(metric_col)

    if max_gen is not None:
        plt.xlim(0, max_gen)

    plt.legend(loc="lower right", fontsize="small", title="GA hyperparameters")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    ensure_dir(os.path.dirname(out_path) if os.path.dirname(out_path) else ".")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def run_hyperparams_analysis(args):
    out_dir = os.path.join(args.out_dir, "hyperparams")
    ensure_dir(out_dir)

    df_all = load_convergence_data(args.conv_glob, args.metric)
    if df_all is None or df_all.empty:
        return

    ranking = df_all.groupby("cfg_id")[args.metric].max().sort_values(ascending=False)
    top_cfgs = ranking.head(args.top).index.tolist()
    bot_cfgs = ranking.tail(args.top).index.tolist()

    plot_convergence_lines(
        df_all,
        top_cfgs,
        args.metric,
        f"TOP {args.top} Configurations",
        os.path.join(out_dir, f"TOP_{args.top}_{args.metric}.png"),
        args.max_gen,
    )
    plot_convergence_lines(
        df_all,
        bot_cfgs,
        args.metric,
        f"BOTTOM {args.top} Configurations",
        os.path.join(out_dir, f"BOTTOM_{args.top}_{args.metric}.png"),
        args.max_gen,
    )


def run_summary_analysis(args):
    out_dir = os.path.join(args.out_dir, "summary")
    ensure_dir(out_dir)

    if not os.path.exists(args.summary_file):
        print(f"[Summary] File {args.summary_file} missing. Skipping summary analysis.")
        return

    df = pd.read_csv(args.summary_file)
    print(f"[Summary] Loaded {len(df)} rows from summary.csv")

    cols = ["best_fitness", "utilization", "seconds", "seed"]
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    if "algo" in df.columns:
        plt.figure(figsize=(10, 6))
        for algo, group in df.groupby("algo"):
            plt.scatter(group["seed"], group["best_fitness"], label=algo, alpha=0.7)
            plt.axhline(
                group["best_fitness"].mean(),
                linestyle="--",
                alpha=0.5,
                label=f"{algo} mean",
            )

        plt.xlabel("Seed")
        plt.ylabel("Best Fitness")
        plt.title("Results Comparison: GA vs Random Search")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(out_dir, "summary_scatter_fitness.png"))
        plt.close()

        plt.figure(figsize=(10, 6))
        for algo, group in df.groupby("algo"):
            plt.scatter(
                group["seed"], group["utilization"] * 100, label=algo, alpha=0.7
            )

        plt.xlabel("Seed")
        plt.ylabel("Warehouse Filling [%]")
        plt.title("Fill Rate: GA vs Random Search")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(out_dir, "summary_scatter_utilization.png"))
        plt.close()

    ga_df = df[df["algo"] == "ga"].copy() if "algo" in df.columns else df.copy()

    if not ga_df.empty and "run_id" in ga_df.columns:
        agg = (
            ga_df.groupby("run_id")[["best_fitness", "utilization", "seconds"]]
            .mean()
            .reset_index()
        )

        agg = agg.sort_values("best_fitness", ascending=True)

        plt.figure(figsize=(12, 6))
        plt.barh(agg["run_id"], agg["best_fitness"])
        plt.xlabel("Average Best Fitness")
        plt.title("GA Configuration Ranking (mean of runs)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "ga_configs_ranking.png"))
        plt.close()

    conv_files = glob.glob(args.conv_glob)
    if not conv_files:
        return

    single_dir = os.path.join(out_dir, "single_runs")
    ensure_dir(single_dir)

    for fp in sorted(conv_files)[:10]:  # Limit to 10 examples
        try:
            cdf = pd.read_csv(fp)
            name = os.path.basename(fp).replace(".csv", "")

            plt.figure(figsize=(8, 5))
            if "avg_report" in cdf.columns:
                plt.plot(cdf["gen"], cdf["avg_report"], label="Avg (Pop)", alpha=0.7)
            if "best_report" in cdf.columns:
                plt.plot(
                    cdf["gen"], cdf["best_report"], label="Best (Ind)", linewidth=2
                )

            plt.title(f"Convergence: {name}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(single_dir, f"{name}.png"))
            plt.close()
        except:
            pass


def main():
    parser = argparse.ArgumentParser(description="Complex GA results analysis (Merger)")

    parser.add_argument(
        "--mode",
        choices=["all", "hyperparams", "summary"],
        default="all",
        help="Analysis mode: 'hyperparams' (Phase A grid), 'summary' (GA vs Random), or 'all'",
    )
    parser.add_argument(
        "--out_dir", type=str, default="runs/plots", help="Output directory"
    )

    parser.add_argument(
        "--summary_file",
        type=str,
        default="runs/summary.csv",
        help="Path to summary file",
    )
    parser.add_argument(
        "--conv_glob",
        type=str,
        default="runs/convergence/**/*.csv",
        help="Glob pattern for convergence files (recursive search)",
    )

    parser.add_argument(
        "--metric",
        default="avg_report",
        help="Metric to analyze (e.g., avg_report, best_report)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="How many configs to show in TOP/BOTTOM plots",
    )
    parser.add_argument(
        "--max_gen", type=int, default=80, help="X-axis limit (generations)"
    )
    parser.add_argument(
        "--score_mode",
        choices=["max", "last", "mean"],
        default="max",
        help="How to aggregate run result to a scalar",
    )

    args = parser.parse_args()

    print(f"--- Starting analysis in mode: {args.mode} ---")

    if args.mode in ["summary", "all"]:
        run_summary_analysis(args)

    if args.mode in ["hyperparams", "all"]:
        run_hyperparams_analysis(args)

    print(f"--- Done. Results in: {args.out_dir} ---")


if __name__ == "__main__":
    main()
