import pandas as pd
import glob
import matplotlib.pyplot as plt
import os
import argparse

# ================= CONFIG =================
FILES_GLOB = "runs/convergence/A/*.csv"
# =========================================


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_get(row: dict, key: str, default="-"):
    if key not in row:
        return default
    v = row.get(key)
    if pd.isna(v):
        return default
    return v


def label_from_row(row: dict) -> str:
    """
    Build a compact legend label from GA hyperparameters stored in the CSV row.
    """
    rtr = safe_get(row, "ratio_to_remove", "-")
    pm = safe_get(row, "p_mut_move", "-")
    prs = safe_get(row, "p_mut_resupport", "-")
    initr = safe_get(row, "init_constructive_ratio", "-")
    pc = safe_get(row, "p_crossover", "-")
    gen_total = safe_get(row, "generations", "-")

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

    if gen_total != "-" and gen_total is not None:
        parts.append(f"gen:{fmt(gen_total)}")

    return " | ".join(parts)


# -------------------------
# Reading + preprocessing
# -------------------------
def load_data(metric_col: str):
    files = glob.glob(FILES_GLOB)
    if not files:
        print("ERROR: No CSV files found.")
        return None

    print(f"Found {len(files)} files. Loading...")

    frames = []
    needed_cols = [
        "gen", metric_col, "cfg_id", "seed",
        "ratio_to_remove", "p_mut_move", "p_mut_resupport",
        "init_constructive_ratio", "p_crossover", "generations",
    ]

    for fp in files:
        df = pd.read_csv(fp)
        if "gen" not in df.columns or metric_col not in df.columns:
            continue

        # fallback for older logs
        if "cfg_id" not in df.columns and "run_id" in df.columns:
            df["cfg_id"] = df["run_id"]

        if "cfg_id" not in df.columns:
            continue

        for c in needed_cols:
            if c not in df.columns:
                df[c] = pd.NA

        frames.append(df[needed_cols].copy())

    if not frames:
        print("ERROR: No valid data loaded.")
        return None

    df_all = pd.concat(frames, ignore_index=True)

    df_all["gen"] = pd.to_numeric(df_all["gen"], errors="coerce").fillna(0).astype(int)
    df_all[metric_col] = pd.to_numeric(df_all[metric_col], errors="coerce").fillna(0.0)
    df_all["seed"] = pd.to_numeric(df_all["seed"], errors="coerce").fillna(0).astype(int)

    # params as numeric (for grouping/pivot)
    for p in ["ratio_to_remove", "p_mut_move", "p_mut_resupport", "init_constructive_ratio", "p_crossover"]:
        df_all[p] = pd.to_numeric(df_all[p], errors="coerce")

    return df_all


# -------------------------
# Scalar per (cfg_id, seed)
# -------------------------
def reduce_to_scalar_per_cfg_seed(df_all: pd.DataFrame, metric_col: str, score_mode: str):
    """
    Returns df_scalar with columns:
      cfg_id, seed, score, ratio_to_remove, p_mut_move, p_mut_resupport, init_constructive_ratio, p_crossover

    score_mode:
      - "max": max(metric) over generations
      - "last": metric value at the last generation
      - "mean": mean(metric) over generations (stability)
    """
    params = ["ratio_to_remove", "p_mut_move", "p_mut_resupport", "init_constructive_ratio", "p_crossover"]

    if score_mode == "last":
        idx = df_all.groupby(["cfg_id", "seed"])["gen"].idxmax()
        df_scalar = df_all.loc[idx].copy()
        df_scalar = df_scalar[["cfg_id", "seed", metric_col] + params].rename(columns={metric_col: "score"})
    elif score_mode == "mean":
        agg = {metric_col: "mean"}
        for p in params:
            agg[p] = "first"
        df_scalar = df_all.groupby(["cfg_id", "seed"]).agg(agg).reset_index().rename(columns={metric_col: "score"})
    else:
        # "max"
        agg = {metric_col: "max"}
        for p in params:
            agg[p] = "first"
        df_scalar = df_all.groupby(["cfg_id", "seed"]).agg(agg).reset_index().rename(columns={metric_col: "score"})

    df_scalar = df_scalar.dropna(subset=params, how="any")
    return df_scalar


# -------------------------
# TOP/BOTTOM convergence plots
# -------------------------
def rank_configs(df_all: pd.DataFrame, metric_col: str, rank_mode: str = "max"):
    """
    Returns a Series: index=cfg_id, value=score used for ranking.

    rank_mode:
      - "max": max(metric) over all generations
      - "last": metric at the last generation
      - "mean": mean(metric) over generations
    """
    if rank_mode == "last":
        idx = df_all.groupby("cfg_id")["gen"].idxmax()
        return df_all.loc[idx].set_index("cfg_id")[metric_col].sort_values(ascending=False)
    if rank_mode == "mean":
        return df_all.groupby("cfg_id")[metric_col].mean().sort_values(ascending=False)
    return df_all.groupby("cfg_id")[metric_col].max().sort_values(ascending=False)


def plot_convergence_lines(df_all, cfg_ids, metric_col, title, out_path, max_gen=80):
    df_subset = df_all[df_all["cfg_id"].isin(cfg_ids)].copy()
    if df_subset.empty:
        print("No data after filtering - nothing to plot.")
        return

    # average across seeds for each generation
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


# -------------------------
# Main effects (1D)
# -------------------------
def plot_main_effect(df_scalar: pd.DataFrame, param: str, out_path: str, title: str, y_label: str):
    """
    1D plot: effect of a single hyperparameter on score (mean/median + min/max band).
    """
    g = df_scalar.groupby(param)["score"]
    stats = g.agg(["mean", "median", "min", "max", "count"]).reset_index().sort_values(param)

    plt.figure(figsize=(10, 6))
    plt.plot(stats[param], stats["mean"], label="mean", marker="o")
    plt.plot(stats[param], stats["median"], label="median", marker="o")
    plt.fill_between(stats[param], stats["min"], stats["max"], alpha=0.2, label="min..max")

    plt.title(title)
    plt.xlabel(param)
    plt.ylabel(y_label)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    ensure_dir(os.path.dirname(out_path) if os.path.dirname(out_path) else ".")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


# -------------------------
# Heatmaps (2D)
# -------------------------
def plot_heatmap_mean(df_scalar: pd.DataFrame, x_param: str, y_param: str, out_path: str, title: str, cbar_label: str):
    """
    Heatmap of mean(score) for a pair of hyperparameters.
    """
    df_mean = df_scalar.groupby([x_param, y_param])["score"].mean().reset_index()
    mat = df_mean.pivot(index=y_param, columns=x_param, values="score")
    mat = mat.sort_index(axis=0).sort_index(axis=1)

    plt.figure(figsize=(10, 7))
    im = plt.imshow(mat.values, aspect="auto", origin="lower")
    plt.title(title)
    plt.xlabel(x_param)
    plt.ylabel(y_param)

    plt.xticks(range(len(mat.columns)), [f"{v:.2f}" for v in mat.columns])
    plt.yticks(range(len(mat.index)), [f"{v:.2f}" for v in mat.index])

    plt.colorbar(im, label=cbar_label)
    plt.tight_layout()

    ensure_dir(os.path.dirname(out_path) if os.path.dirname(out_path) else ".")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_heatmap_slices(df_scalar: pd.DataFrame, x_param: str, y_param: str, slice_param: str, out_dir: str, base_title: str, cbar_label: str):
    """
    Small multiples: one heatmap per value of slice_param.
    """
    vals = sorted([v for v in df_scalar[slice_param].dropna().unique()])
    for v in vals:
        sub = df_scalar[df_scalar[slice_param] == v].copy()
        if sub.empty:
            continue
        out_path = os.path.join(out_dir, f"HEATMAP_{y_param}_vs_{x_param}_slice_{slice_param}_{v:.2f}.png")
        title = f"{base_title} | slice {slice_param}={v:.2f}"
        plot_heatmap_mean(sub, x_param, y_param, out_path, title, cbar_label)


# -------------------------
# Main
# -------------------------
def run_analysis(metric_col, top_n, max_gen, out_dir, rank_mode, score_mode, do_heatmaps, do_main_effects, do_slices):
    df_all = load_data(metric_col)
    if df_all is None or df_all.empty:
        return

    ensure_dir(out_dir)

    # TOP/BOTTOM convergence plots
    ranking = rank_configs(df_all, metric_col, rank_mode)
    top_cfgs = ranking.head(top_n).index.tolist()
    worst_cfgs = ranking.tail(top_n).index.tolist()

    plot_convergence_lines(
        df_all, top_cfgs, metric_col,
        f"TOP {top_n} configurations ({metric_col})",
        os.path.join(out_dir, f"TOP_{top_n}_{metric_col}.png"),
        max_gen
    )

    plot_convergence_lines(
        df_all, worst_cfgs, metric_col,
        f"BOTTOM {top_n} configurations ({metric_col})",
        os.path.join(out_dir, f"BOTTOM_{top_n}_{metric_col}.png"),
        max_gen
    )

    # Reduce to scalar per (cfg_id, seed) for parameter influence analysis
    df_scalar = reduce_to_scalar_per_cfg_seed(df_all, metric_col, score_mode=score_mode)

    # Better y labels for feasibility
    if metric_col == "feasible_rate":
        y_label = "feasible_rate (0..1)"
        cbar_label = "mean(feasible_rate)"
    else:
        y_label = "score"
        cbar_label = "mean(score)"

    # Main effects (1D)
    if do_main_effects:
        params = [
            "ratio_to_remove",
            "p_mut_move",
            "p_mut_resupport",
            "init_constructive_ratio",
            "p_crossover",
        ]
        for p in params:
            plot_main_effect(
                df_scalar,
                param=p,
                out_path=os.path.join(out_dir, f"MAIN_EFFECT_{metric_col}_{p}_{score_mode}.png"),
                title=f"Main effect: {p} -> {metric_col} (score={score_mode})",
                y_label=y_label
            )

    # Heatmaps (2D)
    if do_heatmaps:
        pairs = [
            ("ratio_to_remove", "p_mut_move"),
            ("ratio_to_remove", "p_mut_resupport"),
            ("p_mut_move", "p_mut_resupport"),
            ("ratio_to_remove", "init_constructive_ratio"),
            ("p_mut_move", "init_constructive_ratio"),
            ("ratio_to_remove", "p_crossover"),
            ("p_mut_move", "p_crossover"),
        ]
        for x, y in pairs:
            plot_heatmap_mean(
                df_scalar,
                x_param=x,
                y_param=y,
                out_path=os.path.join(out_dir, f"HEATMAP_{metric_col}_{y}_vs_{x}_{score_mode}.png"),
                title=f"Heatmap: mean({metric_col}) by {y} x {x} (score={score_mode})",
                cbar_label=cbar_label
            )

    # Slices (small multiples)
    if do_slices:
        base_dir = os.path.join(out_dir, "slices")
        ensure_dir(base_dir)

        plot_heatmap_slices(
            df_scalar,
            x_param="ratio_to_remove",
            y_param="p_mut_move",
            slice_param="p_mut_resupport",
            out_dir=os.path.join(base_dir, "slice_presupport"),
            base_title=f"Heatmap pmove x rtr ({metric_col}, score={score_mode})",
            cbar_label=cbar_label
        )
        plot_heatmap_slices(
            df_scalar,
            x_param="ratio_to_remove",
            y_param="p_mut_move",
            slice_param="init_constructive_ratio",
            out_dir=os.path.join(base_dir, "slice_init_ratio"),
            base_title=f"Heatmap pmove x rtr ({metric_col}, score={score_mode})",
            cbar_label=cbar_label
        )
        plot_heatmap_slices(
            df_scalar,
            x_param="ratio_to_remove",
            y_param="p_mut_move",
            slice_param="p_crossover",
            out_dir=os.path.join(base_dir, "slice_pcross"),
            base_title=f"Heatmap pmove x rtr ({metric_col}, score={score_mode})",
            cbar_label=cbar_label
        )

    print(f"Done for metric={metric_col}.")


def main():
    ap = argparse.ArgumentParser(description="GA convergence + hyperparameter influence plots (Phase A only)")
    ap.add_argument("--metric", choices=["avg_report", "best_report", "feasible_rate"], default="avg_report",
                    help="Which metric column from GA logs to analyze")
    ap.add_argument("--top", type=int, default=10, help="How many configs to show for TOP/BOTTOM plots")
    ap.add_argument("--max_gen", type=int, default=80,
                    help="Cut convergence plots at this generation (e.g., 80). Use -1 to disable.")
    ap.add_argument("--out_dir", type=str, default="runs/plots_A", help="Output directory for PNG files")

    ap.add_argument("--rank_mode", choices=["max", "last", "mean"], default="max",
                    help="Ranking method for TOP/BOTTOM: max/last/mean")
    ap.add_argument("--score_mode", choices=["max", "last", "mean"], default="max",
                    help="How to reduce each run to a scalar score for main-effects/heatmaps: max/last/mean")

    ap.add_argument("--main_effects", action="store_true", help="Generate 1D main-effect plots for all hyperparameters")
    ap.add_argument("--heatmaps", action="store_true", help="Generate 2D heatmaps for selected parameter pairs")
    ap.add_argument("--slices", action="store_true", help="Generate sliced heatmaps (pmove x rtr) by other parameters")

    ap.add_argument("--also_feasible", action="store_true",
                    help="If set, runs a second analysis pass for feasible_rate (same flags).")

    args = ap.parse_args()

    max_gen = None if args.max_gen < 0 else args.max_gen

    # primary metric
    run_analysis(
        metric_col=args.metric,
        top_n=args.top,
        max_gen=max_gen,
        out_dir=args.out_dir,
        rank_mode=args.rank_mode,
        score_mode=args.score_mode,
        do_heatmaps=args.heatmaps,
        do_main_effects=args.main_effects,
        do_slices=args.slices,
    )

    # optional feasible_rate pass
    if args.also_feasible and args.metric != "feasible_rate":
        out2 = os.path.join(args.out_dir, "feasible_rate")
        run_analysis(
            metric_col="feasible_rate",
            top_n=args.top,
            max_gen=max_gen,
            out_dir=out2,
            rank_mode=args.rank_mode,
            score_mode=args.score_mode,
            do_heatmaps=args.heatmaps,
            do_main_effects=args.main_effects,
            do_slices=args.slices,
        )


if __name__ == "__main__":
    main()
