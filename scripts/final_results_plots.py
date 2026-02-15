import os
import pandas as pd
import matplotlib.pyplot as plt

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def main(summary_csv="runs/summary.csv", out_dir="runs/plots_final"):
    ensure_dir(out_dir)
    df = pd.read_csv(summary_csv)

    # standaryzacja typów
    for col in ["best_fitness","utilization","seconds","seed"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["algo"] = df["algo"].astype(str)

    # --- wybierz “jedno GA” do porównania z randomem ---
    # jeśli masz wiele run_id, weź ten z najlepszą średnią best_fitness
    ga = df[df["algo"] == "ga"].copy()
    rnd = df[df["algo"] == "random_search"].copy()

    ga_best = ga
    best_run_id = None
    if "run_id" in ga.columns and ga["run_id"].notna().any():
        by = ga.groupby("run_id")["best_fitness"].mean().sort_values(ascending=False)
        best_run_id = by.index[0]
        ga_best = ga[ga["run_id"] == best_run_id].copy()

    # --- tabela stats ---
    def stats(block, name):
        return {
            "name": name,
            "n": int(block["seed"].nunique()),
            "best_fitness_mean": float(block["best_fitness"].mean()),
            "best_fitness_std": float(block["best_fitness"].std()),
            "util_mean": float(block["utilization"].mean()),
            "util_std": float(block["utilization"].std()),
            "sec_mean": float(block["seconds"].mean()),
            "sec_std": float(block["seconds"].std()),
        }

    rows = [stats(rnd, "random_search"), stats(ga_best, f"ga ({best_run_id})" if best_run_id else "ga")]
    stats_df = pd.DataFrame(rows)
    stats_path = os.path.join(out_dir, "final_stats.csv")
    stats_df.to_csv(stats_path, index=False)

    # --- win rate (seed-wise) ---
    # porównujemy per seed: GA vs Random
    merged = pd.merge(
        rnd[["seed","best_fitness"]].rename(columns={"best_fitness":"rnd_best"}),
        ga_best[["seed","best_fitness"]].rename(columns={"best_fitness":"ga_best"}),
        on="seed", how="inner"
    )
    win_rate = (merged["ga_best"] > merged["rnd_best"]).mean() if len(merged) else 0.0
    with open(os.path.join(out_dir, "win_rate.txt"), "w", encoding="utf-8") as f:
        f.write(f"win_rate_ga_vs_random={win_rate:.3f}\n")
        f.write(f"seeds_compared={len(merged)}\n")

    # --- boxplot: best_fitness ---
    plt.figure()
    plt.boxplot([rnd["best_fitness"].dropna(), ga_best["best_fitness"].dropna()], labels=["Random","GA"])
    plt.ylabel("best_fitness")
    plt.title("Final results: Best fitness (across seeds)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "box_best_fitness.png"))
    plt.close()

    # --- boxplot: utilization ---
    plt.figure()
    plt.boxplot([rnd["utilization"].dropna(), ga_best["utilization"].dropna()], labels=["Random","GA"])
    plt.ylabel("utilization")
    plt.title("Final results: Utilization (across seeds)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "box_utilization.png"))
    plt.close()

    # --- boxplot: seconds ---
    plt.figure()
    plt.boxplot([rnd["seconds"].dropna(), ga_best["seconds"].dropna()], labels=["Random","GA"])
    plt.ylabel("seconds")
    plt.title("Runtime (across seeds)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "box_seconds.png"))
    plt.close()

    # --- scatter: quality vs time ---
    plt.figure()
    plt.scatter(rnd["seconds"], rnd["best_fitness"], label="Random")
    plt.scatter(ga_best["seconds"], ga_best["best_fitness"], label="GA")
    plt.xlabel("seconds")
    plt.ylabel("best_fitness")
    plt.title("Quality vs runtime")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "scatter_quality_vs_time.png"))
    plt.close()

    # --- (opcjonalnie) leaderboard GA configs ---
    if "run_id" in ga.columns and ga["run_id"].notna().any():
        lb = ga.groupby("run_id")["best_fitness"].mean().sort_values(ascending=False)
        plt.figure()
        plt.bar(lb.index.astype(str), lb.values)
        plt.xlabel("run_id")
        plt.ylabel("mean best_fitness")
        plt.title("GA configs leaderboard (mean across seeds)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "ga_leaderboard_mean_best_fitness.png"))
        plt.close()

    print("Saved plots & stats to:", out_dir)
    print("Best GA run_id:", best_run_id)
    print("Win rate GA vs Random:", win_rate)

if __name__ == "__main__":
    main()
