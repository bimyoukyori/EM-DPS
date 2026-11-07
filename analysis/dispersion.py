import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def safe_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    a = a[m] - a[m].mean()
    b = b[m] - b[m].mean()
    den = np.sqrt((a*a).sum() * (b*b).sum())
    if den <= 1e-12:
        return np.nan
    return float((a*b).sum() / den)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/guidance_strength/dps_lr0p02_k4_64cases")
    ap.add_argument("--out", default="runs/dispersion")
    args = ap.parse_args()

    run = Path(args.run)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rank_p = run / "dps_fdtd_ranking.csv"
    best_p = run / "dps_best_by_case.csv"
    sample_dir = run / "samples"

    rank = pd.read_csv(rank_p)
    best = pd.read_csv(best_p)

    rows = []

    for case, g in rank.groupby("case"):
        g = g.sort_values("fdtd_mse").reset_index(drop=True)

        best_mse = float(g.loc[0, "fdtd_mse"])
        second_mse = float(g.loc[1, "fdtd_mse"]) if len(g) > 1 else np.nan
        mse_margin = second_mse - best_mse if np.isfinite(second_mse) else np.nan

        best_corr = float(g.loc[0, "fdtd_corr"])
        mean_mse = float(g["fdtd_mse"].mean())
        std_mse = float(g["fdtd_mse"].std(ddof=1))
        mean_corr = float(g["fdtd_corr"].mean())
        std_corr = float(g["fdtd_corr"].std(ddof=1))

        npz = sample_dir / f"case{int(case):03d}.npz"
        eps_std_mean = np.nan
        eps_std_max = np.nan
        eps_range_mean = np.nan

        if npz.exists():
            d = np.load(npz)
            if "eps" in d:
                eps = d["eps"].astype("float32")
                if eps.ndim == 4:
                    eps = eps[:, 0]
                eps_std = eps.std(axis=0)
                eps_std_mean = float(eps_std.mean())
                eps_std_max = float(eps_std.max())
                eps_range_mean = float((eps.max(axis=0) - eps.min(axis=0)).mean())

                plt.figure(figsize=(6, 4))
                plt.imshow(eps_std, cmap="magma", aspect="auto")
                plt.colorbar(label="std of eps candidates")
                plt.title(f"case {int(case):03d} uncertainty map")
                plt.axis("off")
                plt.tight_layout()
                plt.savefig(out / f"case{int(case):03d}_eps_uncertainty.png", dpi=150)
                plt.close()

        rows.append({
            "case": int(case),
            "K": int(len(g)),
            "best_sample": int(g.loc[0, "sample"]),
            "best_mse": best_mse,
            "second_mse": second_mse,
            "mse_margin": mse_margin,
            "best_corr": best_corr,
            "mean_candidate_mse": mean_mse,
            "std_candidate_mse": std_mse,
            "mean_candidate_corr": mean_corr,
            "std_candidate_corr": std_corr,
            "eps_std_mean": eps_std_mean,
            "eps_std_max": eps_std_max,
            "eps_range_mean": eps_range_mean,
            "bad_corr_lt_0p8": bool(best_corr < 0.8),
            "bad_mse_gt_0p02": bool(best_mse > 0.02),
        })

    tab = pd.DataFrame(rows)
    tab["bad"] = tab["bad_corr_lt_0p8"] | tab["bad_mse_gt_0p02"]
    tab.to_csv(out / "uncertainty_summary.csv", index=False)

    summary = pd.DataFrame([{
        "N": len(tab),
        "mean_best_mse": tab["best_mse"].mean(),
        "mean_best_corr": tab["best_corr"].mean(),
        "mean_eps_std_mean": tab["eps_std_mean"].mean(),
        "mean_std_candidate_mse": tab["std_candidate_mse"].mean(),
        "corr_eps_std_vs_best_mse": safe_corr(tab["eps_std_mean"], tab["best_mse"]),
        "corr_eps_std_vs_best_corr": safe_corr(tab["eps_std_mean"], tab["best_corr"]),
        "corr_mse_margin_vs_best_mse": safe_corr(tab["mse_margin"], tab["best_mse"]),
        "bad_cases": int(tab["bad"].sum()),
        "bad_mean_eps_std": tab.loc[tab["bad"], "eps_std_mean"].mean(),
        "good_mean_eps_std": tab.loc[~tab["bad"], "eps_std_mean"].mean(),
        "bad_mean_std_candidate_mse": tab.loc[tab["bad"], "std_candidate_mse"].mean(),
        "good_mean_std_candidate_mse": tab.loc[~tab["bad"], "std_candidate_mse"].mean(),
    }])
    summary.to_csv(out / "uncertainty_global_summary.csv", index=False)

    plt.figure(figsize=(5, 4))
    plt.scatter(tab["eps_std_mean"], tab["best_mse"])
    plt.xlabel("Mean candidate eps std")
    plt.ylabel("Best FDTD MSE")
    plt.tight_layout()
    plt.savefig(out / "scatter_eps_uncertainty_vs_mse.png", dpi=150)
    plt.close()

    plt.figure(figsize=(5, 4))
    plt.scatter(tab["eps_std_mean"], tab["best_corr"])
    plt.xlabel("Mean candidate eps std")
    plt.ylabel("Best FDTD corr")
    plt.tight_layout()
    plt.savefig(out / "scatter_eps_uncertainty_vs_corr.png", dpi=150)
    plt.close()

    print("\n=== Uncertainty global summary ===")
    print(summary.to_string(index=False))

    print("\n=== Worst uncertainty cases ===")
    print(tab.sort_values("eps_std_mean", ascending=False).head(15).to_string(index=False))

    print("\n=== Bad cases with uncertainty ===")
    print(tab[tab["bad"]].sort_values("best_corr").to_string(index=False))

    print("\nsaved to:", out)

if __name__ == "__main__":
    main()
