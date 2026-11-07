import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def load_best(run_dir):
    run_dir = Path(run_dir)
    p = run_dir / "dps_best_by_case.csv"
    if not p.exists():
        raise FileNotFoundError(f"missing {p}")

    df = pd.read_csv(p)

    need = ["case", "sample", "fdtd_mse", "fdtd_mae", "fdtd_corr", "h5_index"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise RuntimeError(f"{p} missing columns {miss}; columns={list(df.columns)}")

    df = df[need].copy()
    df = df.sort_values(["case", "fdtd_mse"]).groupby("case", as_index=False).first()
    return df

def method_summary(all_df):
    rows = []
    for method, g in all_df.groupby("method"):
        rows.append({
            "method": method,
            "N": len(g),
            "mean_mse": g["fdtd_mse"].mean(),
            "median_mse": g["fdtd_mse"].median(),
            "std_mse": g["fdtd_mse"].std(ddof=1),
            "mean_mae": g["fdtd_mae"].mean(),
            "median_mae": g["fdtd_mae"].median(),
            "std_mae": g["fdtd_mae"].std(ddof=1),
            "mean_corr": g["fdtd_corr"].mean(),
            "median_corr": g["fdtd_corr"].median(),
            "std_corr": g["fdtd_corr"].std(ddof=1),
            "bad_corr_lt_0p8": int((g["fdtd_corr"] < 0.8).sum()),
            "bad_mse_gt_0p02": int((g["fdtd_mse"] > 0.02).sum()),
        })
    return pd.DataFrame(rows).sort_values("method")

def compare_to_baseline(all_df, baseline):
    base = all_df[all_df["method"] == baseline].copy()
    if len(base) == 0:
        raise RuntimeError(f"baseline not found: {baseline}")

    rows = []
    per_case_all = []

    for method in sorted(all_df["method"].unique()):
        if method == baseline:
            continue

        cur = all_df[all_df["method"] == method].copy()
        m = base.merge(cur, on="case", suffixes=("_base", "_method"))

        if len(m) == 0:
            continue

        if "h5_index_base" in m.columns and "h5_index_method" in m.columns:
            bad = m[m["h5_index_base"] != m["h5_index_method"]]
            if len(bad) > 0:
                raise RuntimeError(f"h5_index mismatch for {method} vs {baseline}")

        m["mse_gain"] = m["fdtd_mse_base"] - m["fdtd_mse_method"]
        m["mae_gain"] = m["fdtd_mae_base"] - m["fdtd_mae_method"]
        m["corr_gain"] = m["fdtd_corr_method"] - m["fdtd_corr_base"]

        m["mse_gain_pct"] = 100.0 * m["mse_gain"] / m["fdtd_mse_base"].clip(lower=1e-12)
        m["mae_gain_pct"] = 100.0 * m["mae_gain"] / m["fdtd_mae_base"].clip(lower=1e-12)

        rows.append({
            "baseline": baseline,
            "method": method,
            "N": len(m),
            "mse_improved_cases": int((m["mse_gain"] > 0).sum()),
            "mae_improved_cases": int((m["mae_gain"] > 0).sum()),
            "corr_improved_cases": int((m["corr_gain"] > 0).sum()),
            "mean_mse_gain": m["mse_gain"].mean(),
            "median_mse_gain": m["mse_gain"].median(),
            "mean_mse_gain_pct": m["mse_gain_pct"].mean(),
            "median_mse_gain_pct": m["mse_gain_pct"].median(),
            "mean_mae_gain": m["mae_gain"].mean(),
            "median_mae_gain": m["mae_gain"].median(),
            "mean_mae_gain_pct": m["mae_gain_pct"].mean(),
            "median_mae_gain_pct": m["mae_gain_pct"].median(),
            "mean_corr_gain": m["corr_gain"].mean(),
            "median_corr_gain": m["corr_gain"].median(),
            "worst_method_case_by_mse": int(m.loc[m["fdtd_mse_method"].idxmax(), "case"]),
            "worst_method_mse": float(m["fdtd_mse_method"].max()),
            "worst_method_corr": float(m.loc[m["fdtd_mse_method"].idxmax(), "fdtd_corr_method"]),
            "worst_gain_case_by_mse": int(m.loc[m["mse_gain"].idxmin(), "case"]),
            "worst_mse_gain": float(m["mse_gain"].min()),
            "worst_corr_gain": float(m.loc[m["mse_gain"].idxmin(), "corr_gain"]),
        })

        m["baseline"] = baseline
        m["method"] = method
        per_case_all.append(m)

    gain = pd.DataFrame(rows)
    per_case = pd.concat(per_case_all, ignore_index=True) if per_case_all else pd.DataFrame()
    return gain, per_case

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="+", required=True, help="METHOD=RUN_DIR")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    dfs = []
    for item in args.run:
        if "=" not in item:
            raise RuntimeError(f"bad --run item: {item}")
        method, path = item.split("=", 1)
        df = load_best(path)
        df["method"] = method
        df["run_dir"] = path
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)
    all_df.to_csv(out / "all_methods_per_case.csv", index=False)

    ms = method_summary(all_df)
    ms.to_csv(out / "method_summary.csv", index=False)

    gs, pc = compare_to_baseline(all_df, args.baseline)
    gs.to_csv(out / "gain_summary.csv", index=False)
    pc.to_csv(out / "pairwise_per_case.csv", index=False)

    bad = all_df[(all_df["fdtd_corr"] < 0.8) | (all_df["fdtd_mse"] > 0.02)].copy()
    bad = bad.sort_values(["method", "fdtd_corr", "fdtd_mse"], ascending=[True, True, False])
    bad.to_csv(out / "difficult_cases.csv", index=False)

    print("\n=== Method summary ===")
    print(ms.to_string(index=False))

    print("\n=== Gain summary ===")
    print(gs.to_string(index=False))

    print("\n=== Bad cases ===")
    if len(bad) == 0:
        print("None")
    else:
        print(bad.head(40).to_string(index=False))

    print("\nsaved to:", out)

if __name__ == "__main__":
    main()
