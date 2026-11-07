import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def require_cols(df, cols, path):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise RuntimeError(f"{path} missing columns: {miss}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dps", required=True)
    ap.add_argument("--nodps", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    dps_p = Path(args.dps)
    nodps_p = Path(args.nodps)

    dps = pd.read_csv(dps_p)
    nodps = pd.read_csv(nodps_p)

    need = ["case", "sample", "fdtd_mse", "fdtd_mae", "fdtd_corr", "h5_index"]
    require_cols(dps, need, dps_p)
    require_cols(nodps, need, nodps_p)

    dps = dps[need].copy()
    nodps = nodps[need].copy()

    cmp = dps.merge(nodps, on="case", suffixes=("_dps", "_nodps"))

    if len(cmp) == 0:
        raise RuntimeError("No matched cases between DPS and no-DPS outputs.")

    h5_mismatch = cmp[cmp["h5_index_dps"] != cmp["h5_index_nodps"]]
    if len(h5_mismatch) > 0:
        h5_mismatch.to_csv(out / "h5_mismatch_cases.csv", index=False)
        raise RuntimeError(
            f"h5_index mismatch in {len(h5_mismatch)} cases. "
            f"Comparison is invalid. See {out / 'h5_mismatch_cases.csv'}"
        )

    cmp["mse_gain"] = cmp["fdtd_mse_nodps"] - cmp["fdtd_mse_dps"]
    cmp["mae_gain"] = cmp["fdtd_mae_nodps"] - cmp["fdtd_mae_dps"]
    cmp["corr_gain"] = cmp["fdtd_corr_dps"] - cmp["fdtd_corr_nodps"]

    cmp["mse_gain_pct"] = 100.0 * cmp["mse_gain"] / cmp["fdtd_mse_nodps"].clip(lower=1e-12)
    cmp["mae_gain_pct"] = 100.0 * cmp["mae_gain"] / cmp["fdtd_mae_nodps"].clip(lower=1e-12)

    cmp["dps_mse_better"] = cmp["mse_gain"] > 0
    cmp["dps_mae_better"] = cmp["mae_gain"] > 0
    cmp["dps_corr_better"] = cmp["corr_gain"] > 0

    cmp.to_csv(out / "comparison_per_case.csv", index=False)

    method_rows = []
    for name, df in [("DPS", dps), ("No-DPS", nodps)]:
        method_rows.append({
            "method": name,
            "N": len(df),
            "mean_mse": df["fdtd_mse"].mean(),
            "median_mse": df["fdtd_mse"].median(),
            "std_mse": df["fdtd_mse"].std(ddof=1),
            "mean_mae": df["fdtd_mae"].mean(),
            "median_mae": df["fdtd_mae"].median(),
            "std_mae": df["fdtd_mae"].std(ddof=1),
            "mean_corr": df["fdtd_corr"].mean(),
            "median_corr": df["fdtd_corr"].median(),
            "std_corr": df["fdtd_corr"].std(ddof=1),
            "bad_corr_lt_0p8": int((df["fdtd_corr"] < 0.8).sum()),
            "bad_mse_gt_0p02": int((df["fdtd_mse"] > 0.02).sum()),
        })

    method_summary = pd.DataFrame(method_rows)
    method_summary.to_csv(out / "method_summary.csv", index=False)

    gain_summary = pd.DataFrame([{
        "N": len(cmp),
        "mse_improved_cases": int(cmp["dps_mse_better"].sum()),
        "mae_improved_cases": int(cmp["dps_mae_better"].sum()),
        "corr_improved_cases": int(cmp["dps_corr_better"].sum()),
        "mean_mse_gain": cmp["mse_gain"].mean(),
        "median_mse_gain": cmp["mse_gain"].median(),
        "mean_mse_gain_pct": cmp["mse_gain_pct"].mean(),
        "median_mse_gain_pct": cmp["mse_gain_pct"].median(),
        "mean_mae_gain": cmp["mae_gain"].mean(),
        "median_mae_gain": cmp["mae_gain"].median(),
        "mean_mae_gain_pct": cmp["mae_gain_pct"].mean(),
        "median_mae_gain_pct": cmp["mae_gain_pct"].median(),
        "mean_corr_gain": cmp["corr_gain"].mean(),
        "median_corr_gain": cmp["corr_gain"].median(),
        "worst_dps_case_by_mse": int(cmp.loc[cmp["fdtd_mse_dps"].idxmax(), "case"]),
        "worst_dps_mse": float(cmp["fdtd_mse_dps"].max()),
        "worst_dps_corr": float(cmp.loc[cmp["fdtd_mse_dps"].idxmax(), "fdtd_corr_dps"]),
        "worst_gain_case_by_mse": int(cmp.loc[cmp["mse_gain"].idxmin(), "case"]),
        "worst_mse_gain": float(cmp["mse_gain"].min()),
        "worst_corr_gain": float(cmp.loc[cmp["mse_gain"].idxmin(), "corr_gain"]),
    }])
    gain_summary.to_csv(out / "gain_summary.csv", index=False)

    bad = cmp[
        (cmp["fdtd_corr_dps"] < 0.8) |
        (cmp["fdtd_mse_dps"] > 0.02) |
        (cmp["mse_gain"] < 0) |
        (cmp["corr_gain"] < 0)
    ].copy()
    bad.to_csv(out / "regressed_cases.csv", index=False)

    print("\n=== Method summary ===")
    print(method_summary.to_string(index=False))

    print("\n=== DPS gain summary ===")
    print(gain_summary.to_string(index=False))

    print("\n=== Worst DPS cases by MSE ===")
    worst = cmp.sort_values("fdtd_mse_dps", ascending=False).head(12)
    print(worst[[
        "case", "h5_index_dps",
        "fdtd_mse_dps", "fdtd_corr_dps",
        "fdtd_mse_nodps", "fdtd_corr_nodps",
        "mse_gain_pct", "corr_gain"
    ]].to_string(index=False))

    print("\n=== Bad or regressed cases ===")
    if len(bad) == 0:
        print("None")
    else:
        print(bad[[
            "case", "h5_index_dps",
            "fdtd_mse_dps", "fdtd_corr_dps",
            "fdtd_mse_nodps", "fdtd_corr_nodps",
            "mse_gain", "mse_gain_pct", "corr_gain"
        ]].to_string(index=False))

    print("\nsaved to:", out)

if __name__ == "__main__":
    main()
