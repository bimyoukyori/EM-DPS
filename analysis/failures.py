import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import shutil

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dps", default="runs/guidance_strength/dps_lr0p02_k4_64cases/dps_best_by_case.csv")
    ap.add_argument("--nodps", default="runs/candidate_count/nodps_k4_64cases/dps_best_by_case.csv")
    ap.add_argument("--out", default="runs/failures")
    ap.add_argument("--bad_corr", type=float, default=0.8)
    ap.add_argument("--bad_mse", type=float, default=0.02)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    dps = pd.read_csv(args.dps)
    nodps = pd.read_csv(args.nodps)

    cmp = dps.merge(nodps, on="case", suffixes=("_dps", "_nodps"))

    cmp["mse_gain"] = cmp["fdtd_mse_nodps"] - cmp["fdtd_mse_dps"]
    cmp["mae_gain"] = cmp["fdtd_mae_nodps"] - cmp["fdtd_mae_dps"]
    cmp["corr_gain"] = cmp["fdtd_corr_dps"] - cmp["fdtd_corr_nodps"]
    cmp["mse_gain_pct"] = 100 * cmp["mse_gain"] / cmp["fdtd_mse_nodps"].clip(lower=1e-12)

    cmp["bad_by_corr"] = cmp["fdtd_corr_dps"] < args.bad_corr
    cmp["bad_by_mse"] = cmp["fdtd_mse_dps"] > args.bad_mse
    cmp["bad"] = cmp["bad_by_corr"] | cmp["bad_by_mse"]
    cmp["regressed_mse"] = cmp["mse_gain"] < 0
    cmp["regressed_corr"] = cmp["corr_gain"] < 0

    cmp.to_csv(out / "failure_analysis_all_cases.csv", index=False)

    bad = cmp[cmp["bad"]].sort_values(["fdtd_corr_dps", "fdtd_mse_dps"], ascending=[True, False])
    bad.to_csv(out / "bad_cases.csv", index=False)

    worst_mse = cmp.sort_values("fdtd_mse_dps", ascending=False).head(15)
    worst_corr = cmp.sort_values("fdtd_corr_dps", ascending=True).head(15)
    least_gain = cmp.sort_values("mse_gain", ascending=True).head(15)

    worst_mse.to_csv(out / "worst_by_mse.csv", index=False)
    worst_corr.to_csv(out / "worst_by_corr.csv", index=False)
    least_gain.to_csv(out / "least_improved_by_mse_gain.csv", index=False)

    recurrent = pd.concat([
        worst_mse[["case"]],
        worst_corr[["case"]],
        least_gain[["case"]],
        bad[["case"]],
    ])
    recurrent = recurrent["case"].value_counts().reset_index()
    recurrent.columns = ["case", "failure_count"]
    recurrent = recurrent.sort_values(["failure_count", "case"], ascending=[False, True])
    recurrent.to_csv(out / "recurrent_failure_cases.csv", index=False)

    print("\n=== Bad cases ===")
    print(bad[[
        "case", "h5_index_dps", "fdtd_mse_dps", "fdtd_corr_dps",
        "fdtd_mse_nodps", "fdtd_corr_nodps", "mse_gain_pct", "corr_gain"
    ]].to_string(index=False))

    print("\n=== Recurrent failure cases ===")
    print(recurrent.to_string(index=False))

    print("\n=== Least improved cases ===")
    print(least_gain[[
        "case", "h5_index_dps", "fdtd_mse_dps", "fdtd_corr_dps",
        "mse_gain", "mse_gain_pct", "corr_gain"
    ]].to_string(index=False))

    print("\nsaved to:", out)

if __name__ == "__main__":
    main()
