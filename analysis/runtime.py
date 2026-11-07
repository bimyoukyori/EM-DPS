import argparse
from pathlib import Path
import pandas as pd
import os
import glob

def summarize_run(name, run_dir):
    run_dir = Path(run_dir)
    sample_dir = run_dir / "samples"
    files = sorted(sample_dir.glob("case*.npz"))

    if len(files) == 0:
        return {
            "method": name,
            "run_dir": str(run_dir),
            "N_cases": 0,
            "approx_total_seconds": None,
            "approx_seconds_per_case": None,
            "output_size_MB": None,
        }

    mtimes = [p.stat().st_mtime for p in files]
    total = max(mtimes) - min(mtimes) if len(mtimes) > 1 else None

    size = 0
    for root, dirs, fs in os.walk(run_dir):
        for f in fs:
            p = Path(root) / f
            size += p.stat().st_size

    return {
        "method": name,
        "run_dir": str(run_dir),
        "N_cases": len(files),
        "approx_total_seconds": total,
        "approx_seconds_per_case": total / len(files) if total else None,
        "output_size_MB": size / 1024 / 1024,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/runtime_summary.csv")
    args = ap.parse_args()

    runs = [
        ("NoDPS_K4", "runs/candidate_count/nodps_k4_64cases"),
        ("DPS_K4_lr0p02", "runs/guidance_strength/dps_lr0p02_k4_64cases"),
        ("DPS_K1_lr0p02", "runs/candidate_count/dps_k1_64cases"),
        ("DPS_K2_lr0p02", "runs/candidate_count/dps_k2_64cases"),
        ("DPS_K8_lr0p02", "runs/candidate_count/dps_k8_64cases"),
    ]

    rows = [summarize_run(name, path) for name, path in runs]
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(df.to_string(index=False))
    print("\nsaved:", args.out)

if __name__ == "__main__":
    main()
