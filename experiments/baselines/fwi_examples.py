import argparse
import csv
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from methods import emdps
from methods.physics import (
    torch_prep_bscan,
    call_fdtd_torch,
    estimate_train_sigma_median,
    resize_np,
    corr_np,
    official_forward_np,
    tv_loss,
)

def estimate_eps_median(cfg, n=512):
    h5 = cfg["data"]["h5"]
    vals = []
    with h5py.File(h5, "r") as f:
        valid = np.where(f["valid"][:] == 1)[0]
        ids = valid[:min(n, len(valid))]
        for i in ids:
            vals.append(np.median(f["eps"][i].astype("float32")))
    return float(np.median(vals))

def inv_sigmoid(x):
    x = np.clip(x, 1e-5, 1 - 1e-5)
    return np.log(x / (1 - x))

def eps_from_z(z, eps_min, eps_max):
    return eps_min + (eps_max - eps_min) * torch.sigmoid(z)

def norm_corr_t(a, b):
    a = a - a.mean()
    b = b - b.mean()
    den = torch.sqrt((a * a).sum() * (b * b).sum()).clamp_min(1e-12)
    return (a * b).sum() / den

def parse_case_ids(case_ids, cases, n_ds):
    if case_ids.strip():
        out = []
        for x in case_ids.replace(",", " ").split():
            out.append(int(x))
        return out
    return list(range(min(cases, n_ds)))

def official_forward_processed(op, cfg, eps_np, sigma_np, obs_shape, device):
    sim_raw, sim_proc = official_forward_np(op, cfg, eps_np, sigma_np, device)
    if sim_proc.shape != obs_shape:
        sim_proc = resize_np(sim_proc, obs_shape)
    return sim_raw.astype("float32"), sim_proc.astype("float32")

def official_eval_from_sim(sim_proc, obs_proc_np):
    res = sim_proc - obs_proc_np
    return {
        "fdtd_mse": float(np.mean(res ** 2)),
        "fdtd_mae": float(np.mean(np.abs(res))),
        "fdtd_corr": float(corr_np(sim_proc, obs_proc_np)),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--cases", type=int, default=64)
    ap.add_argument("--case_ids", default="", help="Optional comma/space separated dataset case ids, e.g. 14,126,49")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--eps_min", type=float, default=1.0)
    ap.add_argument("--eps_max", type=float, default=35.0)
    ap.add_argument("--eps_init", default="train_median", choices=["train_median", "constant"])
    ap.add_argument("--eps_const", type=float, default=5.5)
    ap.add_argument("--sigma_mode", default="h5_median", choices=["h5_median", "constant", "oracle"])
    ap.add_argument("--sigma_const", type=float, default=0.039093017578125)
    ap.add_argument("--sigma_stats_n", type=int, default=256)
    ap.add_argument("--w_mse", type=float, default=1.0)
    ap.add_argument("--w_ncc", type=float, default=0.05)
    ap.add_argument("--w_tv", type=float, default=1e-4)
    ap.add_argument("--save_samples", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = emdps.load_yaml(args.cfg)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sample_out = out / "samples"
    if args.save_samples:
        sample_out.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("device:", device)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    stats = emdps.Stats(**ckpt["stats"])
    ds = emdps.H5Set(cfg, args.split, stats)
    op = emdps.make_fdtd(cfg, device)

    case_list = parse_case_ids(args.case_ids, args.cases, len(ds))
    print("case_list:", case_list)

    if args.eps_init == "train_median":
        eps0_scalar = estimate_eps_median(cfg, n=512)
    else:
        eps0_scalar = float(args.eps_const)

    if args.sigma_mode == "h5_median":
        sigma_const = estimate_train_sigma_median(cfg, n=args.sigma_stats_n)
    elif args.sigma_mode == "constant":
        sigma_const = float(args.sigma_const)
    else:
        sigma_const = None

    print("eps0_scalar:", eps0_scalar)
    print("sigma_const:", sigma_const)

    H, W = cfg["model"]["x_resolution"]
    rows = []

    with h5py.File(cfg["data"]["h5"], "r") as f:
        for ci in case_list:
            if ci < 0 or ci >= len(ds):
                print(f"[skip] invalid case id {ci}; len(ds)={len(ds)}")
                continue

            b = ds[ci]
            h5_idx = int(b["index"])
            print(f"\n=== smooth FWI case {ci:03d} | h5_idx={h5_idx} ===", flush=True)

            obs_raw_np = f["bscan"][h5_idx].astype("float32")
            obs_raw_t = torch.from_numpy(obs_raw_np).to(device)

            obs_tmp = torch_prep_bscan(obs_raw_t, cfg, obs_scale=None).detach()
            obs_scale = torch.quantile(obs_tmp.abs().flatten(), 0.99).clamp_min(1e-6)
            obs_proc_t = torch_prep_bscan(obs_raw_t, cfg, obs_scale=obs_scale).detach()
            obs_proc_np = emdps.prep_bscan_np(obs_raw_np, cfg).astype("float32")

            if args.sigma_mode == "oracle":
                sigma_np = f["sigma"][h5_idx].astype("float32")
                sigma_np = resize_np(sigma_np, cfg["model"]["x_resolution"])
            else:
                sigma_np = np.full((H, W), sigma_const, dtype="float32")

            sigma_t = torch.from_numpy(sigma_np).to(device)

            eps_true_np = f["eps"][h5_idx].astype("float32")
            if eps_true_np.shape != (H, W):
                eps_true_np = resize_np(eps_true_np, (H, W)).astype("float32")

            eps0_np = np.full((H, W), eps0_scalar, dtype="float32")
            sim_init_raw, sim_init_proc = official_forward_processed(
                op, cfg, eps0_np, sigma_np, obs_proc_np.shape, device
            )
            init_eval = official_eval_from_sim(sim_init_proc, obs_proc_np)

            p0 = (eps0_scalar - args.eps_min) / (args.eps_max - args.eps_min)
            z0 = inv_sigmoid(p0)
            z = torch.full((H, W), float(z0), device=device, requires_grad=True)

            opt = torch.optim.Adam([z], lr=args.lr)

            for it in range(args.steps):
                opt.zero_grad(set_to_none=True)
                eps_t = eps_from_z(z, args.eps_min, args.eps_max)

                sim_raw_t = call_fdtd_torch(op, eps_t, sigma_t, device)
                sim_proc_t = torch_prep_bscan(sim_raw_t, cfg, obs_scale=obs_scale)

                mse = ((sim_proc_t - obs_proc_t) ** 2).mean()
                ncc = norm_corr_t(sim_proc_t, obs_proc_t)
                loss = args.w_mse * mse + args.w_ncc * (1.0 - ncc) + args.w_tv * tv_loss(eps_t)

                loss.backward()
                torch.nn.utils.clip_grad_norm_([z], 1.0)
                opt.step()

                if it % 10 == 0 or it == args.steps - 1:
                    print(
                        f"iter {it:04d}: loss={float(loss):.6g} "
                        f"mse={float(mse):.6g} corr={float(ncc):.4f}",
                        flush=True,
                    )

            eps_final = eps_from_z(z, args.eps_min, args.eps_max).detach().cpu().numpy().astype("float32")
            sim_final_raw, sim_final_proc = official_forward_processed(
                op, cfg, eps_final, sigma_np, obs_proc_np.shape, device
            )
            final_eval = official_eval_from_sim(sim_final_proc, obs_proc_np)

            row = {
                "case": ci,
                "sample": -1,
                "h5_index": h5_idx,
                "K": 1,
                "steps": args.steps,
                "lr": args.lr,
                "eps0_scalar": eps0_scalar,
                "sigma_mode": args.sigma_mode,
                "init_mse": init_eval["fdtd_mse"],
                "init_mae": init_eval["fdtd_mae"],
                "init_corr": init_eval["fdtd_corr"],
                "fdtd_mse": final_eval["fdtd_mse"],
                "fdtd_mae": final_eval["fdtd_mae"],
                "fdtd_corr": final_eval["fdtd_corr"],
                "mse_gain": init_eval["fdtd_mse"] - final_eval["fdtd_mse"],
                "corr_gain": final_eval["fdtd_corr"] - init_eval["fdtd_corr"],
            }
            rows.append(row)

            if args.save_samples:
                np.savez_compressed(
                    sample_out / f"case{ci:03d}.npz",
                    eps=eps_final[None, ...].astype("float32"),
                    eps_final=eps_final.astype("float32"),
                    pred_eps=eps_final.astype("float32"),
                    eps_init=eps0_np.astype("float32"),
                    eps_true=eps_true_np.astype("float32"),
                    sigma=sigma_np.astype("float32"),
                    index=np.array(h5_idx, dtype=np.int64),
                    h5_index=np.array(h5_idx, dtype=np.int64),
                    case=np.array(ci, dtype=np.int64),
                    obs=obs_proc_np.astype("float32"),
                    obs_proc=obs_proc_np.astype("float32"),
                    observed_processed_bscan=obs_proc_np.astype("float32"),
                    sim=sim_final_proc[None, ...].astype("float32"),
                    sim_stack=sim_final_proc[None, ...].astype("float32"),
                    sim_proc=sim_final_proc.astype("float32"),
                    sim_init_proc=sim_init_proc.astype("float32"),
                    sim_final_proc=sim_final_proc.astype("float32"),
                    residual=(sim_final_proc - obs_proc_np).astype("float32"),
                    residual_init=(sim_init_proc - obs_proc_np).astype("float32"),
                    obs_raw=obs_raw_np.astype("float32"),
                )

            print(
                f"official: mse {row['init_mse']:.6g} -> {row['fdtd_mse']:.6g}; "
                f"corr {row['init_corr']:.4f} -> {row['fdtd_corr']:.4f}",
                flush=True,
            )

    if not rows:
        raise RuntimeError("No cases were processed.")

    df_path = out / "dps_best_by_case.csv"
    with open(df_path, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(out / "smooth_fwi_summary.csv", index=False)

    print("\n=== Smooth FWI aggregate ===")
    print("N:", len(df))
    print("mean init_mse:", df["init_mse"].mean())
    print("mean final_mse:", df["fdtd_mse"].mean())
    print("mean init_corr:", df["init_corr"].mean())
    print("mean final_corr:", df["fdtd_corr"].mean())
    print("saved:", df_path)
    if args.save_samples:
        print("saved samples:", sample_out)

if __name__ == "__main__":
    main()
