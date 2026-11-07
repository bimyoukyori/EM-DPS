import argparse
import csv
import os
import copy
import sys
import gc
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

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

def sigma_schedule(sigma_min, sigma_max, steps, rho, device):
    ramp = torch.linspace(0, 1, steps, device=device)
    sigmas = (
        sigma_max ** (1 / rho)
        + ramp * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    sigmas = torch.cat([sigmas, torch.zeros_like(sigmas[:1])])
    return sigmas

def denorm_x_t(x, stats):
    mean = torch.as_tensor(stats.x_mean, device=x.device, dtype=x.dtype)[None, :, None, None]
    std = torch.as_tensor(stats.x_std, device=x.device, dtype=x.dtype)[None, :, None, None]
    return x * std + mean

def clamp_norm_from_eps(x, stats, eps_min, eps_max):
    mean = torch.as_tensor(stats.x_mean, device=x.device, dtype=x.dtype)[None, :, None, None]
    std = torch.as_tensor(stats.x_std, device=x.device, dtype=x.dtype)[None, :, None, None]
    lo = (eps_min - mean) / std
    hi = (eps_max - mean) / std
    return torch.max(torch.min(x, hi), lo)

def norm_corr_t(a, b):
    a = a - a.mean()
    b = b - b.mean()
    den = torch.sqrt((a * a).sum() * (b * b).sum()).clamp_min(1e-12)
    return (a * b).sum() / den

def sample_edm_dps(
    model,
    y,
    obs_raw_t,
    obs_scale,
    op,
    sigma_fixed_t,
    stats,
    cfg,
    args,
    device,
):

    import gc

    K = int(args.K)
    H, W = cfg["model"]["x_resolution"]
    C = int(cfg["model"].get("x_ch", 1))
    assert C == 1, "This DPS script is eps-only and expects x_ch=1."

    sigmas = sigma_schedule(
        args.sigma_min,
        args.sigma_max,
        args.steps,
        args.rho,
        device,
    )

    x = torch.randn(K, C, H, W, device=device) * sigmas[0]
    yK = y.repeat(K, 1, 1, 1).to(device)

    obs_proc_t = torch_prep_bscan(obs_raw_t, cfg, obs_scale=obs_scale).detach()

    x_std = float(np.asarray(stats.x_std).reshape(-1)[0])
    dps_logs = []

    for i in range(len(sigmas) - 1):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        sigma_batch = torch.full([K], float(sigma.item()), device=device)

        with torch.no_grad():
            den = model(x, sigma_batch, yK)
            d = (x - den) / sigma.clamp_min(1e-12)
            x_next = x + (sigma_next - sigma) * d
            x_next = clamp_norm_from_eps(x_next, stats, args.eps_min, args.eps_max)

        do_dps = (
            i >= int(args.dps_start_frac * args.steps)
            and args.dps_every > 0
            and (i % args.dps_every == 0)
            and float(sigma_next.item()) > 0.0
        )

        if do_dps:
            step_losses = []

            for k in range(K):
                
                with torch.no_grad():
                    sig_g = torch.full(
                        [1],
                        float(max(sigma_next.item(), args.sigma_min)),
                        device=device,
                    )
                    den_g = model(x_next[k:k+1], sig_g, yK[k:k+1])
                    den_g = clamp_norm_from_eps(den_g, stats, args.eps_min, args.eps_max)

                eps_g = denorm_x_t(den_g, stats)[0, 0].detach().clone()
                eps_g = eps_g.clamp(args.eps_min, args.eps_max)
                eps_g.requires_grad_(True)

                sim_raw_t = call_fdtd_torch(op, eps_g, sigma_fixed_t, device)
                sim_proc_t = torch_prep_bscan(sim_raw_t, cfg, obs_scale=obs_scale)

                mse = ((sim_proc_t - obs_proc_t) ** 2).mean()
                ncc = norm_corr_t(sim_proc_t, obs_proc_t)

                phys_loss = args.dps_w_mse * mse + args.dps_w_ncc * (1.0 - ncc)
                reg_tv = tv_loss(eps_g)
                loss = phys_loss + args.dps_w_tv * reg_tv

                loss.backward()

                grad_eps = eps_g.grad.detach()
                grad_eps = torch.nan_to_num(grad_eps, nan=0.0, posinf=0.0, neginf=0.0)

                grad_x = grad_eps[None, None] * x_std

                scale = grad_x.abs().mean().clamp_min(1e-8)
                grad_x = grad_x / scale

                noise_factor = float((sigma_next / sigmas[0]).clamp(0.05, 1.0).item())
                eta = args.dps_lr * (noise_factor ** args.dps_power)

                upd = (eta * grad_x).clamp(-args.dps_max_update, args.dps_max_update)

                with torch.no_grad():
                    x_new = x_next[k:k+1] - upd
                    x_new = clamp_norm_from_eps(x_new, stats, args.eps_min, args.eps_max)
                    x_next[k:k+1] = x_new

                step_losses.append({
                    "sample": k,
                    "step": i,
                    "sigma": float(sigma.item()),
                    "sigma_next": float(sigma_next.item()),
                    "eta": float(eta),
                    "mse": float(mse.detach().cpu()),
                    "ncc": float(ncc.detach().cpu()),
                    "loss": float(loss.detach().cpu()),
                })

                del eps_g, sim_raw_t, sim_proc_t, loss, mse, ncc, phys_loss, reg_tv, grad_eps, grad_x, upd
                torch.cuda.empty_cache()
                gc.collect()

            dps_logs.extend(step_losses)

            if args.print_every > 0 and (i % args.print_every == 0):
                m = np.mean([r["mse"] for r in step_losses])
                c = np.mean([r["ncc"] for r in step_losses])
                print(
                    f"[DPS-safe] step {i:03d}/{args.steps} sigma={float(sigma):.4g} "
                    f"mean_mse={m:.5g} mean_corr={c:.4f}",
                    flush=True,
                )

        x = x_next.detach()
        torch.cuda.empty_cache()

    with torch.no_grad():
        sig_final = torch.full([K], args.sigma_min, device=device)
        final = model(x, sig_final, yK)
        final = clamp_norm_from_eps(final, stats, args.eps_min, args.eps_max)
        eps = denorm_x_t(final, stats)[:, 0].detach().cpu().numpy().astype("float32")

    torch.cuda.empty_cache()
    gc.collect()

    return eps, dps_logs

def eval_candidates_official(op, cfg, eps_arr, sigma_np, obs_proc_np, device):
    rows = []
    sims = []

    for k in range(eps_arr.shape[0]):
        eps_k = eps_arr[k].astype("float32")
        _, sim_proc = official_forward_np(op, cfg, eps_k, sigma_np, device)

        if sim_proc.shape != obs_proc_np.shape:
            sim_proc = resize_np(sim_proc, obs_proc_np.shape)

        res = sim_proc - obs_proc_np
        mse = float(np.mean(res ** 2))
        mae = float(np.mean(np.abs(res)))
        corr = corr_np(sim_proc, obs_proc_np)

        rows.append({
            "sample": k,
            "fdtd_mse": mse,
            "fdtd_mae": mae,
            "fdtd_corr": corr,
        })
        sims.append(sim_proc)

    return rows, sims

def plot_case(out_png, eps_true, eps_arr, obs_proc, sims, rows, title):
    K = eps_arr.shape[0]
    showK = min(K, 4)

    fig, axes = plt.subplots(3, showK + 1, figsize=(4 * (showK + 1), 9), constrained_layout=True)

    evmin = min(float(eps_true.min()), float(eps_arr[:showK].min()))
    evmax = max(float(eps_true.max()), float(eps_arr[:showK].max()))

    im = axes[0, 0].imshow(eps_true, cmap="turbo", aspect="auto", vmin=evmin, vmax=evmax)
    axes[0, 0].set_title("true eps")
    axes[0, 0].axis("off")
    plt.colorbar(im, ax=axes[0, 0], fraction=0.046)

    for k in range(showK):
        im = axes[0, k + 1].imshow(eps_arr[k], cmap="turbo", aspect="auto", vmin=evmin, vmax=evmax)
        axes[0, k + 1].set_title(f"DPS sample {k}")
        axes[0, k + 1].axis("off")
        plt.colorbar(im, ax=axes[0, k + 1], fraction=0.046)

    q = float(np.percentile(np.abs(obs_proc), 99))
    q = max(q, 1e-6)

    im = axes[1, 0].imshow(obs_proc, cmap="seismic", aspect="auto", vmin=-q, vmax=q)
    axes[1, 0].set_title("observed")
    axes[1, 0].axis("off")
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

    for k in range(showK):
        im = axes[1, k + 1].imshow(sims[k], cmap="seismic", aspect="auto", vmin=-q, vmax=q)
        axes[1, k + 1].set_title(f"sim {k}\ncorr={rows[k]['fdtd_corr']:.3f}")
        axes[1, k + 1].axis("off")
        plt.colorbar(im, ax=axes[1, k + 1], fraction=0.046)

    axes[2, 0].axis("off")
    axes[2, 0].set_title(title)

    for k in range(showK):
        res = sims[k] - obs_proc
        rq = float(np.percentile(np.abs(res), 99))
        rq = max(rq, 1e-6)
        im = axes[2, k + 1].imshow(res, cmap="seismic", aspect="auto", vmin=-rq, vmax=rq)
        axes[2, k + 1].set_title(f"res {k}\nMSE={rows[k]['fdtd_mse']:.4g}")
        axes[2, k + 1].axis("off")
        plt.colorbar(im, ax=axes[2, k + 1], fraction=0.046)

    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--cases", type=int, default=4)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", default="cuda")

    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--sigma_min", type=float, default=0.002)
    ap.add_argument("--sigma_max", type=float, default=5.0)
    ap.add_argument("--rho", type=float, default=7.0)

    ap.add_argument("--dps_start_frac", type=float, default=0.45)
    ap.add_argument("--dps_every", type=int, default=4)
    ap.add_argument("--dps_lr", type=float, default=0.02)
    ap.add_argument("--dps_power", type=float, default=0.5)
    ap.add_argument("--dps_max_update", type=float, default=0.05)
    ap.add_argument("--dps_w_mse", type=float, default=1.0)
    ap.add_argument("--dps_w_ncc", type=float, default=0.05)
    ap.add_argument("--dps_w_tv", type=float, default=5e-5)
    ap.add_argument("--dps_w_prior", type=float, default=1e-4)

    ap.add_argument("--eps_min", type=float, default=1.0)
    ap.add_argument("--eps_max", type=float, default=35.0)

    ap.add_argument("--sigma_mode", choices=["h5_median", "constant", "oracle"], default="h5_median")
    ap.add_argument("--sigma_const", type=float, default=0.0)
    ap.add_argument("--sigma_stats_n", type=int, default=256)

    ap.add_argument("--guidance_freq_scale", type=float, default=1.0,
                    help="Scale the FDTD source frequency used in DPS guidance.")
    ap.add_argument("--eval_freq_scale", type=float, default=1.0,
                    help="Scale the FDTD source frequency used in final evaluation.")
    ap.add_argument("--print_every", type=int, default=4)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = emdps.load_yaml(args.cfg)
    out = Path(args.out)
    sample_dir = out / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("device:", device)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    stats = emdps.Stats(**ckpt["stats"])

    model = emdps.make_model(cfg, device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    ds = emdps.H5Set(cfg, args.split, stats)
    print("dataset split:", args.split, "N=", len(ds))

    cfg_guidance = copy.deepcopy(cfg)
    cfg_eval = copy.deepcopy(cfg)

    if "fdtd" in cfg_guidance and "freq" in cfg_guidance["fdtd"]:
        cfg_guidance["fdtd"]["freq"] = float(cfg_guidance["fdtd"]["freq"]) * float(args.guidance_freq_scale)
    if "fdtd" in cfg_eval and "freq" in cfg_eval["fdtd"]:
        cfg_eval["fdtd"]["freq"] = float(cfg_eval["fdtd"]["freq"]) * float(args.eval_freq_scale)

    print("guidance_freq_scale:", args.guidance_freq_scale)
    print("eval_freq_scale:", args.eval_freq_scale)
    print("guidance freq:", cfg_guidance.get("fdtd", {}).get("freq", None))
    print("eval freq:", cfg_eval.get("fdtd", {}).get("freq", None))

    op_guidance = emdps.make_fdtd(cfg_guidance, device)
    op_eval = emdps.make_fdtd(cfg_eval, device)

    if args.sigma_mode == "h5_median":
        sigma_const = estimate_train_sigma_median(cfg, n=args.sigma_stats_n)
        print("sigma_const from train median:", sigma_const)
    elif args.sigma_mode == "constant":
        sigma_const = float(args.sigma_const)
        print("sigma_const:", sigma_const)
    else:
        sigma_const = None
        print("oracle sigma: diagnostic only")

    all_rows = []
    all_dps_logs = []

    with h5py.File(cfg["data"]["h5"], "r") as f:
        for ci in range(min(args.cases, len(ds))):
            b = ds[ci]
            y = b["y"][None].to(device)
            x_true = b["x"][None].to(device)
            h5_idx = int(b["index"])

            print(f"\n=== DPS case {ci:03d} | h5_idx={h5_idx} ===", flush=True)

            eps_true = denorm_x_t(x_true, stats)[0, 0].detach().cpu().numpy().astype("float32")

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
                sigma_np = np.full(tuple(cfg["model"]["x_resolution"]), sigma_const, dtype="float32")

            sigma_t = torch.from_numpy(sigma_np).to(device)

            eps_arr, dps_logs = sample_edm_dps(
                model=model,
                y=y,
                obs_raw_t=obs_raw_t,
                obs_scale=obs_scale,
                op=op_guidance,
                sigma_fixed_t=sigma_t,
                stats=stats,
                cfg=cfg,
                args=args,
                device=device,
            )

            rows, sims = eval_candidates_official(op_eval, cfg_eval, eps_arr, sigma_np, obs_proc_np, device)

            for r in rows:
                r.update({
                    "case": ci,
                    "h5_index": h5_idx,
                    "K": args.K,
                    "steps": args.steps,
                    "dps_lr": args.dps_lr,
                    "dps_every": args.dps_every,
                    "dps_start_frac": args.dps_start_frac,
                    "sigma_mode": args.sigma_mode,
                })
                all_rows.append(r)

            for r in dps_logs:
                r.update({"case": ci, "h5_index": h5_idx})
                all_dps_logs.append(r)

            np.savez_compressed(
                sample_dir / f"case{ci:03d}.npz",
                eps=eps_arr,
                eps_true=eps_true,
                index=h5_idx,
                sigma=sigma_np,
            )

            best = min(rows, key=lambda z: z["fdtd_mse"])
            plot_case(
                sample_dir / f"case{ci:03d}_dps.png",
                eps_true,
                eps_arr,
                obs_proc_np,
                sims,
                rows,
                title=f"case={ci}, best sample={best['sample']}, mse={best['fdtd_mse']:.4g}, corr={best['fdtd_corr']:.3f}",
            )

            print(
                f"[case {ci:03d}] best sample={best['sample']} "
                f"mse={best['fdtd_mse']:.6g} mae={best['fdtd_mae']:.6g} corr={best['fdtd_corr']:.4f}",
                flush=True,
            )

    rank_csv = out / "dps_fdtd_ranking.csv"
    with open(rank_csv, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    log_csv = out / "dps_guidance_log.csv"
    if all_dps_logs:
        with open(log_csv, "w", newline="") as fp:
            w = csv.DictWriter(fp, fieldnames=list(all_dps_logs[0].keys()))
            w.writeheader()
            w.writerows(all_dps_logs)

    print("\nsaved:", rank_csv)
    print("saved:", log_csv)

    import pandas as pd
    df = pd.DataFrame(all_rows)
    best_df = df.sort_values(["case", "fdtd_mse"]).groupby("case", as_index=False).first()
    best_df.to_csv(out / "dps_best_by_case.csv", index=False)

    print("\n=== DPS best-by-case ===")
    print(best_df[["case", "sample", "fdtd_mse", "fdtd_mae", "fdtd_corr"]].to_string(index=False))

    print("\n=== Aggregate ===")
    print("N cases:", len(best_df))
    print("mean best MSE:", float(best_df["fdtd_mse"].mean()))
    print("mean best MAE:", float(best_df["fdtd_mae"].mean()))
    print("mean best corr:", float(best_df["fdtd_corr"].mean()))

if __name__ == "__main__":
    main()
