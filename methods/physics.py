import argparse
import csv
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.insert(0, os.getcwd())
from methods import emdps
def resize_np(a, shape):
    if tuple(a.shape) == tuple(shape):
        return a.astype("float32")
    t = torch.from_numpy(a.astype("float32"))[None, None]
    t = F.interpolate(t, size=tuple(shape), mode="bilinear", align_corners=False)
    return t[0, 0].numpy().astype("float32")

def corr_np(a, b):
    a = a.astype("float32") - float(a.mean())
    b = b.astype("float32") - float(b.mean())
    den = float(np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12)
    return float((a * b).sum() / den)

def tv_loss(x):
    return (x[:, 1:] - x[:, :-1]).abs().mean() + (x[1:, :] - x[:-1, :]).abs().mean()

def torch_prep_bscan(raw, cfg, obs_scale=None):

    x = raw

    while x.ndim > 2:
        x = x.squeeze(0)

    if x.ndim != 2:
        raise RuntimeError(f"Expected 2D B-scan after squeeze, got {tuple(x.shape)}")

    prep = cfg.get("prep", {})

    if bool(prep.get("remove_mean_trace", True)):
        x = x - x.mean(dim=1, keepdim=True)

    if bool(prep.get("dewow", True)):
        win = int(prep.get("dewow_win", 121))
        if win > 3:
            if win % 2 == 0:
                win += 1
            xt = x.T[None]  
            ma = F.avg_pool1d(xt, kernel_size=win, stride=1, padding=win // 2)
            x = (xt - ma)[0].T

    mute_top = int(prep.get("mute_top", 70))
    crop_len = int(prep.get("crop_len", 512))
    start = max(0, mute_top)
    end = start + crop_len

    if x.shape[0] < end:
        pad = end - x.shape[0]
        x = F.pad(x, (0, 0, 0, pad))

    x = x[start:end, :]

    yres = cfg.get("model", {}).get("y_resolution", [512, 384])
    H, W = int(yres[0]), int(yres[1])
    x = F.interpolate(x[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0]

    if obs_scale is None:
        obs_scale = torch.quantile(x.detach().abs().flatten(), 0.99).clamp_min(1e-6)

    x = x / obs_scale
    x = x.clamp(-3.0, 3.0)

    return x

def extract_tensor_output(out):
    if isinstance(out, dict):
        for k in ["bscan", "Ez", "field", "rx", "out", "data"]:
            if k in out:
                out = out[k]
                break
        else:
            out = next(iter(out.values()))

    if isinstance(out, (list, tuple)):
        out = out[0]

    if not torch.is_tensor(out):
        raise RuntimeError(
            "FDTD forward returned non-tensor output. "
            "Current interface is not differentiable."
        )

    return out

def call_fdtd_torch(op, eps, sigma, device):

    attempts = []

    if hasattr(emdps, "run_fdtd_torch"):
        attempts.append(lambda e, s: emdps.run_fdtd_torch(op, e, s, device))
        attempts.append(lambda e, s: emdps.run_fdtd_torch(op, e[None], s[None], device))
        attempts.append(lambda e, s: emdps.run_fdtd_torch(op, e[None, None], s[None, None], device))

    if callable(op):
        attempts.append(lambda e, s: op(e, s))
        attempts.append(lambda e, s: op(e[None], s[None]))
        attempts.append(lambda e, s: op(e[None, None], s[None, None]))
        attempts.append(lambda e, s: op({"eps": e, "sigma": s}))
        attempts.append(lambda e, s: op({"eps": e[None, None], "sigma": s[None, None]}))

    errors = []
    for fn in attempts:
        try:
            out = fn(eps, sigma)
            out = extract_tensor_output(out)
            if not out.requires_grad:
                raise RuntimeError("FDTD tensor output does not require grad")
            return out
        except Exception as ex:
            errors.append(repr(ex))

    msg = "\n".join(errors[-6:])
    raise RuntimeError(
        "Could not call a differentiable FDTD forward.\n"
        "Last errors:\n" + msg
    )

def official_forward_np(op, cfg, eps_np, sigma_np, device):
    sim_raw = emdps.run_fdtd(op, eps_np.astype("float32"), sigma_np.astype("float32"), device)
    sim_proc = emdps.prep_bscan_np(sim_raw.astype("float32"), cfg).astype("float32")
    return sim_raw, sim_proc

def estimate_train_sigma_median(cfg, n=256, seed=123):
    h5_path = cfg["data"]["h5"]
    split_seed = int(cfg["data"].get("split_seed", 42))
    train_frac = float(cfg["data"].get("train_frac", 0.8))
    val_frac = float(cfg["data"].get("val_frac", 0.1))

    with h5py.File(h5_path, "r") as f:
        valid = np.where(f["valid"][:] == 1)[0]
        sp = emdps.split_indices(len(valid), split_seed, train_frac, val_frac)
        train_ids = valid[sp["train"]]

        rng = np.random.default_rng(seed)
        use = rng.choice(train_ids, size=min(n, len(train_ids)), replace=False)

        vals = []
        for i in use:
            vals.append(float(np.median(f["sigma"][int(i)].astype("float32"))))

    return float(np.median(vals))

def load_ranked_cases(fdtd_csv, case_ids):
    df = pd.read_csv(fdtd_csv)
    df = df[np.isfinite(df["score"])].copy()
    best = (
        df.sort_values(["case", "score"], ascending=[True, True])
        .groupby("case", as_index=False)
        .first()
    )

    if case_ids is not None:
        best = best[best["case"].isin(case_ids)].copy()

    return best

def inverse_sigmoid_from_eps(eps, lo, hi):
    z = (eps - lo) / (hi - lo)
    z = np.clip(z, 1e-4, 1.0 - 1e-4)
    return np.log(z / (1.0 - z)).astype("float32")

def eps_from_param(p, lo, hi):
    return lo + (hi - lo) * torch.sigmoid(p)

def save_plot(out_png, eps_true, eps_init, eps_final, obs_proc, sim_init, sim_final, title):
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), constrained_layout=True)

    evmin = min(float(eps_true.min()), float(eps_init.min()), float(eps_final.min()))
    evmax = max(float(eps_true.max()), float(eps_init.max()), float(eps_final.max()))

    for ax, img, name in [
        (axes[0, 0], eps_true, "true eps"),
        (axes[0, 1], eps_init, "init eps"),
        (axes[0, 2], eps_final, "FWI-refined eps"),
    ]:
        im = ax.imshow(img, cmap="turbo", aspect="auto", vmin=evmin, vmax=evmax)
        ax.set_title(name)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)

    q = float(np.percentile(np.abs(obs_proc), 99))
    q = max(q, 1e-6)

    for ax, img, name in [
        (axes[1, 0], obs_proc, "observed processed B-scan"),
        (axes[1, 1], sim_init, "sim init processed B-scan"),
        (axes[1, 2], sim_final, "sim FWI processed B-scan"),
    ]:
        im = ax.imshow(img, cmap="seismic", aspect="auto", vmin=-q, vmax=q)
        ax.set_title(name)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)

    res_init = sim_init - obs_proc
    res_final = sim_final - obs_proc
    rq = float(np.percentile(np.abs(np.concatenate([res_init.ravel(), res_final.ravel()])), 99))
    rq = max(rq, 1e-6)

    axes[2, 0].axis("off")
    axes[2, 0].set_title(title)

    for ax, img, name in [
        (axes[2, 1], res_init, "residual init"),
        (axes[2, 2], res_final, "residual FWI"),
    ]:
        im = ax.imshow(img, cmap="seismic", aspect="auto", vmin=-rq, vmax=rq)
        ax.set_title(name)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--fdtd_csv", required=True)
    ap.add_argument("--sample_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--case_ids", nargs="+", type=int, default=None)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--tv_weight", type=float, default=1e-4)
    ap.add_argument("--prior_weight", type=float, default=1e-4)
    ap.add_argument("--eps_min", type=float, default=1.0)
    ap.add_argument("--eps_max", type=float, default=35.0)
    ap.add_argument("--sigma_mode", choices=["h5_median", "constant", "oracle"], default="h5_median")
    ap.add_argument("--sigma_const", type=float, default=0.0)
    ap.add_argument("--sigma_stats_n", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--grad_check_only", action="store_true")
    args = ap.parse_args()

    cfg = emdps.load_yaml(args.cfg)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("device:", device)

    op = emdps.make_fdtd(cfg, device)

    if args.sigma_mode == "h5_median":
        sigma_const = estimate_train_sigma_median(cfg, n=args.sigma_stats_n)
        print("sigma_const from train median:", sigma_const)
    elif args.sigma_mode == "constant":
        sigma_const = float(args.sigma_const)
        print("sigma_const:", sigma_const)
    else:
        sigma_const = None
        print("oracle sigma: diagnostic only")

    ranked = load_ranked_cases(args.fdtd_csv, args.case_ids)
    if len(ranked) == 0:
        raise SystemExit("No selected cases found")

    rows = []

    with h5py.File(cfg["data"]["h5"], "r") as f:
        for _, r in ranked.iterrows():
            case = int(r["case"])
            h5_idx = int(r["h5_index"])
            k = int(r["sample"])
            npz_path = Path(args.sample_dir) / f"case{case:03d}.npz"

            print(f"\n=== case {case:03d} | h5_idx={h5_idx} | init sample={k} ===")

            d = np.load(npz_path)
            eps_samples = d["eps"].astype("float32")
            if eps_samples.ndim == 4:
                eps_samples = eps_samples[:, 0]
            eps_init = eps_samples[k]

            eps_true = d["eps_true"].astype("float32")
            if eps_true.ndim == 3:
                eps_true = eps_true[0]
            eps_true_rs = resize_np(eps_true, eps_init.shape)

            obs_raw_np = f["bscan"][h5_idx].astype("float32")
            obs_raw_t = torch.from_numpy(obs_raw_np).to(device)

            obs_proc_t = torch_prep_bscan(obs_raw_t, cfg, obs_scale=None).detach()
            obs_scale = torch.quantile(obs_proc_t.abs().flatten(), 0.99).clamp_min(1e-6)
            obs_proc_t = torch_prep_bscan(obs_raw_t, cfg, obs_scale=obs_scale).detach()

            obs_proc_np = emdps.prep_bscan_np(obs_raw_np, cfg).astype("float32")

            if args.sigma_mode == "oracle":
                sigma_np = f["sigma"][h5_idx].astype("float32")
                sigma_np = resize_np(sigma_np, eps_init.shape)
            else:
                sigma_np = np.full_like(eps_init, sigma_const, dtype="float32")

            sigma_t = torch.from_numpy(sigma_np).to(device)

            p0 = inverse_sigmoid_from_eps(eps_init, args.eps_min, args.eps_max)
            p = torch.nn.Parameter(torch.from_numpy(p0).to(device))

            opt = torch.optim.Adam([p], lr=args.lr)

            init_t = torch.from_numpy(eps_init).to(device)

            eps_t = eps_from_param(p, args.eps_min, args.eps_max)
            sim_raw_t = call_fdtd_torch(op, eps_t, sigma_t, device)
            sim_proc_t = torch_prep_bscan(sim_raw_t, cfg, obs_scale=obs_scale)
            data_loss = ((sim_proc_t - obs_proc_t) ** 2).mean()
            loss = data_loss + args.tv_weight * tv_loss(eps_t) + args.prior_weight * ((eps_t - init_t) ** 2).mean()
            loss.backward()

            grad_norm = float(p.grad.detach().norm().item())
            print("grad_check loss:", float(loss.detach().item()), "grad_norm:", grad_norm)

            if args.grad_check_only:
                rows.append({
                    "case": case,
                    "h5_index": h5_idx,
                    "init_sample": k,
                    "status": "grad_check_ok",
                    "grad_norm": grad_norm,
                })
                continue

            p.grad.zero_()

            hist = []
            for it in range(args.steps):
                opt.zero_grad(set_to_none=True)

                eps_t = eps_from_param(p, args.eps_min, args.eps_max)
                sim_raw_t = call_fdtd_torch(op, eps_t, sigma_t, device)
                sim_proc_t = torch_prep_bscan(sim_raw_t, cfg, obs_scale=obs_scale)

                data_loss = ((sim_proc_t - obs_proc_t) ** 2).mean()
                reg_tv = tv_loss(eps_t)
                reg_prior = ((eps_t - init_t) ** 2).mean()

                loss = data_loss + args.tv_weight * reg_tv + args.prior_weight * reg_prior
                loss.backward()

                torch.nn.utils.clip_grad_norm_([p], 1.0)
                opt.step()

                if it % 10 == 0 or it == args.steps - 1:
                    print(
                        f"iter {it:04d}: loss={float(loss.detach()):.6g} "
                        f"data={float(data_loss.detach()):.6g} "
                        f"tv={float(reg_tv.detach()):.6g} prior={float(reg_prior.detach()):.6g}",
                        flush=True,
                    )

                hist.append({
                    "iter": it,
                    "loss": float(loss.detach().cpu()),
                    "data_loss": float(data_loss.detach().cpu()),
                    "tv": float(reg_tv.detach().cpu()),
                    "prior": float(reg_prior.detach().cpu()),
                })

            eps_final = eps_from_param(p, args.eps_min, args.eps_max).detach().cpu().numpy().astype("float32")

            sim_raw_init, sim_proc_init = official_forward_np(op, cfg, eps_init, sigma_np, device)
            sim_raw_final, sim_proc_final = official_forward_np(op, cfg, eps_final, sigma_np, device)

            if sim_proc_init.shape != obs_proc_np.shape:
                sim_proc_init = resize_np(sim_proc_init, obs_proc_np.shape)
            if sim_proc_final.shape != obs_proc_np.shape:
                sim_proc_final = resize_np(sim_proc_final, obs_proc_np.shape)

            init_res = sim_proc_init - obs_proc_np
            final_res = sim_proc_final - obs_proc_np

            init_mse = float(np.mean(init_res ** 2))
            final_mse = float(np.mean(final_res ** 2))
            init_mae = float(np.mean(np.abs(init_res)))
            final_mae = float(np.mean(np.abs(final_res)))
            init_corr = corr_np(sim_proc_init, obs_proc_np)
            final_corr = corr_np(sim_proc_final, obs_proc_np)

            np.savez_compressed(
                out_dir / f"case{case:03d}_fwi.npz",
                eps_init=eps_init,
                eps_final=eps_final,
                eps_true=eps_true_rs,
                sigma=sigma_np,
                obs_proc=obs_proc_np,
                sim_proc_init=sim_proc_init,
                sim_proc_final=sim_proc_final,
                case=case,
                h5_index=h5_idx,
                init_sample=k,
                hist=np.array([(h["iter"], h["loss"], h["data_loss"], h["tv"], h["prior"]) for h in hist], dtype="float32"),
            )

            save_plot(
                out_dir / f"case{case:03d}_fwi.png",
                eps_true_rs,
                eps_init,
                eps_final,
                obs_proc_np,
                sim_proc_init,
                sim_proc_final,
                title=f"case={case}, init k={k}\nMSE {init_mse:.4g}->{final_mse:.4g}, corr {init_corr:.3f}->{final_corr:.3f}",
            )

            row = {
                "case": case,
                "h5_index": h5_idx,
                "init_sample": k,
                "sigma_mode": args.sigma_mode,
                "init_mse": init_mse,
                "final_mse": final_mse,
                "delta_mse": final_mse - init_mse,
                "init_mae": init_mae,
                "final_mae": final_mae,
                "delta_mae": final_mae - init_mae,
                "init_corr": init_corr,
                "final_corr": final_corr,
                "delta_corr": final_corr - init_corr,
                "steps": args.steps,
                "lr": args.lr,
                "tv_weight": args.tv_weight,
                "prior_weight": args.prior_weight,
                "grad_norm_initial": grad_norm,
            }
            rows.append(row)

            print(
                f"official: mse {init_mse:.6g} -> {final_mse:.6g}; "
                f"mae {init_mae:.6g} -> {final_mae:.6g}; "
                f"corr {init_corr:.4f} -> {final_corr:.4f}"
            )

    csv_path = out_dir / "fwi_summary.csv"
    with open(csv_path, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\nsaved:", csv_path)

    if "final_mse" in rows[0]:
        df = pd.DataFrame(rows)
        print("\n=== Aggregate ===")
        print("mean init_mse:", float(df["init_mse"].mean()))
        print("mean final_mse:", float(df["final_mse"].mean()))
        print("mean delta_mse:", float(df["delta_mse"].mean()))
        print("mean init_corr:", float(df["init_corr"].mean()))
        print("mean final_corr:", float(df["final_corr"].mean()))
        print("mean delta_corr:", float(df["delta_corr"].mean()))

if __name__ == "__main__":
    main()
