from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import h5py
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def robust_v(a, q=99.0):
    a = np.asarray(a)
    v = np.percentile(np.abs(a[np.isfinite(a)]), q) if np.isfinite(a).any() else 1.0
    if not np.isfinite(v) or v <= 1e-8:
        v = 1.0
    return float(v)

def norm_clip(a, q=99.0, clip=1.0):
    a = np.asarray(a, dtype=np.float32)
    a = a - np.nanmedian(a)
    v = robust_v(a, q=q)
    a = a / v
    a = np.clip(a, -clip, clip)
    return a.astype(np.float32)

def moving_average_time(a, win):

    if win <= 1:
        return np.zeros_like(a, dtype=np.float32)

    win = int(win)
    if win % 2 == 0:
        win += 1

    pad = win // 2
    x = np.pad(a, ((pad, pad), (0, 0)), mode="edge")
    c = np.cumsum(x, axis=0, dtype=np.float64)
    out = (c[win:] - c[:-win]) / float(win)

    if out.shape[0] != a.shape[0]:
        out = out[:a.shape[0]]
        if out.shape[0] < a.shape[0]:
            out = np.pad(out, ((0, a.shape[0] - out.shape[0]), (0, 0)), mode="edge")
    return out.astype(np.float32)

def dewow(a, win=101):
    a = np.asarray(a, dtype=np.float32)
    return (a - moving_average_time(a, win)).astype(np.float32)

def background_remove(a, mode="median"):

    a = np.asarray(a, dtype=np.float32)
    if mode == "mean":
        bg = np.mean(a, axis=1, keepdims=True)
    else:
        bg = np.median(a, axis=1, keepdims=True)
    return (a - bg).astype(np.float32)

def estimate_t0(raw, thr=0.08):

    a = np.asarray(raw, dtype=np.float32)
    e = np.median(np.abs(a), axis=1)
    
    k = 9
    pad = k // 2
    es = np.pad(e, (pad, pad), mode="edge")
    es = np.convolve(es, np.ones(k) / k, mode="valid")

    m = float(es.max())
    if m <= 1e-8 or not np.isfinite(m):
        return 0

    idx = np.where(es >= thr * m)[0]
    if len(idx) == 0:
        return int(np.argmax(es))
    return int(idx[0])

def crop_from_t0(a, t0, crop_len=512, pre=10):
    a = np.asarray(a, dtype=np.float32)
    T, R = a.shape
    start = max(0, int(t0) - int(pre))
    end = start + int(crop_len)

    out = a[start:min(end, T)]

    if out.shape[0] < crop_len:
        pad = crop_len - out.shape[0]
        out = np.pad(out, ((0, pad), (0, 0)), mode="constant")

    return out.astype(np.float32), start

def mute_top(a, mute=70, taper=30):

    a = np.asarray(a, dtype=np.float32).copy()
    n = a.shape[0]

    mute = max(0, min(int(mute), n))
    taper = max(0, min(int(taper), n - mute))

    if mute > 0:
        a[:mute] = 0.0

    if taper > 0:
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, taper, dtype=np.float32))
        a[mute:mute + taper] *= ramp[:, None]

    return a.astype(np.float32)

def resize_2d(a, size=(256, 256), mode="bilinear"):
    x = torch.from_numpy(np.asarray(a, dtype=np.float32))[None, None]
    y = F.interpolate(x, size=size, mode=mode, align_corners=False if mode in ["bilinear", "bicubic"] else None)
    return y[0, 0].numpy().astype(np.float32)

def im(ax, arr, title, cmap="seismic", vmin=None, vmax=None):
    arr = np.asarray(arr).squeeze()
    if vmin is None or vmax is None:
        if cmap == "seismic":
            v = robust_v(arr)
            vmin, vmax = -v, v
    h = ax.imshow(arr, aspect="auto", origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(h, ax=ax, fraction=0.046, pad=0.02)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default="data/raw.h5")
    ap.add_argument("--out", default="runs/preprocessing.png")
    ap.add_argument("-n", type=int, default=6)
    ap.add_argument("--crop_len", type=int, default=512)
    ap.add_argument("--pre", type=int, default=10)
    ap.add_argument("--dewow_win", type=int, default=121)
    ap.add_argument("--mute", type=int, default=70)
    ap.add_argument("--taper", type=int, default=30)
    ap.add_argument("--thr", type=float, default=0.08)
    ap.add_argument("--resize", nargs=2, type=int, default=[256, 256])
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.h5, "r") as f:
        idx = np.where(f["valid"][:] == 1)[0][:args.n]

        rows = []
        for i in idx:
            eps = f["eps"][i]
            sig = f["sigma"][i]
            raw = f["bscan"][i]

            subset = int(f["subset"][i])
            src = f["source"][i]
            if isinstance(src, bytes):
                src = src.decode()

            t0 = estimate_t0(raw, thr=args.thr)

            d = dewow(raw, win=args.dewow_win)
            bg = background_remove(d, mode="median")
            crop, start = crop_from_t0(bg, t0, crop_len=args.crop_len, pre=args.pre)

            muted = mute_top(crop, mute=args.mute, taper=args.taper)
            final_native = norm_clip(muted, q=99.0, clip=1.0)
            final_256 = resize_2d(final_native, size=tuple(args.resize), mode="bilinear")
            final_256 = np.clip(final_256, -1, 1)

            rows.append({
                "i": int(i),
                "subset": subset,
                "src": str(src),
                "eps": eps,
                "sig": sig,
                "raw": raw,
                "dewow": d,
                "bg": bg,
                "crop": crop,
                "muted": final_native,
                "final": final_256,
                "t0": t0,
                "start": start,
            })

    fig, ax = plt.subplots(len(rows), 7, figsize=(31, 3.1 * len(rows)), constrained_layout=True)
    if len(rows) == 1:
        ax = ax[None, :]

    for r, row in enumerate(rows):
        title = f"{row['i']} s{row['subset']}-{row['src']}"

        im(ax[r, 0], row["eps"], title + " eps", "turbo")
        im(ax[r, 1], row["sig"], title + " sigma", "magma")
        im(ax[r, 2], row["raw"], f"raw {row['raw'].shape}", "seismic")
        im(ax[r, 3], row["dewow"], f"dewow win={args.dewow_win}", "seismic")
        im(ax[r, 4], row["bg"], "dewow + bg median", "seismic")
        im(ax[r, 5], row["muted"], f"crop+mute native {row['muted'].shape}\nt0={row['t0']} start={row['start']}", "seismic", -1, 1)
        im(ax[r, 6], row["final"], f"final condition {tuple(args.resize)}", "seismic", -1, 1)

    fig.savefig(out, dpi=180, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print("[saved]", out)
    for row in rows:
        print(
            f"case={row['i']} subset={row['subset']} src={row['src']} "
            f"t0={row['t0']} start={row['start']} "
            f"raw_std={row['raw'].std():.4g} final_std={row['final'].std():.4g}"
        )

if __name__ == "__main__":
    main()
