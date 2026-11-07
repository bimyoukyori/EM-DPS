#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

from fdtd.operator import DifferentiableGPRFDTD

def load_yaml(p: str | Path) -> dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_json(obj: dict[str, Any], p: str | Path) -> None:
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def nkey(x: str | Path):
    s = Path(x).stem if isinstance(x, Path) else str(x)
    try: return int(s)
    except Exception: return s

def mat2d(path: str | Path, key: str) -> np.ndarray:
    m = sio.loadmat(path)
    if key not in m:
        ks = [k for k in m.keys() if not k.startswith("__")]
        raise KeyError(f"{path} missing key={key}; keys={ks}")
    a = np.asarray(m[key], dtype=np.float32).squeeze()
    if a.ndim != 2:
        raise RuntimeError(f"{path}:{key} expected 2D, got {a.shape}")
    if not np.isfinite(a).all():
        raise RuntimeError(f"{path}:{key} contains NaN/Inf")
    return a.astype(np.float32)

def im(ax, arr, title: str, cmap="viridis", vmin=None, vmax=None):
    arr = np.asarray(arr).squeeze()
    h = ax.imshow(arr, aspect="auto", origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(h, ax=ax, fraction=0.046, pad=0.02)

def robust_v(a: np.ndarray, q=99.0) -> float:
    v = float(np.percentile(np.abs(np.asarray(a)), q))
    return v if np.isfinite(v) and v > 1e-8 else 1.0

def fdtd_sim_dict(cfg: dict[str, Any]) -> dict[str, Any]:
    f = cfg["fdtd"]
    dx = float(f["dx"])
    domain_x = float(f.get("domain_x", 1.5))
    domain_z = float(f.get("domain_z", 0.5))
    nx = int(f.get("nx", round(domain_z / dx)))
    ny = int(f.get("ny", round(domain_x / dx)))

    scan_start = float(f.get("scan_start", 0.12))
    scan_end = float(f.get("scan_end", 1.18))
    scan_step = float(f.get("scan_step", 0.025))
    offset = float(f.get("offset", 0.20))
    scan_steps = int(f.get("scan_steps", round((scan_end - scan_start) / scan_step) + 1))
    last_rx = scan_start + (scan_steps - 1) * scan_step + offset
    if last_rx > domain_x + 1e-6:
        raise RuntimeError(f"receiver exceeds domain: last_rx={last_rx:.4f} > domain_x={domain_x:.4f}")

    freq = float(f.get("freq", 6.5e8))
    return {
        "enabled": True,
        "dx": dx,
        "nx": nx,
        "ny": ny,
        "nz": int(f.get("nz", 1)),
        "freq": freq,
        "time_window": float(f.get("time_window", 2.0e-8)),
        "source": list(f.get("source", ["Hertzian", "z"])),
        "source_amp": list(f.get("source_amp", ["gaussian", 1.0, freq])),
        "source_location": [float(f.get("source_depth", 0.085)), scan_start, 0.0],
        "receiver_location": [float(f.get("receiver_depth", 0.085)), scan_start + offset, 0.0],
        "source_step": [0.0, scan_step, 0.0],
        "receiver_step": [0.0, scan_step, 0.0],
        "scan_steps": scan_steps,
        "nrx": int(f.get("nrx", 1)),
        "nsrc": int(f.get("nsrc", 1)),
        "pmlthick": int(f.get("pml", 12)),
        "mr_default": float(f.get("mr", 1.0)),
        "receiver_component_index": int(f.get("receiver_component", 2)),
        "fixed_receiver": bool(f.get("fixed_receiver", False)),
        "air_cells": int(f.get("air_cells", 0)),
        "air_eps": float(f.get("air_eps", 1.0)),
        "air_sigma": float(f.get("air_sigma", 0.0)),
    }

def make_fdtd(cfg: dict[str, Any], device: torch.device) -> DifferentiableGPRFDTD:
    return DifferentiableGPRFDTD(fdtd_sim_dict(cfg), device=device).to(device)

@torch.no_grad()
def run_fdtd(op: DifferentiableGPRFDTD, eps: np.ndarray, sigma: np.ndarray, device: torch.device) -> np.ndarray:
    e = torch.as_tensor(eps, dtype=torch.float32, device=device)[None, None]
    s = torch.as_tensor(sigma, dtype=torch.float32, device=device)[None, None]
    b = op(e, s).detach().float().cpu().squeeze().numpy()
    b = np.squeeze(b)
    if b.ndim != 2:
        raise RuntimeError(f"FDTD output expected 2D after squeeze, got {b.shape}")
    if not np.isfinite(b).all():
        raise RuntimeError("FDTD output contains NaN/Inf")
    return b.astype(np.float32)

def discover(src_root: Path, subsets: list[int], max_per_subset: int = 0) -> list[dict[str, Any]]:
    rows = []
    for sid in subsets:
        root = src_root / str(sid)
        d1, d2 = root / "data1", root / "data2"
        if not d1.exists() or not d2.exists():
            raise FileNotFoundError(f"missing {d1} or {d2}")
        ids = sorted({p.stem for p in d1.glob("*.mat")} & {p.stem for p in d2.glob("*.mat")}, key=nkey)
        if max_per_subset > 0: ids = ids[:max_per_subset]
        print(f"[discover] subset={sid} selected={len(ids)}", flush=True)
        for oid in ids:
            rows.append({"subset": int(sid), "source": str(oid), "eps": str(d1 / f"{oid}.mat"), "sigma": str(d2 / f"{oid}.mat")})
    return rows

def create_h5(path: Path, n: int, eps_shape, bscan_shape, meta: dict[str, Any], overwrite: bool, resume: bool):

    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and overwrite:
        path.unlink()

    if path.exists():
        if not resume:
            raise FileExistsError(f"{path} exists; use --resume or --overwrite")
        return h5py.File(path, "r+")

    f = h5py.File(path, "w")

    comp = dict(compression="gzip", compression_opts=4, shuffle=True)

    f.create_dataset(
        "eps",
        shape=(n, *eps_shape),
        dtype="float16",
        chunks=(1, *eps_shape),
        **comp,
    )
    f.create_dataset(
        "sigma",
        shape=(n, *eps_shape),
        dtype="float16",
        chunks=(1, *eps_shape),
        **comp,
    )
    f.create_dataset(
        "bscan",
        shape=(n, *bscan_shape),
        dtype="float16",
        chunks=(1, *bscan_shape),
        **comp,
    )

    f.create_dataset("subset", shape=(n,), dtype="int16")
    f.create_dataset("source", shape=(n,), dtype=h5py.string_dtype("utf-8"))
    f.create_dataset("valid", shape=(n,), dtype="uint8")

    f.attrs["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
    f.attrs["schema"] = (
        "compact: eps/sigma/bscan are float16 compressed; "
        "bscan is raw current-FDTD; no preprocessing stored"
    )
    f.attrs["storage"] = "float16 + gzip(level=4) + shuffle"
    for k, v in meta.items():
        f.attrs[k] = json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict, tuple)) else str(v)

    return f

def mode_h5(args):
    cfg = load_yaml(args.cfg)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    src = Path(args.src or cfg["data"]["src"])
    out = Path(args.out or cfg["data"].get("h5", "data/raw.h5"))
    subsets = [int(x) for x in (args.subsets or cfg["data"].get("subsets", [1,2,3,4,5,6]))]
    max_per_subset = int(args.max_per_subset if args.max_per_subset is not None else cfg["data"].get("max_per_subset", 0))
    rows = discover(src, subsets, max_per_subset)
    if not rows: raise RuntimeError("no samples discovered")

    print(f"[fdtd] device={device} out={out} n={len(rows)}", flush=True)
    op = make_fdtd(cfg, device)

    eps0 = mat2d(rows[0]["eps"], "data1")
    sig0 = mat2d(rows[0]["sigma"], "data2")
    b0 = run_fdtd(op, eps0, sig0, device)
    print(f"[shape] eps={eps0.shape} bscan={b0.shape} bscan_std={b0.std():.6g}", flush=True)

    f = create_h5(out, len(rows), eps0.shape, b0.shape, {"src": str(src), "subsets": subsets, "fdtd": fdtd_sim_dict(cfg)}, args.overwrite, args.resume)
    prev = []
    try:
        for i in tqdm(range(len(rows)), total=len(rows), dynamic_ncols=True):
            if f["valid"][i] == 1:
                continue
            r = rows[i]
            if i == 0:
                eps, sig, b = eps0, sig0, b0
            else:
                eps, sig = mat2d(r["eps"], "data1"), mat2d(r["sigma"], "data2")
                b = run_fdtd(op, eps, sig, device)
            f["eps"][i] = eps; f["sigma"][i] = sig; f["bscan"][i] = b
            f["subset"][i] = int(r["subset"]); f["source"][i] = str(r["source"]); f["valid"][i] = 1
            if len(prev) < int(cfg["data"].get("preview_n", 6)):
                prev.append((i, r, eps.copy(), sig.copy(), b.copy()))
            if (i + 1) % 25 == 0: f.flush()
        f.attrs["finished"] = "true"; f.attrs["n"] = len(rows); f.flush()
    finally:
        f.close()
    Path(str(out) + ".json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_preview(prev, Path(cfg["data"].get("preview", "runs/h5_preview.png")))
    print(f"[done] {out}", flush=True)

def plot_preview(items, path: Path):
    if not items: return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(len(items), 3, figsize=(15, 3.0 * len(items)), constrained_layout=True)
    if len(items) == 1: ax = ax[None, :]
    for j, (i, r, eps, sig, b) in enumerate(items):
        title = f"{i} s{r['subset']}-{r['source']}"
        im(ax[j,0], eps, title + " eps", "turbo")
        im(ax[j,1], sig, title + " sigma", "magma")
        v = robust_v(b); im(ax[j,2], b, title + " raw bscan", "seismic", -v, v)
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[preview] {path}", flush=True)

@dataclass
class Stats:
    x_mean: list[float]
    x_std: list[float]
    y_mean: float
    y_std: float
    eps_shape: list[int]
    bscan_shape: list[int]

    def save(self, path: Path):
        save_json(self.__dict__, path)

    @staticmethod
    def load(path: Path):
        return Stats(**json.loads(path.read_text()))

def split_indices(n: int, seed: int, train: float, val: float) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    nt = int(round(n * train))
    nv = int(round(n * val))
    return {"train": idx[:nt], "val": idx[nt:nt + nv], "test": idx[nt + nv:]}

def prep_bscan_np(raw, cfg):

    prep = cfg.get("prep", {})
    a = np.asarray(raw, dtype=np.float32).squeeze()

    if a.ndim != 2:
        raise RuntimeError(f"prep_bscan_np expected 2D raw bscan, got {a.shape}")

    if not np.isfinite(a).all():
        raise RuntimeError("raw bscan contains NaN/Inf before preprocessing")

    a = float(prep.get("polarity", 1.0)) * a

    from preprocessing import dewow, background_remove, norm_clip

    d = dewow(a, win=int(prep.get("dewow_win", 121)))
    bg = background_remove(d, mode=str(prep.get("background", "median")).lower())

    out = norm_clip(
        bg,
        q=float(prep.get("norm_q", 99.0)),
        clip=float(prep.get("clip", 1.0)),
    )

    if out.ndim != 2 or out.shape[0] <= 0 or out.shape[1] <= 0:
        raise RuntimeError(f"bad processed condition shape: {out.shape}")

    if not np.isfinite(out).all():
        raise RuntimeError("processed condition contains NaN/Inf")

    return out.astype(np.float32)

def resize_torch(x, size):

    if isinstance(size, dict):
        if "height" in size and "width" in size:
            size = [size["height"], size["width"]]
        elif "h" in size and "w" in size:
            size = [size["h"], size["w"]]
        else:
            raise RuntimeError(f"Unsupported resize size dict: {size}")

    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise RuntimeError(f"resize_torch expects size=[H,W], got {size}")

    size = (int(size[0]), int(size[1]))

    if size[0] <= 0 or size[1] <= 0:
        raise RuntimeError(f"Bad target resize size: {size}")

    if x.ndim != 4:
        raise RuntimeError(f"resize_torch expects [B,C,H,W], got {tuple(x.shape)}")

    return F.interpolate(x.float(), size=size, mode="bilinear", align_corners=False)

def compute_stats_h5(cfg: dict[str, Any], idx: np.ndarray, max_cases: int = 0) -> Stats:

    h5 = Path(cfg["data"]["h5"])

    xsum = np.zeros(1, dtype=np.float64)
    x2sum = np.zeros(1, dtype=np.float64)
    xc = 0

    ysum = 0.0
    y2sum = 0.0
    yc = 0

    use = idx[:max_cases] if max_cases and max_cases > 0 else idx

    with h5py.File(h5, "r") as f:
        for i in tqdm(use, desc="stats", dynamic_ncols=True):
            ii = int(i)

            eps = f["eps"][ii].astype(np.float64)
            raw = f["bscan"][ii].astype(np.float32)
            y = prep_bscan_np(raw, cfg).astype(np.float64)

            xsum[0] += eps.sum()
            x2sum[0] += (eps * eps).sum()
            xc += eps.size

            ysum += y.sum()
            y2sum += (y * y).sum()
            yc += y.size

        eps_shape = list(f["eps"].shape[1:])
        bshape = list(f["bscan"].shape[1:])

    xm = xsum / max(xc, 1)
    var = x2sum / max(xc, 1) - xm * xm
    var = np.nan_to_num(var, nan=1e-12, posinf=1e12, neginf=1e-12)
    xs = np.sqrt(np.maximum(var, 1e-12))

    ym = ysum / max(yc, 1)
    yvar = y2sum / max(yc, 1) - ym * ym
    yvar = float(np.nan_to_num(yvar, nan=1e-12, posinf=1e12, neginf=1e-12))
    ys = math.sqrt(max(yvar, 1e-12))

    return Stats(
        [float(xm[0])],
        [float(xs[0])],
        float(ym),
        float(ys),
        eps_shape,
        bshape,
    )

class H5Set(Dataset):

    def __init__(self, cfg: dict[str, Any], split: str, stats: Stats):
        self.cfg, self.stats = cfg, stats

        with h5py.File(cfg["data"]["h5"], "r") as f:
            valid = np.where(f["valid"][:] == 1)[0]

        sp = split_indices(
            len(valid),
            int(cfg["data"].get("split_seed", 42)),
            float(cfg["data"].get("train_frac", 0.8)),
            float(cfg["data"].get("val_frac", 0.1)),
        )
        self.indices = valid[sp[split]]
        self.h5 = None

    def __len__(self):
        return len(self.indices)

    def _file(self):
        if self.h5 is None:
            self.h5 = h5py.File(self.cfg["data"]["h5"], "r")
        return self.h5

    def __getitem__(self, j: int):
        f = self._file()
        i = int(self.indices[j])

        eps = f["eps"][i].astype(np.float32)
        y = prep_bscan_np(f["bscan"][i], self.cfg)

        x = torch.from_numpy(eps[None, None].astype(np.float32))
        y = torch.from_numpy(y[None, None].astype(np.float32))

        xm = torch.tensor(self.stats.x_mean, dtype=torch.float32)[None, :, None, None]
        xs = torch.tensor(self.stats.x_std, dtype=torch.float32)[None, :, None, None]

        x = (x - xm) / xs.clamp_min(1e-12)
        y = (y - float(self.stats.y_mean)) / max(float(self.stats.y_std), 1e-12)

        x = resize_torch(x, self.cfg["model"]["x_resolution"])[0].contiguous()
        y = resize_torch(y, self.cfg["model"]["y_resolution"])[0].contiguous()

        return {"x": x, "y": y, "index": i}

def gn(ch: int):
    g = min(32, ch)
    while ch % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, ch)

class Emb(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = int(d)
        self.net = nn.Sequential(
            nn.Linear(d, 4 * d),
            nn.SiLU(),
            nn.Linear(4 * d, d),
        )

    def forward(self, c):
        c = c.reshape(-1, 1)
        half = self.d // 2
        freqs = torch.exp(
            torch.arange(half, device=c.device, dtype=c.dtype)
            * (-math.log(10000) / max(half - 1, 1))
        )[None]
        e = torch.cat([torch.sin(c * freqs), torch.cos(c * freqs)], dim=1)
        if e.shape[1] < self.d:
            e = F.pad(e, (0, self.d - e.shape[1]))
        return self.net(e)

class Res(nn.Module):
    def __init__(self, ci, co, ce):
        super().__init__()
        self.n1 = gn(ci)
        self.c1 = nn.Conv2d(ci, co, 3, padding=1)
        self.e = nn.Linear(ce, 2 * co)
        self.n2 = gn(co)
        self.c2 = nn.Conv2d(co, co, 3, padding=1)
        self.s = nn.Conv2d(ci, co, 1) if ci != co else nn.Identity()

    def forward(self, x, e):
        h = self.c1(F.silu(self.n1(x)))
        sc, sh = self.e(F.silu(e))[:, :, None, None].chunk(2, dim=1)
        h = self.n2(h) * (1 + sc) + sh
        h = self.c2(F.silu(h))
        return h + self.s(x)

class TimeSignalEncoder(nn.Module):

    def __init__(self, cin, cout, hidden, t_in, h_out, w_out, blocks=4):
        super().__init__()
        self.t_in = int(t_in)
        self.h_out = int(h_out)
        self.w_out = int(w_out)

        layers = []
        ch = cin + 2  
        for _ in range(int(blocks)):
            layers += [nn.Conv1d(ch, hidden, 5, padding=2), nn.GELU()]
            ch = hidden
        layers += [nn.Conv1d(ch, cout, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, y):
        B, C, T, W = y.shape

        if T != self.t_in or W != self.w_out:
            y = F.interpolate(
                y,
                size=(self.t_in, self.w_out),
                mode="bilinear",
                align_corners=False,
            )
            T, W = self.t_in, self.w_out

        tt = torch.linspace(-1, 1, T, device=y.device, dtype=y.dtype)[None, None, :, None].expand(B, 1, T, W)
        xx = torch.linspace(-1, 1, W, device=y.device, dtype=y.dtype)[None, None, None, :].expand(B, 1, T, W)

        z = torch.cat([y, tt, xx], dim=1)
        z = z.permute(0, 3, 1, 2).reshape(B * W, C + 2, T)

        z = self.net(z)
        z = F.adaptive_avg_pool1d(z, self.h_out)

        Co = z.shape[1]
        return z.reshape(B, W, Co, self.h_out).permute(0, 2, 3, 1).contiguous()

class UNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        m = cfg["model"]

        H, W = [int(v) for v in m["x_resolution"]]
        T, YW = [int(v) for v in m["y_resolution"]]

        base = int(m.get("ch", 80))
        cc = int(m.get("cond_ch", 64))
        ce = int(m.get("emb_ch", 320))
        mult = list(m.get("mult", [1, 2, 2, 4]))

        self.x_ch = int(m.get("x_ch", 1))

        self.cond = TimeSignalEncoder(
            cin=1,
            cout=cc,
            hidden=cc,
            t_in=T,
            h_out=H,
            w_out=W,
            blocks=int(m.get("cond_blocks", 4)),
        )
        self.emb = Emb(ce)

        self.inp = nn.Conv2d(self.x_ch + cc + 2, base, 3, padding=1)

        self.down = nn.ModuleList()
        self.ds = nn.ModuleList()

        ch = base
        skips = []
        for k, mm in enumerate(mult):
            co = base * int(mm)
            self.down.append(Res(ch, co, ce))
            ch = co
            skips.append(ch)
            self.ds.append(
                nn.Conv2d(ch, ch, 3, stride=2, padding=1)
                if k < len(mult) - 1
                else nn.Identity()
            )

        self.mid1 = Res(ch, ch, ce)
        self.mid2 = Res(ch, ch, ce)

        self.up = nn.ModuleList()
        self.us = nn.ModuleList()
        for k, mm in enumerate(reversed(mult)):
            sc = skips.pop()
            co = base * int(mm)
            self.up.append(Res(ch + sc, co, ce))
            ch = co
            self.us.append(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
                if k < len(mult) - 1
                else nn.Identity()
            )

        self.outn = gn(ch)
        self.out = nn.Conv2d(ch, self.x_ch, 3, padding=1)

    def grid(self, B, H, W, device, dtype):
        yy = torch.linspace(-1, 1, H, device=device, dtype=dtype)[None, None, :, None].expand(B, 1, H, W)
        xx = torch.linspace(-1, 1, W, device=device, dtype=dtype)[None, None, None, :].expand(B, 1, H, W)
        return torch.cat([yy, xx], dim=1)

    def forward(self, x, c, y):
        B, _, H, W = x.shape

        e = self.emb(c)
        cond = self.cond(y)

        if cond.shape[-2:] != (H, W):
            cond = F.interpolate(cond, size=(H, W), mode="bilinear", align_corners=False)

        h = self.inp(torch.cat([x, cond, self.grid(B, H, W, x.device, x.dtype)], dim=1))

        ss = []
        for b, d in zip(self.down, self.ds):
            h = b(h, e)
            ss.append(h)
            h = d(h)

        h = self.mid2(self.mid1(h, e), e)

        for b, u in zip(self.up, self.us):
            skip = ss.pop()
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = b(torch.cat([h, skip], dim=1), e)
            h = u(h)

        if h.shape[-2:] != (H, W):
            h = F.interpolate(h, size=(H, W), mode="bilinear", align_corners=False)

        return self.out(F.silu(self.outn(h)))

class EDM(nn.Module):

    def __init__(self, net, sigma_data=0.5):
        super().__init__()
        self.net = net
        self.sd = float(sigma_data)

    def forward(self, x, sigma, y):
        sigma = sigma.reshape(-1, 1, 1, 1).to(x)
        sd = self.sd

        cskip = sd * sd / (sigma * sigma + sd * sd)
        cout = sigma * sd / torch.sqrt(sigma * sigma + sd * sd)
        cin = 1.0 / torch.sqrt(sigma * sigma + sd * sd)
        cnoise = torch.log(sigma.reshape(-1)) / 4.0

        return cskip * x + cout * self.net(cin * x, cnoise, y)

def loss_edm(model, x, y, cfg):

    ncfg = cfg.get("noise", {})
    sd = float(ncfg.get("sigma_data", 0.5))
    p_mean = float(ncfg.get("P_mean", 0.0))
    p_std = float(ncfg.get("P_std", 1.0))

    B = x.shape[0]
    rnd = torch.randn(B, device=x.device, dtype=x.dtype)
    sigma = torch.exp(p_mean + p_std * rnd).to(dtype=x.dtype)
    sigma4 = sigma[:, None, None, None]

    xn = x + torch.randn_like(x) * sigma4

    den = model(xn, sigma, y)

    edm_w = (sigma4 * sigma4 + sd * sd) / ((sigma4 * sd) ** 2).clamp_min(1e-12)
    err2 = (den - x) ** 2

    tcfg = cfg.get("target_loss", {})
    if bool(tcfg.get("enabled", False)):
        thr_z = float(tcfg.get("threshold_z", 2.0))
        roi_weight = float(tcfg.get("roi_weight", 20.0))

        bg = x.flatten(2).median(dim=-1).values[:, :, None, None]
        mask = (x - bg).abs() > thr_z
        spatial_w = 1.0 + (roi_weight - 1.0) * mask.float()
    else:
        spatial_w = 1.0

    loss = (spatial_w * edm_w * err2).mean()

    acfg = cfg.get("cond_anchor", {})
    if bool(acfg.get("enabled", False)):
        anchor_sigma = float(acfg.get("sigma", 5.0))
        anchor_weight = float(acfg.get("weight", 1.0))

        sigma_a = torch.full([B], anchor_sigma, device=x.device, dtype=x.dtype)
        xa = torch.randn_like(x) * anchor_sigma

        den_a = model(xa, sigma_a, y)
        err_a = (den_a - x) ** 2

        loss_anchor = (spatial_w * err_a).mean()
        loss = loss + anchor_weight * loss_anchor

    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite EDM loss")

    return loss

def make_model(cfg, device):
    return EDM(
        UNet(cfg),
        float(cfg["noise"].get("sigma_data", 0.5)),
    ).to(device)

def denorm_x(x, stats: Stats):
    m = torch.tensor(stats.x_mean, device=x.device, dtype=x.dtype)[None, :, None, None]
    s = torch.tensor(stats.x_std, device=x.device, dtype=x.dtype)[None, :, None, None]
    return x * s + m

def mode_train(args):
    cfg = load_yaml(args.cfg)
    seed_all(int(cfg.get("seed", 123)))

    device = torch.device("cuda" if cfg.get("device", "cuda") == "cuda" and torch.cuda.is_available() else "cpu")

    run = Path(cfg["run"])
    (run / "ckpt").mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.cfg, run / "cfg.yaml")

    with h5py.File(cfg["data"]["h5"], "r") as f:
        n = int((f["valid"][:] == 1).sum())

    sp = split_indices(
        n,
        int(cfg["data"].get("split_seed", 42)),
        float(cfg["data"].get("train_frac", 0.8)),
        float(cfg["data"].get("val_frac", 0.1)),
    )

    stats_path = run / "stats.json"
    if stats_path.exists() and not bool(cfg["data"].get("recompute_stats", False)):
        stats = Stats.load(stats_path)
    else:
        stats = compute_stats_h5(cfg, sp["train"], int(cfg["data"].get("stats_max", 0)))
    stats.save(stats_path)

    print("[stats]", stats, flush=True)

    tr = H5Set(cfg, "train", stats)
    va = H5Set(cfg, "val", stats)

    print(f"[data] train={len(tr)} val={len(va)}", flush=True)

    tl = DataLoader(
        tr,
        batch_size=int(cfg["train"].get("batch", 8)),
        shuffle=True,
        num_workers=int(cfg["train"].get("workers", 4)),
        pin_memory=True,
        drop_last=True,
    )
    vl = DataLoader(
        va,
        batch_size=int(cfg["train"].get("val_batch", 8)),
        shuffle=False,
        num_workers=int(cfg["train"].get("workers", 4)),
        pin_memory=True,
    )

    model = make_model(cfg, device)
    print(f"[model] params={sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M", flush=True)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"].get("lr", 1e-4)),
        weight_decay=float(cfg["train"].get("wd", 1e-4)),
    )

    epochs = int(cfg["train"].get("epochs", 300))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=epochs,
        eta_min=float(cfg["train"].get("eta_min", 1e-6)),
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=bool(cfg["train"].get("amp", False)) and device.type == "cuda",
    )

    best = float("inf")
    best_ep = 0
    bad_epochs = 0
    hist = []

    early_patience = int(cfg["train"].get("early_stop_patience", 50))
    early_min_delta = float(cfg["train"].get("early_stop_min_delta", 1e-4))

    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        s = 0.0
        c = 0

        for batch in tqdm(tl, desc=f"epoch {ep:03d}", dynamic_ncols=True, leave=False):
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                loss = loss_edm(model, x, y, cfg)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"].get("clip_grad", 1.0)))
            scaler.step(opt)
            scaler.update()

            s += float(loss.detach().cpu())
            c += 1

        train_loss = s / max(c, 1)

        model.eval()
        vs = 0.0
        vc = 0

        with torch.no_grad():
            for j, batch in enumerate(vl, 1):
                x = batch["x"].to(device, non_blocking=True)
                y = batch["y"].to(device, non_blocking=True)
                vs += float(loss_edm(model, x, y, cfg).detach().cpu())
                vc += 1
                if j >= int(cfg["train"].get("max_val_batches", 60)):
                    break

        val = vs / max(vc, 1)
        sched.step()

        improved = val < (best - early_min_delta)
        mark = ""
        pack = {
            "epoch": ep,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "scaler": scaler.state_dict() if scaler.is_enabled() else None,
            "stats": stats.__dict__,
            "cfg": cfg,
            "best": min(best, val),
        }

        if improved:
            best = val
            best_ep = ep
            bad_epochs = 0
            mark = " ✓"
            torch.save(pack, run / "ckpt" / "best.pt")
        else:
            bad_epochs += 1

        torch.save(pack, run / "ckpt" / "last.pt")

        row = {
            "epoch": ep,
            "train": train_loss,
            "val": val,
            "lr": sched.get_last_lr()[0],
            "sec": time.time() - t0,
            "best": best,
            "best_ep": best_ep,
            "bad_epochs": bad_epochs,
        }
        hist.append(row)

        print(
            f"[epoch {ep:04d}] train={train_loss:.5f} val={val:.5f} "
            f"lr={row['lr']:.2e} time={row['sec']:.1f}s "
            f"best={best:.5f} best_ep={best_ep} bad={bad_epochs}{mark}",
            flush=True,
        )

        import pandas as pd
        pd.DataFrame(hist).to_csv(run / "history.csv", index=False)

        if early_patience > 0 and bad_epochs >= early_patience:
            print(
                f"[early-stop] no val improvement > {early_min_delta:g} "
                f"for {bad_epochs} epochs; best={best:.5f} at epoch {best_ep}",
                flush=True,
            )
            break

@torch.no_grad()
def edm_sample(model, y, shape, cfg, device):
    s = cfg["sample"]

    steps = int(s.get("steps", 64))
    smin = float(s.get("sigma_min", 0.002))
    smax = float(s.get("sigma_max", 80.0))
    rho = float(s.get("rho", 7.0))

    r = torch.linspace(0, 1, steps, device=device)
    sigmas = (smax ** (1 / rho) + r * (smin ** (1 / rho) - smax ** (1 / rho))) ** rho
    sigmas = torch.cat([sigmas, torch.zeros_like(sigmas[:1])])

    x = torch.randn(shape, device=device) * sigmas[0]

    for i in range(steps):
        t = sigmas[i]
        tn = sigmas[i + 1]

        tb = torch.full([shape[0]], float(t), device=device)

        den = model(x, tb, y)
        d = (x - den) / t.clamp_min(1e-12)
        xe = x + (tn - t) * d

        if float(tn) > 0:
            tnb = torch.full([shape[0]], float(tn), device=device)
            den2 = model(xe, tnb, y)
            d2 = (xe - den2) / tn.clamp_min(1e-12)
            x = x + (tn - t) * 0.5 * (d + d2)
        else:
            x = xe

    return x

def mode_sample(args):
    cfg = load_yaml(args.cfg)
    seed_all(int(cfg.get("seed", 123)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = args.ckpt or Path(cfg["run"]) / "ckpt" / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)

    stats = Stats(**ckpt["stats"])
    model = make_model(cfg, device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    ds = H5Set(cfg, args.split, stats)

    out = Path(cfg["run"]) / "samples"
    out.mkdir(parents=True, exist_ok=True)

    ids = list(range(len(ds)))
    random.shuffle(ids)
    ids = ids[:int(cfg["sample"].get("cases", 4))]

    K = int(cfg["sample"].get("K", 8))

    with h5py.File(cfg["data"]["h5"], "r") as f:
        for case_no, j in enumerate(ids):
            b = ds[j]
            idx = int(b["index"])

            y = b["y"][None].to(device)
            yy = y.expand(K, -1, -1, -1).contiguous()

            x = edm_sample(
                model,
                yy,
                (K, 1, *cfg["model"]["x_resolution"]),
                cfg,
                device,
            )

            eps_pred = denorm_x(x, stats).cpu().numpy()[:, 0]
            eps_true = f["eps"][idx].astype(np.float32)

            np.savez_compressed(
                out / f"case{case_no:03d}.npz",
                eps=eps_pred.astype(np.float32),
                eps_true=eps_true.astype(np.float32),
                index=idx,
            )

            fig, ax = plt.subplots(K + 1, 1, figsize=(6, 2.6 * (K + 1)), constrained_layout=True)
            ax = np.atleast_1d(ax)

            im(ax[0], eps_true, f"true eps idx={idx}", "turbo")

            for i in range(K):
                im(ax[i + 1], eps_pred[i], f"sample {i} eps", "turbo")

            fig.savefig(out / f"case{case_no:03d}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    print(f"[sample] {out}", flush=True)

def mode_view(args):
    cfg=load_yaml(args.cfg); h5=Path(args.h5 or cfg["data"]["h5"]); out=Path(args.out or "runs/view.png")
    with h5py.File(h5,"r") as f:
        idx=np.where(f["valid"][:]==1)[0][:int(args.n)]
        items=[(i,{"subset":int(f["subset"][i]),"source":f["source"][i].decode() if isinstance(f["source"][i],bytes) else str(f["source"][i])},f["eps"][i],f["sigma"][i],f["bscan"][i]) for i in idx]
    plot_preview(items,out)

def main():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest="cmd",required=True)
    q=sub.add_parser("h5"); q.add_argument("--cfg",default="cfg.yaml"); q.add_argument("--src"); q.add_argument("--out"); q.add_argument("--device",default="cuda"); q.add_argument("--subsets",nargs="+",type=int); q.add_argument("--max-per-subset",type=int); q.add_argument("--overwrite",action="store_true"); q.add_argument("--resume",action="store_true")
    q=sub.add_parser("view"); q.add_argument("--cfg",default="cfg.yaml"); q.add_argument("--h5"); q.add_argument("--out"); q.add_argument("-n",type=int,default=6)
    q=sub.add_parser("train"); q.add_argument("--cfg",default="cfg.yaml")
    q=sub.add_parser("sample"); q.add_argument("--cfg",default="cfg.yaml"); q.add_argument("--ckpt"); q.add_argument("--split",default="val")
    a=p.parse_args()
    if a.cmd=="h5": mode_h5(a)
    elif a.cmd=="view": mode_view(a)
    elif a.cmd=="train": mode_train(a)
    elif a.cmd=="sample": mode_sample(a)

if __name__ == "__main__":
    main()
