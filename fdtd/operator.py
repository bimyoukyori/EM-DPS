"""EM-DPS differentiable electromagnetic forward operator."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import compute

@dataclass(frozen=True)
class FDTDConfig:
    dx: float
    nx: int
    ny: int
    nz: int
    freq: float
    time_window: float
    pmlthick: int
    nsrc: int
    nrx: int
    scan_steps: int
    source: list[Any]
    source_amp: list[Any]
    source_location: list[float]
    source_step: list[float]
    receiver_location: list[float]
    receiver_step: list[float]
    mr_default: float = 1.0
    dt: float | None = None
    pml_formulation: str = "HORIPML"
    fixed_receiver: bool = False
    air_cells: int = 0
    air_eps: float = 1.0
    air_sigma: float = 0.0

    @staticmethod
    def from_dict(sim: dict) -> "FDTDConfig":
        nz = int(sim.get("nz", 1))
        if nz < 1:
            nz = 1
        return FDTDConfig(
            dx=float(sim["dx"]),
            nx=int(sim["nx"]),
            ny=int(sim["ny"]),
            nz=nz,
            freq=float(sim["freq"]),
            time_window=float(sim["time_window"]),
            pmlthick=int(sim.get("pmlthick", 10)),
            nsrc=int(sim.get("nsrc", 1)),
            nrx=int(sim["nrx"]),
            scan_steps=int(sim.get("scan_steps", sim.get("step", 1))),
            source=list(sim.get("source", ["hertzian", "z"])),
            source_amp=list(sim.get("source_amp", ["ricker", 1.0, sim["freq"]])),
            source_location=list(sim["source_location"]),
            source_step=list(sim["source_step"]),
            receiver_location=list(sim["receiver_location"]),
            receiver_step=list(sim["receiver_step"]),
            mr_default=float(sim.get("mr_default", 1.0)),
            dt=(None if sim.get("dt", None) is None else float(sim["dt"])),
            pml_formulation=str(sim.get("pml_formulation", "HORIPML")),
            fixed_receiver=bool(sim.get("fixed_receiver", False)),
            air_cells=int(sim.get("air_cells", 0)),
            air_eps=float(sim.get("air_eps", 1.0)),
            air_sigma=float(sim.get("air_sigma", 0.0)),
        )

def _to_vec3(values: list[float], device: torch.device) -> torch.Tensor:
    if len(values) != 3:
        raise ValueError(f"expected a 3-vector, got {values!r}")
    return torch.tensor(values, dtype=torch.float32, device=device)

def format_bscan(raw: torch.Tensor, receiver_component_index: int = 2) -> torch.Tensor:

    x = raw
    if x.ndim == 4:
        idx = max(0, min(int(receiver_component_index), x.shape[1] - 1))
        x = x[:, idx]
    if x.ndim == 2:
        return x
    if x.ndim != 3:
        raise ValueError(f"expected raw FDTD output with 2-4 dims, got {tuple(x.shape)}")
    
    scan_steps, time_steps, nrx = x.shape
    if scan_steps == 1:
        return x[0]
    if nrx == 1:
        return x[:, :, 0].transpose(0, 1).contiguous()
    return x.permute(1, 0, 2).reshape(time_steps, scan_steps * nrx).contiguous()

class DifferentiableGPRFDTD(nn.Module):

    def __init__(self, sim: dict, device: torch.device | str | None = None):
        super().__init__()
        self.cfg = FDTDConfig.from_dict(sim)
        self.receiver_component_index = int(sim.get("receiver_component_index", 2))
        self.device_arg = device

    def _prepare_param(self, x: torch.Tensor, device: torch.device, *, kind: str) -> torch.Tensor:
        x = x.to(device=device, dtype=torch.float32)
        if x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 3:
            x = x.unsqueeze(1)
        elif x.ndim != 4:
            raise ValueError(f"{kind} must have 2-4 dims, got {tuple(x.shape)}")
        x = F.interpolate(x, size=(self.cfg.nx, self.cfg.ny), mode="bilinear", align_corners=False)

        air_cells = int(getattr(self.cfg, "air_cells", 0))
        if air_cells > 0:
            air_cells = max(0, min(air_cells, x.shape[-2]))
            if kind == "epsilon_r":
                x[..., :air_cells, :] = float(getattr(self.cfg, "air_eps", 1.0))
            else:
                x[..., :air_cells, :] = float(getattr(self.cfg, "air_sigma", 0.0))

        if kind == "epsilon_r":
            x = x.clamp(min=1.0)
        else:
            x = x.clamp(min=0.0)
        return x

    def forward(self, epsilon_r: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        if self.device_arg is None:
            device = epsilon_r.device
        else:
            device = torch.device(self.device_arg)
        eps = self._prepare_param(epsilon_r, device, kind="epsilon_r")
        sig = self._prepare_param(sigma, device, kind="sigma")
        if eps.shape[0] != sig.shape[0]:
            raise ValueError("epsilon_r and sigma batch sizes differ")

        c = self.cfg
        source_location = _to_vec3(c.source_location, device)
        source_step = _to_vec3(c.source_step, device)
        receiver_location = _to_vec3(c.receiver_location, device)
        receiver_step = _to_vec3(c.receiver_step, device)
        time_window = torch.tensor(c.time_window, dtype=torch.float32, device=device)

        outs = []
        for b in range(eps.shape[0]):
            er3 = eps[b, 0].unsqueeze(-1).repeat(1, 1, c.nz).contiguous()
            se3 = sig[b, 0].unsqueeze(-1).repeat(1, 1, c.nz).contiguous()
            mr = torch.full_like(er3, c.mr_default)
            raw, _dt = compute(
                device=device,
                dx=c.dx,
                ny=c.ny,
                nx=c.nx,
                nz=c.nz,
                source=c.source,
                source_apmlitudes=c.source_amp,
                step=c.scan_steps,
                er=er3,
                se=se3,
                mr=mr,
                nsrc=c.nsrc,
                nrx=c.nrx,
                time_windows=time_window,
                source_location=source_location,
                source_step=source_step,
                receiver_location=receiver_location,
                receiver_step=receiver_step,
                pmlthick=c.pmlthick,
                freq=c.freq,
                dt=c.dt,
                pml_formulation=c.pml_formulation,
                tv=False,
                Fixedreceiver=c.fixed_receiver,
            )
            outs.append(format_bscan(raw, self.receiver_component_index).unsqueeze(0).unsqueeze(0))
        return torch.cat(outs, dim=0)
