"""Check forward output and finite permittivity gradients on a CUDA device."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from methods import emdps
cfg=emdps.load_yaml('configs/emdps.yaml')
assert torch.cuda.is_available(),'CUDA GPU required'
op=emdps.make_fdtd(cfg,torch.device('cuda'))
m=torch.full((1,1,128,384),6.,device='cuda',requires_grad=True)
s=torch.full_like(m,.0391)
y=op(m,s);loss=y.square().mean();loss.backward()
assert torch.isfinite(y).all() and m.grad is not None and torch.isfinite(m.grad).all()
print('FDTD forward/backward smoke check passed:',tuple(y.shape))
