"""Prepare the EM-DPS configuration for an existing HDF5 dataset."""
import argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--h5',required=True);a=p.parse_args()
import yaml
root=Path(__file__).resolve().parents[1]
cfg=yaml.safe_load((root/'configs/emdps.yaml').read_text())
cfg['data']['h5']=str(Path(a.h5).resolve())
for target in [root/'cfg.yaml',root/'runs/emdps/cfg.yaml']:
 target.parent.mkdir(parents=True,exist_ok=True);target.write_text(yaml.safe_dump(cfg,sort_keys=False))
print('Configuration prepared. Place best.pt in runs/emdps/ckpt/.')
