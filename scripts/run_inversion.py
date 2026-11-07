"""Run EM-DPS or its unguided reference using manuscript settings."""
import argparse,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--method',choices=['em-dps','no-dps'],default='em-dps')
p.add_argument('--experiment',choices=['nominal','noise','mismatch'],default='nominal')
p.add_argument('--cfg',default='cfg.yaml');p.add_argument('--ckpt',default='runs/emdps/ckpt/best.pt')
p.add_argument('--out',required=True);p.add_argument('--cases',type=int,default=128);p.add_argument('--K',type=int,default=4)
p.add_argument('--snr',type=float,default=30);p.add_argument('--frequency-scale',type=float,default=.92)
a=p.parse_args();root=Path(__file__).resolve().parents[1]
module={'nominal':'methods.sampling','noise':'experiments.robustness.noise','mismatch':'experiments.robustness.mismatch'}[a.experiment]
cmd=[sys.executable,'-m',module,'--cfg',a.cfg,'--ckpt',a.ckpt,'--out',a.out,'--split','test','--cases',str(a.cases),'--K',str(a.K),'--steps','32','--sigma_min','.002','--sigma_max','5','--rho','7','--seed','123','--eps_min','1','--eps_max','35','--sigma_mode','h5_median','--sigma_stats_n','256','--dps_power','.5','--dps_max_update','.05','--dps_w_prior','0']
if a.method=='em-dps':cmd+=['--dps_start_frac','.45','--dps_every','4','--dps_lr','.02','--dps_w_mse','1','--dps_w_ncc','.05','--dps_w_tv','5e-5']
else:cmd+=['--dps_start_frac','2','--dps_every','999999','--dps_lr','0','--dps_w_mse','0','--dps_w_ncc','0','--dps_w_tv','0']
if a.experiment=='noise':cmd+=['--obs_noise_snr_db',str(a.snr),'--obs_noise_seed','2026']
if a.experiment=='mismatch':cmd+=['--guidance_freq_scale',str(a.frequency_scale),'--eval_freq_scale','1']
subprocess.run(cmd,cwd=root,check=True)
