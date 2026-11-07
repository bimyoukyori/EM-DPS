import csv,gc,hashlib,json,os,sys,zipfile
from pathlib import Path
import numpy as np
import torch,h5py
sys.path.insert(0,os.getcwd())
from methods import emdps
from methods import sampling as ref
out=Path('runs/mechanism')
cfg=emdps.load_yaml('runs/emdps/cfg.yaml');device=torch.device('cuda')
ckpt=torch.load('runs/emdps/ckpt/best.pt',map_location=device,weights_only=False)
stats=emdps.Stats(**ckpt['stats']);model=emdps.make_model(cfg,device)
model.load_state_dict(ckpt['model'],strict=True);model.eval()
for p in model.parameters():p.requires_grad_(False)
ds=emdps.H5Set(cfg,'test',stats);op=emdps.make_fdtd(cfg,device)
levels=ref.sigma_schedule(.002,5.,32,7.,device)
H,W=cfg['model']['x_resolution'];s=float(np.asarray(stats.x_std).reshape(-1)[0])
sigma=torch.full((H,W),ref.estimate_train_sigma_median(cfg,n=256),device=device)
eta=.004472135988319589;slots=[0,16,28];records=[]
with h5py.File(cfg['data']['h5'],'r') as h5:
    for case in [10,24]:
        name=out/f'case{case:03d}_isolated.npz'
        with np.load(name) as data:arrays={k:data[k] for k in data.files}
        b=ds[case];condition=b['y'][None].to(device)
        raw=torch.from_numpy(h5['bscan'][int(b['index'])].astype(np.float32)).to(device)
        tmp=ref.torch_prep_bscan(raw,cfg,obs_scale=None).detach()
        scale=torch.quantile(tmp.abs().flatten(),.99).clamp_min(1e-6)
        obs=ref.torch_prep_bscan(raw,cfg,obs_scale=scale).detach()
        torch.cuda.manual_seed_all(123)
        for _ in range(case+1):x=torch.randn(1,1,H,W,device=device)*levels[0]
        for i in range(32):
            with torch.no_grad():
                den=model(x,torch.full([1],float(levels[i]),device=device),condition)
                x=ref.clamp_norm_from_eps(x+(levels[i+1]-levels[i])*(x-den)/levels[i].clamp_min(1e-12),stats,1.,35.)
            if i not in slots:continue
            with torch.no_grad():
                den_g=model(x,torch.full([1],float(max(levels[i+1],.002)),device=device),condition)
                den_g=ref.clamp_norm_from_eps(den_g,stats,1.,35.)
                estimate=ref.denorm_x_t(den_g,stats)[0,0]
            assert np.max(np.abs(estimate.cpu().numpy()-arrays[f'm_before_s{i:02d}']))<1e-5
            m=estimate.detach().clone().requires_grad_(True)
            pred=ref.torch_prep_bscan(ref.call_fdtd_torch(op,m,sigma,device),cfg,obs_scale=scale)
            objective=((pred-obs)**2).mean()+.05*(1-ref.norm_corr_t(pred,obs))+5e-5*ref.tv_loss(m)
            objective.backward()
            grad=torch.nan_to_num(m.grad.detach(),nan=0.,posinf=0.,neginf=0.)[None,None]*s
            grad=grad/grad.abs().mean().clamp_min(1e-8)
            update=(eta*grad).clamp(-.05,.05)
            with torch.no_grad():nxt=ref.clamp_norm_from_eps(x-update,stats,1.,35.)
            state_delta=(s*(nxt-x))[0,0].cpu().numpy()
            arrays[f'state_delta_s{i:02d}']=state_delta.astype(np.float32)
            immediate=arrays[f'immediate_delta_s{i:02d}'];final=arrays[f'final_delta_s{i:02d}']
            state_norm=float(np.linalg.norm(state_delta))
            records.append(dict(case=case,slot=i,state_l2=state_norm,
                immediate_to_state_norm=float(np.linalg.norm(immediate))/state_norm,
                final_to_state_norm=float(np.linalg.norm(final))/state_norm,
                immediate_final_cosine=float(np.sum(immediate*final)/(np.linalg.norm(immediate)*np.linalg.norm(final)))))
            del den_g,m,pred,objective,grad,update,nxt;gc.collect();torch.cuda.empty_cache()
        np.savez_compressed(name,**arrays);print('STATE_MAPS_DONE',case,flush=True)
with (out/'state_response_metrics.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
protocol=json.loads((out/'protocol.json').read_text())
protocol['state_map_definition']='training standard deviation times (projected updated normalized state minus original state), after clipping and bounds.'
(out/'protocol.json').write_text(json.dumps(protocol,indent=2))
files=[p for p in out.iterdir() if p.is_file() and p.suffix in ['.npz','.csv','.json'] and p.name!='manifest.json']
files += [Path(__file__),Path('experiments/mechanism/diagnose.py')]
manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2));files.append(out/'manifest.json')
with zipfile.ZipFile(out/'mechanism_results_v2.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in files:z.write(p,arcname=p.name)
(out/'V2_DONE').write_text('complete');print(json.dumps(records),flush=True)
