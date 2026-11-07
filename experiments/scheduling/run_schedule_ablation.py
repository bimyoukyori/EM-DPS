import argparse,csv,gc,hashlib,json,os,sys,time
from pathlib import Path
import h5py
import numpy as np
import torch
sys.path.insert(0,os.getcwd())
from methods import emdps
from methods import sampling as ref
SCHEDULES={'Early':[0,4,8,12],'Uniform':[0,10,20,30],'Late':[16,20,24,28]}

def read_csv(p):
    if not p.exists():return []
    with p.open(newline='') as f:return list(csv.DictReader(f))

def write_csv(p,rows):
    if not rows:return
    t=p.with_suffix('.tmp')
    with t.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    t.replace(p)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',required=True)
    parser.add_argument('--cases',type=int,default=32)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);(out/'samples').mkdir(exist_ok=True)
    cfg_file=Path('runs/emdps/cfg.yaml')
    cfg=emdps.load_yaml(str(cfg_file));device=torch.device('cuda')
    torch.manual_seed(123);np.random.seed(123)
    ckpt_path=Path('runs/emdps/ckpt/best.pt')
    ckpt=torch.load(ckpt_path,map_location=device,weights_only=False)
    stats=emdps.Stats(**ckpt['stats'])
    model=emdps.make_model(cfg,device);model.load_state_dict(ckpt['model'],strict=True);model.eval()
    for p in model.parameters():p.requires_grad_(False)
    ds=emdps.H5Set(cfg,'test',stats);op=emdps.make_fdtd(cfg,device)
    s=float(np.asarray(stats.x_std).reshape(-1)[0]);mu=float(np.asarray(stats.x_mean).reshape(-1)[0])
    H,W=cfg['model']['x_resolution']
    sigma_value=ref.estimate_train_sigma_median(cfg,n=256)
    sigma_np=np.full((H,W),sigma_value,dtype=np.float32);sigma=torch.from_numpy(sigma_np).to(device)
    levels=ref.sigma_schedule(.002,5.,32,7.,device)
    eta_values=[.02*float((levels[i+1]/levels[0]).clamp(.05,1))**.5 for i in SCHEDULES['Late']]
    assert np.allclose(eta_values,eta_values[0])
    protocol=dict(cases=args.cases,case_selection='First 32 held-out cases in existing fixed test order; set before running schedule outcomes.',
        K=1,seed=123,steps=32,schedules=SCHEDULES,ordered_update_sizes=eta_values,
        dose_control='The same four ordered update sizes, taken from the original late schedule, are used for every guided treatment. No schedule-specific tuning.',
        objective=dict(mse=1.,decorrelation=.05,tv=5e-5),normalized_step_clipping=.05,eps_bounds=[1.,35.],
        physical_budget_per_guided_case=dict(forward_backward=4,evaluation_forward=1),
        rationale='K=1 removes multi-candidate selection; equal calls and update sizes isolate placement along the denoising trajectory.',
        inference='Observation conditioning, trained denoiser, local physical gradient, preprocessing and final evaluation unchanged.',
        sigma_const=float(sigma_value),x_mean=mu,x_std=s,
        timing='Exclusive RTX 3090; CUDA-synchronized generation plus physical correction plus final evaluation; setup, warm-up and file I/O excluded.',
        source_sha256={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in ['methods/emdps.py','methods/sampling.py','methods/physics.py',str(cfg_file)]},
        checkpoint_sha256=hashlib.sha256(ckpt_path.read_bytes()).hexdigest(),torch_version=torch.__version__,gpu=torch.cuda.get_device_name(device))
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=read_csv(out/'metrics.csv') if args.resume else []
    logs=read_csv(out/'physical_steps.csv') if args.resume else []
    done={int(r['case']) for r in rows if r['method']=='Late'}
    def sample(x0,condition,obs_g,scale,slots):
        x=x0.clone();steps=[];calls=0
        slot_set=set(slots)
        for i in range(32):
            with torch.no_grad():
                den=model(x,torch.full([1],float(levels[i]),device=device),condition)
                nxt=x+(levels[i+1]-levels[i])*(x-den)/levels[i].clamp_min(1e-12)
                nxt=ref.clamp_norm_from_eps(nxt,stats,1.,35.)
            if i in slot_set:
                j=slots.index(i)
                with torch.no_grad():
                    den_g=model(nxt,torch.full([1],float(max(levels[i+1],.002)),device=device),condition)
                    den_g=ref.clamp_norm_from_eps(den_g,stats,1.,35.)
                m=ref.denorm_x_t(den_g,stats)[0,0].detach().clone().clamp(1.,35.).requires_grad_(True)
                sim=ref.call_fdtd_torch(op,m,sigma,device);calls+=1
                pred=ref.torch_prep_bscan(sim,cfg,obs_scale=scale)
                mse=((pred-obs_g)**2).mean();corr=ref.norm_corr_t(pred,obs_g)
                loss=mse+.05*(1-corr)+5e-5*ref.tv_loss(m);loss.backward()
                grad=torch.nan_to_num(m.grad.detach(),nan=0.,posinf=0.,neginf=0.)[None,None]*s
                grad=grad/grad.abs().mean().clamp_min(1e-8)
                eta=eta_values[j];update=(eta*grad).clamp(-.05,.05)
                with torch.no_grad():nxt=ref.clamp_norm_from_eps(nxt-update,stats,1.,35.)
                steps.append(dict(step=i,stage=j,eta=eta,guidance_mse=float(mse.detach()),guidance_corr=float(corr.detach()),guidance_loss=float(loss.detach())))
                del den_g,m,sim,pred,mse,corr,loss,grad,update
                gc.collect();torch.cuda.empty_cache()
            x=nxt.detach();torch.cuda.empty_cache()
        with torch.no_grad():
            final=model(x,torch.full([1],.002,device=device),condition)
            final=ref.clamp_norm_from_eps(final,stats,1.,35.)
            eps=ref.denorm_x_t(final,stats)[:,0].cpu().numpy().astype(np.float32)
        gc.collect();torch.cuda.empty_cache()
        assert calls==len(slots)
        return eps,steps,calls

    with h5py.File(cfg['data']['h5'],'r') as h5:
        b=ds[0];y=b['y'][None].to(device)
        raw=torch.from_numpy(h5['bscan'][int(b['index'])].astype(np.float32)).to(device)
        with torch.no_grad():model(torch.zeros(1,1,H,W,device=device),torch.ones(1,device=device),y)
        prep=ref.torch_prep_bscan(raw,cfg,obs_scale=None).detach()
        scale=torch.quantile(prep.abs().flatten(),.99).clamp_min(1e-6)
        m=torch.full((H,W),mu,device=device,requires_grad=True)
        pred=ref.call_fdtd_torch(op,m,sigma,device)
        ref.torch_prep_bscan(pred,cfg,obs_scale=scale).square().mean().backward()
        ref.official_forward_np(op,cfg,np.full((H,W),mu,dtype=np.float32),sigma_np,device)
        del m,pred,prep,raw,y;gc.collect();torch.cuda.empty_cache()
        torch.cuda.manual_seed_all(123)
        start=time.perf_counter()
        for case in range(args.cases):
            
            x0=torch.randn(1,1,H,W,device=device)*levels[0]
            if case in done:continue
            b=ds[case];index=int(b['index']);condition=b['y'][None].to(device)
            raw_np=h5['bscan'][index].astype(np.float32);raw=torch.from_numpy(raw_np).to(device)
            tmp=ref.torch_prep_bscan(raw,cfg,obs_scale=None).detach()
            scale=torch.quantile(tmp.abs().flatten(),.99).clamp_min(1e-6)
            obs_g=ref.torch_prep_bscan(raw,cfg,obs_scale=scale).detach()
            obs_e=emdps.prep_bscan_np(raw_np,cfg).astype(np.float32)
            true=ref.denorm_x_t(b['x'][None].to(device),stats)[0,0].cpu().numpy()
            case_fields={};case_metrics={}
            treatments=[('No-DPS',[]),*SCHEDULES.items()]
            
            shift=case%4;treatments=treatments[shift:]+treatments[:shift]
            for method,slots in treatments:
                torch.cuda.synchronize();t0=time.perf_counter();rng_state=torch.cuda.get_rng_state()
                eps,trace,calls=sample(x0,condition,obs_g,scale,slots)
                assert torch.equal(rng_state,torch.cuda.get_rng_state()),'sampling consumed additional random draws'
                score,_=ref.eval_candidates_official(op,cfg,eps,sigma_np,obs_e,device)
                torch.cuda.synchronize();seconds=time.perf_counter()-t0
                delta=eps[0]-true
                row=dict(case=case,index=index,method=method,fdtd_mse=score[0]['fdtd_mse'],fdtd_mae=score[0]['fdtd_mae'],fdtd_corr=score[0]['fdtd_corr'],model_rmse=float(np.sqrt(np.mean(delta**2))),model_mae=float(np.mean(np.abs(delta))),seconds=seconds,physical_forward_backward_calls=calls,evaluation_forward_calls=1)
                rows.append(row);case_fields[method.replace('-','_').lower()]=eps[0]
                case_metrics[method]=row['fdtd_mse']
                for step in trace:step.update(case=case,index=index,method=method);logs.append(step)
            np.savez_compressed(out/'samples'/f'case{case:03d}.npz',index=index,eps_true=true,**case_fields)
            write_csv(out/'metrics.csv',rows);write_csv(out/'physical_steps.csv',logs)
            groups={name:[float(r['fdtd_mse']) for r in rows if r['method']==name] for name in ['No-DPS',*SCHEDULES]}
            status=dict(completed_cases=case+1,target_cases=args.cases,last_case_mse=case_metrics,mean_mse={name:float(np.mean(values)) for name,values in groups.items()},elapsed_seconds=time.perf_counter()-start)
            temp=out/'status.tmp';temp.write_text(json.dumps(status,indent=2));temp.replace(out/'status.json')
            print('CASE_COMPLETE',json.dumps(status),flush=True)
            del x0,condition,raw,tmp,obs_g;gc.collect();torch.cuda.empty_cache()
    (out/'DONE').write_text(str(time.time()));print('ALL_DONE',args.cases,flush=True)

if __name__=='__main__':main()
