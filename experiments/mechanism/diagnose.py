import argparse,csv,gc,hashlib,json,os,sys,time,zipfile
from pathlib import Path
import numpy as np
import torch
import h5py
sys.path.insert(0,os.getcwd())
from methods import emdps
from methods import sampling as ref
CASES=[10,24]
SLOTS=[0,16,28]
METHODS=['No-DPS','Early','Uniform','Late']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    original=Path('runs/scheduling')
    cfg=emdps.load_yaml('runs/emdps/cfg.yaml')
    device=torch.device('cuda')
    ckpt=torch.load('runs/emdps/ckpt/best.pt',map_location=device,weights_only=False)
    stats=emdps.Stats(**ckpt['stats']);model=emdps.make_model(cfg,device)
    model.load_state_dict(ckpt['model'],strict=True);model.eval()
    for p in model.parameters():p.requires_grad_(False)
    ds=emdps.H5Set(cfg,'test',stats);op=emdps.make_fdtd(cfg,device)
    levels=ref.sigma_schedule(.002,5.,32,7.,device)
    H,W=cfg['model']['x_resolution'];s=float(np.asarray(stats.x_std).reshape(-1)[0])
    sigma_value=ref.estimate_train_sigma_median(cfg,n=256)
    sigma_np=np.full((H,W),sigma_value,dtype=np.float32)
    sigma=torch.from_numpy(sigma_np).to(device)
    eta=.004472135988319589
    metrics=list(csv.DictReader((original/'metrics.csv').open()))
    baseline=sorted((r for r in metrics if r['method']=='No-DPS'),key=lambda r:float(r['fdtd_mse']))
    assert [int(baseline[r]['case']) for r in [15,23]]==CASES
    protocol=dict(cases=CASES,selection='Ascending No-DPS MSE ranks 16 and 24 of 32 (one-based); selected without looking at guided gains.',
        isolated_slots=SLOTS,isolated_update_size=eta,seed=123,
        counterfactual='One correction at the selected step; all remaining steps unguided. Same baseline trajectory and initial noise.',
        spatial_comparison='Previously saved four-correction schedule reconstructions, no reranking or retraining.',
        diagnostics_not_timing='Additional forward solves are diagnostic measurements and excluded from the original budget/runtime comparison.',
        field_extent=dict(x_m=[0.,cfg['fdtd']['domain_x']],depth_m=[0.,cfg['fdtd']['domain_z']]),
        input_metric_sha256=hashlib.sha256((original/'metrics.csv').read_bytes()).hexdigest())
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[];source_checks=[]

    def clean(x,noise):
        with torch.no_grad():
            den=model(x,torch.full([1],float(max(noise,.002)),device=device),condition)
            den=ref.clamp_norm_from_eps(den,stats,1.,35.)
            return ref.denorm_x_t(den,stats)[0,0].cpu().numpy().astype(np.float32)

    def euler(x,i):
        with torch.no_grad():
            den=model(x,torch.full([1],float(levels[i]),device=device),condition)
            nxt=x+(levels[i+1]-levels[i])*(x-den)/levels[i].clamp_min(1e-12)
            return ref.clamp_norm_from_eps(nxt,stats,1.,35.).detach()

    def sim(eps):
        score,sims=ref.eval_candidates_official(op,cfg,eps[None],sigma_np,obs_e,device)
        return score[0],sims[0].astype(np.float32)

    with h5py.File(cfg['data']['h5'],'r') as h5:
        for case in CASES:
            old=np.load(original/'samples'/f'case{case:03d}.npz')
            b=ds[case];index=int(b['index']);condition=b['y'][None].to(device)
            raw_np=h5['bscan'][index].astype(np.float32)
            obs_e=emdps.prep_bscan_np(raw_np,cfg).astype(np.float32)
            raw=torch.from_numpy(raw_np).to(device)
            tmp=ref.torch_prep_bscan(raw,cfg,obs_scale=None).detach()
            scale=torch.quantile(tmp.abs().flatten(),.99).clamp_min(1e-6)
            obs_g=ref.torch_prep_bscan(raw,cfg,obs_scale=scale).detach()
            arrays=dict(index=index,eps_true=old['eps_true'],obs_e=obs_e)
            spatial=[]
            for method in METHODS:
                key=method.replace('-','_').lower();eps=old[key]
                score,forward=sim(eps)
                source=next(r for r in metrics if int(r['case'])==case and r['method']==method)
                diff=abs(score['fdtd_mse']-float(source['fdtd_mse']))
                assert diff<1e-7,(case,method,diff)
                arrays['eps_'+key]=eps;arrays['sim_'+key]=forward
                arrays['res_'+key]=forward-obs_e
                spatial.append(dict(case=case,index=index,method=method,**score))
                source_checks.append(dict(case=case,method=method,abs_mse_difference=diff))
            with (out/f'case{case:03d}_spatial_metrics.json').open('w') as f:json.dump(spatial,f,indent=2)
            np.savez_compressed(out/f'case{case:03d}_spatial.npz',**arrays)
            print('SPATIAL_DONE',case,flush=True)

            torch.cuda.manual_seed_all(123)
            for _ in range(case+1):x0=torch.randn(1,1,H,W,device=device)*levels[0]
            x=x0.clone();states={}
            for i in range(32):
                x=euler(x,i)
                if i in SLOTS:states[i]=x.clone()
            eps_base=clean(x,.002)
            delta=float(np.max(np.abs(eps_base-old['no_dps'])))
            assert delta<1e-5,(case,delta)
            isolated=dict(index=index,eps_true=old['eps_true'],eps_baseline=eps_base,
                          obs_e=obs_e,sim_baseline=arrays['sim_no_dps'])
            base_mse=float(spatial[0]['fdtd_mse'])
            for slot in SLOTS:
                state=states[slot];noise=float(max(levels[slot+1],.002))
                m_before=clean(state,noise)
                m=torch.from_numpy(m_before).to(device).detach().clone().requires_grad_(True)
                pred=ref.torch_prep_bscan(ref.call_fdtd_torch(op,m,sigma,device),cfg,obs_scale=scale)
                loss=((pred-obs_g)**2).mean()+.05*(1-ref.norm_corr_t(pred,obs_g))+5e-5*ref.tv_loss(m)
                loss.backward()
                grad=torch.nan_to_num(m.grad.detach(),nan=0.,posinf=0.,neginf=0.)[None,None]*s
                grad=grad/grad.abs().mean().clamp_min(1e-8)
                update=(eta*grad).clamp(-.05,.05)
                with torch.no_grad():perturbed=ref.clamp_norm_from_eps(state-update,stats,1.,35.)
                m_after=clean(perturbed,noise)
                final_state=perturbed.detach()
                for i in range(slot+1,32):final_state=euler(final_state,i)
                eps_final=clean(final_state,.002)
                score_before,_=sim(m_before);score_after,_=sim(m_after)
                score_final,sim_final=sim(eps_final)
                immediate=m_after-m_before;retained=eps_final-eps_base
                norms=[float(np.linalg.norm(a)) for a in [immediate,retained]]
                cosine=float(np.sum(immediate*retained)/(norms[0]*norms[1]+1e-20))
                record=dict(case=case,index=index,slot=slot,noise=noise,
                    before_mse=score_before['fdtd_mse'],immediate_mse=score_after['fdtd_mse'],
                    baseline_mse=base_mse,final_mse=score_final['fdtd_mse'],
                    immediate_l2=norms[0],retained_l2=norms[1],
                    norm_ratio=norms[1]/max(norms[0],1e-20),spatial_cosine=cosine)
                rows.append(record)
                for name,value in dict(m_before=m_before,m_after=m_after,immediate_delta=immediate,
                                       final_delta=retained,eps_final=eps_final,sim_final=sim_final).items():
                    isolated[f'{name}_s{slot:02d}']=value
                print('ISOLATED_DONE',json.dumps(record),flush=True)
                del m,pred,loss,grad,update,perturbed,final_state
                gc.collect();torch.cuda.empty_cache()
            np.savez_compressed(out/f'case{case:03d}_isolated.npz',**isolated)
            with (out/'isolated_metrics.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            del states,x,x0,arrays,isolated;gc.collect();torch.cuda.empty_cache()
    (out/'source_regression.json').write_text(json.dumps(source_checks,indent=2))
    files=[p for p in out.iterdir() if p.is_file() and p.suffix in ['.npz','.json','.csv']]
    source=Path(__file__);files.append(source)
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));files.append(out/'manifest.json')
    with zipfile.ZipFile(out/'mechanism_results.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,arcname=p.name)
    (out/'DONE').write_text(str(time.time()));print('ALL_DONE',flush=True)

if __name__=='__main__':main()
