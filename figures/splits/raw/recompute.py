import argparse,csv,gc,hashlib,json,os,sys,time,zipfile
from pathlib import Path
import h5py,numpy as np,torch
import torch.nn.functional as F
sys.path.insert(0,os.getcwd())
from methods import emdps
def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def cross_counts(hashes,split):
    groups={}
    for h,s in zip(hashes,split):
        if h not in groups:groups[h]=[0,0,0]
        groups[h][int(s)]+=1
    crossed=[v for v in groups.values() if sum(n>0 for n in v)>1]
    return dict(cross_split_groups=len(crossed),cross_split_pairs=sum(v[0]*v[1]+v[0]*v[2]+v[1]*v[2] for v in crossed))

def nearest(features,train,queries):

    device=torch.device('cuda');q=torch.from_numpy(features[queries].copy()).to(device)
    qnorm=(q*q).sum(1);nq=len(queries);best=torch.full((nq,4),float('inf'),device=device)
    ids=torch.full((nq,4),-1,dtype=torch.long,device=device)
    for start in range(0,len(train),1024):
        t=torch.from_numpy(features[train[start:start+1024]].copy()).to(device)
        tn=(t*t).sum(1)
        for j in range(0,nq,128):
            d=qnorm[j:j+128,None]+tn[None,:]-2*(q[j:j+128]@t.T)
            values,loc=torch.topk(d,4,dim=1,largest=False)
            values_all=torch.cat((best[j:j+128],values),dim=1)
            ids_all=torch.cat((ids[j:j+128],loc+start),dim=1)
            values_new,which=torch.topk(values_all,4,dim=1,largest=False)
            best[j:j+128]=values_new;ids[j:j+128]=torch.gather(ids_all,1,which)
        del t;torch.cuda.empty_cache()
    candidates=ids.cpu().numpy();dist=[];matched=[]
    del q,best,ids,qnorm;torch.cuda.empty_cache()
    for j,index in enumerate(queries):
        possible=train[candidates[j]]
        delta=features[possible].astype(np.float64)-features[index].astype(np.float64)
        rms=np.sqrt(np.mean(delta*delta,axis=1));k=int(np.argmin(rms))
        dist.append(float(rms[k]));matched.append(int(possible[k]))
    return np.asarray(dist),np.asarray(matched)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    cfg=emdps.load_yaml('runs/emdps/cfg.yaml')
    ckpt_path=Path('runs/emdps/ckpt/best.pt')
    ckpt=torch.load(ckpt_path,map_location='cpu',weights_only=False);stats=ckpt['stats'];del ckpt
    xm=float(np.asarray(stats['x_mean']).reshape(-1)[0]);xs=float(np.asarray(stats['x_std']).reshape(-1)[0])
    ym=float(stats['y_mean']);ys=float(stats['y_std'])
    path=Path(cfg['data']['h5']);start=time.time()
    with h5py.File(path,'r') as f:
        valid=np.flatnonzero(f['valid'][:]==1)
        splits=emdps.split_indices(len(valid),42,.8,.1)
        split=np.empty(len(valid),dtype=np.int8)
        for code,name in enumerate(['train','val','test']):split[splits[name]]=code
        assert [len(splits[k]) for k in ['train','val','test']]==[14400,1800,1800]
        subset=f['subset'][valid];sources=f['source'].asstr()[valid]
        feature_m=np.empty((len(valid),128*384),dtype=np.float32)
        feature_d=np.empty((len(valid),512*384),dtype=np.float32)
        eps_hash=[];pair_hash=[];keys=[]
        for lo in range(0,len(valid),32):
            indices=valid[lo:lo+32]
            eps=f['eps'][indices];sig=f['sigma'][indices];raw=f['bscan'][indices]
            processed=[]
            for j in range(len(indices)):
                eh=hashlib.sha256(eps[j].tobytes()).hexdigest();h=hashlib.sha256()
                for arr in [eps[j],sig[j],raw[j]]:h.update(arr.tobytes())
                eps_hash.append(eh);pair_hash.append(h.hexdigest())
                keys.append((int(subset[lo+j]),str(sources[lo+j])))
                processed.append(emdps.prep_bscan_np(raw[j].astype(np.float32),cfg))
            with torch.no_grad():
                m=F.interpolate(torch.from_numpy(eps.astype(np.float32))[:,None],size=(128,384),mode='bilinear',align_corners=False)[:,0].numpy()
                d=F.interpolate(torch.from_numpy(np.stack(processed))[:,None],size=(512,384),mode='bilinear',align_corners=False)[:,0].numpy()
            feature_m[lo:lo+len(indices)]=((m-xm)/xs).reshape(len(indices),-1)
            feature_d[lo:lo+len(indices)]=((d-ym)/ys).reshape(len(indices),-1)
            if lo%512==0:print('PREPROCESS',lo,len(valid),'seconds',round(time.time()-start,1),flush=True)
        composition=[]
        for code,name in enumerate(['train','val','test']):
            for s in sorted(np.unique(subset)):
                composition.append(dict(split=name,subset=int(s),count=int(np.sum((split==code)&(subset==s))),split_total=int(np.sum(split==code))))
    duplicate=dict(source_keys=cross_counts(keys,split),permittivity=cross_counts(eps_hash,split),complete_pairs=cross_counts(pair_hash,split))
    write_csv(out/'composition.csv',composition)
    write_csv(out/'split_membership.csv',[dict(h5_index=int(i),split=['train','val','test'][int(split[j])],subset=int(subset[j]),source=str(sources[j]),eps_sha256=eps_hash[j],pair_sha256=pair_hash[j]) for j,i in enumerate(valid)])
    np.savez_compressed(out/'split_indices.npz',**{name:valid[idx] for name,idx in splits.items()})
    queries=np.concatenate((splits['val'],splits['test']));train=splits['train']
    print('FEATURES_READY','seconds',round(time.time()-start,1),flush=True)
    dm,im=nearest(feature_m,train,queries);del feature_m;gc.collect()
    print('MODEL_NN_DONE','seconds',round(time.time()-start,1),flush=True)
    dd,idd=nearest(feature_d,train,queries);del feature_d;gc.collect()
    result=[]
    for j,q in enumerate(queries):
        result.append(dict(h5_index=int(valid[q]),split='val' if j<len(splits['val']) else 'test',model_nrmse=float(dm[j]),model_nearest_h5_index=int(valid[im[j]]),data_nrmse=float(dd[j]),data_nearest_h5_index=int(valid[idd[j]])))
    write_csv(out/'nearest_training.csv',result)
    report=dict(valid_cases=len(valid),seed=42,sizes={k:len(v) for k,v in splits.items()},main_cases=128,repeated_cases=64,duplicate_counts=duplicate,
        normalization=dict(model_mean=xm,model_std=xs,data_mean=ym,data_std=ys,source='Training statistics stored in the original best checkpoint'),
        model_grid=[128,384],data_grid=[512,384],
        metric='For each validation/test sample: min over every training sample of sqrt(mean((standardized query - standardized reference)^2)). One scalar training std per space; no per-sample normalization beyond the original processed-B-scan operator.',
        nearest_search='Exhaustive float32 full-grid matrix distances with TF32 disabled; the four closest candidates are refined by direct float64 squared differences.',
        duplicate_definition='Cross-split comparisons on original stored arrays; complete pair is eps + sigma + raw bscan. Source key is (subset, source).',
        provenance=dict(h5_path=str(path),h5_bytes=path.stat().st_size,checkpoint_sha256=hashlib.sha256(ckpt_path.read_bytes()).hexdigest(),emdps_sha256=hashlib.sha256(Path('methods/emdps.py').read_bytes()).hexdigest(),prep_sha256=hashlib.sha256(Path('preprocessing.py').read_bytes()).hexdigest()),
        recomputation_note='Original audit script/data were unavailable. This audit was recomputed with explicit full-grid metric definitions; its numeric curves are not asserted identical to the old PNG.',
        elapsed_seconds=time.time()-start)
    (out/'report.json').write_text(json.dumps(report,indent=2))
    files=[out/n for n in ['composition.csv','split_membership.csv','split_indices.npz','nearest_training.csv','report.json']]+[Path(__file__)]
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));files.append(out/'manifest.json')
    with zipfile.ZipFile(out/'audit_results.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,arcname=p.name)
    (out/'DONE').write_text('complete');print('ALL_DONE',json.dumps(report),flush=True)

if __name__=='__main__':main()
