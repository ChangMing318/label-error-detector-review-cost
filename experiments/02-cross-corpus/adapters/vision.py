"""Remote backbone stress test on the previously frozen benchmark inputs."""
import argparse,csv,hashlib,json,os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import resnet18
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression

HERE=Path(__file__).resolve().parent
H=Path(os.environ.get('CIFAR_H_INPUT',Path.home()/'icde-cifar10h-7gbUcb'))
P=json.loads((HERE/'vision_protocol.json').read_text())
MENU=P['candidate_order']
ADDED=MENU[5:]
WEIGHTS=Path(os.environ.get('RN18_WEIGHTS',Path.home()/'.cache/torch/hub/checkpoints/resnet18-f37072fd.pth'))
DEVICE='cuda'

def digest(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def read_csv(p):
    with p.open(newline='') as f:
        return list(csv.DictReader(f))

def write_csv(p,rows):
    with p.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)

@torch.inference_mode()
def embedding(model,images):
    mean=torch.tensor([.485,.456,.406],device=DEVICE)[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device=DEVICE)[None,:,None,None]
    out=[]
    for a in range(0,len(images),192):
        x=torch.as_tensor(images[a:a+192],device=DEVICE).float()/255
        x=F.interpolate(x,size=(P['resolution'],P['resolution']),mode='bilinear',align_corners=False)
        x=((x-mean)/std).contiguous(memory_format=torch.channels_last)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            z=model(x)
        out.append(F.normalize(z.float(),dim=1).cpu().numpy())
    return np.concatenate(out)

def score(head,ref,y,z,ncls):
    pred=head.decision_function(z)
    nl=logsumexp(pred,axis=1,keepdims=True)-pred
    centers=np.stack([ref[y==i].mean(0) for i in range(ncls)])
    centers=centers/np.maximum(np.linalg.norm(centers,axis=1,keepdims=True),1e-12)
    similarities=z@centers.T
    r=torch.as_tensor(ref,device=DEVICE)
    labels=torch.as_tensor(y,device=DEVICE)
    neighbors=[]
    for a in range(0,len(z),384):
        sim=torch.as_tensor(z[a:a+384],device=DEVICE)@r.T
        idx=torch.argsort(sim,dim=1,descending=True,stable=True)[:,:P['knn']]
        neighbors.append(labels[idx].cpu().numpy())
    nbr=np.concatenate(neighbors)
    disagreement=P['knn']-(nbr[:,:,None]==np.arange(ncls)[None,None,:]).sum(1)
    return dict(zip(ADDED,(nl,similarities.max(1,keepdims=True)-similarities,disagreement)))

def counts(mat,obs,y,ties,k):
    v=mat[np.arange(len(y)),obs]
    order=ties[np.argsort(-v[ties],kind='stable')][:k]
    return int((obs[order]!=y[order]).sum())

def evaluate(dataset,model,out):
    manifest=json.loads((HERE/f'{dataset}_input_sha256.json').read_text())
    for name,obj in manifest.items():
        assert digest(HERE/name)==obj['sha256']
    with np.load(HERE/f'{dataset}_pixels.npz') as q:
        y=q['y']; fit=q['fit']; rest=q['rest']
        assert len(np.unique(np.r_[fit,rest]))==len(y)==50000
        features=embedding(model,q['images'])
    ncls=len(np.unique(y))
    ref_n=5000
    head=LogisticRegression(C=1,max_iter=500,solver='lbfgs').fit(features[fit],y[fit])
    assert np.array_equal(head.classes_,np.arange(ncls))
    base_y=y[rest]
    raw=read_csv(HERE/f'{dataset}_baseline.csv')
    rows=[dict(seed=int(r['seed']),side=r['side'],protocol=r['protocol'],detector=r['detector'],n=int(r['n']),
               k=int(r['k']),hits=int(r['hits']),errors=int(r['errors'])) for r in raw]
    lookup={(r['seed'],r['side'],r['protocol'],r['detector']):r for r in rows}
    assert len(lookup)==len(rows)
    a=np.load(HERE/f'{dataset}_claims.npz')
    ref=a[f"{P['seeds'][0]}_ref"]
    assert len(ref)==ref_n
    mats=score(head,features[rest][ref],base_y[ref],features[rest],ncls)
    for seed in P['seeds']:
        assert np.array_equal(ref,a[f'{seed}_ref'])
        for side,gs in [('source',P['sources']),('target',P['human_targets'][dataset])]:
            ix=a[f'{seed}_{side}']; ties=a[f'{seed}_{side}_ties']; n=len(ix)
            assert len(np.intersect1d(ix,ref))==0
            k=round(P['review_fraction']*n)
            for g in gs:
                key=f"{seed}_source_{P['source_rates'][dataset]:.2f}_{g}" if side=='source' else f'{seed}_human_{g}'
                obs=a[key]
                err=int((obs!=base_y[ix]).sum())
                assert (n,k,err)==tuple(lookup[seed,side,g,'CNN'][key] for key in ('n','k','errors'))
                for method,m in mats.items():
                    h=counts(m[ix],obs,base_y[ix],ties,k)
                    row=dict(seed=seed,side=side,protocol=g,detector=method,n=n,k=k,hits=h,errors=err)
                    rows.append(row);lookup[seed,side,g,method]=row
    write_csv(out/f'{dataset}_scores.csv',rows)
    transfer=[]
    choice=[]
    for size in P['candidate_sizes']:
        names=MENU[:size]
        for seed in P['seeds']:
            for g in P['sources']:
                source=[lookup[seed,'source',g,m]['hits'] for m in names]
                chosen=names[np.argmax(source)]
                top_sorted=sorted(source,reverse=True)
                choice.append(dict(seed=seed,source=g,menu=size,chosen=chosen,source_gap=top_sorted[0]-top_sorted[1]))
                for t in P['human_targets'][dataset]:
                    z=[lookup[seed,'target',t,m] for m in names]
                    result=lookup[seed,'target',t,chosen]
                    n=result['n']; hits=result['hits']
                    opt=max(r['hits'] for r in z)
                    fixed=z[0]['hits']
                    transfer.append(dict(seed=seed,menu=size,source=g,target=t,chosen=chosen,
                        oracle=names[np.argmax([r['hits'] for r in z])],target_n=n,budget=result['k'],
                        hits=hits,oracle_hits=opt,fixed_cnn_hits=fixed,regret_per_10000=10000*(opt-hits)/n,
                        gain_vs_fixed_per_10000=10000*(hits-fixed)/n))
    write_csv(out/f'{dataset}_choices.csv',choice)
    write_csv(out/f'{dataset}_transfer.csv',transfer)
    if dataset=='c10':
        score_h(head,features[rest][ref],base_y[ref],model,choice,out)
    print(f'{dataset}: {len(rows)} metrics and {len(transfer)} transfers',flush=True)

def score_h(head,ref,y,model,choice,out):
    h=np.load(H/'images.npz')
    keep=h['keep']; target=h['target_y'][keep]; n=len(target)
    images=h['target_x'][keep].reshape(n,3,32,32)
    mats=score(head,ref,y,embedding(model,images),10)
    q=np.load(H/'counts.npy')[keep].cumsum(1)
    hp=json.loads((H/'protocol.json').read_text())
    old=read_csv(H/'result'/'target_votes.csv')
    assert len(old)==hp['target_label_draws']
    choices={(r['menu'],r['seed'],r['source']):r['chosen'] for r in choice}
    k=round(P['review_fraction']*n)
    rows=[]
    for j,row in enumerate(old):
        draw=hp['target_label_draw_seed_start']+j
        assert draw==int(row['seed'])
        rng=np.random.default_rng(draw)
        vote=(rng.integers(0,q[:,-1])[:,None]>=q).sum(axis=1)
        ties=rng.permutation(n)
        assert int((target!=vote).sum())==int(row['reference_disagreements'])
        hits={m:int(row[m]) for m in MENU[:5]}
        hits.update({m:counts(v,vote,target,ties,k) for m,v in mats.items()})
        for size in P['candidate_sizes']:
            for seed in P['seeds']:
                for g in ('Chen-ID-mask','Chen-ID-hard'):
                    chosen=choices[size,seed,g]
                    best=max(hits[m] for m in MENU[:size])
                    rows.append(dict(seed=seed,draw=draw,menu=size,source=g,chosen=chosen,target_n=n,
                                     budget=k,hits=hits[chosen],oracle_hits=best,fixed_cnn_hits=hits['CNN']))
    write_csv(out/'c10h_transfer.csv',rows)

def run(dataset):
    torch.set_num_threads(3)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    model=resnet18(weights=None)
    model.load_state_dict(torch.load(WEIGHTS,map_location='cpu',weights_only=True))
    model.fc=torch.nn.Identity()
    model=model.to(device=DEVICE,memory_format=torch.channels_last).eval()
    out=HERE/'vision';out.mkdir(exist_ok=True)
    evaluate(dataset,model,out)
    (out/'model.json').write_text(json.dumps(dict(protocol=P,weights_sha256=digest(WEIGHTS),
        gpu=torch.cuda.get_device_name(),torch=torch.__version__),indent=2)+'\n')

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('dataset',choices=['c10','c100']);b=a.parse_args()
    run(b.dataset)
