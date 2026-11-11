"""SNLI cross-domain replay of the fixed-destination mask experiment."""
import argparse
import csv
import hashlib
import json
import re
import socket
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import sklearn
import torch
from scipy.special import logsumexp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from torch import nn

LABEL = {'entailment':0,'neutral':1,'contradiction':2}
WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def read_split(z, split):
    fn = f'snli_1.0/snli_1.0_{split}.jsonl'
    rows = []
    n = 0
    for line in z.open(fn):
        d = json.loads(line)
        n += 1
        votes = d['annotator_labels']
        if len(votes)!=5 or any(v not in LABEL for v in votes):
            continue
        ref,k = Counter(votes[1:]).most_common(1)[0]
        if k<3:
            continue
        p,h = d['sentence1'],d['sentence2']
        # Also group different captions of the same underlying Flickr image.
        image = d['captionID'].split('#',1)[0]
        rows.append((d['pairID'],image,' '.join(WORD.findall(p.lower())),p,h,LABEL[ref],LABEL[votes[0]]))
    return rows,n


def holdout(rows, higher_priority):
    images={r[1] for r in higher_priority}
    premises={r[2] for r in higher_priority}
    return [r for r in rows if r[1] not in images and r[2] not in premises]


def features(p,h):
    a,b=WORD.findall(p.lower()),WORD.findall(h.lower())
    pwords=set(a)
    tagged=[*[f'p_{w}' for w in a],*[f'h_{w}' for w in b],
            *[f'x_{w}' if w in pwords else f'n_{w}' for w in b]]
    tagged.extend(f'pb_{u}_{v}' for u,v in zip(a,a[1:]))
    tagged.extend(f'hb_{u}_{v}' for u,v in zip(b,b[1:]))
    tagged.extend((f'overlap_{int(10*sum(w in pwords for w in b)/max(len(b),1))}',
                   f'length_delta_{max(-8,min(8,len(b)-len(a)))}'))
    return ' '.join(tagged)


def l2(x):
    return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-8)


class PairMLP(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.repr=nn.Sequential(nn.Linear(dim,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU())
        self.head=nn.Linear(64,3)

    def forward(self,x):
        z=self.repr(x)
        return self.head(z),z


def fit_model(x,y,fit_idx,source_idx,seed,p,track_source):
    torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    model=PairMLP(x.shape[1])
    x=torch.from_numpy(x)
    y=torch.from_numpy(y)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    avg=None
    for epoch in range(p['epochs']):
        model.train()
        for ids in torch.from_numpy(rng.permutation(fit_idx)).split(p['batch']):
            opt.zero_grad()
            logits,_=model(x[ids])
            nn.functional.cross_entropy(logits,y[ids]).backward()
            opt.step()
        if track_source:
            model.eval()
            with torch.no_grad():
                prob=torch.cat([model(x[ids])[0].softmax(1) for ids in torch.from_numpy(source_idx).split(1024)]).numpy()
                avg=prob.copy() if avg is None else avg+prob
        print(f'fitting model {seed}, epoch {epoch+1}/{p["epochs"]}',flush=True)
    model.eval()
    with torch.no_grad():
        outs=[model(chunk) for chunk in x.split(1024)]
        logits=torch.cat([t[0] for t in outs]).numpy()
        z=torch.cat([t[1] for t in outs]).numpy()
    return model,logits,l2(z),None if avg is None else avg/p['epochs']


def score_matrix(z,s,r,y):
    means=l2(np.stack([z[r][y[r]==i].mean(0) for i in range(3)]))
    sim=z[s]@means.T
    return sim.max(1,keepdims=True)-sim


def per_label_matrices(z,pca,rp,logits,refs,probe,y,p):
    nn_ref=NearestNeighbors(n_neighbors=p['knn_neighbors'],algorithm='brute',metric='euclidean').fit(z[refs])
    labels=y[refs[nn_ref.kneighbors(z[probe],return_distance=False)]]
    knn=(labels[:,:,None]!=np.arange(3)[None,None,:]).sum(1)
    logit=logits[probe].astype(np.float64)
    return np.stack([score_matrix(x,probe,refs,y) for x in (z,pca,rp)]
                    +[knn,logsumexp(logit,axis=1,keepdims=True)-logit])


def review(mat,claim,true,ties,cap):
    err=claim!=true
    details=[]
    hits=np.zeros((len(mat),len(err)),dtype=bool)
    for i,scores in enumerate(mat):
        score=scores[np.arange(len(claim)),claim]
        order=ties[np.argsort(-score[ties],kind='stable')]
        ids=order[:cap]
        hits[i,ids]=err[ids]
        details.append((int(hits[i].sum()),roc_auc_score(err,score),average_precision_score(err,score)))
    return details,hits


def crossed_ci(contributions,groups,p):
    labels,ix=np.unique(groups,return_inverse=True)
    sizes=np.bincount(ix)
    by_group=np.stack([np.bincount(ix,weights=c,minlength=len(labels)) for c in contributions])
    rng=np.random.default_rng(p['bootstrap_seed'])
    sims=[]
    n_sources=len(contributions)
    for start in range(0,p['bootstrap_draws'],100):
        b=min(100,p['bootstrap_draws']-start)
        source_weights=rng.multinomial(n_sources,np.full(n_sources,1/n_sources),size=b)/n_sources
        group_draws=rng.integers(0,len(labels),size=(b,len(labels)))
        v=np.take_along_axis(source_weights@by_group,group_draws,axis=1).sum(1)
        sims.extend(10000*v/sizes[group_draws].sum(1))
    return dict(mean=float(10000*contributions.sum(1).mean()/len(groups)),
                ci95=np.quantile(sims,[.025,.975]).tolist())


def save_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def run(args):
    out=args.out
    out.mkdir(exist_ok=True,parents=True)
    p=json.loads(args.protocol.read_text())
    with zipfile.ZipFile(args.data) as z:
        raw={name:read_split(z,name) for name in ('train','dev','test')}
        manifest=json.loads(z.read('input_manifest.json'))
        for name,(_,n) in raw.items():
            assert n==manifest['split_counts'][name]['five_valid_votes']
    test=raw['test'][0]
    dev=holdout(raw['dev'][0],test)
    train=holdout(raw['train'][0],test+dev)
    row_groups=set(r[1] for r in train),set(r[1] for r in dev),set(r[1] for r in test)
    assert not (row_groups[0]&row_groups[1] or row_groups[0]&row_groups[2] or row_groups[1]&row_groups[2])
    rows=train+dev+test
    assert len({r[0] for r in rows})==len(rows)
    groups=np.array([r[1] for r in rows])
    true=np.array([r[5] for r in rows],dtype=np.int64)
    human=np.array([r[6] for r in rows],dtype=np.int64)
    n_tr,n_d,n_te=map(len,(train,dev,test))
    print(f'eligible train={n_tr}, source={n_d}, target={n_te}',flush=True)
    dev_idx=np.arange(n_tr,n_tr+n_d)
    te_idx=np.arange(n_tr+n_d,len(rows))
    train_groups=np.unique(groups[:n_tr])
    refs_group=np.random.default_rng(p['ref_seed']).choice(train_groups,round(len(train_groups)*p['ref_group_fraction']),replace=False)
    is_ref=np.isin(groups[:n_tr],refs_group)
    fit_idx,ref_idx=np.flatnonzero(~is_ref),np.flatnonzero(is_ref)
    sentences=[features(r[3],r[4]) for r in rows]
    vectorizer=TfidfVectorizer(lowercase=False,token_pattern=r'\S+',dtype=np.float32,
                              min_df=2,max_features=p['tfidf_max_features'])
    vectorizer.fit([sentences[i] for i in fit_idx])
    sparse=vectorizer.transform(sentences)
    svd=TruncatedSVD(n_components=p['svd_dim'],random_state=p['feature_seed']).fit(sparse[fit_idx])
    z_svd=l2(svd.transform(sparse))
    w=np.random.default_rng(p['feature_seed']).normal(size=(sparse.shape[1],p['rp_dim'])).astype(np.float32)
    z_rp=l2(sparse@w)
    dense=np.ascontiguousarray(4*np.concatenate([z_svd,z_rp],axis=1),dtype=np.float32)
    model,logits,z,_=fit_model(dense,true,fit_idx,dev_idx,p['fit_seed'],p,False)
    _,_,_,aux=fit_model(dense,true,fit_idx,dev_idx,p['aux_seed'],p,True)
    aux[np.arange(n_d),true[dev_idx]]=-np.inf
    dst,diff=aux.argmax(1),aux.max(1)
    assert (dst!=true[dev_idx]).all()
    torch.save(model.state_dict(),out/'mlp.pt')
    probs=per_label_matrices(z,z_svd,z_rp,logits,ref_idx,np.r_[dev_idx,te_idx],true,p)
    dev_mat,te_mat=probs[:,:n_d],probs[:,n_d:]
    dev_groups=groups[dev_idx]
    unique_dev=np.unique(dev_groups)
    target_cap=round(p['review_fraction']*n_te)
    target_ties=np.random.default_rng(p['target_tie_seed']).permutation(n_te)
    te_detail,te_hits=review(te_mat,human[te_idx],true[te_idx],target_ties,target_cap)
    names=p['detector_order']
    target_metrics=[dict(detector=name,hits=m[0],auc=m[1],ap=m[2],precision=m[0]/target_cap)
                    for name,m in zip(names,te_detail)]
    anchor=names.index(p['fixed_default'])
    oracle=max(range(len(names)),key=lambda j:te_detail[j][0])
    by_seed, per, contributions, delta_fixed=[],[],[],[]
    for seed in p['source_split_seeds']:
        rng=np.random.default_rng(seed)
        subset=rng.choice(unique_dev,round(p['source_group_subsample']*len(unique_dev)),replace=False)
        ids=np.flatnonzero(np.isin(dev_groups,subset))
        k=round(p['source_noise']*len(ids))
        y=true[dev_idx][ids]
        common_ties=rng.permutation(len(ids))
        mask_random=rng.choice(len(ids),k,replace=False)
        mask_hard=np.argsort(-diff[ids],kind='stable')[:k]
        cap=round(p['review_fraction']*len(ids))
        selections=[]
        for arm,mask in [('random',mask_random),('hard',mask_hard)]:
            claim=y.copy()
            claim[mask]=dst[ids][mask]
            assert (claim!=y).sum()==k and (claim[mask]==dst[ids][mask]).all()
            metrics,_=review(dev_mat[:,ids],claim,y,common_ties,cap)
            selections.append(max(range(len(names)),key=lambda j:metrics[j][0]))
            for name,m in zip(names,metrics):
                per.append(dict(seed=seed,arm=arm,detector=name,source_n=len(ids),source_errors=k,review=cap,
                                hits=m[0],auc=m[1],ap=m[2]))
        a,b=selections
        h1,h2=te_detail[a][0],te_detail[b][0]
        by_seed.append(dict(seed=seed,random_choice=names[a],hard_choice=names[b],
                            random_target_hits=h1,hard_target_hits=h2,fixed_target_hits=te_detail[anchor][0],
                            oracle_target_hits=te_detail[oracle][0],common_corrupted=len(np.intersect1d(mask_random,mask_hard))))
        contributions.append(te_hits[a].astype(int)-te_hits[b])
        delta_fixed.append(te_hits[b].astype(int)-te_hits[anchor])
    effect=crossed_ci(np.stack(contributions),groups[te_idx],p)
    assert abs(effect['mean']-np.mean([r['random_target_hits']-r['hard_target_hits'] for r in by_seed])*10000/n_te)<1e-9
    lo,hi=effect['ci95']
    threshold=p['material_loss_per_10000']
    if lo>threshold:
        verdict='CONFIRMED_MATERIAL_LOSS'
    elif hi<threshold:
        verdict='MATERIAL_LOSS_RULED_OUT_ON_THIS_COHORT'
    else:
        verdict='INCONCLUSIVE_AT_MATERIAL_THRESHOLD'
    summary=dict(host=socket.gethostname(),dataset_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
                 protocol_sha256=hashlib.sha256(args.protocol.read_bytes()).hexdigest(),protocol=p,input_manifest=manifest,
                 raw={n:x['original'] for n,x in manifest['split_counts'].items()},consensus_eligible={n:len(x[0]) for n,x in raw.items()},
                 used=dict(fit=len(fit_idx),ref=len(ref_idx),dev=n_d,test=n_te),
                 target_groups=len(np.unique(groups[te_idx])),target_disagreements=int((human[te_idx]!=true[te_idx]).sum()),
                 target_review=target_cap,classification_accuracy=float((logits[te_idx].argmax(1)==true[te_idx]).mean()),
                 target_metrics=target_metrics,oracle=names[oracle],random_minus_hard_per_10000=effect,
                 hard_minus_fixed_per_10000=crossed_ci(np.stack(delta_fixed),groups[te_idx],p),
                 source_choices={arm:dict(Counter(r[arm+'_choice'] for r in by_seed)) for arm in ('random','hard')},
                 selections_differ=sum(r['random_choice']!=r['hard_choice'] for r in by_seed),verdict=verdict,
                 versions=dict(torch=torch.__version__,sklearn=sklearn.__version__,numpy=np.__version__))
    save_csv(out/'source_metrics.csv',per)
    save_csv(out/'paired_transfer.csv',by_seed)
    save_csv(out/'target_metrics.csv',target_metrics)
    save_csv(out/'target_item_hits.csv',[dict(pair_id=r[0],caption_image=r[1],disagreement=int(r[5]!=r[6]),
                  **{name:int(te_hits[j,i]) for j,name in enumerate(names)}) for i,r in enumerate(test)])
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser()
    a.add_argument('--data',type=Path,required=True)
    a.add_argument('--protocol',type=Path,default=Path('protocol.json'))
    a.add_argument('--out',type=Path,default=Path('result'))
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    run(a.parse_args())
