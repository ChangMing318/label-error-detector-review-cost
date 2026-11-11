"""Run one paired synthetic-to-human benchmark cell on a public corpus."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.special import softmax
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, normalize

from corpus_sweep_inputs import key, load

P = json.loads(Path(__file__).with_name("corpus_sweep_protocol.json").read_text())


def cap(part,role,limit):
    return sorted(part,key=lambda r:key(f"{role}:{r[0]}"))[:limit]


def representation(fit,other):
    word = TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=18000,sublinear_tf=True,dtype=np.float32)
    char = TfidfVectorizer(analyzer="char",ngram_range=(3,5),min_df=2,max_features=24000,
                           sublinear_tf=True,dtype=np.float32)
    a,b = word.fit_transform(fit),word.transform(other)
    c,d = char.fit_transform(fit),char.transform(other)
    svd = TruncatedSVD(min(64,a.shape[1]-1,a.shape[0]-1),random_state=26112499).fit(a)
    return (a,b),(c,d),(normalize(svd.transform(a)),normalize(svd.transform(b)))


def text_embed(model, values):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model,local_files_only=True)
    tokenizer.padding_side="left"
    encoder = AutoModel.from_pretrained(model,local_files_only=True,dtype=torch.bfloat16,
                                        attn_implementation="sdpa").cuda().eval()
    rows = []
    with torch.inference_mode():
        for start in range(0,len(values),128):
            tokens=tokenizer(values[start:start+128],padding=True,truncation=True,max_length=64,
                             return_tensors="pt").to("cuda")
            hidden=encoder(**tokens,use_cache=False).last_hidden_state
            rows.append(hidden[:,-1].float().cpu().numpy())
    return normalize(np.concatenate(rows))


def images_embed(values):
    import torch
    from torchvision.models import resnet18
    from vision import WEIGHTS, embedding
    x=np.asarray(values,dtype=np.uint8).reshape(len(values),-1,32,32) if np.asarray(values[0]).size!=784 else None
    if x is None:
        x=np.repeat(np.asarray(values,dtype=np.uint8).reshape(-1,1,28,28),3,axis=1)
    model=resnet18(weights=None)
    model.load_state_dict(torch.load(WEIGHTS,map_location="cpu",weights_only=True))
    model.fc=torch.nn.Identity()
    model=model.to(device="cuda",memory_format=torch.channels_last).eval()
    return embedding(model,x)


def heads(a,b,y):
    lr=LogisticRegression(C=1,max_iter=350).fit(a,y)
    return lr.predict_proba(b)


def geometry(a,b,y):
    a,b=normalize(a),normalize(b)
    classes=np.unique(y)
    centers=normalize(np.stack([a[y==c].mean(0) for c in classes]))
    sim=b@centers.T
    ids=NearestNeighbors(n_neighbors=10,algorithm="brute").fit(a).kneighbors(b,return_distance=False)
    return [sim.max(1,keepdims=True)-sim,(y[ids,None]!=classes[None,None,:]).sum(1)]


def make_scores(fit,rest,y,name,model):
    x,z=[r[2] for r in fit],[r[2] for r in rest]
    text=isinstance(x[0],str)
    if text:
        word,char,latent=representation(x,z)
        clean_proba=heads(*word,y)
        char_proba=heads(*char,y)
        core=["word-head","char-head","margin","10NN"]
        mats=[-np.log(clean_proba),-np.log(char_proba),*geometry(*latent,y)]
        alt=char
        emb=text_embed(model,x+z)
    else:
        a,b=np.asarray(x).reshape(len(x),-1),np.asarray(z).reshape(len(z),-1)
        if name=="AudioSet":
            emb=normalize(np.concatenate([a[:,128:],b[:,128:]]))
            a,b=a[:,:128],b[:,:128]
        else:
            emb=images_embed(x+z)
        scaler=StandardScaler().fit(a)
        a,b=scaler.transform(a),scaler.transform(b)
        clean_proba=heads(a,b,y)
        core=["numeric-head","margin","10NN"]
        mats=[-np.log(clean_proba),*geometry(a,b,y)]
        alt=normalize(a),normalize(b)
    a,b=emb[:len(fit)],emb[len(fit):]
    proba=heads(a,b,y)
    mats.extend([-np.log(proba),*geometry(a,b,y)])
    aux=RidgeClassifier(alpha=1).fit(alt[0],y)
    d=aux.decision_function(alt[1])
    if d.ndim==1:
        d=np.stack([-d,d],axis=1)
    return core,["frozen-head","frozen-margin","frozen-10NN"],np.stack(mats),proba,softmax(d,axis=1)


def orders(mats,proba,claim,tie):
    from cleanlab.filter import find_label_issues
    from cleanlab.rank import get_label_quality_scores
    rank=[]
    for v in mats[:,np.arange(len(claim)),claim]:
        rank.append(tie[np.argsort(-v[tie],kind="stable")])
    issues=find_label_issues(claim,proba,filter_by="prune_by_noise_rate",
                            return_indices_ranked_by="normalized_margin",n_jobs=1)
    rest=tie[~np.isin(tie,issues)]
    quality=get_label_quality_scores(claim,proba,method="normalized_margin")
    rank.append(np.r_[issues,rest[np.argsort(quality[rest],kind="stable")]])
    return np.stack(rank)


def run(a,reuse):
    fit,h,t=load(a.dataset,a,reuse)
    pre_counts=[len(v) for v in (fit,h,t)]
    # Preserve original grouping and remove exact duplicates across roles.
    from corpus_sweep_inputs import norm
    seen,groups=set(),set()
    def unique(part):
        out=[]
        for row in part:
            k=norm(row[2]) if isinstance(row[2],str) else row[1]
            if k in seen or row[1] in groups:
                continue
            seen.add(k)
            out.append(row)
        groups.update(r[1] for r in out)
        return out
    t=unique(cap(t,"target",P["target_limit"]))
    h=unique(cap(h,"history",P["source_limit"]))
    fit=unique(cap(fit,"fit",P["fit_limit"]))
    classes=np.unique([r[3] for r in fit])
    h=[r for r in h if r[3] in classes]
    t=[r for r in t if r[3] in classes and r[4] in classes]
    info=dict(dataset=a.dataset,original=pre_counts,fit=len(fit),history=len(h),target=len(t),
              classes=classes.tolist(),disagreements=sum(r[3]!=r[4] for r in t))
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/"input.json").write_text(json.dumps(info,indent=2)+"\n")
    if len(classes)<2 or min(len(h),len(t))<P["min_source_target"] or info["disagreements"]<P["min_target_disagreements"]:
        (a.out/"excluded.json").write_text(json.dumps(info,indent=2)+"\n")
        print("Insufficient preregistered support",info,flush=True)
        return
    y,h_y,t_y,t_claim=[np.searchsorted(classes,[r[k] for r in group])
                         for group,k in ((fit,3),(h,3),(t,3),(t,4))]
    print(info,flush=True)
    base,added,mats,proba,aux=make_scores(fit,h+t,y,a.dataset,a.model)
    names=base+added+["Confident-Learning"]
    size=len(h)
    e=t_y!=t_claim
    tt=np.argsort([key("tie:"+r[0]) for r in t])
    target=orders(mats[:,size:],proba[size:],t_claim,tt)
    aux=aux[:size].copy()
    aux[np.arange(size),h_y]=-1
    dst,difficulty=aux.argmax(1),aux.max(1)
    rows=[]
    for seed in P["seeds"]:
        rng=np.random.default_rng(seed)
        ix=np.concatenate([rng.permutation(np.flatnonzero(h_y==c))[:max(1,round(.8*(h_y==c).sum()))]
                           for c in np.unique(h_y)])
        assert len(np.unique(ix))==len(ix)
        k=round(P["noise_rate"]*len(ix))
        tie=rng.permutation(len(ix))
        masks=[rng.permutation(len(ix))[:k],tie[np.argsort(-difficulty[ix][tie],kind="stable")[:k]]]
        source_y=h_y[ix]
        history=[]
        for mask in masks:
            claim=source_y.copy()
            claim[mask]=dst[ix[mask]]
            assert (claim!=source_y).sum()==k
            order=orders(mats[:,ix],proba[ix],claim,tie)
            history.append((claim!=source_y)[order])
        for b in P["budgets"]:
            sh=np.stack([v[:,:round(len(ix)*b)].sum(1) for v in history])
            th=e[target[:,:round(len(e)*b)]].sum(1)
            for menu,n in (("core",len(base)),("expanded",len(names))):
                choices=sh[:,:n].argmax(1)
                rows.append(dict(seed=seed,budget=b,menu=menu,source_n=len(ix),noise=k,
                                 mask_overlap=int(np.intersect1d(*masks).size),
                                 random=int(choices[0]),difficulty=int(choices[1]),oracle=int(th[:n].argmax()),
                                 random_hits=int(th[choices[0]]),difficulty_hits=int(th[choices[1]]),
                                 oracle_hits=int(th[:n].max()),random_source=sh[0,:n].tolist(),
                                 difficulty_source=sh[1,:n].tolist()))
    with (a.out/"paired.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    np.savez_compressed(a.out/"target.npz",errors=e,rank=target,names=np.array(names),
                        group=np.array([key(r[1]) for r in t],dtype=np.uint64),
                        ids=np.array([key(r[0]) for r in t],dtype=np.uint64))
    print("Complete",a.dataset,flush=True)


if __name__=="__main__":
    import torch
    from operating_dynasent import helpers
    p=argparse.ArgumentParser()
    p.add_argument("dataset",choices=P["datasets"])
    for k in ("out","stage","crowds","audio","snli","mnli","dyna","new","model","helpers"):
        p.add_argument("--"+k,type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(3)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    run(a,helpers(a.helpers))
