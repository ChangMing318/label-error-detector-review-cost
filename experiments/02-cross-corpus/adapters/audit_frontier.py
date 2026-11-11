"""Estimate the remaining review gain from a paid sample of queue disagreements."""
import argparse
import csv
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.stats import hypergeom

from review_bridge_summary import bootstrap

HERE = Path(__file__).resolve().parent
P = json.loads((HERE/"audit_frontier_protocol.json").read_text())


@lru_cache(maxsize=None)
def bounds(d,m,k):
    """Exact 97.5% interval for one finite-population hit count."""
    totals = np.arange(k,d-m+k+1)
    good = ((hypergeom.cdf(k,d,totals,m)>=.0125) &
            (hypergeom.sf(k-1,d,totals,m)>=.0125))
    valid = totals[good]
    return int(valid[0]),int(valid[-1])


def predict(d,m,h0,h1):
    """Only the two paid audit hit counts enter the gain estimate."""
    l0,u0 = bounds(d,m,h0)
    l1,u1 = bounds(d,m,h1)
    difference = h1-h0
    return (d-m)*difference/m,l1-u0-difference,u1-l0-difference


def trial(group,e,r,names,j):
    n = len(e)
    b = round(.10*n)
    m = min(50,b//2)
    while m:
        core = r[:,:b-m]
        parts = [np.setdiff1d(core[k],core[1-k],assume_unique=True) for k in (0,1)]
        d = len(parts[0])
        if d>=m:
            break
        m -= 1
    baseline = int(e[r[0,:b]].sum())
    if m==0:
        return dict(seed=j,group=group,n=n,budget=b,audit=0,d=0,baseline=names[0],
                    challenger=names[1],h0=0,h1=0,forecast=0.0,low=0,high=0,residual_gain=0,coverage=True,
                    selected=False,fixed=baseline,stay=baseline,switch=baseline,policy=baseline)
    rng = np.random.default_rng(np.random.SeedSequence([P["seed_namespace"],j]))
    a0,a1 = [rng.choice(v,m,replace=False) for v in parts]
    h0,h1 = int(e[a0].sum()),int(e[a1].sum())
    point,low,high = predict(d,m,h0,h1)
    stay_queue,switch_queue = np.r_[core[0],a1],np.r_[core[1],a0]
    assert len(np.unique(stay_queue))==len(np.unique(switch_queue))==b
    stay,switch = int(e[stay_queue].sum()),int(e[switch_queue].sum())
    actual = switch-stay
    selected = low>0
    return dict(seed=j,group=group,n=n,budget=b,audit=2*m,d=d,baseline=names[0],challenger=names[1],
                h0=h0,h1=h1,forecast=point,low=low,high=high,residual_gain=actual,coverage=low<=actual<=high,
                selected=selected,fixed=baseline,stay=stay,switch=switch,policy=switch if selected else stay)


def c10(a):
    with (a.plans/"paired.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    panels = []
    for row in rows:
        with np.load(a.plans/(row["seed"]+".npz")) as z:
            panels.append((int(row["group"]),z["obs"]!=z["gold"],np.stack([z["knn"],z["margin"]]),
                           ["CNN kNN","CNN margin"]))
    return panels


def nli(a,reuse):
    import torch
    from paid_audit import make_data
    torch.set_num_threads(4)
    _,y,c,ix,r,_,setup,_ = make_data(a,reuse,include_scores=True)
    role = "cross_genre" if a.dataset=="nli-holdout" else "train"
    d,t = ix["profile"],ix[role]
    count = (c[d]!=y[d])[r["profile"][:,:round(.10*len(d))]].sum(1)
    pair = np.argsort(-count,kind="stable")[:2]
    return [(0,c[t]!=y[t],r[role][pair],[setup["detectors"][k] for k in pair])]


def get_panels(a,reuse):
    if a.cached is not None:
        with np.load(a.cached) as z:
            return [(int(g),z[f"error{i}"],z[f"order{i}"],z[f"name{i}"].tolist())
                    for i,g in enumerate(z["group"])]
    if a.dataset=="c10":
        return c10(a)
    if a.dataset in ("nli","nli-holdout"):
        return nli(a,reuse)
    import torch
    from paid_audit_more import c100,dyna
    torch.set_num_threads(4)
    return c100(a,panels_only=True) if a.dataset=="c100" else dyna(a,reuse,panels_only=True)


def run(a,reuse):
    panels = get_panels(a,reuse)
    if a.cached is None:
        arrays = {"group":np.array([g for g,_,_,_ in panels])}
        for i,(_,e,r,names) in enumerate(panels):
            arrays.update({f"error{i}":e,f"order{i}":r[:,:round(.10*len(e))].astype(np.uint16),
                           f"name{i}":np.asarray(names)})
        np.savez_compressed(a.out.with_suffix(".panels.npz"),**arrays)
    rows = [trial(*panels[j%len(panels)],j) for j in range(P["replicates"])]
    with a.out.open("w",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    n = np.array([r["n"] for r in rows])
    groups = np.array([r["group"] for r in rows])
    net = 10000*np.array([r["policy"]-r["fixed"] for r in rows])/n
    stay = 10000*np.array([r["stay"]-r["fixed"] for r in rows])/n
    scale = 10000/n
    diff = np.array([r["residual_gain"] for r in rows])
    pred = np.array([r["forecast"] for r in rows])
    summary = dict(dataset=a.dataset,checked=len(rows),interval_coverage=float(np.mean([r["coverage"] for r in rows])),
        switched=sum(r["selected"] for r in rows),
        wrong_switches=sum(r["selected"] and r["residual_gain"]<=0 for r in rows),
        mean_ci_width=float(np.mean(scale*np.array([r["high"]-r["low"] for r in rows]))),
        actual_switch_gain=float((scale*diff).mean()),forecast_switch_gain=float((scale*pred).mean()),
        forecast_mae=float((scale*abs(diff-pred)).mean()),
        same_audit_vs_fixed=bootstrap(stay[:,None],groups),
        gated_vs_fixed=bootstrap(net[:,None],groups))
    a.out.with_suffix(".summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary),flush=True)


if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dataset",choices=("c10","c100","dyna","nli","nli-holdout"))
    for k in ("out","helpers","plans","stage","base","extra","scores","data",
              "source","source-protocol","checkpoint","target","cached"):
        p.add_argument("--"+k,type=Path,required=k=="out")
    p.add_argument("--device",default="cuda")
    a=p.parse_args()
    reuse = None
    if a.cached is None and a.dataset in ("nli","nli-holdout","dyna"):
        from operating_dynasent import helpers
        reuse = helpers(a.helpers)
    run(a,reuse)
