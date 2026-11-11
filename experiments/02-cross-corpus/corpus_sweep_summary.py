"""Recompute every fixed-budget endpoint from saved source decisions and target ranks."""
import argparse
import ast
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

P=json.loads(Path(__file__).with_name("corpus_sweep_protocol.json").read_text())


def interval(x,groups):
    ids,inverse=np.unique(groups,return_inverse=True)
    sz=np.bincount(inverse)
    totals=np.stack([np.bincount(inverse,weights=r,minlength=len(ids)) for r in x])
    n=len(x)
    rng=np.random.default_rng(P["bootstrap_seed"])
    draws=[]
    for start in range(0,P["bootstrap_draws"],100):
        b=min(100,P["bootstrap_draws"]-start)
        w=rng.multinomial(n,np.full(n,1/n),size=b)/n
        sampled=rng.integers(len(ids),size=(b,len(ids)))
        values=np.take_along_axis(w@totals,sampled,axis=1).sum(1)/sz[sampled].sum(1)
        draws.extend(10000*values)
    return [float(x.mean()*10000),*np.quantile(draws,[.025,.975]).tolist()]


def compute(path):
    info=json.loads((path/"input.json").read_text())
    with (path/"paired.csv").open(newline="") as f:
        rows=list(csv.DictReader(f))
    with np.load(path/"target.npz") as data:
        e,rank,names,groups=[data[k] for k in ("errors","rank","names","group")]
    assert len(e)==info["target"] and e.sum()==info["disagreements"]
    assert len(rows)==len(P["seeds"])*len(P["budgets"])*2
    r={}
    flat=[]
    for menu in ("core","expanded"):
        r[menu]={}
        for b in P["budgets"]:
            chosen=[x for x in rows if x["menu"]==menu and float(x["budget"])==b]
            assert [int(x["seed"]) for x in chosen]==P["seeds"]
            n=len(ast.literal_eval(chosen[0]["random_source"]))
            k=round(b*len(e))
            order=rank[:n,:k]
            assert np.all(np.sort(rank[:n],axis=1)==np.arange(len(e)))
            hit=np.zeros((n,len(e)),dtype=np.int8)
            np.put_along_axis(hit,order,e[order].astype(np.int8),axis=1)
            yh=hit.sum(1)
            random=np.array([int(x["random"]) for x in chosen])
            hard=np.array([int(x["difficulty"]) for x in chosen])
            strict,tie_average,tie_min,tie_max=[],[],[],[]
            for row,a,d in zip(chosen,random,hard):
                vals=[np.array(ast.literal_eval(row[k])) for k in ("random_source","difficulty_source")]
                assert [int(v.argmax()) for v in vals]==[a,d]
                assert [int(yh[a]),int(yh[d]),int(yh.max())]==[int(row[k]) for k in ("random_hits","difficulty_hits","oracle_hits")]
                winners=[np.flatnonzero(v==v.max()) for v in vals]
                w0,w1=winners
                strict.append(all(len(w)==1 for w in winners))
                tie_average.append(hit[w1].mean(0)-hit[w0].mean(0))
                tie_min.append(yh[w1].min()-yh[w0].max())
                tie_max.append(yh[w1].max()-yh[w0].min())
            difference=interval(hit[hard]-hit[random],groups)
            result=dict(menu=menu,budget=b,checks=k,n=len(e),groups=len(np.unique(groups)),methods=n,
                choices_differ=int((random!=hard).sum()),strict_switches=int(np.sum((random!=hard)&strict)),
                random=dict(Counter(names[random])),difficulty=dict(Counter(names[hard])),
                oracle=str(names[yh.argmax()]),target_yield=dict(zip(names[:n],(10000*yh/len(e)).tolist())),
                difference=difference,
                tie_average_difference=interval(np.stack(tie_average),groups),
                tie_choice_range=[float(10000*np.mean(v)/len(e)) for v in (tie_min,tie_max)],
                random_regret=float(10000*(yh.max()-yh[random]).mean()/len(e)),
                difficulty_regret=float(10000*(yh.max()-yh[hard]).mean()/len(e)))
            r[menu][str(b)]=result
            flat.append(dict(dataset=path.name,**{k:v for k,v in result.items() if k not in ("random","difficulty","target_yield")}))
    return {**info,"by_menu":r},flat


def run(root,out):
    results,flat,unavailable={},[],{}
    for name in P["datasets"]:
        path=root/name
        if not (path/"paired.csv").exists():
            reason=path/"excluded.json"
            unavailable[name]=json.loads(reason.read_text()) if reason.exists() else "Not scored"
            continue
        results[name],rows=compute(path)
        flat.extend(rows)
    assert results
    expanded=[x["by_menu"]["expanded"][str(P["primary_budget"])] for x in results.values()]
    threshold=P["material_fixes_per_10000"]
    summary=dict(eligible=len(expanded),target_total=sum(r["n"] for r in expanded),
                 any_switches=sum(r["choices_differ"]>0 for r in expanded),
                 strict_switches=sum(r["strict_switches"]>0 for r in expanded),
                 any_difference=sum(r["difference"][0]!=0 for r in expanded),
                 positive_interval=sum(r["difference"][1]>0 for r in expanded),
                 negative_interval=sum(r["difference"][2]<0 for r in expanded),
                 material_positive=sum(r["difference"][1]>threshold for r in expanded),
                 material_negative=sum(r["difference"][2]<-threshold for r in expanded),
                 tie_average_positive=sum(r["tie_average_difference"][1]>0 for r in expanded),
                 tie_average_negative=sum(r["tie_average_difference"][2]<0 for r in expanded))
    out.mkdir(parents=True,exist_ok=True)
    (out/"summary.json").write_text(json.dumps(dict(primary=summary,unavailable=unavailable,corpora=results),indent=2)+"\n")
    with (out/"cells.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]))
        w.writeheader()
        w.writerows(flat)
    print(json.dumps(dict(primary=summary,unavailable=unavailable),indent=2))
    for name,r in results.items():
        m=r["by_menu"]["expanded"]["0.1"]
        print(name,m["n"],m["choices_differ"],m["strict_switches"],np.round(m["difference"],2).tolist(),
              "tie-average",np.round(m["tie_average_difference"],2).tolist())


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("root",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    run(a.root,a.out)
