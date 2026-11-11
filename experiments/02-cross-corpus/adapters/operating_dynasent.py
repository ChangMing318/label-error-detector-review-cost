"""DynaSent profile transfer from Round 1 development to two human targets."""
import argparse
import hashlib
import importlib.util
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.random_projection import GaussianRandomProjection

from operating_envelope import curve, forecast, matched, supported_n

HERE = Path(__file__).resolve().parent
P = json.loads((HERE/"operating_envelope.json").read_text())
C = json.loads((HERE/"dynasent_protocol.json").read_text())
LABELS = {"negative":0, "neutral":1, "positive":2}
NAMES = C["candidates"]


def key(value):
    return int.from_bytes(hashlib.sha256(f"{P['role_seed']}:{value}".encode()).digest()[:8], "big")


def helpers(path):
    spec = importlib.util.spec_from_file_location("snli_shared", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_data(path, normalize, separated_reference=False):
    with path.open("rb") as data:
        assert hashlib.file_digest(data,"sha256").hexdigest()==C["sha256"]
    sections = {
        "r2": ["02-dynabench-test", "02-dynabench-train"],
        "r1": ["01-yelp-test"],
        "development": ["01-yelp-dev"],
        "training": ["01-yelp-train"],
    }
    roles = {name: [] for name in ("r2","r1","profile","source","training")}
    text_seen, group_role = set(), {}
    with zipfile.ZipFile(path) as archive:
        for section, parts in sections.items():
            for part in parts:
                member = f"dynasent-v1.1/dynasent-v1.1-round{part}.jsonl"
                for line in archive.open(member):
                    d = json.loads(line)
                    uid, text = d["text_id"], d["sentence"]
                    group = d.get("review_id") or (d.get("prompt_data") or {}).get("review_id") or uid
                    role = section
                    if role=="development":
                        role = "source" if key(group)%2 else "profile"
                    norm = " ".join(normalize.findall(text.lower()))
                    if not norm or norm in text_seen or group_role.get(group,role)!=role:
                        continue
                    text_seen.add(norm)
                    group_role[group] = role
                    if role=="training":
                        if d["gold_label"] in LABELS:
                            roles[role].append((uid,group,text,LABELS[d["gold_label"]],-1))
                        continue
                    votes = sorted((worker, label) for label, workers in d["label_distribution"].items()
                                   for worker in workers)
                    if len(votes)!=5 or len({w for w,_ in votes})!=5:
                        continue
                    j = key(uid)%5
                    if separated_reference and role in ("r1","r2"):
                        if any(label not in LABELS for _,label in votes):
                            continue
                        claim,w1,w2,ref,ref2 = [LABELS[label] for _,label in votes[j:]+votes[:j]]
                        roles[role].append((uid,group,text,ref,claim,w1,w2,ref2))
                        continue
                    claim = votes.pop(j)[1]
                    ref, count = Counter(label for _,label in votes).most_common(1)[0]
                    if count>=3 and claim in LABELS and ref in LABELS:
                        roles[role].append((uid,group,text,LABELS[ref],LABELS[claim]))
    return roles


def orders(mat, claims, ties):
    return np.stack([ties[np.argsort(-s[ties],kind="stable")] for s in mat[:,np.arange(len(claims)),claims]])


def score_rows(a, reuse):
    separated_reference = getattr(a,"separated_reference",False)
    roles = read_data(a.data, reuse.WORD,separated_reference)
    train = roles.pop("training")
    # Reference and fit partitions are disjoint by original review ID.
    ref_rows = [r for r in train if key(r[1])%4==0]
    fit_rows = [r for r in train if key(r[1])%4!=0]
    rng = np.random.default_rng(C["fit_seed"])
    fit_rows = [fit_rows[i] for i in rng.permutation(len(fit_rows))[:C["fit_max"]]]
    ref_rows = [ref_rows[i] for i in rng.permutation(len(ref_rows))[:C["reference_max"]]]
    rows = fit_rows+ref_rows
    fit = np.arange(len(fit_rows))
    ref = np.arange(len(fit_rows),len(rows))
    start = len(rows)
    positions = {}
    for name in ("source","profile","r1","r2"):
        positions[name] = np.arange(len(rows),len(rows)+len(roles[name]))
        rows += roles[name]
    print({name:len(idx) for name,idx in positions.items()}, flush=True)
    y = np.array([r[3] for r in rows])
    claim = np.array([r[4] for r in rows])
    texts = [r[2] for r in rows]
    tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=2,max_features=25000,
                            sublinear_tf=True,dtype=np.float32)
    tfidf.fit(texts[:len(fit)])
    x = tfidf.transform(texts)
    svd = TruncatedSVD(64,random_state=C["fit_seed"]).fit(x[fit])
    rp = GaussianRandomProjection(192,random_state=C["fit_seed"]).fit(x[fit])
    pca = reuse.l2(svd.transform(x))
    random = reuse.l2(rp.transform(x))
    _,logits,z,_ = reuse.fit_model(pca,y,fit,positions["source"],C["fit_seed"],C,False)
    _,_,_,prob = reuse.fit_model(random,y,fit,positions["source"],C["aux_seed"],C,True)
    scores = reuse.per_label_matrices(z,pca,random,logits,ref,np.arange(start,len(rows)),y,C)
    out = {}
    for name,idx in positions.items():
        for field,data in dict(id=[rows[i][0] for i in idx],group=[rows[i][1] for i in idx],
                               truth=y[idx],claim=claim[idx],scores=scores[:,idx-start]).items():
            out[f"{name}_{field}"] = np.asarray(data)
        if separated_reference and name in ("r1","r2"):
            out[f"{name}_witness"] = np.array([rows[i][5:7] for i in idx])
            out[f"{name}_second_ref"] = np.array([rows[i][7] for i in idx])
    out["aux_probability"] = prob
    out["fit_count"] = len(fit)
    out["reference_count"] = len(ref)
    np.savez_compressed(a.cache,**out)


def evaluate(data, out, reuse):
    caps, rates = P["budgets"], P["dynasent_prevalence"]
    profiles, e, ranking = {}, {}, {}
    for name in ("profile","r1","r2"):
        truth, claim = data[f"{name}_truth"],data[f"{name}_claim"]
        e[name] = claim!=truth
        tie = np.argsort([key(i) for i in data[f"{name}_id"]])
        ranking[name] = orders(data[f"{name}_scores"],claim,tie)
        profiles[name] = [curve(e[name],r) for r in ranking[name]]
    auc = np.array([np.trapezoid(t,f) for t,f in profiles["profile"]])
    src_y,src_score = data["source_truth"],data["source_scores"]
    prob = data["aux_probability"]
    difficulty = 1-prob[np.arange(len(src_y)),src_y]
    wrong_prob = prob.copy()
    wrong_prob[np.arange(len(src_y)),src_y] = -1
    wrong = wrong_prob.argmax(1)
    source_rows = []
    for seed in P["source_seeds"]:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(src_y))[:1500]
        tie = np.argsort([key(i) for i in data["source_id"][idx]])
        k = round(len(idx)*C["source_rate"])
        for mask, corrupt in [("random",rng.permutation(len(idx))[:k]),
                              ("difficulty",tie[np.argsort(-difficulty[idx][tie],kind="stable")][:k])]:
            sy = src_y[idx]
            claims = sy.copy()
            claims[corrupt] = wrong[idx][corrupt]
            rank = orders(src_score[:,idx],claims,tie)
            for b in caps:
                hits = (claims!=sy)[rank[:,:round(len(idx)*b)]].sum(1)
                source_rows.append(dict(seed=seed, mask=mask, cap=b, n=len(idx),
                                        winner=int(np.argmax(hits)),hits=hits.tolist()))
    n = supported_n([e["r1"],e["r2"]],rates,2000)
    assert n>=500
    summary = {"fit":int(data["fit_count"]),"reference":int(data["reference_count"]),
               "profile_auc":auc.tolist(),"matched_n":n,"candidates":NAMES,"targets":{}}
    all_rows, baseline_rows = [], []
    for name in ("r1","r2"):
        prevalence = e[name].mean()
        info = dict(n=len(e[name]),disagreements=int(e[name].sum()),natural_p=float(prevalence),natural=[])
        for b in caps:
            cap = round(len(e[name])*b)
            truth = e[name]
            observed = 10000*truth[ranking[name][:,:cap]].sum(1)/len(truth)
            hits = np.zeros_like(ranking[name],dtype=bool)
            np.put_along_axis(hits,ranking[name][:,:cap],True,axis=1)
            contributions = truth*(hits[3].astype(int)-hits[0])
            info["natural"].append(dict(budget=b,observed=observed.tolist(),
                predicted=[forecast(q,prevalence,b) for q in profiles["profile"]],
                pair_interval=reuse.crossed_ci(contributions[None,:],data[f"{name}_group"],P)))
        summary["targets"][name] = info
        for (p,b),reps in matched(e[name],ranking[name],rates,caps,n,P["target_repeats"],P["role_seed"]).items():
            y = reps.mean(0)
            pred = np.array([forecast(q,p,b) for q in profiles["profile"]])
            natural = [forecast(q,e["profile"].mean(),b) for q in profiles["profile"]]
            ids = dict(profile=int(np.argmax(pred)),auc=int(np.argmax(auc)),natural=int(np.argmax(natural)))
            reg = {k:float(max(y)-y[i]) for k,i in ids.items()}
            for row in source_rows:
                if row["cap"]==b:
                    baseline_rows.append(dict(target=name, prevalence=p,budget=b,seed=row["seed"],
                        mask=row["mask"], regret=float(max(y)-y[row["winner"]]),winner=row["winner"]))
            pair = reps[:,3]-reps[:,0]
            all_rows.append(dict(target=name,prevalence=p,budget=b,n=n,observed=y.tolist(),
                predicted=pred.tolist(),observed_pair=float(pair.mean()),predicted_pair=float(pred[3]-pred[0]),
                draw_quantiles=np.quantile(pair,[.025,.975]).tolist(),
                mae=float(abs(y-pred).mean()),regrets=reg,oracle=int(np.argmax(y)),choices=ids))
    out.mkdir(exist_ok=True,parents=True)
    summary["profile_n"] = len(e["profile"])
    summary["profile_p"] = float(e["profile"].mean())
    summary["cells"] = all_rows
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    for filename,rows in [("cells",all_rows),("source",source_rows),("source_transfer",baseline_rows)]:
        reuse.save_csv(out/f"{filename}.csv",rows)
    print(json.dumps({k:v for k,v in summary.items() if k not in ("cells","targets")}), flush=True)


if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data",type=Path)
    p.add_argument("--helpers",type=Path,required=True)
    p.add_argument("--cache",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--from-cache",action="store_true")
    a = p.parse_args()
    torch.set_num_threads(4)
    reuse = helpers(a.helpers)
    if not a.from_cache:
        score_rows(a,reuse)
    with np.load(a.cache) as data:
        evaluate(data,a.out,reuse)
