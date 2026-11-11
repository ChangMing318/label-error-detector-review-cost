"""Equal-budget review on additional public human annotation datasets."""
import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, normalize

from audit_frontier import predict as interval_gain
from operating_envelope import curve, forecast

HERE = Path(__file__).resolve().parent
P = json.loads((HERE/"external_crowds_protocol.json").read_text())
ROLES = ("fit","history","target")
LIMIT = dict(zip(ROLES,(10000,8000,12000)))


def key(text):
    return int.from_bytes(hashlib.sha256(f"{P['role_seed']}:{text}".encode()).digest()[:8],"big")


def table(path,delimiter=","):
    with path.open(encoding="utf-8-sig",newline="") as f:
        yield from csv.DictReader(f,delimiter=delimiter)


def vote_order(item,votes,k):
    ordered = sorted(votes,key=lambda worker:key(f"{item}:{worker}"))
    return [votes[worker] for worker in ordered[:k]]


def majority(v):
    count = Counter(v)
    return max(v,key=count.get)


def role_of(group):
    v = key(group)%100
    return "fit" if v<50 else "history" if v<75 else "target"


def go(root):
    meta, votes = {},defaultdict(dict)
    for j in (1,2,3):
        for r in table(root/f"goemotions_{j}.csv"):
            if r["example_very_unclear"].lower()=="true":
                continue
            i = r["id"]
            fields = list(r)[list(r).index("admiration"):list(r).index("neutral")+1]
            if not any(int(r[k]) for k in fields):
                continue
            label = int(any(int(r[k]) for k in fields if k!="neutral"))
            meta[i] = (r["link_id"] or i,r["text"])
            votes[i][r["rater_id"]] = label
    out = defaultdict(list)
    for i,v in votes.items():
        if len(v)<5:
            continue
        group,text = meta[i]
        a,b,c,r1,r2 = vote_order(i,v,5)
        out[role_of(group)].append((i,group,text,(a,b,c),(r1,r2),"full"))
    return out


def wiki(root):
    labels = defaultdict(dict)
    for r in table(root/"attack_annotations.tsv","\t"):
        labels[r["rev_id"]][r["worker_id"]] = int(float(r["attack"]))
    out = defaultdict(list)
    order = {"test":0,"dev":1,"train":2}
    role = {"train":"fit","dev":"history","test":"target"}
    seen = set()
    comments = sorted(table(root/"attack_annotated_comments.tsv","\t"),key=lambda r:order[r["split"]])
    for r in comments:
        i = r["rev_id"]
        v = labels[i]
        text = r["comment"].replace("NEWLINE_TOKEN"," ").replace("TAB_TOKEN"," ")
        group = " ".join(text.lower().split())
        if not group or group in seen or len(v)<9:
            continue
        seen.add(group)
        ordered = vote_order(i,v,9)
        out[role[r["split"]]].append((i,group,text,tuple(ordered[:3]),
                                    (majority(ordered[3:6]),majority(ordered[6:])),r.get("sample","full")))
    return out


def dense(root,name):
    p = root/f"{name}.npz"
    out = defaultdict(list)
    with np.load(p) as z:
        x,y,votes = [z[k] for k in ("x","y","votes")]
        for j in range(len(y)):
            keep = np.flatnonzero(votes[j]>=0)
            if len(keep)<3:
                continue
            i = str(j)
            v = vote_order(i,{str(w):int(votes[j,w]) for w in keep},3)
            out[role_of(i)].append((i,i,x[j],tuple(v),(int(y[j]),),"full"))
    return out


def fashion(root):
    with gzip.open(root/"t10k-images-idx3-ubyte.gz","rb") as f:
        x = np.frombuffer(f.read(),dtype=np.uint8,offset=16).reshape(10000,784)
    with gzip.open(root/"t10k-labels-idx1-ubyte.gz","rb") as f:
        y = np.frombuffer(f.read(),dtype=np.uint8,offset=8)
    names = ("t-shirt/top","trouser","pullover","dress","coat","sandal","shirt","sneaker","bag","ankle boot")
    votes = defaultdict(dict)
    with (root/"fmh_raw.csv").open(encoding="utf-8-sig",newline="") as f:
        for row in csv.reader(f):
            item,label,worker = row[:3]
            if not item.endswith(".jpg"):
                continue
            i = int(Path(item).stem)
            votes[i][worker] = int(label) if label.isdigit() else names.index(label.lower())
    out = defaultdict(list)
    for i in sorted(votes):
        if len(votes[i])<3:
            continue
        group = hashlib.sha256(x[i]).hexdigest()
        v = tuple(vote_order(str(i),votes[i],3))
        out[role_of(group)].append((str(i),group,x[i],v,(int(y[i]),),"full"))
    return out


def take(rows,role):
    return sorted(rows,key=lambda r:key("cap:"+r[0]))[:LIMIT[role]]


def feature_models(fit,rest):
    y = np.array([r[4][0] for r in fit])
    classes = np.unique(y)
    x = [r[2] for r in fit]
    z = [r[2] for r in rest]
    heads = []
    if isinstance(x[0],str):
        names = ["word head","char head","word SVD margin","word SVD 10NN"]
        for analyzer,ngram in (("word",(1,2)),("char",(3,5))):
            tf = TfidfVectorizer(analyzer=analyzer,ngram_range=ngram,min_df=2,max_features=25000,
                                 sublinear_tf=True,dtype=np.float32)
            train = tf.fit_transform(x)
            test = tf.transform(z)
            head = LogisticRegression(C=1,max_iter=500).fit(train,y)
            heads.append(-head.predict_log_proba(test))
            if analyzer=="word":
                svd = TruncatedSVD(min(64,train.shape[1]-1),random_state=P["role_seed"]).fit(train)
                a,b = normalize(svd.transform(train)),normalize(svd.transform(test))
    else:
        names = ["numeric head","cosine margin","10NN"]
        scaler = StandardScaler().fit(x)
        a,b = scaler.transform(x),scaler.transform(z)
        heads.append(-LogisticRegression(C=1,max_iter=500).fit(a,y).predict_log_proba(b))
        a,b = normalize(a),normalize(b)
    centers = normalize(np.stack([a[y==j].mean(0) for j in classes]))
    sim = b@centers.T
    neighbors = NearestNeighbors(n_neighbors=min(10,len(y)),algorithm="brute").fit(a).kneighbors(b,return_distance=False)
    scores = np.stack([*heads,sim.max(1,keepdims=True)-sim,
                       (y[neighbors,None]!=classes[None,None,:]).sum(1)])
    return names,scores,classes


def rank(scores,classes,claim,tie):
    j = np.searchsorted(classes,claim)
    assert np.all(j<len(classes)) and np.array_equal(classes[j],claim)
    vals = scores[:,np.arange(len(claim)),j]
    return np.stack([tie[np.argsort(-s[tie],kind="stable")] for s in vals])


def order_choices(e,r,n,b,audit):
    cap = (b-audit)/(n-audit)
    profiles = [curve(e,q) for q in r]
    return np.array([int(np.argmax([forecast(p,h/audit,cap) for p in profiles]))
                     for h in range(audit+1)])


def queue(prefix,a,b):
    return np.r_[a,prefix[~np.isin(prefix,a)][:b-len(a)]]


def estimated_net(q,prefix,lookup,a,anchor):
    n = len(q)
    b = len(anchor)
    base = q[anchor].sum()
    rng = np.random.default_rng(P["audit_seed"]-1)
    delta = []
    for _ in range(512):
        sample = rng.choice(n,a,replace=False)
        k = int((rng.random(a)<q[sample]).sum())
        ids = queue(prefix[lookup[k]],sample,b)
        delta.append(q[ids].sum()-base)
    return float(10000*np.mean(delta)/n)


def stats(delta,group):
    """Resample audit draws and item blocks with paired contributions."""
    _,ix = np.unique(group,return_inverse=True)
    count = np.bincount(ix)
    mass = np.stack([np.bincount(ix,weights=v,minlength=len(count)) for v in delta])
    rng = np.random.default_rng(P["bootstrap_seed"])
    means = []
    for lo in range(0,P["bootstrap_draws"],200):
        b = min(200,P["bootstrap_draws"]-lo)
        weights = rng.multinomial(len(mass),np.full(len(mass),1/len(mass)),size=b)/len(mass)
        item = rng.integers(len(count),size=(b,len(count)))
        means.extend(10000*np.take_along_axis(weights@mass,item,axis=1).sum(1)/count[item].sum(1))
    point = float(10000*delta.mean())
    sims = np.asarray(means)
    p = (1+int(np.sum(sims-point>=point)))/(len(sims)+1)
    return dict(mean=point,ci95=np.quantile(sims,[.025,.975]).tolist(),p_positive=p)


def evaluate(name,h,t,hs,ts,classes,names,source_mode):
    historical = np.array([majority(r[3]) if source_mode=="majority" else r[3][0] for r in h])
    source_error = historical!=np.array([r[4][0] for r in h])
    if source_error.sum()==0 or source_error.all():
        return None
    sh = rank(hs,classes,historical,np.argsort([key("tie:"+r[0]) for r in h]))
    priority = np.argsort(-source_error[sh[:,:round(.10*len(h))]].sum(1),kind="stable")
    fixed,runner = priority[:2]
    n = len(t)
    b = round(.10*n)
    a = max(2,min(100,2*round(.05*b)))
    votes = np.array([r[3] for r in t])
    tie = np.argsort([key("tie:"+r[0]) for r in t])
    rt = rank(ts,classes,votes[:,0],tie)
    prefix = rt[:,:b]
    lookup = order_choices(source_error,sh,n,b,a)
    q = (votes[:,1:]!=votes[:,:1]).mean(1)
    direct = int(q[prefix].sum(1).argmax())
    witness = tie[np.argsort(-q[tie],kind="stable")][:b]
    predicted = estimated_net(q,prefix,lookup,a,prefix[direct])
    # Selections, including the pre-audit decision, are fixed before evaluating references.
    report = dict(name=name,source_mode=source_mode,history_n=len(h),target_n=n,budget=b,audit=a,
                  history_choice=names[fixed],direct_choice=names[direct],challenger=names[runner],
                  predicted_audit_vs_direct=predicted,references=[])
    m = a//2
    while m:
        core = rt[[fixed,runner],:b-m]
        diff = [np.setdiff1d(core[k],core[1-k],assume_unique=True) for k in (0,1)]
        if len(diff[0])>=m:
            break
        m -= 1
    group = np.array([key("block:"+r[1])%min(100,max(10,n//12)) for r in t])
    raw = []
    for ref in range(len(t[0][4])):
        e = votes[:,0]!=np.array([r[4][ref] for r in t])
        policies = dict(historical=prefix[fixed],direct=prefix[direct],witness=witness)
        outcomes = {k:np.zeros((P["replicates"],n),dtype=np.int8) for k in (*policies,"uniform","exclusive")}
        for trial in range(P["replicates"]):
            rng = np.random.default_rng(P["audit_seed"]+trial)
            u = rng.choice(n,a,replace=False)
            selected = lookup[e[u].sum()]
            uniform = queue(prefix[selected],u,b)
            if m:
                left,right = [rng.choice(z,m,replace=False) for z in diff]
                _,lower,_ = interval_gain(len(diff[0]),m,int(e[left].sum()),int(e[right].sum()))
                switch = lower>0
                targeted = np.r_[core[int(switch)],left if switch else right]
            else:
                switch = False
                targeted = prefix[fixed]
            chosen = dict(**policies,uniform=uniform,exclusive=targeted)
            for k,ids in chosen.items():
                assert len(np.unique(ids))==b
                outcomes[k][trial,ids] = e[ids]
            raw.append(dict(target=name,source_mode=source_mode,reference=ref,seed=trial,
                            switched=switch,n=n,b=b,**{k:int(outcomes[k][trial].sum()) for k in chosen}))
        comp = {}
        for policy in ("uniform","exclusive","direct","witness"):
            controls = ("historical","direct","witness") if policy in ("uniform","exclusive") else ("historical",)
            comp[policy] = {c:stats(outcomes[policy]-outcomes[c],group) for c in controls}
        report["references"].append(dict(reference=ref,target_rate=float(e.mean()),comp=comp,
            oracle_gap=float(10000*(e[prefix].sum(1).max()-e[prefix[fixed]].sum())/n)))
    return report,raw


def run(a):
    out = a.out
    out.mkdir(parents=True,exist_ok=True)
    loaders = {"go":go,"wiki":wiki,"fashion":fashion}
    raw = loaders[a.dataset](a.data) if a.dataset in loaders else dense(a.data,a.dataset)
    kept = {role:take(raw[role],role) for role in ROLES}
    print(a.dataset,"eligible by role", {r:len(raw[r]) for r in ROLES},flush=True)
    sets = [("full",kept["history"],kept["target"])]
    if a.dataset=="wiki":
        sets.append(("random_to_blocked",
            take([r for r in raw["history"] if r[5]=="random"],"history"),
            take([r for r in raw["target"] if r[5]=="blocked"],"target")))
    records,results = [],[]
    for name,h,t in sets:
        if min(len(kept["fit"]),len(h),len(t))<60 or round(.10*len(t))<12:
            results.append(dict(name=name,status="insufficient_public_roles",history_n=len(h),target_n=len(t)))
            continue
        names,scores,classes = feature_models(kept["fit"],h+t)
        for mode in ("majority","single"):
            v = evaluate(name,h,t,scores[:,:len(h)],scores[:,len(h):],classes,names,mode)
            if v is not None:
                summary,rows = v
                results.append(summary)
                records.extend(rows)
    manifest = dict(dataset=a.dataset,eligible={r:len(raw[r]) for r in ROLES},
                    fit_n=len(kept["fit"]),cohorts=results)
    (out/"summary.json").write_text(json.dumps(manifest,indent=2)+"\n")
    if records:
        with (out/"paired.csv").open("w",newline="") as f:
            w = csv.DictWriter(f,fieldnames=list(records[0]))
            w.writeheader()
            w.writerows(records)
    print(a.dataset,"saved",len(records),"paired outcomes",flush=True)


if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dataset",choices=("go","wiki","fashion","music","sentiment","labelme"))
    for k in ("data","out"):
        p.add_argument("--"+k,type=Path,required=True)
    run(p.parse_args())
