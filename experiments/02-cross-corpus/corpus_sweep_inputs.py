"""Load public human claims and references without materializing text in the results."""
import csv
import hashlib
import json
import tarfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from io import TextIOWrapper

import numpy as np

ROLES = ("fit","history","target")


def key(text):
    return int.from_bytes(hashlib.sha256(str(text).encode()).digest()[:8],"big")


def norm(text):
    return " ".join(text.casefold().split())


def role(group):
    p = key("split:"+str(group))%100
    return "fit" if p<50 else "history" if p<75 else "target"


def panel(rows):
    parts = defaultdict(list)
    seen = set()
    for i,g,x,y,c in rows:
        text = norm(x) if isinstance(x,str) else str(i)
        if text in seen:
            continue
        seen.add(text)
        parts[role(g)].append((i,g,x,y,c))
    return tuple(parts[k] for k in ROLES)


def existing(name,root):
    from blind_rate import datasets
    from fresh_votes import mbic, polite, sbic

    registry = {"GoEmotions":"go","Wikipedia-Attack":"wiki","D3code":"d3",
                "Unhealthy-Comments":"ucc","Fashion-MNIST-H":"fashion"}
    if name in registry:
        parts = datasets(root,registry[name])["full"]
    elif name=="Stanford-Politeness":
        parts = polite(root)["stack_transfer"]
    else:
        parts = mbic(root)["full"]
    return tuple([(i,g,x,ref[0],votes[0]) for i,g,x,votes,ref,_ in part] for part in parts)


def separate(item,votes):
    if len(votes)<3:
        return None
    index = key("claim:"+item)%len(votes)
    remain = votes[:index]+votes[index+1:]
    ref,n = Counter(remain).most_common(1)[0]
    return (ref,votes[index]) if n>len(remain)/2 else None


def hate(path):
    rows = []
    labels = {"hatespeech":0,"offensive":1,"normal":2}
    for i,d in sorted(json.loads(path.read_text(encoding="utf-8")).items()):
        text = " ".join(d["post_tokens"])
        votes = [labels[v["label"]] for v in sorted(d["annotators"],key=lambda r:r["annotator_id"])]
        pair = separate(i,votes)
        if pair is not None and norm(text):
            rows.append((i,norm(text),text,*pair))
    return panel(rows)


def davidson(path):
    rows = []
    with path.open(encoding="utf-8-sig",newline="") as f:
        for j,r in enumerate(csv.DictReader(f)):
            i = str(j)
            text = r["tweet"]
            votes = [k for k,n in enumerate(("hate_speech","offensive_language","neither")) for _ in range(int(r[n]))]
            assert len(votes)==int(r["count"])
            pair = separate(i,votes)
            if pair is not None and norm(text):
                rows.append((i,norm(text),text,*pair))
    return panel(rows)


def sbic_full(path):
    data,votes={},defaultdict(dict)
    with tarfile.open(path,"r:gz") as archive:
        for m in sorted(archive.getmembers(),key=lambda x:x.name):
            if not any(m.name.endswith(f"SBIC.v2.{suffix}.csv") for suffix in ("trn","dev","tst")):
                continue
            with TextIOWrapper(archive.extractfile(m),encoding="utf-8-sig",newline="") as f:
                for r in csv.DictReader(f):
                    w,v=r["WorkerId"],r["offensiveYN"]
                    if not w or v not in ("0","0.0","0.5","1","1.0"):
                        continue
                    text=r["post"]
                    g=norm(text)
                    data.setdefault(g,text)
                    votes[g][w]=int(float(v)==1)
    return panel([(str(key(g)),g,data[g],*pair) for g in sorted(data)
                  if (pair:=separate(str(key(g)),[votes[g][w] for w in sorted(votes[g])])) is not None])


def mhs_full(path):
    import pyarrow.parquet as pq

    data,votes={},defaultdict(dict)
    for batch in pq.ParquetFile(path).iter_batches(columns=["comment_id","annotator_id","text","hatespeech"]):
        for r in batch.to_pylist():
            if str(r["hatespeech"]) not in ("0","1","2","0.0","1.0","2.0"):
                continue
            i=str(r["comment_id"])
            data.setdefault(i,r["text"])
            votes[i][str(r["annotator_id"])]=int(float(r["hatespeech"])==2)
    return panel([(i,norm(data[i]),data[i],*pair) for i in sorted(data)
                  if (pair:=separate(i,[votes[i][w] for w in sorted(votes[i])])) is not None])


def cifar(stage,name):
    token = "c10" if name=="CIFAR-10N" else "c100"
    claim_key = "random_label1" if token=="c10" else "noisy_label"
    with np.load(stage/f"{token}_pixels.npz") as z,np.load(stage/f"{token}_claims.npz") as q:
        x,y,fit,rest = [z[k] for k in ("images","y","fit","rest")]
        source = rest[q["260109_source"]]
        target = rest[q["260109_target"]]
        claims = q[f"260109_human_{claim_key}"]
    fit = np.array(sorted(fit,key=lambda j:key(f"image-fit:{j}"))[:4000])
    source = np.array(sorted(source,key=lambda j:key(f"image-source:{j}"))[:4000])
    ids = sorted(range(len(target)),key=lambda j:key(f"image-target:{target[j]}"))[:6000]
    assert not np.intersect1d(source,target).size and not np.intersect1d(fit,target).size
    def record(j,obs):
        return (str(j),hashlib.sha256(x[j]).hexdigest(),x[j],int(y[j]),int(obs))
    return ([record(j,y[j]) for j in fit],[record(j,y[j]) for j in source],
            [record(target[j],claims[j]) for j in ids])


def snli(path,reuse):
    with zipfile.ZipFile(path) as z:
        test = reuse.read_split(z,"test")[0]
        dev = reuse.holdout(reuse.read_split(z,"dev")[0],test)
        train = reuse.holdout(reuse.read_split(z,"train")[0],test+dev)
    return tuple([(i,g,p+" [SEP] "+h,y,c) for i,g,_,p,h,y,c in part] for part in (train,dev,test))


def multinli(path,reuse):
    parts,seen = defaultdict(list),set()
    with zipfile.ZipFile(path) as z:
        for split in ("mismatched","matched"):
            member = f"multinli_1.0/multinli_1.0_dev_{split}.jsonl"
            for line in z.open(member):
                r=json.loads(line)
                p,h = r["sentence1"],r["sentence2"]
                g = norm(p)
                if split=="matched" and g in seen:
                    continue
                votes = r["annotator_labels"]
                if len(votes)!=5 or any(v not in reuse.LABEL for v in votes):
                    continue
                ref,n = Counter(votes[1:]).most_common(1)[0]
                if n<3:
                    continue
                split_role = "target" if split=="mismatched" else ("fit" if key(g)%2 else "history")
                parts[split_role].append((r["pairID"],g,p+" [SEP] "+h,reuse.LABEL[ref],reuse.LABEL[votes[0]]))
                if split=="mismatched":
                    seen.add(g)
    return tuple(parts[k] for k in ROLES)


def dyna(path,reuse):
    from operating_dynasent import read_data
    d=read_data(path,reuse.WORD)
    return tuple(d[k] for k in ("training","profile","r2"))


def alpha(path):
    rows=[]
    with path.open(encoding="utf-8") as f:
        for line in f:
            r=json.loads(line)
            ex=r["example"]
            dist=r["label_counter"]
            ref,n=max(dist.items(),key=lambda kv:kv[1])
            old=r["old_label"]
            if n<=sum(dist.values())/2 or ref not in ("1","2") or str(old) not in ("1","2"):
                continue
            text=" [SEP] ".join(ex[k] for k in ("obs1","hyp1","hyp2","obs2"))
            rows.append((str(r["uid"]),norm(ex["obs1"]+" "+ex["obs2"]),text,int(ref)-1,int(old)-1))
    return panel(rows)


def audio(root):
    import sys
    sys.path.insert(0,str(root))
    from acoustic import embed
    from audioset_replay import weak
    release = weak(root/"weak_eval.csv")
    ontology = json.loads((root/"ontology.json").read_text())
    children = {v["id"]:v["child_ids"] for v in ontology}
    speech = next(v["id"] for v in ontology if v["name"]=="Speech")
    family, todo = set(),[speech]
    while todo:
        k=todo.pop()
        if k not in family:
            family.add(k)
            todo.extend(children.get(k,[]))
    coverage = {}
    with (root/"strong.tsv").open(newline="",encoding="utf-8") as f:
        for r in csv.DictReader(f,delimiter="\t"):
            video,ms = r["segment_id"].rsplit("_",1)
            q = video,int(ms)
            event = r["label"] in family and float(r["end_time_seconds"])-float(r["start_time_seconds"])>=.5
            coverage[q] = int(coverage.get(q,False) or event)
    features=embed(root/"features/eval",release)
    rows=[]
    for q in sorted(coverage.keys()&release.keys()&features.keys()):
        i=f"{q[0]}:{q[1]}"
        rows.append((i,q[0],features[q],coverage[q],int(bool(release[q][1]&family))))
    return panel(rows)


def load(name,a,reuse):
    if name.startswith("CIFAR"):
        return cifar(a.stage,name)
    if name=="AudioSet":
        return audio(a.audio)
    if name=="SNLI":
        return snli(a.snli,reuse)
    if name=="MultiNLI":
        return multinli(a.mnli,reuse)
    if name=="DynaSent":
        return dyna(a.dyna,reuse)
    if name=="HateXplain":
        return hate(a.new/"hatexplain.json")
    if name=="Davidson":
        return davidson(a.new/"davidson.csv")
    if name=="SBIC":
        return sbic_full(a.crowds/"SBIC.v2.tgz")
    if name=="Measuring-Hate-Speech":
        return mhs_full(a.crowds/"measuring-hate-speech.parquet")
    if name=="Chaos-alphaNLI":
        return alpha(a.new/"chaosNLI_alphanli.jsonl")
    return existing(name,a.crowds)
