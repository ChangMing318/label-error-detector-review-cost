"""Audit public AudioSet proposal provenance and replay it against the separate temporal labels."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import numpy as np

def weak(path):
    with path.open(encoding="utf-8",newline="") as stream:
        for _ in range(3):
            next(stream)
        return {(video,round(float(start)*1000)):(float(end),set(labels.split(",")))
                for video,start,end,labels in csv.reader(stream,skipinitialspace=True)}

def main(a):
    root=a.input
    files=["weak_eval.csv","weak_train.csv","classes.csv","strong.tsv","ontology.json","audioset_mturk.json"]
    source_hashes={f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in files}
    test=weak(root/"weak_eval.csv")
    train=weak(root/"weak_train.csv")
    assert len(set(key[0] for key in test)&set(key[0] for key in train))==0
    cls=list(csv.DictReader((root/"classes.csv").open(encoding="utf-8-sig",newline="")))
    mid_names={r["mid"]:r["display_name"] for r in cls}
    def name_key(name):
        return ' '.join(name.replace(',', '').split()).casefold()
    name_to_mid={name_key(value):key for key,value in mid_names.items()}
    assert len(mid_names)==len(name_to_mid)==527
    ontology=json.loads((root/"ontology.json").read_text(encoding="utf-8"))
    children={r["id"]:set(r["child_ids"]) for r in ontology}
    def descend(key):
        frontier=children.get(key,set()).copy()
        visited=set()
        while frontier:
            v=frontier.pop()
            if v not in visited:
                visited.add(v)
                frontier|=children.get(v,set())-visited
        return visited
    down={key:descend(key) for key in children}
    related={key:{key}|down.get(key,set())|{p for p,sub in down.items() if key in sub} for key in mid_names}
    strong=defaultdict(dict)
    vocabulary=set()
    event_rows=0
    with (root/"strong.tsv").open(encoding="utf-8",newline="") as stream:
        for row in csv.DictReader(stream,delimiter="\t"):
            event_rows+=1
            video,millis=row["segment_id"].rsplit("_",1)
            duration=float(row["end_time_seconds"])-float(row["start_time_seconds"])
            assert duration>=0
            mid=row["label"]
            key=(video,int(millis))
            strong[key][mid]=max(duration,strong[key].get(mid,0.))
            vocabulary.add(mid)
    assert len(test)==20371 and event_rows==139538 and len(strong)==16996
    universe=sorted(set(strong)&set(test))
    known_classes=sorted(vocabulary&set(mid_names))
    index={key:j for j,key in enumerate(mid_names)}
    shared=np.array([index[key] for key in known_classes])
    method_names=["cooccurrence","prior"]
    acoustic,prediction_counts=None,{}
    if a.features:
        from acoustic import predict
        acoustic,prediction_counts=predict(a.features,train,test,index)
        method_names.append("acoustic_ridge")
    elif a.acoustic_scores:
        table=csv.DictReader(a.acoustic_scores.open(newline=""))
        acoustic={(r["video_id"],int(r["start_ms"])):float(r["acoustic_score"]) for r in table}
        method_names.append("acoustic_ridge")
    historical={}
    weak_disagreement=[]
    rows=json.loads((root/"audioset_mturk.json").read_text())
    for r in rows:
        url=urlparse(r["url"])
        parms=parse_qs(url.query)
        key=(url.path.strip("/"),round(float(parms["start"][0])*1000))
        assert key not in historical
        historical[key]=r
        if key in test:
            end,labels=test[key]
            assert int(end)==int(parms["end"][0])
            announced={name_to_mid.get(name_key(n),n) for n in r["given_original_labels"]}
            if announced!=labels:
                weak_disagreement.append(key)
    eligible=[key for key in universe if key not in set(weak_disagreement)]
    assert len(rows)==307
    if acoustic is not None:
        eligible=[key for key in eligible if key in acoustic]
    # Aggregate the original weak training labels only. A fixed pseudo-count
    # shrinks a conditional estimate back toward the independent class prior.
    co=np.zeros((len(index),len(index)),dtype=np.int64)
    for _,labels in train.values():
        ids=[index[label] for label in labels]
        co[np.ix_(ids,ids)]+=1
    counts=co.diagonal()
    prior=counts/len(train)
    conditional=(co+50*prior[None,:])/(counts[:,None]+50)
    k=len(set(historical)&set(eligible))
    n=len(eligible)
    assert 0<k<n
    logged=np.zeros(n,dtype=bool)
    archive_hits=np.zeros(n,dtype=int)
    count_half=np.zeros(n,dtype=int)
    count_one=np.zeros(n,dtype=int)
    scores=np.zeros((len(method_names),n))
    for i,key in enumerate(eligible):
        labels=test[key][1]
        allowed=np.ones(len(index),dtype=bool)
        banned=set().union(*(related[c] for c in labels))
        allowed[[index[c] for c in banned if c in index]]=False
        possible=shared[allowed[shared]]
        current=[index[mid] for mid in labels]
        scores[0,i]=conditional[np.ix_(current,possible)].max(initial=0)
        scores[1,i]=prior[possible].max(initial=0)
        if a.features:
            scores[2,i]=acoustic[key][possible].max(initial=0)
        elif a.acoustic_scores:
            scores[2,i]=acoustic[key]
        count_half[i]=sum(mid in vocabulary and mid in index and mid not in banned and sec>=.5
                          for mid,sec in strong[key].items())
        count_one[i]=sum(mid in vocabulary and mid in index and mid not in banned and sec>=1
                         for mid,sec in strong[key].items())
        if key in historical:
            r=historical[key]
            logged[i]=True
            archive_hits[i]=len({mid for name in r["our_guessed_labels"] if (mid:=name_to_mid.get(name_key(name))) in index
                                 and mid in vocabulary and mid not in banned and r["mturk"].get(name,0)>=3})
    ids=np.arange(n)
    selections={"historical":logged}
    for name,score in zip(method_names,scores):
        order=np.lexsort((ids,-score))
        selection=np.zeros(n,dtype=bool)
        selection[order[:k]]=True
        selections[name]=selection
    truth=count_half>0
    report={}
    for name,selected in selections.items():
        assert int(selected.sum())==k
        added=int((truth & selected & ~logged).sum())
        lost=int((truth & logged & ~selected).sum())
        bootstrap=np.random.default_rng(26121424).multinomial(n,[lost/n,added/n,1-(added+lost)/n],size=20000)
        report[name]=dict(archived_omission_clips=int((selected & (archive_hits>0)).sum()),
                          same_reference_inside_archive=int((selected&truth&logged).sum()),
                          independent_omission_clips=int((selected&truth).sum()),
                          independent_omission_clips_one_second=int((selected&(count_one>0)).sum()),
                          total_independently_confirmed_missing_classes=int(count_half[selected].sum()),
                          independently_confirmed_omission_clips_outside_archive=added,
                          delta_vs_historical=added-lost,
                          delta_ci95=np.quantile(bootstrap[:,1]-bootstrap[:,0],[.025,.975]).tolist())
    a.output.mkdir(exist_ok=True,parents=True)
    result=dict(weak_eval=len(test),strong_eval_with_events=len(strong),aligned_before_log_label_check=len(universe),
                historical_proposals=len(rows),historical_match_original=int(len(set(historical)&set(test))),
                historical_label_disagreement=len(weak_disagreement),eligible=n,review_budget=k,
                shared_label_vocabulary=len(shared),independent_positive=int(truth.sum()),
                report=report,input_sha256=source_hashes,**prediction_counts)
    (a.output/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    with (a.output/"clips.csv").open("w",newline="") as stream:
        w=csv.writer(stream)
        w.writerow(["video_id","start_ms","historically_reviewed","archived_omitted_classes",
                    "strong_omitted_classes_05s","strong_omitted_classes_1s",*[m+"_score" for m in method_names],
                    *[m+"_queue" for m in method_names]])
        for i,key in enumerate(eligible):
            w.writerow([*key,int(logged[i]),archive_hits[i],count_half[i],count_one[i],*scores[:,i],
                        *(int(selections[m][i]) for m in method_names)])
    if a.features:
        with (a.output/"acoustic_scores.csv").open("w",newline="") as stream:
            w=csv.writer(stream)
            w.writerow(["video_id","start_ms","acoustic_score"])
            w.writerows([(*key,scores[2,i]) for i,key in enumerate(eligible)])
    print(json.dumps(result,indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("input",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--features",type=Path)
    p.add_argument("--acoustic-scores",type=Path)
    main(p.parse_args())
