"""Measure the utility of item-linked witness information at equal cost."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression

from blind_rate import datasets, interval, P as BLIND, write
from external_crowds import feature_models, key, take

P = json.loads(Path(__file__).with_name("located_votes_protocol.json").read_text())


def inputs(rows, scores, classes):
    votes = np.array([r[3] for r in rows])
    claim = votes[:, 0]
    j = np.searchsorted(classes, claim)
    class_ind = np.eye(len(classes))[j]
    count = (votes[:, 1:] != claim[:, None]).sum(1)
    pattern = count + ((count == 2) & (votes[:, 1] == votes[:, 2]))
    signal = np.eye(4)[pattern]
    values = scores[:, np.arange(len(claim)), j]
    percentiles = rankdata(values, axis=1).T/(len(claim)+1)
    return claim, class_ind, count, signal, percentiles


def ordered(score, tie):
    return tie[np.argsort(-score[tie], kind="stable")]


def fit_model(cols, error):
    return LogisticRegression(C=1, max_iter=500).fit(np.column_stack(cols), error)


def queues(h, t, scores, classes, frozen, candidate_prefixes):
    h_claim, h_class, _, h_wit, h_scores = inputs(h, scores[:, :len(h)], classes)
    t_claim, t_class, t_count, t_wit, t_scores = inputs(t, scores[:, len(h):], classes)
    error = h_claim != np.array([r[4][0] for r in h])
    b = frozen["b"]
    tie = np.argsort([key("tie:"+r[0]) for r in t])
    hist = {
        "claim": (h_class,),
        "score": (h_class, h_scores),
        "witness": (h_class, h_wit),
        "full": (h_class, h_scores, h_wit),
    }
    public = {
        "claim": (t_class,),
        "score": (t_class, t_scores),
        "witness": (t_class, t_wit),
        "full": (t_class, t_scores, t_wit),
    }
    model = {arm: fit_model(fields, error) for arm, fields in hist.items()}
    value = {arm: model[arm].decision_function(np.column_stack(fields)) for arm, fields in public.items()}
    q = {arm: candidate_prefixes[frozen["choices"][arm]] for arm in ("matched", "votes", "budget", "auc")}
    q.update({arm: ordered(v, tie)[:b] for arm, v in value.items()})
    q["count"] = ordered(t_count, tie)[:b]
    score_order = ordered(value["score"], tie)
    q["count_then_score"] = ordered(t_count, score_order)[:b]
    groups = [np.flatnonzero(t_claim == c) for c in classes]
    shuffled = []
    for seed in range(P["permutation_seed"], P["permutation_seed"]+P["permutations"]):
        rng = np.random.default_rng(seed)
        new_wit = t_wit.copy()
        for ids in groups:
            new_wit[ids] = t_wit[rng.permutation(ids)]
        value = model["full"].decision_function(np.column_stack((t_class, t_scores, new_wit)))
        shuffled.append(ordered(value, tie)[:b])
    return q, np.stack(shuffled), t_claim


def evaluate(orders, shuffled, e, groups):
    outcomes = {}
    for arm, order in orders.items():
        hits = np.zeros(len(e), dtype=np.int8)
        hits[order] = e[order]
        outcomes[arm] = hits
    hits = {k: int(v.sum()) for k, v in outcomes.items()}
    comp = {k: interval(outcomes["full"]-v, groups) for k, v in outcomes.items() if k != "full"}
    permuted = e[shuffled].sum(1)
    p = (1+int((permuted >= hits["full"]).sum()))/(len(permuted)+1)
    out = dict(hits=hits, contrasts=comp, permuted_hits=float(permuted.mean()),
               linked_minus_permuted=float(10000*(hits["full"]-permuted.mean())/len(e)),
               permutation_p=p)
    return out, permuted


def run(args):
    summaries = {}
    args.out.mkdir(parents=True, exist_ok=True)
    blind = args.blind/args.dataset
    completed = json.loads((blind/"summary.json").read_text())
    for name, roles in datasets(args.data, args.dataset).items():
        if completed[name]["status"] != "complete":
            continue
        f, h, t = [take(r, tag) for r, tag in zip(roles, ("fit", "history", "target"))]
        public = [r[:4] for r in t]
        frozen = json.loads((blind/f"{name}.plan.json").read_text())
        _, scores, classes = feature_models(f, h+public)
        with np.load(blind/f"{name}.receipts.npz") as z:
            orders, shuffled, claim = queues(h, public, scores, classes, frozen, z["prefix"])
        paths = [*orders.values(), *shuffled]
        assert all(len(np.unique(q)) == frozen["b"] for q in paths)
        plan = args.out/f"{name}.plan.npz"
        np.savez_compressed(plan, arms=np.array(list(orders)), orders=np.stack(list(orders.values())),
                            shuffled=shuffled)
        digest = hashlib.sha256(plan.read_bytes()).hexdigest()
        ref = np.array([r[4] for r in t])
        e = (ref != claim[:, None]).T
        _, groups = np.unique([r[1] for r in public], return_inverse=True)
        rows = [evaluate(orders, shuffled, panel, groups) for panel in e]
        np.savez_compressed(args.out/f"{name}.judge.npz", errors=e, groups=groups,
                            shuffled_hits=np.stack([r[1] for r in rows]))
        summaries[name] = dict(n=len(t), b=frozen["b"], plan_sha256=digest, references=[r[0] for r in rows])
        print(args.dataset, name, summaries[name]["references"][0]["hits"], flush=True)
    write(args.out/"summary.json", summaries)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=("go", "wiki", "fashion", "d3", "ucc"))
    for k in ("data", "out", "blind"):
        p.add_argument("--"+k, type=Path, required=True)
    run(p.parse_args())
