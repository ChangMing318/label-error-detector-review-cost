"""Freeze blind detector choices, then judge them at one human review budget."""
import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from external_crowds import fashion, feature_models, go, key, majority, rank, role_of, table, take, vote_order, wiki
from operating_envelope import curve, forecast

P = json.loads(Path(__file__).with_name("blind_rate_protocol.json").read_text())
ARMS = ("auc", "budget", "matched", "votes")


def text_group(text):
    return hashlib.sha256(" ".join(text.lower().split()).encode()).hexdigest()


def make_rows(items, votes):
    roles = defaultdict(list)
    for item, (text, tag) in sorted(items.items()):
        if len(votes[item]) < 5:
            continue
        group = text_group(text)
        selected = vote_order(item, votes[item], 5)
        roles[role_of(group)].append((item, group, text, tuple(selected[:3]), tuple(selected[3:]), tag))
    return roles


def d3(root):
    regions = {r["rater_id"]: r["Region"] for r in table(root/"d3-raters.csv")}
    items = {r["item_id"]: (r["text"], r["category"]) for r in table(root/"d3-items.csv")}
    votes, by_region = defaultdict(dict), defaultdict(lambda: defaultdict(dict))
    for row in table(root/"d3-ratings.csv"):
        if row["rating_binary"] not in ("0", "1", "0.0", "1.0"):
            continue
        i, w = row["item_id"], row["rater_id"]
        value = int(float(row["rating_binary"]))
        votes[i][w] = value
        by_region[regions[w]][i][w] = value
    roles = make_rows(items, votes)
    extra = {"random_to_selected": (roles["fit"], [r for r in roles["history"] if r[5] == "random"],
                                    [r for r in roles["target"] if r[5] != "random"])}
    for region, panel in sorted(by_region.items()):
        extra["region_" + region.replace(" ", "_")] = (roles["fit"], roles["history"], make_rows(items, panel)["target"])
    return roles, extra


def mhs(root):
    items, votes = {}, defaultdict(dict)
    with gzip.open(root/"mhs-selected.csv.gz", "rt", encoding="utf-8", newline="") as stream:
        for r in csv.DictReader(stream):
            if r["hatespeech"] not in ("0", "1", "2", "0.0", "1.0", "2.0"):
                continue
            i = r["comment_id"]
            items[i] = (r["text"], r["platform"])
            votes[i][r["annotator_id"]] = int(float(r["hatespeech"]) == 2)
    roles = make_rows(items, votes)
    extra = {f"platform_{tag}": ([r for r in roles["fit"] if r[5] != tag],
                                 [r for r in roles["history"] if r[5] != tag],
                                 [r for r in roles["target"] if r[5] == tag])
             for tag in sorted({r[5] for r in roles["target"]})}
    return roles, extra


def ucc(root):
    items, votes = {}, defaultdict(dict)
    labels = {"no": 0, "yes": 1, "false": 0, "true": 1, "0": 0, "1": 1,
              "unhealthy": 0, "healthy": 1}
    with gzip.open(root/"unhealthy_full.csv.gz", "rt", encoding="utf-8-sig", newline="") as stream:
        for r in csv.DictReader(stream):
            value = r["healthy"].strip().lower()
            if value not in labels or not r["_worker_id"]:
                continue
            i = r["_unit_id"]
            items[i] = (r["comment"], "full")
            votes[i][r["_worker_id"]] = labels[value]
    return make_rows(items, votes), {}


def datasets(root, name):
    loaders = {"go": go, "wiki": wiki, "fashion": fashion}
    if name in loaders:
        roles = loaders[name](root)
        extra = {}
        if name == "wiki":
            extra["random_to_blocked"] = (roles["fit"], [r for r in roles["history"] if r[5] == "random"],
                                         [r for r in roles["target"] if r[5] == "blocked"])
    else:
        roles, extra = {"d3": d3, "mhs": mhs, "ucc": ucc}[name](root)
    return {"full": tuple(roles[k] for k in ("fit", "history", "target")), **extra}


def auc(e, scores):
    m, n = int(e.sum()), len(e)
    return (rankdata(scores, axis=1)[:, e].sum(1) - m*(m+1)/2)/(m*(n-m))


def vote_disagreement(votes):
    return (votes[:, 1:] != votes[:, :1]).sum(1)


def score_plans(history, target_public, hs, ts, classes, names):
    # target_public has no target reference labels.
    hv = np.array([r[3] for r in history])
    tv = np.array([r[3] for r in target_public])
    hy = np.array([r[4][0] for r in history])
    hc = np.array([majority(row) for row in hv])
    e = hc != hy
    if not 0 < e.sum() < len(e):
        raise ValueError("source needs both disagreements and agreements")
    h_order = rank(hs, classes, hc, np.argsort([key("tie:"+r[0]) for r in history]))
    t_order = rank(ts, classes, tv[:, 0], np.argsort([key("tie:"+r[0]) for r in target_public]))
    n = len(tv)
    b = round(P["cap"]*n)
    cap = b/n
    prefix = t_order[:, :b]
    profiles = [curve(e, r) for r in h_order]
    sv, tvotes = vote_disagreement(hv), vote_disagreement(tv)
    he = hv[:, 0] != hy
    source_risk = [(he[sv == k].sum()+.5)/((sv == k).sum()+1) if np.any(sv == k) else he.mean()
                   for k in range(3)]
    rate = float(np.asarray(source_risk)[tvotes].mean())
    values = {
        "auc": auc(e, hs[:, np.arange(len(e)), np.searchsorted(classes, hc)]),
        "budget": np.array([forecast(p, e.mean(), cap) for p in profiles]),
        "matched": np.array([forecast(p, rate, cap) for p in profiles]),
        "votes": tvotes[prefix].sum(1),
    }
    a = max(2, min(100, 2*round(.05*b)))
    paid_cap = (b-a)/(n-a)
    lookup = [int(np.argmax([forecast(p, h/a, paid_cap) for p in profiles])) for h in range(a+1)]
    plan = dict(names=names, n=n, b=b, historical_rate=float(e.mean()), target_rate_estimate=rate,
                witness_risk=source_risk, choices={k: int(v.argmax()) for k, v in values.items()},
                values={k: v.tolist() for k, v in values.items()}, paid_audit=a, paid_lookup=lookup)
    return plan, t_order, profiles


def interval(delta, groups):
    size = np.bincount(groups)
    hits = np.bincount(groups, weights=delta)
    signatures, counts = np.unique(np.stack((size, hits), axis=1), return_counts=True, axis=0)
    rng = np.random.default_rng(P["bootstrap_seed"])
    draw = rng.multinomial(len(size), counts/len(size), size=P["bootstrap_draws"])
    sims = 10000*(draw @ signatures[:, 1])/(draw @ signatures[:, 0])
    point = float(10000*delta.mean())
    return dict(mean=point, ci95=np.quantile(sims, [.025, .975]).tolist(),
                p_positive=(1+int(np.sum(sims >= 2*point-1e-12)))/(len(sims)+1))


def paid_orders(e, order, plan):
    b, a = plan["b"], plan["paid_audit"]
    for seed in range(P["audit_seed"], P["audit_seed"]+P["audit_draws"]):
        audit = np.random.default_rng(seed).choice(len(e), a, replace=False)
        chosen = plan["paid_lookup"][int(e[audit].sum())]
        queue = np.r_[audit, order[chosen][~np.isin(order[chosen], audit)][:b-a]]
        assert len(np.unique(queue)) == b
        yield seed, chosen, int(e[audit].sum()), int(e[queue].sum())


def judge(plan, order, profiles, claim, references, groups):
    results = []
    prefix = order[:, :plan["b"]]
    rows = []
    for r, reference in enumerate(references.T):
        e = claim != reference
        n = len(e)
        by_candidate = e[prefix].sum(1)
        oracle = int(by_candidate.argmax())
        known = int(np.argmax([forecast(p, e.mean(), plan["b"]/n) for p in profiles]))
        choices = {**plan["choices"], "known_rate": known, "oracle": oracle}
        hits = {k: int(by_candidate[j]) for k, j in choices.items()}
        outcomes = np.zeros((len(choices), n), dtype=np.int8)
        for i, j in enumerate(choices.values()):
            outcomes[i, prefix[j]] = e[prefix[j]]
        contrasts = {}
        for arm, baseline in (("budget", "auc"), ("matched", "auc"), ("matched", "budget"),
                              ("matched", "votes"), ("known_rate", "budget")):
            k, j = [list(choices).index(x) for x in (arm, baseline)]
            contrasts[f"{arm}-{baseline}"] = interval(outcomes[k]-outcomes[j], groups)
        paid = list(paid_orders(e, order, plan))
        rows.append((e, outcomes, np.asarray(paid)))
        results.append(dict(reference=r, target_rate=float(e.mean()), choices=choices, hits=hits,
                            candidate_hits=by_candidate.tolist(), contrasts=contrasts,
                            regrets={k: float(10000*(hits["oracle"]-v)/n) for k, v in hits.items()},
                            paid_net=float(10000*(np.mean([row[-1] for row in paid])-hits["budget"])/n),
                            paid_switches=sum(row[1] != choices["budget"] for row in paid)))
    return results, rows


def write(path, result):
    path.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    complete = {}
    for name, roles in datasets(args.data, args.dataset).items():
        sizes = list(map(len, roles))
        if min(sizes) < P["minimum_fit_history_target"]:
            complete[name] = dict(status="below_fixed_support", eligible=sizes)
            continue
        f, h, t = [take(r, tag) for r, tag in zip(roles, ("fit", "history", "target"))]
        public = [r[:4] for r in t]
        names, scores, classes = feature_models(f, h+public)
        if len(classes) < 2 or any(v not in classes for r in h+public for v in r[3]):
            complete[name] = dict(status="missing_fit_classes", eligible=sizes)
            continue
        plan, order, profiles = score_plans(h, public, scores[:, :len(h)], scores[:, len(h):], classes, names)
        plan.update(eligible=sizes, fit_n=len(f), source_n=len(h))
        plan_path = args.out/f"{name}.plan.json"
        write(plan_path, plan)
        digest = hashlib.sha256(plan_path.read_bytes()).hexdigest()
        reference = np.array([r[4] for r in t])
        _, groups = np.unique([r[1] for r in public], return_inverse=True)
        result, rows = judge(plan, order, profiles, np.array([r[3][0] for r in public]), reference, groups)
        np.savez_compressed(args.out/f"{name}.receipts.npz", groups=groups, prefix=order[:, :plan["b"]],
                            errors=np.stack([r[0] for r in rows]), outcomes=np.stack([r[1] for r in rows]),
                            paid=np.stack([r[2] for r in rows]))
        complete[name] = dict(status="complete", plan_sha256=digest, references=result)
        print(args.dataset, name, "committed and judged", len(t), flush=True)
    write(args.out/"summary.json", complete)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=("go", "wiki", "fashion", "d3", "mhs", "ucc"))
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    run(p.parse_args())
