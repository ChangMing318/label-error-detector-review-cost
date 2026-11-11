"""Compare catch-up screening rules without using the target's reference to fit or select."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

import numpy as np

from assisted_audit import features, fit_pieces
from precision_audit import allocated
from review_cost import exclusive, sharper_power, write


P = json.loads(Path(__file__).with_name("screen_controls_protocol.json").read_text())
METHODS = P["primary_controls"]


def read_inputs(root, second):
    data = {}
    for folder in sorted(root.iterdir()):
        if not (folder/"target.npz").exists():
            continue
        with np.load(folder/"target.npz") as z:
            rank = z["rank"]
            references = {1: z["errors"].astype(bool)}
        if folder.name in second:
            references[2] = second[folder.name].astype(bool)
        n = rank.shape[1]
        percentiles = (np.argsort(rank, axis=1)+.5)/n
        design = features(rank)
        mean_log = design[:, 4]
        bucket = np.argsort(np.argsort(mean_log, kind="stable"), kind="stable")*5//n
        with (folder/"paired.csv").open(encoding="utf-8", newline="") as stream:
            choices = [row for row in csv.DictReader(stream) if row["menu"] == "expanded"]
        assert len(choices) == 30
        variants = dict(history_ridge=design, one_feature=design[:, [0, 4]],
                        rank_bins=np.eye(5)[bucket])
        pieces = {key: fit_pieces(x, references[1]) for key, x in variants.items()}
        panels = []
        for row in choices:
            frac = float(row["budget"])
            b = round(frac*n)
            k = math.ceil(b/10)
            indexes = [int(row[col]) for col in ("random", "difficulty")]
            u, v = (np.asarray(json.loads(row[f"{col}_source"])) for col in ("random", "difficulty"))
            margin = int(u[indexes[0]]-u[indexes[1]]-v[indexes[1]]+v[indexes[0]])
            proxies = dict(borda=1-percentiles.mean(axis=0), log_rank=mean_log/np.log(2*n),
                           top_votes=(percentiles <= b/n).mean(axis=0),
                           median_rank=1-design[:, 3],
                           two_rank=1-percentiles[indexes].mean(axis=0))
            for i, direction in enumerate(("hard_trails", "random_trails")):
                challenger, incumbent = rank[indexes[::1 if i == 0 else -1]]
                a, c = exclusive(challenger[:b], incumbent[:b+k-1])
                sizes = len(a), len(c)
                cost = min(round(.10*b), sum(sizes)) if sizes[0] else 0
                scores = {name: (float(p[a].sum())-float(p[c].sum()))/max(1, sum(sizes))
                          for name, p in proxies.items()}
                scores.update(audit_all=1., overlap_half=sizes[0]/b-.5,
                              source_margin=(-1 if i else 1)*margin)
                panels.append(dict(seed=int(row["seed"]), budget=frac, direction=direction,
                                   B=b, cost=cost, sets=(a, c), sizes=sizes, scores=scores))
        data[folder.name] = dict(pieces=pieces, variants=variants, references=references, panels=panels)
    return data


def model_for(data, excluded):
    models = {}
    for method in ("history_ridge", "one_feature", "rank_bins"):
        pieces = [x["pieces"][method] for name, x in data.items() if name not in excluded]
        a, b = (sum(p[i] for p in pieces) for i in range(2))
        if method == "rank_bins":
            models[method] = b / np.diag(a)
        else:
            models[method] = np.linalg.solve(a+.05*np.eye(len(a)), b)
    return models


def predictions(corpus, model):
    return {name: np.clip(corpus["variants"][name] @ coef, 0, 1) for name, coef in model.items()}


def scores(panel, predicted):
    a, c = panel["sets"]
    mass = max(1, len(a)+len(c))
    return dict(**panel["scores"], **{name: (p[a].sum()-p[c].sum())/mass
                                     for name, p in predicted.items()})


def learned_thresholds(data, held):
    minimum = {name: float("inf") for name in METHODS}
    for name, corpus in data.items():
        if name in held:
            continue
        model = model_for(data, {*held, name})
        predicted = predictions(corpus, model)
        truth = corpus["references"][1]
        for panel in corpus["panels"]:
            if (panel["budget"], panel["direction"]) != (.10, "hard_trails"):
                continue
            a, c = panel["sets"]
            if not panel["cost"] or truth[a].sum() <= truth[c].sum():
                continue
            for method, s in scores(panel, predicted).items():
                minimum[method] = min(minimum[method], float(s))
    return {method: s if np.isfinite(s) else -float("inf") for method, s in minimum.items()}


def run(data):
    names = sorted(data, key=lambda n: hashlib.sha256(n.encode()).hexdigest())
    for fold in range(4):
        held = names[fold::4]
        model = model_for(data, held)
        learned = learned_thresholds(data, held)
        for name in held:
            corpus = data[name]
            predicted = predictions(corpus, model)
            for panel in corpus["panels"]:
                a, c = panel["sets"]
                values = scores(panel, predicted)
                size = panel["sizes"]
                cost = panel["cost"]
                draws = allocated(size, cost, "proportional") if cost else (0, 0)
                policies = [("fixed", method, value > 0 or (method == "overlap_half" and value == 0))
                            for method, value in values.items()]
                policies.extend(("nested", method, value >= learned[method])
                                for method, value in values.items() if method != "audit_all")
                for ref, truth in corpus["references"].items():
                    hits = int(truth[a].sum()), int(truth[c].sum())
                    power = sharper_power(size, hits, draws, 1) if cost else 0.
                    for mode, method, audit in policies:
                        yield dict(dataset=name, fold=fold, reference=ref, seed=panel["seed"],
                                   budget=panel["budget"], direction=panel["direction"], mode=mode,
                                   method=method, B=panel["B"], eligible=int(cost>0),
                                   gap=hits[0]-hits[1], score=float(values[method]),
                                   threshold=learned[method] if mode=="nested" else 0.,
                                   baseline_cost=cost, cost=cost if audit else 0, baseline_power=power,
                                   power=power if audit else 0.)


def summarize(rows):
    paired = {r["dataset"] for r in rows if r["reference"] == 2}
    result = []
    for cohort in ("all_primary", "matched_primary", "matched_second"):
        ref = 2 if cohort == "matched_second" else 1
        by = {}
        for r in rows:
            if r["reference"] != ref or (cohort != "all_primary" and r["dataset"] not in paired):
                continue
            key = r["budget"], r["direction"], r["mode"], r["method"]
            by.setdefault(key, []).append(r)
        for (budget, direction, mode, method), values in sorted(by.items()):
            material = [r for r in values if r["gap"] > 0]
            lost = [r for r in material if not r["cost"]]
            result.append(dict(cohort=cohort, budget=budget, direction=direction, mode=mode,
                               method=method, baseline_cost=sum(r["baseline_cost"] for r in values),
                               cost=sum(r["cost"] for r in values), material=len(material),
                               material_lost=len(lost),
                               power=mean(r["power"] for r in material) if material else 0.,
                               baseline_power=mean(r["baseline_power"] for r in material) if material else 0.,
                               high_power=sum(r["power"] >= .8 for r in material),
                               expected_false_certificates=sum(r["power"] for r in values if r["gap"] <= 0),
                               lost_corpora=",".join(sorted({r["dataset"] for r in lost}))))
    return result


def deployment(rows, path):
    key = lambda row: (row["dataset"], int(row["seed"]), int(row["reference"]))
    with path.open(newline="", encoding="utf-8") as stream:
        paid = {key(row): row for row in csv.DictReader(stream)
                if float(row["audit_fraction_limit"]) == .10}
    outcomes = {}
    for row in rows:
        if (row["mode"], row["budget"], row["direction"]) != ("fixed", .10, "hard_trails"):
            continue
        q = paid[key(row)]
        value = float(q["expected_yield"] if row["cost"] else q["yield_difficulty"])
        changes = {f"vs_{arm}": 10000*(value-float(q[arm]))/int(q["n"]) for arm in (
            "expected_yield", "yield_random", "yield_difficulty", "yield_interleave")}
        outcomes.setdefault((row["method"], row["reference"]), []).append(changes)
    for (name, reference), values in sorted(outcomes.items()):
        yield dict(method=name, reference=reference,
                   **{column: mean(r[column] for r in values) for column in values[0]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("reference2", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--deployment", type=Path)
    args = parser.parse_args()
    with np.load(args.reference2) as z:
        data = read_inputs(args.inputs, dict(z))
    rows = list(run(data))
    args.out.mkdir(parents=True, exist_ok=True)
    write(args.out/"screen_controls.csv", rows)
    write(args.out/"screen_controls_summary.csv", summarize(rows))
    if args.deployment:
        write(args.out/"screen_controls_deployment.csv", list(deployment(rows, args.deployment)))
