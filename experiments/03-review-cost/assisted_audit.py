"""Evaluate rank-assisted audit scores with strictly held-out corpora."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from precision_audit import allocated
from review_cost import exclusive, write


P = json.loads(Path(__file__).with_name("assisted_protocol.json").read_text())


def features(ranks):
    n = ranks.shape[1]
    percentiles = (np.argsort(ranks, axis=1)+.5)/n
    sorted_ranks = np.sort(percentiles, axis=0)
    return np.stack(
        [np.ones(n), sorted_ranks[0], sorted_ranks[-1], np.median(sorted_ranks, axis=0),
         np.mean(-np.log(percentiles), axis=0),
         *[(percentiles <= t).mean(axis=0) for t in (.01, .02, .05, .10, .20, .40)]],
        axis=1)


def read_panels(inputs, references):
    result = {}
    for index, path in enumerate(sorted(inputs.iterdir())):
        if not (path/"target.npz").exists():
            continue
        with np.load(path/"target.npz") as data:
            ranks, errors = data["rank"], data["errors"].astype(bool)
        with (path/"paired.csv").open(newline="", encoding="utf-8") as stream:
            choices = [r for r in csv.DictReader(stream)
                       if r["menu"] == "expanded" and float(r["budget"]) == .10]
        n = len(errors)
        budget = round(.10*n)
        k = math.ceil(budget/10)
        panels = []
        for j, row in enumerate(choices):
            r, h = ranks[[int(row[col]) for col in ("random", "difficulty")]]
            sides = exclusive(r[:budget], h[:budget+k-1])
            sizes = tuple(map(len, sides))
            cost = min(round(budget*P["audit_fraction"]), sum(sizes))
            allocation = allocated(sizes, cost, "proportional")
            rng = np.random.default_rng(np.random.SeedSequence([P["seed"], index, j]))
            sample = [np.array([rng.choice(s, m, replace=False) for _ in range(P["draws"])])
                      for s, m in zip(sides, allocation)]
            panels.append((int(row["seed"]), sides, sample, budget, cost))
        truth = {1: errors}
        if path.name in references:
            truth[2] = references[path.name].astype(bool)
        result[path.name] = (features(ranks), truth, panels)
    return result


def fit_pieces(x, y):
    return x.T @ x / len(x), x.T @ y / len(x)


def estimate(panel, predicted, errors):
    seed, sides, samples, budget, cost = panel
    n = np.array([len(s) for s in sides])
    m = np.array([s.shape[1] for s in samples])
    observed = [errors[s] for s in samples]
    marginal = np.array([p.mean(1) if count else np.zeros(P["draws"])
                         for p, count in zip(observed, m)])
    forecast = np.array([predicted[s].sum() for s in sides])
    residuals = [y-predicted[s] for y, s in zip(observed, samples)]
    avg_residual = np.array([e.mean(1) if count else np.zeros(P["draws"])
                             for e, count in zip(residuals, m)])
    signs = np.array([1, -1])
    true = int(signs @ [errors[s].sum() for s in sides])
    raw = signs @ (n[:, None]*marginal)
    residual = signs @ (forecast[:, None]+n[:, None]*avg_residual)
    shrunk = signs @ (forecast[:, None]+(n*m/(m+4))[:, None]*avg_residual)
    guess = np.full(P["draws"], signs @ forecast)
    eligible = n[0] > 0 and cost > 0
    return dict(seed=seed, B=budget, cost=cost, true_gap=true,
                material=int(true > 0), eligible=eligible,
                scores={name: value/max(1, n.sum()) for name, value in
                        zip(P["methods"], (raw, guess, residual, shrunk))})


def weighted_threshold(controls, name, alpha):
    per_target = {}
    for corpus, outcome in controls:
        per_target.setdefault(corpus, []).append(outcome["scores"][name])
    pooled = np.concatenate([s for outcomes in per_target.values() for s in outcomes])
    weights = np.concatenate([np.full(len(s), 1/(len(per_target)*len(outcomes)*len(s)))
                              for outcomes in per_target.values() for s in outcomes])
    order = np.argsort(pooled, kind="stable")
    limit = np.searchsorted(np.cumsum(weights[order]), 1-alpha)
    return float(pooled[order[limit]])


def run(inputs, references, out):
    with np.load(references) as z:
        data = read_panels(inputs, dict(z))
    pieces = {name: fit_pieces(x, truth[1]) for name, (x, truth, _) in data.items()}
    whole = tuple(sum(p[i] for p in pieces.values()) for i in range(2))
    rows = []
    for heldout, (x, truth, panels) in data.items():
        fit = tuple(a-b for a, b in zip(whole, pieces[heldout]))
        def predict(features, parts):
            m, y = parts
            return np.clip(features @ np.linalg.solve(m+.05*np.eye(len(m)), y), 0, 1)
        fitted = predict(x, fit)
        controls = []
        for other, (other_x, other_truth, other_panels) in data.items():
            if other == heldout:
                continue
            other_fit = tuple(a-b for a, b in zip(fit, pieces[other]))
            other_prediction = predict(other_x, other_fit)
            controls.extend((other, q) for p in other_panels
                            if not (q := estimate(p, other_prediction, other_truth[1]))["material"]
                            and q["eligible"])
        thresholds = {name: weighted_threshold(controls, name, .05) for name in P["methods"]}
        for reference, errors in truth.items():
            for panel in panels:
                q = estimate(panel, fitted, errors)
                for name, score in q["scores"].items():
                    rows.append(dict(dataset=heldout, reference=reference, method=name,
                                     **{key: v for key, v in q.items() if key != "scores"},
                                     threshold=thresholds[name],
                                     alert=float((score>thresholds[name]).mean()) if q["eligible"] else 0.,
                                     rmse_gap=float(np.sqrt(np.mean(
                                         (score*sum(map(len, panel[1]))-q["true_gap"])**2)))))
        print(heldout, flush=True)
    write(out, rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("reference2", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    run(args.inputs, args.reference2, args.out)
