"""Replay Wake Vision class allocations at equal budget and equal yield."""
import argparse
import csv
from itertools import product
from pathlib import Path
import random
from statistics import mean

import numpy as np

from review_cost import signed_debt, write


CAPS = (.05, .10, .20)
SEEDS = range(55001, 55011)
POLICIES = ("historical", "population", "balanced", "25pct_positive", "75pct_positive")


def read_compact(path, panel):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len({r["image"] for r in rows}) == len(rows)
    label = np.array([int(r["original"]) for r in rows])
    errors = np.array([r["original"] != r["reference"] for r in rows])
    covered = np.array([r[f"hist_{panel}"] == "1" for r in rows])
    score = np.array([float(r[f"machine_{panel}"] or 0) for r in rows])
    return label, errors, covered, score


def curves(data, seed):
    label, errors, covered, score = data
    n = len(label)
    rng = random.Random(seed)
    groups = [sorted(np.flatnonzero(label == c), key=lambda i: ((2*c-1)*score[i], rng.random()))
              for c in (0, 1)]
    prefix = lambda values: [np.r_[0, np.cumsum(values[g])] for g in groups]
    full, visible = prefix(errors), prefix(errors & covered)
    caps = np.arange(n+1)
    fractions = (label[covered].mean(), label.mean(), .5, .25, .75)
    for policy, fraction in zip(POLICIES, fractions):
        q = np.fromiter((round(fraction*int(t)) for t in caps), dtype=int)
        q = np.minimum(len(groups[1]), np.maximum(caps-len(groups[0]), q))
        yield policy, (full[0][caps-q]+full[1][q], visible[0][caps-q]+visible[1][q])


def compare(r, h, budget):
    debt = signed_debt(r[0], h[0], budget)
    if debt:
        leader, trailer = (r[0], h[0]) if debt > 0 else (h[0], r[0])
        t = budget+abs(debt)
        assert trailer[t-1] < leader[budget] <= trailer[t]
    return dict(challenger_hits=int(r[0][budget]), historical_hits=int(h[0][budget]),
                gain=int(r[0][budget]-h[0][budget]),
                signed_extra_checks=debt, relative_extra_checks=debt/budget,
                historical_visibility_gain=int(r[1][budget]-h[1][budget]))


def winners(ranks, budget, reference):
    best = max(y[reference][budget] for y in ranks.values())
    return [p for p, y in ranks.items() if y[reference][budget] == best]


def run(root):
    with (root/"wake_allocations.csv").open(newline="", encoding="utf-8") as stream:
        reference = {(r["split"], r["panel"], int(r["seed"]), float(r["budget_share"]), r["policy"]):
                     (int(r["independent_errors"]), int(r["historically_visible_errors"]))
                     for r in csv.DictReader(stream)}
    for panel in ("family", "person"):
        data = {s: read_compact(root/f"wake_{s}_compact.csv", panel) for s in ("val", "test")}
        for seed in SEEDS:
            ranks = {s: dict(curves(d, seed)) for s, d in data.items()}
            for frac in CAPS:
                sizes = {s: round(len(d[0])*frac) for s, d in data.items()}
                for split, rr in ranks.items():
                    budget = sizes[split]
                    for p, outcome in rr.items():
                        assert tuple(o[budget] for o in outcome) == reference[split, panel, seed, frac, p]
                    yield dict(split=split, panel=panel, seed=seed, fraction=frac, B=budget,
                               comparison="fixed_allocation", challenger="population", historical="historical",
                               **compare(rr["population"], rr["historical"], budget))
                independent = winners(ranks["val"], sizes["val"], 0)
                visible = winners(ranks["val"], sizes["val"], 1)
                for r, h in product(independent, visible):
                    yield dict(split="test", panel=panel, seed=seed, fraction=frac, B=sizes["test"],
                               comparison="validation_selection", challenger=r, historical=h,
                               **compare(ranks["test"][r], ranks["test"][h], sizes["test"]))


def summarize(rows):
    cells = {}
    for r in rows:
        key = tuple(r[k] for k in ("split", "panel", "fraction", "B", "comparison"))
        cells.setdefault(key, {}).setdefault(r["seed"], []).append(r)
    for key, by_seed in cells.items():
        assert len(by_seed) == 10
        metrics = ("challenger_hits", "historical_hits", "gain", "signed_extra_checks",
                   "relative_extra_checks", "historical_visibility_gain")
        result = {k: mean(mean(r[k] for r in group) for group in by_seed.values()) for k in metrics}
        debts = [r["signed_extra_checks"] for group in by_seed.values() for r in group]
        yield dict(zip(("split", "panel", "fraction", "B", "comparison"), key), **result,
                   min_extra=min(debts), max_extra=max(debts),
                   positive_seeds=sum(mean(r["signed_extra_checks"] for r in group)>0
                                      for group in by_seed.values()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    rows = list(run(args.inputs))
    args.out.mkdir(parents=True, exist_ok=True)
    write(args.out/"wake_cost_trials.csv", rows)
    write(args.out/"wake_cost_summary.csv", list(summarize(rows)))
