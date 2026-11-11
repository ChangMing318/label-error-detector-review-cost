"""Try pair sampling with an exact sign test on the discordant sampled pairs."""
import argparse
import csv
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.stats import hypergeom

from review_cost import exclusive, write


@lru_cache(maxsize=1500)
def cutoffs(size, sample):
    cuts = []
    for d in range(sample+1):
        possible = np.arange(d, size-sample+d+1)
        tail = hypergeom.sf(np.arange(d+1)[:, None]-1, possible, possible//2, d)
        eligible = np.flatnonzero(np.max(tail, axis=1) <= .05-1e-12) if d else []
        cuts.append(int(eligible[0]) if len(eligible) else d+1)
    return np.array(cuts)


def calculate(contrast, sample):
    size = len(contrast)
    positive = int((contrast == 1).sum())
    negative = int((contrast == -1).sum())
    if sample == size:
        return float(positive > negative)
    if not sample or not positive:
        return 0.
    total = positive+negative
    d = np.arange(sample+1)
    return float(hypergeom.pmf(d, size, total, sample) @
                 np.nan_to_num(hypergeom.sf(cutoffs(size, sample)-1, total, positive, d)))


def run(inputs, references, out):
    with np.load(references) as z:
        secondary = dict(z)
    rows = []
    for i, path in enumerate(sorted(inputs.iterdir())):
        if not (path/"target.npz").exists():
            continue
        with np.load(path/"target.npz") as z:
            ranks = z["rank"]
            truth = {1: z["errors"]}
        if path.name in secondary:
            truth[2] = secondary[path.name]
        with (path/"paired.csv").open(newline="", encoding="utf-8") as stream:
            choices = [r for r in csv.DictReader(stream)
                       if r["menu"] == "expanded" and float(r["budget"]) == .10]
        n = ranks.shape[1]
        budget = round(.10*n)
        k = math.ceil(budget/10)
        consensus = np.mean(-np.log((np.argsort(ranks, axis=1)+.5)/n), axis=0)
        consensus = np.r_[consensus, -1.]
        for j, row in enumerate(choices):
            r, h = ranks[[int(row[c]) for c in ("random", "difficulty")]]
            sides = exclusive(r[:budget], h[:budget+k-1])
            size = max(map(len, sides))
            padded = [np.pad(s, (0, size-len(s)), constant_values=n) for s in sides]
            rng = np.random.default_rng(np.random.SeedSequence([26112645, i, j]))
            ordered = [p[np.argsort(consensus[p], kind="stable")] for p in padded]
            variants = {"rank_matched": ordered, "rank_reversed": (ordered[0], ordered[1][::-1]),
                        "shuffled": [rng.permutation(p) for p in padded]}
            for reference, errors in truth.items():
                errors = np.r_[errors, False].astype(int)
                for method, (a, b) in variants.items():
                    contrast = errors[a]-errors[b]
                    for fraction in (.05, .10, .25):
                        cost = min(round(fraction*budget), sum(map(len, sides)))
                        m = size if cost == sum(map(len, sides)) else min(size, cost//2)
                        probability = calculate(contrast, m)
                        rows.append(dict(dataset=path.name, seed=int(row["seed"]), reference=reference,
                                         method=method, fraction=fraction, budget=budget,
                                         check_cap=cost, pairs=m, material=int(contrast.sum() > 0),
                                         probability=probability))
    write(out, rows)


def self_check():
    for size in (6, 9):
        for sample in range(1, size+1):
            for pos in range(size+1):
                for neg in range(pos, size-pos+1):
                    contrast = np.array([1]*pos+[-1]*neg+[0]*(size-pos-neg))
                    assert calculate(contrast, sample) <= .05+1e-12


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("reference2", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    self_check()
    run(args.inputs, args.reference2, args.out)
