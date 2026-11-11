"""Spend early-stopped audit checks and follow-up checks from the same cap."""
import argparse
import csv
import math
from pathlib import Path
from statistics import mean

import numpy as np

from futility_audit import extremes, schedule
from review_cost import audit_strata, bounds, interleave, write


DRAWS = 600


def fill(order, reviewed, budget, errors):
    reviewed = set(map(int, reviewed))
    for item in order:
        if len(reviewed) == budget:
            break
        reviewed.add(int(item))
    assert len(reviewed) == budget
    return int(errors[np.fromiter(reviewed, dtype=int)].sum())


def evaluate(random, hard, errors, b, rng):
    r, h, m = audit_strata(random, hard, b, math.floor(.10*b/2))
    direct_h = int(errors[hard[:b]].sum())
    direct_r = int(errors[random[:b]].sum())
    mixed = int(errors[interleave(random, hard, b)].sum())
    if m == 0:
        return dict(fixed_checks=0, early_checks=0., fixed_hits=direct_h, early_hits=direct_h,
                    direct_h=direct_h, direct_r=direct_r, interleave=mixed, paired_se=0.)
    x = np.arange(m+1)
    low, high = bounds(len(r), m)
    decision = (low-x)[:, None] > (high-x)[None, :]
    assert (np.diff(decision.astype(int), axis=0) >= 0).all()
    assert (np.diff(decision.astype(int), axis=1) <= 0).all()
    sample = [np.array([rng.choice(side, m, replace=False) for _ in range(DRAWS)])
              for side in (r, h)]
    cx, cy = [np.c_[np.zeros(DRAWS, dtype=int), errors[s].cumsum(axis=1)] for s in sample]
    final = decision[cx[:, -1], cy[:, -1]]
    taken = np.full((DRAWS, 2), m, dtype=int)
    live = np.ones(DRAWS, dtype=bool)
    for i, j in schedule((m, m), "proportional"):
        worst, best = extremes(decision, i, j)
        a, c = cx[:, i], cy[:, j]
        stop = live & (worst[a, c] | ~best[a, c])
        assert np.array_equal(worst[a[stop], c[stop]], final[stop])
        taken[stop] = i, j
        live[stop] = False
        if not live.any():
            break
    assert not live.any()
    fixed = np.where(final, errors[random[:b-m]].sum()+cy[:, -1],
                     errors[hard[:b-m]].sum()+cx[:, -1])
    early = np.array([fill(random if final[t] else hard,
                           np.r_[sample[0][t, :i], sample[1][t, :j]], b, errors)
                      for t, (i, j) in enumerate(taken)])
    return dict(fixed_checks=2*m, early_checks=float(taken.sum(1).mean()),
                fixed_hits=float(fixed.mean()), early_hits=float(early.mean()),
                direct_h=direct_h, direct_r=direct_r, interleave=mixed,
                paired_se=float(np.std(early-fixed, ddof=1)/np.sqrt(DRAWS)))


def run(inputs, second):
    for i, folder in enumerate(sorted(inputs.iterdir())):
        if not (folder/"target.npz").exists():
            continue
        with np.load(folder/"target.npz") as data:
            rank = data["rank"]
            truth = {1: data["errors"]}
        if folder.name in second:
            truth[2] = second[folder.name]
        with (folder/"paired.csv").open(newline="", encoding="utf-8") as stream:
            choices = [r for r in csv.DictReader(stream)
                       if r["menu"] == "expanded" and float(r["budget"]) == .10]
        n = rank.shape[1]
        b = round(.10*n)
        for j, row in enumerate(choices):
            r, h = rank[[int(row[col]) for col in ("random", "difficulty")]]
            for ref, errors in truth.items():
                rng = np.random.default_rng(np.random.SeedSequence([26112647, i, j]))
                yield dict(dataset=folder.name, seed=int(row["seed"]), reference=ref, n=n, B=b,
                           **evaluate(r, h, errors, b, rng))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    with np.load(args.second) as z:
        rows = list(run(args.inputs, dict(z)))
    write(args.out/"futility_deployment.csv", rows)
    summaries = []
    for ref in (1, 2):
        subset = [r for r in rows if r["reference"] == ref]
        summaries.append(dict(reference=ref, fixed_checks=sum(r["fixed_checks"] for r in subset),
                              early_checks=sum(r["early_checks"] for r in subset),
                              **{f"vs_{arm}": mean((r["early_hits"]-r[arm])*10000/r["n"] for r in subset)
                                 for arm in ("fixed_hits", "direct_h", "direct_r", "interleave")}))
    write(args.out/"futility_deployment_summary.csv", summaries)
