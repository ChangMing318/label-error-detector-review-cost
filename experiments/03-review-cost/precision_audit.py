"""Compare finite-population audit statistics at the same label cost."""
import argparse
import csv
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.stats import betabinom, hypergeom

from review_cost import exclusive, write
from threshold import allocation


P = json.loads(Path(__file__).with_name("precision_protocol.json").read_text())
ALPHA = P["alpha"]


@lru_cache(maxsize=1500)
def pmf(size, sample):
    if not size:
        return np.ones((1, 1))
    return hypergeom.pmf(np.arange(sample+1), size, np.arange(size+1)[:, None], sample)


def statistics(sizes, draws):
    nr, nh = sizes
    mr, mh = draws
    x, y = np.arange(mr+1)[:, None], np.arange(mh+1)[None, :]
    raw = nr*max(1, mh)*x - nh*max(1, mr)*y
    # Fit an approximate common total under the null. This variance is only
    # used to order outcomes. The rejection probability is calibrated exactly.
    total = (x+y+.5)/(mr/max(1, nr)+mh/max(1, nh)+1/max(1, min(nr, nh)))
    total = np.clip(total, 0, min(nr, nh))
    variance = sum(size*size/max(1, m)*(size-m)/max(1, size-1) *
                   (total/max(1, size))*(1-total/max(1, size))
                   for size, m in zip(sizes, draws))
    pooled = raw / np.sqrt(variance + 1)

    posterior = []
    for size, m in zip(sizes, draws):
        audited = np.arange(m+1)[:, None]
        rest = np.arange(size+1) - audited
        posterior.append(betabinom.pmf(rest, size-m, audited+.5, m-audited+.5))
    hard_cdf = np.c_[np.zeros(mh+1), np.cumsum(posterior[1], axis=1)]
    posterior_win = posterior[0] @ hard_cdf[:, np.minimum(np.arange(nr+1), nh+1)].T

    # Signed square-root likelihood ratio, constrained at the finite null.
    pr, ph = pmf(nr, mr), pmf(nh, mh)
    prefix_r = np.maximum.accumulate(pr, axis=0)
    null = np.zeros((mr+1, mh+1))
    for h, dist in enumerate(ph):
        np.maximum(null, prefix_r[min(nr, h), :, None]*dist, out=null)
    unrestricted = pr.max(0)[:, None]*ph.max(0)[None, :]
    likelihood = np.log(unrestricted / np.maximum(null, np.finfo(float).tiny))
    likelihood = np.where(raw <= 0, -np.abs(raw)/max(1, np.max(np.abs(raw))), likelihood)
    # Ensure increasing-in-x, decreasing-in-y upper sets. That property
    # permits exact calibration on the boundary of the composite null.
    return {name: np.maximum.accumulate(np.maximum.accumulate(
                np.round(score, 13), axis=0)[:, ::-1], axis=1)[:, ::-1]
            for name, score in zip(P["methods"], (raw, pooled, posterior_win, likelihood))}


def critical(score, pr, ph, threshold=1):
    nr, nh = len(pr)-1, len(ph)-1
    hard = np.arange(max(0, 1-threshold), nh+1)
    if not len(hard):
        return np.ones(score.shape, dtype=bool), 0.
    random = np.minimum(nr, hard + threshold - 1)
    pr0 = pr[random]
    cdf0 = np.c_[np.zeros(len(hard)), ph[hard].cumsum(axis=1)]
    grid = np.unique(score)
    left, right = 0, len(grid)
    while left < right:
        i = (left+right)//2
        cut = np.sum(score >= grid[i], axis=1)
        size = float(np.max(np.sum(pr0*cdf0[:, cut], axis=1)))
        if size <= ALPHA-1e-12:
            right = i
        else:
            left = i+1
    reject = score >= grid[left] if left < len(grid) else np.zeros(score.shape, dtype=bool)
    cut = reject.sum(1)
    size = float(np.max(np.sum(pr0*cdf0[:, cut], axis=1)))
    return reject, size


@lru_cache(maxsize=1800)
def tests(sizes, draws):
    nr, nh = sizes
    mr, mh = draws
    pr, ph = pmf(nr, mr), pmf(nh, mh)
    if sum(draws) == sum(sizes):
        reject = np.arange(mr+1)[:, None] > np.arange(mh+1)[None, :]
        return {name: (reject, 0.) for name in P["methods"]}
    if not sum(draws) or not nr:
        return {name: (np.zeros((mr+1, mh+1), dtype=bool), 0.) for name in P["methods"]}
    return {name: critical(score, pr, ph) for name, score in statistics(sizes, draws).items()}


def allocated(sizes, cost, design):
    if not cost:
        return 0, 0
    if design == "proportional":
        return tuple(map(int, allocation(sizes, [cost])[0]))
    nr, nh = sizes
    desired = round(cost*{"balanced": .5, "random60": .6, "difficulty60": .4}[design])
    floor, ceil = max(0, cost-nh), min(nr, cost)
    if nr and nh and cost >= 2:
        floor, ceil = max(1, floor), min(cost-1, ceil)
    x = min(ceil, max(floor, desired))
    return x, cost-x


def compare(sizes, hits, budget, frac):
    cost = min(round(budget*frac), sum(sizes))
    for design in P["allocation"]:
        draws = allocated(sizes, cost, design)
        assert sum(draws) == cost
        probabilities = [pmf(n, m)[h] for n, m, h in zip(sizes, draws, hits)]
        for name, (reject, size) in tests(sizes, draws).items():
            power = float(probabilities[0] @ reject @ probabilities[1])
            yield dict(allocation=design, statistic=name, cost=cost,
                       random_checks=draws[0], difficulty_checks=draws[1],
                       probability=min(1., power), worst_case_false_positive=size)


def run(inputs, reference_path, out):
    rows = []
    with np.load(reference_path) as z:
        second = dict(z)
    for path in sorted(inputs.iterdir()):
        if not (path/"target.npz").exists():
            continue
        with np.load(path/"target.npz") as data:
            ranks = data["rank"]
            references = {1: data["errors"].astype(bool)}
        if path.name in second:
            references[2] = second[path.name].astype(bool)
        with (path/"paired.csv").open(newline="", encoding="utf-8") as stream:
            choices = [r for r in csv.DictReader(stream)
                       if r["menu"] == "expanded" and float(r["budget"]) == .10]
        assert len(choices) == 10
        for choice in choices:
            n = ranks.shape[1]
            budget = round(.10*n)
            k = math.ceil(budget/10)
            r, h = ranks[[int(choice[col]) for col in ("random", "difficulty")]]
            sides = exclusive(r[:budget], h[:budget+k-1])
            sizes = tuple(map(len, sides))
            for reference, errors in references.items():
                hits = tuple(int(errors[s].sum()) for s in sides)
                for frac in P["audit_fractions"]:
                    for result in compare(sizes, hits, budget, frac):
                        rows.append(dict(dataset=path.name, seed=int(choice["seed"]), reference=reference,
                                         B=budget, extra_threshold=k, fraction=frac,
                                         random_only=sizes[0], difficulty_only=sizes[1],
                                         random_only_hits=hits[0], difficulty_only_hits=hits[1],
                                         material=int(hits[0] > hits[1]), **result))
        print(path.name, flush=True)
    write(out, rows)


def self_check():
    for sizes, draws in [((4, 5), (3, 3)), ((6, 3), (3, 2)), ((3, 0), (2, 0))]:
        pr, ph = [pmf(n, m) for n, m in zip(sizes, draws)]
        for reject, size in tests(sizes, draws).values():
            observed = pr @ reject @ ph.T
            under_null = np.arange(len(pr))[:, None] <= np.arange(len(ph))[None, :]
            assert np.isclose(observed[under_null].max(), size)
            assert size <= ALPHA


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("reference2", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    self_check()
    run(args.inputs, args.reference2, args.out)
