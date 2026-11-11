"""Stop an exact fixed-cap audit once its final decision is determined."""
import argparse
import json
from functools import lru_cache
from itertools import product
from pathlib import Path
from statistics import mean

import numpy as np
from scipy.stats import hypergeom

from precision_audit import allocated
from review_cost import exact_critical, sharper_power, write
from screen_controls import read_inputs


@lru_cache(maxsize=3000)
def terminal(sizes, draws):
    nr, nh = sizes
    mr, mh = draws
    x = np.arange(mr+1)[:, None]
    y = np.arange(mh+1)[None, :]
    if sum(draws) == sum(sizes):
        return x > y
    if not sum(draws) or not nr:
        return np.zeros((mr+1, mh+1), dtype=bool)
    return nr*max(1, mh)*x-nh*max(1, mr)*y >= exact_critical(sizes, draws, 1)


def schedule(draws, order):
    seen = [0, 0]
    yield tuple(seen)
    mr, mh = draws
    for _ in range(mr+mh):
        if seen[1] == mh:
            side = 0
        elif seen[0] == mr:
            side = 1
        elif order == "challenger_first":
            side = 0
        elif order == "default_first":
            side = 1
        else:
            side = int(seen[0]*mh > seen[1]*mr)
        seen[side] += 1
        yield tuple(seen)


def extremes(reject, i, j):
    mr, mh = np.array(reject.shape)-1
    # All unseen challenger labels miss and all unseen default labels hit
    # in the least favorable completion. Swap these in the best completion.
    worst = reject[:i+1, mh-j:]
    best = reject[mr-i:, :j+1]
    return worst, best


def early_decision(labels, reject, order):
    final = bool(reject[sum(labels[0]), sum(labels[1])])
    for i, j in schedule(tuple(map(len, labels)), order):
        worst, best = extremes(reject, i, j)
        x, y = sum(labels[0][:i]), sum(labels[1][:j])
        if worst[x, y] or not best[x, y]:
            return i+j, bool(worst[x, y]), final
    raise AssertionError("A complete audit must terminate")


def pmfs(size, hits, samples):
    if size == 0:
        return [np.ones(1)]
    return [hypergeom.pmf(np.arange(i+1), size, hits, i) for i in range(samples+1)]


@lru_cache(maxsize=5000)
def replay(sizes, hits, draws):
    reject = terminal(sizes, draws)
    px, py = [pmfs(n, h, d) for n, h, d in zip(sizes, hits, draws)]
    p = float(px[-1] @ reject @ py[-1])
    if sizes[0]:
        assert abs(p-sharper_power(sizes, hits, draws, 1)) < 1e-10
    result = {}
    for order in ("proportional", "challenger_first", "default_first"):
        spent = futility = 0.
        for i, j in schedule(draws, order):
            if i+j == sum(draws):
                break
            worst, best = extremes(reject, i, j)
            spent += float(px[i] @ (best & ~worst) @ py[j])
            if order == "proportional":
                futility += float(px[i] @ best @ py[j])
        result[order] = spent
        if order == "proportional":
            result["futility_only"] = futility
    return p, result


def self_check():
    for draws in ((2, 3), (3, 2)):
        sizes = (6, 7)
        for reject in (terminal(sizes, draws),
                       np.arange(draws[0]+1)[:, None] > np.arange(draws[1]+1)[None, :]):
            for labels in product(product((0, 1), repeat=draws[0]), product((0, 1), repeat=draws[1])):
                for order in ("proportional", "challenger_first", "default_first"):
                    cost, result, final = early_decision(labels, reject, order)
                    assert result == final and 0 <= cost <= sum(draws)


def run(data):
    for name, corpus in data.items():
        for panel in corpus["panels"]:
            a, c = panel["sets"]
            sizes = panel["sizes"]
            cost = panel["cost"]
            draws = allocated(sizes, cost, "proportional")
            for ref, truth in corpus["references"].items():
                hits = int(truth[a].sum()), int(truth[c].sum())
                power, expected = replay(sizes, hits, draws)
                yield dict(dataset=name, reference=ref, seed=panel["seed"], budget=panel["budget"],
                           direction=panel["direction"], B=panel["B"], fixed_checks=cost,
                           material=int(hits[0]>hits[1]), certificate_probability=power,
                           **expected)


def summarize(rows):
    paired = {r["dataset"] for r in rows if r["reference"] == 2}
    for cohort in ("all_primary", "matched_primary", "matched_second"):
        ref = 2 if cohort == "matched_second" else 1
        for cap, direction in product((.05, .10, .20), ("hard_trails", "random_trails")):
            selected = [r for r in rows if r["reference"] == ref and r["budget"] == cap
                        and r["direction"] == direction
                        and (cohort == "all_primary" or r["dataset"] in paired)]
            good = [r for r in selected if r["material"]]
            yield dict(cohort=cohort, budget=cap, direction=direction,
                       replays=len(selected), material=len(good),
                       mean_power=mean(r["certificate_probability"] for r in good) if good else 0.,
                       high_power=sum(r["certificate_probability"] >= .80 for r in good),
                       **{key: sum(r[key] for r in selected) for key in ("fixed_checks", "proportional",
                                   "futility_only", "challenger_first", "default_first")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("reference2", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    self_check()
    with np.load(args.reference2) as z:
        data = read_inputs(args.inputs, dict(z))
    rows = list(run(data))
    args.out.mkdir(parents=True, exist_ok=True)
    write(args.out/"futility_audit.csv", rows)
    write(args.out/"futility_audit_summary.csv", list(summarize(rows)))
