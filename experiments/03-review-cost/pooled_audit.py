"""Distribute a fixed audit budget across a batch of unseen targets."""
import argparse
import hashlib
import heapq
import json
import math
from pathlib import Path

import numpy as np

from assisted_audit import fit_pieces, read_panels
from precision_audit import allocated, pmf, tests
from review_cost import write


P = json.loads(Path(__file__).with_name("pooled_audit_protocol.json").read_text())


def certificate(sizes, hits, cost):
    if not sizes[0] or not cost:
        return 0., 0.
    draws = allocated(sizes, cost, "proportional")
    reject, size = tests(sizes, draws)["count"]
    p, q = (pmf(n, m)[h] for n, m, h in zip(sizes, draws, hits))
    return min(1., float(p @ reject @ q)), size


def allocate(cohort, method):
    if method == "positive_only":
        return [q["baseline_cost"] if q["priority"] > 0 else 0 for q in cohort]
    paid = [q["audit_cost"] for q in cohort]
    remaining = sum(q["baseline_cost"] for q in cohort)-sum(paid)
    if method == "descending":
        for i in sorted(range(len(cohort)), key=lambda i: -cohort[i]["priority"]):
            increment = min(remaining, cohort[i]["max_cost"]-paid[i])
            paid[i] += increment
            remaining -= increment
    else:
        extra = [0]*len(cohort)
        weights = [q["baseline_cost"]*math.exp(8*q["priority"]) for q in cohort]
        queue = [(-w, i) for i, w in enumerate(weights) if paid[i] < cohort[i]["max_cost"]]
        heapq.heapify(queue)
        while remaining:
            _, i = heapq.heappop(queue)
            extra[i] += 1
            paid[i] += 1
            remaining -= 1
            if paid[i] < cohort[i]["max_cost"]:
                heapq.heappush(queue, (-weights[i]/(extra[i]+1), i))
    assert remaining == 0 and sum(paid) == sum(q["baseline_cost"] for q in cohort)
    return paid


def run(inputs, references, out):
    with np.load(references) as z:
        data = read_panels(inputs, dict(z))
    names = sorted(data, key=lambda name: hashlib.sha256(name.encode()).hexdigest())
    pieces = {name: fit_pieces(x, truth[1]) for name, (x, truth, _) in data.items()}
    rows = []
    for fold in range(P["folds"]):
        held = names[fold::P["folds"]]
        train = [value for name, value in pieces.items() if name not in held]
        a, b = (sum(piece[i] for piece in train) for i in range(2))
        model = np.linalg.solve(a + .05*np.eye(len(a)), b)
        for reference in (1, 2):
            cohort = []
            for name in held:
                features, truth, panels = data[name]
                if reference not in truth:
                    continue
                pred = np.clip(features @ model, 0, 1)
                for seed, sides, _, b, _ in panels:
                    sizes = tuple(map(len, sides))
                    hits = tuple(int(truth[reference][s].sum()) for s in sides)
                    mass = sum(sizes)
                    priority = (pred[sides[0]].sum()-pred[sides[1]].sum())/max(1, mass)
                    caps = [min(round(f*b), mass) if sizes[0] else 0 for f in (
                        P["initial_audit"], P["baseline_audit"], P["maximum_audit"])]
                    cohort.append(dict(dataset=name, reference=reference, fold=fold, seed=seed,
                                       B=b, priority=priority, sizes=sizes, hits=hits,
                                       audit_cost=caps[0], baseline_cost=caps[1], max_cost=caps[2]))
            for method in ("positive_only", "descending", "smooth"):
                for q, cost in zip(cohort, allocate(cohort, method)):
                    baseline, original_size = certificate(q["sizes"], q["hits"], q["baseline_cost"])
                    detection, new_size = certificate(q["sizes"], q["hits"], cost)
                    rows.append(dict(**{k: v for k, v in q.items() if k not in ("sizes", "hits", "audit_cost")},
                                     method=method, audit_cost=cost,
                                     material=int(q["hits"][0] > q["hits"][1]), gap=q["hits"][0]-q["hits"][1],
                                     baseline=baseline, prioritized=detection,
                                     baseline_worst_case=original_size, prioritized_worst_case=new_size))
    # A single-panel gate can train on every other corpus. This checks
    # sensitivity to withholding an entire block of targets from training.
    for name, (features, truth, panels) in data.items():
        a, b = (sum(p[i] for other, p in pieces.items() if other != name) for i in range(2))
        pred = np.clip(features @ np.linalg.solve(a+.05*np.eye(len(a)), b), 0, 1)
        for reference, errors in truth.items():
            for seed, sides, _, budget, _ in panels:
                sizes = tuple(map(len, sides))
                hits = tuple(int(errors[s].sum()) for s in sides)
                mass = sum(sizes)
                priority = (pred[sides[0]].sum()-pred[sides[1]].sum())/max(1, mass)
                old = min(round(P["baseline_audit"]*budget), mass) if sizes[0] else 0
                new = old if priority > 0 else 0
                baseline, size = certificate(sizes, hits, old)
                detection = baseline if new else 0.
                rows.append(dict(dataset=name, reference=reference, fold=-1, seed=seed,
                                 B=budget, priority=priority, baseline_cost=old, max_cost=old,
                                 method="positive_only_loco", audit_cost=new, material=int(hits[0]>hits[1]),
                                 gap=hits[0]-hits[1], baseline=baseline, prioritized=detection,
                                 baseline_worst_case=size, prioritized_worst_case=size if new else 0.))
    write(out, rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("reference2", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    run(args.inputs, args.reference2, args.out)
