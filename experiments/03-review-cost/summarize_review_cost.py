"""Summarize the frozen audit cost study without pooling records across tasks."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def table(root, file):
    with (root / file).open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            result = {}
            for key, value in row.items():
                try:
                    result[key] = float(value) if "." in value or "e-" in value else int(value)
                except ValueError:
                    result[key] = value
            yield result


def group(rows, *fields):
    result = defaultdict(list)
    for row in rows:
        result[tuple(row[field] for field in fields)].append(row)
    return result


def summary(root):
    measures = list(table(root, "paired_costs.csv"))
    ties = list(table(root, "source_tie_sensitivity.csv"))
    audit = list(table(root, "fixed_audit_power.csv"))
    policy = list(table(root, "paid_policy_exact.csv"))
    primary = [r for r in measures if r["reference"] == 1]
    costs_by_budget = {}
    tie_groups = group(ties, "dataset", "seed", "budget", "reference")
    for (budget,), rows in group(primary, "budget").items():
        costs_by_budget[budget] = dict(
            rows=len(rows),
            different_detector=sum(r["random_detector"] != r["difficulty_detector"] for r in rows),
            ruled_out_repairs=sum(r["ruled_out_by_overlap"] for r in rows),
            ruled_out_hard_debt=sum(r["difficulty_debt_ruled_out"] for r in rows),
            ruled_out_random_debt=sum(r["random_debt_ruled_out"] for r in rows),
            hard_repair_loss=sum(r["gap"] >= r["material_fixes"] for r in rows),
            random_repair_loss=sum(-r["gap"] >= r["material_fixes"] for r in rows),
            hard_debt=sum(r["signed_extra_checks"] >= r["material_checks"] for r in rows),
            random_debt=sum(-r["signed_extra_checks"] >= r["material_checks"] for r in rows),
            hard_debt_all_ties=sum(
                min(t["signed_extra_checks"] for t in tie_groups[
                    (r["dataset"], r["seed"], budget, 1)]) >= r["material_checks"] for r in rows),
        )
    cases = []
    for (name, reference), rows in group(
            (r for r in measures if r["budget"] == .10), "dataset", "reference").items():
        cases.append(dict(dataset=name, reference=reference, n=rows[0]["n"], B=rows[0]["checks"],
                          random_hits=mean(r["random_hits"] for r in rows),
                          hard_hits=mean(r["difficulty_hits"] for r in rows),
                          mean_signed_extra=mean(r["signed_extra_checks"] for r in rows),
                          hard_material=sum(r["signed_extra_checks"] >= r["material_checks"] for r in rows),
                          random_material=sum(-r["signed_extra_checks"] >= r["material_checks"] for r in rows),
                          hard_trails=sum(r["signed_extra_checks"] > 0 for r in rows),
                          random_trails=sum(r["signed_extra_checks"] < 0 for r in rows),
                          min_extra=min(r["signed_extra_checks"] for r in rows),
                          max_extra=max(r["signed_extra_checks"] for r in rows)))
    audits = {}
    for (test, reference), rows in group(
            (r for r in audit if r["material"]), "test", "reference").items():
        by_case = group(rows, "dataset", "seed")
        checkpoints = []
        for f in (.05, .10, .25, .50, 1., 1.5, 2., 2.1):
            q = [max([r["exact_tail_power"] for r in case if r["cost"] <= round(f*r["checks"])]
                     or [0.]) for case in by_case.values()]
            p = [max([r["power"] for r in case if r["cost"] <= round(f*r["checks"])]
                     or [0.]) for case in by_case.values()]
            checkpoints.append(dict(max_budget_fraction=f, mean_power=mean(q),
                                    panels_ge_80=sum(x >= .8 for x in q),
                                    conservative_mean_power=mean(p),
                                    conservative_panels_ge_80=sum(x >= .8 for x in p)))
        first = []
        for (name, seed), case in by_case.items():
            point = min((r for r in case if r["exact_tail_power"] >= .8), key=lambda r: r["cost"])
            first.append(dict(dataset=name, seed=seed, cost=point["cost"], B=point["checks"],
                              power=point["exact_tail_power"]))
        audits[f"{test}:{reference}"] = dict(material_panels=len(by_case), curve=checkpoints,
                                             first_measured_80pct_power=first)
    policies = []
    for (fraction, reference), rows in group(policy, "audit_fraction_limit", "reference").items():
        policies.append(dict(
            audit_fraction=fraction, reference=reference, datasets=len({r["dataset"] for r in rows}),
            **{f"vs_{arm}_per_10000": mean((r["expected_yield"] - r[f"yield_{arm}"])*10000/r["n"]
                                          for r in rows)
               for arm in ("random", "difficulty", "interleave")}))
    return dict(counts_by_budget=costs_by_budget, cases=cases, audits=audits, policies=policies)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    print(json.dumps(summary(args.results), indent=2))
