"""Summarize exact audit power, check costs and held-out screens."""
import argparse
import json
from pathlib import Path
from statistics import mean

from summarize_review_cost import group, table


def summarize(root):
    audit = [r for r in table(root, "precision_audit.csv") if r["fraction"] == .10]
    signs = [r for r in table(root, "paired_sign_audit.csv") if r["fraction"] == .10]
    priority = list(table(root, "pooled_audit.csv"))
    paid = [r for r in table(root, "paid_policy_exact.csv") if r["audit_fraction_limit"] == .10]
    screened = {(r["dataset"], r["seed"], r["reference"]): r for r in priority
                if r["method"] == "positive_only"}
    for r in paid:
        decision = screened[r["dataset"], r["seed"], r["reference"]]
        r["screened_yield"] = r["expected_yield"] if decision["audit_cost"] else r["yield_difficulty"]
    summaries = {"exact_statistics": [], "paired_sign": [], "rank_priority": [], "deployment": []}
    for data, field, kind in ((audit, "statistic", "exact_statistics"), (signs, "method", "paired_sign")):
        keys = ["reference", field] + (["allocation"] if data is audit else [])
        for ident, rows in group(data, *keys).items():
            true = [r for r in rows if r["material"]]
            summaries[kind].append(dict(zip(keys, ident), panels=len(true),
                                        mean_power=mean(r["probability"] for r in true),
                                        high_power=sum(r["probability"] >= .8 for r in true),
                                        worst_case_false_positive=max(
                                            r.get("worst_case_false_positive", .05) for r in rows)))
    for (reference, method), rows in group(priority, "reference", "method").items():
        true = [r for r in rows if r["material"]]
        summaries["rank_priority"].append(dict(
            reference=reference, method=method, material=len(true),
            baseline_checks=sum(r["baseline_cost"] for r in rows),
            screened_checks=sum(r["audit_cost"] for r in rows),
            material_skipped=sum(r["audit_cost"] == 0 for r in true),
            baseline_power=mean(r["baseline"] for r in true),
            new_power=mean(r["prioritized"] for r in true),
            baseline_high_power=sum(r["baseline"] >= .8 for r in true),
            new_high_power=sum(r["prioritized"] >= .8 for r in true),
            unchanged_on_every_material=all(r["baseline"] == r["prioritized"] for r in true),
            worst_case_false_positive=max(r["prioritized_worst_case"] for r in rows)))
    for (reference,), rows in group(paid, "reference").items():
        summaries["deployment"].append(dict(
            reference=reference,
            **{f"vs_{key}": mean(10000*(r["screened_yield"]-r[key])/r["n"] for r in rows)
               for key in ("expected_yield", "yield_random", "yield_difficulty", "yield_interleave")}))
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.results), indent=2))
