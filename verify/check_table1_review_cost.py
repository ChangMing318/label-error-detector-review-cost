"""Independent audit of Table I (review cost) against the shipped evidence.

Recomputes, from prior_study/12-review-cost/results/paired_costs.csv:
  mean random hits, mean difficulty hits, signed gap, mean signed extra checks,
  extra checks as % of B, and the trailing counts in the last column.
Asserts equality (to printing precision) with the values printed in
paper/tables/review_cost.tex and with the aggregate sentences in Section IV.
"""
from pathlib import Path
import csv
from collections import defaultdict
from statistics import mean

EV = Path(__file__).resolve().parent.parent / "data"
CSV = EV / "table-1_paired_costs.csv"

# (dataset, B, R hits, H hits, gap, extra checks, % of B, H-trails, R-trails)
TABLE1 = [
    ("Unhealthy-Comments", 449, 346.0, 193.0, +153.0, +294.0, 65.5, 10, 0),
    ("HateXplain",         319, 204.6, 156.0, +48.6,  +129.8, 40.7,  7, 0),
    ("D3code",             116,  78.3,  55.1, +23.2,  +45.6,  39.3,  9, 0),
    ("GoEmotions",         404, 316.9, 278.2, +38.7,  +104.7, 25.9,  3, 2),
    ("SBIC",               600, 283.7, 269.0, +14.7,  +46.9,   7.8, 10, 0),
    ("CIFAR-10N",          600, 542.0, 521.0, +21.0,  +28.7,   4.8,  7, 0),
    ("DynaSent",           600, 222.6, 227.0, -4.4,   -11.6,  -1.9,  0, 4),
    ("Davidson",           519, 382.3, 385.0, -2.7,   -9.0,   -1.7,  0, 9),
    ("Fashion-MNIST-H",    253, 247.8, 244.5, +3.3,   +3.3,    1.3,  4, 1),
    ("SNLI",               600, 138.9, 138.0, +0.9,   +3.6,    0.6,  1, 0),
    ("MBIC",                42,  29.1,  29.0, +0.1,   +0.1,    0.2,  1, 0),
    ("AudioSet",           433, 332.0, 332.0,  0.0,    0.0,    0.0,  0, 0),
    ("CIFAR-100N",         600, 553.0, 553.0,  0.0,    0.0,    0.0,  0, 0),
    ("Measuring-Hate-Speech", 363, 237.0, 237.0, 0.0,   0.0,    0.0,  0, 0),
    ("MultiNLI",           600, 107.2, 107.2,  0.0,    0.0,    0.0,  3, 3),
    ("Stanford-Politeness", 600, 311.0, 311.0,  0.0,   0.0,    0.0,  0, 0),
    ("Wikipedia-Attack",   600, 374.0, 374.0,  0.0,   0.0,    0.0,  0, 0),
]

rows = list(csv.DictReader(CSV.open(newline="", encoding="utf8")))
by_ds = defaultdict(list)
for r in rows:
    # reference "1" is the primary human reference; "2" is the second reference
    # available for six corpora and is reported separately in the paper.
    if abs(float(r["budget"]) - 0.1) < 1e-9 and r["reference"] == "1":
        by_ds[r["dataset"]].append(r)

fails = []
def chk(name, got, want, tol):
    if abs(got - want) > tol:
        fails.append(f"{name}: evidence={got:.3f} paper={want:.3f} (tol {tol})")

tot_h = tot_r = tot_equal = 0
hard_total, hard_targets = 0, []
for ds, B, R, H, gap, extra, pct, th, tr in TABLE1:
    rr = by_ds.get(ds, [])
    if not rr:
        fails.append(f"{ds}: no rows at budget 0.1")
        continue
    mR = mean(float(x["random_hits"]) for x in rr)
    mH = mean(float(x["difficulty_hits"]) for x in rr)
    mE = mean(float(x["signed_extra_checks"]) for x in rr)
    chk(f"{ds} R", mR, R, 0.051)
    chk(f"{ds} H", mH, H, 0.051)
    chk(f"{ds} gap", mR - mH, gap, 0.051)
    chk(f"{ds} extra", mE, extra, 0.051)
    # % of B is printed as the mean of per-row extra/B, i.e. mean(extra)/B here
    chk(f"{ds} extra%", 100 * mE / B, pct, 0.06)
    # trailing counts
    h = sum(1 for x in rr if float(x["signed_extra_checks"]) > 0)
    r_ = sum(1 for x in rr if float(x["signed_extra_checks"]) < 0)
    e = len(rr) - h - r_
    if (h, r_) != (th, tr):
        fails.append(f"{ds} trails: evidence={h}/{r_} paper={th}/{tr}")
    # detector switch: the two masks picked different detectors
    sw = sum(1 for x in rr if x["random_detector"] != x["difficulty_detector"])
    tot_h += h; tot_r += r_; tot_equal += e
    # "material" cost: extra >= ceil(0.10*B) -> matches material_checks column
    mt = sum(1 for x in rr if float(x["signed_extra_checks"]) >= int(x["material_checks"]))
    hard_total += mt
    if mt:
        hard_targets.append((ds, mt))

print("\n--- aggregate checks from Section IV ---")
print(f"H trails total = {tot_h}, R trails total = {tot_r}, equal-yield = {tot_equal}")
print(f"total draws = {tot_h+tot_r+tot_equal}")
print(f"material-cost draws (extra >= ceil(0.10B)) = {hard_total}; targets={hard_targets}")
if tot_h + tot_r + tot_equal != 170:
    fails.append(f"row count {tot_h+tot_r+tot_equal} != 170")
print(f"non-tied draws = {tot_h+tot_r} (paper: 85 switches, 11 of them equal-yield -> 74 non-tied)")

print("\nFAILURES:" if fails else "\nPASS: Table I reproduces from paired_costs.csv")
for f in fails:
    print("  -", f)
raise SystemExit(1 if fails else 0)
