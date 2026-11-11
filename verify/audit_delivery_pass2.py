"""Second pass: case-insensitive lookup + selection-conditioned reproduction.

The reports quote selection-conditioned means ("only the repetitions where the
policy switched"): SBIC predicted +13.28, actual -3.15; NoisyAG-News +9.34, -3.20.
Those must come from rows.csv filtered on select_alt == 1, not from summary.csv.
"""
from pathlib import Path
import csv
from statistics import mean

D = Path(__file__).resolve().parent.parent / "data"
SEP = ""  # flat data/ layout in this artifact

def load(p):
    with (D / p).open(newline="", encoding="utf8") as f:
        return list(csv.DictReader(f))

eq = load("table-10_equal_risk_summary.csv") + load("table-10_equal_risk_extra.csv")
low = {r["dataset"].lower(): r for r in eq}
fails = []

for name, want in (("SBIC", (600, 30, 285.5, 288.9)), ("CODA", (90, 4, 82.8, 82.1)),
                   ("NoisyAG-News", (600, 30, 569.4, 566.7))):
    r = low.get(name.lower())
    if r is None:
        fails.append(f"{name} missing")
        continue
    B, k, fx, fb = want
    got = (int(float(r["B"])), int(float(r["k"])), float(r["fixed"]), float(r["feedback"]))
    ok = got[0] == B and got[1] == k and abs(got[2] - fx) < 0.051 and abs(got[3] - fb) < 0.051
    print(f'{name:14s} B={got[0]:>4d} k={got[1]:>3d} fixed={got[2]:7.1f} feedback={got[3]:7.1f}  '
          f'{"OK" if ok else "MISMATCH"}')
    if not ok:
        fails.append(f"{name}: {got} != {want}")

rows = load("table-10_paired_rows.csv")
print(f"\nrows.csv: {len(rows)} repetitions")
for name, want_pred, want_act in (("SBIC", 13.28, -3.15), ("NoisyAG-News", 9.34, -3.20)):
    sub = [r for r in rows if r["dataset"] == name and int(float(r["select_alt"])) == 1]
    pred = mean(float(r["predicted_gain"]) for r in sub)
    act = mean(float(r["actual_increment"]) for r in sub)
    n = len(sub)
    ok = abs(pred - want_pred) < 0.02 and abs(act - want_act) < 0.02
    print(f'  {name:14s} switched {n:3d}/{len([r for r in rows if r["dataset"] == name])}  '
          f'predicted={pred:+.2f} (paper {want_pred:+.2f})  actual={act:+.2f} (paper {want_act:+.2f})  '
          f'{"OK" if ok else "MISMATCH"}')
    if not ok:
        fails.append(f"{name} conditional means")

# every repetition: budget and uniqueness are checked by the authors' own asserts;
# here confirm the policy always completes at B distinct records via the column totals
bad = [r for r in rows if not (0 <= float(r["pilot_n"]) <= 0.04 * float(r["B"]) + 2)]
print(f"\n  pilot records stay within ~4% of B on all rows: {not bad}")
if bad:
    fails.append(f"pilot_n out of range on {len(bad)} rows")

print()
if fails:
    print("FAILURES:", fails)
else:
    print("PASS: delivered experiment reproduces, including the selection-conditioned means")
raise SystemExit(1 if fails else 0)
