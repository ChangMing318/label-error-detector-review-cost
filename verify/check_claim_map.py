"""Check that every claim in the paper has a data file and a code path in this
artifact. Prints one line per claim; exits non-zero if any row is missing.

    python verify/check_claim_map.py
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
ok = True

# (claim, data file, code that produced it, note)
ROWS = [
    ("Table I, Section III aggregates", "data/table-1_paired_costs.csv",
     "experiments/03-review-cost/review_cost.py", "row_metrics()"),
    ("Table II, paid audit", "data/table-2_paid_policy_exact.csv",
     "experiments/03-review-cost/assisted_audit.py", "six cap rows"),
    ("Table IV, reference pairing", "data/table-4_reference_pairing.csv",
     "experiments/02-cross-corpus/corpus_sweep.py", "external annotations"),
    ("Table V, target AUC matrix", "data/table-5_c10_metrics.csv",
     "experiments/01-crossed-protocols/cross_protocol.py", "11 x 9 cells"),
    ("Table V, HWDB block", "data/table-5_hwdb_metrics.csv",
     "experiments/01-crossed-protocols/cross_protocol.py", "archived checkpoint"),
    ("Table V, CIFAR-100 block", "data/table-5_c100_metrics.csv",
     "experiments/01-crossed-protocols/cross_protocol.py", "same"),
    ("Table VI, candidate set", "data/table-6_extension_summary.json",
     "experiments/02-cross-corpus/corpus_sweep.py", "expanded menu"),
    ("Table VII, human holdouts", "data/table-7_human_holdout.json",
     "experiments/01-crossed-protocols/cross_protocol.py", "vote draws"),
    ("Tables VIII-IX, equal cost", "data/table-8_c100_linear.json",
     "experiments/03-review-cost/futility_deployment.py", "and three siblings"),
    ("Table X, four policies", "data/table-10_equal_risk_summary.csv",
     "experiments/03-review-cost/review_cost.py", "deployment()"),
    ("Fig. 2, neighbour dose-response", "data/fig-2_neighbour_count.json",
     "experiments/01-crossed-protocols/cross_protocol.py", "binned by corrupted neighbours"),
    ("Fig. 3, regret matrix", "data/fig-3_c100_regret.csv",
     "experiments/01-crossed-protocols/cross_protocol.py", "source-to-target loss"),
    ("Fig. 4, cap sweep", "data/table-10_paired_rows.csv",
     "reproduce.py", "recomputed from the policy rows"),
    ("Propositions 1 and 2", "(none: analytic)", "code/review_cost.py",
     "bounds() and signed_debt()"),
]

print(f"{'claim':<34}{'data':<6}{'code':<6}  note")
for claim, data, code, note in ROWS:
    has_d = data.startswith("(") or (ROOT / data).exists()
    has_c = (ROOT / code).exists()
    ok = ok and has_d and has_c
    print(f"{claim:<34}{'ok' if has_d else 'MISS':<6}{'ok' if has_c else 'MISS':<6}  {note}")

print()
if not ok:
    print("FAIL: a claim has no data or no code path in this artifact")
    sys.exit(1)
print("PASS: every claim maps to a shipped data file and a shipped code path")
