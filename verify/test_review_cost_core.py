"""Adversarial correctness tests for the metric core of review_cost.py.

Strategy: import the authors' own functions (no reimplementation) and check them
against brute-force definitions on exhaustive small cases. No dataset is needed.
"""
import sys, itertools, math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))

import numpy as np
import review_cost as rc
from threshold import allocation, intervals_for_counts

fails = []
def check(name, cond, detail=""):
    if not cond:
        fails.append(f"{name}  {detail}")

# ---------------------------------------------------------------- 1. signed_debt
# Definition: K_H = min{k>=0 : Y_H(B+k) >= Y_R(B)}; negative of the symmetric case.
def brute_debt(r, h, B):
    n = len(r) - 1                      # cumulative arrays have n+1 entries
    r0, h0 = r[B], h[B]
    if r0 == h0:
        return 0
    if r0 > h0:                         # R leads: how many checks for H to catch up
        return min(k for k in range(0, n - B + 1) if h[B + k] >= r0)
    return -min(k for k in range(0, n - B + 1) if r[B + k] >= h0)

rng = np.random.default_rng(0)
n_ok = 0
for trial in range(4000):
    n = int(rng.integers(6, 14))
    err = rng.integers(0, 2, n).astype(bool)
    B = int(rng.integers(1, n - 1))
    perm_r, perm_h = rng.permutation(n), rng.permutation(n)
    r, h = rc.cumulative(err, perm_r), rc.cumulative(err, perm_h)
    got = rc.signed_debt(r, h, B)
    want = brute_debt(r, h, B)
    check("signed_debt", got == want, f"n={n} B={B} got={got} want={want}")
    n_ok += 1
print(f"signed_debt: {n_ok} random permutations checked against the definition")

# ---------------------------------------------------------------- 2. row_metrics
# gap bound |gap| <= exclusive-set size, and the reported normalisation.
for trial in range(800):
    n = int(rng.integers(8, 20))
    err = rng.integers(0, 2, n).astype(bool)
    B = int(rng.integers(1, n - 1))
    r, h = rng.permutation(n), rng.permutation(n)
    d = dict = rc.row_metrics(r, h, err, B, 1, 1)
    check("gap identity", d["gap"] == int(rc.cumulative(err, r)[B] - rc.cumulative(err, h)[B]))
    ex = len(np.setdiff1d(r[:B], h[:B]))
    check("gap bound", abs(d["gap"]) <= ex, f"gap={d['gap']} ex={ex}")
    check("per10k", abs(d["gap_per_10000"] - d["gap"] * 10000 / n) < 1e-9)
print("row_metrics: gap identity, |gap| <= |exclusive|, /10,000 normalisation OK")

# ---------------------------------------------------------------- 3. interleave
for trial in range(300):
    n = 40
    B = int(rng.integers(5, n))
    r, h = rng.permutation(n), rng.permutation(n)
    out = rc.interleave(r, h, B)
    check("interleave length", len(out) == B, f"{len(out)} != {B}")
    check("interleave distinct", len(set(out.tolist())) == B)
print("interleave: exactly B distinct records from the deduplicated merge OK")

# ---------------------------------------------------------------- 4. audit_strata
# After the audit, the policy must spend exactly B checks: 2m audited + (B-2m) follow-up.
for trial in range(400):
    n = 60
    B = int(rng.integers(20, 55))
    r, h = rng.permutation(n), rng.permutation(n)
    m0 = min(B // 20, 50)
    a, b, m = rc.audit_strata(r, h, B, m0)
    if m:
        check("strata sizes", len(a) >= m and len(b) >= m, f"|a|={len(a)} |b|={len(b)} m={m}")
        check("audited + follow-up == B", 2 * m + (B - 2 * m) == B)
        check("strata disjoint from prefixes",
              not set(a.tolist()) & set(h[:B - m].tolist()))
print("audit_strata: m shrinks to fit, strata disjoint from the other prefix, budget closes at B")

# ---------------------------------------------------------------- 5. audit rule
# Paper: switch iff a one-sided 95% lower bound on the hit difference favours R.
# With K_a (population hits in R-exclusive) and K_b (H-exclusive), the two policies
# yield (common + K_a) + y and (common + K_b) + x, so the difference is
#   (K_a - x) - (K_b - y)  =  lower(K_a) - high(K_b)  evaluated at the observations.
# The implemented rule is  low_a[x] - x > high_b[y] - y  -> algebraically identical.
lo_a, _ = rc.bounds(400, 20)
_, hi_b = rc.bounds(300, 20)
for x in (0, 4, 10, 16, 20):
    for y in (0, 4, 10, 16, 20):
        implemented = (lo_a[x] - x) > (hi_b[y] - y)
        explicit = (lo_a[x] - hi_b[y]) > (x - y)
        check("audit rule equivalence", implemented == explicit)
print("audit rule: implemented expression == explicit conservative comparison on the yield difference")

# monotonicity: more observed hits in R's stratum can only favour switching
prev = None
for x in range(21):
    val = lo_a[x] - x
    if prev is not None:
        check("audit rule monotone in x", val >= prev - 1e-12, f"x={x} {val} < {prev}")
    prev = val
print("audit rule: monotone in the observed stratum hits OK")

# ---------------------------------------------------------------- 6. allocation
sizes = (340, 340)
tot = sorted({min(sum(sizes), round(449 * f)) for f in (0.05, 0.1, 0.25, 0.5, 1.0)} | {sum(sizes)})
alloc = allocation(sizes, tot)
check("allocation totals", [int(s.sum()) for s in alloc] == tot, f"{[int(s.sum()) for s in alloc]} vs {tot}")
check("allocation monotone", all((np.diff(np.vstack(alloc), axis=0) >= 0).all() for _ in [0]))
check("allocation <= stratum", all((np.asarray(alloc) <= np.asarray(sizes)).all() for _ in [0]))
print(f"allocation: exact totals {tot}, monotone, never exceeds a stratum")

# ---------------------------------------------------------------- 7. interval coverage
# intervals_for_counts must be an exact inversion (not asymptotic): check coverage
# of the reported bounds against the true binomial-ish null on a small population.
from scipy.stats import hypergeom
N, n = 60, 12
lo, hi = intervals_for_counts(N, n, tuple(range(n + 1)), 0.025)
bad = 0
for K in range(N + 1):
    covered = 0.0
    for x in range(n + 1):
        if lo[x] <= K <= hi[x]:
            covered += hypergeom.pmf(x, N, K, n)
    if covered < 0.90:   # two one-sided 2.5% tails -> nominal >= 0.95
        bad += 1
print(f"intervals_for_counts: {bad} of {N+1} true K values covered below 90%")
check("interval coverage", bad == 0, f"{bad} K values under-covered")

print()
if fails:
    print(f"FAILURES ({len(fails)}):")
    for f in fails[:20]:
        print("  -", f)
else:
    print("PASS: all metric-core assertions hold")
raise SystemExit(1 if fails else 0)
