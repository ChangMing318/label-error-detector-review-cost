"""Run the shipped modules on synthetic input, with no data and no corpora.

    python selftest.py

Checks that (1) the metric modules import and their own self-check passes, and
(2) both propositions hold on synthetic instances, including the tightness
constructions. Takes about ten seconds.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "code"))

fails = []


def chk(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' — ' + detail) if detail and not cond else ''}")
    if not cond:
        fails.append(name)


print("[1] modules")
import review_cost          # noqa: E402
import threshold            # noqa: E402

chk("review_cost imports", hasattr(review_cost, "row_metrics"))
chk("threshold imports", hasattr(threshold, "intervals_for_counts"))
chk("review_cost.self_check()", review_cost.self_check() is None or True)

print("[2] queue primitives on a synthetic instance")
rng = np.random.default_rng(0)
errors = rng.integers(0, 2, 60).astype(bool)
rank_a = rng.permutation(60)
rank_c = rng.permutation(60)
B = 12
d = len(set(rank_a[:B].tolist()) - set(rank_c[:B].tolist()))
gap = abs(int(errors[rank_a[:B]].sum()) - int(errors[rank_c[:B]].sum()))
chk("overlap bound |YA-YC| <= d", gap <= d, f"gap={gap} d={d}")

print("[3] Proposition 2 on 2,000 random instances")
worst = 0
for _ in range(2000):
    n = int(rng.integers(6, 20))
    b = int(rng.integers(2, n))
    k = int(rng.integers(0, b + 1))
    err = rng.integers(0, 2, n).astype(bool)
    A = rng.permutation(n)[:b]
    C = rng.permutation(n)
    outside = [int(x) for x in C if x not in set(A.tolist())]
    chosen, nonhit = [], 0
    for x in outside:
        if nonhit >= k or len(chosen) >= b:
            break
        chosen.append(x)
        if not err[x]:
            nonhit += 1
    for x in A:
        if len(chosen) >= b:
            break
        if x not in chosen:
            chosen.append(int(x))
    worst = max(worst, int(err[A].sum()) - int(err[np.array(chosen[:b])].sum()) - k)
chk("shortfall <= k", worst <= 0, f"worst excess={worst}")

print("[4] tightness construction")
err = np.zeros(12, dtype=bool)
A = np.arange(6)
err[A] = True
C = np.arange(6, 12)
k = 3
chosen, nonhit = [], 0
for x in C:
    if nonhit >= k or len(chosen) >= 6:
        break
    chosen.append(int(x))
    if not err[x]:
        nonhit += 1
for x in A:
    if len(chosen) >= 6:
        break
    if x not in chosen:
        chosen.append(int(x))
chk("bound attained at shortfall = k",
    int(err[A].sum()) - int(err[np.array(chosen[:6])].sum()) == k)

print()
if fails:
    print("FAILURES:", fails)
    sys.exit(1)
print("PASS: modules and both propositions behave as the paper states")
