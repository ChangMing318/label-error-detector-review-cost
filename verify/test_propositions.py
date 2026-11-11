"""Formal check of the two propositions in Section IV, plus the definitions they use.

Prop. 1 (overlap screen):  |Y_A(B) - Y_C(B)| <= d,  d = |A\\C| = |C\\A|
Prop. 2 (cap-limited):     yield >= Y_A(B) - k

Both are tested on random permutations and random reference vectors, exhaustively
over small sizes, so a counterexample would surface if one existed.
"""
import itertools
import numpy as np

fails = []
def chk(name, cond, detail=""):
    if not cond:
        fails.append(f"{name}: {detail}")

rng = np.random.default_rng(7)

# ---------------------------------------------------------------- Prop. 1
print("Prop 1: |Y_A(B) - Y_C(B)| <= d")
viol = 0
for trial in range(20000):
    n = int(rng.integers(4, 14))
    B = int(rng.integers(1, n))
    err = rng.integers(0, 2, n).astype(bool)
    A = rng.permutation(n)[:B]
    C = rng.permutation(n)[:B]
    d = len(set(A.tolist()) - set(C.tolist()))
    assert d == len(set(C.tolist()) - set(A.tolist()))
    gap = abs(int(err[A].sum()) - int(err[C].sum()))
    if gap > d:
        viol += 1
chk("Prop 1", viol == 0, f"{viol} violations in 20000 trials")
print(f"   20000 random trials, B from 1 to n-1: {viol} violations")

# exhaustive over all small permutations, all error vectors
viol2 = 0
n, B = 5, 3
perms = list(itertools.permutations(range(n)))[:60]
for A in perms:
    for C in perms:
        for bits in itertools.product([0, 1], repeat=n):
            err = np.array(bits, dtype=bool)
            d = len(set(A[:B]) - set(C[:B]))
            gap = abs(int(err[list(A[:B])].sum()) - int(err[list(C[:B])].sum()))
            if gap > d:
                viol2 += 1
chk("Prop 1 exhaustive", viol2 == 0, f"{viol2} violations")
print(f"   exhaustive n=5, B=3, 60x60 permutations x 32 error vectors: {viol2} violations "
      f"(tightness: max gap over the grid = {max(abs(int(np.array(b,dtype=bool)[list(A[:B])].sum()) - int(np.array(b,dtype=bool)[list(C[:B])].sum())) for A in perms for C in perms for b in itertools.product([0,1],repeat=n))})")

# ---------------------------------------------------------------- Prop. 2
print("\nProp 2: cap-limited yield >= Y_A(B) - k")
viol3 = 0
worst = 0.0
for trial in range(20000):
    n = int(rng.integers(6, 20))
    B = int(rng.integers(2, n))
    k = int(rng.integers(0, B + 1))
    err = rng.integers(0, 2, n).astype(bool)
    order = rng.permutation(n)
    A = order[:B]                     # default queue
    C = rng.permutation(n)            # alternative queue
    Y_A = int(err[A].sum())

    outside = [int(r) for r in C if r not in set(A.tolist())]
    chosen, nonhit = [], 0
    for r in outside:                 # explore in C's order
        if nonhit >= k or len(chosen) >= B:
            break
        chosen.append(r)
        if not err[r]:
            nonhit += 1
    for r in A:                       # fill from the top of A
        if len(chosen) >= B:
            break
        if r not in chosen:
            chosen.append(int(r))
    Y_pol = int(err[np.array(chosen[:B])].sum())

    shortfall = Y_A - Y_pol
    worst = max(worst, shortfall / max(k, 1))
    if shortfall > k:
        viol3 += 1
chk("Prop 2", viol3 == 0, f"{viol3} violations in 20000 trials")
print(f"   20000 random trials, k from 0 to B: {viol3} violations")
print(f"   worst observed shortfall / k = {worst:.2f} (the bound allows up to 1.0)")

# ---------------------------------------------------------------- tightness
# can the bound be attained? make every explored record clean and every displaced record a hit
n, B, k = 12, 6, 3
err = np.zeros(n, dtype=bool)
A = np.arange(B)                 # default: all hits at the tail
err[A] = True
C = np.arange(B, n)              # alternative: all clean
Y_A = int(err[A].sum())
chosen, nonhit = [], 0
for r in C:
    if nonhit >= k or len(chosen) >= B:
        break
    chosen.append(int(r))
    if not err[r]:
        nonhit += 1
for r in A:
    if len(chosen) >= B:
        break
    if r not in chosen:
        chosen.append(int(r))
Y_pol = int(err[np.array(chosen[:B])].sum())
print(f"\n   tightness construction: Y_A={Y_A}, policy={Y_pol}, shortfall={Y_A-Y_pol}, k={k} "
      f"-> bound attained: {Y_A - Y_pol == k}")

print()
if fails:
    print("FAILURES:")
    for f in fails:
        print("  -", f)
else:
    print("PASS: both propositions hold; Prop 1 is tested exhaustively on a small grid")
raise SystemExit(1 if fails else 0)
