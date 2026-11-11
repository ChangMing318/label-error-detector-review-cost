"""Forecast review yield from class-conditional ranking profiles."""
from collections import defaultdict

import numpy as np


def curve(errors, order):
    errors = np.asarray(errors, dtype=bool)[order]
    positives = np.r_[0, np.cumsum(errors)]
    negatives = np.arange(len(positives)) - positives
    return positives / positives[-1], negatives / negatives[-1]


def forecast(profile, prevalence, cap):
    """Expected error mass at a fixed total review mass."""
    tp, fp = profile
    mass = prevalence * tp + (1-prevalence) * fp
    return 10000 * prevalence * np.interp(cap, mass, tp)


def matched(errors, orders, rates, caps, n, repeat, seed):
    """Rerank by filtering the same frozen detector order to identical records."""
    rng = np.random.default_rng(seed)
    errors = np.asarray(errors,dtype=bool)
    e, c = np.flatnonzero(errors), np.flatnonzero(~errors)
    out = defaultdict(list)
    eligible = np.empty(len(errors),dtype=bool)
    for _ in range(repeat):
        pe, pc = rng.permutation(e), rng.permutation(c)
        for p in rates:
            k = round(p*n)
            eligible.fill(False)
            eligible[pe[:k]] = True
            eligible[pc[:n-k]] = True
            ranks = [errors[order[eligible[order]]] for order in orders]
            for b in caps:
                out[p,b].append([10000 * r[:round(n*b)].sum()/n for r in ranks])
    return {k: np.array(v) for k,v in out.items()}


def supported_n(errors, rates, limit):
    """Common support across all requested rates."""
    n = min(limit, min(round(sum(e)/max(rates)) for e in errors),
            min(round(sum(~e)/(1-min(rates))) for e in errors))
    n = n // 100 * 100
    while any(round(p*n)>sum(e) or n-round(p*n)>sum(~e) for e in errors for p in rates):
        n -= 100
    return n


def oracle_contrast(predicted, observed):
    chosen = int(np.argmax(predicted))
    return dict(choice=chosen, regret=float(np.max(observed)-observed[chosen]),
                forecast_mae=float(np.abs(np.asarray(predicted)-observed).mean()))
