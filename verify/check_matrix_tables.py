"""Assert Table IV (matrix AUC) and Table V (paired gaps) against the shipped metrics CSVs.

Table IV = mean target AUC at 10% corruption, 10% budget, per protocol x detector.
"""
from pathlib import Path
import sys
import pandas as pd

EV = Path(__file__).resolve().parent.parent / "data"

TABLE4 = {
    # protocol        CASIA CNN/PCA/RP     C10 CNN/PCA/RP      C100 CNN/PCA/RP
    "NC-CNN":      (0.891, 0.672, 0.654, 0.689, 0.568, 0.574, 0.266, 0.426, 0.480),
    "NC-PCA":      (0.969, 0.254, 0.439, 0.880, 0.234, 0.422, 0.723, 0.061, 0.273),
    "NC-RP":       (0.973, 0.490, 0.225, 0.883, 0.406, 0.256, 0.748, 0.305, 0.073),
    "NC-Aux":      (0.922, 0.656, 0.637, 0.758, 0.583, 0.583, 0.414, 0.456, 0.484),
    "AQuA-ID-CNN": (0.998, 0.919, 0.889, 0.932, 0.711, 0.705, 0.868, 0.727, 0.708),
    "AQuA-CM-CNN": (0.958, 0.770, 0.748, 0.885, 0.657, 0.651, 0.787, 0.649, 0.649),
    "AQuA-ID-Aux": (0.999, 0.923, 0.900, 0.927, 0.713, 0.710, 0.886, 0.723, 0.720),
    "AQuA-CM-Aux": (0.962, 0.770, 0.746, 0.884, 0.648, 0.651, 0.781, 0.654, 0.646),
    "Chen-ID-mask":(0.935, 0.697, 0.676, 0.782, 0.573, 0.577, 0.527, 0.456, 0.490),
    "Chen-ID-hard":(0.744, 0.552, 0.557, 0.496, 0.476, 0.480, 0.368, 0.349, 0.404),
    "Uniform":     (0.999, 0.925, 0.902, 0.933, 0.711, 0.709, 0.880, 0.733, 0.726),
}
DS = ("hwdb", "c10", "c100")
fails, checked = [], 0
frames = {}
for ds in DS:
    d = pd.read_csv(EV / f"table-5_{ds}_metrics.csv")
    frames[ds] = d[(d.side == "target") & (abs(d.rate - 0.10) < 1e-9) & (d.budget == 0.1)]
for proto, vals in TABLE4.items():
    for di, ds in enumerate(DS):
        d = frames[ds]
        means = d[d.protocol == proto].groupby("detector").auc.mean()
        for k, det in enumerate(("CNN", "PCA", "RP")):
            want = vals[di * 3 + k]
            got = float(means[det])
            checked += 1
            if abs(round(got, 3) - want) > 0.0006:
                fails.append(f"Table IV {proto:12s} {ds:5s} {det}: evidence={got:.4f} paper={want:.3f}")

# Table V: paired PCA-RP gaps at all three rates, target side.
TABLE5 = {
    ("hwdb", 0.10): (-0.185, +0.265), ("hwdb", 0.20): (-0.187, +0.271), ("hwdb", 0.40): (-0.188, +0.268),
    ("c10", 0.10): (-0.188, +0.150), ("c10", 0.20): (-0.187, +0.149), ("c10", 0.40): (-0.187, +0.147),
    ("c100", 0.10): (-0.212, +0.232), ("c100", 0.20): (-0.213, +0.235), ("c100", 0.40): (-0.215, +0.236),
}
for ds in DS:
    d = pd.read_csv(EV / f"table-5_{ds}_metrics.csv")
    tgt = d[(d.side == "target") & (d.budget == 0.1)]
    for rate in (0.10, 0.20, 0.40):
        sub = tgt[abs(tgt.rate - rate) < 1e-9]
        gap_nc_pca, gap_nc_rp = [], []
        for seed, g in sub.groupby("seed"):
            piv = g[g.protocol.str.startswith("NC-")].pivot_table(index="protocol", columns="detector", values="auc")
            # NC-PCA row: PCA detector minus RP detector
            gap_nc_pca.append(piv.loc["NC-PCA", "PCA"] - piv.loc["NC-PCA", "RP"])
            gap_nc_rp.append(piv.loc["NC-RP", "PCA"] - piv.loc["NC-RP", "RP"])
        got = (sum(gap_nc_pca) / len(gap_nc_pca), sum(gap_nc_rp) / len(gap_nc_rp))
        want = TABLE5[(ds, rate)]
        for k, name in enumerate(("NC-PCA", "NC-RP")):
            checked += 1
            if abs(round(got[k], 3) - want[k]) > 0.0015:
                fails.append(f"Table V {ds:5s} {rate:.2f} {name}: evidence={got[k]:+.4f} paper={want[k]:+.3f}")

print(f"checked {checked} cells (Table IV = 99, Table V = 18)")
print("FAILURES:" if fails else "PASS: Tables IV and V reproduce from *_metrics.csv")
for f in fails:
    print("  -", f)
sys.exit(1 if fails else 0)
