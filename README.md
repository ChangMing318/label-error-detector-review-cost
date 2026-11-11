# Artifact — Selecting Label-Error Detectors: Review Cost, Downstream Quality, and Bounded Loss

Everything needed to (a) recompute every number in the paper, (b) run the two metric
modules on synthetic input, and (c) re-run the correspondence check between the paper's
claims, the shipped data and the code that produced it.

```bash
python reproduce.py                 # ~30 s, recomputes Table I, V, X and both bounds
python selftest.py                  # ~10 s, modules + propositions on synthetic input
python verify/check_claim_map.py    # instant, every claim -> data file -> code path
```

`reproduce.sh` and `reproduce.ps1` are one-line wrappers around `reproduce.py` for
bash and PowerShell, so the same command works without remembering the interpreter name.

Requirements: Python 3.9+, NumPy, SciPy. Nothing else, and **no dataset access**: the
first two commands run on the frozen measurements that ship here.

---

## What runs without the corpora

| tier | command | needs | what it establishes |
|---|---|---|---|
| 1 | `python reproduce.py` | this folder only | Table I (170 draws, 85 switches, 96 equal-yield, 28 material draws over 4 targets), Table V's NC-CNN row, Table X (6/22/5, interleave tied-or-best on 21), and both propositions with zero violations |
| 2 | `python selftest.py` | this folder only | `code/review_cost.py` and `code/threshold.py` import and self-check; the overlap bound and the cap bound hold on synthetic instances; both are attained |
| 3 | `python verify/check_claim_map.py` | this folder only | every table, figure and proposition in the paper has a shipped data file and a shipped code path |
| 4 | `experiments/` | the original corpora and encoders | the drivers that produced `data/` |

Tier 4 is shipped for inspection and re-execution where the corpora are available; it is
documented in `docs/PIPELINE.md` and is not needed for tiers 1–3.

## Layout

```
reproduce.py        tier 1
selftest.py         tier 2
data/               frozen decisions and measurements (19 files), one per table or figure
code/               review_cost.py, threshold.py, protocol file
experiments/        the drivers, grouped by study, with their protocol JSONs
verify/             the checks used while preparing the paper, incl. check_claim_map.py
docs/PIPELINE.md    what each driver needs and the order to run them
docs/CLAIM_MAP.md   claim -> data -> code, with the status the checker printed
CHANGELOG.md        release history; v1.0-submission is the reviewed snapshot
CITATION.cff        how to cite; add the deposited DOI before submission
MANIFEST.sha256     SHA-256 of every file except itself
```

## What is not here

The corpora (CIFAR-10N/10H, SNLI, HateXplain, Davidson, SBIC, Unhealthy Comments,
Measuring Hate Speech, GERMS-AT, ALIA, Fashion-MNIST, CASIA-HWDB1.1, …) are not
redistributed: they keep their own licences and are obtained from their releases. The
encoder checkpoints are frozen, and the paper reports the study as retrospective.
Reference disagreements are disagreements with the reference the dataset specifies, not
adjudicated ground-truth errors.

## Versioning

| release | what it is | may change? |
|---|---|---|
| `v1.0-submission` | the snapshot submitted with the paper; the DOI in Section X points here | **no** |
| later tags | packaging fixes, extra documentation, an added check | yes, with a `CHANGELOG.md` entry |

Deposit with Zenodo and cite the **version DOI** in the paper; keep the **concept DOI** in
this README. ICDE reviews single-blind, so naming the authors here is fine.
