# Pipeline

The paper's numbers come from four studies. Each is a standalone set of scripts that reads
public corpora, writes per-run rows, and is then summarised. `data/` holds the summaries
and the record-level arrays; the drivers are under `experiments/`.

## Order

```
corpora -> features -> 01-crossed-protocols -> tables IV, V, VII, Fig. 2, Fig. 3
                    -> 02-cross-corpus       -> tables IV, VI
                    -> 03-review-cost        -> tables I, II, VIII, IX, X, Fig. 4
```

## 01-crossed-protocols  (`experiments/01-crossed-protocols/`)

| file | role |
|---|---|
| `cross_protocol.py` | builds the eleven generators x nine detectors matrix on CASIA-HWDB, CIFAR-10 and CIFAR-100; writes `*_metrics.csv` |
| `prepare_features.py` | frozen encoder features and cached scores |
| `summarize.py` | aggregates the per-run rows into the tables |
| `protocol.json` | the preset: rates, budgets, menus, seeds |

Needs: the three corpora, the frozen checkpoints, and cached features. Produces the AUC
matrix (Table V), the legacy rule table (Table IV), the human holdouts (Table VII), the
neighbour bins (Fig. 2) and the regret matrix (Fig. 3).

## 02-cross-corpus  (`experiments/02-cross-corpus/`)

| file | role |
|---|---|
| `corpus_sweep.py` | detector selection under two source masks on 19 human-annotated corpora, at a fixed review budget |
| `adapters/` | per-corpus loaders and vote handling |
| `corpus_sweep_protocol.json` | the preset |

Needs: the corpora listed in the protocol file. Produces the reference-pairing table
(Table IV) and the candidate-set study (Table VI).

## 03-review-cost  (`experiments/03-review-cost/`)

| file | role |
|---|---|
| `review_cost.py` | the review-cost measurement: yield, catch-up checks, interleave, deployment policies, `row_metrics()` |
| `assisted_audit.py` | the paid yield audit at six caps (Table II) |
| `futility_deployment.py` | the equal-cost downstream comparison (Tables VIII, IX) |
| `paired_sign_audit.py` | the sign audit behind Section III |
| `*_protocol.json` | presets, including the 10% budget and 20% corruption rate |

Needs: the per-corpus rankings written by 02-cross-corpus. Produces Tables I, II, VIII, IX,
X and Fig. 4.

## Reproducing without the corpora

Tiers 1–3 in the README do not touch any driver. They read `data/` only, which is the
point: the claims can be checked by a reviewer in under a minute, and the drivers are
available for anyone who has the corpora.

## What is not shipped from `experiments/`

The upstream `base/` toolkit (feature extraction and the noisy-label training loop for CASIA-HWDB) is not included: it is third-party research code with its own licence and distribution terms, and the frozen features it produced are already what `data/` records. `cross_protocol.py` reads those features and can be re-run wherever the toolkit is installed.

## Checks that ship here vs elsewhere

`verify/` holds the checks that read `data/` or the shipped modules, so they run from a clean copy of this artifact. The checks that audit the manuscript source (citation closure, page structure, number preservation) live with the paper sources, not here, because they need `main.tex` and the built PDF.
