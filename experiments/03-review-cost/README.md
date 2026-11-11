# Review queue budget costs

The inputs are frozen source-selection receipts and full target rankings from the 17-corpus sweep. The primary evaluation uses 10% of the target rows. The 5% and 20% caps are also replayed. The analysis counts each review as one record-level judgment and evaluates hits against the designated published reference.

## Replay

With Python 3.12, NumPy 2.2.6 and SciPy 1.15.2:

```sh
python 12-review-cost/review_cost.py 11-cross-corpus/results replay --second 12-review-cost/reference2.npz
python 12-review-cost/summarize_review_cost.py replay > replay/summary.json
```

The first command verifies that the target hit counts match the original source-choice receipts. It also compares exact test power to direct enumeration on small populations and verifies the first-reaching prefix for each observed cost.

## Receipts

- `11-cross-corpus/results/<dataset>/paired.csv` contains the source-only candidate hit counts and selected detector identities. Each dataset has ten source draws.
- In the same directory, `target.npz` contains candidate rank permutations, target reference disagreement flags and hashed IDs and groups. `input.json` records the upstream scoring split and dimensions. The full fit and scoring code and input manifest are in directory 11.
- `reference2.npz` stores a second disagreement flag for each of six corpora in the identical target order. This reference uses another worker, or another disjoint group of three workers for Wikipedia Attack. `cost_second_reference.py` regenerates the flags using the public vote adapters in directory 11. It checks the original disagreement flag against the upstream target file for every aligned row.
- `paired_costs.csv` reports the observed differences, extra checks and label-free screens. Positive signed extra checks mean the difficulty-mask queue trails; a negative value means the random-mask queue trails.
- `source_tie_sensitivity.csv` exhausts the combinations of tied source winners on both references.
- `fixed_audit_power.csv` reports the finite-population certificate probability under both the two-bound test and the exact worst-case-null test. Each cost is a separate fixed-budget measurement design. The grid includes the full exclusive census.
- `paid_policy_exact.csv` evaluates same-budget review yield for the audited policies by enumerating the joint distribution of the two sampled counts. `paid_policy_replay.csv` contains the original 100 seeded audit replays per panel. Their sampled hit counts determine switching and credited yield.
- `summary.json` aggregates the exact measurements and lists the first evaluated costs reaching 80% detection probability. These costs are taken from the declared grid.

The source choices and queue prefixes use no target reference judgments. The full reference defines the evaluation population and the exact power of the audit. A simulated audit only supplies the sampled hit counts to the policy. The experimental caps count checks and are independent of individual annotation time.

## Wake Vision external replay

The compact inputs retain the image identifiers, original and manually revised binary labels, Open Images machine scores, and historical coverage indicators. No image pixels are included. Wake Vision annotation sources are pinned to [this revision](https://github.com/harvard-edge/Wake_Vision/tree/bab1ce4a8ef20a7ef90858279b1a2a0796125946/cleaned_csvs). The Open Images inputs use the [V7 release](https://storage.googleapis.com/openimages/web/download_v7.html), with raw checksums in `wake_inputs/vision_sources.sha256`. The compact annotations are released under CC BY 4.0. The historical indicator records where human verification took place. All hits use the Wake Vision label convention.

```sh
python 12-review-cost/wake_cost.py 12-review-cost/wake_inputs 12-review-cost/results/wake_cost
```

The replay reproduces the archived yield totals for all five policies and three caps. Each allocation remains fixed while its queue is extended. The output records both coverage definitions, both splits, every tie seed, and all combinations of tied validation winners.

## Low-cost identification follow-up

The paper uses the decision-preserving stopping rule. It stops the prespecified fixed audit as soon as the best and worst possible final outcomes give the same decision. It uses the existing ranks to define the two audit sets and the sampled judgments to determine when to stop. It does not need a calibration model.

```sh
python 12-review-cost/futility_audit.py 11-cross-corpus/results 12-review-cost/reference2.npz replay/futility
python 12-review-cost/futility_deployment.py 11-cross-corpus/results 12-review-cost/reference2.npz replay/futility
```

The first command integrates the exact sampling distribution and records expected check costs and unchanged decision probabilities by target, cap, and direction. The second replays 600 paired draws per comparison under the original total review cap. The costs in the two experiments describe different tasks: certifying a catch-up threshold and deciding when to switch queues within a deployment cap.

The historical and label-free gate comparisons are retained as experimental results:

```sh
python 12-review-cost/screen_controls.py 11-cross-corpus/results 12-review-cost/reference2.npz replay/screen_controls --deployment replay/paid_policy_exact.csv
```

From the archive root, use the same Python environment:

```sh
python 12-review-cost/precision_audit.py 11-cross-corpus/results 12-review-cost/reference2.npz replay/precision_audit.csv
python 12-review-cost/paired_sign_audit.py 11-cross-corpus/results 12-review-cost/reference2.npz replay/paired_sign_audit.csv
python 12-review-cost/assisted_audit.py 11-cross-corpus/results 12-review-cost/reference2.npz replay/assisted_audit.csv
python 12-review-cost/pooled_audit.py 11-cross-corpus/results 12-review-cost/reference2.npz replay/pooled_audit.csv
python 12-review-cost/summarize_precision.py replay > replay/precision_summary.json
```

`precision_audit.csv` compares four outcome orderings and four label allocations under a fixed exact test size. `paired_sign_audit.csv` checks matching, reverse matching, and shuffled pairs. `assisted_audit.csv` compares empirical screening scores under nested held-out-corpus calibration. These historical cutoffs are screening scores only. `pooled_audit.csv` records the catch-up gap estimated from other corpora, exact test power and false-positive bounds, and charges every target check in the fixed budget and the cost-saving comparisons. `positive_only` is the sign gate with four corpus-level folds. `positive_only_loco` trains on all other corpora and is the sensitivity check. The other two methods reallocate a pooled budget across comparisons. A diagnostic check count is a sum over replays. It is separate from the total review cap in the deployment replay.
