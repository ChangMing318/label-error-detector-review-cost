# Supplemental methods and validation

The conference manuscript has no appendix. This note documents inputs and additional result indexing for independent reproduction.

## Reuse and scope

- CASIA input uses the archived curated 100-character study, 40 held-out samples per class, and the archived CNN detector. Historical five-split results are listed separately from the new ten splits. The original font map is stored in inputs/font_map.npz. In the newly crossed study, the Aux encoder is fit only once.
- The human-score caches in the source archives were suitable for calibration and the historical sign study. They could not rescore new wrong classes under a changed generator or supply common detector scores on new partitions. We fit the two CIFAR encoder instances per dataset and used the existing CNN architecture. No other candidate was fitted on target labels.
- Historical retraining accuracy was recovered from archived per-seed logs, not rerun. The new intervention trained five conditions with each of ten seeds. Geometry summaries quantify a subgroup property, while contrasts among the label interventions address the effects of changing those labels.
- Source and target records do not overlap within a split. Dataset and model reuse across different splits means the intervals estimate variability of the split procedure on these fixed resources. The all-seed results and bootstrap code make the claim verifiable without replacing any seed.

## Result indexing

- Directory `12-review-cost` provides the fixed-cap hit gap and the extra-check count at each source draw and cap. The signed cost is computed on the original target ranking. The second reference and exhaustive tied-winner comparisons keep the same source and target queues. The audit replay credits every audited hit once and caps total checks at the same value as direct review.
- The archived automatic relabeling experiments compare each policy with its own unchanged labels. The calibrated setup changes accuracy by -1.79 [-2.85, -0.73] percentage points and uses 5,000 audit checks. Fixed 10% relabeling changes accuracy by +2.97 [1.95, 4.04] points and uses 30,000 trusted fitting and reference labels. The paper reports paired review-queue retraining separately.
- The rate-based audit's full review intervals and test results are in `paper/tables/paid_transfer.tex`. Direct SNLI-to-MultiNLI review improves on no review by +0.27 [+0.02, +0.51] points on same-genre MultiNLI. Audit minus direct review changes accuracy by +0.04 [-0.07, +0.15] on the cross-genre set and +0.72 [-0.51, +1.92] on the secondary official CIFAR-10 test.

- Rank comparisons with different masks and all destination pairs are in each dataset's ranking CSV. Filter rate and use protocol1/protocol2. The Kendall coefficient uses three candidate detectors and counts ties via tau-b. Strict inversions omit ties.
- The paper reports the 10% cap in the main regret matrix. Each regret CSV also provides the other three caps, all three prevalence levels, and two source decision rules. The human target rows are indexed by released annotation key and matched source prevalence.
- The per-split mechanism CSV includes the class-weighted modal destination share and the fraction of wrong destinations matching the auxiliary nearest prototype. The main metric CSV includes mean margins, zero-score masses, and conditional nearest-class accuracy. These are distinct diagnostics of class concentration, label plausibility under a reference representation, and difficulty.
- In the historical calibration replay, audits are charged and only follow-up corrections count as gains. In the new capped comparison audited corrections are also credited, as a deployment policy can directly use the audit labels. For calibrated review, a nonempty audit with zero follow-up has positive total cost and can still repair audited errors.
- The downstream fixes-only, damage-only, and perfect-review arms use privileged reference labels. The auto arm uses only the selected records and proposals from the frozen detector. It can change an already wrong label to a different wrong label.

## Theory checks

For nonnegative scores, let a be the strictly positive mass among errors and b the zero mass among non-errors. The zero-versus-zero contribution to AUC is b(1-a)/2 and the positive-versus-zero contribution is ab. The strictly positive pair term is a(1-b)q with tie-adjusted q. The code evaluates these terms on every generated cell. When a fixed geometry generates the nearest wrong class, a coincides with true-nearest accuracy on errors except for closest-class ties. The zero mass b is the true class attaining maximum similarity among non-errors. These conditional probabilities can differ even with unchanged clean representation quality.

The transfer bound uses an observed common pool and fixed masks. For each positive record and detector, compute the negative mid-rank of the positive score in both label vectors. The mean absolute mid-rank change bounds the absolute AUC shift. For fixed review, exchange the prefixes first with the error set held fixed, then change the error set on the original prefix. The prefix symmetric difference divided by two bounds the hits that can change in the first step. These are sufficient finite-pool certificates. The release checks every certified sign and records the cases that the bound leaves undecided.


## Secondary downstream analysis

The main study measures lost corrections under a review cap. This separate experiment measures test accuracy after changing labels. It holds the cleaning pool, detector, selected records, architecture, and seed fixed between label interventions. We use 20,000 CIFAR-10N worst images, disjoint prototypes, and the frozen CNN margin's 2,000-record prefix. We train the original three-convolution architecture for 12 epochs and evaluate on the official clean test set. Each of ten training seeds runs all five label conditions.

| Label intervention | Net correct labels | Test accuracy difference in points [95% paired interval] |
| --- | ---: | ---: |
| Auto | +1,408 | +4.14 [+0.60, +7.59] |
| Fix only | +1,462 | +2.46 [-1.85, +6.10] |
| Damage only | -54 | -0.56 [-1.82, +0.73] |
| Review | +1,946 | +3.47 [-0.68, +7.25] |

The automatic arm fixes 1,462 errors, damages 54 clean labels, and leaves 484 selected errors incorrect. It can change an already incorrect label to a different incorrect class. The damage-only comparison isolates the selected damage in this intervention and its interval spans zero. The nearest-to-second-nearest similarity gap is 0.327 on fixed errors and 0.096 on selected errors that remain incorrect. Damaged clean labels have mean similarity 0.128 to their true prototype, compared with 0.765 for all clean labels.

The archived run used a two-fold detector and a calibrated automatic prefix of about 28% on a different training pool. Its paired accuracy estimate was -2.48 [-5.16, +0.37] points. The archive's prior analysis reported a positive label gain at that operating point. Its per-seed retraining log records accuracy and budget, with no matching per-seed net-correction counts. The archived interval spans zero, so these data leave a reproducible accuracy decrease unresolved. The new matched intervention supports a positive automatic-versus-unchanged accuracy effect in its fixed setting. Across the archived and new settings, changes in detector, pool, and selection rule preclude attributing the different estimates to the action budget. Neither experiment quantifies the effect of source-protocol detector selection on downstream test accuracy.

## Expanded detector set

The nested sets are (1) CNN, PCA, and RP prototype margins, (2) those margins plus CNN kNN disagreement, and (3) all preceding scores plus classifier-head confidence. CNN kNN counts the disagreement among 10 Euclidean nearest neighbors of the normalized embedding in the same trusted reference pool. It shares the frozen CNN backbone, but uses local records rather than class prototypes. The head scores the negative log probability of the claimed class using the frozen CosFace class weights, a scale of 30, and no true-class adjustment. Its ordering agrees with Cleanlab's self-confidence ranking. Neither candidate is a full reimplementation of SimiFeat or Confident Learning. The new candidates use no target reference labels during scoring. All non-reference human claims remain unmodified.

The original margin-only evaluation remains as a controlled comparison. The candidate extension is a follow-up selected before inspecting its outcome. It reuses datasets and splits from the earlier analysis, so estimates remain conditional on the two fitted checkpoints per dataset. It does not represent an independent replication on a new domain.

At CIFAR-100N with 40% NC-Aux source noise and 10% review, RP gives 6.01 fixes per 100 records in the original menu. Adding kNN selects it and yields 5.61. Adding the classifier head selects it and yields 8.29, versus 7.59 for the fixed CNN prototype margin. In the full set, NC-Aux has zero loss in 11 of 12 rate-by-budget cells. In the remaining cell a single split selects kNN. The head is the oracle at 10% review in all ten CIFAR-100N splits. The CNN margin is target-optimal in all 50 CIFAR-10N split-variant comparisons at 10% review.

In the full set, the adapted Chen hard mask at 20% source noise selects CNN kNN in all ten splits. On the three individual CIFAR-10N label variants it loses 1.13 to 1.31 fixes per 100 records at 10% review. Its matched random-mask counterpart chooses the target winner throughout. The AQuA confusion source from the auxiliary CNN at 10% source noise chooses the CNN head and the CNN prototype margin five times each. At 10% review on aggregate labels it loses 0.45 fixes per 100 records. All other directions and caps, including zero-regret cases, are in the raw CSVs. These outcomes describe the effects of available scorers and source construction on this benchmark.

## Unified 100-seed downstream replication

We compare automatic relabeling with unchanged labels in each of two operating points. Both use the same 100 fresh seed identifiers (310000 through 310099), the same original three-convolution classifier, 12 epochs, batch size 128, and the official 10,000-record CIFAR-10 test set. For each seed and operating point, initial weights, minibatch order, and image augmentations are shared by the two arms. We use BF16 training forward with FP32 weights, loss and evaluation on CUDA. Intervals are percentile intervals from the same 20,000 paired-seed resamples in each operating point. They describe seed variation on fixed data.

| Setup | Training records | Mean action fraction | Mean net fixes | Unchanged accuracy | Auto accuracy | Paired change in points [95% interval] | Paired SD | Positive / ties / negative |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Legacy calibrated | 50,000 | 29.18% | 1,328.9 | 57.72% | 55.92% | -1.79 [-2.85, -0.73] | 5.41 | 33 / 0 / 67 |
| Fixed 10% | 20,000 | 10.00% | 1,408.0 | 50.56% | 53.54% | +2.97 [+1.95, +4.04] | 5.31 | 79 / 0 / 21 |

The original two-fold detector's full scores were unavailable. We regenerate them once with the original eight-epoch FP32 training recipe, disjoint folds, and human labels, then share the resulting scores across the 100 audit draws. The calibrated procedure pays for a random 10% audit, selects the action prefix by the original 0.70 precision rule, and retains the historical allowance for overlap between the audit and the action prefix. The fixed 10% setup uses the prior frozen CNN and the original 20,000-record pool and action prefix. The paired training-seed contrast between setups does not identify an effect of action fraction because their pools and detectors differ. The original ten-seed results remain separately documented above.

### Trusted supervision costs

We count records for which the system needs trusted information. Full class labels and correctness checks are listed separately because their labor costs can differ.

| Setting | Clean fitting labels at setup | Clean reference labels at setup | Correctness checks per deployment | Mean automatic actions |
| --- | ---: | ---: | ---: | ---: |
| Legacy calibrated | 0 | 0 | 5,000 | 14,590 |
| Fixed 10% | 25,000 | 5,000 | 0 additional | 2,000 |

The legacy detector uses the observed human labels in its fitting folds. Its audit implementation reads a reference label for 5,000 records, although the policy itself requires only a correct/incorrect verdict to choose the cutoff. Mean overlap between the audit and automatic actions is 1,455.42 records. An overlapping record is counted once as an audit, and the proposed class still comes from the detector. The fixed policy assumes access to 30,000 trusted labels, acquired or reused before deployment. Its 20,000 target claims are scored without their reference labels. Reference labels used only to calculate benchmark outcomes are not inputs to the operating policy. The 100 seeds repeat the experiment on the same dataset and do not represent 100 separately acquired training corpora.

For the main CIFAR policy comparison, the trusted setup is common to all compared policies. Its review and audit caps describe incremental deployment cost under that setup. We report access counts, with no measured labor time or monetary price. The two downstream setups have different supervision requirements and do not constitute an equal-total-cost policy comparison.
