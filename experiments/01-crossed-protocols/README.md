# Label-Error Detection Under Shifting Noise-Generation Protocols

This artifact contains the English manuscript, paired per-seed data, generator assignments, scores summarized at fixed budgets, corrected calibration, and a matched downstream intervention. The paper and artifact distinguish archived experiments from newly executed runs.

## Rebuild figures, tables, and PDF

Use Python 3.12 or compatible Python 3.10 with the packages in requirements.txt. Install PyTorch separately using the wheel for your platform. This run used NumPy 2.2.6, SciPy 1.15.1, scikit-learn 1.6.1, pandas 2.2.3, and Matplotlib 3.10.0 for the CPU evaluation and PyTorch 2.12.1+cu130 on a GeForce RTX 4070 Laptop GPU for training and checkpoint inference. CPU-only PyTorch 2.9.1 was used to load the public labels. Set OPENBLAS_NUM_THREADS and OMP_NUM_THREADS to 4 for the recorded CPU execution.

    python check_results.py
    python summarize.py --paper paper
    python paper_findings.py
    python summarize_extension.py
    cd paper
    pdflatex -interaction=nonstopmode -halt-on-error main.tex
    pdflatex -interaction=nonstopmode -halt-on-error main.tex

The manuscript uses IEEEtran 1.8b. The class file is included for consistent compilation. If the submission author details are available, edit the author block in paper/main.tex before compiling the single-blind submission. A standard openly accessible supplement URL is an author-side submission requirement.

## Reproduce the crossed study without retraining

1. Prepare data under base/. Use the official public data downloader, or copy the files from the original input bundle to the same relative paths.

       python prepare_data.py --cifar all --verify

   The CIFAR input inventory and SHA-256 fingerprints are in input_hashes.json. Human annotation files are pinned to UCSC-REAL revision 49df7d8a69e355470c77c1c2f2424916325a394b, and clean label ordering is checked against the original CIFAR labels. The CIFAR source is the [official Toronto dataset distribution](https://www.cs.toronto.edu/~kriz/cifar.html).

2. For the CASIA arm, use the original input bundle's data/processed/hwdb11_test_32.npz or obtain Gnt1.1Test.zip from the [official CASIA distribution](https://nlpr.ia.ac.cn/databases/handwriting/Download.html) and run:

       python prepare_data.py --gnt /path/to/casia-archives

   Use the offline character-sample archive, not the 512-dimensional feature archive. The training archives Gnt1.1TrainPart1.zip and Gnt1.1TrainPart2.zip are needed only to refit the auxiliary CASIA CNN. Pass --rebuild-train to prepare them. The local experiment used the supplied processed caches; it did not redownload the CASIA archives. The archive identifiers and hashes identify the local supplied inputs. A rebuilt compressed NPZ can differ in bytes because of packaging or compression. Its classes, writer order, and sample content determine evaluation reproducibility. CASIA pixels and the original archive are not redistributed in this artifact.

3. Generate embeddings using the distributed checkpoints. The small Chen inputs preserve the top wrong class and its epoch-averaged probability on the same fixed pool in original order. They avoid storing whole training trajectories or dense features.

       python prepare_features.py --base base --frozen hwdb
       python prepare_features.py --base base --frozen c10
       python prepare_features.py --base base --frozen c100
       python cross_protocol.py --base base hwdb
       python cross_protocol.py --base base c10
       python cross_protocol.py --base base c100
       python transfer_bounds.py --base base
       python extend_detectors.py --base base

   Using the shipped weights on the execution GPU, the reconstructed final features and logits match the arrays used in the formal runs. CPU execution is supported by passing --device cpu to feature preparation. A different math backend can change ties or borderline ranks. For a separate replay, pass --out to feature preparation and --cache / --out to the crossed experiment. Raw result rows supplied in results/ preserve the observed outcomes.

4. Corrected calibration and archived sign diagnostics use the bundled confidence and margin probes.

       python 108_calibration_coverage.py --base base
       python base/scripts/110_sign_outofsample.py

   Reproduce the historical CASIA five-split protocol with python base/scripts/109_protocol_rules.py. This uses the stored font destination map rather than requiring the original four Windows font files. The historical and new studies use different splits and report separately. The original downstream accuracy logs are in legacy/endtoend_more.json, and the archived implementation is base/scripts/107_endtoend_more.py. The archived study is separate from the new matched intervention.

5. Refit missing encoders by omitting --frozen, preferably with a separate --out directory. Reproduce the new downstream intervention with:

       python downstream.py --base base

   This consumes inputs/c10_features.npz, runs the 10 paired seeds across all 5 arms, and writes results/downstream.json. It trains from scratch on the 20,000-record cleaning pool. The official CIFAR-10 test data are reserved for that downstream endpoint. Use --device cpu if necessary.

## Files and interpretation

- protocol.json fixes the split seeds, budgets, training settings, and human prevalence matching.
- results/*_metrics.csv records per-run AUC, AP, selected counts, precision, recall, correction, damage, and score mechanism fields.
- results/*_regret.csv records source and AUC selections, the target comparator, and the actual missed corrections. No target labels were used to select from the source.
- results/*_ranking.csv gives paired rank inversions and Kendall correlations. results/*_assignments.npz contains disjoint partitions, deterministic score tie order, masks, and exact claims. For CIFAR, indices are relative to the post-fit permutation returned by prepare_features.py.
- results/*_policies.csv and legacy_coverage_*.csv contain audit costs, executed review, gain, conditional precision, and unconditional attainment. Blank precision means no follow-up.
- results/transfer_bounds.csv records the finite-pool AUC and prefix certificate experiments. results/paired_summary.json contains paired bootstrap summaries, conditional on the observed datasets and models.
- results/downstream.json has every paired run and the fixed, damaged, and unfixed subgroup geometry.
- PROTOCOL_AUDIT.md records the audited code paths and how the released protocols were instantiated.
- SHA256SUMS inventories the release. input_hashes.json identifies the external clean and human data actually used.

## Provenance of the supplied inputs

The source archives were used read-only. revision_2026-11-09 (2).zip SHA-256 is 4069631bc4582d36c8ed18ed47028dcc215272c42e7d9c47769ce22f67f23718. labelerr_icde_2026-11-09.zip SHA-256 is b4000bcf85336df18f51357cf4c2981367955b50778d8b074640c0a5a012eb5a. We reused the CASIA detector checkpoint, the curated subset, the font destination map, the CIFAR data, and the archived human probe and training logs. Earlier coverage tables with phantom reviews are superseded by the corrected replay. The original broad result archives remain the record of work not used by this manuscript.

## Expanded-candidate human transfer

The frozen extension scores trusted-reference CNN kNN disagreement and the classifier-head self-confidence ranking on the identical CIFAR claims, then uses the existing margin rows. Its setup is in extension_protocol.json. It evaluates three nested candidate sets, three source noise rates per human target, four review caps, and source-review, source-AUC, and fixed CNN margin policies. No model is refitted. A replay checks that the original candidate set exactly reproduces the archived matched-prevalence selection, selected hits, and target oracle.

- results/*_extended_metrics.csv contains the additional scorers' per-split metrics on all existing CIFAR claims and budgets.
- results/*_extended_regret.csv contains every human-target source selection and fixed baseline. The oracle is specific to the candidate set. The delta_vs_fixed_cnn_pp column uses the same fixed policy in all sets.
- results/extension_summary.json contains all paired split intervals and choice counts. summarize_extension.py generates the manuscript tables and text for this extension.

## Unified downstream replication

The 100-seed extension reruns the calibrated automatic relabeling setup on all 50,000 training records and the fixed 10% automatic relabeling setup on the existing 20,000-record cleaning pool. Both use the same fresh 100 seeds, 12 epochs, and the original three-convolution downstream classifier. Within each seed and setup, the two arms use the same initialization, image order, and augmentation. They differ only in assigned labels. The reference endpoint is the official 10,000-image CIFAR-10 test set.

       python prepare_features.py --base base --frozen c10
       python downstream100.py --base base
       python summarize_downstream100.py --raw results/downstream100_runs.json
       python paper_findings.py

The historical two-fold scoring cache can be reproduced by removing legacy_oof_100.npz from a fresh copy of the artifact before executing. It is provided for reuse. The code reuses completed pairs when resuming the same protocol. For an independent replay, use a fresh --out directory, optionally copy legacy_oof_100.npz there, and point the statistics command to the new raw result. The formal CUDA runs use BF16 training forward with FP32 loss, weights and optimizer state, fused Adam, deterministic cuDNN, and FP32 evaluation. The original CPU logs are kept separate. The two legacy detector folds are trained once in FP32 and fixed across all 100 audit draws. The analysis reports paired accuracy differences in percentage points using the same 20,000 bootstrap draws, and counts both corrected and damaged labels directly.

The downstream summary separates the cost of trusted fitting labels, trusted reference labels, and deployment audit checks. The calibrated setup uses 5,000 checks per deployment. The fixed setup assumes 25,000 trusted fitting labels and 5,000 trusted reference labels, with zero additional checks. These counts measure required access to supervision. They do not equate the labor cost of a correctness check with that of assigning a full class label.
