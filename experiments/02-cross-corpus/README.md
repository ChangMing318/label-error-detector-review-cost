# Cross-corpus synthetic-to-human queue transfer

Run the offline evidence replay from the archive root:

~~~bash
OPENBLAS_NUM_THREADS=4 python corpus_sweep_summary.py results summary
~~~

The frozen design is in **corpus_sweep_protocol.json**. Every eligible
corpus has **paired.csv**, **input.json**, and **target.npz**. The CSV
stores all candidates' source hit counts, the two selected detectors,
the retrospective target best, and their target hits. The NPZ stores
the fixed target orders, disagreement indicators, and hashed IDs and
group IDs. Original messages, annotator identities, and model weights
are external inputs. The replay verifies source choices, every target
prefix total, and the full set of prespecified source seeds.

## Data and claims

| Corpus | Observed label and scoring reference |
| --- | --- |
| CIFAR-10N, CIFAR-100N | First personal CIFAR-10N annotation or original CIFAR-100N noisy fine label, versus the corresponding original CIFAR class |
| Fashion-MNIST-H | Deterministic individual human vote, versus the original Fashion-MNIST class |
| AudioSet | Weak speech-family presence, versus speech-family events lasting at least 0.5 s in the separate temporal annotation |
| GoEmotions, D3code, Unhealthy Comments | A real individual binary vote and a different worker's binary vote under the published input mapping |
| Wikipedia Attack | A real individual attack vote and the majority of three different workers |
| Stanford Politeness, MBIC | Disjoint individual votes after the fixed binary mapping in the earlier input adapters |
| Measuring Hate Speech, SBIC | One real vote versus strict majority of the remaining votes in the complete release, retaining at least three distinct workers |
| DynaSent | One real Round 2 ternary vote and a majority of at least three of the four remaining votes |
| SNLI, MultiNLI | Original author's label and the majority of at least three of the four independent validators |
| HateXplain, Davidson | One human vote, deterministically selected without replacement, versus strict majority of the remaining votes. Davidson's individual vote multiset comes from the released counts |
| Chaos-alphaNLI | Original answer versus a strict majority of the independently collected judgments, when available |

The AudioSet comparison is binary speech presence. It uses the published
temporal annotations over entire eligible clips and treats overlapping
ontology descendants as speech. The labels describe disagreement with
these reference rules. In subjective tasks, disagreement can reflect
reasonable differences in judgment.

The preregistered Chaos-alphaNLI entry was not scored. The authors'
download returned an HTML page in place of the ZIP. Its entry remains
in the design and in the summary's unavailable list.

## Model and source replication

The source and target roles are disjoint by the retained original group
and by exact normalized text or exact image. Candidate scores are fitted
on a separate reference partition, which is a shared supervision cost.
Each source subset uses the same records, exact number of flips, and
per-record wrong class across the paired masks. The auxiliary ridge head
sets the wrong destination and ranks its likelihood. Its class scores
are not candidate scores. The target keeps its original claim or
deterministically held-out human vote. The reference only scores
already fixed target orders. A larger candidate menu changes the
retrospective comparator, so both absolute repair differences and regret
are recorded.

Reference and feature setup reuse the previous paper artifact's staged
CIFAR pixels/claims, original SNLI vote ZIP, MultiNLI validated members,
DynaSent v1.1 and the human-vote adapters. The required legacy adapter
modules are copied under **adapters**. **run_corpus_sweep.sh** records
the execution layout. For an independent run, point the runner at the
same named input files, set PYTHONPATH to the adapter and prior
robustness directories, and use the frozen model revisions in the
protocol. The model runner uses CUDA for Qwen and ResNet inference.
Install Cleanlab 2.7.1 for the expanded ranking. The recorded machine
used Python 3.12, NumPy 2.3.5, scikit-learn 1.9.1 and PyTorch
2.13.0+cu130. The result replay requires NumPy only.

Official release references:

- [CIFAR-N](https://github.com/UCSC-REAL/cifar-10-100n),
  [Fashion-MNIST-H](https://github.com/ishida-lab/irreducible),
  [AudioSet temporal annotation](https://research.google.com/audioset/download_strong.html).
- [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions),
  [Wikipedia Detox](https://meta.wikimedia.org/wiki/Research:Detox/Data_Release),
  [D3code](https://aclanthology.org/2024.emnlp-main.1029/),
  [Measuring Hate Speech](https://aclanthology.org/2022.nlperspectives-1.11/),
  [Unhealthy Comments](https://aclanthology.org/2020.alw-1.15/).
- [Stanford Politeness](https://arxiv.org/abs/1306.6078),
  [SBIC](https://maartensap.com/social-bias-frames/),
  [MBIC](https://github.com/Media-Bias-Group/MBIC-A-Media-Bias-Annotation-Dataset-Including-Annotator-Characteristics),
  [DynaSent](https://github.com/cgpotts/dynasent),
  [SNLI](https://nlp.stanford.edu/projects/snli/),
  [MultiNLI](https://cims.nyu.edu/~sbowman/multinli/).
- [HateXplain](https://github.com/hate-alert/HateXplain),
  [Davidson et al.](https://github.com/t-davidson/hate-speech-and-offensive-language),
  [ChaosNLI](https://github.com/easonnie/ChaosNLI).
