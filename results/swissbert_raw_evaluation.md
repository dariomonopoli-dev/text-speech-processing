# Original Approach 3: SwissBERT on annotated Swiss German

Corrected training results are available in the [10 November local comparison](local_comparison_20261110/report.md). The figures below describe the original checkpoints.

**Training caveat:** these scores describe the original checkpoints. A later audit found a gradient-accumulation normalization issue in their training environment; see [the updated audit](swissbert_approach3_audit.md). The scores reproduce, but any corrected training runs must be reported separately.

Evaluation: 9 November 2026. Existing GAHD-trained checkpoints were used without retraining. Each was selected by development macro-F1 with five epochs, learning rate 2e-5 and training adapter de_CH. Inference uses gsw, FP32, batch size 8 and a maximum of 128 tokens on an NVIDIA GeForce RTX 3080.

The three annotation files contain 900 complete binary judgements on 700 unique sentences: 100 shared and 600 single_annotator. Gold labels use majority vote on the shared sentences and the original label elsewhere. There are 26 hate and 674 non-hate sentences. The annotators are unanimous on 99/100 shared sentences; Fleiss kappa is 0.9464.

The classification threshold is fixed at 0.5. Results are the mean of the metrics of seeds 42, 43 and 44, rather than the metric of averaged predictions. SD is the sample standard deviation across seeds. The 95% intervals use 1,000 bootstrap resamples of sentences with seed 42; each resample is scored for every seed and the metrics are averaged. Resamples without positives are excluded.

| Subset | Metric | Mean (%) | SD (percentage points) | 95% bootstrap interval (%) |
| --- | --- | ---: | ---: | --- |
| all | f1-macro | 51.84 | 0.56 | 48.47 to 55.75 |
| all | f1-pos | 13.03 | 2.25 | 6.71 to 20.00 |
| all | precision | 8.06 | 1.14 | 3.93 to 13.00 |
| all | recall | 34.62 | 10.18 | 19.44 to 50.54 |
| all | acc | 83.14 | 1.96 | 80.62 to 85.52 |
| all | pr-auc | 10.36 | 3.19 | 5.34 to 20.30 |
| all | roc-auc | 71.21 | 1.18 | 61.69 to 79.71 |
| shared | f1-macro | 54.73 | 3.03 | 46.62 to 63.19 |
| shared | f1-pos | 19.20 | 3.49 | 3.17 to 34.81 |
| shared | precision | 14.70 | 3.87 | 2.08 to 31.75 |
| shared | recall | 28.57 | 0.00 | 8.33 to 53.33 |
| shared | acc | 82.67 | 4.16 | 76.33 to 88.67 |
| shared | pr-auc | 16.61 | 3.50 | 6.39 to 37.96 |
| shared | roc-auc | 71.63 | 2.83 | 56.02 to 86.06 |

Hate-class performance is low: seeds 42, 43 and 44 detect 6, 10 and 11 of the 26 gold-hate sentences, with 83, 105 and 115 false positives respectively. The low positive prevalence makes accuracy alone misleading. These test results were not used for tuning.

The saved best_model weights were checked by SHA-256 against the development-selected checkpoint: checkpoint-1205 for seeds 42 and 43, and checkpoint-964 for seed 44. Runtime: Python 3.12.15, PyTorch 2.11.0+cu128, Transformers 5.19.0. Full metadata and input/checkpoint hashes are included in [the metrics JSON](swissbert_raw_metrics.json).

Files: [combined gold set](../data/eval_gold_700.csv), [all scored sentences](swissbert_raw_predictions.csv), [per-seed metrics and confusion counts](swissbert_raw_by_seed.csv). Mean probability in the scored export is provided for inspection; the reported metrics evaluate each seed separately.

Validation: all predictions have one finite probability per gold sentence; all six per-seed/subset metric rows were recomputed from the exported scores and matched. The targeted prediction, annotation, agreement and evaluation tests passed (31 tests). Python compilation and git diff whitespace checks passed. Black and Pyflakes are not installed in the available environments.
