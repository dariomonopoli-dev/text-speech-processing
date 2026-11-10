# Local comparison of Approaches 1, 2 and 3

Completed 2026-11-10. Classifier training and inference used the same NVIDIA GeForce RTX 3080 and Python environment. The existing Haiku and Apertus translations were reused; their generation was not repeated on this machine.

All variants use the same 700 gold sentences (26 hate), seeds [42, 43, 44], threshold 0.5, and corrected accumulation-loss normalization. Checkpoints were selected using GAHD development macro-F1. Test labels were used only for evaluation.

| Variant | Macro-F1 (%) | Hate F1 (%) | Precision (%) | Recall (%) |
| --- | ---: | ---: | ---: | ---: |
| GBERT, raw Swiss German | 36.37 ± 4.99 | 10.13 ± 1.59 | 5.42 ± 0.92 | 78.21 ± 2.22 |
| GBERT, Haiku | 53.00 ± 0.41 | 16.36 ± 0.60 | 9.86 ± 0.13 | 48.72 ± 8.01 |
| GBERT, Apertus-8B | 45.33 ± 2.39 | 11.91 ± 1.90 | 6.60 ± 1.12 | 61.54 ± 3.85 |
| SwissBERT, raw Swiss German | 53.05 ± 0.51 | 13.60 ± 1.49 | 8.87 ± 0.76 | 29.49 ± 5.88 |

For context, always predicting non-hate gives macro-F1 49.05%, hate F1 0% and accuracy 96.29%. The low positive prevalence makes accuracy alone misleading.

SwissBERT before and after corrected training on the same labels:

| Metric | Original (%) | Corrected (%) | Change (percentage points) |
| --- | ---: | ---: | ---: |
| f1-macro | 51.84 | 53.05 | +1.21 |
| f1-pos | 13.03 | 13.60 | +0.57 |

Values are mean ± sample SD over the 3 seeds. The summary CSV also includes accuracy, ROC-AUC, PR-AUC, 95% bootstrap intervals and the shared-only subset. Mean probabilities in the prediction export are for inspection; reported metrics are averages of per-seed metrics, not ensemble metrics.

GBERT uses three epochs and maximum length 512; SwissBERT uses five epochs and maximum length 128. Both use learning rate 2e-5, FP32 and effective batch size 32 (8 × 4 accumulation steps). SwissBERT freezes embeddings and language adapters, trains with de_CH and predicts with gsw. Their established recipes were retained; this is not a comparison under identical training hyperparameters.

The original SwissBERT runs had a fourfold gradient-scale defect in the installed Transformers loss handling. Those checkpoints and scores remain available separately. Corrected training is reported here; this does not establish that the loss defect caused the original low scores, because clipping/Adam can offset a uniform gradient scale and final uneven batches also affect weighting.

The [SwissBERT model card](https://huggingface.co/ZurichNLP/swissbert#training-details) states that its gsw adapter was pretrained on SwissCrawl. This test set also comes from SwissCrawl, so possible exposure to unlabeled test text should be disclosed. Sentence-level pretraining overlap has not been established.

The main comparison retains all translated sentences, including unchanged/flagged inputs. Haiku has 2 unchanged translations and 2 anomaly flags; Apertus has 49 unchanged translations and 50 anomaly flags. These counts describe translation quality checks, not confirmed translation errors.

[Per-seed results](by_seed.csv) · [Means, SDs and confidence intervals](summary.csv) · [All sentence scores](predictions.csv) · [Environment and checkpoint hashes](metadata.json)

All exported per-seed metrics were independently recomputed from the stored probabilities and matched their evaluation JSON files. All prediction files contain exactly one finite probability per gold sentence. Training-data hashes, hardware and package versions match across the 6 trained models.

Additional analyses: [paired GBERT differences](../local_gbert_paired_differences_20261110.json), [GBERT unflagged subset](../local_gbert_unflagged_subset_20261110.json), and [validation record](validation.json). The unflagged subset retains 650 sentences, including all 26 hate sentences.

For these three trained GBERT seeds, the paired 95% bootstrap interval for Haiku minus raw-text macro-F1 is +13.39 to +19.73 percentage points (point difference +16.63). Haiku minus Apertus is +5.21 to +10.06 points (point difference +7.67). These intervals resample test sentences with the trained models held fixed.

The `pr-auc` field records scikit-learn average precision. Modified Python files pass Black and Pyflakes; the repository-wide checks still report existing formatting issues and two undefined constants in an unrelated diagnostics plotting module, as recorded in validation.json.
