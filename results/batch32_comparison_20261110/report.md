# Physical-training-batch-32 comparison of Approaches 1, 2 and 3

Completed 2026-11-10. Classifier training and inference used the same NVIDIA GeForce RTX 3080 and Python environment. The existing Haiku and Apertus translations were reused; their generation was not repeated on this machine.

All variants use the same 700 gold sentences (26 hate), seeds [42, 43, 44], threshold 0.5, and corrected accumulation-loss normalization. Checkpoints were selected using GAHD development macro-F1. Test labels were used only for evaluation.

| Variant | Macro-F1 (%) | Hate F1 (%) | Precision (%) | Recall (%) |
| --- | ---: | ---: | ---: | ---: |
| GBERT, raw Swiss German | 39.11 ± 4.27 | 10.13 ± 1.52 | 5.47 ± 0.84 | 70.51 ± 16.01 |
| GBERT, Haiku | 53.75 ± 0.77 | 17.35 ± 1.82 | 10.50 ± 1.00 | 50.00 ± 7.69 |
| GBERT, Apertus-8B | 47.44 ± 1.35 | 13.17 ± 1.42 | 7.38 ± 0.82 | 61.54 ± 6.66 |
| SwissBERT, raw Swiss German | 52.89 ± 2.02 | 12.63 ± 3.17 | 8.55 ± 2.34 | 24.36 ± 4.44 |

For context, always predicting non-hate gives macro-F1 49.05%, hate F1 0% and accuracy 96.29%. The low positive prevalence makes accuracy alone misleading.

SwissBERT original checkpoints versus the selected runs on the same labels (training recipes may differ):

| Metric | Original (%) | Selected runs (%) | Change (percentage points) |
| --- | ---: | ---: | ---: |
| f1-macro | 51.84 | 52.89 | +1.05 |
| f1-pos | 13.03 | 12.63 | -0.40 |

Values are mean ± sample SD over the 3 seeds. The summary CSV also includes accuracy, ROC-AUC, PR-AUC, 95% bootstrap intervals and the shared-only subset. Mean probabilities in the prediction export are for inspection; reported metrics are averages of per-seed metrics, not ensemble metrics.

GBERT uses 3 epochs, maximum length 512, physical batch 32 and 1 accumulation step(s). SwissBERT uses 5 epochs, maximum length 128, physical batch 32 and 1 accumulation step(s). Both use learning rate 2e-5, FP32 and effective batch size 32. SwissBERT freezes embeddings and language adapters, trains with de_CH and predicts with gsw. This is not a comparison under identical architecture-specific training hyperparameters.

The original SwissBERT runs had a fourfold gradient-scale defect in the installed Transformers loss handling. Those checkpoints and scores remain available separately. Corrected training is reported here; this does not establish that the loss defect caused the original low scores, because clipping/Adam can offset a uniform gradient scale and final uneven batches also affect weighting. Any batching changes also affect the stochastic training trajectory, so this historical comparison does not isolate the loss correction when batches differ.

The [SwissBERT model card](https://huggingface.co/ZurichNLP/swissbert#training-details) states that its gsw adapter was pretrained on SwissCrawl. This test set also comes from SwissCrawl, so possible exposure to unlabeled test text should be disclosed. Sentence-level pretraining overlap has not been established.

The main comparison retains all translated sentences, including unchanged/flagged inputs. Haiku has 2 unchanged translations and 2 anomaly flags; Apertus has 49 unchanged translations and 50 anomaly flags. These counts describe translation quality checks, not confirmed translation errors.

[Per-seed results](by_seed.csv) · [Means, SDs and confidence intervals](summary.csv) · [All sentence scores](predictions.csv) · [Environment and checkpoint hashes](metadata.json)

All exported per-seed metrics were independently recomputed from the stored probabilities and matched their evaluation JSON files. All prediction files contain exactly one finite probability per gold sentence. Training-data hashes, hardware and package versions match across the 6 trained models.

GBERT batch-32 controls use activation checkpointing with RNG preservation to fit GPU memory. Its loss and gradients matched exactly with checkpointing disabled in the real-model probe. SwissBERT uses ordinary training without activation checkpointing.

## SwissBERT: physical training batch 32 versus corrected 8 x 4

Both experiments use the corrected loss, the same seeds (42, 43, 44), the same 700 test sentences and the same development-based checkpoint selection. All other recorded training settings match. Inference batches remain 8 in both experiments; the change is to training batches only.

| Metric | Corrected 8 x 4 | Physical 32 x 1 | Difference (percentage points) |
| --- | ---: | ---: | ---: |
| Macro-F1 | 0.530 +/- 0.005 | 0.529 +/- 0.020 | -0.16 |
| Hate F1 | 0.136 +/- 0.015 | 0.126 +/- 0.032 | -0.97 |
| Precision | 0.089 +/- 0.008 | 0.085 +/- 0.023 | -0.32 |
| Recall | 0.295 +/- 0.059 | 0.244 +/- 0.044 | -5.13 |
| PR-AUC | 0.098 +/- 0.031 | 0.095 +/- 0.009 | -0.31 |

Means and sample SDs are on the 0-1 scale in this table. The new runs do not improve average macro-F1 or hate F1. A paired sentence bootstrap (1,000 resamples, seed 42, holding the three trained seeds fixed) gives a macro-F1 difference of -0.16 percentage points, 95% interval [-2.16, +1.79], and a hate-F1 difference of -0.97 points, interval [-4.81, +2.60]. These intervals include zero and do not include uncertainty over new training seeds.

For the new SwissBERT report row, state a physical training batch size of 32 and gradient accumulation of 1. The previous 8 x 4 results remain a separate experiment. The report supplied during the investigation specifies SwissBERT 8 x 4; using these new scores requires updating that description.

[Batching comparison CSV](swissbert_batching_comparison.csv) | [Paired differences](swissbert_batching_paired.json) | [Run validation](validation.json)
