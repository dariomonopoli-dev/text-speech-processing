# Investigation of GBERT result differences

Investigated 10 November 2026. The strongest confirmed contributor is the physical training batch: the colleagues' report specifies batches of 32; the local rerun used batches of 8 with four accumulation steps. These share effective batch size 32 but do not reproduce identical dropout masks or training trajectories. Data and scoring mismatches were ruled out. The remaining difference cannot be uniquely attributed without the original runs and environment records.

## Three-seed controlled comparison

Values below are means over seeds 42, 43 and 44. The batch-32 controls use the same local GPU, pretrained weights, GAHD contents, learning-rate schedule, FP32 precision and development-selection criterion as the local 8 x 4 run. They retain the cleaned 1,648-example development split. Memory checkpointing with RNG preservation allows full batches to fit; its loss and gradients were bit-identical in a real-model probe.

| Input | Metric | Colleagues | Local 8 x 4 | Local batch 32 control |
| --- | --- | ---: | ---: | ---: |
| raw | f1-macro | 0.395 +/- 0.011 | 0.364 +/- 0.050 | 0.391 +/- 0.043 |
| raw | f1-pos | 0.087 +/- 0.009 | 0.101 +/- 0.016 | 0.101 +/- 0.015 |
| raw | recall | 0.577 +/- 0.077 | 0.782 +/- 0.022 | 0.705 +/- 0.160 |
| raw | pr-auc | 0.057 +/- 0.009 | 0.077 +/- 0.006 | 0.061 +/- 0.010 |
| Apertus-8B | f1-macro | 0.464 +/- 0.022 | 0.453 +/- 0.024 | 0.474 +/- 0.013 |
| Apertus-8B | f1-pos | 0.115 +/- 0.014 | 0.119 +/- 0.019 | 0.132 +/- 0.014 |
| Apertus-8B | recall | 0.538 +/- 0.000 | 0.615 +/- 0.038 | 0.615 +/- 0.067 |
| Apertus-8B | pr-auc | 0.118 +/- 0.021 | 0.163 +/- 0.069 | 0.142 +/- 0.051 |
| Haiku | f1-macro | 0.536 +/- 0.007 | 0.530 +/- 0.004 | 0.537 +/- 0.008 |
| Haiku | f1-pos | 0.166 +/- 0.019 | 0.164 +/- 0.006 | 0.174 +/- 0.018 |
| Haiku | recall | 0.462 +/- 0.102 | 0.487 +/- 0.080 | 0.500 +/- 0.077 |
| Haiku | pr-auc | 0.158 +/- 0.027 | 0.159 +/- 0.032 | 0.185 +/- 0.048 |

Matching batches reduces the raw macro-F1 discrepancy from approximately 0.031 to 0.004 and the Haiku macro-F1 discrepancy from approximately 0.006 to 0.001. It does not reproduce every result: raw recall remains 0.705 versus the colleagues' 0.577, and Apertus macro-F1 is 0.474 versus 0.464. Consequently, batching is a demonstrated contributor, not a complete reconstruction of the original training environment.

## What was ruled out

- The colleagues' tracked score export and the local evaluation contain exactly the same 700 IDs, texts, gold labels and both translations (zero differences).
- The local GAHD CSV and a fresh download of public revision 5d9a908a8aadb911d0dd441164976ef8cf13214e have the same SHA-256 and all 10,996 rows match. The only intentional loader difference is removing one blank development example; no training examples were removed.
- Evaluation, seed aggregation, gold-label construction and prediction code have identical Git blob hashes at the colleagues' analysis commit 25ccc5c and the current committed version.
- Repeating inference in batches of 32 rather than 8 changes zero classifications across all nine original local GBERT prediction files. The largest probability difference is about 2.22e-6.
- Adding the blank development example back, under either possible predicted class, cannot change the selected checkpoint for any of the three local reference models. The validation-score bounds do not overlap competing checkpoints.

## Why batching matters

The gradient probe uses identical pretrained weights, examples, seed and per-example objective. With dropout disabled, gradients for batch 32 versus 8 x 4 are nearly identical (relative difference about 0.013%; cosine approximately 1). With dropout enabled, the relative gradient difference is about 132% and the cosine is 0.430. Splitting forward passes changes which dropout draws are applied to which examples, and dynamic padding also changes tensor shapes and random-number consumption. Matching the effective batch therefore preserves the intended gradient averaging, not an identical stochastic run.

For seed 42, batching alone changes Swiss German raw recall from 0.769 to 0.577 while German test macro-F1 changes only from 0.7872 to 0.7893. Seed 43 moves in the opposite direction on Swiss recall, so this is seed-dependent training sensitivity rather than a constant correction to probabilities. Across all three controls, German test macro-F1 changes by less than 0.009 per seed.

The accumulation-loss defect previously found is not evidence that the colleagues' original GBERT training was wrong: their batch-32 configuration uses one accumulation step and does not suffer the fourfold scaling issue. Both sets of local runs here use the explicit corrected loss.

## What the supplied report adds

The draft explicitly lists batches of 32, seeds 42/43/44, three epochs, learning rate 2e-5 and 7,701/1,649/1,646 GAHD splits. Its limitations section already documents a same-seed rerun on different hardware changing raw Swiss German recall from 0.81 to 0.69 while GAHD macro-F1 stays within 0.01. That is consistent with target-domain training sensitivity. It does not establish that hardware alone explains the current gap, since our same-hardware batching control also changes results.

The report does not identify the colleagues' actual GBERT device or installed Python/PyTorch/Transformers versions. Repository requirements are declared pins, not a runtime manifest; they differ from the local runtime (PyTorch 2.11.0+cu128, Transformers 5.19.0). Kernel arithmetic, library/tokenizer/optimizer implementations, stochastic training and consequent checkpoint choices remain possible contributors to the residual differences. Their individual contributions cannot be measured from the available files.

## Interpretation

The earlier local results remain valid measurements of the documented 8 x 4 recipe. They should not be described as an exact reproduction of the original physical-batch-32 run. The batch-32 results are included in the separate [physical-batch-32 comparison](../batch32_comparison_20261110/report.md); the earlier 8 x 4 comparison remains available. For a report, choose and document one run protocol consistently; if using the new runs, update the old prediction-based error analysis as well.

With only 26 positives, changes to a few predictions and positive-example ranks materially affect recall and average precision. The colleagues' report uses mean per-seed metrics for its main table and thresholded mean scores for its error analysis; these are different aggregations, and their counts need not match. Our comparisons of main metrics use mean per-seed metrics.

[Comparison including precision and SDs](comparison.csv) | [Batch-32 results by seed](batch32_control_by_seed.csv) | [Machine-readable evidence and limitations](evidence.json)

Original colleague per-seed models, probabilities and runtime manifests were unavailable. Controls preserve the same GPU and packages but add memory checkpointing; the checkpointing probe matched exactly. The local seed-44 reference was resumed, whereas the control was fresh. These limits prevent claiming a unique, exact causal decomposition of every reported metric difference.

The GBERT control training and prediction recipe is available in [run_gbert_batch32.py](../../src/run_gbert_batch32.py). The [README](../../README.md) documents training, evaluation and export commands.
