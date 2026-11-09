# Approach 3 correctness audit

Audited on 9 November 2026. No correctness error was found that explains the weak Swiss German results. All three saved models reproduce every previously saved classification on GAHD dev, GAHD test and the 700 Swiss German sentences. The reported test results remain unchanged.

| Evaluation | Mean macro-F1 (%) | Mean hate F1 (%) |
| --- | ---: | ---: |
| GAHD test, de_CH | 79.98 | 77.18 |
| Swiss German, gsw (reported) | 51.84 | 13.03 |
| Swiss German, de_CH (diagnostic) | 45.76 | 11.34 |
| Swiss German, gsw, up to 512 tokens (diagnostic) | 51.87 | 13.06 |

The de_CH and longer-context runs are diagnostics performed after seeing test results. They are not newly selected configurations or replacements for the reported experiment.

Checks performed:

- Independently read all three annotation CSVs with Python csv.DictReader and reconstructed majority votes by sentence ID. The resulting 700 labels, sentence texts and shared flags exactly match the evaluation exports. All 900 labels are present and binary; there are 100 shared and 600 single_annotator sentences, with 26 gold positives.
- Reloaded all three checkpoints with no missing, unexpected or mismatched weights. Label 1 maps to hate. The saved best_model files exactly match the development-selected checkpoint files by SHA-256. Their selection scores are the maximum development macro-F1 values in each training history.
- Compared all 245 frozen parameter tensors in each checkpoint with the cached pretrained model: every embedding and language-adapter tensor is unchanged. All 192 shared parameter tensors present in the cached model changed during training. The classifier is functional, as confirmed by reproduction of the German results.
- Verified that the saved tokenizer produces the same token IDs as the cached pretrained tokenizer for all 700 Swiss German inputs. No text contains a Unicode replacement character. Token lengths have median 30 and maximum 136; just one non-hate sentence exceeds 128 tokens. Extending that sentence to full length changes one classification for seed 43 and none for the other seeds. Mean macro-F1 changes by only about 0.03 percentage points.
- Observed nonempty adapter batches in all 12 layers: only gsw processes the Swiss German samples when gsw is selected. Explicit lang_ids and set_default_language produce identical probabilities on the checked batch. Keeping de_CH active lowers mean Swiss German macro-F1.
- Independently reran inference and sklearn metrics without importing our prediction or gold-label helpers. There are zero classification differences across all three seeds and all three evaluated datasets. Maximum probability differences are below 1.5e-7 for German and 3e-8 for Swiss German, consistent with floating-point/CSV round-trip precision.
- Checked exact text overlap after stripping whitespace and case-folding: none between GAHD train/dev/test splits, and none between the Swiss German test set and GAHD. The highest seed-42 development score among the saved configurations belongs to the chosen five-epoch, 2e-5 setup.

The training procedure agrees with the [official X-MOD fine-tuning and transfer guidance](https://huggingface.co/docs/transformers/model_doc/xmod): freeze embeddings and language adapters while fine-tuning shared layers and the task head, then activate the target-language adapter for transfer.

One test weakness was corrected: the old adapter test counted calls even for empty input batches, so a call alone did not demonstrate active routing. The strengthened test counts actual samples and includes a de_CH negative control. All six SwissBERT prediction tests pass, including the strengthened routing test. This changes verification, not model predictions.

Two reproducibility gaps remain in the historical training record. The experiment JSONs contain a null base-model revision; the cached revision c1294db2e2eb48bfed66ee0b536861601164b92c matches all frozen tensors and tokenization, but the original revision was not recorded. The actual environment uses PyTorch 2.11.0+cu128 and Transformers 5.19.0, whereas the shared requirements file pins 2.14.1 and 5.18.0. The audit uses the original training environment and records this distinction in the JSON.

The performance drop cannot be attributed to dialect alone from this comparison: GAHD and SwissCrawl differ in domain and class balance as well. GAHD training is 42.4% hate, compared with 3.7% here. Mean Swiss German ROC-AUC is 71.2%, indicating some ranking signal despite poor precision and recall at threshold 0.5. Any threshold selection should use separate development data rather than these test labels.

The [SwissBERT model card](https://huggingface.co/ZurichNLP/swissbert#training-details) states that the gsw adapter was pretrained on SwissCrawl. Since this test set also comes from SwissCrawl, possible exposure to the unlabeled evaluation text should be disclosed; this audit does not establish sentence-level pretraining overlap.

Human label interpretation was not adjudicated. Two examples merit a guideline discussion, without assuming the model is correct: sentence 204878 is labelled 0 despite an ethnic stereotype, while sentence 56249 is labelled 1 and mixes a regional insult with gendered abuse. The stated guideline excludes regional-group insults but includes group-directed abuse. These are review candidates, not confirmed label errors, and no labels were changed.

[Machine-readable audit evidence](swissbert_approach3_audit.json). The local reproduction script is tmp/audit_approach3.py; replayed probabilities are in tmp/approach3_audit/. The original results and checkpoints are preserved.
