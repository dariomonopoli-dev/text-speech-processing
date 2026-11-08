# Hate Speech Detection for Swiss German

Code and annotations for the UZH course project "Essentials of Text and Speech Processing" by Marjorie Heim, Yannis Timon Kessler and Dario Monopoli: a 700-sentence Swiss German hate speech test set built from SwissCrawl, and a comparison of three ways to bridge the dialect gap for classifiers trained on the Standard German GAHD dataset.

The task is binary classification under the GAHD definition. Label `1` is abusive or discriminatory language against a marginalised group, or against a person as a member of one. Label `0` is everything else, including profanity, insults at individuals, and insults at regional or majority groups.

Sentences labelled by all three annotators are `shared` (100). The other 200 per annotator are `single_annotator`.

## Setup

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

A `.env` file holds `HF_TOKEN` (model downloads) and `ANTHROPIC_API_KEY` (only for a new translation request). The GAHD data is downloaded from the Hugging Face hub on first use. `mlx` and `mlx-lm`, which translate with a local model in Approach 2, install on Apple silicon only.

## Run Experiments

### Test set

The step scripts build the 700-sentence test set from the filtered SwissCrawl candidates. The raw API checkpoint was not kept, so rerunning `src/translate_master.py` sends a new paid request.

```bash
bash scripts/build_swisscrawl_testset.sh --resume
```

Annotate `label` with `1` or `0` in `data/annotation_<annotator>_stratified.csv` and use `comment` for uncertainty. Annotators see neither GAHD scores nor translations.

```bash
python src/compute_annotation_agreement.py --path_annotations data --path_agreement results/annotation_agreement.json
```

### Approach 1: GBERT on raw Swiss German

`deepset/gbert-base` fine-tuned on GAHD with seeds 42, 43, 44 (3 epochs, batch size 32, learning rate `2e-5`), then applied to the raw Swiss German text.

```bash
python src/build_eval_sentences.py
bash scripts/train_gbert_raw.sh
```

`build_eval_sentences.py` collects the 700 sentences from the three annotation files into `data/eval_sentences.csv`. Each run writes `runs/gahd_gbert_raw/run<seed>/` with `gahd_test_predictions.csv`, `gahd_test_metrics.json`, the best checkpoint and `predictions_swiss_german_text.csv`. The configs in `configs/` have one section per step (`train`, `predict`, `predict_translation_haiku`, `predict_translation_apertus8b`) whose keys equal the argparse flags. The two translation sections read the files prepared under Approach 2, so run those steps before the launcher.

### Approach 2: GBERT on normalised input

The same fine-tuned models applied to Standard German translations of the 700 sentences from two normalisers:

- `claude-haiku-5-5` with a fixed four-example prompt (`data/haiku_translations_700.csv`).
- `mlx-community/Apertus-8B-Instruct-2509-bf16`, run locally with MLX in batches of 50 sentences with greedy decoding. It replaces the Apertus-70B model named in the proposal, which does not fit into 48 GB of memory. This must be stated in the report.

Translate with Apertus. Batch size 50 left 2 to 4 percent of the sentences unchanged on two test samples, batch sizes 10, 100 and 200 left 18 to 23 percent.

```bash
python src/translate_local.py --path_output data/eval_sentences_apertus8b.csv --path_checkpoint tmp/apertus8b_translation.jsonl --model_name mlx-community/Apertus-8B-Instruct-2509-bf16 --batch_size 50
```

Flag translations that equal the Swiss German source (`translation_unchanged`, status `manual_review`) and prepare the prediction inputs. The flagged sentences stay in the predictions.

```bash
python src/audit_translations.py --path_translations data/haiku_translations_700.csv --path_output data/haiku_translations_700_audited.csv
python src/prepare_prediction_input.py --path_translations data/haiku_translations_700_audited.csv --input_column translation_haiku --path_output data/pred_input_haiku.csv
python src/prepare_prediction_input.py --path_translations data/eval_sentences_apertus8b.csv --input_column translation_apertus8b --path_output data/pred_input_apertus8b.csv
```

`scripts/train_gbert_raw.sh` writes `predictions_swiss_german_translation_haiku.csv` and `predictions_swiss_german_translation_apertus8b.csv` into each run directory. For runs that are already trained, call `predict.py` directly:

```bash
for seed in 42 43 44; do
  for name in haiku apertus8b; do
    python src/predict.py --experiment_name gahd_gbert_raw --path_eval_set data/pred_input_$name.csv --input_column translation_$name --batch_size 32 --no_fp16 --path_runs_dir runs --seed $seed
  done
done
```

### Approach 3: SwissBERT on raw Swiss German

`notebooks/train_approach3.ipynb` fine-tunes SwissBERT on GAHD (`de_CH` adapter for training, `gsw` for the dialect, five epochs, learning rate `2e-5`, seeds 42, 43, 44). Its Swiss German predictions have to be exported as a CSV with the columns `sentence_id` and `prob_hate` to be scored with the commands below.

### Baseline: TF-IDF logistic regression

Character 2- to 5-gram TF-IDF features within word boundaries and a logistic regression, trained on GAHD and applied to the raw text and both translations. The model is deterministic, so all seeds give the same predictions.

```bash
for seed in 42 43 44; do
  python src/train_baseline.py --seed $seed
done
```

Each run writes `runs/gahd_tfidf_logreg/run<seed>/` with the GAHD test metrics and `predictions_swiss_german_{text,translation_haiku,translation_apertus8b}.csv`.

### Evaluate

```bash
bash scripts/evaluate_swiss_german.sh
python src/evaluate_unflagged_subset.py --path_translations data/eval_sentences_apertus8b.csv data/haiku_translations_700_audited.csv --path_output results/unflagged_subset_metrics.json
python src/compare_variants.py --path_output results/paired_differences.json
```

`evaluate_swiss_german.sh` recomputes the annotation agreement, then scores the `text`, `translation_haiku` and `translation_apertus8b` predictions of every seed. For one prediction file, call `evaluate_predictions.py` and `parse_results_into_csv.py` as the script does. `evaluate_predictions.py` writes one flat metrics JSON (`acc`, `f1-macro`, `f1-pos`, `f1-neg`, `precision`, `recall`, `pr-auc`, `roc-auc`, ...) and a bootstrap interval file next to it. `parse_results_into_csv.py` writes `raw_results.csv` and `seed_avg_results.csv` (mean and standard deviation over seeds). Gold labels are the majority of the three annotators on the shared sentences and the single label elsewhere.

`evaluate_unflagged_subset.py` scores only the sentences that neither normaliser left unchanged. `compare_variants.py` bootstraps the metric differences between variants (Haiku and Apertus-8B against the raw text, and against each other) on shared resamples of the test sentences.

### Figures

```bash
python src/plot_report_figures.py --path_figures_dir figures
```

## Layout

```
src/        one script per step plus small helper modules
configs/    one JSON config per training run
scripts/    bash launchers for the experiment families
notebooks/  exploration, baseline and Yannis's Approach 3 training
tests/      pytest suite for the pipeline, evaluation and training helpers
data/       inputs, annotations, translations and prediction inputs
results/    annotation agreement, subset metrics and paired differences
runs/       training output (gitignored)
```

Repository conventions are in `CLAUDE.md`. They follow the research code of Janis Goldzycher (GAHD, `xnli4xhsd`, `CL-UZH-EDOS-2023`) and AustroTox.

## Reproducibility

- Seeds 42, 43 and 44 for training, seed 42 for sampling and bootstrap resampling.
- GAHD scorer pinned to `jagoldz/gahd` revision `90c6e4f50d109c42bebdeb0f43fb5c3e619612fc`.
- Translation model `claude-haiku-5-5` with prompt version `swiss-german-standard-few-shot-batch-v2`. The Haiku CSV records no prompt version.
- Translation model `mlx-community/Apertus-8B-Instruct-2509-bf16` with prompt version `swiss-german-standard-few-shot-local-v1`, batch size 50, greedy decoding, `mlx==0.32.4` and `mlx-lm==0.32.2`. The prompt text is the one of the Haiku run.
- Package versions are pinned in `requirements.txt`. Before reporting results, record the Python version, hardware, device and the selected checkpoint of every run.
- Annotation CSVs are tracked in Git. Raw corpora, model checkpoints and API keys are not.

## Verification

```bash
python -m pytest
python -m black --check src tests
python -m pyflakes src tests
```
