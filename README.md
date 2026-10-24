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

A `.env` file holds `HF_TOKEN` (model downloads) and `ANTHROPIC_API_KEY` (only for a new translation request). The GAHD data is downloaded from the Hugging Face hub on first use.

## Run Experiments

### Test set

The step scripts build the 700-sentence test set from the filtered SwissCrawl candidates. The A2 input is the `translation` column of `data/master_sample_stratified.csv`. The raw API checkpoint was not kept, so rerunning `src/translate_master.py` sends a new paid request.

```bash
bash scripts/build_swisscrawl_testset.sh --resume
```

Annotate `label` with `1` or `0` in `data/annotation_<annotator>_stratified.csv` and use `comment` for uncertainty. Annotators see neither GAHD scores nor translations.

```bash
python src/compute_annotation_agreement.py --path_annotations data --path_agreement results/annotation_agreement.json
```

### Approach 1: GBERT on raw Swiss German

`deepset/gbert-base` fine-tuned on GAHD with seeds 42, 43, 44, then applied to the raw Swiss German text.

```bash
bash scripts/train_gbert_raw.sh
```

Each run writes `runs/gahd_gbert_raw/run<seed>/` with `gahd_test_predictions.csv`, `gahd_test_metrics.json`, the best checkpoint and `predictions_swiss_german_text.csv`. The configs in `configs/` have one section per step (`train`, `predict`, `predict_translation`) whose keys equal the argparse flags.

### Approach 2: GBERT on normalised input

The same fine-tuned model applied to the Standard German `translation` column, produced by `claude-haiku-5-5` with a fixed four-example prompt. This replaces the Apertus-70B model named in the proposal and must be stated in the report. The `predict_translation` config section writes `predictions_swiss_german_translation.csv` into each run directory.

### Approach 3: SwissBERT on raw Swiss German

`notebooks/train_approach3.ipynb` fine-tunes SwissBERT on GAHD (`de_CH` adapter for training, `gsw` for the dialect, five epochs, learning rate `2e-5`, seeds 42, 43, 44). Its Swiss German predictions have to be exported as a CSV with the columns `sentence_id` and `prob_hate` to be scored with the commands below.

### Evaluate

```bash
python src/evaluate_predictions.py --path_predictions runs/gahd_gbert_raw/run42/predictions_swiss_german_text.csv --path_metrics runs/gahd_gbert_raw/run42/swiss_german_text_metrics.json
python src/parse_results_into_csv.py --path_runs runs/gahd_gbert_raw --num_runs 3
```

`evaluate_predictions.py` writes one flat metrics JSON (`acc`, `f1-macro`, `f1-pos`, `f1-neg`, `precision`, `recall`, `pr-auc`, `roc-auc`, ...) and a bootstrap interval file next to it. `parse_results_into_csv.py` writes `raw_results.csv` and `seed_avg_results.csv` (mean and standard deviation over seeds). Gold labels are the majority of the three annotators on the shared sentences and the single label elsewhere.

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
data/       inputs, annotations and the 700-sentence master sample
runs/       training output (gitignored)
```

Repository conventions are in `CLAUDE.md`. They follow the research code of Janis Goldzycher (GAHD, `xnli4xhsd`, `CL-UZH-EDOS-2023`) and AustroTox.

## Reproducibility

- Seeds 42, 43 and 44 for training, seed 42 for sampling and bootstrap resampling.
- GAHD scorer pinned to `jagoldz/gahd` revision `90c6e4f50d109c42bebdeb0f43fb5c3e619612fc`.
- Translation model `claude-haiku-5-5` with prompt version `swiss-german-standard-few-shot-batch-v2`.
- Package versions are pinned in `requirements.txt`. Before reporting results, record the Python version, hardware, device and the selected checkpoint of every run.
- Annotation CSVs are tracked in Git. Raw corpora, model checkpoints and API keys are not.

## Verification

```bash
python -m pytest
python -m black --check src tests
python -m pyflakes src tests
```
