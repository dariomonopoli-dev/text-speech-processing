# Hate Speech Detection for Swiss German

This repository implements the project proposed by Marjorie Heim, Yannis Timon
Kessler, and Dario Monopoli. It compares three ways of bridging the dialect gap
between the Standard German GAHD training data and a manually annotated Swiss
German evaluation set built from SwissCrawl.

The task is binary classification under the GAHD definition:

- `1`: abusive or discriminatory language targeting a marginalized group, or a
  person because of membership in such a group
- `0`: not hate speech

Generic profanity or an individual insult without protected-group targeting is
not sufficient for label `1`.

## Experimental approaches

All approaches use the same GAHD train, development, and test splits. Final
model selection must use the GAHD development split. The manually annotated
Swiss German test set is reserved for final evaluation.

### Approach 1: GBERT on raw Swiss German

`code/train.ipynb` fine-tunes `deepset/gbert-base` on GAHD and evaluates the
result directly on raw Swiss German. This is the no-dialect-handling condition
from the proposal.

Set `SEED` to `42`, `43`, and `44` in separate runs. Each run writes to
`runs/gbert_seed<SEED>/`:

- `best_model/`
- `test_results.json`
- `gahd_test_predictions.csv`

After all runs, execute `code/analyze_seeds_approach1.ipynb` to report every
seed plus the mean and sample standard deviation.

### Approach 2: GBERT on normalized input

Approach 2 reuses the fine-tuned GBERT model from Approach 1 but evaluates the
Standard German `translation` instead of the raw Swiss German `text`.
`code/normalize.ipynb` records the translation comparison experiments.

The proposal named locally run Apertus-70B-Instruct. The final implemented
test-set pipeline instead uses `claude-haiku-5-5` with effort `medium` and the
approved four-example German prompt. This implementation change must be stated
in the report. Haiku translates only the selected 700 sentences, not the full
candidate pool.

### Approach 3: SwissBERT on raw Swiss German

`code/train_approach3.ipynb` fine-tunes SwissBERT on GAHD. It activates `de_CH`
for German training and validation, then `gsw` for final raw Swiss German
evaluation. The final configuration uses five epochs, learning rate `2e-5`, and
seeds `42`, `43`, and `44`; outputs are stored in
`runs/swissbert_seed<SEED>_epochs5_lr2e-5/`.

### Baseline

`code/baseline.ipynb` provides a TF-IDF logistic-regression reference model. It
is useful as a sanity check but is not one of the proposal's three main
approaches.

## SwissCrawl evaluation set

`code/score_swisscrawl_gahd_colab.ipynb` is a Google Colab data-construction
notebook. It is not the Approach 1 multi-seed training script. It scores the
complete filtered SwissCrawl candidate pool with the pinned `jagoldz/gahd`
classifier, creates the fixed 700-row stratified sample, and exports the master
sample and sampling report. Haiku is not called by this notebook.

The equivalent local pipeline is:

```bash
python code/build_swisscrawl_testset.py score --resume
python code/build_swisscrawl_testset.py sample --resume
python code/build_swisscrawl_testset.py translate --resume
python code/build_swisscrawl_testset.py annotations --resume
python code/build_swisscrawl_testset.py verify --resume
```

The project uses the AustroTox score intervals with a documented,
project-specific allocation of `134/134/134/164/134`. GAHD scores are sampling
signals, never gold labels. Annotation is always performed on the original
Swiss German sentence.

Each annotator receives 300 rows:

- 100 shared rows labeled independently by all three annotators
- 200 private rows

Use the committed XLSX files for manual annotation. Fill `label` with `1` or
`0`, and use `comment` for uncertainty. The CSV files are canonical
machine-readable copies. Translations, GAHD scores, and strata are excluded from
annotator files to avoid bias.

The detailed sampling and validation rules are in the
[test-set design](docs/superpowers/specs/2026-10-08-swisscrawl-stratified-testset-design.md).

## Reproducibility

Create the environment from the repository root:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
```

`requirements-lock.txt` captures the complete Python 3.10 environment used for
the experiments. `requirements.txt` lists only direct dependencies and is
intended for development when exact environment reproduction is unnecessary.

Reproducibility controls include:

- fixed experiment seeds `42`, `43`, and `44`
- SwissCrawl sampling seed `42`
- pinned GAHD model revision
  `90c6e4f50d109c42bebdeb0f43fb5c3e619612fc`
- versioned Haiku model name and prompt
- corpus fingerprints bound to scoring and translation checkpoints
- resumable SQLite scoring checkpoints and atomic artifact writes
- committed sampling and verification reports
- per-seed model metrics and predictions rather than only aggregate scores

Set `ANTHROPIC_API_KEY` in `.env` only for a new Haiku request. The successful
batch checkpoint is retained as an audit artifact, so the translated master can
be reconstructed without repeating the paid request. API keys, raw corpora,
model checkpoints, and local SQLite files are ignored by Git.

Before reporting results, record the Python version, package versions, hardware,
device, model revisions, seed, training duration, and selected checkpoint for
every run. Report mean and sample standard deviation across the three seeds.

## Verification

```bash
python -m pytest -q
python -m pyflakes code/*.py tests/*.py
black --check code/*.py tests/*.py
python code/build_swisscrawl_testset.py verify --resume
```
