# Code conventions for this repository

This project follows the way Janis Goldzycher writes research code (GAHD, `xnli4xhsd`, `CL-UZH-EDOS-2023`) and the AustroTox repository (Pachinger et al., co-authored by him). Follow these rules for every Python file, shell script, config and README change. Where they disagree with `ows-code-guidelines` on naming or layout, these rules win in this repository. The em dash ban and the comment quota from the global instructions still hold.

Sources studied, with the files each rule comes from: `xnli4xhsd/src/{train,evaluation,get_loggers,parse_results_into_csv}.py`, `CL-UZH-EDOS-2023/src/{exec_scripts,get_loggers,train}.py`, `austrotox/src/{compute_metrics,predict}.py`, and the READMEs of all four repositories. The data-stage and file-naming rules below come from `efficient-low-resource-hate-detection`. That repository is Paul Rottger's code (EMNLP 2022). Janis's fork of it has no commits of his own, so treat those rules as the lab's shared convention, not as his.

## Layout

- One flat `src/` of single-purpose scripts, one file per pipeline step, named `verb_object.py`: `train.py`, `predict.py`, `evaluate_predictions.py`, `compute_data_stats.py`, `parse_results_into_csv.py`, `delete_checkpoints.py`.
- Shared helpers live in their own small modules (`get_loggers.py`, `dataset.py`, `config.py`), never copied between scripts.
- `configs/` holds one JSON file per run. `scripts/` holds bash launchers named `run_eval_<model>_<variant>.sh` and `training_runs_<what>.sh`. `notebooks/` holds plots and exploration only, no pipeline logic. `data/` holds inputs, `tests/` holds pytest files.
- Paths come from a gitignored `paths.json` (`data_dir`, `output_dir`, `configs_dir`) or from `--path_*` flags, never from hard-coded absolute paths.

## Entry points

- Every script is `def main(args: argparse.Namespace) -> None` plus an `if __name__ == "__main__":` block that builds the `argparse.ArgumentParser`, calls `parse_args()` into `cmd_args` and calls `main(cmd_args)`.
- Flags are long snake_case. Paths are `--path_<thing>`, counts are `--num_<thing>`, the seed is `--seed` with default 42. Booleans use `action="store_true"`. Closed sets use `choices=[...]`. Every flag has `help=`. Group flags with one-word comments such as `# data`, `# hyperparams`, `# saving`.
- A config JSON section has the same keys as the argparse dests of the step it feeds. A thin launcher runs `step.main(argparse.Namespace(**config[key]))`, so the CLI and the config share one vocabulary.

## Naming

- Functions are `verb_object`: `load_` for loaders (`load_predictions`), `compute_` for metrics and statistics (`compute_binary_metrics`), `get_` for look-ups (`get_true_labels`), `check_` for validation, `extract_`, `parse_`, `map_`, `to_` for conversions. Variants take a suffix: `compute_metrics_hatecheck`, `compute_metrics_default`.
- Variables name their role: `true_labels`, `pred_labels`, `train_set`, `eval_set`, `raw_preds`, `preds_per_cat`, `run_path`, `cmd_args`. File handles are `fin` and `fout`. Paths carry the `path_` prefix.
- Constants are `UPPER_SNAKE` (`SEED`, `N_SPLITS`, `MAX_SEQ_LENGTH`, `CHECKPOINT_REGEX`). Classes are short `CamelCase` with the parent as suffix (`Dataset`, `DefaultDataset`, `HateCheckDataset`). Base methods raise `NotImplementedError`.
- Dataset ids are `<ABBREV><YY>_<LANG>` (`BAS19_ES`). Result files are `<DATASET>_eval.json`. Prediction files that depend on a threshold end in `_metrics_<threshold>.json`.
- Experiment names encode every factor and end in the run: `<DATASET>_<TASK>_<FACTOR>_<value>_..._run<N>`. The same stem is the config file name, `experiment_name`, `run_name` and output directory. Run directories are `run<N>` and the seed is the run number.
- This project's vocabulary: label `1` is hate, `0` is not hate. Sentences labelled by all three annotators are `shared`, the other 200 per annotator are `single_annotator`. Use these two terms everywhere (code, figures, report).

## Data stages and file names

- Number the data stages of any new pipeline by order and let each stage read only the previous stage's output: `data/0_raw/`, `data/1_clean/`, and so on. This repository keeps its existing flat `data/` because most files are large, local and gitignored.
- Encode size and seed in the file name: `train_<n>_rs<k>.csv`, `dev_<n>.csv`, `test_<n>.csv`, where `<n>` is the real number of rows. Use the same `<model>_<dataset>_<n>_<seed>` pattern for model directories and result files. Never parse names by `split("_")` index, keep the parts in a dict or dataclass.
- Seeds are integers: `SEEDS`, then `df.sample(n, random_state=seed)` for each seed. The train, dev and test split uses one fixed `random_state` constant. Sample sizes are one shared constant list.
- Write one prediction CSV per (model, test set, seed) with the columns `index,label,prediction` and `prob_hate` when scores exist. Compute metrics later from these files. Aggregation flattens results into a long dataframe and asserts the expected number of seeds per cell before averaging.
- Fix the label space (`label_list = [0, 1]`) so tiny splits that contain one class still work.

## Metrics and results

- Metrics are explicit sklearn calls returned as one flat dict with lowercase hyphenated keys: `acc`, `f1-macro`, `f1-pos`, `f1-neg`, `f1-micro`, `recall`, `precision`, `num_examples`, `num_true_hate`, `num_pred_hate`, `true_hs_ratio`, `pred_hs_ratio`. Always report both the hate-class F1 and the macro F1.
- Store raw predictions next to the metrics, so scoring can be rerun without inference. Metrics are computed by a separate script from stored predictions.
- Everything is JSON or JSONL (`ensure_ascii=False, indent=4`), with `id`, `text`, `label`, `split` as record keys. One results file per run, aggregated by a separate `parse_results_into_csv.py` that writes `raw_results.csv` and `seed_avg_results.csv` (mean and standard deviation over seeds with `groupby`).
- Choose the best checkpoint by `eval_f1-macro` on the development split with early stopping. Test data is never used for selection.
- Refuse to overwrite: if the output directory is not empty, raise an exception. Runs never mix.

## Reproducibility

- Seed `torch`, `random` and `numpy` at the top of `main` from `args.seed`. Subsampling takes the seed as an argument.
- `requirements.txt` has exact `==` pins and no notebook-only packages. The README names the Python version and gives the venv commands.
- Log through `get_logger(name)` from `get_loggers.py` (format `[%(asctime)s] [%(levelname)s] %(message)s`, file handler plus stdout handler), one logger per module named after its role (`train`, `eval`, `pred`). Log the full `cmd_args` as JSON at the start of every script. Never use bare `print` in pipeline code.

## Errors and logging

- Validate at the boundary (flags, config keys, input files, CSV columns) and fail fast with a specific exception and a message that names the offending value: `FileNotFoundError` for missing paths, `ValueError` for bad values, `KeyError` listing the missing columns.
- Catch only what the code can handle, with the narrowest exception type. Never use a bare `except:`, never swallow an error. To add context, log it with `logger.exception(...)` or `logger.error(...)` and re-raise.
- Scripts that fail exit non-zero. Long runs (training, API calls) log the stage start and end (`'Start training.'`, `'Finished training.'`), log warnings for recoverable problems (`logger.warning`), and write checkpoints so a rerun can resume.
- Log levels: `info` for stages and counts, `warning` for recoverable data problems, `error` for failures. Never log secrets or API keys.

## Comments and docstrings

- Sparse. Roughly 3 to 7 percent of lines at most, and fewer in new code. A one-line docstring on public functions, optional `Args:` block only when a parameter is not obvious. No narrating comments, no section banners, no commented-out code, no TODOs.
- Cite the source of copied logic in a comment (`# adjusted from: <url>`).

## README

- First line links the paper or states what the repository provides. Then `## Setup` (environment, data), then `## Run Experiments` with one fenced bash block per experiment family, then the command that aggregates the results. No prose about internals.
- Dataset README lists every column in backticks with a one-line gloss and the value legend (`0` = not hate speech), then the download link and a BibTeX block.

## What not to copy

- Bare `except:`, `import pdb`, `from typing import *`, `from module import *`, global mutable state fed into metric functions, duplicated helpers, hard-coded cluster paths, commented-out blocks, `type=bool` flags, a `pip freeze` containing notebook packages, metrics on hard predictions only without uncertainty.
- Add what the course requires and these repositories lack: type hints on every signature, unit tests with pytest in `tests/`, a Kanban board on the GitHub project, bootstrapped 95 percent intervals next to seed means.
