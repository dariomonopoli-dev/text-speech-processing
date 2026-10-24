"""Collect the metrics of all runs of one experiment into raw and seed-averaged CSVs."""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from evaluate_predictions import (
    check_predictions_cover_gold,
    check_true_labels,
    compute_bootstrap_intervals,
    get_eval_set,
    load_predictions,
    SUBSETS,
    write_json,
)
from get_loggers import get_logger
from gold_labels import load_gold_labels

logger = get_logger("parse")

RUN_DIR_REGEX = re.compile(r"^run(\d+)$")
GROUP_COLUMNS = ["experiment_name", "eval_name"]
RUN_COLUMNS = ["experiment_name", "run", "seed", "eval_name"]


def get_run_paths(path_runs: Path) -> dict[int, Path]:
    """Map each seed to its run<seed> directory."""
    if not path_runs.is_dir():
        raise FileNotFoundError(f"run directory not found: {path_runs}")
    run_paths = {}
    for run_path in sorted(path_runs.iterdir()):
        match = RUN_DIR_REGEX.match(run_path.name)
        if run_path.is_dir() and match:
            run_paths[int(match.group(1))] = run_path
    if not run_paths:
        raise FileNotFoundError(f"no run<seed> directories in {path_runs}")
    return run_paths


def get_metrics_paths(run_path: Path, metrics_pattern: str) -> list[Path]:
    """List the flat metrics files of a run, without bootstrap intervals."""
    return [
        path_metrics
        for path_metrics in sorted(run_path.glob(metrics_pattern))
        if not path_metrics.name.endswith("_bootstrap.json")
    ]


def load_raw_results(path_runs: Path, metrics_pattern: str) -> pd.DataFrame:
    """Load every metrics file of every run as one row."""
    rows = []
    for seed, run_path in get_run_paths(path_runs).items():
        for path_metrics in get_metrics_paths(run_path, metrics_pattern):
            with open(path_metrics, encoding="utf-8") as fin:
                metrics = json.load(fin)
            rows.append(
                {
                    "experiment_name": path_runs.name,
                    "run": seed,
                    "seed": seed,
                    "eval_name": path_metrics.stem,
                    **metrics,
                }
            )
    if not rows:
        raise FileNotFoundError(f"no files matching {metrics_pattern} in {path_runs}")
    return pd.DataFrame(rows)


def check_num_runs(raw_results: pd.DataFrame, num_runs: int) -> None:
    """Raise unless every evaluation has exactly num_runs runs."""
    runs_per_group = raw_results.groupby(GROUP_COLUMNS)["run"].apply(sorted)
    for group, runs in runs_per_group.items():
        if len(runs) != num_runs:
            raise ValueError(f"{group} has runs {runs}, expected {num_runs} runs")


def compute_seed_avg_results(raw_results: pd.DataFrame) -> pd.DataFrame:
    """Mean and sample standard deviation of every metric over seeds."""
    metric_columns = [
        column for column in raw_results.columns if column not in RUN_COLUMNS
    ]
    grouped = raw_results.groupby(GROUP_COLUMNS)
    seed_avg_results = grouped.size().rename("num_runs").to_frame()
    for metric in metric_columns:
        seed_avg_results[f"mean-{metric}"] = grouped[metric].mean()
        seed_avg_results[f"{metric}-stddev"] = grouped[metric].std()
    return seed_avg_results.reset_index()


def compute_seed_avg_bootstrap(args: argparse.Namespace) -> dict[str, float]:
    """Bootstrap intervals that score each resample on every seed, then average."""
    true_labels = load_gold_labels(Path(args.path_annotations))
    eval_set = get_eval_set(true_labels, args.subset)
    eval_labels = eval_set["label"].to_numpy()
    check_true_labels(eval_labels, args.subset)
    scores_per_run = []
    for run_path in get_run_paths(Path(args.path_runs)).values():
        path_predictions = run_path / args.predictions_name
        raw_preds = load_predictions(path_predictions, args.score_column)
        check_predictions_cover_gold(true_labels, raw_preds, path_predictions)
        scores_per_run.append(raw_preds.loc[eval_set.index].to_numpy())
    if len(scores_per_run) != args.num_runs:
        raise ValueError(
            f"found {len(scores_per_run)} runs in {args.path_runs}, "
            f"expected {args.num_runs}"
        )
    return compute_bootstrap_intervals(
        eval_labels, scores_per_run, args.threshold, args.num_resamples, args.seed
    )


def main(args: argparse.Namespace) -> None:
    logger.info(json.dumps(vars(args), ensure_ascii=False, default=str))
    if args.bootstrap and not args.predictions_name:
        raise ValueError("--bootstrap needs --predictions_name")
    path_runs = Path(args.path_runs)
    path_output_dir = Path(args.path_output_dir or path_runs)
    path_output_dir.mkdir(parents=True, exist_ok=True)

    raw_results = load_raw_results(path_runs, args.metrics_pattern)
    raw_results.to_csv(path_output_dir / "raw_results.csv", index=False)
    check_num_runs(raw_results, args.num_runs)
    seed_avg_results = compute_seed_avg_results(raw_results)
    seed_avg_results.to_csv(path_output_dir / "seed_avg_results.csv", index=False)
    logger.info(f"Wrote {len(raw_results)} runs to {path_output_dir}.")

    if args.bootstrap:
        logger.info(f"Start bootstrap with {args.num_resamples} resamples.")
        intervals = compute_seed_avg_bootstrap(args)
        path_bootstrap = path_output_dir / f"seed_avg_bootstrap_{args.subset}.json"
        write_json(intervals, path_bootstrap)
        logger.info(f"Wrote bootstrap intervals to {path_bootstrap}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    # data
    parser.add_argument(
        "--path_runs",
        required=True,
        help="Directory runs/<experiment_name>/ holding run<seed>/ subdirectories.",
    )
    parser.add_argument(
        "--metrics_pattern",
        default="*metrics*.json",
        help="Glob for the metrics files inside each run directory.",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=3,
        help="Number of runs every evaluation must have before averaging.",
    )
    # bootstrap
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Also write seed-mean bootstrap intervals from the stored predictions.",
    )
    parser.add_argument(
        "--predictions_name",
        help="File name of the prediction CSV inside each run directory.",
    )
    parser.add_argument(
        "--path_annotations",
        default="data",
        help="Directory with annotation_<annotator>_stratified.csv files.",
    )
    parser.add_argument(
        "--score_column",
        default="prob_hate",
        help="Column of the prediction CSVs that holds the hate score.",
    )
    parser.add_argument(
        "--subset",
        choices=SUBSETS,
        default="all",
        help="Bootstrap over all 700 sentences or only the 100 shared ones.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Scores at or above the threshold are predicted as hate.",
    )
    parser.add_argument(
        "--num_resamples",
        type=int,
        default=1000,
        help="Number of bootstrap resamples that contain at least one label 1.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed of the bootstrap resampling."
    )
    # saving
    parser.add_argument(
        "--path_output_dir",
        help="Directory for the two result CSVs, default --path_runs.",
    )
    cmd_args = parser.parse_args()
    main(cmd_args)
