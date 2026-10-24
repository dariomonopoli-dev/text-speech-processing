"""Compute metrics for one prediction file on the annotated Swiss German test set."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from get_loggers import get_logger
from gold_labels import load_gold_labels

logger = get_logger("eval")

SUBSETS = ["all", "shared"]
BOOTSTRAP_METRICS = [
    "acc",
    "f1-macro",
    "f1-pos",
    "f1-neg",
    "precision",
    "recall",
    "pr-auc",
    "roc-auc",
]


def load_predictions(path_predictions: Path, score_column: str) -> pd.Series:
    """Load the hate scores of one prediction CSV indexed by sentence_id."""
    if not Path(path_predictions).is_file():
        raise FileNotFoundError(f"prediction file not found: {path_predictions}")
    predictions = pd.read_csv(path_predictions, encoding="utf-8-sig")
    missing_columns = [
        column
        for column in ["sentence_id", score_column]
        if column not in predictions.columns
    ]
    if missing_columns:
        raise KeyError(f"{path_predictions} lacks columns {missing_columns}")
    return predictions.set_index("sentence_id")[score_column]


def check_predictions_cover_gold(
    true_labels: pd.DataFrame, raw_preds: pd.Series, path_predictions: Path
) -> None:
    """Raise if any gold sentence has no prediction."""
    missing_ids = true_labels.index.difference(raw_preds.index)
    if len(missing_ids):
        raise ValueError(
            f"{len(missing_ids)} test sentences have no prediction in "
            f"{path_predictions}, for example {list(missing_ids[:5])}"
        )


def get_eval_set(true_labels: pd.DataFrame, subset: str) -> pd.DataFrame:
    """Select the gold labels of one subset."""
    if subset == "all":
        return true_labels
    if subset == "shared":
        return true_labels[true_labels["is_shared"]]
    raise ValueError(f"unknown subset {subset!r}, expected one of {SUBSETS}")


def compute_binary_metrics(
    true_labels: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, float]:
    """Compute threshold and ranking metrics with label 1 as the positive class."""
    pred_labels = (scores >= threshold).astype(int)
    num_true_hate = int(true_labels.sum())
    num_pred_hate = int(pred_labels.sum())
    has_both_classes = 0 < num_true_hate < len(true_labels)
    return {
        "acc": accuracy_score(true_labels, pred_labels),
        "f1-macro": f1_score(
            true_labels, pred_labels, average="macro", zero_division=0
        ),
        "f1-pos": f1_score(true_labels, pred_labels, zero_division=0),
        "f1-neg": f1_score(true_labels, pred_labels, pos_label=0, zero_division=0),
        "precision": precision_score(true_labels, pred_labels, zero_division=0),
        "recall": recall_score(true_labels, pred_labels, zero_division=0),
        "pr-auc": (
            average_precision_score(true_labels, scores) if has_both_classes else np.nan
        ),
        "roc-auc": roc_auc_score(true_labels, scores) if has_both_classes else np.nan,
        "num_examples": len(true_labels),
        "num_true_hate": num_true_hate,
        "num_pred_hate": num_pred_hate,
        "true_hs_ratio": num_true_hate / len(true_labels),
        "pred_hs_ratio": num_pred_hate / len(true_labels),
    }


def compute_seed_mean_metrics(
    true_labels: np.ndarray, scores_per_run: Sequence[np.ndarray], threshold: float
) -> dict[str, float]:
    """Average the metrics of several runs scored on the same gold labels."""
    metrics_per_run = pd.DataFrame(
        [
            compute_binary_metrics(true_labels, scores, threshold)
            for scores in scores_per_run
        ]
    )
    return metrics_per_run.mean().to_dict()


def compute_bootstrap_intervals(
    true_labels: np.ndarray,
    scores_per_run: Sequence[np.ndarray],
    threshold: float,
    num_resamples: int,
    seed: int,
) -> dict[str, float]:
    """95 percent percentile intervals of seed-mean metrics over test resamples."""
    generator = np.random.default_rng(seed)
    resampled_metrics = []
    while len(resampled_metrics) < num_resamples:
        indices = generator.integers(0, len(true_labels), len(true_labels))
        # PR-AUC and ROC-AUC are undefined without positives.
        if true_labels[indices].sum() == 0:
            continue
        resampled_metrics.append(
            compute_seed_mean_metrics(
                true_labels[indices],
                [scores[indices] for scores in scores_per_run],
                threshold,
            )
        )
    resampled_df = pd.DataFrame(resampled_metrics)
    intervals = {}
    for metric in BOOTSTRAP_METRICS:
        intervals[f"{metric}-ci_low"] = float(resampled_df[metric].quantile(0.025))
        intervals[f"{metric}-ci_high"] = float(resampled_df[metric].quantile(0.975))
    intervals["num_resamples"] = num_resamples
    intervals["seed"] = seed
    return intervals


def check_true_labels(true_labels: np.ndarray, subset: str) -> None:
    """Raise if the gold labels of a subset cannot be scored."""
    if len(true_labels) == 0:
        raise ValueError(f"subset {subset!r} has no gold labels")
    if true_labels.min() == true_labels.max():
        raise ValueError(
            f"subset {subset!r} has only gold label {true_labels[0]}, AUC is undefined"
        )


def get_path_bootstrap(path_metrics: Path) -> Path:
    """Return the bootstrap interval file that sits next to a metrics file."""
    return path_metrics.with_name(f"{path_metrics.stem}_bootstrap.json")


def check_output_paths(output_paths: Sequence[Path], overwrite: bool) -> None:
    """Refuse to overwrite existing outputs unless asked to."""
    existing_paths = [str(path) for path in output_paths if path.exists()]
    if existing_paths and not overwrite:
        raise FileExistsError(
            f"{existing_paths} already exist, pass --overwrite to replace them"
        )


def write_json(output: dict[str, float], path_output: Path) -> None:
    """Write a flat dict as indented UTF-8 JSON."""
    path_output.parent.mkdir(parents=True, exist_ok=True)
    with open(path_output, "w", encoding="utf-8") as fout:
        json.dump(output, fout, ensure_ascii=False, indent=4)


def main(args: argparse.Namespace) -> None:
    logger.info(json.dumps(vars(args), ensure_ascii=False, default=str))
    path_metrics = Path(args.path_metrics)
    path_bootstrap = get_path_bootstrap(path_metrics)
    check_output_paths([path_metrics, path_bootstrap], args.overwrite)

    true_labels = load_gold_labels(Path(args.path_annotations))
    raw_preds = load_predictions(Path(args.path_predictions), args.score_column)
    check_predictions_cover_gold(true_labels, raw_preds, args.path_predictions)
    eval_set = get_eval_set(true_labels, args.subset)
    eval_labels = eval_set["label"].to_numpy()
    check_true_labels(eval_labels, args.subset)
    eval_scores = raw_preds.loc[eval_set.index].to_numpy()

    metrics = compute_binary_metrics(eval_labels, eval_scores, args.threshold)
    write_json(metrics, path_metrics)
    logger.info(f"Wrote metrics to {path_metrics}: {metrics}")

    logger.info(f"Start bootstrap with {args.num_resamples} resamples.")
    intervals = compute_bootstrap_intervals(
        eval_labels, [eval_scores], args.threshold, args.num_resamples, args.seed
    )
    write_json(intervals, path_bootstrap)
    logger.info(f"Wrote bootstrap intervals to {path_bootstrap}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    # data
    parser.add_argument(
        "--path_predictions",
        required=True,
        help="Prediction CSV with a sentence_id column and a score column.",
    )
    parser.add_argument(
        "--path_annotations",
        default="data",
        help="Directory with annotation_<annotator>_stratified.csv files.",
    )
    parser.add_argument(
        "--score_column",
        default="prob_hate",
        help="Column of the prediction CSV that holds the hate score.",
    )
    parser.add_argument(
        "--subset",
        choices=SUBSETS,
        default="all",
        help="Score all 700 sentences or only the 100 shared ones.",
    )
    # hyperparams
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
        "--path_metrics",
        required=True,
        help="Metrics JSON to write; intervals go to <stem>_bootstrap.json beside it.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing metrics and bootstrap files.",
    )
    cmd_args = parser.parse_args()
    main(cmd_args)
