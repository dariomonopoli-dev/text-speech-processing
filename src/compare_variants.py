"""Paired bootstrap of the metric differences between prediction variants."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_predictions import (
    check_predictions_cover_gold,
    check_true_labels,
    compute_seed_mean_metrics,
    load_predictions,
    write_json,
)
from get_loggers import get_logger
from gold_labels import load_gold_labels
from parse_results_into_csv import get_run_paths

logger = get_logger("compare")

DIFFERENCE_METRICS = ["f1-macro", "f1-pos", "acc", "precision", "recall", "pr-auc", "roc-auc"]
DEFAULT_PAIRS = [
    "translation_haiku:text",
    "translation_apertus8b:text",
    "translation_haiku:translation_apertus8b",
]


def parse_pair(pair: str) -> tuple[str, str]:
    """Split 'a:b' into the variant a and the variant b whose metrics are subtracted."""
    parts = pair.split(":")
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"pair {pair!r} must look like 'variant_a:variant_b'")
    return parts[0], parts[1]


def compute_paired_differences(
    true_labels: np.ndarray,
    scores_per_variant: dict[str, list[np.ndarray]],
    pairs: list[tuple[str, str]],
    threshold: float,
    num_resamples: int,
    seed: int,
) -> dict[str, dict[str, dict[str, float]]]:
    """Difference of seed-mean metrics per pair with a 95 percent interval over shared resamples."""
    unknown_variants = sorted({v for pair in pairs for v in pair} - scores_per_variant.keys())
    if unknown_variants:
        raise KeyError(f"no predictions loaded for {unknown_variants}")
    full_metrics = {
        variant: compute_seed_mean_metrics(true_labels, scores_per_run, threshold)
        for variant, scores_per_run in scores_per_variant.items()
    }
    generator = np.random.default_rng(seed)
    resampled_metrics: dict[str, list[dict[str, float]]] = {v: [] for v in scores_per_variant}
    num_done = 0
    while num_done < num_resamples:
        indices = generator.integers(0, len(true_labels), len(true_labels))
        # PR-AUC and ROC-AUC are undefined without positives.
        if true_labels[indices].sum() == 0:
            continue
        for variant, scores_per_run in scores_per_variant.items():
            resampled_metrics[variant].append(
                compute_seed_mean_metrics(
                    true_labels[indices], [scores[indices] for scores in scores_per_run], threshold
                )
            )
        num_done += 1

    resampled_df = {v: pd.DataFrame(rows) for v, rows in resampled_metrics.items()}
    results: dict[str, dict[str, dict[str, float]]] = {}
    for variant_a, variant_b in pairs:
        pair_results = {}
        for metric in DIFFERENCE_METRICS:
            differences = resampled_df[variant_a][metric] - resampled_df[variant_b][metric]
            pair_results[metric] = {
                "difference": float(full_metrics[variant_a][metric] - full_metrics[variant_b][metric]),
                "ci_low": float(differences.quantile(0.025)),
                "ci_high": float(differences.quantile(0.975)),
                "share_resamples_above_zero": float((differences > 0).mean()),
            }
        results[f"{variant_a}_minus_{variant_b}"] = pair_results
    return results


def main(args: argparse.Namespace) -> None:
    logger.info(json.dumps(vars(args), ensure_ascii=False, default=str))
    if args.path_output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.path_output} exists, pass --overwrite to replace it")
    pairs = [parse_pair(pair) for pair in args.pairs]

    true_labels = load_gold_labels(args.path_annotations)
    check_true_labels(true_labels["label"].to_numpy(), "all")
    variants = sorted({variant for pair in pairs for variant in pair})
    scores_per_variant = {}
    for variant in variants:
        scores_per_run = []
        for run_path in get_run_paths(args.path_runs).values():
            path_predictions = run_path / f"predictions_swiss_german_{variant}.csv"
            raw_preds = load_predictions(path_predictions, args.score_column)
            check_predictions_cover_gold(true_labels, raw_preds, path_predictions)
            scores_per_run.append(raw_preds.loc[true_labels.index].to_numpy())
        scores_per_variant[variant] = scores_per_run

    logger.info(f"Start paired bootstrap with {args.num_resamples} resamples for {args.pairs}.")
    results = compute_paired_differences(
        true_labels["label"].to_numpy(),
        scores_per_variant,
        pairs,
        args.threshold,
        args.num_resamples,
        args.seed,
    )
    write_json(results, args.path_output)
    logger.info(f"Wrote results to {args.path_output}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    # data
    parser.add_argument(
        "--path_annotations",
        type=Path,
        default=Path("data"),
        help="Directory with annotation_<annotator>_stratified.csv files.",
    )
    parser.add_argument(
        "--path_runs",
        type=Path,
        default=Path("runs/gahd_gbert_raw"),
        help="Directory with the run<seed> folders.",
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=DEFAULT_PAIRS,
        help="Pairs 'variant_a:variant_b'; the metrics of b are subtracted from those of a.",
    )
    parser.add_argument(
        "--score_column", default="prob_hate", help="Column with the hate score."
    )
    # hyperparams
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Scores at or above the threshold are predicted as hate.",
    )
    parser.add_argument(
        "--num_resamples", type=int, default=1000, help="Number of bootstrap resamples."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed of the bootstrap resampling."
    )
    # saving
    parser.add_argument(
        "--path_output",
        type=Path,
        required=True,
        help="JSON file that receives the differences per pair and metric.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing output file."
    )
    cmd_args = parser.parse_args()
    main(cmd_args)
