"""Score the seed-averaged predictions on sentences no normaliser left unchanged."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_predictions import (
    check_predictions_cover_gold,
    check_true_labels,
    compute_bootstrap_intervals,
    compute_seed_mean_metrics,
    load_predictions,
    write_json,
)
from get_loggers import get_logger
from gold_labels import load_gold_labels
from parse_results_into_csv import get_run_paths

logger = get_logger("eval")

FLAG_COLUMN = "translation_has_anomaly"


def load_flagged_ids(path_translations: Path) -> set[int]:
    """Return the sentence_ids flagged for manual review in one audited translation CSV."""
    if not path_translations.is_file():
        raise FileNotFoundError(f"translation file not found: {path_translations}")
    translations = pd.read_csv(path_translations, encoding="utf-8-sig")
    missing_columns = [
        column for column in ["sentence_id", FLAG_COLUMN] if column not in translations
    ]
    if missing_columns:
        raise KeyError(f"{path_translations} lacks columns {missing_columns}")
    return set(translations.loc[translations[FLAG_COLUMN], "sentence_id"])


def compute_subset_results(
    true_labels: np.ndarray,
    scores_per_run: list[np.ndarray],
    threshold: float,
    num_resamples: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Seed-mean metrics and bootstrap intervals of one prediction variant."""
    return {
        "metrics": compute_seed_mean_metrics(true_labels, scores_per_run, threshold),
        "intervals": compute_bootstrap_intervals(
            true_labels, scores_per_run, threshold, num_resamples, seed
        ),
    }


def main(args: argparse.Namespace) -> None:
    logger.info(json.dumps(vars(args), ensure_ascii=False, default=str))
    if args.path_output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.path_output} exists, pass --overwrite to replace it")

    flagged_ids = set()
    for path_translations in args.path_translations:
        flagged_ids |= load_flagged_ids(path_translations)
    logger.info(f"{len(flagged_ids)} sentences are flagged in at least one translation file.")

    true_labels = load_gold_labels(args.path_annotations)
    kept_labels = true_labels.loc[~true_labels.index.isin(flagged_ids)]
    check_true_labels(kept_labels["label"].to_numpy(), "unflagged")
    logger.info(
        f"Scoring {len(kept_labels)} sentences, {int(kept_labels['label'].sum())} labelled hate."
    )

    run_paths = get_run_paths(args.path_runs)
    results = {
        "num_examples": len(kept_labels),
        "num_true_hate": int(kept_labels["label"].sum()),
        "num_excluded": len(true_labels) - len(kept_labels),
        "variants": {},
    }
    for variant in args.variants:
        scores_per_run = []
        for seed, run_path in run_paths.items():
            path_predictions = run_path / f"predictions_swiss_german_{variant}.csv"
            raw_preds = load_predictions(path_predictions, args.score_column)
            check_predictions_cover_gold(true_labels, raw_preds, path_predictions)
            scores_per_run.append(raw_preds.loc[kept_labels.index].to_numpy())
        results["variants"][variant] = compute_subset_results(
            kept_labels["label"].to_numpy(),
            scores_per_run,
            args.threshold,
            args.num_resamples,
            args.seed,
        )
        logger.info(f"{variant}: {results['variants'][variant]['metrics']}")
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
        "--path_translations",
        type=Path,
        nargs="+",
        required=True,
        help="Audited translation CSVs; sentences flagged in any of them are excluded.",
    )
    parser.add_argument(
        "--path_runs",
        type=Path,
        default=Path("runs/gahd_gbert_raw"),
        help="Directory with the run<seed> folders.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["text", "translation_haiku", "translation_apertus8b"],
        help="Input columns whose prediction files are scored.",
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
        help="JSON file that receives metrics and intervals per variant.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing output file."
    )
    cmd_args = parser.parse_args()
    main(cmd_args)
