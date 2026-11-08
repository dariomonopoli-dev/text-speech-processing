"""Export the gold-hate sentences with both translations and seed-mean hate scores."""

import argparse
import json
from pathlib import Path

import pandas as pd

from evaluate_predictions import load_predictions
from get_loggers import get_logger
from gold_labels import load_gold_labels
from parse_results_into_csv import get_run_paths
from pipeline_io import write_csv

logger = get_logger("export")

SCORE_COLUMNS = {
    "text": "mean_prob_hate_raw",
    "translation_apertus8b": "mean_prob_hate_apertus8b",
    "translation_haiku": "mean_prob_hate_haiku",
}


def load_translations(path_translations: Path) -> pd.DataFrame:
    """Load a translation CSV indexed by sentence_id."""
    if not path_translations.is_file():
        raise FileNotFoundError(f"translation file not found: {path_translations}")
    translations = pd.read_csv(path_translations, encoding="utf-8-sig")
    missing_columns = sorted(
        {"sentence_id", "text", "translation"} - set(translations.columns)
    )
    if missing_columns:
        raise KeyError(f"{path_translations} lacks columns {missing_columns}")
    return translations.set_index("sentence_id")


def compute_mean_scores(path_runs: Path, variant: str, score_column: str) -> pd.Series:
    """Average the hate score of one prediction variant over all seeds."""
    scores_per_run = [
        load_predictions(
            run_path / f"predictions_swiss_german_{variant}.csv", score_column
        )
        for run_path in get_run_paths(path_runs).values()
    ]
    return pd.concat(scores_per_run, axis=1).mean(axis=1)


def check_covers_ids(frame: pd.DataFrame, sentence_ids: pd.Index, name: str) -> None:
    missing_ids = sentence_ids.difference(frame.index)
    if len(missing_ids):
        raise KeyError(f"{name} lacks sentence_ids {list(missing_ids[:5])}")


def main(args: argparse.Namespace) -> None:
    logger.info(json.dumps(vars(args), ensure_ascii=False, default=str))
    if args.path_output.exists() and not args.overwrite:
        raise FileExistsError(
            f"{args.path_output} exists, pass --overwrite to replace it"
        )

    gold = load_gold_labels(args.path_annotations)
    hate_ids = gold.index if args.subset == "all" else gold.index[gold["label"] == 1]
    apertus = load_translations(args.path_apertus)
    haiku = load_translations(args.path_haiku)
    check_covers_ids(apertus, hate_ids, str(args.path_apertus))
    check_covers_ids(haiku, hate_ids, str(args.path_haiku))

    different_texts = hate_ids[
        (apertus.loc[hate_ids, "text"] != haiku.loc[hate_ids, "text"]).to_numpy()
    ]
    if len(different_texts):
        raise ValueError(
            f"{len(different_texts)} sentences have different texts in the two "
            f"translation files, for example {list(different_texts[:5])}"
        )

    hate_sentences = pd.DataFrame(
        {
            "text": apertus.loc[hate_ids, "text"],
            "gold_label": gold.loc[hate_ids, "label"],
            "translation_haiku": haiku.loc[hate_ids, "translation"],
            "translation_apertus8b": apertus.loc[hate_ids, "translation"],
        }
    )
    for variant, column in SCORE_COLUMNS.items():
        mean_scores = compute_mean_scores(args.path_runs, variant, args.score_column)
        check_covers_ids(mean_scores.to_frame(), hate_ids, f"predictions for {variant}")
        hate_sentences[column] = mean_scores.loc[hate_ids]

    hate_sentences = hate_sentences.sort_index().rename_axis("sentence_id").reset_index()
    write_csv(hate_sentences, args.path_output)
    logger.info(f"wrote {len(hate_sentences)} sentences to {args.path_output}")


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
        "--path_haiku",
        type=Path,
        default=Path("data/haiku_translations_700.csv"),
        help="CSV with sentence_id, text and the Haiku translation.",
    )
    parser.add_argument(
        "--path_apertus",
        type=Path,
        default=Path("data/eval_sentences_apertus8b.csv"),
        help="CSV with sentence_id, text and the Apertus translation.",
    )
    parser.add_argument(
        "--path_runs",
        type=Path,
        default=Path("runs/gahd_gbert_raw"),
        help="Directory with the run<seed> folders.",
    )
    parser.add_argument(
        "--score_column", default="prob_hate", help="Column with the hate score."
    )
    parser.add_argument(
        "--subset",
        choices=["gold_hate", "all"],
        default="gold_hate",
        help="Export only the gold-hate sentences or all sentences.",
    )
    # saving
    parser.add_argument(
        "--path_output",
        type=Path,
        default=Path("results/gold_hate_sentences.csv"),
        help="CSV with one row per exported sentence.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing output file."
    )
    cmd_args = parser.parse_args()
    main(cmd_args)
