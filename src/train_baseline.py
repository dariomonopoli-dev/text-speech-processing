"""Train the TF-IDF logistic regression baseline on GAHD and predict the Swiss German test set."""

import argparse
import json
import os
import random

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline

from config import get_run_path
from dataset import load_gahd
from get_loggers import get_logger
from predict import get_predictions_path, load_eval_set
from train import (
    GAHD_COLUMNS,
    check_columns,
    check_empty_output_dir,
    compute_binary_metrics,
)

NGRAM_RANGE = (2, 5)
MAX_ITER = 1000
INPUT_COLUMNS = ["text", "translation_haiku", "translation_apertus8b"]

train_logger = get_logger("train")


def make_baseline() -> Pipeline:
    """Character n-gram TF-IDF features into a logistic regression."""
    return make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=NGRAM_RANGE),
        LogisticRegression(max_iter=MAX_ITER),
    )


def predict_eval_set(
    baseline: Pipeline, eval_set: pd.DataFrame, input_column: str
) -> pd.DataFrame:
    """Return one row per sentence with sentence_id, prediction and prob_hate."""
    prob_hate = baseline.predict_proba(eval_set[input_column])[:, 1]
    return pd.DataFrame(
        {
            "sentence_id": eval_set["sentence_id"].to_numpy(),
            "prediction": (prob_hate >= 0.5).astype(int),
            "prob_hate": prob_hate,
        }
    )


def write_test_results(
    baseline: Pipeline, test_set: pd.DataFrame, run_path: str
) -> None:
    prob_hate = baseline.predict_proba(test_set["text"])[:, 1]
    pred_labels = (prob_hate >= 0.5).astype(int)
    pd.DataFrame(
        {
            "gahd_id": test_set["gahd_id"].to_numpy(),
            "text": test_set["text"].to_numpy(),
            "label": test_set["label"].to_numpy(),
            "prob_hate": prob_hate,
            "pred": pred_labels,
        }
    ).to_csv(os.path.join(run_path, "gahd_test_predictions.csv"), index=False)
    metrics = compute_binary_metrics(test_set["label"].to_numpy(), pred_labels)
    with open(os.path.join(run_path, "gahd_test_metrics.json"), "w") as fout:
        json.dump(metrics, fout, ensure_ascii=False, indent=4)
    train_logger.info(f"GAHD test metrics: {json.dumps(metrics, indent=4)}")


def check_args(args: argparse.Namespace) -> None:
    if len(args.path_eval_sets) != len(args.input_columns):
        raise ValueError(
            f"Got {len(args.path_eval_sets)} --path_eval_sets but "
            f"{len(args.input_columns)} --input_columns."
        )


def main(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)

    check_args(args)
    run_path = get_run_path(args.path_runs_dir, args.experiment_name, args.seed)
    check_empty_output_dir(run_path)
    train_logger.info(f"CMD args: {json.dumps(args.__dict__, indent=4)}")
    eval_sets = {
        input_column: load_eval_set(path_eval_set, input_column)
        for path_eval_set, input_column in zip(args.path_eval_sets, args.input_columns)
    }

    train_logger.info("Load GAHD.")
    train_set, _, test_set = load_gahd()
    for name, split in [("train", train_set), ("test", test_set)]:
        check_columns(split, GAHD_COLUMNS, f"GAHD {name} split")

    train_logger.info("Start training.")
    baseline = make_baseline().fit(train_set["text"], train_set["label"])
    train_logger.info("Finished training.")

    write_test_results(baseline, test_set, run_path)
    for input_column, eval_set in eval_sets.items():
        path_predictions = get_predictions_path(run_path, input_column)
        predict_eval_set(baseline, eval_set, input_column).to_csv(
            path_predictions, index=False
        )
        train_logger.info(f"Wrote {len(eval_set)} predictions to {path_predictions}.")


def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    # data
    parser.add_argument(
        "--path_eval_sets",
        nargs="+",
        default=[
            "data/eval_sentences.csv",
            "data/pred_input_haiku.csv",
            "data/pred_input_apertus8b.csv",
        ],
        help="Swiss German test set CSVs, one per input column.",
    )
    parser.add_argument(
        "--input_columns",
        nargs="+",
        choices=INPUT_COLUMNS,
        default=INPUT_COLUMNS,
        help="Column of each evaluation set to classify, in the same order.",
    )
    # saving
    parser.add_argument(
        "--experiment_name",
        default="gahd_tfidf_logreg",
        help="Name of the experiment directory under --path_runs_dir.",
    )
    parser.add_argument(
        "--path_runs_dir", default="runs", help="Directory that holds all runs."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser


if __name__ == "__main__":
    cmd_args = get_argparser().parse_args()
    main(cmd_args)
