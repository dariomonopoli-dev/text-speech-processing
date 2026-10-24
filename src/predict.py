"""Predict hate probabilities for the Swiss German test set with a run's best checkpoint."""

import argparse
import json
import os

import pandas as pd
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from config import get_run_path
from delete_checkpoints import get_checkpoint_path
from get_loggers import get_logger
from train import check_columns, encode_dataset, to_prob_hate

pred_logger = get_logger("pred")


def get_predictions_path(path_run_dir: str, input_column: str) -> str:
    return os.path.join(path_run_dir, f"predictions_swiss_german_{input_column}.csv")


def load_eval_set(path_eval_set: str, input_column: str) -> pd.DataFrame:
    """Load the test set and check that every row has an id and an input text."""
    if not os.path.isfile(path_eval_set):
        raise FileNotFoundError(f"Evaluation set '{path_eval_set}' does not exist.")
    eval_set = pd.read_csv(path_eval_set)
    check_columns(eval_set, ["sentence_id", input_column], path_eval_set)
    num_missing_texts = int(eval_set[input_column].isna().sum())
    if num_missing_texts:
        raise ValueError(
            f"{num_missing_texts} rows of '{path_eval_set}' have no '{input_column}'."
        )
    return eval_set[["sentence_id", input_column]]


def main(args: argparse.Namespace) -> None:
    pred_logger.info(f"CMD args: {json.dumps(args.__dict__, indent=4)}")
    if args.batch_size <= 0:
        raise ValueError(f"--batch_size must be positive, got {args.batch_size}.")
    run_path = get_run_path(args.path_runs_dir, args.experiment_name, args.seed)
    path_predictions = get_predictions_path(run_path, args.input_column)
    if os.path.exists(path_predictions):
        raise FileExistsError(f"Output file '{path_predictions}' exists already.")
    path_checkpoint = get_checkpoint_path(run_path)
    eval_set = load_eval_set(args.path_eval_set, args.input_column)
    pred_logger.info(f"Loaded {len(eval_set)} examples from {args.path_eval_set}.")

    pred_logger.info(f"Load tokenizer {args.model_name} and model {path_checkpoint}.")
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    model = BertForSequenceClassification.from_pretrained(path_checkpoint)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=run_path,
            per_device_eval_batch_size=args.batch_size,
            fp16=not args.no_fp16,
            report_to="none",
        ),
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    enc_eval_set = encode_dataset(
        eval_set[[args.input_column]], tokenizer, args.max_length, args.input_column
    )

    pred_logger.info("Start prediction.")
    logits = trainer.predict(enc_eval_set).predictions
    pred_logger.info("Finished prediction.")
    pd.DataFrame(
        {
            "sentence_id": eval_set["sentence_id"].to_numpy(),
            "prediction": logits.argmax(axis=-1),
            "prob_hate": to_prob_hate(logits),
        }
    ).to_csv(path_predictions, index=False)
    pred_logger.info(f"Wrote predictions to {path_predictions}.")


def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    # main
    parser.add_argument(
        "--experiment_name",
        required=True,
        help="Experiment stem of the trained run, e.g. gahd_gbert_raw.",
    )
    parser.add_argument(
        "-m",
        "--model_name",
        default="deepset/gbert-base",
        help="Hugging Face name of the tokenizer the run was trained with.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Truncation length in tokens. Default: the tokenizer's model_max_length.",
    )
    # data
    parser.add_argument(
        "--path_eval_set",
        default="data/master_sample_stratified.csv",
        help="CSV with a sentence_id column and the input column.",
    )
    parser.add_argument(
        "--input_column",
        choices=["text", "translation"],
        default="text",
        help="text: raw Swiss German (approach 1); "
        "translation: Standard German normalisation (approach 2).",
    )
    # hyperparams
    parser.add_argument(
        "-b", "--batch_size", type=int, default=32, help="Per-device batch size."
    )
    parser.add_argument(
        "--no_fp16", action="store_true", help="Predict in fp32 instead of fp16."
    )
    # saving
    parser.add_argument(
        "--path_runs_dir",
        default="runs",
        help="Root directory; the run is read from <root>/<experiment_name>/run<seed>.",
    )
    # reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed of the trained run, which is also its run number.",
    )
    return parser


if __name__ == "__main__":
    cmd_args = get_argparser().parse_args()
    try:
        main(cmd_args)
    except Exception:
        pred_logger.exception(f"Prediction failed for {cmd_args.experiment_name}.")
        raise
