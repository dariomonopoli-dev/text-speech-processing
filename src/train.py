"""Fine-tune GBERT or SwissBERT on GAHD and write test predictions and metrics."""

import argparse
import hashlib
import json
import os
import platform
import random
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertForSequenceClassification,
    BertTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.trainer_utils import get_last_checkpoint

import delete_checkpoints
from classification_loss import compute_classification_loss
from config import ID2LABEL, LABEL2ID, LABEL_LIST, get_run_name, get_run_path
from dataset import load_gahd
from get_loggers import get_logger

SAVE_TOTAL_LIMIT = 3
GAHD_COLUMNS = ["gahd_id", "text", "label"]

train_logger = get_logger("train")


def compute_binary_metrics(
    true_labels: np.ndarray, pred_labels: np.ndarray
) -> dict[str, float]:
    """Return the flat metric dict with label 1 (hate) as the positive class."""
    f1_per_label = f1_score(
        true_labels, pred_labels, labels=LABEL_LIST, average=None, zero_division=0
    )
    num_examples = len(true_labels)
    num_true_hate = int(np.sum(true_labels == 1))
    num_pred_hate = int(np.sum(pred_labels == 1))
    return {
        "acc": float(accuracy_score(true_labels, pred_labels)),
        "f1-macro": float(
            f1_score(
                true_labels,
                pred_labels,
                labels=LABEL_LIST,
                average="macro",
                zero_division=0,
            )
        ),
        "f1-pos": float(f1_per_label[1]),
        "f1-neg": float(f1_per_label[0]),
        "precision": float(
            precision_score(true_labels, pred_labels, pos_label=1, zero_division=0)
        ),
        "recall": float(
            recall_score(true_labels, pred_labels, pos_label=1, zero_division=0)
        ),
        "num_examples": num_examples,
        "num_true_hate": num_true_hate,
        "num_pred_hate": num_pred_hate,
        "true_hs_ratio": num_true_hate / num_examples,
        "pred_hs_ratio": num_pred_hate / num_examples,
    }


def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    logits, true_labels = eval_pred
    return compute_binary_metrics(true_labels, logits.argmax(-1))


def to_prob_hate(logits: np.ndarray) -> np.ndarray:
    return torch.softmax(torch.from_numpy(logits), dim=-1)[:, 1].numpy()


def encode_dataset(
    df: pd.DataFrame,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int | None,
    text_column: str = "text",
) -> Dataset:
    """Tokenize text_column with truncation; padding is left to the data collator."""
    return Dataset.from_pandas(df).map(
        lambda batch: tokenizer(
            batch[text_column], truncation=True, max_length=max_length
        ),
        batched=True,
    )


def check_args(args: argparse.Namespace) -> None:
    for name in [
        "epochs",
        "batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "patience",
        "log_interval",
    ]:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name} must be positive, got {getattr(args, name)}.")
    if args.max_length is not None and args.max_length <= 0:
        raise ValueError(f"--max_length must be positive, got {args.max_length}.")


def check_empty_output_dir(path_out_dir: str) -> None:
    """Create the output directory, or refuse if it already has content."""
    os.makedirs(path_out_dir, exist_ok=True)
    if os.listdir(path_out_dir):
        raise FileExistsError(f"Output directory '{path_out_dir}' is not empty.")


def get_resume_checkpoint(args: argparse.Namespace, run_path: str) -> str | None:
    """Resume only an unfinished run with matching configuration and full state."""
    if not args.resume:
        check_empty_output_dir(run_path)
        return None
    path_run = Path(run_path).resolve()
    if (path_run / "gahd_test_metrics.json").exists():
        raise FileExistsError(f"Training is already complete in {run_path}")
    previous = json.loads((path_run / "experiment.json").read_text(encoding="utf-8"))
    differences = [
        key
        for key, value in vars(args).items()
        if key != "resume" and previous.get(key) != value
    ]
    if differences:
        raise ValueError(f"Resume configuration differs for {differences}")
    if (
        previous.get("loss_normalization")
        != "sum_cross_entropy_over_accumulated_examples"
    ):
        raise ValueError("Cannot resume a run with different loss normalization")
    if args.path_gahd is not None:
        current_hash = hashlib.sha256(Path(args.path_gahd).read_bytes()).hexdigest()
        if previous.get("gahd_sha256") != current_hash:
            raise ValueError("GAHD data changed since the interrupted run")
    checkpoint = get_last_checkpoint(str(path_run))
    if checkpoint is None:
        raise FileNotFoundError(f"No resumable checkpoint in {run_path}")
    path_checkpoint = Path(checkpoint).resolve()
    if path_checkpoint.parent != path_run:
        raise ValueError(f"Checkpoint resolves outside run: {path_checkpoint}")
    for name in [
        "model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "trainer_state.json",
    ]:
        if not (path_checkpoint / name).is_file():
            raise FileNotFoundError(
                f"Resume checkpoint lacks {name}: {path_checkpoint}"
            )
    return str(path_checkpoint)


def check_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing_columns = [column for column in columns if column not in df.columns]
    if missing_columns:
        raise KeyError(f"{name} is missing the columns {missing_columns}.")


def write_test_results(
    test_set: pd.DataFrame, logits: np.ndarray, path_out_dir: str
) -> None:
    pred_labels = logits.argmax(axis=-1)
    pd.DataFrame(
        {
            "gahd_id": test_set["gahd_id"].to_numpy(),
            "text": test_set["text"].to_numpy(),
            "label": test_set["label"].to_numpy(),
            "prob_hate": to_prob_hate(logits),
            "pred": pred_labels,
        }
    ).to_csv(os.path.join(path_out_dir, "gahd_test_predictions.csv"), index=False)
    metrics = compute_binary_metrics(test_set["label"].to_numpy(), pred_labels)
    with open(os.path.join(path_out_dir, "gahd_test_metrics.json"), "w") as fout:
        json.dump(metrics, fout, ensure_ascii=False, indent=4)
    train_logger.info(f"GAHD test metrics: {json.dumps(metrics, indent=4)}")


def load_model(model_name: str, model_family: str) -> PreTrainedModel:
    """Load the classifier and apply the SwissBERT training adapter policy."""
    model_class = (
        AutoModelForSequenceClassification
        if model_family == "swissbert"
        else BertForSequenceClassification
    )
    model = model_class.from_pretrained(
        model_name,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    if model_family == "swissbert":
        if model.config.model_type != "xmod" or "gsw" not in model.config.languages:
            raise ValueError(
                "SwissBERT requires an X-MOD checkpoint with a gsw adapter"
            )
        model.set_default_language("de_CH")
        model.freeze_embeddings_and_language_adapters()
    return model


def main(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    check_args(args)
    run_path = get_run_path(args.path_runs_dir, args.experiment_name, args.seed)
    resume_checkpoint = get_resume_checkpoint(args, run_path)
    train_logger.info(f"CMD args: {json.dumps(args.__dict__, indent=4)}")

    train_logger.info("Load GAHD.")
    train_set, dev_set, test_set = load_gahd(args.path_gahd)
    for name, split in [("train", train_set), ("dev", dev_set), ("test", test_set)]:
        check_columns(split, GAHD_COLUMNS, f"GAHD {name} split")
    train_logger.info(
        f"Loaded {len(train_set)} train, {len(dev_set)} dev, {len(test_set)} test examples."
    )

    train_logger.info(f"Load tokenizer: {args.model_name}")
    tokenizer_class = (
        AutoTokenizer if args.model_family == "swissbert" else BertTokenizer
    )
    tokenizer = tokenizer_class.from_pretrained(args.model_name)
    enc_train_set = encode_dataset(
        train_set[["text", "label"]], tokenizer, args.max_length
    )
    enc_dev_set = encode_dataset(dev_set[["text", "label"]], tokenizer, args.max_length)
    enc_test_set = encode_dataset(
        test_set[["text", "label"]], tokenizer, args.max_length
    )

    train_logger.info(f"Load model: {args.model_name}")
    model = load_model(args.model_name, args.model_family)
    optimizer_args = (
        {"optim": "adamw_torch"} if args.model_family == "swissbert" else {}
    )
    training_args = TrainingArguments(
        output_dir=run_path,
        run_name=get_run_name(args.experiment_name, args.seed),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=delete_checkpoints.BEST_METRIC,
        save_total_limit=SAVE_TOTAL_LIMIT,
        logging_steps=args.log_interval,
        seed=args.seed,
        report_to="tensorboard",
        fp16=not args.no_fp16,
        **optimizer_args,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=enc_train_set,
        eval_dataset=enc_dev_set,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        compute_loss_func=compute_classification_loss,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )
    experiment = {
        **vars(args),
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "loss_normalization": "sum_cross_entropy_over_accumulated_examples",
        "resumed_from_checkpoint": resume_checkpoint,
        "optimizer": str(training_args.optim),
        "max_grad_norm": training_args.max_grad_norm,
        "training_adapter": getattr(model.config, "default_language", None),
        "embeddings_and_language_adapters_frozen": args.model_family == "swissbert",
        "python_version": platform.python_version(),
        "package_versions": {
            name: version(name)
            for name in ["torch", "transformers", "datasets", "pandas", "scikit-learn"]
        },
        "device": str(training_args.device),
        "hardware": (
            torch.cuda.get_device_name(training_args.device)
            if training_args.device.type == "cuda"
            else platform.processor()
        ),
        "num_train": len(train_set),
        "num_dev": len(dev_set),
        "num_test": len(test_set),
        "tokenizer_max_length": tokenizer.model_max_length,
        "gahd_sha256": (
            hashlib.sha256(Path(args.path_gahd).read_bytes()).hexdigest()
            if args.path_gahd is not None
            else None
        ),
    }
    path_experiment = Path(run_path) / "experiment.json"
    path_experiment.write_text(json.dumps(experiment, indent=4), encoding="utf-8")
    train_logger.info("Start training.")
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_state()
    experiment.update(
        best_model_checkpoint=trainer.state.best_model_checkpoint,
        best_dev_macro_f1=trainer.state.best_metric,
        train_runtime=train_result.metrics["train_runtime"],
    )
    path_experiment.write_text(json.dumps(experiment, indent=4), encoding="utf-8")
    train_logger.info("Finished training.")

    train_logger.info("Predict on the GAHD test split with the best checkpoint.")
    test_output = trainer.predict(enc_test_set, metric_key_prefix="test")
    write_test_results(test_set, test_output.predictions, run_path)
    delete_checkpoints.main(argparse.Namespace(path_run_dir=run_path))


def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    # main
    parser.add_argument(
        "--experiment_name",
        required=True,
        help="Experiment stem encoding every factor, e.g. gahd_gbert_raw.",
    )
    parser.add_argument(
        "-m",
        "--model_name",
        default="deepset/gbert-base",
        help="Hugging Face name of the model and tokenizer to fine-tune.",
    )
    parser.add_argument(
        "--model_family",
        choices=["bert", "swissbert"],
        default="bert",
        help="SwissBERT uses de_CH and freezes embeddings and language adapters.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the latest full checkpoint of an unfinished matching run.",
    )
    parser.add_argument(
        "--path_gahd",
        default=None,
        help="Local GAHD CSV. Default: download from the Hugging Face hub.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Microbatches per optimizer step; effective batch size is their product.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Truncation length in tokens. Default: the tokenizer's model_max_length.",
    )
    # hyperparams
    parser.add_argument(
        "-E", "--epochs", type=float, default=3.0, help="Number of training epochs."
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=32,
        help="Per-device batch size for training and evaluation.",
    )
    parser.add_argument(
        "-l", "--learning_rate", type=float, default=2e-5, help="Learning rate."
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="AdamW weight decay."
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Early stopping patience in evaluations without dev macro F1 gain.",
    )
    parser.add_argument(
        "--no_fp16", action="store_true", help="Train in fp32 instead of fp16."
    )
    # reporting
    parser.add_argument(
        "-i",
        "--log_interval",
        type=int,
        default=50,
        help="Number of steps between training loss logs.",
    )
    # saving
    parser.add_argument(
        "--path_runs_dir",
        default="runs",
        help="Root directory; the run is written to <root>/<experiment_name>/run<seed>.",
    )
    # reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for torch, numpy, random and the Trainer; also the run number.",
    )
    return parser


if __name__ == "__main__":
    cmd_args = get_argparser().parse_args()
    try:
        main(cmd_args)
    except Exception:
        train_logger.exception(f"Training failed for {cmd_args.experiment_name}.")
        raise
