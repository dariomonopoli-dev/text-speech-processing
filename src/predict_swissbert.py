"""Predict raw Swiss German text with a locally saved SwissBERT checkpoint."""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import pandas as pd
import torch
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from get_loggers import get_logger

logger = get_logger("pred_swissbert")


def load_eval_set(path_eval_set: Path) -> pd.DataFrame:
    """Load nonempty, uniquely identified sentences without changing their order."""
    eval_set = pd.read_csv(path_eval_set, dtype={"sentence_id": "string"})
    missing_columns = sorted({"sentence_id", "text"} - set(eval_set.columns))
    if missing_columns:
        raise KeyError(f"{path_eval_set} lacks columns {missing_columns}")
    if eval_set.empty:
        raise ValueError(f"{path_eval_set} has no sentences")
    if eval_set["sentence_id"].isna().any() or not eval_set["sentence_id"].is_unique:
        raise ValueError(f"{path_eval_set} has missing or duplicate sentence_id values")
    if eval_set["text"].fillna("").str.strip().eq("").any():
        raise ValueError(f"{path_eval_set} has missing or blank text")
    return eval_set[["sentence_id", "text"]]


def predict_batches(
    model: torch.nn.Module,
    tokenizer: object,
    texts: list[str],
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Return two-class probabilities using FP32 and inference mode."""
    model.to(device=device, dtype=torch.float32).eval()
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            inputs = tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            logits = model(**inputs).logits
            probabilities.append(logits.softmax(dim=-1).cpu())
    return torch.cat(probabilities)


def main(args: argparse.Namespace) -> None:
    logger.info(json.dumps(vars(args), default=str))
    if args.batch_size <= 0 or args.max_length <= 0:
        raise ValueError("batch_size and max_length must be positive")
    path_metadata = args.path_output.with_suffix(".metadata.json")
    for path_output in [args.path_output, path_metadata]:
        if path_output.exists():
            raise FileExistsError(f"Output {path_output} already exists")
    eval_set = load_eval_set(args.path_eval_set)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.path_checkpoint, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.path_checkpoint, local_files_only=True
    )
    if model.config.model_type != "xmod" or "gsw" not in model.config.languages:
        raise ValueError(f"{args.path_checkpoint} is not an X-MOD model with gsw")
    if model.config.label2id != {"not_hate": 0, "hate": 1}:
        raise ValueError(f"Unexpected label mapping: {model.config.label2id}")
    model.set_default_language("gsw")
    logger.info(f"Start prediction: {len(eval_set)} sentences, gsw adapter, {device}.")
    probabilities = predict_batches(
        model,
        tokenizer,
        eval_set["text"].tolist(),
        args.batch_size,
        args.max_length,
        device,
    )
    if probabilities.shape != (len(eval_set), 2) or not torch.isfinite(
        probabilities
    ).all():
        raise ValueError("Prediction probabilities are invalid")
    predictions = eval_set[["sentence_id"]].copy()
    predictions["prediction"] = probabilities.argmax(dim=-1).numpy()
    predictions["prob_hate"] = probabilities[:, 1].numpy()
    metadata = {
        **vars(args),
        "num_examples": len(eval_set),
        "language": model.config.default_language,
        "precision": "float32",
        "device": str(device),
        "hardware": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else platform.processor()
        ),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "input_sha256": hashlib.sha256(args.path_eval_set.read_bytes()).hexdigest(),
    }
    path_state = args.path_checkpoint.parent / "trainer_state.json"
    if path_state.is_file():
        state = json.loads(path_state.read_text(encoding="utf-8"))
        metadata["best_model_checkpoint"] = state["best_model_checkpoint"]
        metadata["best_dev_metric"] = state["best_metric"]
    args.path_output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.path_output, index=False)
    path_metadata.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=4, default=str),
        encoding="utf-8",
    )
    logger.info(f"Finished prediction. Wrote {args.path_output} and {path_metadata}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path_checkpoint",
        type=Path,
        required=True,
        help="Local best_model directory.",
    )
    parser.add_argument(
        "--path_eval_set",
        type=Path,
        default=Path("data/eval_sentences.csv"),
        help="CSV with sentence_id and raw Swiss German text.",
    )
    parser.add_argument(
        "--path_output", type=Path, required=True, help="Prediction CSV to create."
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Number of sentences per batch."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Truncation length used in training.",
    )
    cmd_args = parser.parse_args()
    main(cmd_args)
