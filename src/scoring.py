"""GAHD hate-speech scores for the SwissCrawl candidate sentences."""

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

GAHD_MODEL = "jagoldz/gahd"
GAHD_REVISION = "90c6e4f50d109c42bebdeb0f43fb5c3e619612fc"
GAHD_HATE_LABELS = frozenset({"LABEL_1", "1", "HATE", "HATE SPEECH"})

Classifier = Callable[..., Sequence[Sequence[Mapping[str, Any]]]]


def load_gahd_classifier(device: str | None = None) -> Classifier:
    """Load the pinned official GAHD classifier on MPS when available."""
    from transformers import pipeline

    if device is None:
        import torch

        device = "mps" if torch.backends.mps.is_available() else "cpu"
    return pipeline(
        "text-classification",
        model=GAHD_MODEL,
        revision=GAHD_REVISION,
        device=device,
    )


def is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def extract_hate_probability(prediction: Any) -> float:
    if not isinstance(prediction, Sequence) or isinstance(prediction, (str, bytes)):
        raise ValueError(
            "Each classifier prediction must be a sequence of label-score records."
        )
    hate_items = [
        item
        for item in prediction
        if isinstance(item, Mapping)
        and str(item.get("label", "")).strip().upper() in GAHD_HATE_LABELS
    ]
    if len(hate_items) != 1:
        raise ValueError(
            "Classifier output must contain the GAHD hate label exactly once."
        )
    try:
        probability = float(hate_items[0]["score"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("The GAHD hate label must have a numeric score.") from error
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("The GAHD hate probability must be between 0 and 1.")
    return probability


def compute_hate_scores(
    texts: Iterable[str], classifier: Classifier, batch_size: int = 32
) -> pd.Series:
    """Return the GAHD probability of the hate label for every text."""
    text_list = list(texts)
    if not text_list:
        return pd.Series(dtype=float, name="hate_score")
    if not all(is_non_empty_string(text) for text in text_list):
        raise ValueError("Every text must be a non-empty string.")
    predictions = classifier(
        text_list,
        batch_size=batch_size,
        truncation=True,
        top_k=None,
    )
    if len(predictions) != len(text_list):
        raise ValueError(
            "Classifier must return exactly one prediction per input text."
        )
    probabilities = [extract_hate_probability(prediction) for prediction in predictions]
    return pd.Series(probabilities, name="hate_score")


def add_hate_scores(
    candidates: pd.DataFrame, classifier: Classifier, batch_size: int
) -> pd.DataFrame:
    """Score every candidate in bounded inference batches."""
    missing_columns = sorted({"sentence_id", "text"}.difference(candidates.columns))
    if missing_columns:
        raise KeyError("Missing candidate columns: " + ", ".join(missing_columns))
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    sentence_ids = candidates["sentence_id"]
    if sentence_ids.isna().any() or sentence_ids.duplicated().any():
        raise ValueError("sentence_id values must be unique and non-null.")
    if not sentence_ids.map(is_non_empty_string).all():
        raise ValueError("Every sentence_id must be a non-empty string.")
    if not candidates["text"].map(is_non_empty_string).all():
        raise ValueError("Every text must be a non-empty string.")

    hate_scores: list[float] = []
    for start in range(0, len(candidates), batch_size):
        batch = candidates.iloc[start : start + batch_size]
        batch_scores = compute_hate_scores(
            batch["text"].tolist(), classifier=classifier, batch_size=batch_size
        )
        hate_scores.extend(batch_scores.tolist())

    scored_candidates = candidates.copy()
    scored_candidates["hate_score"] = hate_scores
    return scored_candidates
