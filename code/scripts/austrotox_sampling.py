"""Score German text and apply an AustroTox-inspired sampling protocol."""

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


GAHD_MODEL = "jagoldz/gahd"
GAHD_REVISION = "90c6e4f50d109c42bebdeb0f43fb5c3e619612fc"
GAHD_HATE_LABELS = frozenset({"LABEL_1", "1", "HATE", "HATE SPEECH"})
LOWER_STRATA = ("0.0-0.3", "0.3-0.5", "0.5-0.7", "0.7-0.9")
ALL_STRATA = LOWER_STRATA + ("0.9-1.0",)
FIXED_STRATUM_TARGETS = {
    "0.0-0.3": 134,
    "0.3-0.5": 134,
    "0.5-0.7": 134,
    "0.7-0.9": 164,
    "0.9-1.0": 134,
}
AUDIT_COLUMNS = (
    "missing",
    "meta_commentary",
    "multiple_paragraphs",
    "length_ratio",
    "extreme_length",
    "has_anomaly",
)
META_COMMENTARY_PATTERN = re.compile(
    r"(?:^\s*(?:hinweis(?:\s+zur\s+übersetzung)?|übersetzung|translation|note)\s*:|"
    r"\b(?:let me correct|bitte prüfen|ich habe .* übersetzt)\b)",
    flags=re.IGNORECASE | re.DOTALL,
)


def audit_translations(
    source_texts: Iterable[str],
    translations: Iterable[str | None],
    min_length_ratio: float = 0.25,
    max_length_ratio: float = 3.0,
) -> pd.DataFrame:
    """Flag malformed translations without filtering words or social groups.

    These checks identify missing output, model commentary, multiple paragraphs,
    and extreme source-to-translation length ratios. They deliberately contain
    no lexical blocklist for identity terms, profanity, or offensive language.
    """
    sources = list(source_texts)
    outputs = list(translations)
    if len(sources) != len(outputs):
        raise ValueError("source_texts and translations must have the same length.")
    if not 0 < min_length_ratio < max_length_ratio:
        raise ValueError("Length ratio bounds must satisfy 0 < min < max.")
    if not all(isinstance(source, str) and source.strip() for source in sources):
        raise ValueError("Every source text must be a non-empty string.")

    records = []
    for source, output in zip(sources, outputs):
        missing = output is None or pd.isna(output) or not str(output).strip()
        normalized_output = "" if missing else str(output).strip()
        length_ratio = (
            np.nan if missing else len(normalized_output) / len(source.strip())
        )
        meta_commentary = not missing and bool(
            META_COMMENTARY_PATTERN.search(normalized_output)
        )
        multiple_paragraphs = not missing and bool(
            re.search(r"\n\s*\n", normalized_output)
        )
        extreme_length = (
            not missing and not min_length_ratio <= length_ratio <= max_length_ratio
        )
        records.append(
            {
                "missing": missing,
                "meta_commentary": meta_commentary,
                "multiple_paragraphs": multiple_paragraphs,
                "length_ratio": length_ratio,
                "extreme_length": extreme_length,
                "has_anomaly": missing
                or meta_commentary
                or multiple_paragraphs
                or extreme_length,
            }
        )
    return pd.DataFrame.from_records(records, columns=AUDIT_COLUMNS)


def assign_austrotox_strata(scores: Iterable[float]) -> pd.Series:
    """Assign the five toxicity bands described in the AustroTox paper.

    Adjacent intervals are left-closed and right-open. The last interval also
    contains 1.0, so every valid classifier probability belongs to one band.
    """
    values = pd.Series(scores, dtype=float)
    if values.isna().any() or not values.between(0, 1).all():
        raise ValueError("Toxicity scores must all be between 0 and 1.")

    labels = np.select(
        [
            values < 0.3,
            values < 0.5,
            values < 0.7,
            values < 0.9,
        ],
        LOWER_STRATA,
        default="0.9-1.0",
    )
    return pd.Series(
        pd.Categorical(labels, categories=ALL_STRATA, ordered=True),
        index=values.index,
        name="austrotox_stratum",
    )


def sample_fixed_strata(
    corpus: pd.DataFrame,
    score_column: str = "hate_score",
    random_state: int = 42,
) -> pd.DataFrame:
    """Sample exactly 700 rows using the scaled AustroTox stratum allocation."""
    if score_column not in corpus:
        raise KeyError(f"Missing score column: {score_column}")
    if "sentence_id" not in corpus:
        raise KeyError("Missing identity column: sentence_id")
    if corpus["sentence_id"].isna().any() or corpus["sentence_id"].duplicated().any():
        raise ValueError("sentence_id values must be unique and non-null.")

    candidates = corpus.copy()
    candidates["austrotox_stratum"] = assign_austrotox_strata(
        candidates[score_column]
    ).to_numpy()
    available_counts = candidates["austrotox_stratum"].value_counts(sort=False)
    shortages = [
        f"{stratum}: available={int(available_counts.get(stratum, 0))}, required={target}"
        for stratum, target in FIXED_STRATUM_TARGETS.items()
        if int(available_counts.get(stratum, 0)) < target
    ]
    if shortages:
        raise ValueError("Undersized strata: " + "; ".join(shortages))

    parts = [
        candidates[candidates["austrotox_stratum"] == stratum].sample(
            n=target,
            random_state=random_state,
        )
        for stratum, target in FIXED_STRATUM_TARGETS.items()
    ]
    return (
        pd.concat(parts, ignore_index=True)
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )


def sample_like_austrotox(
    corpus: pd.DataFrame,
    total: int = 700,
    score_column: str = "hate_score",
    random_state: int = 42,
) -> pd.DataFrame:
    """Compatibility wrapper for the fixed AustroTox-adapted allocation.

    The revised study protocol has a fixed total of 700. Callers requesting a
    different total must choose and document a different allocation explicitly.
    """
    fixed_total = sum(FIXED_STRATUM_TARGETS.values())
    if total != fixed_total:
        raise ValueError(
            f"The fixed AustroTox-adapted allocation requires total={fixed_total}."
        )
    return sample_fixed_strata(
        corpus,
        score_column=score_column,
        random_state=random_state,
    )


def score_with_gahd(
    texts: Iterable[str],
    classifier: Callable[..., Sequence[Sequence[Mapping[str, Any]]]] | None = None,
    batch_size: int = 32,
) -> pd.Series:
    """Return the official GAHD model's probability for hate speech."""
    text_list = list(texts)
    if not text_list:
        return pd.Series(dtype=float, name="hate_score")
    if not all(isinstance(text, str) and text.strip() for text in text_list):
        raise ValueError("Every text must be a non-empty string.")

    if classifier is None:
        from transformers import pipeline

        classifier = pipeline(
            "text-classification",
            model=GAHD_MODEL,
            revision=GAHD_REVISION,
        )

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

    probabilities = []
    for prediction in predictions:
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
            raise ValueError(
                "The GAHD hate label must have a numeric score."
            ) from error
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("The GAHD hate probability must be between 0 and 1.")
        probabilities.append(probability)
    return pd.Series(probabilities, name="hate_score")
