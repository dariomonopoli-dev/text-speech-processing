"""AustroTox-inspired stratified sampling of the 700-row master sample."""

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from scoring import GAHD_MODEL, GAHD_REVISION

LOWER_STRATA = ("0.0-0.3", "0.3-0.5", "0.5-0.7", "0.7-0.9")
ALL_STRATA = LOWER_STRATA + ("0.9-1.0",)
FIXED_STRATUM_TARGETS = {
    "0.0-0.3": 134,
    "0.3-0.5": 134,
    "0.5-0.7": 134,
    "0.7-0.9": 164,
    "0.9-1.0": 134,
}


def map_scores_to_strata(scores: Iterable[float]) -> pd.Series:
    """Map scores to the five AustroTox bands, left-closed, with 1.0 in the last."""
    values = pd.Series(scores, dtype=float)
    if values.isna().any() or not values.between(0, 1).all():
        raise ValueError("Toxicity scores must all be between 0 and 1.")
    strata = np.select(
        [values < 0.3, values < 0.5, values < 0.7, values < 0.9],
        LOWER_STRATA,
        default="0.9-1.0",
    )
    return pd.Series(
        pd.Categorical(strata, categories=ALL_STRATA, ordered=True),
        index=values.index,
        name="austrotox_stratum",
    )


def check_strata_match_scores(frame: pd.DataFrame) -> None:
    """Require every stored stratum to agree with its GAHD probability."""
    missing = sorted({"hate_score", "austrotox_stratum"}.difference(frame.columns))
    if missing:
        raise KeyError("Missing score or stratum columns: " + ", ".join(missing))
    hate_scores = pd.to_numeric(frame["hate_score"], errors="coerce")
    if hate_scores.isna().any() or not hate_scores.between(0.0, 1.0).all():
        raise ValueError("hate_score values must be probabilities in [0, 1].")
    expected_strata = map_scores_to_strata(hate_scores).astype("string")
    stored_strata = frame["austrotox_stratum"].astype("string")
    if not stored_strata.reset_index(drop=True).equals(
        expected_strata.reset_index(drop=True)
    ):
        raise ValueError("austrotox_stratum does not match hate_score for every row.")


def sample_fixed_strata(
    candidates: pd.DataFrame, seed: int = 42, score_column: str = "hate_score"
) -> pd.DataFrame:
    """Sample exactly 700 rows using the scaled AustroTox stratum allocation."""
    if score_column not in candidates:
        raise KeyError(f"Missing score column: {score_column}")
    if "sentence_id" not in candidates:
        raise KeyError("Missing identity column: sentence_id")
    sentence_ids = candidates["sentence_id"]
    if sentence_ids.isna().any() or sentence_ids.duplicated().any():
        raise ValueError("sentence_id values must be unique and non-null.")

    stratified = candidates.copy()
    stratified["austrotox_stratum"] = map_scores_to_strata(
        stratified[score_column]
    ).to_numpy()
    available_counts = stratified["austrotox_stratum"].value_counts(sort=False)
    shortages = [
        f"{stratum}: available={int(available_counts.get(stratum, 0))}, required={target}"
        for stratum, target in FIXED_STRATUM_TARGETS.items()
        if int(available_counts.get(stratum, 0)) < target
    ]
    if shortages:
        raise ValueError("Undersized strata: " + "; ".join(shortages))

    samples_per_stratum = [
        stratified[stratified["austrotox_stratum"] == stratum].sample(
            n=target, random_state=seed
        )
        for stratum, target in FIXED_STRATUM_TARGETS.items()
    ]
    return (
        pd.concat(samples_per_stratum, ignore_index=True)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )


def count_per_stratum(strata: pd.Series) -> dict[str, int]:
    counts = strata.value_counts(sort=False)
    return {stratum: int(counts.get(stratum, 0)) for stratum in FIXED_STRATUM_TARGETS}


def compute_sampling_report(
    scored_candidates: pd.DataFrame,
    source_strata: pd.Series,
    master: pd.DataFrame,
    seed: int,
) -> dict[str, Any]:
    """Summarise source and selected counts and score ranges per stratum."""
    score_summaries = {}
    for stratum in FIXED_STRATUM_TARGETS:
        hate_scores = scored_candidates.loc[source_strata == stratum, "hate_score"]
        score_summaries[stratum] = {
            "min": float(hate_scores.min()),
            "median": float(hate_scores.median()),
            "max": float(hate_scores.max()),
        }
    return {
        "method": "AustroTox-inspired project-specific fixed allocation",
        "sample_size": int(len(master)),
        "random_seed": int(seed),
        "score_model": GAHD_MODEL,
        "score_revision": GAHD_REVISION,
        "source_counts": count_per_stratum(source_strata),
        "target_counts": dict(FIXED_STRATUM_TARGETS),
        "selected_counts": count_per_stratum(master["austrotox_stratum"]),
        "score_summaries": score_summaries,
    }


def sample_master(
    scored_candidates: pd.DataFrame, seed: int = 42
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select the fixed 700-row sample and return its diagnostic report."""
    if "hate_score" not in scored_candidates:
        raise KeyError("Missing score column: hate_score")
    source_strata = map_scores_to_strata(scored_candidates["hate_score"])
    master = sample_fixed_strata(scored_candidates, seed=seed)
    report = compute_sampling_report(scored_candidates, source_strata, master, seed)
    return master, report
