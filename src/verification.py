"""Checks that the master sample and annotation files form a valid test set."""

from collections.abc import Mapping
from typing import Any

import pandas as pd

from annotation_assignment import (
    ANNOTATION_COLUMNS,
    NUM_SHARED,
    check_ids_unique,
    check_translations_terminal,
)
from sampling import FIXED_STRATUM_TARGETS, check_strata_match_scores, count_per_stratum
from translation import TRANSLATION_COLUMNS

NUM_ANNOTATORS = 3
NUM_ANNOTATION_ROWS = 300
NUM_SINGLE_ANNOTATOR = 200


def check_master(master: pd.DataFrame) -> dict[str, int]:
    """Validate the master sample and return its per-stratum counts."""
    required = {
        "sentence_id",
        "text",
        "hate_score",
        "austrotox_stratum",
        *TRANSLATION_COLUMNS,
    }
    missing = sorted(required.difference(master.columns))
    if missing:
        raise ValueError(
            "master translation columns or required fields are missing: "
            + ", ".join(missing)
        )
    if len(master) != sum(FIXED_STRATUM_TARGETS.values()):
        raise ValueError("master must contain exactly 700 rows.")
    check_ids_unique(master, "master sentence_id values must be unique and non-null.")
    check_translations_terminal(master)
    stratum_counts = count_per_stratum(master["austrotox_stratum"])
    if stratum_counts != FIXED_STRATUM_TARGETS:
        raise ValueError(
            f"master stratum counts do not match targets: {stratum_counts}"
        )
    check_strata_match_scores(master)
    return stratum_counts


def check_annotation_set(
    annotator: str, annotation_set: pd.DataFrame, master: pd.DataFrame
) -> None:
    if annotation_set.columns.tolist() != ANNOTATION_COLUMNS:
        raise ValueError(
            f"{annotator} annotation columns must be exactly {ANNOTATION_COLUMNS}."
        )
    if len(annotation_set) != NUM_ANNOTATION_ROWS:
        raise ValueError(f"{annotator} must have exactly 300 annotation rows.")
    check_ids_unique(
        annotation_set, f"{annotator} annotation sentence_id values must be unique."
    )
    if not set(annotation_set["sentence_id"]).issubset(set(master["sentence_id"])):
        raise ValueError(f"{annotator} contains IDs outside the master sample.")
    master_texts = master.set_index("sentence_id")["text"]
    if not annotation_set["text"].equals(
        annotation_set["sentence_id"].map(master_texts)
    ):
        raise ValueError(
            f"{annotator} does not preserve the original Swiss German text."
        )
    for column in ("label", "comment"):
        if not annotation_set[column].fillna("").eq("").all():
            raise ValueError(f"{annotator} {column} values must be empty.")


def check_assignment_split(
    id_sets: Mapping[str, set], master_ids: set
) -> tuple[set, dict[str, set]]:
    """Return the shared ids and each annotator's single_annotator ids."""
    shared = set.intersection(*id_sets.values())
    if len(shared) != NUM_SHARED:
        raise ValueError(
            f"shared overlap must contain exactly 100 IDs, got {len(shared)}."
        )
    single_annotator = {name: ids - shared for name, ids in id_sets.items()}
    if any(len(ids) != NUM_SINGLE_ANNOTATOR for ids in single_annotator.values()):
        raise ValueError(
            "Every single_annotator assignment must contain exactly 200 IDs."
        )
    names = list(single_annotator)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            if single_annotator[left] & single_annotator[right]:
                raise ValueError(
                    f"single_annotator assignments overlap between {left} and {right}."
                )
    if set.union(*id_sets.values()) != master_ids:
        raise ValueError(
            "Annotation assignments do not cover the complete master sample."
        )
    return shared, single_annotator


def check_testset(
    master: pd.DataFrame, annotation_sets: Mapping[str, pd.DataFrame]
) -> dict[str, Any]:
    """Validate the final sample and leakage-free annotation assignments."""
    stratum_counts = check_master(master)
    if len(annotation_sets) != NUM_ANNOTATORS:
        raise ValueError("Exactly three annotation frames are required.")
    for annotator, annotation_set in annotation_sets.items():
        check_annotation_set(annotator, annotation_set, master)
    id_sets = {
        annotator: set(annotation_set["sentence_id"])
        for annotator, annotation_set in annotation_sets.items()
    }
    shared, single_annotator = check_assignment_split(
        id_sets, set(master["sentence_id"])
    )

    unresolved = master["translation_has_anomaly"].fillna(True).astype(bool) | master[
        "translation_status"
    ].ne("success")
    unresolved_ids = master.loc[unresolved, "sentence_id"].astype(str).tolist()
    status_counts = {
        str(status): int(count)
        for status, count in master["translation_status"]
        .fillna("missing")
        .value_counts()
        .items()
    }
    return {
        "status": "passed",
        "master_rows": int(len(master)),
        "stratum_counts": stratum_counts,
        "annotator_rows": {
            annotator: int(len(annotation_set))
            for annotator, annotation_set in annotation_sets.items()
        },
        "shared_count": int(len(shared)),
        "single_annotator_counts": {
            annotator: int(len(ids)) for annotator, ids in single_annotator.items()
        },
        "translation_status_counts": status_counts,
        "unresolved_translation_anomaly_count": len(unresolved_ids),
        "unresolved_translation_anomaly_ids": unresolved_ids,
    }
