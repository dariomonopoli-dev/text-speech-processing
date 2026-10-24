"""Split the master sample into shared and single_annotator annotation files."""

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from translation import TERMINAL_TRANSLATION_STATUSES

ANNOTATORS = ("dario", "marjorie", "yannis")
NUM_SHARED = 100
ANNOTATION_COLUMNS = ["sentence_id", "text", "label", "comment"]


def get_annotation_paths(path_annotation_dir: Path) -> dict[str, Path]:
    return {
        annotator: path_annotation_dir / f"annotation_{annotator}_stratified.csv"
        for annotator in ANNOTATORS
    }


def get_workbook_paths(path_annotation_dir: Path) -> dict[str, Path]:
    return {
        annotator: path.with_suffix(".xlsx")
        for annotator, path in get_annotation_paths(path_annotation_dir).items()
    }


def check_ids_unique(frame: pd.DataFrame, message: str) -> None:
    if frame["sentence_id"].isna().any() or frame["sentence_id"].duplicated().any():
        raise ValueError(message)


def check_translations_terminal(master: pd.DataFrame) -> None:
    if (~master["translation_status"].isin(TERMINAL_TRANSLATION_STATUSES)).any():
        raise ValueError("Every master row must have a terminal translation status.")


def assign_annotations(
    master: pd.DataFrame,
    annotators: Sequence[str] = ANNOTATORS,
    num_shared: int = NUM_SHARED,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Give every annotator the shared rows plus an equal single_annotator share."""
    required = {"sentence_id", "text", "translation_status"}
    missing_columns = sorted(required.difference(master.columns))
    if missing_columns:
        raise KeyError("Missing master columns: " + ", ".join(missing_columns))
    names = tuple(annotators)
    if not names or len(set(names)) != len(names):
        raise ValueError("Annotator names must be non-empty and unique.")
    check_ids_unique(master, "master sentence_id values must be unique and non-null.")
    check_translations_terminal(master)
    if not isinstance(num_shared, int) or num_shared < 0:
        raise ValueError("shared_count must be a non-negative integer.")
    num_remaining = len(master) - num_shared
    if num_remaining < 0 or num_remaining % len(names):
        raise ValueError(
            "Rows remaining after the shared overlap must divide evenly among annotators."
        )

    shuffled = (
        master.loc[:, ["sentence_id", "text"]]
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )
    shared = shuffled.iloc[:num_shared]
    remaining = shuffled.iloc[num_shared:]
    num_single_annotator = num_remaining // len(names)
    annotation_sets = {}
    for index, annotator in enumerate(names):
        start = index * num_single_annotator
        single_annotator = remaining.iloc[start : start + num_single_annotator]
        annotation_set = (
            pd.concat([shared, single_annotator], ignore_index=True)
            .sample(frac=1, random_state=seed + index)
            .reset_index(drop=True)
        )
        annotation_set["label"] = ""
        annotation_set["comment"] = ""
        annotation_sets[annotator] = annotation_set[ANNOTATION_COLUMNS]
    return annotation_sets
