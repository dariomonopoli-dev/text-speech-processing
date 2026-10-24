"""Load the stratified annotation files and resolve them into gold labels."""

from pathlib import Path

import pandas as pd

ANNOTATORS = ["dario", "marjorie", "yannis"]
ANNOTATION_COLUMNS = ["sentence_id", "label"]


def get_path_annotation_file(path_annotations: Path, annotator: str) -> Path:
    """Return the stratified annotation file of one annotator."""
    return Path(path_annotations) / f"annotation_{annotator}_stratified.csv"


def load_annotations(path_annotations: Path) -> dict[str, pd.Series]:
    """Load each annotator's labels indexed by sentence_id."""
    annotations = {}
    for annotator in ANNOTATORS:
        path_annotation_file = get_path_annotation_file(path_annotations, annotator)
        if not path_annotation_file.is_file():
            raise FileNotFoundError(
                f"annotation file not found: {path_annotation_file}"
            )
        annotation_df = pd.read_csv(path_annotation_file, encoding="utf-8-sig")
        missing_columns = [
            column
            for column in ANNOTATION_COLUMNS
            if column not in annotation_df.columns
        ]
        if missing_columns:
            raise KeyError(f"{path_annotation_file} lacks columns {missing_columns}")
        annotations[annotator] = annotation_df.set_index("sentence_id")["label"]
    return annotations


def get_shared_labels(annotations: dict[str, pd.Series]) -> pd.DataFrame:
    """Return one column per annotator for the sentences all of them labelled."""
    return pd.concat(annotations, axis=1, join="inner")


def load_gold_labels(path_annotations: Path) -> pd.DataFrame:
    """Majority label on shared sentences, the single label everywhere else."""
    annotations = load_annotations(path_annotations)
    shared_labels = get_shared_labels(annotations)
    single_annotator_labels = pd.concat(
        [labels.drop(shared_labels.index) for labels in annotations.values()]
    )
    majority_labels = (shared_labels.sum(axis=1) >= 2).astype(int)
    true_labels = pd.concat([majority_labels, single_annotator_labels])
    return pd.DataFrame(
        {"label": true_labels, "is_shared": true_labels.index.isin(shared_labels.index)}
    ).rename_axis("sentence_id")
