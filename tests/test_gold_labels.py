from pathlib import Path

import pandas as pd
import pytest

from gold_labels import ANNOTATORS, load_annotations, load_gold_labels

PATH_DATA = Path(__file__).resolve().parent.parent / "data"
SHARED_VOTES = {1: [1, 1, 1], 2: [1, 1, 0], 3: [1, 0, 0], 4: [0, 0, 0]}
SINGLE_ANNOTATOR_LABELS = {"dario": {10: 1}, "marjorie": {20: 0}, "yannis": {30: 1}}


@pytest.fixture
def path_annotations(tmp_path: Path) -> Path:
    for position, annotator in enumerate(ANNOTATORS):
        labels = {
            sentence_id: votes[position] for sentence_id, votes in SHARED_VOTES.items()
        }
        labels.update(SINGLE_ANNOTATOR_LABELS[annotator])
        pd.DataFrame(
            {
                "sentence_id": list(labels),
                "text": "satz",
                "label": list(labels.values()),
            }
        ).to_csv(tmp_path / f"annotation_{annotator}_stratified.csv", index=False)
    return tmp_path


def test_shared_sentences_take_the_majority_label(path_annotations: Path) -> None:
    true_labels = load_gold_labels(path_annotations)

    shared = true_labels[true_labels["is_shared"]]["label"].to_dict()
    assert shared == {1: 1, 2: 1, 3: 0, 4: 0}


def test_single_annotator_sentences_keep_their_label(path_annotations: Path) -> None:
    true_labels = load_gold_labels(path_annotations)

    single = true_labels[~true_labels["is_shared"]]["label"].to_dict()
    assert single == {10: 1, 20: 0, 30: 1}


def test_missing_annotation_file_names_the_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="annotation_dario_stratified.csv"):
        load_annotations(tmp_path)


def test_missing_label_column_is_named(path_annotations: Path) -> None:
    path_dario = path_annotations / "annotation_dario_stratified.csv"
    pd.read_csv(path_dario).drop(columns="label").to_csv(path_dario, index=False)

    with pytest.raises(KeyError, match="label"):
        load_annotations(path_annotations)


@pytest.mark.skipif(
    not (PATH_DATA / "annotation_dario_stratified.csv").is_file(),
    reason="annotation files are not available",
)
def test_real_annotations_resolve_to_700_gold_labels() -> None:
    true_labels = load_gold_labels(PATH_DATA)

    assert len(true_labels) == 700
    assert true_labels["is_shared"].sum() == 100
    assert (~true_labels["is_shared"]).sum() == 600
    assert set(true_labels["label"].unique()) <= {0, 1}
