import pandas as pd
import pytest

from annotation_assignment import assign_annotations

ANNOTATORS = ("dario", "marjorie", "yannis")


@pytest.fixture
def master() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sentence_id": [str(i) for i in range(700)],
            "text": [f"satz {i}" for i in range(700)],
            "translation_status": "success",
        }
    )


def test_each_annotator_gets_100_shared_and_200_single_annotator_rows(master):
    assignments = assign_annotations(master, seed=42)
    id_sets = {name: set(frame["sentence_id"]) for name, frame in assignments.items()}
    shared = set.intersection(*id_sets.values())
    single_annotator = {name: ids - shared for name, ids in id_sets.items()}

    assert list(assignments) == list(ANNOTATORS)
    assert all(len(frame) == 300 for frame in assignments.values())
    assert len(shared) == 100
    assert all(len(ids) == 200 for ids in single_annotator.values())
    assert len(set.union(*single_annotator.values())) == 600
    assert set.union(*id_sets.values()) == set(master["sentence_id"])


def test_assignment_files_carry_source_text_and_empty_label_columns(master):
    frame = assign_annotations(master, seed=42)["dario"]

    assert frame.columns.tolist() == ["sentence_id", "text", "label", "comment"]
    assert frame["label"].eq("").all() and frame["comment"].eq("").all()
    assert (frame["text"] == "satz " + frame["sentence_id"]).all()


def test_assignment_is_deterministic_and_shuffled_per_annotator(master):
    first = assign_annotations(master, seed=42)
    second = assign_annotations(master, seed=42)

    for name in ANNOTATORS:
        pd.testing.assert_frame_equal(first[name], second[name])
    assert first["dario"]["sentence_id"].tolist() != sorted(
        first["dario"]["sentence_id"], key=int
    )


def test_assignment_requires_a_terminal_translation_status(master):
    master.loc[0, "translation_status"] = "pending"

    with pytest.raises(ValueError, match="terminal translation status"):
        assign_annotations(master)


def test_assignment_requires_rows_to_divide_evenly(master):
    with pytest.raises(ValueError, match="divide evenly"):
        assign_annotations(master.iloc[:699])
