import pandas as pd
import pytest

from annotation_assignment import assign_annotations
from sampling import sample_fixed_strata
from translation import add_translations
from verification import check_testset


@pytest.fixture
def master(scored_candidates) -> pd.DataFrame:
    sample = sample_fixed_strata(scored_candidates, seed=42)
    return add_translations(sample, sample["text"].tolist(), attempts=1)


@pytest.fixture
def annotation_sets(master) -> dict[str, pd.DataFrame]:
    return assign_annotations(master, seed=42)


def test_verification_report_summarises_a_valid_testset(master, annotation_sets):
    report = check_testset(master, annotation_sets)

    assert report == {
        "status": "passed",
        "master_rows": 700,
        "stratum_counts": {
            "0.0-0.3": 134,
            "0.3-0.5": 134,
            "0.5-0.7": 134,
            "0.7-0.9": 164,
            "0.9-1.0": 134,
        },
        "annotator_rows": {"dario": 300, "marjorie": 300, "yannis": 300},
        "shared_count": 100,
        "single_annotator_counts": {"dario": 200, "marjorie": 200, "yannis": 200},
        "translation_status_counts": {"success": 700},
        "unresolved_translation_anomaly_count": 0,
        "unresolved_translation_anomaly_ids": [],
    }


def test_verification_lists_translations_that_need_review(master, annotation_sets):
    master.loc[3, "translation_status"] = "manual_review"

    report = check_testset(master, annotation_sets)

    assert report["translation_status_counts"] == {"success": 699, "manual_review": 1}
    assert report["unresolved_translation_anomaly_ids"] == [
        str(master.loc[3, "sentence_id"])
    ]


def test_verification_rejects_filled_in_labels(master, annotation_sets):
    annotation_sets["yannis"].loc[0, "label"] = "1"

    with pytest.raises(ValueError, match="yannis label values must be empty"):
        check_testset(master, annotation_sets)


def test_verification_rejects_edited_source_text(master, annotation_sets):
    annotation_sets["dario"].loc[5, "text"] = "verändert"

    with pytest.raises(ValueError, match="dario does not preserve"):
        check_testset(master, annotation_sets)


def test_verification_rejects_overlapping_single_annotator_rows(
    master, annotation_sets
):
    dario = annotation_sets["dario"]
    marjorie = annotation_sets["marjorie"]
    shared = set(dario["sentence_id"]) & set(marjorie["sentence_id"])
    shared &= set(annotation_sets["yannis"]["sentence_id"])
    marjorie_only = marjorie[~marjorie["sentence_id"].isin(shared)].iloc[0]
    dario_only_index = dario.index[~dario["sentence_id"].isin(shared)][0]
    dario.loc[dario_only_index, ["sentence_id", "text"]] = marjorie_only[
        ["sentence_id", "text"]
    ].tolist()

    with pytest.raises(ValueError, match="exactly 200 IDs|overlap"):
        check_testset(master, annotation_sets)


def test_verification_rejects_a_wrong_stratum_allocation(master, annotation_sets):
    with pytest.raises(ValueError, match="exactly 700 rows"):
        check_testset(master.iloc[:699], annotation_sets)
