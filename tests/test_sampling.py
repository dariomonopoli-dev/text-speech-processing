import pandas as pd
import pytest

from sampling import (
    FIXED_STRATUM_TARGETS,
    check_strata_match_scores,
    map_scores_to_strata,
    sample_fixed_strata,
    sample_master,
)


def test_strata_are_left_closed_and_last_band_contains_one():
    strata = map_scores_to_strata([0.0, 0.2999, 0.3, 0.5, 0.7, 0.8999, 0.9, 1.0])

    assert strata.astype(str).tolist() == [
        "0.0-0.3",
        "0.0-0.3",
        "0.3-0.5",
        "0.5-0.7",
        "0.7-0.9",
        "0.7-0.9",
        "0.9-1.0",
        "0.9-1.0",
    ]


@pytest.mark.parametrize("score", [-0.01, 1.01, float("nan")])
def test_strata_reject_scores_outside_unit_interval(score):
    with pytest.raises(ValueError, match="between 0 and 1"):
        map_scores_to_strata([0.5, score])


def test_fixed_strata_sample_has_700_rows_in_fixed_allocation(scored_candidates):
    sample = sample_fixed_strata(scored_candidates, seed=42)

    assert len(sample) == 700
    assert sample["austrotox_stratum"].value_counts().to_dict() == {
        "0.0-0.3": 134,
        "0.3-0.5": 134,
        "0.5-0.7": 134,
        "0.7-0.9": 164,
        "0.9-1.0": 134,
    }
    assert not sample["sentence_id"].duplicated().any()


def test_fixed_strata_sample_is_deterministic_for_a_seed(scored_candidates):
    first = sample_fixed_strata(scored_candidates, seed=7)
    second = sample_fixed_strata(scored_candidates, seed=7)
    other_seed = sample_fixed_strata(scored_candidates, seed=8)

    pd.testing.assert_frame_equal(first, second)
    assert first["sentence_id"].tolist() != other_seed["sentence_id"].tolist()


def test_fixed_strata_sample_names_undersized_strata(scored_candidates):
    sparse = scored_candidates[scored_candidates["hate_score"] < 0.9].copy()
    sparse = pd.concat([sparse, scored_candidates.iloc[-10:]], ignore_index=True)

    with pytest.raises(
        ValueError, match=r"Undersized strata: 0\.9-1\.0: available=10, required=134"
    ):
        sample_fixed_strata(sparse)


def test_fixed_strata_sample_rejects_duplicate_ids(scored_candidates):
    scored_candidates.loc[1, "sentence_id"] = scored_candidates.loc[0, "sentence_id"]

    with pytest.raises(ValueError, match="unique and non-null"):
        sample_fixed_strata(scored_candidates)


def test_master_sample_report_counts_source_and_selected_rows(scored_candidates):
    master, report = sample_master(scored_candidates, seed=42)

    assert report["sample_size"] == 700
    assert report["random_seed"] == 42
    assert report["source_counts"] == {
        stratum: 200 for stratum in FIXED_STRATUM_TARGETS
    }
    assert report["selected_counts"] == FIXED_STRATUM_TARGETS
    assert report["target_counts"] == FIXED_STRATUM_TARGETS
    assert report["score_summaries"]["0.3-0.5"]["min"] == pytest.approx(0.3)
    assert report["score_summaries"]["0.3-0.5"]["max"] < 0.5
    pd.testing.assert_frame_equal(
        master, sample_fixed_strata(scored_candidates, seed=42)
    )


def test_stratum_alignment_rejects_a_stratum_that_disagrees_with_its_score():
    frame = pd.DataFrame(
        {"hate_score": [0.1, 0.95], "austrotox_stratum": ["0.0-0.3", "0.7-0.9"]}
    )

    with pytest.raises(ValueError, match="does not match hate_score"):
        check_strata_match_scores(frame)
