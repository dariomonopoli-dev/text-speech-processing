import argparse

import pandas as pd
import pytest

import analyse_false_alarms


@pytest.fixture
def scores() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sentence_id": [1, 2, 3, 4, 5, 6],
            "text": [f"satz {i}" for i in range(1, 7)],
            "gold_label": [0, 0, 0, 0, 1, 1],
            "translation_haiku": [f"haiku {i}" for i in range(1, 7)],
            "translation_apertus8b": [f"apertus {i}" for i in range(1, 7)],
            "mean_prob_hate_raw": [0.9, 0.1, 0.6, 0.2, 0.8, 0.3],
            "mean_prob_hate_apertus8b": [0.4, 0.7, 0.1, 0.2, 0.9, 0.6],
            "mean_prob_hate_haiku": [0.1, 0.2, 0.3, 0.55, 0.7, 0.4],
        }
    )


@pytest.fixture
def args(tmp_path, scores) -> argparse.Namespace:
    scores.to_csv(tmp_path / "scores.csv", index=False)
    return argparse.Namespace(
        step="select",
        path_scores=tmp_path / "scores.csv",
        path_tag_file=tmp_path / "tag.csv",
        path_summary=tmp_path / "summary.csv",
        threshold=0.5,
        top_k=1,
        overwrite=False,
    )


def test_compute_alarm_counts_separates_false_alarms_from_hits(scores):
    counts = analyse_false_alarms.compute_alarm_counts(scores, 0.5)

    assert counts["raw"] == {
        "false_alarms": 2,
        "num_not_hate": 4,
        "hate_caught": 1,
        "num_hate": 2,
    }
    assert counts["hk"]["false_alarms"] == 1
    assert counts["ap"]["hate_caught"] == 2


def test_select_false_alarms_takes_the_top_sentences_of_every_variant(scores):
    selected = analyse_false_alarms.select_false_alarms(scores, top_k=1)

    assert selected["sentence_id"].tolist() == [1, 2, 4]
    assert selected["in_top_k"].tolist() == ["raw", "ap", "hk"]
    assert (selected["category"] == "").all()


def test_select_false_alarms_marks_a_sentence_in_several_top_lists(scores):
    scores.loc[scores["sentence_id"] == 1, "mean_prob_hate_haiku"] = 0.99

    selected = analyse_false_alarms.select_false_alarms(scores, top_k=1)

    assert selected.loc[selected["sentence_id"] == 1, "in_top_k"].item() == "raw/hk"


def test_run_select_writes_the_tag_file(args):
    analyse_false_alarms.main(args)

    tag_file = pd.read_csv(args.path_tag_file, encoding="utf-8-sig")
    assert tag_file["sentence_id"].tolist() == [1, 2, 4]


def test_run_select_refuses_to_overwrite_manual_tags(args):
    args.path_tag_file.write_text("tagged")

    with pytest.raises(FileExistsError, match="--overwrite"):
        analyse_false_alarms.main(args)


def test_load_scores_names_missing_columns(tmp_path):
    pd.DataFrame({"sentence_id": [1]}).to_csv(tmp_path / "scores.csv", index=False)

    with pytest.raises(KeyError, match="gold_label"):
        analyse_false_alarms.load_scores(tmp_path / "scores.csv")


def test_summarise_categories_counts_scores_at_the_threshold(scores):
    tagged = scores[scores["gold_label"] == 0].assign(
        category=["slur", "slur", "insult", "insult"]
    )

    summary = analyse_false_alarms.summarise_categories(tagged, 0.5)

    assert summary["category"].tolist() == ["insult", "slur", "All"]
    slur = summary[summary["category"] == "slur"].iloc[0]
    assert slur["n"] == 2
    assert slur["above_threshold_raw"] == 1
    assert slur["mean_raw"] == 0.5
    assert summary.iloc[-1]["n"] == 4


def test_run_summarise_ignores_untagged_rows(args):
    analyse_false_alarms.main(args)
    tag_file = pd.read_csv(args.path_tag_file, encoding="utf-8-sig")
    tag_file["category"] = ["slur", "slur", None]
    tag_file.to_csv(args.path_tag_file, index=False)
    args.step = "summarise"

    analyse_false_alarms.main(args)

    summary = pd.read_csv(args.path_summary, encoding="utf-8-sig")
    assert summary["category"].tolist() == ["slur", "All"]
    assert summary["n"].tolist() == [2, 2]


def test_run_summarise_refuses_to_overwrite_an_existing_summary(args):
    analyse_false_alarms.main(args)
    args.step = "summarise"
    args.path_summary.write_text("old")

    with pytest.raises(FileExistsError, match="--overwrite"):
        analyse_false_alarms.main(args)
