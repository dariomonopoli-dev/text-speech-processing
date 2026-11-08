import argparse

import pandas as pd
import pytest

import analyse_hate


@pytest.fixture
def sentences() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sentence_id": [1, 2, 3],
            "Target": ["Women", "Women & Disability", "Disability"],
            "Mechanism": ["Slur", "Implicit / mockery", "Slur"],
            "mean_prob_hate_raw": [0.9, 0.5, 0.2],
            "mean_prob_hate_apertus8b": [0.4, 0.6, 0.7],
            "mean_prob_hate_haiku": [0.1, 0.49, 0.8],
        }
    )


def test_split_tags_separates_two_targets():
    assert analyse_hate.split_tags("Women & Disability") == [
        "Women",
        "Disability",
    ]


def test_compute_analysis_table_counts_a_sentence_once_per_target(sentences):
    table = analyse_hate.compute_analysis_table(sentences, 0.5)

    women = table[table["name"] == "Women"].iloc[0]
    assert (women["n"], women["raw"], women["apertus"], women["haiku"]) == (2, 2, 1, 0)
    disability = table[table["name"] == "Disability"].iloc[0]
    assert (disability["n"], disability["raw"]) == (2, 1)


def test_compute_analysis_table_counts_each_sentence_once_in_all(sentences):
    table = analyse_hate.compute_analysis_table(sentences, 0.5)

    everything = table[table["name"] == "All"].iloc[0]
    assert (everything["n"], everything["raw"], everything["haiku"]) == (3, 2, 1)


def test_threshold_is_inclusive(sentences):
    table = analyse_hate.compute_analysis_table(sentences, 0.5)

    implicit = table[table["name"] == "Implicit / mockery"].iloc[0]
    assert (implicit["raw"], implicit["haiku"]) == (1, 0)


def test_misspelled_tag_is_listed_as_unexpected(sentences):
    sentences.loc[0, "Mechanism"] = "slur"

    table = analyse_hate.compute_analysis_table(sentences, 0.5)

    assert "slur" in table["name"].tolist()
    mismatches = analyse_hate.find_mismatches(table)
    unexpected = {"name": "slur", "column": "tag", "expected": "absent", "actual": "present"}
    assert unexpected in mismatches


def test_find_mismatches_reports_every_differing_count(sentences):
    table = analyse_hate.compute_analysis_table(sentences, 0.5)

    mismatches = analyse_hate.find_mismatches(table)

    women_n = [m for m in mismatches if m["name"] == "Women" and m["column"] == "n"]
    assert women_n == [{"name": "Women", "column": "n", "expected": 9, "actual": 2}]


def test_format_latex_rows_ends_rows_with_a_line_break(sentences):
    table = analyse_hate.compute_analysis_table(sentences, 0.5)

    rows = analyse_hate.format_latex_rows(table)

    assert rows[-1] == "All & 3 & 2 & 2 & 1 \\\\"


def test_load_hate_sentences_names_an_empty_tag(tmp_path, sentences):
    sentences.loc[1, "Target"] = None
    sentences.to_csv(tmp_path / "hate.csv", index=False)

    with pytest.raises(ValueError, match=r"Target is empty for sentence_ids \[2\]"):
        analyse_hate.load_hate_sentences(tmp_path / "hate.csv")


def test_load_hate_sentences_names_missing_columns(tmp_path, sentences):
    sentences.drop(columns="Mechanism").to_csv(tmp_path / "hate.csv", index=False)

    with pytest.raises(KeyError, match="Mechanism"):
        analyse_hate.load_hate_sentences(tmp_path / "hate.csv")


def test_main_writes_the_table_and_refuses_to_overwrite(tmp_path, sentences):
    sentences.to_csv(tmp_path / "hate.csv", index=False)
    args = argparse.Namespace(
        path_sentences=tmp_path / "hate.csv",
        threshold=0.5,
        margin=0.02,
        path_output=tmp_path / "table.csv",
        overwrite=False,
    )

    analyse_hate.main(args)

    written = pd.read_csv(args.path_output, encoding="utf-8-sig")
    assert written["name"].iloc[-1] == "All"
    with pytest.raises(FileExistsError, match="--overwrite"):
        analyse_hate.main(args)
