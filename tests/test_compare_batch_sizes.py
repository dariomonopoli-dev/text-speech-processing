import argparse

import pandas as pd
import pytest

import compare_batch_sizes

TEXTS = ["grüezi mitenand", "das isch guet", "i gang hei"]


def write_output(path, translations, anomalies) -> None:
    pd.DataFrame(
        {
            "sentence_id": ["1", "2", "3"],
            "text": TEXTS,
            "translation": translations,
            "translation_has_anomaly": anomalies,
        }
    ).to_csv(path, index=False, encoding="utf-8-sig")


@pytest.fixture
def args(tmp_path) -> argparse.Namespace:
    write_output(
        tmp_path / "b50.csv",
        ["Grüezi miteinander", "Das ist gut", "Ich gehe heim"],
        [False, False, False],
    )
    write_output(
        tmp_path / "b100.csv",
        ["Grüezi miteinander", "das isch guet", " I GANG HEI "],
        [False, True, True],
    )
    return argparse.Namespace(
        path_outputs=[tmp_path / "b50.csv", tmp_path / "b100.csv"],
        num_examples=8,
        path_summary=tmp_path / "summary.csv",
        overwrite=False,
    )


def test_main_counts_unchanged_anomalous_and_identical_translations(args):
    compare_batch_sizes.main(args)

    summary = pd.read_csv(args.path_summary, encoding="utf-8-sig")
    assert summary["output"].tolist() == ["b50", "b100"]
    assert summary["num_unchanged"].tolist() == [0, 2]
    assert summary["num_anomalies"].tolist() == [0, 2]
    assert summary["num_identical_to_first"].tolist() == [3, 1]


def test_main_aligns_outputs_by_sentence_id(args):
    shuffled = pd.read_csv(args.path_outputs[0], encoding="utf-8-sig").iloc[::-1]
    shuffled.to_csv(args.path_outputs[0], index=False, encoding="utf-8-sig")

    compare_batch_sizes.main(args)

    summary = pd.read_csv(args.path_summary, encoding="utf-8-sig")
    assert summary["num_identical_to_first"].tolist() == [3, 1]


def test_main_rejects_outputs_with_different_sentences(args):
    output = pd.read_csv(args.path_outputs[1], encoding="utf-8-sig")
    output.iloc[:2].to_csv(args.path_outputs[1], index=False, encoding="utf-8-sig")

    with pytest.raises(ValueError, match="different sentence_ids"):
        compare_batch_sizes.main(args)


def test_main_names_a_missing_column(args):
    output = pd.read_csv(args.path_outputs[1], encoding="utf-8-sig")
    output.drop(columns="translation_has_anomaly").to_csv(
        args.path_outputs[1], index=False, encoding="utf-8-sig"
    )

    with pytest.raises(KeyError, match="translation_has_anomaly"):
        compare_batch_sizes.main(args)


def test_main_needs_at_least_two_outputs(args):
    args.path_outputs = args.path_outputs[:1]

    with pytest.raises(ValueError, match="at least two"):
        compare_batch_sizes.main(args)


def test_main_refuses_an_existing_summary_without_overwrite(args):
    args.path_summary.write_text("old")

    with pytest.raises(FileExistsError, match="--overwrite"):
        compare_batch_sizes.main(args)
