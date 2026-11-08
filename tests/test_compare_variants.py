import argparse
import json

import numpy as np
import pandas as pd
import pytest

import compare_variants

SENTENCE_IDS = list(range(1, 41))
HATE_IDS = set(range(1, 9))


@pytest.fixture
def args(tmp_path) -> argparse.Namespace:
    for annotator in ["dario", "marjorie", "yannis"]:
        pd.DataFrame(
            {
                "sentence_id": SENTENCE_IDS,
                "label": [int(i in HATE_IDS) for i in SENTENCE_IDS],
            }
        ).to_csv(tmp_path / f"annotation_{annotator}_stratified.csv", index=False)

    good_scores = [0.9 if i in HATE_IDS else 0.1 for i in SENTENCE_IDS]
    bad_scores = [0.1 if i in HATE_IDS else 0.9 for i in SENTENCE_IDS]
    for seed in [42, 43]:
        run_path = tmp_path / "runs" / f"run{seed}"
        run_path.mkdir(parents=True)
        for variant, scores in [("good", good_scores), ("bad", bad_scores), ("same", good_scores)]:
            pd.DataFrame({"sentence_id": SENTENCE_IDS, "prob_hate": scores}).to_csv(
                run_path / f"predictions_swiss_german_{variant}.csv", index=False
            )

    return argparse.Namespace(
        path_annotations=tmp_path,
        path_runs=tmp_path / "runs",
        pairs=["good:bad", "good:same"],
        score_column="prob_hate",
        threshold=0.5,
        num_resamples=50,
        seed=42,
        path_output=tmp_path / "result.json",
        overwrite=False,
    )


def test_parse_pair_splits_two_variants():
    assert compare_variants.parse_pair("a:b") == ("a", "b")


@pytest.mark.parametrize("pair", ["a", "a:b:c", "a:", ":b"])
def test_parse_pair_rejects_a_malformed_pair(pair):
    with pytest.raises(ValueError, match="must look like"):
        compare_variants.parse_pair(pair)


def test_main_reports_a_clear_difference_with_an_interval_above_zero(args):
    compare_variants.main(args)

    with open(args.path_output) as fin:
        results = json.load(fin)
    f1_macro = results["good_minus_bad"]["f1-macro"]
    assert f1_macro["difference"] == pytest.approx(1.0)
    assert f1_macro["ci_low"] > 0
    assert f1_macro["share_resamples_above_zero"] == pytest.approx(1.0)


def test_main_reports_no_difference_for_identical_predictions(args):
    compare_variants.main(args)

    with open(args.path_output) as fin:
        results = json.load(fin)
    f1_pos = results["good_minus_same"]["f1-pos"]
    assert f1_pos["difference"] == pytest.approx(0.0)
    assert f1_pos["ci_low"] == pytest.approx(0.0)
    assert f1_pos["ci_high"] == pytest.approx(0.0)


def test_main_refuses_an_existing_output_without_overwrite(args):
    args.path_output.write_text("old")

    with pytest.raises(FileExistsError, match="--overwrite"):
        compare_variants.main(args)


def test_main_fails_for_a_variant_without_prediction_files(args):
    args.pairs = ["good:missing"]

    with pytest.raises(FileNotFoundError, match="missing"):
        compare_variants.main(args)


def test_compute_paired_differences_rejects_an_unknown_variant():
    with pytest.raises(KeyError, match="nope"):
        compare_variants.compute_paired_differences(
            np.array([1, 0, 1, 0]),
            {"a": [np.array([0.9, 0.1, 0.8, 0.2])]},
            [("a", "nope")],
            threshold=0.5,
            num_resamples=5,
            seed=42,
        )
