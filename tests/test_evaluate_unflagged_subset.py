import argparse
import json

import numpy as np
import pandas as pd
import pytest

import evaluate_unflagged_subset

SENTENCE_IDS = list(range(1, 11))
HATE_IDS = {1, 2, 3}


@pytest.fixture
def args(tmp_path) -> argparse.Namespace:
    for annotator in ["dario", "marjorie", "yannis"]:
        pd.DataFrame(
            {
                "sentence_id": SENTENCE_IDS,
                "label": [int(i in HATE_IDS) for i in SENTENCE_IDS],
            }
        ).to_csv(tmp_path / f"annotation_{annotator}_stratified.csv", index=False)

    flags = [False] * 10
    flags[3] = True
    pd.DataFrame(
        {"sentence_id": SENTENCE_IDS, "translation_has_anomaly": flags}
    ).to_csv(tmp_path / "translations_a.csv", index=False)
    other_flags = [False] * 10
    other_flags[5] = True
    pd.DataFrame(
        {"sentence_id": SENTENCE_IDS, "translation_has_anomaly": other_flags}
    ).to_csv(tmp_path / "translations_b.csv", index=False)

    for seed in [42, 43]:
        run_path = tmp_path / "runs" / f"run{seed}"
        run_path.mkdir(parents=True)
        pd.DataFrame(
            {
                "sentence_id": SENTENCE_IDS,
                "prob_hate": [0.9 if i in HATE_IDS else 0.1 for i in SENTENCE_IDS],
            }
        ).to_csv(run_path / "predictions_swiss_german_text.csv", index=False)

    return argparse.Namespace(
        path_annotations=tmp_path,
        path_translations=[tmp_path / "translations_a.csv", tmp_path / "translations_b.csv"],
        path_runs=tmp_path / "runs",
        variants=["text"],
        score_column="prob_hate",
        threshold=0.5,
        num_resamples=20,
        seed=42,
        path_output=tmp_path / "result.json",
        overwrite=False,
    )


def test_load_flagged_ids_returns_only_the_flagged_sentences(args):
    flagged_ids = evaluate_unflagged_subset.load_flagged_ids(args.path_translations[0])

    assert flagged_ids == {4}


def test_load_flagged_ids_names_a_missing_column(args, tmp_path):
    pd.DataFrame({"sentence_id": [1]}).to_csv(tmp_path / "bad.csv", index=False)

    with pytest.raises(KeyError, match=evaluate_unflagged_subset.FLAG_COLUMN):
        evaluate_unflagged_subset.load_flagged_ids(tmp_path / "bad.csv")


def test_main_excludes_the_sentences_flagged_in_either_file(args):
    evaluate_unflagged_subset.main(args)

    with open(args.path_output) as fin:
        results = json.load(fin)
    assert results["num_examples"] == 8
    assert results["num_excluded"] == 2
    assert results["num_true_hate"] == 3
    assert results["variants"]["text"]["metrics"]["f1-pos"] == pytest.approx(1.0)


def test_main_refuses_an_existing_output_without_overwrite(args):
    args.path_output.write_text("old")

    with pytest.raises(FileExistsError, match="--overwrite"):
        evaluate_unflagged_subset.main(args)


def test_main_fails_when_every_hate_sentence_is_excluded(args, tmp_path):
    pd.DataFrame(
        {
            "sentence_id": SENTENCE_IDS,
            "translation_has_anomaly": [i in HATE_IDS for i in SENTENCE_IDS],
        }
    ).to_csv(args.path_translations[0], index=False)

    with pytest.raises(ValueError, match="only gold label"):
        evaluate_unflagged_subset.main(args)


def test_compute_subset_results_returns_metrics_and_intervals():
    true_labels = np.array([1, 0, 1, 0, 0, 0])
    scores = np.array([0.9, 0.1, 0.8, 0.2, 0.3, 0.1])

    results = evaluate_unflagged_subset.compute_subset_results(
        true_labels, [scores], threshold=0.5, num_resamples=10, seed=42
    )

    assert results["metrics"]["f1-pos"] == pytest.approx(1.0)
    assert "f1-pos-ci_low" in results["intervals"]
