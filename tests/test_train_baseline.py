import argparse

import pandas as pd
import pytest

import train_baseline

HATEFUL = ["alle ausländer sind dreck", "frauen sind dumm und wertlos"]
NOT_HATEFUL = ["das wetter ist heute schön", "wir gehen morgen wandern"]


def make_gahd_split(split: str) -> pd.DataFrame:
    texts = (HATEFUL + NOT_HATEFUL) * 5
    return pd.DataFrame(
        {
            "gahd_id": range(len(texts)),
            "text": texts,
            "label": ([1] * len(HATEFUL) + [0] * len(NOT_HATEFUL)) * 5,
            "split": split,
        }
    )


@pytest.fixture
def fake_gahd(monkeypatch):
    monkeypatch.setattr(
        train_baseline,
        "load_gahd",
        lambda: tuple(make_gahd_split(split) for split in ["train", "dev", "test"]),
    )


def make_args(tmp_path, path_eval_set) -> argparse.Namespace:
    return argparse.Namespace(
        path_eval_sets=[str(path_eval_set)],
        input_columns=["text"],
        experiment_name="gahd_tfidf_logreg",
        path_runs_dir=str(tmp_path / "runs"),
        seed=42,
    )


def test_run_writes_predictions_the_evaluator_reads(tmp_path, fake_gahd):
    path_eval_set = tmp_path / "eval_sentences.csv"
    pd.DataFrame(
        {"sentence_id": [11, 12], "text": ["alle ausländer sind dreck", "schön"]}
    ).to_csv(path_eval_set, index=False)

    train_baseline.main(make_args(tmp_path, path_eval_set))

    run_path = tmp_path / "runs" / "gahd_tfidf_logreg" / "run42"
    predictions = pd.read_csv(run_path / "predictions_swiss_german_text.csv")
    assert list(predictions.columns) == ["sentence_id", "prediction", "prob_hate"]
    assert predictions["sentence_id"].tolist() == [11, 12]
    assert predictions["prediction"].tolist() == [1, 0]
    assert (run_path / "gahd_test_metrics.json").is_file()


def test_existing_run_is_not_overwritten(tmp_path, fake_gahd):
    path_eval_set = tmp_path / "eval_sentences.csv"
    pd.DataFrame({"sentence_id": [11], "text": ["schön"]}).to_csv(
        path_eval_set, index=False
    )
    train_baseline.main(make_args(tmp_path, path_eval_set))

    with pytest.raises(FileExistsError):
        train_baseline.main(make_args(tmp_path, path_eval_set))


def test_mismatched_eval_sets_and_columns_are_refused(tmp_path):
    args = make_args(tmp_path, tmp_path / "eval_sentences.csv")
    args.input_columns = ["text", "translation_haiku"]

    with pytest.raises(ValueError, match="--input_columns"):
        train_baseline.main(args)
