import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evaluate_predictions import (
    check_predictions_cover_gold,
    compute_binary_metrics,
    compute_bootstrap_intervals,
    load_predictions,
    main,
)

TRUE_LABELS = np.array([1, 1, 0, 0, 0])
SCORES = np.array([0.9, 0.4, 0.6, 0.1, 0.2])


def test_binary_metrics_match_hand_computation() -> None:
    metrics = compute_binary_metrics(TRUE_LABELS, SCORES, threshold=0.5)

    assert metrics == pytest.approx(
        {
            "acc": 3 / 5,
            "f1-macro": (1 / 2 + 2 / 3) / 2,
            "f1-pos": 1 / 2,
            "f1-neg": 2 / 3,
            "precision": 1 / 2,
            "recall": 1 / 2,
            "pr-auc": 5 / 6,
            "roc-auc": 5 / 6,
            "num_examples": 5,
            "num_true_hate": 2,
            "num_pred_hate": 2,
            "true_hs_ratio": 2 / 5,
            "pred_hs_ratio": 2 / 5,
        }
    )


def test_threshold_is_inclusive() -> None:
    metrics = compute_binary_metrics(TRUE_LABELS, SCORES, threshold=0.4)

    assert metrics["num_pred_hate"] == 3


def test_ranking_metrics_are_nan_without_both_classes() -> None:
    metrics = compute_binary_metrics(np.zeros(3, dtype=int), SCORES[:3], 0.5)

    assert math.isnan(metrics["pr-auc"])
    assert math.isnan(metrics["roc-auc"])


def test_bootstrap_is_deterministic_for_a_fixed_seed() -> None:
    first = compute_bootstrap_intervals(TRUE_LABELS, [SCORES], 0.5, 50, seed=7)
    second = compute_bootstrap_intervals(TRUE_LABELS, [SCORES], 0.5, 50, seed=7)
    other_seed = compute_bootstrap_intervals(TRUE_LABELS, [SCORES], 0.5, 50, seed=8)

    assert first == second
    assert first != other_seed
    assert first["f1-macro-ci_low"] <= first["f1-macro-ci_high"]


def test_missing_prediction_rows_are_reported() -> None:
    true_labels = pd.DataFrame({"label": [1, 0, 0]}, index=[1, 2, 3])
    raw_preds = pd.Series([0.3, 0.7], index=[1, 2])

    with pytest.raises(ValueError, match=r"1 test sentences .* \[3\]"):
        check_predictions_cover_gold(true_labels, raw_preds, Path("preds.csv"))


def test_missing_score_column_is_named(tmp_path: Path) -> None:
    path_predictions = tmp_path / "preds.csv"
    pd.DataFrame({"sentence_id": [1], "score": [0.2]}).to_csv(path_predictions)

    with pytest.raises(KeyError, match="prob_hate"):
        load_predictions(path_predictions, "prob_hate")


def test_existing_metrics_are_not_overwritten(tmp_path: Path) -> None:
    path_metrics = tmp_path / "metrics.json"
    path_metrics.write_text("{}")
    args = argparse.Namespace(
        path_predictions=str(tmp_path / "preds.csv"),
        path_annotations=str(tmp_path),
        score_column="prob_hate",
        subset="all",
        threshold=0.5,
        num_resamples=10,
        seed=42,
        path_metrics=str(path_metrics),
        overwrite=False,
    )

    with pytest.raises(FileExistsError, match="--overwrite"):
        main(args)
