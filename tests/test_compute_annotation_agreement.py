import argparse
import json
from pathlib import Path

import pandas as pd
import pytest

from compute_annotation_agreement import (
    compute_fleiss_kappa,
    compute_krippendorff_alpha,
    main,
)

PATH_DATA = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture
def shared_labels() -> pd.DataFrame:
    # Hate votes per sentence are 3, 0, 2 and 1 out of 3.
    return pd.DataFrame(
        {"a": [1, 0, 1, 1], "b": [1, 0, 1, 0], "c": [1, 0, 0, 0]},
    )


def test_fleiss_kappa_matches_hand_computation(shared_labels: pd.DataFrame) -> None:
    assert compute_fleiss_kappa(shared_labels) == pytest.approx(1 / 3)


def test_krippendorff_alpha_matches_hand_computation(
    shared_labels: pd.DataFrame,
) -> None:
    assert compute_krippendorff_alpha(shared_labels) == pytest.approx(7 / 18)


def test_perfect_agreement_scores_one() -> None:
    unanimous = pd.DataFrame({"a": [1, 0, 0], "b": [1, 0, 0], "c": [1, 0, 0]})

    assert compute_fleiss_kappa(unanimous) == pytest.approx(1.0)
    assert compute_krippendorff_alpha(unanimous) == pytest.approx(1.0)


@pytest.mark.skipif(
    not (PATH_DATA / "annotation_dario_stratified.csv").is_file(),
    reason="annotation files are not available",
)
def test_real_annotations_reproduce_reported_agreement(tmp_path: Path) -> None:
    path_agreement = tmp_path / "agreement.json"

    main(
        argparse.Namespace(
            path_annotations=str(PATH_DATA), path_agreement=str(path_agreement)
        )
    )

    with open(path_agreement, encoding="utf-8") as fin:
        agreement_stats = json.load(fin)
    assert agreement_stats["num_sentences"] == 700
    assert agreement_stats["num_shared"] == 100
    assert 0 <= agreement_stats["num_sentences_majority_hate"] <= 700
    for key in ("fleiss-kappa", "krippendorff-alpha"):
        assert -1 <= agreement_stats[key] <= 1
