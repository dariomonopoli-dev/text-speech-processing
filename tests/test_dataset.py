from pathlib import Path

import pandas as pd
import pytest

from dataset import load_gahd


def test_local_gahd_preserves_splits_labels_and_removes_blank_text(
    tmp_path: Path,
) -> None:
    path_gahd = tmp_path / "gahd.csv"
    pd.DataFrame(
        {
            "gahd_id": [1, 2, 3, 4, 5],
            "text": [" train ", "dev", "test", " ", None],
            "label": [1, 0, 1, 0, 0],
            "split": ["train", "dev", "test", "dev", "train"],
        }
    ).to_csv(path_gahd, index=False)

    train_set, dev_set, test_set = load_gahd(str(path_gahd))

    assert train_set[["gahd_id", "text", "label"]].to_dict("records") == [
        {"gahd_id": 1, "text": " train ", "label": 1}
    ]
    assert dev_set["gahd_id"].tolist() == [2]
    assert test_set["gahd_id"].tolist() == [3]


def test_local_gahd_rejects_nonbinary_labels(tmp_path: Path) -> None:
    path_gahd = tmp_path / "gahd.csv"
    pd.DataFrame(
        {"gahd_id": [1], "text": ["text"], "label": [2], "split": ["train"]}
    ).to_csv(path_gahd, index=False)

    with pytest.raises(ValueError, match="binary"):
        load_gahd(str(path_gahd))
