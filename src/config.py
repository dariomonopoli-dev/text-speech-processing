"""Label space and run naming shared by train.py and predict.py."""

import os

ID2LABEL = {0: "not_hate", 1: "hate"}
LABEL2ID = {label: label_id for label_id, label in ID2LABEL.items()}
LABEL_LIST = [0, 1]


def get_run_name(experiment_name: str, seed: int) -> str:
    """Return the run stem, e.g. gahd_gbert_raw_run42."""
    return f"{experiment_name}_run{seed}"


def get_run_path(path_runs_dir: str, experiment_name: str, seed: int) -> str:
    """Return the run directory, e.g. runs/gahd_gbert_raw/run42."""
    return os.path.join(path_runs_dir, experiment_name, f"run{seed}")
