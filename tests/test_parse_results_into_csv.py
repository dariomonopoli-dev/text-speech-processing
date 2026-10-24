import argparse
import json
from pathlib import Path

import pandas as pd
import pytest

from parse_results_into_csv import main

F1_MACRO_PER_SEED = {42: 0.5, 43: 0.6, 44: 0.7}


def write_runs(path_runs: Path, seeds: list[int]) -> None:
    for seed in seeds:
        run_path = path_runs / f"run{seed}"
        run_path.mkdir(parents=True)
        metrics = {"f1-macro": F1_MACRO_PER_SEED[seed], "num_examples": 700}
        with open(
            run_path / "swiss_german_metrics.json", "w", encoding="utf-8"
        ) as fout:
            json.dump(metrics, fout)
        with open(
            run_path / "swiss_german_metrics_bootstrap.json", "w", encoding="utf-8"
        ) as fout:
            json.dump({"f1-macro-ci_low": 0.0}, fout)


def make_args(path_runs: Path, path_output_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        path_runs=str(path_runs),
        path_output_dir=str(path_output_dir),
        metrics_pattern="*metrics*.json",
        num_runs=3,
        bootstrap=False,
        predictions_name=None,
    )


def test_writes_one_row_per_run_and_seed_averages(tmp_path: Path) -> None:
    path_runs = tmp_path / "runs" / "gbert_swiss_german"
    write_runs(path_runs, [42, 43, 44])

    main(make_args(path_runs, tmp_path / "results"))

    raw_results = pd.read_csv(tmp_path / "results" / "raw_results.csv")
    assert raw_results["seed"].tolist() == [42, 43, 44]
    assert set(raw_results["eval_name"]) == {"swiss_german_metrics"}
    seed_avg_results = pd.read_csv(tmp_path / "results" / "seed_avg_results.csv")
    assert len(seed_avg_results) == 1
    row = seed_avg_results.iloc[0]
    assert row["experiment_name"] == "gbert_swiss_german"
    assert row["num_runs"] == 3
    assert row["mean-f1-macro"] == pytest.approx(0.6)
    assert row["f1-macro-stddev"] == pytest.approx(0.1)
    assert row["num_examples-stddev"] == 0


def test_missing_seed_fails_before_averaging(tmp_path: Path) -> None:
    path_runs = tmp_path / "runs" / "gbert_swiss_german"
    write_runs(path_runs, [42, 44])

    with pytest.raises(ValueError, match=r"runs \[42, 44\], expected 3"):
        main(make_args(path_runs, tmp_path / "results"))

    assert not (tmp_path / "results" / "seed_avg_results.csv").exists()


def test_missing_run_directory_is_named(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no_such_experiment"):
        main(make_args(tmp_path / "no_such_experiment", tmp_path / "results"))
