import argparse

import pandas as pd
import pytest

import sample_master
from pipeline_io import PipelineError, load_csv, load_json, write_csv


@pytest.fixture
def args(tmp_path, scored_candidates) -> argparse.Namespace:
    write_csv(scored_candidates, tmp_path / "scored.csv")
    return argparse.Namespace(
        path_scored=tmp_path / "scored.csv",
        path_master=tmp_path / "master.csv",
        path_sampling_report=tmp_path / "sampling_report.json",
        seed=42,
        resume=False,
    )


def test_sample_step_writes_master_and_report(args):
    sample_master.main(args)

    master = load_csv(args.path_master)
    assert len(master) == 700
    assert load_json(args.path_sampling_report)["selected_counts"]["0.7-0.9"] == 164


def test_sample_step_refuses_to_overwrite_without_resume(args):
    sample_master.main(args)

    with pytest.raises(PipelineError, match="pass --resume"):
        sample_master.main(args)


def test_sample_step_overwrites_with_resume(args):
    sample_master.main(args)
    first = load_csv(args.path_master)
    args.resume = True

    sample_master.main(args)

    pd.testing.assert_frame_equal(load_csv(args.path_master), first)


def test_sample_step_requires_the_scored_input(args):
    args.path_scored.unlink()

    with pytest.raises(PipelineError, match="Required input does not exist"):
        sample_master.main(args)
