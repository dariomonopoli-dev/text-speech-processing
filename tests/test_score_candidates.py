import argparse

import pytest

import score_candidates
from pipeline_io import PipelineError, write_csv


@pytest.fixture
def args(tmp_path, scored_candidates, monkeypatch) -> argparse.Namespace:
    monkeypatch.setattr(score_candidates, "load_dotenv", lambda override: None)
    write_csv(scored_candidates.drop(columns="hate_score"), tmp_path / "candidates.csv")
    write_csv(scored_candidates, tmp_path / "scored.csv")
    return argparse.Namespace(
        path_candidates=tmp_path / "candidates.csv",
        path_scored=tmp_path / "scored.csv",
        batch_size=32,
        device="cpu",
        resume=True,
    )


def test_score_step_with_resume_reuses_a_matching_scored_output(args, monkeypatch):
    def fail_to_load(device):
        raise AssertionError("the classifier must not be loaded")

    monkeypatch.setattr(score_candidates, "load_gahd_classifier", fail_to_load)

    score_candidates.main(args)


def test_score_step_refuses_to_overwrite_without_resume(args):
    args.resume = False

    with pytest.raises(PipelineError, match="pass --resume"):
        score_candidates.main(args)


def test_score_step_rejects_a_scored_output_for_other_candidates(
    args, scored_candidates
):
    write_csv(scored_candidates.iloc[1:], args.path_scored)

    with pytest.raises(PipelineError, match="does not match the candidate corpus"):
        score_candidates.main(args)


def test_score_step_rejects_a_non_positive_batch_size(args):
    args.batch_size = 0

    with pytest.raises(PipelineError, match="--batch_size must be positive"):
        score_candidates.main(args)
