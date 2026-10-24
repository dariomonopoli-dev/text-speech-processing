import argparse

import pytest

import translate_master
from pipeline_io import PipelineError, load_csv, write_csv, write_json
from sampling import sample_fixed_strata
from translation import build_translation_checkpoint


@pytest.fixture
def args(tmp_path, scored_candidates, monkeypatch) -> argparse.Namespace:
    monkeypatch.setattr(translate_master, "load_dotenv", lambda override: None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    write_csv(sample_fixed_strata(scored_candidates, seed=42), tmp_path / "master.csv")
    return argparse.Namespace(
        path_master=tmp_path / "master.csv",
        path_translation_checkpoint=tmp_path / "checkpoint.json",
        num_api_attempts=5,
        resume=False,
    )


def write_checkpoint(args: argparse.Namespace, master, status: str) -> None:
    raw_response = "\n".join(f"{i}. übersetzt {i}" for i in range(1, len(master) + 1))
    checkpoint = build_translation_checkpoint(
        master,
        status=status,
        attempts=1,
        input_tokens=0,
        output_tokens=0,
        raw_response=raw_response,
        error=None,
        attempt_history=[],
    )
    write_json(checkpoint, args.path_translation_checkpoint)


def test_translate_step_requires_an_api_key_without_a_reusable_checkpoint(args):
    with pytest.raises(PipelineError, match="ANTHROPIC_API_KEY is required"):
        translate_master.main(args)


def test_translate_step_restores_a_successful_checkpoint_without_an_api_key(args):
    master = load_csv(args.path_master)
    write_checkpoint(args, master, status="success")
    args.resume = True

    translate_master.main(args)

    translated_master = load_csv(args.path_master)
    assert translated_master["translation"].iloc[0] == "übersetzt 1"
    assert translated_master["translation_status"].eq("success").all()


def test_translate_step_refuses_an_existing_checkpoint_without_resume(args):
    write_checkpoint(args, load_csv(args.path_master), status="success")

    with pytest.raises(PipelineError, match="pass --resume to reuse it"):
        translate_master.main(args)


def test_translate_step_rejects_a_checkpoint_from_another_sample(args, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-real-key")
    other_master = load_csv(args.path_master).iloc[::-1].reset_index(drop=True)
    write_checkpoint(args, other_master, status="invalid")
    args.resume = True

    with pytest.raises(PipelineError, match="belongs to another sample"):
        translate_master.main(args)


def test_translate_step_rejects_a_master_with_the_wrong_row_count(args, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-real-key")
    write_csv(load_csv(args.path_master).iloc[:699], args.path_master)

    with pytest.raises(PipelineError, match="exactly 700 rows, got 699"):
        translate_master.main(args)
