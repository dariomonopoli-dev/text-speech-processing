import argparse
import json

import pandas as pd
import pytest

import translate_local
from pipeline_io import PipelineError, write_csv

MODEL_NAME = "fake/model"


def fake_generate(prompt: str) -> str:
    """Translate the numbered sentences at the end of the prompt by upper-casing them."""
    numbered_lines = prompt.split("Sätze:")[1].strip().splitlines()
    return "\n".join(line.upper() for line in numbered_lines)


@pytest.fixture
def sentences() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sentence_id": pd.array(["11", "12", "13", "14", "15"], dtype="string"),
            "text": [f"satz {i} isch kurz" for i in range(1, 6)],
        }
    )


@pytest.fixture
def args(tmp_path, sentences, monkeypatch) -> argparse.Namespace:
    monkeypatch.setattr(
        translate_local, "make_generate", lambda model_name, max_tokens: fake_generate
    )
    write_csv(sentences, tmp_path / "sentences.csv")
    return argparse.Namespace(
        path_sentences=tmp_path / "sentences.csv",
        path_output=tmp_path / "translated.csv",
        path_checkpoint=tmp_path / "checkpoint.jsonl",
        model_name=MODEL_NAME,
        batch_size=2,
        max_tokens=64,
        resume=False,
    )


def test_translate_batch_parses_one_translation_per_sentence():
    translations = translate_local.translate_batch(["eins", "zwei"], fake_generate)

    assert translations == ["EINS", "ZWEI"]


def test_translate_batch_splits_a_batch_whose_reply_has_the_wrong_length():
    calls = []

    def generate_one_line_for_two(prompt: str) -> str:
        calls.append(prompt)
        lines = prompt.split("Sätze:")[1].strip().splitlines()
        return lines[0] if len(lines) == 2 else "\n".join(lines)

    translations = translate_local.translate_batch(
        ["eins", "zwei"], generate_one_line_for_two
    )

    assert translations == ["eins", "zwei"]
    assert len(calls) == 3


def test_translate_batch_returns_none_for_a_sentence_that_cannot_be_parsed():
    translations = translate_local.translate_batch(["eins"], lambda prompt: "kein format")

    assert translations == [None]


def test_translate_sentences_keeps_the_input_order_and_records_the_model(
    sentences, tmp_path
):
    translated = translate_local.translate_sentences(
        sentences, fake_generate, MODEL_NAME, 2, tmp_path / "c.jsonl"
    )

    assert translated["sentence_id"].tolist() == ["11", "12", "13", "14", "15"]
    assert translated["translation"].notna().all()
    assert translated["translation_model"].eq(MODEL_NAME).all()


def test_translate_sentences_flags_a_missing_translation_for_review(sentences, tmp_path):
    translated = translate_local.translate_sentences(
        sentences, lambda prompt: "kein format", MODEL_NAME, 5, tmp_path / "c.jsonl"
    )

    assert translated["translation_missing"].all()
    assert translated["translation_status"].eq("manual_review").all()


def test_translate_sentences_flags_a_translation_identical_to_the_source(
    sentences, tmp_path
):
    def echo_generate(prompt: str) -> str:
        numbered_lines = prompt.split("Sätze:")[1].strip().splitlines()
        return "\n".join(numbered_lines)

    translated = translate_local.translate_sentences(
        sentences, echo_generate, MODEL_NAME, 5, tmp_path / "c.jsonl"
    )

    assert translated["translation_unchanged"].all()
    assert translated["translation_has_anomaly"].all()
    assert translated["translation_status"].eq("manual_review").all()


def test_translate_sentences_does_not_flag_a_real_translation(sentences, tmp_path):
    def rewording_generate(prompt: str) -> str:
        numbered_lines = prompt.split("Sätze:")[1].strip().splitlines()
        return "\n".join(line.replace("isch", "ist") for line in numbered_lines)

    translated = translate_local.translate_sentences(
        sentences, rewording_generate, MODEL_NAME, 5, tmp_path / "c.jsonl"
    )

    assert not translated["translation_unchanged"].any()
    assert translated["translation_status"].eq("success").all()


def test_resume_only_translates_sentences_missing_from_the_checkpoint(
    sentences, tmp_path
):
    path_checkpoint = tmp_path / "c.jsonl"
    translate_local.append_checkpoint(path_checkpoint, ["11", "12"], ["ERSTE", "ZWEITE"], MODEL_NAME)
    prompts = []

    def recording_generate(prompt: str) -> str:
        prompts.append(prompt)
        return fake_generate(prompt)

    translated = translate_local.translate_sentences(
        sentences, recording_generate, MODEL_NAME, 10, path_checkpoint
    )

    assert translated["translation"].tolist()[:2] == ["ERSTE", "ZWEITE"]
    assert len(prompts) == 1
    assert "satz 1 isch kurz" not in prompts[0]


def test_load_checkpoint_refuses_records_of_another_model(tmp_path):
    path_checkpoint = tmp_path / "c.jsonl"
    translate_local.append_checkpoint(path_checkpoint, ["11"], ["ERSTE"], "other/model")

    with pytest.raises(PipelineError, match="other/model"):
        translate_local.load_checkpoint(path_checkpoint, MODEL_NAME)


def test_main_writes_translations_and_a_checkpoint(args):
    translate_local.main(args)

    translated = pd.read_csv(args.path_output, encoding="utf-8-sig")
    assert translated["translation"].tolist()[0] == "SATZ 1 ISCH KURZ"
    assert len(translated) == 5
    records = [json.loads(line) for line in args.path_checkpoint.read_text().splitlines()]
    assert len(records) == 5


def test_main_refuses_an_existing_output_without_resume(args):
    args.path_output.write_text("old")

    with pytest.raises(PipelineError, match="--resume"):
        translate_local.main(args)


def test_main_rejects_sentences_without_the_text_column(args):
    write_csv(pd.DataFrame({"sentence_id": ["1"]}), args.path_sentences)

    with pytest.raises(KeyError, match="text"):
        translate_local.main(args)


def test_main_rejects_duplicate_sentence_ids(args, sentences):
    write_csv(pd.concat([sentences, sentences.iloc[:1]]), args.path_sentences)

    with pytest.raises(ValueError, match="unique"):
        translate_local.main(args)
