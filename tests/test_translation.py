from types import SimpleNamespace

import pandas as pd
import pytest

from translation import (
    add_translations,
    build_translation_checkpoint,
    check_translation_checkpoint,
    compute_fingerprint,
    compute_translation_audits,
    parse_translation_response,
    restore_translations,
    to_translation_prompt,
    translate_master,
)


@pytest.fixture
def selected() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sentence_id": pd.array(["11", "12", "13"], dtype="string"),
            "text": ["das isch guet", "de chunnt spöter", "mir gönd hei"],
        }
    )


def make_response(text: str, stop_reason: str = "end_turn") -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=1000, output_tokens=200),
    )


class FakeStream:
    def __init__(self, response):
        self.response = response

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def get_final_message(self):
        return self.response


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.messages = self

    def stream(self, **request):
        return FakeStream(self.responses.pop(0))


VALID_RESPONSE = "1. das ist gut\n2. der kommt später\n3. wir gehen heim"


def test_audit_accepts_a_faithful_translation():
    audit = compute_translation_audits(["das isch guet"], ["das ist gut"])

    assert not audit.loc[0, "has_anomaly"]
    assert audit.loc[0, "length_ratio"] == pytest.approx(11 / 13)


@pytest.mark.parametrize(
    "translation, flag",
    [
        (None, "missing"),
        ("   ", "missing"),
        ("Hinweis: das ist gut", "meta_commentary"),
        ("Translation: das ist gut", "meta_commentary"),
        ("das ist\n\ngut", "multiple_paragraphs"),
        ("gut", "extreme_length"),
        ("das ist gut " * 10, "extreme_length"),
    ],
)
def test_audit_flags_malformed_translations(translation, flag):
    audit = compute_translation_audits(["das isch guet"], [translation])

    assert audit.loc[0, flag]
    assert audit.loc[0, "has_anomaly"]


def test_audit_rejects_unequal_lengths():
    with pytest.raises(ValueError, match="same length"):
        compute_translation_audits(["a", "b"], ["a"])


def test_prompt_numbers_sentences_and_flattens_line_breaks():
    prompt = to_translation_prompt(["erschti\nzeile", "zwöiti"])

    assert prompt.endswith("Sätze:\n\n1. erschti zeile\n2. zwöiti")


def test_response_parsing_returns_one_translation_per_row():
    translations = parse_translation_response(VALID_RESPONSE, 3)

    assert translations == ["das ist gut", "der kommt später", "wir gehen heim"]


@pytest.mark.parametrize(
    "response, message",
    [
        (None, "received no text"),
        ("1. a\n2. b", "received 2 lines"),
        ("1. a\n3. b\n2. c", "line 2 is invalid"),
        ("1. a\n2.b\n3. c", "line 2 is invalid"),
    ],
)
def test_response_parsing_rejects_malformed_responses(response, message):
    with pytest.raises(ValueError, match=message):
        parse_translation_response(response, 3)


def test_merge_marks_anomalous_translations_for_manual_review(selected):
    selected["translation"] = "stale"
    merged = add_translations(
        selected, ["das ist gut", "Hinweis: kommt", "wir gehen heim"], attempts=2
    )

    assert merged["translation"].tolist() == [
        "das ist gut",
        "Hinweis: kommt",
        "wir gehen heim",
    ]
    assert merged["translation_status"].tolist() == [
        "success",
        "manual_review",
        "success",
    ]
    assert merged["translation_meta_commentary"].tolist() == [False, True, False]
    assert merged["translation_attempts"].eq(2).all()
    assert merged["translation_error"].isna().all()


def test_fingerprint_is_stable_and_order_sensitive(selected):
    reordered = selected.iloc[::-1].reset_index(drop=True)

    assert compute_fingerprint(selected) == compute_fingerprint(selected.copy())
    assert compute_fingerprint(selected) != compute_fingerprint(reordered)
    assert compute_fingerprint(selected) != compute_fingerprint(
        selected.astype({"sentence_id": object})
    )


def test_checkpoint_restores_translations_without_an_api_call(selected):
    checkpoint = build_translation_checkpoint(
        selected,
        status="success",
        attempts=1,
        input_tokens=1000,
        output_tokens=200,
        raw_response=VALID_RESPONSE,
        error=None,
        attempt_history=[],
    )

    restored = restore_translations(selected, checkpoint)

    assert checkpoint["estimated_cost_usd"] == pytest.approx(0.0002)
    assert restored["translation"].tolist() == [
        "das ist gut",
        "der kommt später",
        "wir gehen heim",
    ]


def test_checkpoint_for_other_sources_is_rejected(selected):
    checkpoint = build_translation_checkpoint(
        selected,
        status="success",
        attempts=1,
        input_tokens=0,
        output_tokens=0,
        raw_response=VALID_RESPONSE,
        error=None,
        attempt_history=[],
    )
    changed = selected.assign(text=["anders", "de chunnt spöter", "mir gönd hei"])

    with pytest.raises(ValueError, match="does not match: source_fingerprint"):
        check_translation_checkpoint(changed, checkpoint)


def test_translation_retries_a_malformed_response_and_checkpoints_each_attempt(
    selected,
):
    client = FakeClient([make_response("1. nur eine"), make_response(VALID_RESPONSE)])
    checkpoints = []

    translated, report = translate_master(
        selected,
        client,
        max_api_attempts=2,
        sleeper=lambda seconds: None,
        save_checkpoint=checkpoints.append,
    )

    assert [checkpoint["status"] for checkpoint in checkpoints] == [
        "invalid",
        "success",
    ]
    assert report["attempts"] == 2
    assert report["input_tokens"] == 2000
    assert report["attempt_history"][0]["error"].startswith("Expected 3 numbered")
    assert translated["translation"].tolist()[0] == "das ist gut"
    assert translated["translation_attempts"].eq(2).all()


def test_translation_raises_after_the_last_malformed_response(selected):
    client = FakeClient([make_response(VALID_RESPONSE, stop_reason="max_tokens")])

    with pytest.raises(ValueError, match="non-success stop_reason: 'max_tokens'"):
        translate_master(selected, client, max_api_attempts=1)


def test_translation_resumes_attempt_count_from_an_invalid_checkpoint(selected):
    prior = build_translation_checkpoint(
        selected,
        status="invalid",
        attempts=3,
        input_tokens=500,
        output_tokens=50,
        raw_response="1. kaputt",
        error="Expected 3 numbered translations, received 1 lines.",
        attempt_history=[{"attempt": 3}],
    )

    _, report = translate_master(
        selected,
        FakeClient([make_response(VALID_RESPONSE)]),
        prior_checkpoint=prior,
    )

    assert report["attempts"] == 4
    assert report["input_tokens"] == 1500
    assert len(report["attempt_history"]) == 2
