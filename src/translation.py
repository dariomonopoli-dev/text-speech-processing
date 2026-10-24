"""Batch translation of the master sample into Standard German with Haiku."""

import hashlib
import json
import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

TRANSLATION_MODEL = "claude-haiku-5-5"
TRANSLATION_PROMPT_VERSION = "swiss-german-standard-few-shot-batch-v2"
HAIKU_MAX_OUTPUT_TOKENS = 128_000
HAIKU_INPUT_USD_PER_MILLION = 0.10
HAIKU_OUTPUT_USD_PER_MILLION = 0.50
RETRYABLE_ERROR_NAMES = frozenset(
    {"APIConnectionError", "APITimeoutError", "InternalServerError", "RateLimitError"}
)
RETRYABLE_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 504})
TERMINAL_TRANSLATION_STATUSES = {
    "success",
    "manual_review",
    "refusal",
    "malformed_response",
    "api_error",
}
AUDIT_COLUMNS = (
    "missing",
    "meta_commentary",
    "multiple_paragraphs",
    "length_ratio",
    "extreme_length",
    "has_anomaly",
)
TRANSLATION_COLUMNS = {
    "translation",
    "translation_status",
    "translation_attempts",
    "translation_input_tokens",
    "translation_output_tokens",
    "translation_estimated_cost_usd",
    "translation_error",
    *(f"translation_{column}" for column in AUDIT_COLUMNS),
}
META_COMMENTARY_PATTERN = re.compile(
    r"(?:^\s*(?:hinweis(?:\s+zur\s+übersetzung)?|übersetzung|translation|note)\s*:|"
    r"\b(?:let me correct|bitte prüfen|ich habe .* übersetzt)\b)",
    flags=re.IGNORECASE | re.DOTALL,
)

TRANSLATION_PROMPT = """Übersetze die folgenden Schweizerdeutschen Sätze ins Hochdeutsche. Behalte Bedeutung, Ton und alle Beleidigungen oder vulgären Ausdrücke wörtlich bei, schwäche nichts ab.

Beispiele:

Schweizerdeutsch: De Sportverein en Nüereschdoorf bietet während de Sommermönet Yogaüebige onter freiem Hemmel of de Wese ah.
Hochdeutsch: Der Sportverein in Nürensdorf bietet während der Sommermonate Yogaübungen unter freiem Himmel auf der Wiese an.

Schweizerdeutsch: In Deitige isch es üblich, die monatliche Fixköste für de "V-Pass bequem per Duuruftrag z regle.
Hochdeutsch: In Deitingen ist es üblich, die monatlichen Fixkosten für den ÖV-Pass bequem per Dauerauftrag zu regeln.

Schweizerdeutsch: D Omleitig förs kantonale Schötzefescht füehrt öber Hofdere ond Horb tüüf is ländliche Gebiet.
Hochdeutsch: Die Umleitung für das kantonale Schützenfest führt über Hochdorf und Horw tief in das ländliche Gebiet.

Schweizerdeutsch: Nach de Entgleisig vomne Dienstfahrzüg in Remismüli blibt d Strecki fürd Spuresicherig polizeilich gsperrt.
Hochdeutsch: Nach der Entgleisung eines Dienstfahrzeugs in Rämismühle bleibt die Strecke für die Spurensicherung polizeilich gesperrt.

Antworte nur mit den Übersetzungen, eine pro Zeile, mit derselben Nummer wie im Original (z.B. "1. ..."). Keine Erklärungen.

Sätze:"""


def compute_fingerprint(frame: pd.DataFrame) -> str:
    """Return an order-sensitive digest of a frame's schema and values."""
    digest = hashlib.sha256()
    schema = [(str(column), str(dtype)) for column, dtype in frame.dtypes.items()]
    digest.update(
        json.dumps(schema, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
    digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    return digest.hexdigest()


def compute_source_fingerprint(master: pd.DataFrame) -> str:
    return compute_fingerprint(
        master.loc[:, ["sentence_id", "text"]].reset_index(drop=True)
    )


def compute_translation_audits(
    source_texts: Iterable[str],
    translations: Iterable[str | None],
    min_length_ratio: float = 0.25,
    max_length_ratio: float = 3.0,
) -> pd.DataFrame:
    """Flag malformed translations without filtering words or social groups."""
    sources = list(source_texts)
    outputs = list(translations)
    if len(sources) != len(outputs):
        raise ValueError("source_texts and translations must have the same length.")
    if not 0 < min_length_ratio < max_length_ratio:
        raise ValueError("Length ratio bounds must satisfy 0 < min < max.")
    if not all(isinstance(source, str) and source.strip() for source in sources):
        raise ValueError("Every source text must be a non-empty string.")

    records = []
    for source, output in zip(sources, outputs):
        missing = output is None or pd.isna(output) or not str(output).strip()
        normalized_output = "" if missing else str(output).strip()
        length_ratio = (
            np.nan if missing else len(normalized_output) / len(source.strip())
        )
        meta_commentary = not missing and bool(
            META_COMMENTARY_PATTERN.search(normalized_output)
        )
        multiple_paragraphs = not missing and bool(
            re.search(r"\n\s*\n", normalized_output)
        )
        extreme_length = (
            not missing and not min_length_ratio <= length_ratio <= max_length_ratio
        )
        records.append(
            {
                "missing": missing,
                "meta_commentary": meta_commentary,
                "multiple_paragraphs": multiple_paragraphs,
                "length_ratio": length_ratio,
                "extreme_length": extreme_length,
                "has_anomaly": missing
                or meta_commentary
                or multiple_paragraphs
                or extreme_length,
            }
        )
    return pd.DataFrame.from_records(records, columns=AUDIT_COLUMNS)


def compute_translation_audit(source: str, translation: str | None) -> dict[str, Any]:
    audit = compute_translation_audits([source], [translation]).iloc[0].to_dict()
    ratio = audit["length_ratio"]
    audit["length_ratio"] = None if pd.isna(ratio) else float(ratio)
    return {
        key: bool(value) if key != "length_ratio" else value
        for key, value in audit.items()
    }


def to_translation_prompt(texts: Sequence[str]) -> str:
    """Build the approved few-shot prompt containing every selected sentence."""
    values = list(texts)
    if not values:
        raise ValueError("At least one translation text is required.")
    if not all(isinstance(text, str) and text.strip() for text in values):
        raise ValueError("Every translation text must be a non-empty string.")
    numbered = "\n".join(
        f"{number}. {' '.join(text.splitlines())}"
        for number, text in enumerate(values, start=1)
    )
    return f"{TRANSLATION_PROMPT}\n\n{numbered}"


def parse_translation_response(
    response_text: str | None, expected_count: int
) -> list[str]:
    """Validate and parse exactly one numbered translation per source row."""
    if not isinstance(response_text, str) or not response_text.strip():
        raise ValueError(
            f"Expected {expected_count} numbered translations, received no text."
        )
    lines = response_text.strip().splitlines()
    if len(lines) != expected_count:
        raise ValueError(
            f"Expected {expected_count} numbered translations, received "
            f"{len(lines)} lines."
        )
    translations: list[str] = []
    for expected_number, line in enumerate(lines, start=1):
        match = re.fullmatch(r"([0-9]+)\.\s+(.+)", line.strip())
        if match is None or match.group(1) != str(expected_number):
            raise ValueError(
                f"Expected {expected_count} numbered translations in order; "
                f"line {expected_number} is invalid."
            )
        translations.append(match.group(2).strip())
    return translations


def add_translations(
    master: pd.DataFrame, translations: Sequence[str], attempts: int
) -> pd.DataFrame:
    """Replace any previous translation columns with these translations and audits."""
    stale_columns = [column for column in TRANSLATION_COLUMNS if column in master]
    translated = master.drop(columns=stale_columns).copy()
    translated["translation"] = list(translations)
    audits = [
        compute_translation_audit(source, translation)
        for source, translation in zip(translated["text"], translations, strict=True)
    ]
    translated["translation_status"] = [
        "manual_review" if audit["has_anomaly"] else "success" for audit in audits
    ]
    translated["translation_attempts"] = attempts
    translated["translation_input_tokens"] = pd.NA
    translated["translation_output_tokens"] = pd.NA
    translated["translation_estimated_cost_usd"] = pd.NA
    translated["translation_error"] = None
    for audit_column in AUDIT_COLUMNS:
        translated[f"translation_{audit_column}"] = [
            audit[audit_column] for audit in audits
        ]
    return translated


def check_translation_checkpoint(
    master: pd.DataFrame, checkpoint: Mapping[str, Any]
) -> None:
    """Reject a checkpoint created for different sources or settings."""
    expected_metadata = {
        "translation_model": TRANSLATION_MODEL,
        "translation_prompt_version": TRANSLATION_PROMPT_VERSION,
        "row_count": int(len(master)),
        "source_fingerprint": compute_source_fingerprint(master),
    }
    mismatches = [
        key for key, value in expected_metadata.items() if checkpoint.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            "Batch translation checkpoint does not match: " + ", ".join(mismatches)
        )


def restore_translations(
    master: pd.DataFrame, checkpoint: Mapping[str, Any]
) -> pd.DataFrame:
    """Restore a validated batch response without paying for another API call."""
    check_translation_checkpoint(master, checkpoint)
    if checkpoint.get("status") != "success":
        raise ValueError("Batch translation checkpoint is not successful.")
    translations = parse_translation_response(
        checkpoint.get("raw_response"), len(master)
    )
    return add_translations(
        master, translations, attempts=int(checkpoint.get("attempts", 1))
    )


def build_translation_checkpoint(
    master: pd.DataFrame,
    *,
    status: str,
    attempts: int,
    input_tokens: int,
    output_tokens: int,
    raw_response: str | None,
    error: str | None,
    attempt_history: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "status": status,
        "translation_model": TRANSLATION_MODEL,
        "translation_prompt_version": TRANSLATION_PROMPT_VERSION,
        "row_count": int(len(master)),
        "attempts": attempts,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": (
            input_tokens * HAIKU_INPUT_USD_PER_MILLION
            + output_tokens * HAIKU_OUTPUT_USD_PER_MILLION
        )
        / 1_000_000,
        "source_fingerprint": compute_source_fingerprint(master),
        "raw_response": raw_response,
        "error": error,
        "attempt_history": list(attempt_history),
    }


def check_prior_checkpoint(
    master: pd.DataFrame, prior_checkpoint: Mapping[str, Any]
) -> None:
    check_translation_checkpoint(master, prior_checkpoint)
    if prior_checkpoint.get("status") == "success":
        raise ValueError(
            "Batch translation checkpoint is already successful; restore it."
        )
    if prior_checkpoint.get("status") != "invalid":
        raise ValueError("Prior batch translation checkpoint has invalid status.")
    for key in ("attempts", "input_tokens", "output_tokens"):
        value = prior_checkpoint.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Prior batch translation checkpoint has invalid {key}.")
    if not isinstance(prior_checkpoint.get("attempt_history"), list):
        raise ValueError(
            "Prior batch translation checkpoint has invalid attempt_history."
        )


def get_response_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def extract_response_text(response: Any) -> str | None:
    texts = []
    for block in get_response_value(response, "content", []) or []:
        if get_response_value(block, "type") == "text":
            text = get_response_value(block, "text")
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
    return "\n".join(texts) if texts else None


def get_response_tokens(response: Any) -> tuple[int, int]:
    usage = get_response_value(response, "usage", {})
    return (
        int(get_response_value(usage, "input_tokens", 0) or 0),
        int(get_response_value(usage, "output_tokens", 0) or 0),
    )


def is_retryable_api_error(error: BaseException) -> bool:
    status_code = getattr(error, "status_code", None)
    return (
        error.__class__.__name__ in RETRYABLE_ERROR_NAMES
        or status_code in RETRYABLE_STATUS_CODES
    )


def request_translation(client: Any, prompt: str) -> Any:
    with client.messages.stream(
        model=TRANSLATION_MODEL,
        max_tokens=HAIKU_MAX_OUTPUT_TOKENS,
        output_config={"effort": "medium"},
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        return stream.get_final_message()


def to_attempt_record(
    attempt: int,
    stop_reason: Any,
    input_tokens: int,
    output_tokens: int,
    response_text: str | None,
    parse_error: ValueError | None,
) -> dict[str, Any]:
    return {
        "attempt": attempt,
        "stop_reason": stop_reason,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "response_sha256": (
            hashlib.sha256(response_text.encode("utf-8")).hexdigest()
            if response_text is not None
            else None
        ),
        "response_characters": len(response_text) if response_text is not None else 0,
        "error": str(parse_error) if parse_error else None,
    }


def translate_master(
    master: pd.DataFrame,
    client: Any,
    max_api_attempts: int = 1,
    sleeper: Callable[[float], None] = time.sleep,
    prior_checkpoint: Mapping[str, Any] | None = None,
    save_checkpoint: Callable[[Mapping[str, Any]], None] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Translate the complete master sample in one strictly numbered request."""
    missing_columns = sorted({"sentence_id", "text"}.difference(master.columns))
    if missing_columns:
        raise KeyError("Missing selected columns: " + ", ".join(missing_columns))
    if master["sentence_id"].isna().any() or master["sentence_id"].duplicated().any():
        raise ValueError("sentence_id values must be unique and non-null.")
    if not isinstance(max_api_attempts, int) or max_api_attempts <= 0:
        raise ValueError("max_api_attempts must be a positive integer.")

    prompt = to_translation_prompt(master["text"].tolist())
    prior_checkpoint = dict(prior_checkpoint or {})
    if prior_checkpoint:
        check_prior_checkpoint(master, prior_checkpoint)
    last_error: Exception | None = None
    total_input_tokens = int(prior_checkpoint.get("input_tokens", 0))
    total_output_tokens = int(prior_checkpoint.get("output_tokens", 0))
    attempt_history = list(prior_checkpoint.get("attempt_history", []))
    translations: list[str] | None = None
    attempts = int(prior_checkpoint.get("attempts", 0))
    run_attempts = 0

    while run_attempts < max_api_attempts:
        run_attempts += 1
        attempts += 1
        try:
            response = request_translation(client, prompt)
        except Exception as error:
            last_error = error
            if not is_retryable_api_error(error) or run_attempts >= max_api_attempts:
                raise
            sleeper(float(2 ** (run_attempts - 1)))
            continue

        attempt_input_tokens, attempt_output_tokens = get_response_tokens(response)
        total_input_tokens += attempt_input_tokens
        total_output_tokens += attempt_output_tokens
        response_text = extract_response_text(response)
        stop_reason = get_response_value(response, "stop_reason")
        parse_error: ValueError | None = None
        try:
            if stop_reason != "end_turn":
                raise ValueError(
                    f"Batch response has non-success stop_reason: {stop_reason!r}."
                )
            translations = parse_translation_response(response_text, len(master))
        except ValueError as error:
            parse_error = error
            last_error = error
        attempt_history.append(
            to_attempt_record(
                attempts,
                stop_reason,
                attempt_input_tokens,
                attempt_output_tokens,
                response_text,
                parse_error,
            )
        )
        checkpoint = build_translation_checkpoint(
            master,
            status="invalid" if parse_error else "success",
            attempts=attempts,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            raw_response=response_text,
            error=str(parse_error) if parse_error else None,
            attempt_history=attempt_history,
        )
        if save_checkpoint is not None:
            save_checkpoint(checkpoint)
        if parse_error is not None:
            if run_attempts >= max_api_attempts:
                raise parse_error
            continue
        break

    if translations is None:
        raise RuntimeError(
            "Batch translation ended without valid output."
        ) from last_error
    return add_translations(master, translations, attempts), checkpoint
