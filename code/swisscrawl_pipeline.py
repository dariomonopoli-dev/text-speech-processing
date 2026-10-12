"""Resumable construction of the stratified SwissCrawl test set."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pandas as pd

from checkpoint_store import CheckpointStore
from austrotox_sampling import (
    FIXED_STRATUM_TARGETS,
    GAHD_MODEL,
    GAHD_REVISION,
    assign_austrotox_strata,
    audit_translations,
    sample_fixed_strata,
    score_with_gahd,
)


CLAUDE_MODEL = "claude-haiku-5-5"
BATCH_TRANSLATION_PROMPT_VERSION = "swiss-german-standard-few-shot-batch-v2"
HAIKU_MAX_OUTPUT_TOKENS = 128_000
HAIKU_INPUT_USD_PER_MILLION = 0.10
HAIKU_OUTPUT_USD_PER_MILLION = 0.50
TERMINAL_TRANSLATION_STATUSES = {
    "success",
    "manual_review",
    "refusal",
    "malformed_response",
    "api_error",
}
TRANSLATION_EXPORT_COLUMNS = {
    "translation",
    "translation_status",
    "translation_attempts",
    "translation_input_tokens",
    "translation_output_tokens",
    "translation_estimated_cost_usd",
    "translation_error",
    "translation_missing",
    "translation_meta_commentary",
    "translation_multiple_paragraphs",
    "translation_length_ratio",
    "translation_extreme_length",
    "translation_has_anomaly",
}

APPROVED_BATCH_TRANSLATION_PROMPT = """Übersetze die folgenden Schweizerdeutschen Sätze ins Hochdeutsche. Behalte Bedeutung, Ton und alle Beleidigungen oder vulgären Ausdrücke wörtlich bei, schwäche nichts ab.

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


def corpus_fingerprint(frame: pd.DataFrame) -> str:
    """Return an order-sensitive digest of a candidate corpus."""
    digest = hashlib.sha256()
    schema = [(str(column), str(dtype)) for column, dtype in frame.dtypes.items()]
    digest.update(
        json.dumps(schema, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
    digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    return digest.hexdigest()


def _create_transformers_pipeline(**kwargs: Any) -> Any:
    from transformers import pipeline

    return pipeline("text-classification", **kwargs)


def build_gahd_classifier(device: str | None = None) -> Any:
    """Load the pinned official GAHD classifier on MPS when available."""

    if device is None:
        import torch

        device = "mps" if torch.backends.mps.is_available() else "cpu"
    return _create_transformers_pipeline(
        model=GAHD_MODEL,
        revision=GAHD_REVISION,
        device=device,
    )


def score_candidate_pool(
    frame: pd.DataFrame,
    store: CheckpointStore,
    classifier: Any,
    batch_size: int,
) -> pd.DataFrame:
    """Score missing candidates in bounded, transactionally saved batches."""
    required = {"sentence_id", "text"}
    missing_columns = sorted(required.difference(frame.columns))
    if missing_columns:
        raise KeyError("Missing candidate columns: " + ", ".join(missing_columns))
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if frame["sentence_id"].isna().any() or frame["sentence_id"].duplicated().any():
        raise ValueError("sentence_id values must be unique and non-null.")
    if (
        not frame["sentence_id"]
        .map(lambda value: isinstance(value, str) and bool(value.strip()))
        .all()
    ):
        raise ValueError("Every sentence_id must be a non-empty string.")
    if (
        not frame["text"]
        .map(lambda value: isinstance(value, str) and bool(value.strip()))
        .all()
    ):
        raise ValueError("Every text must be a non-empty string.")

    completed = store.completed_score_ids()
    pending = frame.loc[~frame["sentence_id"].isin(completed), ["sentence_id", "text"]]
    for start in range(0, len(pending), batch_size):
        batch = pending.iloc[start : start + batch_size]
        scores = score_with_gahd(
            batch["text"].tolist(),
            classifier=classifier,
            batch_size=batch_size,
        )
        store.save_scores(
            {
                "sentence_id": sentence_id,
                "hate_score": score,
            }
            for sentence_id, score in zip(batch["sentence_id"], scores)
        )

    score_by_id = store.load_scores()
    result = frame.copy()
    result["hate_score"] = result["sentence_id"].map(score_by_id)
    if result["hate_score"].isna().any():
        missing_ids = result.loc[result["hate_score"].isna(), "sentence_id"].tolist()
        raise RuntimeError(
            "Checkpoint is missing scores after scoring: " + ", ".join(missing_ids[:10])
        )
    return result


def select_master_sample(
    scored: pd.DataFrame,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select the fixed 700-row sample and return its diagnostic report."""
    if "hate_score" not in scored:
        raise KeyError("Missing score column: hate_score")
    source = scored.copy()
    source["austrotox_stratum"] = assign_austrotox_strata(
        source["hate_score"]
    ).to_numpy()
    sample = sample_fixed_strata(scored, random_state=random_state)

    source_counts = source["austrotox_stratum"].value_counts(sort=False)
    selected_counts = sample["austrotox_stratum"].value_counts(sort=False)
    score_summaries = {}
    for stratum in FIXED_STRATUM_TARGETS:
        scores = source.loc[source["austrotox_stratum"] == stratum, "hate_score"]
        score_summaries[stratum] = {
            "min": float(scores.min()),
            "median": float(scores.median()),
            "max": float(scores.max()),
        }

    report = {
        "method": "AustroTox-inspired project-specific fixed allocation",
        "sample_size": int(len(sample)),
        "random_seed": int(random_state),
        "score_model": GAHD_MODEL,
        "score_revision": GAHD_REVISION,
        "source_counts": {
            stratum: int(source_counts.get(stratum, 0))
            for stratum in FIXED_STRATUM_TARGETS
        },
        "target_counts": dict(FIXED_STRATUM_TARGETS),
        "selected_counts": {
            stratum: int(selected_counts.get(stratum, 0))
            for stratum in FIXED_STRATUM_TARGETS
        },
        "score_summaries": score_summaries,
    }
    return sample, report


def validate_score_stratum_alignment(frame: pd.DataFrame) -> None:
    """Require every stored stratum to agree with its GAHD probability."""
    required = {"hate_score", "austrotox_stratum"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError("Missing score or stratum columns: " + ", ".join(missing))
    scores = pd.to_numeric(frame["hate_score"], errors="coerce")
    if scores.isna().any() or not scores.between(0.0, 1.0).all():
        raise ValueError("hate_score values must be probabilities in [0, 1].")
    expected = assign_austrotox_strata(scores).astype("string")
    actual = frame["austrotox_stratum"].astype("string")
    if not actual.reset_index(drop=True).equals(expected.reset_index(drop=True)):
        raise ValueError("austrotox_stratum does not match hate_score for every row.")


def build_batch_translation_prompt(texts: Sequence[str]) -> str:
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
    return f"{APPROVED_BATCH_TRANSLATION_PROMPT}\n\n{numbered}"


def _parse_batch_translation_response(
    response_text: str | None,
    expected_count: int,
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


def _merge_batch_translations(
    frame: pd.DataFrame,
    translations: Sequence[str],
    attempts: int,
) -> pd.DataFrame:
    result = frame.drop(
        columns=[
            column for column in TRANSLATION_EXPORT_COLUMNS if column in frame.columns
        ]
    ).copy()
    result["translation"] = list(translations)
    audits = [
        _audit_translation(source, translation)
        for source, translation in zip(result["text"], translations, strict=True)
    ]
    result["translation_status"] = [
        "manual_review" if audit["has_anomaly"] else "success" for audit in audits
    ]
    result["translation_attempts"] = attempts
    result["translation_input_tokens"] = pd.NA
    result["translation_output_tokens"] = pd.NA
    result["translation_estimated_cost_usd"] = pd.NA
    result["translation_error"] = None
    for output_column, audit_key in (
        ("translation_missing", "missing"),
        ("translation_meta_commentary", "meta_commentary"),
        ("translation_multiple_paragraphs", "multiple_paragraphs"),
        ("translation_length_ratio", "length_ratio"),
        ("translation_extreme_length", "extreme_length"),
        ("translation_has_anomaly", "has_anomaly"),
    ):
        result[output_column] = [audit[audit_key] for audit in audits]
    return result


def validate_batch_translation_checkpoint(
    frame: pd.DataFrame,
    report: Mapping[str, Any],
) -> None:
    """Reject a batch checkpoint created for different sources or settings."""
    expected_fingerprint = corpus_fingerprint(
        frame.loc[:, ["sentence_id", "text"]].reset_index(drop=True)
    )
    expected_metadata = {
        "translation_model": CLAUDE_MODEL,
        "translation_prompt_version": BATCH_TRANSLATION_PROMPT_VERSION,
        "row_count": int(len(frame)),
        "source_fingerprint": expected_fingerprint,
    }
    mismatches = [
        key for key, value in expected_metadata.items() if report.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            "Batch translation checkpoint does not match: " + ", ".join(mismatches)
        )


def restore_batch_translation(
    frame: pd.DataFrame,
    report: Mapping[str, Any],
) -> pd.DataFrame:
    """Restore a validated batch response without paying for another API call."""
    validate_batch_translation_checkpoint(frame, report)
    if report.get("status") != "success":
        raise ValueError("Batch translation checkpoint is not successful.")
    translations = _parse_batch_translation_response(
        report.get("raw_response"), len(frame)
    )
    return _merge_batch_translations(
        frame,
        translations,
        attempts=int(report.get("attempts", 1)),
    )


def _batch_translation_report(
    frame: pd.DataFrame,
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
        "translation_model": CLAUDE_MODEL,
        "translation_prompt_version": BATCH_TRANSLATION_PROMPT_VERSION,
        "row_count": int(len(frame)),
        "attempts": attempts,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": (
            input_tokens * HAIKU_INPUT_USD_PER_MILLION
            + output_tokens * HAIKU_OUTPUT_USD_PER_MILLION
        )
        / 1_000_000,
        "source_fingerprint": corpus_fingerprint(
            frame.loc[:, ["sentence_id", "text"]].reset_index(drop=True)
        ),
        "raw_response": raw_response,
        "error": error,
        "attempt_history": list(attempt_history),
    }


def translate_selected_batch(
    frame: pd.DataFrame,
    client: Any,
    max_api_attempts: int = 1,
    sleeper: Any = time.sleep,
    prior_report: Mapping[str, Any] | None = None,
    checkpoint_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Translate the complete selected set in one strictly numbered request."""
    required = {"sentence_id", "text"}
    missing_columns = sorted(required.difference(frame.columns))
    if missing_columns:
        raise KeyError("Missing selected columns: " + ", ".join(missing_columns))
    if frame["sentence_id"].isna().any() or frame["sentence_id"].duplicated().any():
        raise ValueError("sentence_id values must be unique and non-null.")
    if not isinstance(max_api_attempts, int) or max_api_attempts <= 0:
        raise ValueError("max_api_attempts must be a positive integer.")

    prompt = build_batch_translation_prompt(frame["text"].tolist())
    prior_report = dict(prior_report or {})
    if prior_report:
        validate_batch_translation_checkpoint(frame, prior_report)
        if prior_report.get("status") == "success":
            raise ValueError(
                "Batch translation checkpoint is already successful; restore it."
            )
        if prior_report.get("status") != "invalid":
            raise ValueError("Prior batch translation checkpoint has invalid status.")
        for key in ("attempts", "input_tokens", "output_tokens"):
            value = prior_report.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"Prior batch translation checkpoint has invalid {key}."
                )
        if not isinstance(prior_report.get("attempt_history"), list):
            raise ValueError(
                "Prior batch translation checkpoint has invalid attempt_history."
            )
    last_error: Exception | None = None
    total_input_tokens = int(prior_report.get("input_tokens", 0))
    total_output_tokens = int(prior_report.get("output_tokens", 0))
    attempt_history = list(prior_report.get("attempt_history", []))
    response_text: str | None = None
    translations: list[str] | None = None
    attempts = int(prior_report.get("attempts", 0))
    run_attempts = 0

    while run_attempts < max_api_attempts:
        run_attempts += 1
        attempts += 1
        try:
            with client.messages.stream(
                model=CLAUDE_MODEL,
                max_tokens=HAIKU_MAX_OUTPUT_TOKENS,
                output_config={"effort": "medium"},
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                response = stream.get_final_message()
        except Exception as error:
            last_error = error
            if not _is_retryable_api_error(error) or run_attempts >= max_api_attempts:
                raise
            sleeper(float(2 ** (run_attempts - 1)))
            continue

        attempt_input_tokens, attempt_output_tokens = _response_tokens(response)
        total_input_tokens += attempt_input_tokens
        total_output_tokens += attempt_output_tokens
        response_text = _extract_response_text(response)
        stop_reason = _response_value(response, "stop_reason")
        parse_error: ValueError | None = None
        try:
            if stop_reason != "end_turn":
                raise ValueError(
                    "Batch response has non-success stop_reason: " f"{stop_reason!r}."
                )
            translations = _parse_batch_translation_response(response_text, len(frame))
        except ValueError as error:
            parse_error = error
            last_error = error
        attempt_history.append(
            {
                "attempt": attempts,
                "stop_reason": stop_reason,
                "input_tokens": attempt_input_tokens,
                "output_tokens": attempt_output_tokens,
                "response_sha256": (
                    hashlib.sha256(response_text.encode("utf-8")).hexdigest()
                    if response_text is not None
                    else None
                ),
                "response_characters": (
                    len(response_text) if response_text is not None else 0
                ),
                "error": str(parse_error) if parse_error else None,
            }
        )
        checkpoint = _batch_translation_report(
            frame,
            status="invalid" if parse_error else "success",
            attempts=attempts,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            raw_response=response_text,
            error=str(parse_error) if parse_error else None,
            attempt_history=attempt_history,
        )
        if checkpoint_callback is not None:
            checkpoint_callback(checkpoint)
        if parse_error is not None:
            if run_attempts >= max_api_attempts:
                raise parse_error
            continue
        break

    if translations is None:
        raise RuntimeError(
            "Batch translation ended without valid output."
        ) from last_error

    result = _merge_batch_translations(frame, translations, attempts)

    return result, checkpoint


def _response_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _extract_response_text(response: Any) -> str | None:
    texts = []
    for block in _response_value(response, "content", []) or []:
        if _response_value(block, "type") == "text":
            text = _response_value(block, "text")
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
    return "\n".join(texts) if texts else None


def _response_tokens(response: Any) -> tuple[int, int]:
    usage = _response_value(response, "usage", {})
    return (
        int(_response_value(usage, "input_tokens", 0) or 0),
        int(_response_value(usage, "output_tokens", 0) or 0),
    )


def _is_retryable_api_error(error: BaseException) -> bool:
    retryable_names = {
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "RateLimitError",
    }
    status_code = getattr(error, "status_code", None)
    return error.__class__.__name__ in retryable_names or status_code in {
        408,
        409,
        429,
        500,
        502,
        503,
        504,
    }


def _audit_translation(source: str, response: str | None) -> dict[str, Any]:
    audit = audit_translations([source], [response]).iloc[0].to_dict()
    ratio = audit["length_ratio"]
    audit["length_ratio"] = None if pd.isna(ratio) else float(ratio)
    return {
        key: bool(value) if key != "length_ratio" else value
        for key, value in audit.items()
    }


def build_annotation_frames(
    master: pd.DataFrame,
    annotators: Sequence[str] = ("dario", "marjorie", "yannis"),
    shared_count: int = 100,
    random_state: int = 42,
) -> dict[str, pd.DataFrame]:
    """Build deterministic shared and private annotation assignments."""
    required = {"sentence_id", "text", "translation_status"}
    missing_columns = sorted(required.difference(master.columns))
    if missing_columns:
        raise KeyError("Missing master columns: " + ", ".join(missing_columns))
    names = tuple(annotators)
    if not names or len(set(names)) != len(names):
        raise ValueError("Annotator names must be non-empty and unique.")
    if master["sentence_id"].isna().any() or master["sentence_id"].duplicated().any():
        raise ValueError("master sentence_id values must be unique and non-null.")
    nonterminal = ~master["translation_status"].isin(TERMINAL_TRANSLATION_STATUSES)
    if nonterminal.any():
        raise ValueError("Every master row must have a terminal translation status.")
    if not isinstance(shared_count, int) or shared_count < 0:
        raise ValueError("shared_count must be a non-negative integer.")
    remaining_count = len(master) - shared_count
    if remaining_count < 0 or remaining_count % len(names):
        raise ValueError(
            "Rows remaining after the shared overlap must divide evenly among annotators."
        )

    shuffled = (
        master.loc[:, ["sentence_id", "text"]]
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )
    shared = shuffled.iloc[:shared_count]
    remaining = shuffled.iloc[shared_count:]
    private_count = remaining_count // len(names)
    assignments = {}
    for index, annotator in enumerate(names):
        private = remaining.iloc[index * private_count : (index + 1) * private_count]
        assignment = pd.concat([shared, private], ignore_index=True).sample(
            frac=1,
            random_state=random_state + index,
        )
        assignment = assignment.reset_index(drop=True)
        assignment["label"] = ""
        assignment["comment"] = ""
        assignments[annotator] = assignment[["sentence_id", "text", "label", "comment"]]
    return assignments


def verify_artifacts(
    master: pd.DataFrame,
    annotations: Mapping[str, pd.DataFrame],
) -> dict[str, Any]:
    """Validate the final sample and leakage-free annotation assignments."""
    required_master = {
        "sentence_id",
        "text",
        "hate_score",
        "austrotox_stratum",
        *TRANSLATION_EXPORT_COLUMNS,
    }
    missing_master = sorted(required_master.difference(master.columns))
    if missing_master:
        raise ValueError(
            "master translation columns or required fields are missing: "
            + ", ".join(missing_master)
        )
    if len(master) != sum(FIXED_STRATUM_TARGETS.values()):
        raise ValueError("master must contain exactly 700 rows.")
    if master["sentence_id"].isna().any() or master["sentence_id"].duplicated().any():
        raise ValueError("master sentence_id values must be unique and non-null.")
    nonterminal = ~master["translation_status"].isin(TERMINAL_TRANSLATION_STATUSES)
    if nonterminal.any():
        raise ValueError("Every master row must have a terminal translation status.")

    stratum_counts = master["austrotox_stratum"].value_counts(sort=False).to_dict()
    normalized_counts = {
        stratum: int(stratum_counts.get(stratum, 0))
        for stratum in FIXED_STRATUM_TARGETS
    }
    if normalized_counts != FIXED_STRATUM_TARGETS:
        raise ValueError(
            f"master stratum counts do not match targets: {normalized_counts}"
        )
    validate_score_stratum_alignment(master)
    if len(annotations) != 3:
        raise ValueError("Exactly three annotation frames are required.")

    expected_columns = ["sentence_id", "text", "label", "comment"]
    master_ids = set(master["sentence_id"])
    master_text = master.set_index("sentence_id")["text"]
    id_sets = {}
    for annotator, frame in annotations.items():
        if frame.columns.tolist() != expected_columns:
            raise ValueError(
                f"{annotator} annotation columns must be exactly {expected_columns}."
            )
        if len(frame) != 300:
            raise ValueError(f"{annotator} must have exactly 300 annotation rows.")
        if frame["sentence_id"].isna().any() or frame["sentence_id"].duplicated().any():
            raise ValueError(
                f"{annotator} annotation sentence_id values must be unique."
            )
        ids = set(frame["sentence_id"])
        if not ids.issubset(master_ids):
            raise ValueError(f"{annotator} contains IDs outside the master sample.")
        expected_text = frame["sentence_id"].map(master_text)
        if not frame["text"].equals(expected_text):
            raise ValueError(
                f"{annotator} does not preserve the original Swiss German text."
            )
        for column in ("label", "comment"):
            if not frame[column].fillna("").eq("").all():
                raise ValueError(f"{annotator} {column} values must be empty.")
        id_sets[annotator] = ids

    shared = set.intersection(*id_sets.values())
    if len(shared) != 100:
        raise ValueError(
            f"shared overlap must contain exactly 100 IDs, got {len(shared)}."
        )
    private = {name: ids - shared for name, ids in id_sets.items()}
    if any(len(ids) != 200 for ids in private.values()):
        raise ValueError("Every private assignment must contain exactly 200 IDs.")
    names = list(private)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            if private[left] & private[right]:
                raise ValueError(
                    f"private assignments overlap between {left} and {right}."
                )
    if set.union(*id_sets.values()) != master_ids:
        raise ValueError(
            "Annotation assignments do not cover the complete master sample."
        )

    unresolved = master["translation_has_anomaly"].fillna(True).astype(bool) | master[
        "translation_status"
    ].ne("success")
    unresolved_ids = master.loc[unresolved, "sentence_id"].astype(str).tolist()
    status_counts = {
        str(status): int(count)
        for status, count in master["translation_status"]
        .fillna("missing")
        .value_counts()
        .items()
    }
    return {
        "status": "passed",
        "master_rows": int(len(master)),
        "stratum_counts": normalized_counts,
        "annotator_rows": {
            name: int(len(frame)) for name, frame in annotations.items()
        },
        "shared_count": int(len(shared)),
        "private_counts": {name: int(len(ids)) for name, ids in private.items()},
        "translation_status_counts": status_counts,
        "unresolved_translation_anomaly_count": len(unresolved_ids),
        "unresolved_translation_anomaly_ids": unresolved_ids,
    }
