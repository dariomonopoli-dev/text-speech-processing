"""Command-line orchestration for the SwissCrawl stratified test set."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from austrotox_sampling import FIXED_STRATUM_TARGETS
from swisscrawl_pipeline import (
    build_annotation_frames,
    build_gahd_classifier,
    score_candidate_pool,
    select_master_sample,
    restore_batch_translation,
    translate_selected_batch,
    validate_batch_translation_checkpoint,
    validate_score_stratum_alignment,
    verify_artifacts,
)


DEFAULT_CANDIDATE = Path("data/swisscrawl_filtered.csv")
DEFAULT_SCORED = Path("data/swisscrawl_scored_gahd.csv")
DEFAULT_MASTER = Path("data/master_sample_stratified.csv")
DEFAULT_SAMPLING_REPORT = Path("data/stratified_sampling_report.json")
DEFAULT_VERIFICATION_REPORT = Path("data/stratified_verification_report.json")
DEFAULT_TRANSLATION_CHECKPOINT = Path("data/checkpoints/haiku_batch_translation.json")
ANNOTATORS = ("dario", "marjorie", "yannis")


class PipelineCliError(RuntimeError):
    pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the 700-row SwissCrawl stratified test set."
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        metavar="{score,sample,translate,annotations,verify,all}",
    )
    for command in ("score", "sample", "translate", "annotations", "verify", "all"):
        subparser = subparsers.add_parser(command)
        _add_common_arguments(subparser)
    return parser


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--scored", type=Path, default=DEFAULT_SCORED)
    parser.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--sampling-report", type=Path, default=DEFAULT_SAMPLING_REPORT)
    parser.add_argument(
        "--verification-report", type=Path, default=DEFAULT_VERIFICATION_REPORT
    )
    parser.add_argument(
        "--translation-checkpoint",
        type=Path,
        default=DEFAULT_TRANSLATION_CHECKPOINT,
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("cpu", "mps"))
    parser.add_argument("--max-api-attempts", type=int, default=5)
    parser.add_argument("--resume", action="store_true")


def _annotation_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        annotator: args.output_dir / f"annotation_{annotator}_stratified.csv"
        for annotator in ANNOTATORS
    }


def _annotation_workbook_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        annotator: path.with_suffix(".xlsx")
        for annotator, path in _annotation_paths(args).items()
    }


def _require_files(paths: Sequence[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise PipelineCliError("Required input does not exist: " + ", ".join(missing))


def _protect_outputs(paths: Sequence[Path], resume: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not resume:
        raise PipelineCliError(
            "Output already exists; pass --resume to reuse or replace it: "
            + ", ".join(existing)
        )


def _preflight(args: argparse.Namespace, environ: Mapping[str, str]) -> None:
    command = args.command
    if command in {"score", "all"}:
        _require_files([args.candidate])
    elif command == "sample":
        _require_files([args.scored])
    elif command == "translate":
        _require_files([args.master])
    elif command == "annotations":
        _require_files([args.master])
    elif command == "verify":
        _require_files([args.master, *_annotation_paths(args).values()])

    reusable_translation = False
    if (
        command in {"translate", "all"}
        and args.master.is_file()
        and args.translation_checkpoint.is_file()
    ):
        try:
            with args.translation_checkpoint.open(encoding="utf-8") as handle:
                checkpoint = json.load(handle)
            if checkpoint.get("status") == "success":
                master = _read_csv(args.master)
                validate_batch_translation_checkpoint(master, checkpoint)
                reusable_translation = True
        except (OSError, ValueError, TypeError, KeyError):
            reusable_translation = False
    requires_api_key = command in {"translate", "all"} and not reusable_translation
    if requires_api_key and not environ.get("ANTHROPIC_API_KEY"):
        raise PipelineCliError(
            "ANTHROPIC_API_KEY is required for the translate and all stages."
        )

    if command == "score":
        _protect_outputs([args.scored], args.resume)
    elif command == "sample":
        _protect_outputs([args.master, args.sampling_report], args.resume)
    elif command == "translate" and not args.resume:
        header = pd.read_csv(args.master, nrows=0)
        if "translation" in header.columns or args.translation_checkpoint.exists():
            raise PipelineCliError(
                "Translated master output or batch checkpoint already exists; "
                "pass --resume to reuse it."
            )
    elif command == "annotations":
        _protect_outputs(
            [
                *_annotation_paths(args).values(),
                *_annotation_workbook_paths(args).values(),
            ],
            args.resume,
        )
    elif command == "verify":
        _protect_outputs([args.verification_report], args.resume)
    elif command == "all":
        _protect_outputs(
            [
                args.scored,
                args.master,
                args.sampling_report,
                args.verification_report,
                args.translation_checkpoint,
                *_annotation_paths(args).values(),
                *_annotation_workbook_paths(args).values(),
            ],
            args.resume,
        )

    for name in ("batch_size", "max_api_attempts"):
        if getattr(args, name) <= 0:
            raise PipelineCliError(f"--{name.replace('_', '-')} must be positive.")


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"sentence_id": "string"})


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.replace(path)


def _validated_scored_output(candidate: pd.DataFrame, path: Path) -> pd.DataFrame:
    scored = _read_csv(path)
    required = {"sentence_id", "text", "hate_score"}
    if not required.issubset(scored.columns) or not scored[
        ["sentence_id", "text"]
    ].equals(candidate[["sentence_id", "text"]]):
        raise PipelineCliError(
            "Existing scored output does not match the candidate corpus."
        )
    scores = pd.to_numeric(scored["hate_score"], errors="coerce")
    if scores.isna().any() or not scores.between(0.0, 1.0).all():
        raise PipelineCliError("Existing scored output contains invalid scores.")
    return scored


def _write_xlsx(frame: pd.DataFrame, path: Path) -> None:
    from openpyxl import Workbook

    path.parent.mkdir(parents=True, exist_ok=True)
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "annotations"
    sheet.append(frame.columns.tolist())
    for values in frame.itertuples(index=False, name=None):
        sheet.append([None if pd.isna(value) else value for value in values])
    for row in sheet.iter_rows():
        for cell in row:
            if isinstance(cell.value, str):
                cell.data_type = "s"
    workbook.save(path)


def _run_score(args: argparse.Namespace) -> None:
    candidate = _read_csv(args.candidate)
    if args.resume and args.scored.is_file():
        scored = _validated_scored_output(candidate, args.scored)
        print(f"score: reused {len(scored)} rows from {args.scored}")
        return
    classifier = build_gahd_classifier(args.device)
    scored = score_candidate_pool(
        candidate,
        classifier,
        batch_size=args.batch_size,
    )
    _write_csv(scored, args.scored)
    print(f"score: wrote {len(scored)} rows to {args.scored}")


def _run_sample(args: argparse.Namespace) -> None:
    scored = _read_csv(args.scored)
    master, report = select_master_sample(scored, random_state=42)
    _write_csv(master, args.master)
    _write_json(report, args.sampling_report)
    print(f"sample: wrote {len(master)} rows to {args.master}")


def _run_translate(args: argparse.Namespace, api_key: str) -> None:
    master = _read_csv(args.master)
    _validate_translation_master(master)
    prior_report = None
    if args.translation_checkpoint.is_file():
        try:
            with args.translation_checkpoint.open(encoding="utf-8") as handle:
                prior_report = json.load(handle)
            validate_batch_translation_checkpoint(master, prior_report)
        except (OSError, ValueError, TypeError, KeyError) as error:
            raise PipelineCliError(
                "Translation checkpoint is invalid or belongs to another sample: "
                f"{error}"
            ) from error
        if prior_report.get("status") == "success":
            translated = restore_batch_translation(master, prior_report)
            _write_csv(translated, args.master)
            print(
                f"translate: wrote {len(translated)} rows to {args.master} "
                "(reused batch checkpoint)"
            )
            return

    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    translated, report = translate_selected_batch(
        master,
        client,
        max_api_attempts=args.max_api_attempts,
        prior_report=prior_report,
        checkpoint_callback=lambda checkpoint: _write_json(
            checkpoint, args.translation_checkpoint
        ),
    )
    _write_json(report, args.translation_checkpoint)
    source = "one Haiku batch request"
    _write_csv(translated, args.master)
    print(f"translate: wrote {len(translated)} rows to {args.master} " f"({source})")


def _validate_translation_master(master: pd.DataFrame) -> None:
    required = {"sentence_id", "text", "hate_score", "austrotox_stratum"}
    missing = sorted(required.difference(master.columns))
    if missing:
        raise PipelineCliError(
            "Translation master is missing columns: " + ", ".join(missing)
        )
    if len(master) != 700:
        raise PipelineCliError(
            f"The batch translation stage requires exactly 700 rows, got {len(master)}."
        )
    if master["sentence_id"].isna().any() or master["sentence_id"].duplicated().any():
        raise PipelineCliError("Translation master sentence_id values must be unique.")
    if (
        not master["text"]
        .map(lambda value: isinstance(value, str) and bool(value.strip()))
        .all()
    ):
        raise PipelineCliError("Translation master contains empty source text.")
    scores = pd.to_numeric(master["hate_score"], errors="coerce")
    if scores.isna().any() or not scores.between(0.0, 1.0).all():
        raise PipelineCliError("Translation master contains invalid hate_score values.")
    try:
        validate_score_stratum_alignment(master)
    except (KeyError, ValueError) as error:
        raise PipelineCliError(str(error)) from error
    actual = master["austrotox_stratum"].value_counts().to_dict()
    expected = dict(FIXED_STRATUM_TARGETS)
    if actual != expected:
        raise PipelineCliError(
            f"Translation master stratum counts are {actual}, expected {expected}."
        )


def _run_annotations(args: argparse.Namespace) -> None:
    master = _read_csv(args.master)
    annotations = build_annotation_frames(master, random_state=42)
    for annotator, frame in annotations.items():
        path = _annotation_paths(args)[annotator]
        if path.exists():
            print(f"annotations: preserved existing {path}")
        else:
            _write_csv(frame, path)
            print(f"annotations: wrote {len(frame)} rows to {path}")
        workbook_path = _annotation_workbook_paths(args)[annotator]
        if workbook_path.exists():
            print(f"annotations: preserved existing {workbook_path}")
        else:
            _write_xlsx(frame, workbook_path)
            print(f"annotations: wrote {len(frame)} rows to {workbook_path}")


def _run_verify(args: argparse.Namespace) -> None:
    master = _read_csv(args.master)
    annotations = {
        annotator: _read_csv(path)
        for annotator, path in _annotation_paths(args).items()
    }
    report = verify_artifacts(master, annotations)
    _write_json(report, args.verification_report)
    print(f"verify: passed and wrote {args.verification_report}")


def _run_all(args: argparse.Namespace, api_key: str) -> None:
    _run_score(args)
    _run_sample(args)
    _run_translate(args, api_key)
    _run_annotations(args)
    _run_verify(args)


def _default_handlers(
    environ: Mapping[str, str],
) -> dict[str, Callable[[argparse.Namespace], None]]:
    api_key = environ.get("ANTHROPIC_API_KEY", "")
    return {
        "score": _run_score,
        "sample": _run_sample,
        "translate": lambda args: _run_translate(args, api_key),
        "annotations": _run_annotations,
        "verify": _run_verify,
        "all": lambda args: _run_all(args, api_key),
    }


def main(
    argv: Sequence[str] | None = None,
    *,
    stage_handlers: Mapping[str, Callable[[argparse.Namespace], Any]] | None = None,
    environ: Mapping[str, str] | None = None,
) -> int:
    if environ is None:
        load_dotenv(override=False)
        environ = os.environ
    args = build_parser().parse_args(argv)
    try:
        _preflight(args, environ)
        handlers = dict(stage_handlers or _default_handlers(environ))
        handler = handlers.get(args.command)
        if handler is None:
            raise PipelineCliError(f"No handler configured for stage {args.command}.")
        handler(args)
    except PipelineCliError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
