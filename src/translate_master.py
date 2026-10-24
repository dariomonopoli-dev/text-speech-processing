"""Translate the master sample into Standard German in one Haiku batch request."""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from get_loggers import get_logger
from pipeline_io import (
    PipelineError,
    check_inputs_exist,
    check_positive,
    load_csv,
    load_json,
    write_csv,
    write_json,
)
from sampling import FIXED_STRATUM_TARGETS, check_strata_match_scores
from translation import (
    check_translation_checkpoint,
    restore_translations,
    translate_master,
)

logger = get_logger("translate")

CHECKPOINT_ERRORS = (OSError, ValueError, TypeError, KeyError)


def has_reusable_checkpoint(path_master: Path, path_checkpoint: Path) -> bool:
    if not (path_master.is_file() and path_checkpoint.is_file()):
        return False
    try:
        checkpoint = load_json(path_checkpoint)
        if checkpoint.get("status") != "success":
            return False
        check_translation_checkpoint(load_csv(path_master), checkpoint)
    except CHECKPOINT_ERRORS as error:
        logger.warning(f"Translation checkpoint is not reusable: {error}")
        return False
    return True


def check_translation_master(master: pd.DataFrame) -> None:
    required = {"sentence_id", "text", "hate_score", "austrotox_stratum"}
    missing = sorted(required.difference(master.columns))
    if missing:
        raise PipelineError(
            "Translation master is missing columns: " + ", ".join(missing)
        )
    if len(master) != 700:
        raise PipelineError(
            f"The batch translation stage requires exactly 700 rows, got {len(master)}."
        )
    if master["sentence_id"].isna().any() or master["sentence_id"].duplicated().any():
        raise PipelineError("Translation master sentence_id values must be unique.")
    if (
        not master["text"]
        .map(lambda value: isinstance(value, str) and bool(value.strip()))
        .all()
    ):
        raise PipelineError("Translation master contains empty source text.")
    hate_scores = pd.to_numeric(master["hate_score"], errors="coerce")
    if hate_scores.isna().any() or not hate_scores.between(0.0, 1.0).all():
        raise PipelineError("Translation master contains invalid hate_score values.")
    try:
        check_strata_match_scores(master)
    except (KeyError, ValueError) as error:
        raise PipelineError(str(error)) from error
    stratum_counts = master["austrotox_stratum"].value_counts().to_dict()
    if stratum_counts != dict(FIXED_STRATUM_TARGETS):
        raise PipelineError(
            f"Translation master stratum counts are {stratum_counts}, "
            f"expected {dict(FIXED_STRATUM_TARGETS)}."
        )


def check_preconditions(args: argparse.Namespace) -> None:
    check_inputs_exist([args.path_master])
    if not os.environ.get("ANTHROPIC_API_KEY") and not has_reusable_checkpoint(
        args.path_master, args.path_translation_checkpoint
    ):
        raise PipelineError(
            "ANTHROPIC_API_KEY is required for the translate and all stages."
        )
    if not args.resume:
        header = pd.read_csv(args.path_master, nrows=0)
        if "translation" in header.columns or args.path_translation_checkpoint.exists():
            raise PipelineError(
                "Translated master output or batch checkpoint already exists; "
                "pass --resume to reuse it."
            )
    check_positive("num_api_attempts", args.num_api_attempts)


def load_prior_checkpoint(master: pd.DataFrame, path_checkpoint: Path) -> dict | None:
    if not path_checkpoint.is_file():
        return None
    try:
        prior_checkpoint = load_json(path_checkpoint)
        check_translation_checkpoint(master, prior_checkpoint)
    except CHECKPOINT_ERRORS as error:
        raise PipelineError(
            "Translation checkpoint is invalid or belongs to another sample: "
            f"{error}"
        ) from error
    return prior_checkpoint


def main(args: argparse.Namespace) -> None:
    logger.info(json.dumps(vars(args), default=str))
    load_dotenv(override=False)
    check_preconditions(args)

    master = load_csv(args.path_master)
    check_translation_master(master)
    prior_checkpoint = load_prior_checkpoint(master, args.path_translation_checkpoint)
    if prior_checkpoint is not None and prior_checkpoint.get("status") == "success":
        translated_master = restore_translations(master, prior_checkpoint)
        write_csv(translated_master, args.path_master)
        logger.info(
            f"wrote {len(translated_master)} rows to {args.path_master} "
            "(reused batch checkpoint)"
        )
        return

    import anthropic

    logger.info("Start translation.")
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    translated_master, checkpoint = translate_master(
        master,
        client,
        max_api_attempts=args.num_api_attempts,
        prior_checkpoint=prior_checkpoint,
        save_checkpoint=lambda checkpoint: write_json(
            checkpoint, args.path_translation_checkpoint
        ),
    )
    write_json(checkpoint, args.path_translation_checkpoint)
    write_csv(translated_master, args.path_master)
    logger.info("Finished translation.")
    logger.info(
        f"wrote {len(translated_master)} rows to {args.path_master} "
        "(one Haiku batch request)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Translate the master sample with Claude Haiku, resumably."
    )
    # data
    parser.add_argument(
        "--path_master",
        type=Path,
        default=Path("data/master_sample_stratified.csv"),
        help="Master sample CSV; translation columns are written back into it.",
    )
    parser.add_argument(
        "--path_translation_checkpoint",
        type=Path,
        default=Path("data/checkpoints/haiku_batch_translation.json"),
        help="JSON checkpoint of the batch response, reused instead of a new call.",
    )
    # api
    parser.add_argument(
        "--num_api_attempts",
        type=int,
        default=5,
        help="Maximum number of API attempts in this run.",
    )
    # saving
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse an existing checkpoint and overwrite existing translations.",
    )
    cmd_args = parser.parse_args()
    try:
        main(cmd_args)
    except PipelineError as error:
        logger.error(str(error))
        sys.exit(1)
