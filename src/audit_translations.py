"""Audit a translation CSV with the checks shared by all normalisers."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from get_loggers import get_logger
from pipeline_io import PipelineError, check_inputs_exist, load_csv, write_csv
from translation import API_USAGE_COLUMNS, add_translations, flag_unchanged

logger = get_logger("audit")

REQUIRED_COLUMNS = ["sentence_id", "text", "translation"]


def check_translations(translations: pd.DataFrame, path_translations: Path) -> None:
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(translations.columns))
    if missing_columns:
        raise KeyError(f"{path_translations} lacks columns {missing_columns}")
    if translations["sentence_id"].isna().any() or translations["sentence_id"].duplicated().any():
        raise ValueError(f"{path_translations} needs unique, non-null sentence_id values.")


def audit_translations(translations: pd.DataFrame) -> pd.DataFrame:
    """Recompute the audit columns and the unchanged flag, keeping the translations as they are."""
    audited = add_translations(
        translations[["sentence_id", "text"]].reset_index(drop=True),
        list(translations["translation"]),
        attempts=1,
    )
    return flag_unchanged(audited.drop(columns=list(API_USAGE_COLUMNS)))


def main(args: argparse.Namespace) -> None:
    logger.info(json.dumps(vars(args), default=str))
    check_inputs_exist([args.path_translations])
    if args.path_output.exists() and not args.overwrite:
        raise PipelineError(
            f"Output {args.path_output} exists; pass --overwrite to replace it."
        )
    if args.path_output.resolve() == args.path_translations.resolve():
        raise PipelineError("The output must be a different file than the input.")

    translations = load_csv(args.path_translations)
    check_translations(translations, args.path_translations)

    audited = audit_translations(translations)
    num_unchanged = int(audited["translation_unchanged"].sum())
    num_anomalies = int(audited["translation_has_anomaly"].sum())
    logger.info(
        f"{len(audited)} rows, {num_unchanged} unchanged, {num_anomalies} flagged for review."
    )
    write_csv(audited, args.path_output)
    logger.info(f"wrote {len(audited)} rows to {args.path_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flag unchanged and malformed translations in a translation CSV."
    )
    # data
    parser.add_argument(
        "--path_translations",
        type=Path,
        required=True,
        help="CSV with the columns sentence_id, text and translation.",
    )
    parser.add_argument(
        "--path_output",
        type=Path,
        required=True,
        help="New CSV with the audit columns; never the input file.",
    )
    # saving
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output file if it exists.",
    )
    cmd_args = parser.parse_args()
    try:
        main(cmd_args)
    except PipelineError as error:
        logger.error(str(error))
        sys.exit(1)
