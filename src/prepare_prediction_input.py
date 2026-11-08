"""Write the prediction input of one normaliser: sentence_id and one named translation column."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from get_loggers import get_logger
from pipeline_io import PipelineError, check_inputs_exist, load_csv

logger = get_logger("prepare")

INPUT_COLUMNS = ["translation_haiku", "translation_apertus8b"]
REQUIRED_COLUMNS = ["sentence_id", "translation"]


def check_translations(translations: pd.DataFrame, path_translations: Path) -> None:
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(translations.columns))
    if missing_columns:
        raise KeyError(f"{path_translations} lacks columns {missing_columns}")
    sentence_ids = translations["sentence_id"]
    if sentence_ids.isna().any() or sentence_ids.duplicated().any():
        raise ValueError(f"{path_translations} needs unique, non-null sentence_id values.")
    is_blank = translations["translation"].fillna("").astype(str).str.strip().eq("")
    if is_blank.any():
        raise ValueError(
            f"{path_translations} has {int(is_blank.sum())} rows without a translation."
        )


def to_prediction_input(translations: pd.DataFrame, input_column: str) -> pd.DataFrame:
    """Keep the ids and name the translation column after its normaliser."""
    return translations[REQUIRED_COLUMNS].rename(columns={"translation": input_column})


def main(args: argparse.Namespace) -> None:
    logger.info(json.dumps(vars(args), default=str))
    check_inputs_exist([args.path_translations])
    if args.path_output.resolve() == args.path_translations.resolve():
        raise PipelineError("The output must be a different file than the input.")
    if args.path_output.exists() and not args.overwrite:
        raise PipelineError(
            f"Output {args.path_output} exists; pass --overwrite to replace it."
        )
    translations = load_csv(args.path_translations)
    check_translations(translations, args.path_translations)

    prediction_input = to_prediction_input(translations, args.input_column)
    args.path_output.parent.mkdir(parents=True, exist_ok=True)
    prediction_input.to_csv(args.path_output, index=False)
    logger.info(
        f"wrote {len(prediction_input)} rows with column {args.input_column} "
        f"to {args.path_output}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    # data
    parser.add_argument(
        "--path_translations",
        type=Path,
        required=True,
        help="CSV with the columns sentence_id and translation.",
    )
    parser.add_argument(
        "--input_column",
        required=True,
        choices=INPUT_COLUMNS,
        help="Name of the translation column in the output, also the --input_column of predict.py.",
    )
    # saving
    parser.add_argument(
        "--path_output",
        type=Path,
        required=True,
        help="CSV for predict.py; never the input file.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing output file."
    )
    cmd_args = parser.parse_args()
    try:
        main(cmd_args)
    except PipelineError as error:
        logger.error(str(error))
        sys.exit(1)
