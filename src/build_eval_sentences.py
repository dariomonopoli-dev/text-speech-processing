"""Collect the evaluation sentences from the three stratified annotation files."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from get_loggers import get_logger
from gold_labels import ANNOTATORS, get_path_annotation_file
from pipeline_io import PipelineError

logger = get_logger("build")

SOURCE_COLUMNS = ["sentence_id", "text"]


def load_annotated_sentences(path_annotations: Path) -> pd.DataFrame:
    """Return each annotated sentence once, with sentence_id and text."""
    frames = []
    for annotator in ANNOTATORS:
        path_annotation_file = get_path_annotation_file(path_annotations, annotator)
        if not path_annotation_file.is_file():
            raise FileNotFoundError(
                f"annotation file not found: {path_annotation_file}"
            )
        annotation_df = pd.read_csv(
            path_annotation_file, encoding="utf-8-sig", dtype={"sentence_id": "string"}
        )
        missing_columns = sorted(set(SOURCE_COLUMNS) - set(annotation_df.columns))
        if missing_columns:
            raise KeyError(f"{path_annotation_file} lacks columns {missing_columns}")
        frames.append(annotation_df[SOURCE_COLUMNS])
    sentences = pd.concat(frames)
    num_texts_per_id = sentences.groupby("sentence_id")["text"].nunique()
    conflicting_ids = list(num_texts_per_id[num_texts_per_id > 1].index)
    if conflicting_ids:
        raise ValueError(
            f"{len(conflicting_ids)} sentence_ids have different texts in the "
            f"annotation files, for example {conflicting_ids[:5]}"
        )
    return sentences.drop_duplicates("sentence_id").reset_index(drop=True)


def main(args: argparse.Namespace) -> None:
    logger.info(json.dumps(vars(args), default=str))
    if args.path_output.exists() and not args.overwrite:
        raise PipelineError(
            f"Output {args.path_output} exists; pass --overwrite to replace it."
        )
    sentences = load_annotated_sentences(args.path_annotations)
    args.path_output.parent.mkdir(parents=True, exist_ok=True)
    sentences.to_csv(args.path_output, index=False)
    logger.info(f"wrote {len(sentences)} sentences to {args.path_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    # data
    parser.add_argument(
        "--path_annotations",
        type=Path,
        default=Path("data"),
        help="Directory with annotation_<annotator>_stratified.csv files.",
    )
    # saving
    parser.add_argument(
        "--path_output",
        type=Path,
        default=Path("data/eval_sentences.csv"),
        help="CSV with the columns sentence_id and text.",
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
