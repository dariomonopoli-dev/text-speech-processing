"""Aggregate the three stratified annotation files into one labeled test set."""

import argparse
import json
from pathlib import Path

import pandas as pd

from get_loggers import get_logger
from gold_labels import ANNOTATORS, get_path_annotation_file, load_gold_labels

logger = get_logger("build_labeled_testset")


def main(args: argparse.Namespace) -> None:
    logger.info(json.dumps(vars(args), indent=4, default=str))
    path_annotations = Path(args.path_annotations)
    gold_df = load_gold_labels(path_annotations)

    texts = pd.concat(
        [
            pd.read_csv(
                get_path_annotation_file(path_annotations, annotator),
                encoding="utf-8-sig",
            ).set_index("sentence_id")["text"]
            for annotator in ANNOTATORS
        ]
    )
    texts = texts[~texts.index.duplicated()]

    test_set_df = (gold_df[["label"]].join(texts.rename("text")).reset_index())[
        ["sentence_id", "text", "label"]
    ]
    test_set_df.to_csv(args.path_output, index=False, encoding="utf-8")
    logger.info(
        f"Wrote {len(test_set_df)} sentences "
        f"({int(test_set_df['label'].sum())} hate) to {args.path_output}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument(
        "--path_annotations",
        default="data",
        help="Directory with the annotation_<name>_stratified.csv files.",
    )
    # saving
    parser.add_argument(
        "--path_output",
        default="data/test_set.csv",
        help="Where to write sentence_id,text,label.",
    )
    cmd_args = parser.parse_args()
    main(cmd_args)
