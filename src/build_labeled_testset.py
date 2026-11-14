"""Add the gold labels to the stratified master sample."""

import argparse
import json
from pathlib import Path

import pandas as pd

from get_loggers import get_logger
from gold_labels import ANNOTATORS, load_annotations, load_gold_labels

logger = get_logger("build_labeled_testset")


def main(args: argparse.Namespace) -> None:
    logger.info(json.dumps(vars(args), indent=4, default=str))
    path_master = Path(args.path_master)
    path_output = Path(args.path_output)
    if not path_master.is_file():
        raise FileNotFoundError(f"master sample not found: {path_master}")

    master_df = pd.read_csv(path_master, encoding="utf-8-sig")
    gold_df = load_gold_labels(Path(args.path_annotations))
    unlabeled_ids = set(master_df["sentence_id"]) - set(gold_df.index)
    if unlabeled_ids:
        raise ValueError(f"{len(unlabeled_ids)} sentences have no label")

    annotations = load_annotations(Path(args.path_annotations))
    annotator_df = pd.DataFrame(
        {f"label_{name}": annotations[name] for name in ANNOTATORS}
    )
    added_columns = ["label", "is_shared", "split", *annotator_df.columns]
    labeled_df = master_df.drop(columns=added_columns, errors="ignore")
    labeled_df = labeled_df.join(gold_df, on="sentence_id").join(
        annotator_df, on="sentence_id"
    )
    labeled_df["split"] = labeled_df["is_shared"].map(
        {True: "shared", False: "single_annotator"}
    )
    labeled_df.to_csv(path_output, index=False, encoding="utf-8-sig")
    logger.info(
        f"Wrote {len(labeled_df)} labeled sentences "
        f"({int(labeled_df['label'].sum())} hate) to {path_output}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument(
        "--path_master",
        default="data/master_sample_stratified.csv",
        help="Stratified master sample with the translations.",
    )
    parser.add_argument(
        "--path_annotations",
        default="data",
        help="Directory with the annotation_<name>_stratified.csv files.",
    )
    # saving
    parser.add_argument(
        "--path_output",
        default="data/master_sample_stratified.csv",
        help="Where to write the labeled test set.",
    )
    cmd_args = parser.parse_args()
    main(cmd_args)
