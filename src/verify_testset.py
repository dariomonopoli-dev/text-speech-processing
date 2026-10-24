"""Verify the master sample and unlabelled annotation files, and write a report."""

import argparse
import json
import sys
from pathlib import Path

from annotation_assignment import get_annotation_paths
from get_loggers import get_logger
from pipeline_io import (
    PipelineError,
    check_inputs_exist,
    check_outputs_absent,
    load_csv,
    write_json,
)
from verification import check_testset

logger = get_logger("verify")


def main(args: argparse.Namespace) -> None:
    logger.info(json.dumps(vars(args), default=str))
    path_annotations = get_annotation_paths(args.path_annotation_dir)
    check_inputs_exist([args.path_master, *path_annotations.values()])
    check_outputs_absent([args.path_verification_report], args.resume)

    master = load_csv(args.path_master)
    annotation_sets = {
        annotator: load_csv(path) for annotator, path in path_annotations.items()
    }
    report = check_testset(master, annotation_sets)
    write_json(report, args.path_verification_report)
    logger.info(f"passed and wrote {args.path_verification_report}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check row counts, strata, overlap and leakage of the test set."
    )
    # data
    parser.add_argument(
        "--path_master",
        type=Path,
        default=Path("data/master_sample_stratified.csv"),
        help="Translated master sample CSV.",
    )
    parser.add_argument(
        "--path_annotation_dir",
        type=Path,
        default=Path("data"),
        help="Directory holding annotation_<annotator>_stratified.csv.",
    )
    # saving
    parser.add_argument(
        "--path_verification_report",
        type=Path,
        default=Path("data/stratified_verification_report.json"),
        help="Output JSON verification report.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Overwrite an existing report instead of refusing.",
    )
    cmd_args = parser.parse_args()
    try:
        main(cmd_args)
    except PipelineError as error:
        logger.error(str(error))
        sys.exit(1)
