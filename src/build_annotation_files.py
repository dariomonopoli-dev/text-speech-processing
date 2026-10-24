"""Write one CSV and one XLSX annotation file per annotator."""

import argparse
import json
import sys
from pathlib import Path

from annotation_assignment import (
    assign_annotations,
    get_annotation_paths,
    get_workbook_paths,
)
from get_loggers import get_logger
from pipeline_io import (
    PipelineError,
    check_inputs_exist,
    check_outputs_absent,
    load_csv,
    write_csv,
    write_xlsx,
)

logger = get_logger("annotate")


def main(args: argparse.Namespace) -> None:
    logger.info(json.dumps(vars(args), default=str))
    path_annotations = get_annotation_paths(args.path_annotation_dir)
    path_workbooks = get_workbook_paths(args.path_annotation_dir)
    check_inputs_exist([args.path_master])
    check_outputs_absent(
        [*path_annotations.values(), *path_workbooks.values()], args.resume
    )

    master = load_csv(args.path_master)
    annotation_sets = assign_annotations(master, seed=args.seed)
    for annotator, annotation_set in annotation_sets.items():
        for path, write in (
            (path_annotations[annotator], write_csv),
            (path_workbooks[annotator], write_xlsx),
        ):
            if path.exists():
                logger.info(f"preserved existing {path}")
            else:
                write(annotation_set, path)
                logger.info(f"wrote {len(annotation_set)} rows to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Assign 100 shared and 200 single_annotator rows per annotator."
    )
    # data
    parser.add_argument(
        "--path_master",
        type=Path,
        default=Path("data/master_sample_stratified.csv"),
        help="Translated master sample CSV.",
    )
    # saving
    parser.add_argument(
        "--path_annotation_dir",
        type=Path,
        default=Path("data"),
        help="Directory for annotation_<annotator>_stratified.csv and .xlsx.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep existing annotation files and write only the missing ones.",
    )
    # hyperparams
    parser.add_argument("--seed", type=int, default=42, help="Assignment seed.")
    cmd_args = parser.parse_args()
    try:
        main(cmd_args)
    except PipelineError as error:
        logger.error(str(error))
        sys.exit(1)
