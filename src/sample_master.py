"""Draw the 700-row stratified master sample from the scored candidates."""

import argparse
import json
import sys
from pathlib import Path

from get_loggers import get_logger
from pipeline_io import (
    PipelineError,
    check_inputs_exist,
    check_outputs_absent,
    load_csv,
    write_csv,
    write_json,
)
from sampling import sample_master

logger = get_logger("sample")


def main(args: argparse.Namespace) -> None:
    logger.info(json.dumps(vars(args), default=str))
    check_inputs_exist([args.path_scored])
    check_outputs_absent([args.path_master, args.path_sampling_report], args.resume)

    scored_candidates = load_csv(args.path_scored)
    master, report = sample_master(scored_candidates, seed=args.seed)
    write_csv(master, args.path_master)
    write_json(report, args.path_sampling_report)
    logger.info(f"wrote {len(master)} rows to {args.path_master}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample 700 rows with the fixed AustroTox stratum allocation."
    )
    # data
    parser.add_argument(
        "--path_scored",
        type=Path,
        default=Path("data/swisscrawl_scored_gahd.csv"),
        help="CSV of scored candidates from score_candidates.py.",
    )
    # saving
    parser.add_argument(
        "--path_master",
        type=Path,
        default=Path("data/master_sample_stratified.csv"),
        help="Output CSV of the 700-row master sample.",
    )
    parser.add_argument(
        "--path_sampling_report",
        type=Path,
        default=Path("data/stratified_sampling_report.json"),
        help="Output JSON with per-stratum counts and score ranges.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Overwrite existing outputs instead of refusing.",
    )
    # hyperparams
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    cmd_args = parser.parse_args()
    try:
        main(cmd_args)
    except PipelineError as error:
        logger.error(str(error))
        sys.exit(1)
