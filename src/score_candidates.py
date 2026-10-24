"""Score every SwissCrawl candidate sentence with the GAHD classifier."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from get_loggers import get_logger
from pipeline_io import (
    PipelineError,
    check_inputs_exist,
    check_outputs_absent,
    check_positive,
    load_csv,
    write_csv,
)
from scoring import add_hate_scores, load_gahd_classifier

logger = get_logger("score")


def load_scored_candidates(candidates: pd.DataFrame, path_scored: Path) -> pd.DataFrame:
    """Load an earlier scoring output, provided it scores exactly these candidates."""
    scored_candidates = load_csv(path_scored)
    required = {"sentence_id", "text", "hate_score"}
    if not required.issubset(scored_candidates.columns) or not scored_candidates[
        ["sentence_id", "text"]
    ].equals(candidates[["sentence_id", "text"]]):
        raise PipelineError(
            "Existing scored output does not match the candidate corpus."
        )
    hate_scores = pd.to_numeric(scored_candidates["hate_score"], errors="coerce")
    if hate_scores.isna().any() or not hate_scores.between(0.0, 1.0).all():
        raise PipelineError("Existing scored output contains invalid scores.")
    return scored_candidates


def main(args: argparse.Namespace) -> None:
    logger.info(json.dumps(vars(args), default=str))
    load_dotenv(override=False)
    check_inputs_exist([args.path_candidates])
    check_outputs_absent([args.path_scored], args.resume)
    check_positive("batch_size", args.batch_size)

    candidates = load_csv(args.path_candidates)
    if args.resume and args.path_scored.is_file():
        scored_candidates = load_scored_candidates(candidates, args.path_scored)
        logger.info(f"reused {len(scored_candidates)} rows from {args.path_scored}")
        return
    logger.info("Start scoring.")
    classifier = load_gahd_classifier(args.device)
    scored_candidates = add_hate_scores(
        candidates, classifier, batch_size=args.batch_size
    )
    write_csv(scored_candidates, args.path_scored)
    logger.info("Finished scoring.")
    logger.info(f"wrote {len(scored_candidates)} rows to {args.path_scored}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Score the SwissCrawl candidates with the pinned GAHD classifier."
    )
    # data
    parser.add_argument(
        "--path_candidates",
        type=Path,
        default=Path("data/swisscrawl_filtered.csv"),
        help="CSV of filtered SwissCrawl candidates with sentence_id and text.",
    )
    parser.add_argument(
        "--path_scored",
        type=Path,
        default=Path("data/swisscrawl_scored_gahd.csv"),
        help="Output CSV: the candidates plus a hate_score column.",
    )
    # inference
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Inference batch size."
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "mps"],
        help="Inference device; defaults to mps when available, else cpu.",
    )
    # saving
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse an existing scored output if it matches the candidates.",
    )
    cmd_args = parser.parse_args()
    try:
        main(cmd_args)
    except PipelineError as error:
        logger.error(str(error))
        sys.exit(1)
