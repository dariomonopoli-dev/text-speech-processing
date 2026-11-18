"""Compare translation outputs of the same sentences produced with different batch sizes."""

import argparse
import json
from pathlib import Path

import pandas as pd

from get_loggers import get_logger
from pipeline_io import write_csv
from translation import compute_unchanged

logger = get_logger("batch_sizes")

REQUIRED_COLUMNS = {"sentence_id", "text", "translation", "translation_has_anomaly"}


def load_translation_output(path_output: Path) -> pd.DataFrame:
    """Load one translation output CSV indexed by sentence_id."""
    if not path_output.is_file():
        raise FileNotFoundError(f"translation output not found: {path_output}")
    output = pd.read_csv(
        path_output, encoding="utf-8-sig", dtype={"sentence_id": "string"}
    )
    missing_columns = sorted(REQUIRED_COLUMNS - set(output.columns))
    if missing_columns:
        raise KeyError(f"{path_output} lacks columns {missing_columns}")
    return output.set_index("sentence_id")


def check_same_sentences(outputs: dict[str, pd.DataFrame]) -> None:
    reference_name, reference = next(iter(outputs.items()))
    for name, output in outputs.items():
        if set(output.index) != set(reference.index):
            raise ValueError(f"{name} and {reference_name} contain different sentence_ids")


def compute_batch_size_summary(outputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per output: unchanged and anomaly counts, and agreement with the first output."""
    reference = next(iter(outputs.values()))
    rows = []
    for name, output in outputs.items():
        aligned = output.loc[reference.index]
        rows.append(
            {
                "output": name,
                "num_sentences": len(output),
                "num_unchanged": int(compute_unchanged(output).sum()),
                "num_anomalies": int(output["translation_has_anomaly"].sum()),
                "num_identical_to_first": int(
                    (aligned["translation"] == reference["translation"]).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def log_differing_examples(outputs: dict[str, pd.DataFrame], num_examples: int) -> None:
    reference_name, reference = next(iter(outputs.items()))
    for name, output in list(outputs.items())[1:]:
        aligned = output.loc[reference.index]
        differing_ids = reference.index[aligned["translation"] != reference["translation"]]
        for sentence_id in differing_ids[:num_examples]:
            logger.info(
                f"[{sentence_id}] {reference.loc[sentence_id, 'text']}\n"
                f"  {reference_name}: {reference.loc[sentence_id, 'translation']}\n"
                f"  {name}: {aligned.loc[sentence_id, 'translation']}"
            )


def main(args: argparse.Namespace) -> None:
    logger.info(json.dumps(vars(args), ensure_ascii=False, default=str))
    if len(args.path_outputs) < 2:
        raise ValueError("--path_outputs needs at least two translation outputs")
    if args.path_summary.exists() and not args.overwrite:
        raise FileExistsError(f"{args.path_summary} exists, pass --overwrite to replace it")

    outputs = {path.stem: load_translation_output(path) for path in args.path_outputs}
    if len(outputs) < len(args.path_outputs):
        raise ValueError("--path_outputs contains two files with the same name")
    check_same_sentences(outputs)

    summary = compute_batch_size_summary(outputs)
    logger.info("\n" + summary.to_string(index=False))
    log_differing_examples(outputs, args.num_examples)
    write_csv(summary, args.path_summary)
    logger.info(f"wrote the summary to {args.path_summary}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    # data
    parser.add_argument(
        "--path_outputs",
        type=Path,
        nargs="+",
        required=True,
        help="Translation output CSVs of the same sentences, the first is the reference.",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=8,
        help="Differing translations logged per output.",
    )
    # saving
    parser.add_argument(
        "--path_summary",
        type=Path,
        required=True,
        help="CSV with one row of counts per translation output.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing summary file."
    )
    cmd_args = parser.parse_args()
    main(cmd_args)
