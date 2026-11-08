"""Count the gold-hate sentences each variant detects per target and mechanism."""

import argparse
import json
from pathlib import Path

import pandas as pd

from get_loggers import get_logger
from pipeline_io import write_csv

logger = get_logger("hate_analysis")

SCORE_COLUMNS = {
    "raw": "mean_prob_hate_raw",
    "apertus": "mean_prob_hate_apertus8b",
    "haiku": "mean_prob_hate_haiku",
}
TAG_COLUMNS = ["Target", "Mechanism"]
TAG_SEPARATOR = "&"
TARGET_NAMES = [
    "Nationality / ethnicity",
    "Women",
    "Sexual orientation / gender identity",
    "Disability",
]
MECHANISM_NAMES = ["Slur", "Generalisation / stereotype", "Implicit / mockery"]
TAG_NAMES = {"Target": TARGET_NAMES, "Mechanism": MECHANISM_NAMES}
# n, raw, apertus, haiku as counted by hand for the report table
EXPECTED_COUNTS = {
    "Nationality / ethnicity": (9, 5, 4, 3),
    "Women": (9, 4, 5, 3),
    "Sexual orientation / gender identity": (6, 5, 5, 5),
    "Disability": (4, 3, 1, 1),
    "Slur": (12, 6, 6, 7),
    "Generalisation / stereotype": (6, 2, 2, 2),
    "Implicit / mockery": (8, 7, 5, 2),
    "All": (26, 15, 13, 11),
}
COUNT_COLUMNS = ["n", *SCORE_COLUMNS]


def split_tags(cell: str) -> list[str]:
    """Split a tag cell with several tags into its tags."""
    return [tag.strip() for tag in str(cell).split(TAG_SEPARATOR) if tag.strip()]


def load_hate_sentences(path_sentences: Path) -> pd.DataFrame:
    """Load the gold-hate sentences with their scores and tags."""
    if not path_sentences.is_file():
        raise FileNotFoundError(f"sentence file not found: {path_sentences}")
    sentences = pd.read_csv(path_sentences, encoding="utf-8-sig")
    required = {"sentence_id", *TAG_COLUMNS, *SCORE_COLUMNS.values()}
    missing_columns = sorted(required - set(sentences.columns))
    if missing_columns:
        raise KeyError(f"{path_sentences} lacks columns {missing_columns}")
    for column in TAG_COLUMNS:
        tag_counts = sentences[column].map(
            lambda cell: len(split_tags(cell)) if pd.notna(cell) else 0
        )
        empty_ids = sentences.loc[tag_counts == 0, "sentence_id"].tolist()
        if empty_ids:
            raise ValueError(f"{column} is empty for sentence_ids {empty_ids}")
    return sentences


def select_sentences(
    sentences: pd.DataFrame, name: str, columns: list[str] = TAG_COLUMNS
) -> pd.DataFrame:
    """Return the sentences that carry the tag, or all sentences for 'All'."""
    if name == "All":
        return sentences
    has_tag = sentences[columns].apply(
        lambda row: any(name in split_tags(cell) for cell in row), axis=1
    )
    return sentences[has_tag]


def count_detections(group: pd.DataFrame, threshold: float) -> dict[str, int]:
    counts = {"n": len(group)}
    for variant, column in SCORE_COLUMNS.items():
        counts[variant] = int((group[column] >= threshold).sum())
    return counts


def list_tag_names(sentences: pd.DataFrame, column: str) -> list[str]:
    """Return the expected names of a tag column followed by any other tags found."""
    found = {tag for cell in sentences[column] for tag in split_tags(cell)}
    return TAG_NAMES[column] + sorted(found - set(TAG_NAMES[column]))


def compute_analysis_table(sentences: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Count n and detections per target, per mechanism and over all sentences."""
    rows = []
    for column in TAG_COLUMNS:
        for name in list_tag_names(sentences, column):
            group = select_sentences(sentences, name, [column])
            rows.append({"group": column, "name": name, **count_detections(group, threshold)})
    rows.append({"group": "All", "name": "All", **count_detections(sentences, threshold)})
    return pd.DataFrame(rows)


def find_mismatches(table: pd.DataFrame) -> list[dict]:
    """Compare the table with the hand counts and list every differing cell."""
    mismatches = []
    for name, expected in EXPECTED_COUNTS.items():
        row = table[table["name"] == name]
        actual = row.iloc[0] if len(row) else pd.Series(0, index=COUNT_COLUMNS)
        for column, expected_count in zip(COUNT_COLUMNS, expected):
            if int(actual[column]) != expected_count:
                mismatches.append(
                    {
                        "name": name,
                        "column": column,
                        "expected": expected_count,
                        "actual": int(actual[column]),
                    }
                )
    unexpected = sorted(set(table["name"]) - set(EXPECTED_COUNTS))
    mismatches.extend(
        {"name": name, "column": "tag", "expected": "absent", "actual": "present"}
        for name in unexpected
    )
    return mismatches


def format_sentence(sentence: pd.Series) -> str:
    scores = ", ".join(
        f"{variant}={sentence[column]:.3f}" for variant, column in SCORE_COLUMNS.items()
    )
    return (
        f"  {sentence['sentence_id']}: Target={sentence['Target']!r} "
        f"Mechanism={sentence['Mechanism']!r} {scores}"
    )


def log_mismatches(sentences: pd.DataFrame, mismatches: list[dict]) -> None:
    for mismatch in mismatches:
        logger.warning(
            f"{mismatch['name']} / {mismatch['column']}: "
            f"hand count {mismatch['expected']}, script {mismatch['actual']}"
        )
        group = select_sentences(sentences, mismatch["name"])
        for _, sentence in group.iterrows():
            logger.warning(format_sentence(sentence))


def log_close_scores(sentences: pd.DataFrame, threshold: float, margin: float) -> None:
    """List sentences whose score is within margin of the threshold for any variant."""
    scores = sentences[list(SCORE_COLUMNS.values())]
    close = sentences[((scores - threshold).abs() <= margin).any(axis=1)]
    logger.info(f"{len(close)} sentences have a score within {margin} of {threshold}")
    for _, sentence in close.iterrows():
        logger.info(format_sentence(sentence))


def format_latex_rows(table: pd.DataFrame) -> list[str]:
    return [
        " & ".join(str(row[column]) for column in ["name", *COUNT_COLUMNS]) + " \\\\"
        for _, row in table.iterrows()
    ]


def main(args: argparse.Namespace) -> None:
    logger.info(json.dumps(vars(args), ensure_ascii=False, default=str))
    if args.path_output.exists() and not args.overwrite:
        raise FileExistsError(
            f"{args.path_output} exists, pass --overwrite to replace it"
        )
    sentences = load_hate_sentences(args.path_sentences)
    table = compute_analysis_table(sentences, args.threshold)
    logger.info("\n" + table.drop(columns="group").to_string(index=False))
    logger.info("LaTeX rows:\n" + "\n".join(format_latex_rows(table)))

    mismatches = find_mismatches(table)
    if mismatches:
        logger.warning(f"{len(mismatches)} cells differ from the hand counts")
        log_mismatches(sentences, mismatches)
    else:
        logger.info("all cells equal the hand counts")
    log_close_scores(sentences, args.threshold, args.margin)

    write_csv(table, args.path_output)
    logger.info(f"wrote the table to {args.path_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    # data
    parser.add_argument(
        "--path_sentences",
        type=Path,
        default=Path("results/gold_hate_sentences.csv"),
        help="CSV with the gold-hate sentences, seed-mean scores, Target and Mechanism.",
    )
    # hyperparams
    parser.add_argument("--threshold", type=float, default=0.5, help="Hate threshold.")
    parser.add_argument(
        "--margin",
        type=float,
        default=0.02,
        help="Report sentences with a score this close to the threshold.",
    )
    # saving
    parser.add_argument(
        "--path_output",
        type=Path,
        default=Path("results/hate_analysis_table.csv"),
        help="CSV with the analysis table.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing output file."
    )
    cmd_args = parser.parse_args()
    main(cmd_args)
