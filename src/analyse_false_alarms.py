"""Select the top-scoring non-hateful sentences for manual tagging and summarise the tags."""

import argparse
import json
from pathlib import Path

import pandas as pd

from get_loggers import get_logger
from pipeline_io import write_csv

logger = get_logger("false_alarms")

VARIANTS = {
    "raw": "mean_prob_hate_raw",
    "ap": "mean_prob_hate_apertus8b",
    "hk": "mean_prob_hate_haiku",
}
SCORE_COLUMNS = list(VARIANTS.values())
REQUIRED_COLUMNS = {
    "sentence_id",
    "text",
    "gold_label",
    "translation_haiku",
    "translation_apertus8b",
    *SCORE_COLUMNS,
}
TAG_COLUMNS = ["category", "label_noise"]


def load_scores(path_scores: Path) -> pd.DataFrame:
    """Load the CSV of all test sentences with gold labels and seed-mean scores."""
    if not path_scores.is_file():
        raise FileNotFoundError(f"scores file not found: {path_scores}")
    scores = pd.read_csv(path_scores, encoding="utf-8-sig")
    missing_columns = sorted(REQUIRED_COLUMNS - set(scores.columns))
    if missing_columns:
        raise KeyError(f"{path_scores} lacks columns {missing_columns}")
    return scores


def compute_alarm_counts(scores: pd.DataFrame, threshold: float) -> dict[str, dict]:
    """Count false alarms among non-hate and hits among hate sentences per variant."""
    hate = scores[scores["gold_label"] == 1]
    not_hate = scores[scores["gold_label"] == 0]
    return {
        name: {
            "false_alarms": int((not_hate[column] >= threshold).sum()),
            "num_not_hate": len(not_hate),
            "hate_caught": int((hate[column] >= threshold).sum()),
            "num_hate": len(hate),
        }
        for name, column in VARIANTS.items()
    }


def select_false_alarms(scores: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """Combine the top_k highest-scoring non-hate sentences of every variant."""
    not_hate = scores[scores["gold_label"] == 0]
    top_ids = {
        name: set(not_hate.nlargest(top_k, column)["sentence_id"])
        for name, column in VARIANTS.items()
    }
    selected = not_hate[not_hate["sentence_id"].isin(set().union(*top_ids.values()))].copy()
    selected["in_top_k"] = [
        "/".join(name for name in VARIANTS if sentence_id in top_ids[name])
        for sentence_id in selected["sentence_id"]
    ]
    selected["category"] = ""
    selected["label_noise"] = ""
    selected = selected.assign(max_score=selected[SCORE_COLUMNS].max(axis=1))
    selected = selected.sort_values("max_score", ascending=False)
    return selected[
        [
            "sentence_id",
            *TAG_COLUMNS,
            *SCORE_COLUMNS,
            "in_top_k",
            "text",
            "translation_haiku",
            "translation_apertus8b",
        ]
    ].round(2)


def summarise_row(name: str, group: pd.DataFrame, threshold: float) -> dict:
    row = {"category": name, "n": len(group)}
    for variant, column in VARIANTS.items():
        row[f"mean_{variant}"] = round(group[column].mean(), 2)
        row[f"above_threshold_{variant}"] = int((group[column] >= threshold).sum())
    return row


def summarise_categories(tagged: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Per category: sentence count, mean score and count at or above the threshold."""
    rows = [
        summarise_row(category, group, threshold)
        for category, group in tagged.groupby("category")
    ]
    rows.append(summarise_row("All", tagged, threshold))
    return pd.DataFrame(rows)


def run_select(args: argparse.Namespace) -> None:
    if args.path_tag_file.exists() and not args.overwrite:
        raise FileExistsError(
            f"{args.path_tag_file} exists and may hold manual tags, "
            "pass --overwrite to replace it"
        )
    scores = load_scores(args.path_scores)
    counts = compute_alarm_counts(scores, args.threshold)
    for name, count in counts.items():
        logger.info(f"{name}: {json.dumps(count)}")
    selected = select_false_alarms(scores, args.top_k)
    write_csv(selected, args.path_tag_file)
    logger.info(f"wrote {len(selected)} sentences to {args.path_tag_file}")


def run_summarise(args: argparse.Namespace) -> None:
    if not args.path_tag_file.is_file():
        raise FileNotFoundError(f"tag file not found: {args.path_tag_file}")
    if args.path_summary.exists() and not args.overwrite:
        raise FileExistsError(
            f"{args.path_summary} exists, pass --overwrite to replace it"
        )
    tagged =pd.read_csv(args.path_tag_file, encoding="utf-8-sig")
    missing_columns = sorted(set(TAG_COLUMNS + SCORE_COLUMNS) - set(tagged.columns))
    if missing_columns:
        raise KeyError(f"{args.path_tag_file} lacks columns {missing_columns}")
    untagged = int(tagged["category"].isna().sum())
    if untagged:
        logger.warning(f"{untagged} sentences have no category and are skipped")
    tagged_rows = tagged[tagged["category"].notna()]
    summary = summarise_categories(tagged_rows, args.threshold)
    logger.info("\n" + summary.to_string(index=False))
    write_csv(summary, args.path_summary)
    logger.info(f"wrote the summary to {args.path_summary}")
    label_noise = tagged["label_noise"].fillna("").value_counts().to_dict()
    logger.info(f"label noise: {json.dumps(label_noise, ensure_ascii=False)}")

    changes = tagged_rows.assign(
        change_hk=tagged_rows[VARIANTS["hk"]] - tagged_rows[VARIANTS["raw"]]
    )
    columns = ["sentence_id", "category", *SCORE_COLUMNS]
    logger.info(
        "biggest drops with Haiku:\n"
        + changes.nsmallest(5, "change_hk")[columns].to_string(index=False)
    )
    logger.info(
        "biggest rises with Haiku:\n"
        + changes.nlargest(5, "change_hk")[columns].to_string(index=False)
    )


def main(args: argparse.Namespace) -> None:
    logger.info(json.dumps(vars(args), ensure_ascii=False, default=str))
    {"select": run_select, "summarise": run_summarise}[args.step](args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--step",
        choices=["select", "summarise"],
        required=True,
        help="select writes the file to tag, summarise reads it after tagging.",
    )
    # data
    parser.add_argument(
        "--path_scores",
        type=Path,
        default=Path("results/test_set_scores.csv"),
        help="CSV from export_hate_sentences.py --subset all.",
    )
    parser.add_argument(
        "--path_tag_file",
        type=Path,
        default=Path("results/false_alarms_to_tag.csv"),
        help="CSV for manual tagging with category and label_noise columns.",
    )
    parser.add_argument(
        "--path_summary",
        type=Path,
        default=Path("results/false_alarms_summary.csv"),
        help="CSV with the per-category summary written by the summarise step.",
    )
    # hyperparams
    parser.add_argument("--threshold", type=float, default=0.5, help="Hate threshold.")
    parser.add_argument(
        "--top_k", type=int, default=20, help="Non-hate sentences taken per variant."
    )
    # saving
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing tag file or summary file."
    )
    cmd_args = parser.parse_args()
    main(cmd_args)
