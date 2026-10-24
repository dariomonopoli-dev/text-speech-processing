"""Compute label counts and inter-annotator agreement for the stratified annotations."""

import argparse
import json
from itertools import combinations
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from get_loggers import get_logger
from gold_labels import get_shared_labels, load_annotations

logger = get_logger("agreement")


def compute_fleiss_kappa(shared_labels: pd.DataFrame) -> float:
    """Fleiss' kappa for binary labels, one column per annotator."""
    num_raters = shared_labels.shape[1]
    num_pos = shared_labels.sum(axis=1)
    num_neg = num_raters - num_pos
    agreement = (num_pos * (num_pos - 1) + num_neg * (num_neg - 1)) / (
        num_raters * (num_raters - 1)
    )
    pos_ratio = shared_labels.to_numpy().mean()
    expected = pos_ratio**2 + (1 - pos_ratio) ** 2
    return float((agreement.mean() - expected) / (1 - expected))


def compute_krippendorff_alpha(shared_labels: pd.DataFrame) -> float:
    """Krippendorff's alpha for binary labels without missing values."""
    num_raters = shared_labels.shape[1]
    num_pos = shared_labels.sum(axis=1)
    num_judgements = len(shared_labels) * num_raters
    observed = (
        2
        * (num_pos * (num_raters - num_pos)).sum()
        / (num_judgements * (num_raters - 1))
    )
    num_pos_total = num_pos.sum()
    expected = (
        2
        * num_pos_total
        * (num_judgements - num_pos_total)
        / (num_judgements * (num_judgements - 1))
    )
    return float(1 - observed / expected)


def compute_label_counts(annotations: dict[str, pd.Series]) -> dict[str, int]:
    """Count hate labels per annotator and over the union of all sentences."""
    labels_per_sentence = pd.concat(annotations.values()).groupby(level=0).agg(list)
    label_counts = {}
    for annotator, labels in annotations.items():
        label_counts[f"num_examples-{annotator}"] = len(labels)
        label_counts[f"num_true_hate-{annotator}"] = int(labels.sum())
    label_counts["num_sentences"] = len(labels_per_sentence)
    label_counts["num_judgements"] = sum(map(len, annotations.values()))
    label_counts["num_judgements_hate"] = int(
        sum(labels.sum() for labels in annotations.values())
    )
    label_counts["num_sentences_any_hate"] = int(labels_per_sentence.map(max).sum())
    label_counts["num_sentences_majority_hate"] = int(
        labels_per_sentence.map(lambda votes: sum(votes) * 2 > len(votes)).sum()
    )
    return label_counts


def compute_shared_agreement(shared_labels: pd.DataFrame) -> dict[str, float]:
    """Compute unanimity and agreement statistics on the shared sentences."""
    num_hate_votes = shared_labels.sum(axis=1)
    num_raters = shared_labels.shape[1]
    agreement_stats = {"num_shared": len(shared_labels)}
    for annotator in shared_labels.columns:
        agreement_stats[f"num_shared_hate-{annotator}"] = int(
            shared_labels[annotator].sum()
        )
    agreement_stats["num_shared_unanimous"] = int(
        (shared_labels.nunique(axis=1) == 1).sum()
    )
    agreement_stats["num_shared_unanimous_hate"] = int(
        (num_hate_votes == num_raters).sum()
    )
    agreement_stats["num_shared_unanimous_not_hate"] = int((num_hate_votes == 0).sum())
    agreement_stats["num_shared_majority_hate"] = int((num_hate_votes >= 2).sum())
    agreement_stats["fleiss-kappa"] = compute_fleiss_kappa(shared_labels)
    agreement_stats["krippendorff-alpha"] = compute_krippendorff_alpha(shared_labels)
    for first, second in combinations(shared_labels.columns, 2):
        agreement_stats[f"agreement-{first}-{second}"] = float(
            (shared_labels[first] == shared_labels[second]).mean()
        )
        agreement_stats[f"cohen-kappa-{first}-{second}"] = float(
            cohen_kappa_score(shared_labels[first], shared_labels[second])
        )
    return agreement_stats


def log_agreement_stats(
    agreement_stats: dict[str, float], annotators: list[str]
) -> None:
    """Log the statistics in the order of the original report."""
    logger.info("label 1 per annotator")
    for annotator in annotators:
        num_hate = agreement_stats[f"num_true_hate-{annotator}"]
        num_examples = agreement_stats[f"num_examples-{annotator}"]
        logger.info(
            f"  {annotator}: {num_hate} / {num_examples} ({num_hate / num_examples:.1%})"
        )
    logger.info(f"unique sentences: {agreement_stats['num_sentences']}")
    logger.info(
        f"  annotations with label 1 (all {agreement_stats['num_judgements']} judgements): "
        f"{agreement_stats['num_judgements_hate']}"
    )
    logger.info(
        f"  sentences with at least one label 1: {agreement_stats['num_sentences_any_hate']}"
    )
    logger.info(
        f"  sentences with majority label 1: {agreement_stats['num_sentences_majority_hate']}"
    )
    logger.info(f"shared sentences: {agreement_stats['num_shared']}")
    shared_hate = {
        annotator: agreement_stats[f"num_shared_hate-{annotator}"]
        for annotator in annotators
    }
    logger.info(f"  label 1 in shared set: {shared_hate}")
    logger.info(
        f"  unanimous: {agreement_stats['num_shared_unanimous']} "
        f"(all 1: {agreement_stats['num_shared_unanimous_hate']}, "
        f"all 0: {agreement_stats['num_shared_unanimous_not_hate']})"
    )
    logger.info(f"  majority label 1: {agreement_stats['num_shared_majority_hate']}")
    logger.info(f"  Fleiss kappa: {agreement_stats['fleiss-kappa']:.3f}")
    logger.info(f"  Krippendorff alpha: {agreement_stats['krippendorff-alpha']:.3f}")
    for first, second in combinations(annotators, 2):
        agreement = agreement_stats[f"agreement-{first}-{second}"]
        kappa = agreement_stats[f"cohen-kappa-{first}-{second}"]
        logger.info(
            f"  {first} vs {second}: agreement {agreement:.2f}, Cohen kappa {kappa:.3f}"
        )


def main(args: argparse.Namespace) -> None:
    logger.info(json.dumps(vars(args), ensure_ascii=False, default=str))
    annotations = load_annotations(Path(args.path_annotations))
    shared_labels = get_shared_labels(annotations)
    if shared_labels.empty:
        raise ValueError(f"no sentence in {args.path_annotations} has all labels")
    agreement_stats = {
        **compute_label_counts(annotations),
        **compute_shared_agreement(shared_labels),
    }
    log_agreement_stats(agreement_stats, list(annotations))

    path_agreement = Path(args.path_agreement)
    path_agreement.parent.mkdir(parents=True, exist_ok=True)
    with open(path_agreement, "w", encoding="utf-8") as fout:
        json.dump(agreement_stats, fout, ensure_ascii=False, indent=4)
    logger.info(f"Wrote agreement statistics to {path_agreement}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    # data
    parser.add_argument(
        "--path_annotations",
        default="data",
        help="Directory with annotation_<annotator>_stratified.csv files.",
    )
    # saving
    parser.add_argument(
        "--path_agreement",
        default="results/annotation_agreement.json",
        help="Path of the flat JSON file with all counts and agreement scores.",
    )
    cmd_args = parser.parse_args()
    main(cmd_args)
