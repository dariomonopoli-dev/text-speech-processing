"""Plot the annotation and GAHD score figures used in the report."""

import argparse
import json
from collections.abc import Callable
from itertools import combinations
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    average_precision_score,
    cohen_kappa_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from get_loggers import get_logger
from gold_labels import get_shared_labels, load_annotations, load_gold_labels
from visualizations import (
    GRID_COLOR,
    HATE_COLOR,
    MUTED_INK,
    NOT_HATE_COLOR,
    SECONDARY_INK,
    SEQUENTIAL_BLUE,
    apply_style,
)

logger = get_logger("plot")

FIGURE_WIDTH = 900
FIGURE_HEIGHT = 360
MIN_SENTENCES_PER_DOMAIN = 20
NUM_SCORE_BINS = 20
DECISION_THRESHOLD = 0.5
TITLE_X = 0.015
MASTER_SAMPLE_COLUMNS = ["austrotox_stratum", "hate_score", "domain"]
SUBSET_LEGEND = {
    "shared": "shared by all 3 annotators (100)",
    "single-annotator": "labelled by 1 annotator (200 each)",
}


def load_scored_sentences(
    path_annotations: Path, path_master_sample: Path
) -> pd.DataFrame:
    """Join the gold labels with the GAHD score, stratum and domain of each sentence."""
    if not path_master_sample.is_file():
        raise FileNotFoundError(f"master sample not found: {path_master_sample}")
    master_sample = pd.read_csv(path_master_sample, encoding="utf-8-sig")
    missing_columns = [
        column
        for column in ["sentence_id", *MASTER_SAMPLE_COLUMNS]
        if column not in master_sample.columns
    ]
    if missing_columns:
        raise KeyError(f"{path_master_sample} lacks columns {missing_columns}")
    master_sample = master_sample.set_index("sentence_id")
    return load_gold_labels(path_annotations).join(master_sample[MASTER_SAMPLE_COLUMNS])


def compute_hate_counts(sentences: pd.DataFrame, by: str) -> pd.DataFrame:
    """Count hate and total sentences per group."""
    grouped = sentences.groupby(by)["label"]
    return pd.DataFrame({"hate": grouped.sum(), "total": grouped.size()})


def compute_annotator_hate_counts(path_annotations: Path) -> pd.DataFrame:
    """Count each annotator's hate labels on the shared and single-annotator subsets."""
    annotations = load_annotations(path_annotations)
    shared_ids = get_shared_labels(annotations).index
    rows = []
    for annotator, labels in annotations.items():
        is_shared = labels.index.isin(shared_ids)
        for subset, subset_labels in (
            ("shared", labels[is_shared]),
            ("single-annotator", labels[~is_shared]),
        ):
            rows.append(
                {
                    "annotator": annotator.capitalize(),
                    "subset": subset,
                    "hate": int(subset_labels.sum()),
                    "total": len(subset_labels),
                }
            )
    return pd.DataFrame(rows)


def compute_pairwise_kappas(shared_labels: pd.DataFrame) -> pd.DataFrame:
    """Square matrix of Cohen's kappa between annotators, 1 on the diagonal."""
    annotators = list(shared_labels.columns)
    kappas = pd.DataFrame(1.0, index=annotators, columns=annotators)
    for first, second in combinations(annotators, 2):
        kappa = cohen_kappa_score(shared_labels[first], shared_labels[second])
        kappas.loc[first, second] = kappas.loc[second, first] = kappa
    return kappas.rename(index=str.capitalize, columns=str.capitalize)


def to_count_labels(hate_counts: pd.DataFrame) -> list[str]:
    """Format hate counts as hate/total bar labels."""
    return [
        f"{hate}/{total}"
        for hate, total in zip(hate_counts["hate"], hate_counts["total"])
    ]


def plot_annotation_overview(
    path_annotations: Path, path_master_sample: Path
) -> go.Figure:
    """Hate rate per GAHD score stratum and per annotator."""
    strata = compute_hate_counts(
        load_scored_sentences(path_annotations, path_master_sample),
        by="austrotox_stratum",
    )
    annotators = compute_annotator_hate_counts(path_annotations)
    figure = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.5, 0.5],
        subplot_titles=(
            "Hate rate by GAHD score stratum",
            "Hate rate by annotator",
        ),
        horizontal_spacing=0.12,
    )
    figure.add_trace(
        go.Bar(
            x=strata.index,
            y=100 * strata["hate"] / strata["total"],
            text=to_count_labels(strata),
            textposition="outside",
            marker_color=HATE_COLOR,
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    for subset, color in (
        ("shared", NOT_HATE_COLOR),
        ("single-annotator", HATE_COLOR),
    ):
        subset_counts = annotators[annotators["subset"] == subset]
        figure.add_trace(
            go.Bar(
                name=SUBSET_LEGEND[subset],
                x=subset_counts["annotator"],
                y=100 * subset_counts["hate"] / subset_counts["total"],
                text=to_count_labels(subset_counts),
                textposition="outside",
                marker_color=color,
            ),
            row=1,
            col=2,
        )
    figure.update_yaxes(title_text="Label 1 (%)", range=[0, 18], row=1, col=1)
    figure.update_yaxes(range=[0, 18], row=1, col=2)
    figure.update_xaxes(title_text="GAHD hate score", row=1, col=1)
    figure.update_layout(
        title_text="Sentences labelled hate, by score stratum and by annotator",
        barmode="group",
        width=FIGURE_WIDTH,
        height=FIGURE_HEIGHT + 40,
        margin=dict(l=60, r=20, t=130, b=80),
        legend=dict(y=1.15),
    )
    figure.update_annotations(font=dict(size=13, color=SECONDARY_INK))
    figure.update_traces(textfont=dict(size=11, color=MUTED_INK))
    return figure


def plot_agreement_overview(
    path_annotations: Path, path_master_sample: Path
) -> go.Figure:
    """Pairwise Cohen's kappa and the number of hate votes per shared sentence."""
    shared_labels = get_shared_labels(load_annotations(path_annotations))
    kappas = compute_pairwise_kappas(shared_labels)
    num_raters = shared_labels.shape[1]
    hate_votes = (
        shared_labels.sum(axis=1)
        .value_counts()
        .reindex(range(num_raters + 1), fill_value=0)
    )
    figure = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.45, 0.55],
        subplot_titles=(
            "Pairwise agreement between annotators (Cohen's kappa)",
            f"How many of the {num_raters} annotators voted hate, per shared sentence",
        ),
        horizontal_spacing=0.14,
    )
    figure.add_trace(
        go.Heatmap(
            z=kappas.to_numpy(),
            x=kappas.columns,
            y=kappas.index,
            zmin=0.4,
            zmax=1,
            zsmooth=False,
            colorscale=SEQUENTIAL_BLUE,
            text=kappas.round(2).to_numpy(),
            texttemplate="%{text}",
            showscale=False,
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            x=[f"{votes} of {num_raters}" for votes in hate_votes.index],
            y=hate_votes.to_numpy(),
            text=hate_votes.to_numpy(),
            textposition="outside",
            marker_color=[NOT_HATE_COLOR] + [HATE_COLOR] * num_raters,
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    figure.update_yaxes(autorange="reversed", row=1, col=1)
    figure.update_yaxes(
        title_text=f"Shared sentences (n={len(shared_labels)})",
        range=[0, 105],
        row=1,
        col=2,
    )
    figure.update_xaxes(title_text="Annotators voting hate", row=1, col=2)
    figure.update_layout(
        title_text=f"Annotator agreement on the {len(shared_labels)} shared sentences",
        width=FIGURE_WIDTH,
        height=FIGURE_HEIGHT,
        margin=dict(l=80, r=20, t=90, b=80),
    )
    figure.update_annotations(font=dict(size=12, color=SECONDARY_INK))
    figure.update_traces(
        textfont=dict(size=11, color=MUTED_INK), selector=dict(type="bar")
    )
    return figure


def plot_score_distribution(
    path_annotations: Path, path_master_sample: Path
) -> go.Figure:
    """Histogram of GAHD scores per human label."""
    sentences = load_scored_sentences(path_annotations, path_master_sample)
    figure = go.Figure()
    for label, label_name, color in (
        (0, "not hate", NOT_HATE_COLOR),
        (1, "hate", HATE_COLOR),
    ):
        is_label = sentences["label"] == label
        figure.add_trace(
            go.Histogram(
                x=sentences.loc[is_label, "hate_score"],
                name=f"human label: {label_name} (n={int(is_label.sum())})",
                histnorm="probability",
                xbins=dict(start=0, end=1, size=1 / NUM_SCORE_BINS),
                marker_color=color,
            )
        )
    figure.add_vline(
        x=DECISION_THRESHOLD,
        line_dash="dash",
        line_color=MUTED_INK,
        line_width=1.5,
        annotation_text=f"decision threshold {DECISION_THRESHOLD}",
        annotation_position="top left",
        annotation_font_color=SECONDARY_INK,
    )
    figure.update_layout(
        title_text=(
            f"GAHD classifier score of the {len(sentences)} test sentences, "
            "split by the human label"
        ),
        barmode="group",
        width=FIGURE_WIDTH,
        height=FIGURE_HEIGHT,
        margin=dict(l=80, r=20, t=90, b=80),
        xaxis_title="Hate score of the GAHD classifier (0 to 1)",
        yaxis_title="Share of sentences<br>with that human label",
    )
    return figure


def plot_scorer_curves(path_annotations: Path, path_master_sample: Path) -> go.Figure:
    """Precision-recall and ROC curves of the GAHD score against the gold labels."""
    sentences = load_scored_sentences(path_annotations, path_master_sample)
    true_labels, scores = sentences["label"], sentences["hate_score"]
    precision, recall, _ = precision_recall_curve(true_labels, scores)
    false_positive_rate, true_positive_rate, _ = roc_curve(true_labels, scores)
    average_precision = average_precision_score(true_labels, scores)
    true_hs_ratio = true_labels.mean()
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"Precision-recall (average precision {average_precision:.2f}, "
            f"dashed = share of hate {true_hs_ratio:.1%})",
            f"ROC curve (AUC {roc_auc_score(true_labels, scores):.2f}, "
            "dashed = chance)",
        ),
        horizontal_spacing=0.12,
    )
    figure.add_trace(
        go.Scatter(
            x=recall[:-1],
            y=precision[:-1],
            mode="lines",
            line=dict(color=HATE_COLOR, width=2.5),
        ),
        row=1,
        col=1,
    )
    figure.add_hline(
        y=true_hs_ratio, line_dash="dash", line_color=MUTED_INK, row=1, col=1
    )
    figure.add_trace(
        go.Scatter(
            x=false_positive_rate,
            y=true_positive_rate,
            mode="lines",
            line=dict(color=NOT_HATE_COLOR, width=2.5),
        ),
        row=1,
        col=2,
    )
    figure.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", line=dict(color=MUTED_INK, dash="dash")
        ),
        row=1,
        col=2,
    )
    figure.update_xaxes(title_text="Recall", row=1, col=1)
    figure.update_yaxes(title_text="Precision", range=[0, 1], row=1, col=1)
    figure.update_xaxes(title_text="False positive rate", row=1, col=2)
    figure.update_yaxes(title_text="True positive rate", range=[0, 1], row=1, col=2)
    figure.update_layout(
        title_text="The GAHD classifier scores as a detector on our human labels",
        width=FIGURE_WIDTH,
        height=FIGURE_HEIGHT,
        margin=dict(l=60, r=20, t=90, b=80),
        showlegend=False,
    )
    figure.update_annotations(font=dict(size=12, color=SECONDARY_INK))
    return figure


def plot_domain_hate_rates(
    path_annotations: Path, path_master_sample: Path
) -> go.Figure:
    """Hate rate per source website with enough sampled sentences."""
    domains = compute_hate_counts(
        load_scored_sentences(path_annotations, path_master_sample), by="domain"
    )
    domains = domains[domains["total"] >= MIN_SENTENCES_PER_DOMAIN]
    domains = domains.assign(rate=100 * domains["hate"] / domains["total"])
    domains = domains.sort_values("rate")
    figure = go.Figure(
        go.Bar(
            x=domains["rate"],
            y=domains.index,
            orientation="h",
            text=to_count_labels(domains),
            textposition="outside",
            marker_color=HATE_COLOR,
        )
    )
    figure.update_xaxes(
        title_text="Sentences labelled hate (%)",
        range=[0, 18],
        showgrid=True,
        gridcolor=GRID_COLOR,
    )
    figure.update_yaxes(showgrid=False)
    figure.update_layout(
        title=dict(
            text="Share of test sentences labelled hate, by source website",
            subtitle=dict(
                text=(
                    "labels show hate sentences / sampled sentences, websites with "
                    f"at least {MIN_SENTENCES_PER_DOMAIN} sentences"
                ),
                font=dict(size=12, color=SECONDARY_INK),
            ),
        ),
        width=FIGURE_WIDTH,
        height=FIGURE_HEIGHT + 60,
        margin=dict(l=150, r=30, t=90, b=80),
    )
    figure.update_traces(textfont=dict(size=11, color=MUTED_INK))
    return figure


FIGURES: dict[str, Callable[[Path, Path], go.Figure]] = {
    "annotation_overview": plot_annotation_overview,
    "agreement_overview": plot_agreement_overview,
    "score_distribution": plot_score_distribution,
    "scorer_curves": plot_scorer_curves,
    "domain_hate_rates": plot_domain_hate_rates,
}


def main(args: argparse.Namespace) -> None:
    logger.info(json.dumps(vars(args), ensure_ascii=False, default=str))
    path_figures_dir = Path(args.path_figures_dir)
    path_figures_dir.mkdir(parents=True, exist_ok=True)
    apply_style()
    for figure_name, plot_figure in FIGURES.items():
        figure = plot_figure(Path(args.path_annotations), Path(args.path_master_sample))
        figure.update_layout(title_xref="container", title_x=TITLE_X)
        figure.write_image(path_figures_dir / f"{figure_name}.pdf")
        figure.write_html(
            path_figures_dir / f"{figure_name}.html", include_plotlyjs="cdn"
        )
        logger.info(f"Wrote {figure_name} to {path_figures_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    # data
    parser.add_argument(
        "--path_annotations",
        default="data",
        help="Directory with annotation_<annotator>_stratified.csv files.",
    )
    parser.add_argument(
        "--path_master_sample",
        default="data/master_sample_stratified.csv",
        help="Master sample CSV with GAHD score, stratum and domain per sentence.",
    )
    # saving
    parser.add_argument(
        "--path_figures_dir",
        required=True,
        help="Directory for one .pdf and one .html file per figure.",
    )
    cmd_args = parser.parse_args()
    main(cmd_args)
