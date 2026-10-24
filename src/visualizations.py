"""Plotly style and plotting helpers shared by the notebooks."""

import re
from collections import Counter
from collections.abc import Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    recall_score,
)
from sklearn.metrics.pairwise import paired_cosine_distances

NOT_HATE_COLOR = "#2a78d6"
HATE_COLOR = "#eb6834"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED_INK = "#898781"
GRID_COLOR = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"
SURFACE = "#fcfcfb"
SEQUENTIAL_BLUE = ["#cde2fb", "#86b6ef", "#2a78d6", "#104281"]
FONT_FAMILY = "system-ui, -apple-system, Segoe UI, sans-serif"
LABEL_NAMES = ["not hate", "hate"]
CLASS_COLORS = {"not hate": NOT_HATE_COLOR, "hate": HATE_COLOR}
ROUNDS = ["R1", "R2", "R3", "R4"]
DECISION_THRESHOLD = 0.5

N_RESAMPLES = 1000
N_SIMILARITY_BINS = 4
N_WORDS_PER_CLASS = 15
MIN_WORD_COUNT = 5
MIN_WORD_LENGTH = 3


def apply_style() -> None:
    """Register the gahd plotly template and make it the default."""
    axis = dict(
        gridcolor=GRID_COLOR,
        linecolor=AXIS_COLOR,
        zeroline=False,
        ticks="",
        tickfont=dict(color=SECONDARY_INK),
        title=dict(font=dict(color=SECONDARY_INK)),
    )
    pio.templates["gahd"] = go.layout.Template(
        layout=dict(
            font=dict(family=FONT_FAMILY, color=INK, size=13),
            paper_bgcolor=SURFACE,
            plot_bgcolor=SURFACE,
            title=dict(x=0, xref="paper", font=dict(size=16)),
            xaxis=dict(axis, showgrid=False, showline=True),
            yaxis=dict(axis, showgrid=True, showline=False),
            legend=dict(
                orientation="h",
                x=1,
                xanchor="right",
                y=1.02,
                yanchor="bottom",
                title_font=dict(size=13, color=SECONDARY_INK),
            ),
            hoverlabel=dict(
                bgcolor="white",
                bordercolor=AXIS_COLOR,
                font=dict(family=FONT_FAMILY, color=INK),
            ),
            bargap=0.3,
        )
    )
    pio.templates.default = "plotly_white+gahd"
    pio.renderers.default = "plotly_mimetype+notebook_connected"


def with_predictions(examples: pd.DataFrame, hate_log_odds: ArrayLike) -> pd.DataFrame:
    """Add predicted label, hate probability and correctness from hate log-odds."""
    pred = (np.asarray(hate_log_odds) > 0).astype(int)
    return examples.assign(
        pred=pred,
        prob_hate=1 / (1 + np.exp(-np.asarray(hate_log_odds))),
        correct=pred == examples["label"].to_numpy(),
    )


def add_reference_line(
    fig: go.Figure,
    value: float,
    label: str,
    axis: str = "y",
    position: str = "top right",
) -> None:
    """Draw a labelled dashed reference line on either axis."""
    add_line = fig.add_hline if axis == "y" else fig.add_vline
    add_line(
        value,
        line_dash="dash",
        line_color=MUTED_INK,
        line_width=1.5,
        annotation_text=label,
        annotation_position=position,
        annotation_font_color=SECONDARY_INK,
    )


def plot_confusion_matrix(results: pd.DataFrame) -> None:
    """Show the row-normalised confusion matrix with raw counts."""
    counts = confusion_matrix(results["label"], results["pred"])
    rates = counts / counts.sum(axis=1, keepdims=True)

    fig = px.imshow(
        rates,
        x=LABEL_NAMES,
        y=LABEL_NAMES,
        zmin=0,
        zmax=1,
        color_continuous_scale=SEQUENTIAL_BLUE,
        labels=dict(x="Predicted", y="Ground truth", color="Share"),
        title="Confusion matrix, GAHD test",
        width=460,
        height=440,
    )
    fig.update_traces(
        text=counts,
        texttemplate="<b>%{z:.0%}</b><br>%{text}",
        textfont_size=16,
        xgap=2,
        ygap=2,
        hovertemplate="Ground truth %{y}, predicted %{x}<br><b>%{z:.1%}</b> (%{text} examples)<extra></extra>",
    )
    fig.update_coloraxes(showscale=False)
    fig.update_xaxes(showline=False)
    fig.update_yaxes(showgrid=False)
    fig.show()


def compute_macro_f1_interval(
    truth: ArrayLike, predicted: ArrayLike, seed: int = 0
) -> np.ndarray:
    """Percentile 95 percent bootstrap interval of macro F1."""
    truth, predicted = np.asarray(truth), np.asarray(predicted)
    resamples = np.random.default_rng(seed).integers(
        0, len(truth), size=(N_RESAMPLES, len(truth))
    )
    scores = [
        f1_score(
            truth[r], predicted[r], average="macro", labels=[0, 1], zero_division=0
        )
        for r in resamples
    ]
    return np.percentile(scores, [2.5, 97.5])


def compute_round_summary(results: pd.DataFrame) -> pd.Series:
    """Macro F1 with interval and per-class recall for one set of results."""
    low, high = compute_macro_f1_interval(results["label"], results["pred"])
    return pd.Series(
        {
            "examples": len(results),
            "hate_share": results["label"].mean(),
            "f1_macro": f1_score(results["label"], results["pred"], average="macro"),
            "ci_low": low,
            "ci_high": high,
            "not hate": recall_score(
                results["label"], results["pred"], pos_label=0, zero_division=np.nan
            ),
            "hate": recall_score(
                results["label"], results["pred"], pos_label=1, zero_division=np.nan
            ),
        }
    )


def plot_rounds(results: pd.DataFrame) -> None:
    """Show macro F1 and per-class recall per GAHD collection round."""
    by_round = pd.concat(
        [
            results.groupby("round")
            .apply(compute_round_summary, include_groups=False)
            .loc[ROUNDS],
            compute_round_summary(results).to_frame("all").T,
        ]
    )
    by_round["round"] = [
        f"<b>{name}</b><br>n={row.examples:.0f}<br>{row.hate_share:.0%} hate"
        for name, row in by_round.iterrows()
    ]

    fig = px.scatter(
        by_round,
        x="round",
        y="f1_macro",
        error_y=by_round["ci_high"] - by_round["f1_macro"],
        error_y_minus=by_round["f1_macro"] - by_round["ci_low"],
        hover_data=dict(round=False, f1_macro=":.3f", ci_low=":.3f", ci_high=":.3f"),
        labels=dict(
            round="Collection round",
            f1_macro="Macro F1",
            ci_low="95% CI low",
            ci_high="95% CI high",
        ),
        title="Macro F1 per collection round (95% bootstrap CI)",
        height=420,
    )
    fig.update_traces(
        marker=dict(size=12, color=INK, line=dict(color=SURFACE, width=2)),
        error_y=dict(color=MUTED_INK, thickness=2, width=0),
    )
    fig.update_yaxes(range=[0, 1])
    fig.show()

    recall = by_round.melt(
        id_vars="round",
        value_vars=LABEL_NAMES,
        var_name="ground_truth",
        value_name="recall",
    )
    fig = px.bar(
        recall,
        x="round",
        y="recall",
        color="ground_truth",
        barmode="group",
        color_discrete_map=CLASS_COLORS,
        hover_data=dict(round=False, recall=":.1%"),
        labels=dict(
            round="Collection round", recall="Recall", ground_truth="Ground truth"
        ),
        title="Recall per class and collection round",
        height=420,
    )
    fig.update_traces(marker_cornerradius=4)
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    fig.show()


def get_contrastive_examples(results: pd.DataFrame, gahd: pd.DataFrame) -> pd.DataFrame:
    """Return label-flipping contrastive pairs with their character similarity."""
    by_id = gahd.set_index("gahd_id")
    contrastive = results.assign(
        original_text=results["contrastive_gahd_id"].map(by_id["text"]),
        original_label=results["contrastive_gahd_id"].map(by_id["label"]),
    )
    contrastive = contrastive[
        contrastive["original_label"].notna()
        & (contrastive["label"] != contrastive["original_label"])
    ]
    character_ngrams = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5)).fit(
        gahd["text"]
    )
    similarity = 1 - paired_cosine_distances(
        character_ngrams.transform(contrastive["text"]),
        character_ngrams.transform(contrastive["original_text"]),
    )
    return contrastive.assign(similarity=similarity)


def plot_contrastive(results: pd.DataFrame, gahd: pd.DataFrame) -> None:
    """Show accuracy on contrastive examples by similarity quartile."""
    contrastive = get_contrastive_examples(results, gahd)
    contrastive["similarity_bin"] = pd.qcut(
        contrastive["similarity"], N_SIMILARITY_BINS
    )
    by_similarity = (
        contrastive.groupby("similarity_bin", observed=True)["correct"]
        .agg(accuracy="mean", correct="sum", examples="size")
        .reset_index()
    )
    by_similarity["similarity"] = [
        f"{interval.left:.2f}-{interval.right:.2f}<br>n={n}"
        for interval, n in zip(
            by_similarity["similarity_bin"], by_similarity["examples"]
        )
    ]
    other_accuracy = results.loc[
        ~results.index.isin(contrastive.index), "correct"
    ].mean()

    fig = px.bar(
        by_similarity,
        x="similarity",
        y="accuracy",
        hover_data=dict(similarity=False, accuracy=":.1%", correct=True, examples=True),
        labels=dict(
            similarity="Character n-gram similarity to the original (quartiles)",
            accuracy="Accuracy",
            correct="Correct",
            examples="Examples",
        ),
        title=f"Contrastive R4 test examples (n={len(contrastive)})",
        height=420,
    )
    fig.update_traces(marker_color=NOT_HATE_COLOR, marker_cornerradius=4, width=0.6)
    add_reference_line(fig, other_accuracy, f"rest of test set {other_accuracy:.0%}")
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    fig.show()


def plot_precision_recall(results: pd.DataFrame, model_name: str) -> None:
    """Show the precision-recall curve with the default threshold marked."""
    precision, recall, thresholds = precision_recall_curve(
        results["label"], results["prob_hate"]
    )
    curve = pd.DataFrame(
        dict(
            recall=recall, precision=precision, threshold=np.append(thresholds, np.nan)
        )
    )
    default_point = curve.iloc[np.searchsorted(thresholds, DECISION_THRESHOLD)]
    average_precision = average_precision_score(results["label"], results["prob_hate"])
    hate_share = results["label"].mean()

    fig = px.area(
        curve,
        x="recall",
        y="precision",
        hover_data=dict(threshold=":.3f", precision=":.3f", recall=":.3f"),
        labels=dict(
            recall="Recall (hate)", precision="Precision (hate)", threshold="Threshold"
        ),
        title=f"Precision-recall curve, {model_name} (AP={average_precision:.2f})",
        width=560,
        height=520,
    )
    fig.update_traces(line_color=HATE_COLOR, fillcolor="rgba(235, 104, 52, 0.12)")
    fig.add_scatter(
        x=[default_point["recall"]],
        y=[default_point["precision"]],
        mode="markers",
        name=f"threshold {DECISION_THRESHOLD}",
        marker=dict(size=12, color=HATE_COLOR, line=dict(color=SURFACE, width=2)),
        hovertemplate=f"Threshold {DECISION_THRESHOLD}<br>Precision %{{y:.3f}}<br>Recall %{{x:.3f}}<extra></extra>",
    )
    add_reference_line(
        fig, hate_share, f"hate share {hate_share:.0%}", position="bottom left"
    )
    fig.update_yaxes(range=[0, 1.02], scaleanchor="x", scaleratio=1)
    fig.update_xaxes(range=[0, 1], constrain="domain", showgrid=True)
    fig.show()


def frequent_words(texts: Iterable[str]) -> list[str]:
    """Return words that occur often enough and are long enough to score."""
    counts = Counter(re.findall(r"\w+", " ".join(texts).lower()))
    return [
        word
        for word, count in counts.items()
        if count >= MIN_WORD_COUNT and len(word) >= MIN_WORD_LENGTH
    ]


def compute_word_training_stats(
    words: Sequence[str], training_examples: pd.DataFrame
) -> pd.DataFrame:
    """Count training sentences per word and their hate share."""
    texts = training_examples["text"].str.lower()
    contains = [texts.str.contains(rf"\b{re.escape(word)}\b") for word in words]
    return pd.DataFrame(
        dict(
            sentences=[mask.sum() for mask in contains],
            hate_share=[
                training_examples.loc[mask, "label"].mean() for mask in contains
            ],
        ),
        index=words,
    )


def plot_word_scores(word_scores: pd.Series, training_examples: pd.DataFrame) -> None:
    """Show the words with the most extreme log-odds of hate."""
    ranked = word_scores.sort_values()
    extremes = pd.concat(
        [ranked.head(N_WORDS_PER_CLASS), ranked.tail(N_WORDS_PER_CLASS)]
    )
    words = (
        compute_word_training_stats(extremes.index, training_examples)
        .assign(
            log_odds=extremes,
            prob_hate=1 / (1 + np.exp(-extremes)),
            classified_as=np.where(extremes > 0, "hate", "not hate"),
        )
        .rename_axis("word")
        .reset_index()
    )

    fig = px.bar(
        words,
        x="log_odds",
        y="word",
        color="classified_as",
        orientation="h",
        color_discrete_map=CLASS_COLORS,
        category_orders=dict(word=list(words["word"][::-1]), classified_as=LABEL_NAMES),
        hover_data=dict(
            classified_as=False,
            log_odds=":.2f",
            prob_hate=":.1%",
            sentences=True,
            hate_share=":.0%",
        ),
        labels=dict(
            log_odds="Log-odds of hate for the word alone",
            word="",
            classified_as="Classified as",
            prob_hate="P(hate)",
            sentences="Training sentences with the word",
            hate_share="Of those, labelled hate",
        ),
        title="Words classified on their own",
        height=780,
    )
    fig.update_traces(marker_cornerradius=4)
    fig.add_vline(x=0, line_color=AXIS_COLOR, line_width=1)
    fig.update_xaxes(showgrid=True, showline=False)
    fig.update_yaxes(showgrid=False, dtick=1)
    fig.show()
