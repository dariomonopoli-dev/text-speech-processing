"""Shared Plotly colors, labels, and style configuration."""

import plotly.graph_objects as go
import plotly.io as pio

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
    """Register the gahd Plotly template and make it the default."""
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
