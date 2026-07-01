from __future__ import annotations

import math

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from vmecdash.theme import PlotTheme

# Dashboard panels, in order. Each entry is (profile key, subplot title, trace name, color).
# Edit this one list to change which profiles the Summary Dashboard shows.
OVERVIEW_PROFILES = (
    ("iotaf", "Rotational Transform (iota)", "iota", "#22b8cf"),
    ("q", "Safety Factor (q)", "q", "#fd7e14"),
    ("presf", "Pressure Profile", "pres", "#fa5252"),
    ("dpds", "dP/ds", "dP/ds", "#12b886"),
    ("vp", "Volume Derivative (Vp)", "Vp", "#7950f2"),
    ("bdotb", "Flux Avg <B.B>", "<B.B>", "#0ca678"),
)
_OVERVIEW_COLS = 3


def overview_figure(vmec, theme: PlotTheme) -> go.Figure:
    rows = max(1, math.ceil(len(OVERVIEW_PROFILES) / _OVERVIEW_COLS))
    fig = make_subplots(
        rows=rows,
        cols=_OVERVIEW_COLS,
        subplot_titles=tuple(title for _, title, _, _ in OVERVIEW_PROFILES),
        vertical_spacing=0.15,
        horizontal_spacing=0.08,
    )

    def get_data_safe(key):
        try:
            return vmec.get_1d_data(key)
        except Exception:
            return [], []

    for index, (key, _title, name, color) in enumerate(OVERVIEW_PROFILES):
        s_values, y_values = get_data_safe(key)
        row = index // _OVERVIEW_COLS + 1
        col = index % _OVERVIEW_COLS + 1
        fig.add_trace(
            go.Scatter(x=s_values, y=y_values, name=name, line=dict(color=color, width=3)),
            row=row,
            col=col,
        )

    fig.update_layout(
        title_text="Equilibrium Summary",
        template=theme.fig_template,
        paper_bgcolor=theme.paper_bg,
        plot_bgcolor=theme.plot_bg,
        showlegend=False,
        uirevision=f"overview-{theme.reset_seed}",
    )
    return fig
