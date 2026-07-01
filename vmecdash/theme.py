from __future__ import annotations

from dataclasses import dataclass

import plotly.graph_objects as go


@dataclass
class PlotTheme:
    fig_template: str
    paper_bg: str
    plot_bg: str
    dark_mode: bool
    text_color: str
    reset_seed: int


def build_theme(dark_mode: bool, reset_seed: int = 0) -> PlotTheme:
    fig_template = "plotly_dark" if dark_mode else "plotly_white"
    paper_bg = "#2e2e2e" if dark_mode else "white"
    plot_bg = "#2e2e2e" if dark_mode else "white"
    text_color = "#adb5bd" if dark_mode else "#495057"
    return PlotTheme(fig_template, paper_bg, plot_bg, dark_mode, text_color, reset_seed)


def make_empty_figure(theme: PlotTheme, message: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        autosize=True,
        template=theme.fig_template,
        paper_bgcolor=theme.paper_bg,
        plot_bgcolor=theme.plot_bg,
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    fig.add_annotation(text=message, showarrow=False, font=dict(size=20, color=theme.text_color))
    return fig

