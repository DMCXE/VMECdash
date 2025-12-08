from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import dash_mantine_components as dmc
import plotly.graph_objects as go

from ui.components import create_stat_card


@dataclass
class PlotTheme:
    fig_template: str
    paper_bg: str
    plot_bg: str
    dark_mode: bool
    text_color: str
    reset_seed: int


def build_theme(dark_mode: bool, reset_seed: int) -> PlotTheme:
    fig_template = "plotly_dark" if dark_mode else "plotly_white"
    # Use solid backgrounds so downloads match on-screen dark mode rendering.
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


def base_stat_cards(scalars: dict[str, Any]) -> dmc.SimpleGrid:
    beta = scalars.get("beta_total", 0.0)
    vol = scalars.get("volume", 0.0)
    curr = scalars.get("ctor", 0.0)
    aspect_ratio = scalars.get("Rmajor") / scalars.get("Aminor") if scalars.get("Aminor") else math.nan
    ar_str = f"{aspect_ratio:.2f}" if aspect_ratio == aspect_ratio else "N/A"

    return dmc.SimpleGrid(
        cols=4,
        children=[
            create_stat_card("Beta Total", f"{beta*100:.2f}%", "mdi:percent", "red"),
            create_stat_card("Volume", f"{vol:.2f} m³", "mdi:cube-outline", "blue"),
            create_stat_card("Aspect Ratio", ar_str, "lucide:torus", "orange"),
            create_stat_card("Toroidal Current", f"{curr:.2f} A", "mdi:current-ac", "teal"),
        ],
    )
