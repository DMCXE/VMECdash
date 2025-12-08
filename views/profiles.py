from __future__ import annotations

import dash_mantine_components as dmc
import plotly.graph_objects as go
from dash import dcc, html

from ui.components import get_icon
from views.shared import PlotTheme


def controls():
    return dmc.Paper(
        id="wrapper-1d",
        withBorder=True,
        shadow="xs",
        radius="md",
        p="md",
        style={"display": "none"},
        children=[
            dmc.Text("Select a flux-surface averaged quantity to inspect versus normalized flux (s).", size="sm", c="dimmed", mb="sm"),
            dmc.ScrollArea(
                h=300,
                type="auto",
                mb="md",
                children=dmc.RadioGroup(id="ctrl-1d-var", label="Profile Variable", value="iotaf", children=[], size="sm"),
            ),
            dmc.Alert(
                "Plots 1D flux surface averaged quantities vs normalized flux (s).",
                color="blue",
                variant="light",
                icon=get_icon("mdi:chart-bell-curve"),
            ),
        ],
    )


def render_profile(vmec, var_name: str, theme: PlotTheme):
    s, y = vmec.get_1d_data(var_name or "iotaf")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s, y=y, mode="lines", line=dict(color="#3bc9db", width=4)))
    fig.update_layout(
        title=f"Profile: {var_name}",
        xaxis_title="Normalized Flux (s)",
        yaxis_title=var_name,
        template=theme.fig_template,
        paper_bgcolor=theme.paper_bg,
        plot_bgcolor=theme.plot_bg,
        uirevision=f"1d-{theme.reset_seed}",
    )
    return fig

