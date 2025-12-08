from __future__ import annotations

import numpy as np
import dash_mantine_components as dmc
import plotly.graph_objects as go
from dash import dcc

from ui.components import get_icon
from views.shared import PlotTheme


def controls():
    return dmc.Paper(
        id="wrapper-3d",
        withBorder=True,
        shadow="xs",
        radius="md",
        p="md",
        style={"display": "none"},
        children=[
            dmc.Text("Rendering 3D surfaces using the accelerated JAX backend.", size="sm", c="dimmed", mb="sm"),
            dmc.Select(id="ctrl-3d-var", label="Surface Color", value="modB", data=[], mb="md", allowDeselect=False),
            dmc.Stack(
                [
                    dmc.Text("Flux Surface (s)", size="sm", fw=500),
                    dcc.Slider(
                        id="ctrl-s3d-idx",
                        min=0,
                        max=10,
                        step=1,
                        value=10,
                        marks={0: "Axis", 10: "Edge"},
                        tooltip={"placement": "bottom"},
                    ),
                    dmc.Switch(
                        id="ctrl-3d-bg",
                        label="Coordinate-free Background",
                        checked=True,
                        color="cyan",
                        mt="sm",
                    ),
                ],
                gap="xs",
            ),
        ],
    )


def render_3d(vmec, s_val: int, v_name: str, field_label: str, coord_free: bool, theme: PlotTheme):
    fig = go.Figure()
    x, y, z, val = vmec.compute_3d_surface(s_idx=s_val, var_name=v_name, resolution=100)

    if v_name == "geometry":
        surf_color = np.full_like(z, 0.5)
        cscale = "Greys"
    else:
        surf_color = val
        cscale = "Jet"

    fig.add_trace(
        go.Surface(
            x=x,
            y=y,
            z=z,
            surfacecolor=surf_color,
            colorscale=cscale,
            colorbar=dict(title=field_label, len=0.6) if v_name != "geometry" else None,
        )
    )

    axis_color = "#dee2e6" if theme.dark_mode else "#495057"
    grid_color = "#5c677d" if theme.dark_mode else "#ced4da"
    axis_style = (
        dict(visible=False)
        if coord_free
        else dict(
            visible=True,
            backgroundcolor="rgba(0,0,0,0)",
            gridcolor=grid_color,
            zerolinecolor=grid_color,
            color=axis_color,
            title=dict(font=dict(color=axis_color)),
            tickfont=dict(color=axis_color),
        )
    )
    title_txt = f"Surface s={s_val/(vmec.ns-1):.2f}"
    if v_name != "geometry":
        title_txt += f" colored by {field_label}"

    fig.update_layout(
        title=title_txt,
        template=theme.fig_template,
        paper_bgcolor=theme.paper_bg,
        scene=dict(bgcolor=theme.plot_bg, xaxis=axis_style, yaxis=axis_style, zaxis=axis_style, aspectmode="data"),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    return fig
