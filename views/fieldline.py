from __future__ import annotations

import numpy as np
import dash_mantine_components as dmc
import plotly.graph_objects as go
from dash import dcc, html
from plotly.subplots import make_subplots

from ui.components import get_icon
from views.shared import PlotTheme


def controls():
    return dmc.Paper(
        id="wrapper-fieldline",
        withBorder=True,
        shadow="xs",
        radius="md",
        p="md",
        style={"display": "none"},
        children=[
            dmc.Text("Trace magnetic field lines and visualize |B| in (alpha, zeta) coordinates.", size="sm", c="dimmed", mb="sm"),
            dmc.Select(
                id="ctrl-fl-type",
                label="Plot Type",
                value="2d_modB",
                data=[
                    {"value": "2d_modB", "label": "2D |B| (alpha, zeta)"},
                    {"value": "1d_lines", "label": "1D Field Lines (Multiple Alphas)"},
                    {"value": "single_trace", "label": "Single Field Line Trace"},
                ],
                mb="md",
                allowDeselect=False,
            ),
            dmc.Stack(
                [
                    dmc.Text("Flux Surface (s)", size="sm", fw=500),
                    dcc.Slider(
                        id="ctrl-fl-s-idx",
                        min=0,
                        max=10,
                        step=1,
                        value=10,
                        marks={0: "Axis", 10: "Edge"},
                        tooltip={"placement": "bottom"},
                    ),
                    dmc.Group(
                        [dmc.Text("Number of Lines:", size="sm"), dmc.NumberInput(id="ctrl-fl-nlines", value=6, min=1, max=20, step=1, w=80)],
                        id="group-fl-nlines",
                        style={"display": "none"},
                        mt="sm",
                    ),
                    dmc.Group(
                        [dmc.Text("Transits:", size="sm"), dmc.NumberInput(id="ctrl-fl-transits", value=50, min=10, max=500, step=10, w=80)],
                        id="group-fl-transits",
                        style={"display": "none"},
                        mt="sm",
                    ),
                    dmc.Group(
                        [dmc.Text("Alpha Start:", size="sm"), dmc.NumberInput(id="ctrl-fl-alpha0", value=0.0, step=0.1, w=80)],
                        id="group-fl-alpha0",
                        style={"display": "none"},
                        mt="sm",
                    ),
                    dmc.Text("Resolution (Grid Points)", size="sm", fw=500, mt="sm"),
                    dmc.Slider(
                        id="ctrl-fl-res",
                        min=64,
                        max=512,
                        step=64,
                        value=128,
                        marks=[
                            {"value": 64, "label": "64"},
                            {"value": 128, "label": "128"},
                            {"value": 256, "label": "256"},
                            {"value": 512, "label": "512"},
                        ],
                        mb="sm",
                    ),
                    dmc.Switch(
                        id="ctrl-fl-zeta-shift",
                        label="Shift zeta by π/NFP",
                        checked=False,
                        color="cyan",
                        mt="sm",
                    ),
                ],
                gap="xs",
            ),
        ],
    )


def render_fieldline_heatmap(vmec, s_idx: int, res: int, theme: PlotTheme, shift_zeta: bool = False):
    zeta_offset = float(np.pi / vmec.nfp) if shift_zeta else 0.0
    alpha_grid, zeta_grid, b_data = vmec.compute_field_line_properties(
        s_idx=s_idx, alpha_points=res, zeta_points=res, single_line=False, zeta_offset=zeta_offset
    )
    fig = go.Figure(
        data=go.Contour(
            z=b_data,
            x=zeta_grid,
            y=alpha_grid,
            colorscale="Viridis",
            ncontours=50,
            contours=dict(coloring="fill", showlines=True, showlabels=False),
            line=dict(color="black", width=0.5),
            colorbar=dict(title="|B| [T]"),
        )
    )
    fig.update_layout(
        title=f"|B| on Flux Surface s={s_idx} (JAX Computed)",
        xaxis_title="Zeta [rad]",
        yaxis_title="Alpha [rad]",
        template=theme.fig_template,
        paper_bgcolor=theme.paper_bg,
        plot_bgcolor=theme.plot_bg,
    )
    return fig


def render_fieldline_lines(vmec, s_idx: int, n_lines: int, res: int, theme: PlotTheme, shift_zeta: bool = False):
    zeta_offset = float(np.pi / vmec.nfp) if shift_zeta else 0.0
    alpha_grid, zeta_grid, b_data = vmec.compute_field_line_properties(
        s_idx=s_idx, alpha_points=res, zeta_points=res, single_line=False, zeta_offset=zeta_offset
    )
    idx_list = np.linspace(0, len(alpha_grid) - 1, n_lines, dtype=int)
    fig = go.Figure()
    for idx in idx_list:
        alpha_val = alpha_grid[idx]
        b_line = b_data[idx, :]
        fig.add_trace(go.Scatter(x=zeta_grid, y=b_line, mode="lines", name=f"alpha={alpha_val:.2f}"))

    fig.update_layout(
        title=f"|B| along field lines (s={s_idx})",
        xaxis_title="Zeta [rad]",
        yaxis_title="|B| [T]",
        template=theme.fig_template,
        paper_bgcolor=theme.paper_bg,
        plot_bgcolor=theme.plot_bg,
    )
    return fig


def render_single_trace(vmec, s_idx: int, transits: int, alpha0: float, res: int, theme: PlotTheme):
    zeta_grid, alpha_segments, b_data = vmec.compute_field_line_properties(
        s_idx=s_idx, zeta_points=res, n_transits=transits, alpha0=alpha0, single_line=True
    )

    z_scatter = np.tile(zeta_grid, (len(alpha_segments), 1)).ravel()
    a_scatter = np.tile(alpha_segments[:, None], (1, len(zeta_grid))).ravel()
    b_flat = b_data.ravel()
    d_zeta = zeta_grid[1] - zeta_grid[0]
    zeta_long = np.arange(b_flat.size) * d_zeta

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(f"Field Line Coverage ({len(alpha_segments)} transits)", "|B| along Field Line"),
        vertical_spacing=0.15,
    )

    fig.add_trace(
        go.Scattergl(
            x=z_scatter,
            y=a_scatter,
            mode="markers",
            marker=dict(
                size=2,
                color=b_flat,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="|B| [T]", len=0.45, y=0.78),
                opacity=0.8,
            ),
            text=[f"|B|={b:.3f}" for b in b_flat],
            hoverinfo="text+x+y",
            name="Coverage",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(x=zeta_long, y=b_flat, mode="lines", line=dict(width=0.5, color="#3bc9db"), name="|B| Trace"),
        row=2,
        col=1,
    )

    fig.update_layout(template=theme.fig_template, paper_bgcolor=theme.paper_bg, plot_bgcolor=theme.plot_bg, height=800, showlegend=False)
    fig.update_xaxes(title_text="Zeta (1 period) [rad]", row=1, col=1)
    fig.update_yaxes(title_text="Alpha [rad]", range=[0, 2 * np.pi], row=1, col=1)
    fig.update_xaxes(title_text="Zeta (Continuous) [rad]", row=2, col=1)
    fig.update_yaxes(title_text="|B| [T]", row=2, col=1)
    return fig
