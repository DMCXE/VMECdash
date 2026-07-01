from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from vmecdash.theme import PlotTheme


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
        # SVG scatter (not Scattergl): the webview caps live WebGL contexts, and 3D + 2D-lambda
        # already use them; this coverage plot's point count is small enough for SVG.
        go.Scatter(
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

    fig.update_layout(template=theme.fig_template, paper_bgcolor=theme.paper_bg, plot_bgcolor=theme.plot_bg, showlegend=False)
    fig.update_xaxes(title_text="Zeta (1 period) [rad]", row=1, col=1)
    fig.update_yaxes(title_text="Alpha [rad]", range=[0, 2 * np.pi], row=1, col=1)
    fig.update_xaxes(title_text="Zeta (Continuous) [rad]", row=2, col=1)
    fig.update_yaxes(title_text="|B| [T]", row=2, col=1)
    return fig

