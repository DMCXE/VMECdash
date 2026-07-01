from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

from vmecdash.theme import PlotTheme, make_empty_figure


def build_geometry_cross_section_figure(vmec, phi_angle, s_idx, geo_count, dark_mode, fig_template, paper_bg, plot_bg, reset_seed):
    count = geo_count if geo_count and geo_count > 0 else 15
    surfaces = np.linspace(0, vmec.ns - 1, int(count), dtype=int)
    if s_idx not in surfaces:
        surfaces = np.append(surfaces, s_idx)
    surfaces = np.sort(surfaces)
    r_grid, z_grid, _ = vmec.get_cross_section_data(phi_angle, "geometry", res_s=vmec.ns, res_u=160)
    sel_color = "#1c7ed6" if not dark_mode else "#22b8cf"
    ghost_color = "#5c677d" if not dark_mode else "rgba(255,255,255,0.25)"
    fig = go.Figure()
    for s_i in surfaces:
        if s_i >= len(r_grid):
            continue
        r_l = np.append(r_grid[s_i], r_grid[s_i][0])
        z_l = np.append(z_grid[s_i], z_grid[s_i][0])
        color = sel_color if s_i == s_idx else ghost_color
        width = 3 if s_i == s_idx else 1
        fig.add_trace(go.Scatter(x=r_l, y=z_l, mode="lines", line=dict(color=color, width=width), hoverinfo="skip"))
    fig.update_layout(title=f"Flux Surfaces @ phi={phi_angle:.2f} rad")
    fig.update_xaxes(title="R [m]")
    fig.update_yaxes(title="Z [m]", scaleanchor="x", scaleratio=1)
    fig.update_layout(template=fig_template, paper_bgcolor=paper_bg, plot_bgcolor=plot_bg, uirevision=f"2d-{reset_seed}", showlegend=False)
    return fig


def _carpet_colorscale(values: np.ndarray):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return "Viridis", False, 0.0, 1.0
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if vmin < 0.0 < vmax:
        bound = max(abs(vmin), abs(vmax))
        return "RdBu", True, -bound, bound
    if np.isclose(vmin, vmax):
        pad = max(abs(vmin) * 0.01, 1e-9)
        vmin -= pad
        vmax += pad
    return "Viridis", False, vmin, vmax


def _sample_field_color(value: float, colorscale, reversescale: bool, zmin: float, zmax: float) -> str:
    if not np.isfinite(value) or np.isclose(zmin, zmax):
        level = 0.5
    else:
        level = float(np.clip((value - zmin) / (zmax - zmin), 0.0, 1.0))
    if reversescale:
        level = 1.0 - level
    return sample_colorscale(colorscale, [level])[0]


def _add_lambda_axis_fill(fig: go.Figure, r_nodes, z_nodes, val_nodes, colorscale, reversescale, zmin, zmax) -> None:
    axis_value = float(np.nanmean(val_nodes[1, :-1]))
    fill_color = _sample_field_color(axis_value, colorscale, reversescale, zmin, zmax)
    fig.add_trace(
        go.Scatter(
            x=r_nodes[1],
            y=z_nodes[1],
            mode="none",
            fill="toself",
            fillcolor=fill_color,
            line=dict(color=fill_color, width=0),
            hoverinfo="skip",
            showlegend=False,
            name="Axis fill",
        )
    )


def _add_lambda_sampled_field(
    fig: go.Figure,
    r_nodes: np.ndarray,
    z_nodes: np.ndarray,
    val_nodes: np.ndarray,
    colorscale,
    reversescale: bool,
    zmin: float,
    zmax: float,
    field_label: str,
) -> None:
    if r_nodes.shape[0] < 3 or r_nodes.shape[1] < 2:
        return

    r_centers = 0.25 * (r_nodes[1:-1, :-1] + r_nodes[2:, :-1] + r_nodes[2:, 1:] + r_nodes[1:-1, 1:])
    z_centers = 0.25 * (z_nodes[1:-1, :-1] + z_nodes[2:, :-1] + z_nodes[2:, 1:] + z_nodes[1:-1, 1:])
    cell_values = 0.25 * (
        val_nodes[1:-1, :-1] + val_nodes[2:, :-1] + val_nodes[2:, 1:] + val_nodes[1:-1, 1:]
    )
    finite_mask = np.isfinite(r_centers) & np.isfinite(z_centers) & np.isfinite(cell_values)
    if not np.any(finite_mask):
        return

    fig.add_trace(
        go.Scattergl(
            x=r_centers[finite_mask],
            y=z_centers[finite_mask],
            mode="markers",
            marker=dict(
                symbol="square",
                size=6,
                color=cell_values[finite_mask],
                colorscale=colorscale,
                reversescale=reversescale,
                cmin=zmin,
                cmax=zmax,
                showscale=True,
                colorbar=dict(title=field_label),
                line=dict(width=0),
            ),
            hoverinfo="skip",
            showlegend=False,
            name="Lambda samples",
        )
    )


def _add_lcfs_outline(fig: go.Figure, r_nodes, z_nodes, lcfs_color: str) -> None:
    fig.add_trace(
        go.Scatter(
            x=r_nodes[-1],
            y=z_nodes[-1],
            mode="lines",
            line=dict(color=lcfs_color, width=1.2),
            hoverinfo="skip",
            showlegend=False,
        )
    )


def _add_carpet_contour(fig: go.Figure, r_nodes, z_nodes, val_nodes, colorscale, reversescale, zmin, zmax, field_label):
    n_s, n_theta = val_nodes.shape
    theta_vals = np.linspace(0.0, 2.0 * np.pi, n_theta)
    s_vals = np.linspace(0.0, 1.0, n_s)
    a_flat = np.tile(theta_vals, n_s)
    b_flat = np.repeat(s_vals, n_theta)
    x_flat = r_nodes.reshape(-1)
    y_flat = z_nodes.reshape(-1)
    z_flat = val_nodes.reshape(-1)
    step = (zmax - zmin) / 40.0
    carpet_id = "cross-section"
    hidden_axis = dict(showgrid=False, showticklabels="none", showline=False, startline=False, endline=False, smoothing=0)

    fig.add_trace(
        go.Carpet(carpet=carpet_id, a=a_flat, b=b_flat, x=x_flat, y=y_flat, aaxis=hidden_axis, baxis=hidden_axis)
    )
    fig.add_trace(
        go.Contourcarpet(
            carpet=carpet_id,
            a=a_flat,
            b=b_flat,
            z=z_flat,
            colorscale=colorscale,
            reversescale=reversescale,
            contours=dict(start=zmin, end=zmax, size=step, coloring="fill"),
            line=dict(width=0),
            colorbar=dict(title=field_label),
        )
    )


def render_cross_section_field(vmec, phi_angle: float, var_name: str, field_label: str, theme: PlotTheme):
    res_u = 480 if var_name == "lambda" else 160
    r_nodes, z_nodes, val_nodes = vmec.get_cross_section_mesh(phi_angle, var_name, res_u=res_u)
    if r_nodes is None or z_nodes is None or val_nodes is None:
        return make_empty_figure(theme, f"No cross-section data for {field_label}")

    colorscale, reversescale, zmin, zmax = _carpet_colorscale(val_nodes)
    lcfs_color = "#f8f9fa" if theme.dark_mode else "#212529"
    fig = go.Figure()

    if var_name == "lambda" and val_nodes.shape[0] > 1:
        _add_lambda_axis_fill(fig, r_nodes, z_nodes, val_nodes, colorscale, reversescale, zmin, zmax)
        _add_lambda_sampled_field(fig, r_nodes, z_nodes, val_nodes, colorscale, reversescale, zmin, zmax, field_label)
    else:
        _add_carpet_contour(fig, r_nodes, z_nodes, val_nodes, colorscale, reversescale, zmin, zmax, field_label)

    _add_lcfs_outline(fig, r_nodes, z_nodes, lcfs_color)

    fig.update_xaxes(title="R [m]")
    fig.update_yaxes(title="Z [m]", scaleanchor="x", scaleratio=1)
    fig.update_layout(
        title=f"{field_label} on Cross-Section at phi={phi_angle:.2f} rad",
        template=theme.fig_template,
        paper_bgcolor=theme.paper_bg,
        plot_bgcolor=theme.plot_bg,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
        uirevision=f"2d-cross-{theme.reset_seed}-{var_name}",
    )
    return fig


def render_flux_surface(vmec, s_idx: int, var_name: str, field_label: str, theme: PlotTheme):
    theta, zeta, val = vmec.get_flux_surface_data(s_idx, var_name, res_u=128, res_v=128)
    fig = go.Figure()
    if theta is not None:
        fig.add_trace(
            go.Contour(
                x=zeta,
                y=theta,
                z=val,
                ncontours=50,
                colorscale="Viridis",
                colorbar=dict(title=field_label),
                contours=dict(coloring="fill"),
            )
        )
        fig.update_layout(
            title=f"{field_label} on Flux Surface s={s_idx/(vmec.ns-1):.2f}",
            xaxis_title="Zeta (toroidal) [rad]",
            yaxis_title="Theta (poloidal) [rad]",
            xaxis=dict(range=[0, 2 * np.pi / vmec.nfp]),
            yaxis=dict(range=[0, 2 * np.pi]),
        )
    fig.update_layout(
        template=theme.fig_template,
        paper_bgcolor=theme.paper_bg,
        plot_bgcolor=theme.plot_bg,
        uirevision=f"2d-{theme.reset_seed}",
    )
    return fig

