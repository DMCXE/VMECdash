from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

from vmecdash.theme import (
    CONTOUR_LEVELS,
    CONTOUR_LINE_COLOR,
    ColorScale,
    PlotTheme,
    make_empty_figure,
    resolve_colorscale,
)


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


def _sample_field_color(value: float, cs: ColorScale) -> str:
    """The single colour a scalar maps to under ``cs`` - used to flat-fill the axis cell."""
    if not np.isfinite(value) or np.isclose(cs.zmin, cs.zmax):
        level = 0.5
    else:
        level = float(np.clip((value - cs.zmin) / (cs.zmax - cs.zmin), 0.0, 1.0))
    if cs.reversescale:
        level = 1.0 - level
    return sample_colorscale(cs.scale, [level])[0]


def _add_lambda_axis_fill(fig: go.Figure, r_nodes, z_nodes, val_nodes, cs: ColorScale) -> None:
    axis_value = float(np.nanmean(val_nodes[1, :-1]))
    fill_color = _sample_field_color(axis_value, cs)
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
    cs: ColorScale,
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
                colorscale=cs.scale,
                reversescale=cs.reversescale,
                cmin=cs.zmin,
                cmax=cs.zmax,
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


#: The three toroidal cuts of an LPK plot, as fractions of one field period. Under
#: stellarator symmetry phi in [0, half-period] carries all the information, so these
#: three - the endpoints and the midpoint - are the complete set, not an arbitrary
#: sample. VMECplot.m hardcodes the same trio.
LPK_FRACTIONS = (0.0, 0.25, 0.5)

#: Colour follows VMECplot's blue/green/red so the figure stays recognisable, but each
#: curve also gets its own dash pattern: green-vs-red is the worst pair for deuteranopia,
#: curve identity is the entire content of this plot, and a dash pattern still reads once
#: the figure is printed in black and white for a paper.
_LPK_STYLES = (
    ("#1c7ed6", "solid", "0"),
    ("#2f9e44", "dash", "1/4"),
    ("#e03131", "dashdot", "1/2"),
)


#: Per-cut alpha when three filled cross-sections are stacked on one axes. Three layers
#: at this alpha still let the lowest one read through; much higher and the last cut drawn
#: simply wins, much lower and every cut washes out into the page.
LPK_FILL_OPACITY = 0.45


def render_lpk(
    vmec,
    s_idx: int,
    theme: PlotTheme,
    var_name: str = "geometry",
    field_label: str = "",
    colormap: str | None = None,
    res_u: int | None = None,
    contour_lines: bool = False,
):
    """Flux-surface evolution: the same surface at three toroidal angles, on one axes.

    LPK is a viewpoint rather than a plot type, so it follows the current colour variable.
    With ``geometry`` there is nothing to fill and this is the classic VMECplot LPK -
    three bare outlines. With a field selected each cut is filled exactly as the ordinary
    cross-section is, and the three fills are stacked semi-transparently.

    Stacking translucent fills is a genuine compromise: where the cuts overlap the colours
    blend, so a pixel in the overlap is not a faithful reading of any single cut. The
    opaque dashed outlines drawn on top are what keeps the figure interpretable - they say
    which boundary belongs to which angle.
    """
    period = 2.0 * np.pi / max(vmec.nfp, 1)
    s_idx = int(s_idx) if s_idx is not None else vmec.ns - 1
    geometry_only = var_name == "geometry"
    res = int(res_u) if res_u else auto_res_u(vmec.ns, var_name)
    angles = [(frac * period, style) for frac, style in zip(LPK_FRACTIONS, _LPK_STYLES, strict=True)]

    fig = go.Figure()
    outlines = []

    if geometry_only:
        for phi, (color, dash, label) in angles:
            r, z = vmec.get_surface_curve(s_idx, phi, res_u=res)
            outlines.append((r, z, color, dash, label, phi))
    else:
        meshes = []
        for phi, style in angles:
            r_mesh, z_mesh, val_mesh = vmec.get_cross_section_mesh(phi, var_name, res_u=res)
            if r_mesh is None:
                r, z = vmec.get_surface_curve(s_idx, phi, res_u=res)
                outlines.append((r, z, style[0], style[1], style[2], phi))
            else:
                meshes.append((r_mesh, z_mesh, val_mesh, style, phi))

        if meshes:
            # One scale across all three cuts, or the blend in the overlap would be
            # comparing values that were never on the same footing.
            cs = resolve_colorscale(np.concatenate([m[2].ravel() for m in meshes]), colormap)
            for index, (r_mesh, z_mesh, val_mesh, style, phi) in enumerate(meshes):
                _add_carpet_contour(
                    fig, r_mesh, z_mesh, val_mesh, cs, field_label,
                    contour_style="fill", contour_lines=contour_lines,
                    carpet_id=f"lpk-{index}",
                    opacity=LPK_FILL_OPACITY,
                    showscale=index == 0,
                )
                row = min(s_idx, r_mesh.shape[0] - 1)
                outlines.append((r_mesh[row], z_mesh[row], style[0], style[1], style[2], phi))

    # Outlines last so they sit above every fill and stay fully opaque.
    for r, z, color, dash, label, phi in outlines:
        fig.add_trace(
            go.Scatter(
                x=r, y=z, mode="lines",
                line=dict(color=color, width=2, dash=dash),
                name=f"\u03c6 = {label} period",
                hovertemplate="R=%{x:.4f} m<br>Z=%{y:.4f} m<extra></extra>",
            )
        )
        r_axis, z_axis = vmec.get_surface_curve(0, phi, res_u=res)
        fig.add_trace(
            go.Scatter(
                x=[float(r_axis[0])], y=[float(z_axis[0])], mode="markers",
                marker=dict(symbol="cross-thin", size=9, line=dict(color=color, width=1.5)),
                showlegend=False, hoverinfo="skip",
            )
        )

    suffix = "" if geometry_only else f" - {field_label}"
    fig.update_xaxes(title="R [m]")
    fig.update_yaxes(title="Z [m]", scaleanchor="x", scaleratio=1)
    fig.update_layout(
        title=f"Flux Surface Evolution (s index {s_idx}, {vmec.nfp} field periods){suffix}",
        template=theme.fig_template,
        paper_bgcolor=theme.paper_bg,
        plot_bgcolor=theme.plot_bg,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        uirevision=f"2d-lpk-{theme.reset_seed}-{var_name}",
    )
    return fig


def auto_res_u(ns: int, var_name: str) -> int:
    """Poloidal sample count when the user has not chosen one.

    Raising this costs transport, not compute: render time is essentially flat from 160
    to 960 poloidal samples while the payload grows linearly with ``ns * res_u``. So the
    automatic value is scaled down on radially dense equilibria to keep the transfer
    bounded, rather than being a constant.
    """
    base = 480 if var_name == "lambda" else 160
    if ns > 200:
        base = max(120, int(base * 200 / ns))
    return base


def _add_carpet_contour(
    fig: go.Figure,
    r_nodes,
    z_nodes,
    val_nodes,
    cs: ColorScale,
    field_label,
    contour_style: str = "fill",
    contour_lines: bool = True,
    carpet_id: str = "cross-section",
    opacity: float | None = None,
    showscale: bool = True,
):
    n_s, n_theta = val_nodes.shape
    theta_vals = np.linspace(0.0, 2.0 * np.pi, n_theta)
    s_vals = np.linspace(0.0, 1.0, n_s)
    a_flat = np.tile(theta_vals, n_s)
    b_flat = np.repeat(s_vals, n_theta)
    x_flat = r_nodes.reshape(-1)
    y_flat = z_nodes.reshape(-1)
    z_flat = val_nodes.reshape(-1)
    step = (cs.zmax - cs.zmin) / CONTOUR_LEVELS
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
            colorscale=cs.scale,
            reversescale=cs.reversescale,
            # Contourcarpet has no "heatmap" coloring - a curvilinear mesh cannot be
            # smooth-shaded by Plotly, so the caller must not offer that option here.
            contours=dict(
                start=cs.zmin,
                end=cs.zmax,
                size=step,
                coloring="lines" if contour_style == "lines" else "fill",
            ),
            line=dict(width=0.5 if (contour_lines or contour_style == "lines") else 0, color=CONTOUR_LINE_COLOR),
            showscale=showscale,
            colorbar=dict(title=field_label) if showscale else None,
            opacity=opacity,
        )
    )


def render_cross_section_field(
    vmec,
    phi_angle: float,
    var_name: str,
    field_label: str,
    theme: PlotTheme,
    colormap: str | None = None,
    res_u: int | None = None,
    contour_style: str = "fill",
    contour_lines: bool = True,
):
    res_u = int(res_u) if res_u else auto_res_u(vmec.ns, var_name)
    r_nodes, z_nodes, val_nodes = vmec.get_cross_section_mesh(phi_angle, var_name, res_u=res_u)
    if r_nodes is None or z_nodes is None or val_nodes is None:
        return make_empty_figure(theme, f"No cross-section data for {field_label}")

    cs = resolve_colorscale(val_nodes, colormap)
    lcfs_color = "#f8f9fa" if theme.dark_mode else "#212529"
    fig = go.Figure()

    if var_name == "lambda" and val_nodes.shape[0] > 1:
        _add_lambda_axis_fill(fig, r_nodes, z_nodes, val_nodes, cs)
        _add_lambda_sampled_field(fig, r_nodes, z_nodes, val_nodes, cs, field_label)
    else:
        _add_carpet_contour(fig, r_nodes, z_nodes, val_nodes, cs, field_label, contour_style, contour_lines)

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


def render_flux_surface(
    vmec,
    s_idx: int,
    var_name: str,
    field_label: str,
    theme: PlotTheme,
    colormap: str | None = None,
    res_u: int | None = None,
    contour_style: str = "fill",
    contour_lines: bool = True,
):
    res = int(res_u) if res_u else 128
    theta, zeta, val = vmec.get_flux_surface_data(s_idx, var_name, res_u=res, res_v=res)
    fig = go.Figure()
    if theta is not None:
        cs = resolve_colorscale(val, colormap)
        # Unlike the R-Z carpet, this is a rectilinear Contour, so "heatmap" (smooth) is
        # available here.
        coloring = {"smooth": "heatmap", "lines": "lines"}.get(contour_style, "fill")
        fig.add_trace(
            go.Contour(
                x=zeta,
                y=theta,
                z=val,
                ncontours=CONTOUR_LEVELS,
                colorscale=cs.scale,
                reversescale=cs.reversescale,
                zmin=cs.zmin,
                zmax=cs.zmax,
                colorbar=dict(title=field_label),
                contours=dict(coloring=coloring, showlines=contour_lines or coloring == "lines"),
                line=dict(width=0.5, color=CONTOUR_LINE_COLOR),
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

