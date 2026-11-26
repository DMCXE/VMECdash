from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from vmecdash.theme import CONTOUR_LEVELS, CONTOUR_LINE_COLOR, PlotTheme, resolve_colorscale


def _surface_label(vmec, s_idx: int, diag: dict | None = None) -> str:
    if diag is not None and "s_idx" in diag and "s_half" in diag:
        idx = int(diag["s_idx"])
        s_val = float(diag["s_half"])
    else:
        idx = vmec._sanitize_half_s(s_idx) if hasattr(vmec, "_sanitize_half_s") else int(s_idx)
        s_val = vmec.half_mesh_s(idx) if hasattr(vmec, "half_mesh_s") else float(idx) / max(1, vmec.ns - 1)
    return f"s_idx={idx} (s={s_val:.4f})"


def _diagnostic_suffix(diag: dict | None) -> str:
    if not diag:
        return ""
    status = diag.get("status", "monotone")
    n_invalid = int(diag.get("n_invalid", 0))
    n_total = int(diag.get("n_total", 0))
    if status in ("folded", "singular", "nonfinite"):
        min_slope = diag.get("min_dtheta_star_du", float("nan"))
        return f" — [{status}: min(1+λ_u)={min_slope:.2e}]"
    if n_invalid > 0:
        return f" — [{n_invalid}/{n_total} invalid points]"
    return ""


def _near_rational_return(iota: float, nfp: int, n_periods: int) -> tuple[int, float]:
    """Fundamental one-poloidal-turn return period count and signed Δα in degrees."""
    if not np.isfinite(iota) or abs(iota) == 0.0 or nfp <= 0:
        return 1, 0.0
    two_pi = 2.0 * np.pi
    period = two_pi / int(nfp)
    fund_k = max(1, int(round(int(nfp) / abs(iota))))
    delta_rad = float(((fund_k * iota * period + np.pi) % two_pi) - np.pi)
    return fund_k, float(np.degrees(delta_rad))


def render_fieldline_heatmap(
    vmec, s_idx: int, res: int, theme: PlotTheme, shift_zeta: bool = False, colormap: str | None = None,
    contour_style: str = "fill", contour_lines: bool = True,
):
    zeta_offset = float(np.pi / vmec.nfp) if shift_zeta else 0.0
    alpha_grid, zeta_grid, b_data, diag = vmec.compute_field_line_properties(
        s_idx=s_idx,
        alpha_points=res,
        zeta_points=res,
        single_line=False,
        zeta_offset=zeta_offset,
        return_diagnostics=True,
    )
    cs = resolve_colorscale(b_data, colormap)
    coloring = {"smooth": "heatmap", "lines": "lines"}.get(contour_style, "fill")
    fig = go.Figure(
        data=go.Contour(
            z=b_data,
            x=zeta_grid,
            y=alpha_grid,
            colorscale=cs.scale,
            reversescale=cs.reversescale,
            zmin=cs.zmin,
            zmax=cs.zmax,
            ncontours=CONTOUR_LEVELS,
            contours=dict(coloring=coloring, showlines=contour_lines or coloring == "lines", showlabels=False),
            line=dict(color=CONTOUR_LINE_COLOR, width=0.5),
            colorbar=dict(title="|B| [T]"),
        )
    )
    surf_lbl = _surface_label(vmec, s_idx, diag)
    fig.update_layout(
        title=f"|B| on Flux Surface {surf_lbl} (half-mesh){_diagnostic_suffix(diag)}",
        xaxis_title="Zeta [rad]",
        yaxis_title="Alpha [rad]",
        template=theme.fig_template,
        paper_bgcolor=theme.paper_bg,
        plot_bgcolor=theme.plot_bg,
    )
    return fig


def render_fieldline_lines(
    vmec, s_idx: int, n_lines: int, res: int, theme: PlotTheme, shift_zeta: bool = False,
):
    n_lines_i = max(1, int(n_lines))
    zeta_offset = float(np.pi / vmec.nfp) if shift_zeta else 0.0
    # Request exactly n_lines equidistant alpha_j = 2*pi*j/n_lines directly,
    # avoiding both wasted res x res work and the endpoint near-overlap artifact.
    alpha_grid, zeta_grid, b_data, diag = vmec.compute_field_line_properties(
        s_idx=s_idx,
        alpha_points=n_lines_i,
        zeta_points=res,
        single_line=False,
        zeta_offset=zeta_offset,
        return_diagnostics=True,
    )
    fig = go.Figure()
    for idx, alpha_val in enumerate(alpha_grid):
        b_line = b_data[idx, :]
        fig.add_trace(
            go.Scatter(
                x=zeta_grid,
                y=b_line,
                mode="lines",
                connectgaps=False,
                name=f"alpha={alpha_val:.2f} rad ({np.degrees(alpha_val):.1f}°)",
            )
        )

    surf_lbl = _surface_label(vmec, s_idx, diag)
    fig.update_layout(
        title=f"|B| along {n_lines_i} Field Lines ({surf_lbl}, half-mesh){_diagnostic_suffix(diag)}",
        xaxis_title="Zeta [rad]",
        yaxis_title="|B| [T]",
        template=theme.fig_template,
        paper_bgcolor=theme.paper_bg,
        plot_bgcolor=theme.plot_bg,
    )
    return fig


def render_single_trace(
    vmec, s_idx: int, transits: int, alpha0: float, res: int, theme: PlotTheme, colormap: str | None = None,
):
    zeta_grid, alpha_segments, b_data, diag = vmec.compute_field_line_properties(
        s_idx=s_idx,
        zeta_points=res,
        n_transits=transits,
        alpha0=alpha0,
        single_line=True,
        return_diagnostics=True,
    )

    n_periods = len(alpha_segments)
    nfp = max(1, int(diag.get("nfp", vmec.nfp)))
    iota = float(diag.get("iota", 0.0))
    toroidal_turns = n_periods / float(nfp)
    ret_k, ret_deg = _near_rational_return(iota, nfp, n_periods)

    invalid_mask = np.asarray(diag["invalid_mask"], dtype=bool)
    b_masked = np.where(invalid_mask, np.nan, b_data)

    z_scatter = np.tile(zeta_grid, (n_periods, 1)).ravel()
    a_scatter = np.tile(alpha_segments[:, None], (1, len(zeta_grid))).ravel()
    invalid_flat = invalid_mask.ravel()
    b_flat = b_masked.ravel()
    # Break markers cleanly at invalid points while preserving continuous zeta axis for the trace.
    z_scatter_plot = np.where(invalid_flat, np.nan, z_scatter)
    a_scatter_plot = np.where(invalid_flat, np.nan, a_scatter)

    d_zeta = zeta_grid[1] - zeta_grid[0] if len(zeta_grid) > 1 else 0.0
    zeta_long = np.arange(b_flat.size) * d_zeta

    surf_lbl = _surface_label(vmec, s_idx, diag)
    coverage_title = (
        f"Segment-Start α Folded to 1 Field Period "
        f"({n_periods} periods = {toroidal_turns:.2f} toroidal turns; "
        f"near-rational return: {ret_k} periods, Δα = {ret_deg:+.2f}°)"
    )
    trace_title = f"|B| along Continuous Field Line ({surf_lbl}, half-mesh){_diagnostic_suffix(diag)}"

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(coverage_title, trace_title),
        vertical_spacing=0.15,
    )

    fig.add_trace(
        # SVG scatter (not Scattergl): the webview caps live WebGL contexts, and 3D + 2D-lambda
        # already use them; this coverage plot's point count is small enough for SVG.
        go.Scatter(
            x=z_scatter_plot,
            y=a_scatter_plot,
            mode="markers",
            marker=dict(
                size=2,
                color=b_flat,
                colorscale=resolve_colorscale(b_flat, colormap).scale,
                showscale=True,
                colorbar=dict(title="|B| [T]", len=0.45, y=0.78),
                opacity=0.8,
            ),
            text=[f"|B|={b:.3f}" if np.isfinite(b) else "|B|=invalid" for b in b_flat],
            hoverinfo="text+x+y",
            name="Coverage",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=zeta_long,
            y=b_flat,
            mode="lines",
            connectgaps=False,
            line=dict(width=0.5, color=theme.palette.line),
            name="|B| Trace",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title=f"Single Field Line Trace — {surf_lbl} (half-mesh){_diagnostic_suffix(diag)}",
        template=theme.fig_template,
        paper_bgcolor=theme.paper_bg,
        plot_bgcolor=theme.plot_bg,
        showlegend=False,
    )
    fig.update_xaxes(title_text="Zeta (1 field period) [rad]", row=1, col=1)
    fig.update_yaxes(title_text="Segment-Start α (folded) [rad]", range=[0, 2 * np.pi], row=1, col=1)
    fig.update_xaxes(title_text="Zeta (Continuous) [rad]", row=2, col=1)
    fig.update_yaxes(title_text="|B| [T]", row=2, col=1)
    return fig
