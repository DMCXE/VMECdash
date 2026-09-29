from __future__ import annotations

from functools import lru_cache

import numpy as np
import plotly.graph_objects as go

from vmecdash.theme import PlotTheme, resolve_colorscale

#: Poloidal points, and toroidal points per field period, when the reader leaves Mesh
#: Resolution on Auto. About DESC's ``plot_3d`` default (101 x 101 per period).
AUTO_POINTS = 128

#: Auto lowers the toroidal density of high-``nfp`` devices so the mesh stays under this
#: many vertices - enough that WebGL never stutters and the payload stays a few MB.
AUTO_MAX_VERTICES = 2**17

#: The floor Auto will not go below, however many field periods there are.
AUTO_MIN_POINTS_PER_PERIOD = 48

#: Neutral shade for the bare geometry: Plotly's "Greys" at its midpoint, as before.
GEOMETRY_COLOR = "rgb(150,150,150)"


def resolve_3d_resolution(resolution, nfp: int) -> tuple[int, int]:
    """``(n_theta, n_zeta_per_period)`` for a Mesh Resolution control value.

    A number is taken literally, for both angles. "auto" (or nothing) keeps the poloidal
    count and trims toroidal density only when ``nfp`` would push the mesh past
    ``AUTO_MAX_VERTICES``.
    """
    if resolution not in (None, "", "auto"):
        points = int(resolution)
        return points, points
    periods = max(int(nfp), 1)
    per_period = min(AUTO_POINTS, AUTO_MAX_VERTICES // (AUTO_POINTS * periods))
    return AUTO_POINTS, max(per_period, AUTO_MIN_POINTS_PER_PERIOD)


@lru_cache(maxsize=8)
def torus_triangles(n_theta: int, n_zeta: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Triangle indices closing an ``n_theta x n_zeta`` periodic grid into a torus.

    Vertex ``(a, b)`` is ``a * n_zeta + b`` - the row-major flattening of the grid. Every
    grid cell is split into two triangles, and both angles wrap, so the last row and
    column connect back to the first: the surface has no seam. Plotly then averages face
    normals across it like anywhere else instead of shading the edge one-sided.
    """
    a, b = np.meshgrid(np.arange(n_theta), np.arange(n_zeta), indexing="ij")
    a_next = (a + 1) % n_theta
    b_next = (b + 1) % n_zeta
    p00 = (a * n_zeta + b).ravel()
    p01 = (a * n_zeta + b_next).ravel()
    p10 = (a_next * n_zeta + b).ravel()
    p11 = (a_next * n_zeta + b_next).ravel()
    ijk = (np.concatenate([p00, p01]), np.concatenate([p01, p11]), np.concatenate([p10, p10]))
    for index in ijk:
        index.flags.writeable = False  # cached and shared between figures
    return ijk


def render_3d(
    vmec,
    s_val: int,
    v_name: str,
    field_label: str,
    coord_free: bool,
    theme: PlotTheme,
    colormap: str | None = None,
    resolution=None,
):
    n_theta, n_zeta_per_period = resolve_3d_resolution(resolution, vmec.nfp)
    x, y, z, val = vmec.compute_3d_surface(
        s_idx=s_val, var_name=v_name, n_theta=n_theta, n_zeta_per_period=n_zeta_per_period
    )
    i, j, k = torus_triangles(*x.shape)

    # A triangle mesh rather than go.Surface, as DESC's plot_3d does. Surface estimates
    # its normals from the open parameter grid, which leaves dark kinked streaks along
    # the sharp bends of a stellarator surface and a one-sided seam at theta = zeta = 0;
    # Mesh3d averages true face normals over a closed triangulation, and interpolates the
    # field value - not the colour - across each triangle.
    mesh = dict(x=x.ravel(), y=y.ravel(), z=z.ravel(), i=i, j=j, k=k, flatshading=False)
    if v_name == "geometry":
        # Flat colour: the surface is the subject, there is no field to read off it.
        surface = go.Mesh3d(**mesh, color=GEOMETRY_COLOR, hoverinfo="x+y+z")
    else:
        # Previously hardcoded "Jet", which is perceptually non-uniform, not colour-blind
        # safe, and puts no neutral point at zero - so signed fields like j^u read as
        # magnitudes. Share the policy every other view uses.
        cs = resolve_colorscale(val, colormap)
        surface = go.Mesh3d(
            **mesh,
            intensity=val.ravel(),
            intensitymode="vertex",
            colorscale=cs.scale,
            reversescale=cs.reversescale,
            cmin=cs.zmin,
            cmax=cs.zmax,
            colorbar=dict(title=field_label, len=0.6),
            name=field_label,
        )

    fig = go.Figure()
    fig.add_trace(surface)

    axis_color = theme.palette.axis
    grid_color = theme.palette.grid
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
