from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from vmecdash.theme import PlotTheme


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

