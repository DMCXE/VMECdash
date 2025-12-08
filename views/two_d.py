from __future__ import annotations

import numpy as np
import dash_mantine_components as dmc
from dash import dcc, html
import plotly.graph_objects as go

from ui.components import get_icon
from views.shared import PlotTheme


def controls():
    return dmc.Paper(
        id="wrapper-2d",
        withBorder=True,
        shadow="xs",
        radius="md",
        p="md",
        style={"display": "none"},
        children=[
            dmc.Text("Configure cross section or flux-surface renderings.", size="sm", c="dimmed", mb="sm"),
            dmc.Select(
                id="ctrl-2d-type",
                label="Plot Type",
                value="cross_section",
                data=[
                    {"value": "cross_section", "label": "Cross Section (R-Z)"},
                    {"value": "flux_surface", "label": "Flux Surface (u-v)"},
                ],
                mb="md",
                allowDeselect=False,
            ),
            dmc.Select(id="ctrl-2d-var", label="Color Variable", value="geometry", data=[], mb="md", allowDeselect=False),
            dmc.Stack(
                [
                    dmc.Text("Toroidal Angle (Phi)", size="sm", fw=500),
                    dmc.Group(
                        [
                            dmc.ActionIcon(get_icon("mdi:minus"), id="btn-phi-dec", variant="light", color="gray", size="lg"),
                            html.Div(
                                dcc.Slider(
                                    id="ctrl-phi",
                                    min=0,
                                    max=1,
                                    step=0.01,
                                    value=0,
                                    marks={0: "0", 0.5: "π", 1: "2π/N"},
                                    tooltip={"placement": "bottom"},
                                    updatemode="drag",
                                ),
                                style={"flexGrow": 1},
                            ),
                            dmc.ActionIcon(get_icon("mdi:plus"), id="btn-phi-inc", variant="light", color="gray", size="lg"),
                        ],
                        gap="xs",
                    ),
                    dmc.Text("Flux Surface Index (s)", size="sm", fw=500, mt="sm"),
                    dmc.Group(
                        [
                            dmc.ActionIcon(get_icon("mdi:minus"), id="btn-s-dec", variant="light", color="gray", size="lg"),
                            html.Div(
                                dcc.Slider(
                                    id="ctrl-s-idx",
                                    min=0,
                                    max=10,
                                    step=1,
                                    value=10,
                                    marks={0: "Axis", 10: "Edge"},
                                    tooltip={"placement": "bottom"},
                                    updatemode="drag",
                                ),
                                style={"flexGrow": 1},
                            ),
                            dmc.ActionIcon(get_icon("mdi:plus"), id="btn-s-inc", variant="light", color="gray", size="lg"),
                        ],
                        gap="xs",
                    ),
                    dmc.Group(
                        [
                            dmc.Text("Visible Surfaces (Count):", size="sm"),
                            dmc.NumberInput(id="ctrl-geo-stride", value=15, min=1, max=200, step=1, w=80),
                        ],
                        id="group-geo-stride",
                        style={"display": "none"},
                        mt="sm",
                    ),
                ],
                gap="xs",
            ),
        ],
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
    fig.update_layout(title=f"Flux Surfaces @ φ={phi_angle:.2f} rad")
    fig.update_xaxes(title="R [m]")
    fig.update_yaxes(title="Z [m]", scaleanchor="x", scaleratio=1)
    fig.update_layout(template=fig_template, paper_bgcolor=paper_bg, plot_bgcolor=plot_bg, uirevision=f"2d-{reset_seed}", showlegend=False)
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


def precompute_frames(vmec, var_name: str):
    field_lookup = {opt["value"]: opt.get("label", opt["value"]) for opt in vmec.available_fields()}
    field_label = field_lookup.get(var_name, var_name)
    frames = []
    steps = np.linspace(0, 1, 21)
    for phi_frac in steps:
        phi = phi_frac * 2 * np.pi / vmec.nfp
        r, z, val = vmec.get_cross_section_grid(phi, var_name, res_grid=140)
        if r is None or z is None or val is None:
            continue
        frames.append({"r": r.tolist(), "z": z.tolist(), "val": np.where(np.isnan(val), None, val).tolist()})
    if not frames:
        return None
    return {"frames": frames, "var_key": var_name, "var_label": field_label}

