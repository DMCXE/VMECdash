from typing import List, Optional, Sequence, Tuple

import dash_mantine_components as dmc
from dash_iconify import DashIconify
import plotly.graph_objects as go
import numpy as np


def get_icon(icon: str, size: int = 18):
    return DashIconify(icon=icon, width=size)


def create_stat_card(title: str, value: str, icon: str, color: str):
    return dmc.Paper(
        withBorder=True,
        shadow="xs",
        p="md",
        radius="md",
        children=[
            dmc.Group(
                [
                    dmc.Text(title, c="dimmed", size="xs", fw=700, tt="uppercase"),
                    dmc.ThemeIcon(get_icon(icon), color=color, variant="light", size="sm"),
                ],
                justify="space-between",
                mb="xs",
            ),
            dmc.Text(value, fw=700, size="xl"),
        ],
    )


def create_nav_link(label: str, icon: str, id_val: str, description: Optional[str] = None):
    return dmc.NavLink(
        id=id_val,
        label=label,
        leftSection=get_icon(icon),
        variant="light",
        n_clicks=0,
        description=description,
        style={"borderRadius": "6px", "marginBottom": "4px"},
    )


def build_select_data(options: Optional[Sequence[dict]]):
    if not options:
        return []
    payload = []
    for item in options:
        value = item.get("value") if isinstance(item, dict) else None
        if value is None:
            continue
        label = item.get("label") or str(value)
        payload.append({"label": label, "value": value})
    return payload


def unique_options(options: Sequence[dict]):
    seen = set()
    unique = []
    for opt in options:
        value = opt.get("value")
        if value is None or value in seen:
            continue
        seen.add(value)
        unique.append(opt)
    return unique


def create_hero_stat(title: str, value: str, unit: str, icon: str, color: str, sub_text: Optional[str] = None):
    return dmc.Paper(
        withBorder=True,
        radius="md",
        p="md",
        style={"backgroundColor": "var(--mantine-color-body)", "overflow": "hidden"},
        children=dmc.Group(
            [
                dmc.ThemeIcon(
                    get_icon(icon, size=24),
                    size=48,
                    radius="md",
                    color=color,
                    variant="light",
                ),
                dmc.Stack(
                    [
                        dmc.Text(title, c="dimmed", fw=700, size="xs", tt="uppercase"),
                        dmc.Group(
                            [
                                dmc.Text(value, fw=800, size="1.7rem", style={"lineHeight": 1}),
                                dmc.Text(
                                    unit,
                                    c="dimmed",
                                    fw=500,
                                    size="sm",
                                    style={"alignSelf": "flex-end", "marginBottom": "4px"},
                                ),
                            ],
                            gap=4,
                        ),
                        dmc.Text(sub_text, size="xs", c=color) if sub_text else None,
                    ],
                    gap=2,
                ),
            ],
            align="center",
        ),
    )


def create_spec_row(label: str, value: str, unit: Optional[str] = None, icon: Optional[str] = None):
    return dmc.Group(
        [
            dmc.Group([
                get_icon(icon, size=16) if icon else None,
                dmc.Text(label, size="sm", c="dimmed"),
            ], gap="xs"),
            dmc.Box(
                style={
                    "flexGrow": 1,
                    "borderBottom": "1px dashed var(--mantine-color-dimmed)",
                    "opacity": 0.3,
                    "margin": "0 8px",
                }
            ),
            dmc.Group(
                [
                    dmc.Text(value, fw=600, size="sm"),
                    dmc.Text(unit, size="xs", c="dimmed") if unit else None,
                ],
                gap=4,
            ),
        ],
        justify="space-between",
        mb=8,
    )


def create_solver_badge(ftol: Optional[float] = None, iterations: Optional[int] = None):
    if ftol is None and iterations is None:
        return None
    if ftol is None:
        ftol = 1e-8
    if iterations is None:
        iterations = "N/A"
    is_good = ftol < 1e-8
    color = "green" if is_good else "yellow"
    icon = "mdi:check-circle" if is_good else "mdi:alert-circle"
    return dmc.Paper(
        withBorder=True,
        p="xs",
        radius="sm",
        children=dmc.Group(
            [
                dmc.ThemeIcon(get_icon(icon), color=color, variant="filled", size="sm", radius="xl"),
                dmc.Text("Solver Status:", size="xs", fw=700, c=color),
                dmc.Text(f"F_tol = {ftol:.2e}", size="xs", c="dimmed"),
                dmc.Divider(orientation="vertical"),
                dmc.Text(f"{iterations} Iterations", size="xs", c="dimmed"),
            ],
            gap="sm",
        ),
    )


def create_detail_card(title: str, icon: str, items: List[Tuple[str, str, Optional[str]]]):
    rows = []
    for label, value, unit in items:
        rows.append(
            dmc.Group(
                [
                    dmc.Text(label, size="sm", c="dimmed"),
                    dmc.Box(
                        style={
                            "flexGrow": 1,
                            "borderBottom": "1px dashed var(--mantine-color-dimmed)",
                            "opacity": 0.2,
                            "margin": "0 8px",
                        }
                    ),
                    dmc.Group(
                        [
                            dmc.Text(value, fw=600, size="sm"),
                            dmc.Text(unit, size="xs", c="dimmed") if unit else None,
                        ],
                        gap=4,
                    ),
                ],
                justify="space-between",
                mb=6,
            )
        )

    return dmc.Card(
        children=[
            dmc.Group([
                dmc.ThemeIcon(get_icon(icon), size="md", radius="xl", variant="light"),
                dmc.Text(title, fw=700, size="md"),
            ], mb="md"),
            dmc.Stack(rows, gap=2),
        ],
        withBorder=True,
        radius="md",
        p="lg",
    )


def build_geometry_cross_section_figure(vmec, phi_angle: float, s_idx: int, geo_count: Optional[int], dark_mode: bool, fig_template: str, paper_bg: str, plot_bg: str, reset_seed: int):
    count = geo_count if geo_count and geo_count > 0 else 15
    surfaces = np.linspace(0, vmec.ns - 1, int(count), dtype=int)
    if s_idx not in surfaces:
        surfaces = np.append(surfaces, s_idx)
    surfaces = np.sort(surfaces)
    r_grid, z_grid, _ = vmec.get_cross_section_data(
        phi_angle, 'geometry', res_s=vmec.ns, res_u=160
    )
    sel_color = '#1c7ed6' if not dark_mode else '#22b8cf'
    ghost_color = '#5c677d' if not dark_mode else 'rgba(255,255,255,0.25)'
    fig = go.Figure()
    for s_i in surfaces:
        if s_i >= len(r_grid):
            continue
        r_l = np.append(r_grid[s_i], r_grid[s_i][0])
        z_l = np.append(z_grid[s_i], z_grid[s_i][0])
        color = sel_color if s_i == s_idx else ghost_color
        width = 3 if s_i == s_idx else 1
        fig.add_trace(
            go.Scatter(
                x=r_l,
                y=z_l,
                mode='lines',
                line=dict(color=color, width=width),
                hoverinfo='skip'
            )
        )
    fig.update_layout(title=f'Flux Surfaces @ φ={phi_angle:.2f} rad')
    fig.update_xaxes(title='R [m]')
    fig.update_yaxes(title='Z [m]', scaleanchor='x', scaleratio=1)
    fig.update_layout(
        template=fig_template,
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        uirevision=f"2d-{reset_seed}",
        showlegend=False
    )
    return fig
