from __future__ import annotations

import dash_mantine_components as dmc
from dash import dcc, html

from vmecdash.dash_app.cards import get_icon


def overview_controls():
    return dmc.Paper(
        id="wrapper-overview",
        withBorder=True,
        shadow="xs",
        radius="md",
        p="md",
        style={"display": "block"},
        children=[
            dmc.Group([get_icon("mdi:information-outline", 20), dmc.Text("Summary dashboard highlights six canonical VMEC plots.", size="sm")], gap="xs"),
            dmc.Alert(
                "Displays the standard VMEC equilibrium summary (2x3 panel).",
                title="Overview Mode",
                color="gray",
                variant="light",
                icon=get_icon("mdi:view-dashboard-outline"),
            ),
            dmc.Button(
                "Export Report",
                id="btn-export-report",
                variant="light",
                color="grape",
                fullWidth=True,
                leftSection=get_icon("mdi:file-document"),
                mt="md",
            ),
        ],
    )


def profiles_controls():
    return dmc.Paper(
        id="wrapper-1d",
        withBorder=True,
        shadow="xs",
        radius="md",
        p="md",
        style={"display": "none"},
        children=[
            dmc.Text("Select a flux-surface averaged quantity to inspect versus normalized flux (s).", size="sm", c="dimmed", mb="sm"),
            dmc.ScrollArea(
                h=300,
                type="auto",
                mb="md",
                children=dmc.RadioGroup(id="ctrl-1d-var", label="Profile Variable", value="iotaf", children=[], size="sm"),
            ),
            dmc.Alert(
                "Plots 1D flux surface averaged quantities vs normalized flux (s).",
                color="blue",
                variant="light",
                icon=get_icon("mdi:chart-bell-curve"),
            ),
        ],
    )


def two_d_controls():
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
                                    marks={0: "0", 0.5: "pi", 1: "2pi/N"},
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


def three_d_controls():
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


def fieldline_controls():
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
                        min=1,
                        max=10,
                        step=1,
                        value=10,
                        marks={1: "Near axis", 10: "Edge"},
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
                        label="Shift zeta by pi/NFP",
                        checked=False,
                        color="cyan",
                        mt="sm",
                    ),
                ],
                gap="xs",
            ),
        ],
    )

