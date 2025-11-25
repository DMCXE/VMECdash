from dash import dcc, html
import dash_mantine_components as dmc

from components.ui import create_nav_link, get_icon
from layouts.controls import controls_1d, controls_2d, controls_3d, controls_overview


layout = dmc.MantineProvider(
    id="theme-provider",
    forceColorScheme="dark",
    theme={
        "primaryColor": "cyan",
        "fontFamily": "'Inter', sans-serif",
        "defaultRadius": "md",
        "components": {
            "Card": {"defaultProps": {"withBorder": True}},
            "Paper": {"defaultProps": {"withBorder": True}},
        },
    },
    children=[
        dcc.Store(id='stored-filepath'),
        dcc.Store(id='vmec-meta'),
        dcc.Store(id='current-view', data='overview'),
        dcc.Store(id='store-2d-data'),
        dcc.Download(id='download-report'),

        dmc.AppShell(
            header={"height": 60},
            navbar={"width": 280, "breakpoint": "sm"},
            aside={"width": 320, "breakpoint": "md"},
            padding="md",
            children=[
                dmc.AppShellHeader(
                    px="lg",
                    children=dmc.Group(
                        justify="space-between",
                        align="center",
                        h="100%",
                        children=[
                            dmc.Group([
                                dmc.ThemeIcon(
                                    get_icon("picon:infinity"),
                                    variant="gradient",
                                    gradient={"from": "cyan", "to": "indigo"},
                                    size="lg",
                                    radius="xl",
                                ),
                                dmc.Stack([
                                    dmc.Group([
                                        dmc.Text("VMEC ProViz", fw=700, size="xl", style={"letterSpacing": "-0.5px"}),
                                        dmc.Badge("Neo UI", color="gray", variant="outline"),
                                    ], gap="xs"),
                                    dmc.Text("Interactive VMEC explorer", size="sm", c="dimmed"),
                                ], gap=0),
                            ], gap="md"),
                            dmc.Group([
                                dmc.Text(id="header-filename", children="No File Loaded", size="sm", c="dimmed", fw=500),
                                dmc.Switch(
                                    id="toggle-theme",
                                    label="Dark Mode",
                                    checked=True,
                                    color="cyan",
                                    size="md",
                                ),
                            ], gap="xs"),
                        ],
                    ),
                ),

                dmc.AppShellNavbar(
                    p="md",
                    children=dmc.ScrollArea([
                        dmc.Text("DATA SOURCE", size="xs", fw=700, c="dimmed", mb="sm"),
                        dcc.Upload(
                            id='upload-data',
                            children=dmc.Button(
                                "Load .nc File",
                                leftSection=get_icon("mdi:upload"),
                                variant="outline",
                                fullWidth=True,
                                mb="sm",
                            ),
                            multiple=False,
                        ),
                        dmc.Text("PHYSICS MODULES", size="xs", fw=700, c="dimmed", mt="lg", mb="xs"),
                        create_nav_link("Summary Dashboard", "mdi:view-dashboard-outline", "nav-overview", "Global snapshots"),
                        create_nav_link("1D Profiles", "mdi:chart-bell-curve", "nav-1d", "Flux averaged"),
                        create_nav_link("2D Cross-Section", "mdi:chart-scatter-plot", "nav-2d", "R-Z or u-v"),
                        create_nav_link("3D Geometry", "mdi:shape-outline", "nav-3d", "Boundary surfaces"),
                        dmc.Divider(label="Utilities", labelPosition="center", my="md"),
                        dmc.Button(
                            "High-Res Screenshot",
                            id="btn-download",
                            variant="light",
                            color="gray",
                            fullWidth=True,
                            leftSection=get_icon("mdi:camera"),
                            disabled=True,
                        ),
                    ]),
                ),

                dmc.AppShellAside(
                    p="md",
                    children=[
                        dmc.Group([get_icon("mdi:tune", 20), dmc.Text("View Controls", fw=700)], mb="md"),
                        dmc.Stack([
                            controls_overview,
                            controls_1d,
                            controls_2d,
                            controls_3d,
                            dmc.Alert(
                                "Control visibility responds to navigation selection.",
                                color="gray",
                                variant="light",
                                icon=get_icon("mdi:gesture-tap-button"),
                            ),
                        ], gap="sm"),
                    ],
                ),

                dmc.AppShellMain(
                    children=[
                        dmc.SimpleGrid(
                            cols=1,
                            spacing="md",
                            children=[
                                dmc.Stack([
                                    dmc.Group([
                                        dmc.SegmentedControl(
                                            id="seg-dark-toggle",
                                            value="dark",
                                            data=[
                                                {"label": "Dark", "value": "dark"},
                                                {"label": "Light", "value": "light"},
                                            ],
                                            size="xs",
                                            radius="md",
                                            color="cyan",
                                ),
                                dmc.Badge("Experimental", color="yellow", variant="light"),
                                dmc.Button(
                                    "Reset 3D Camera",
                                    id="btn-reset-camera",
                                    variant="outline",
                                    color="gray",
                                    leftSection=get_icon("mdi:axis")
                                ),
                            ], justify="space-between"),
                            dmc.Alert(
                                id="status-alert",
                                title="Status",
                                children="Ready",
                                color="gray",
                                variant="light",
                                hide=True,
                                icon=get_icon("mdi:information-outline"),
                            ),
                            dmc.Grid([
                                dmc.GridCol(
                                    dmc.Card(
                                        children=dcc.Graph(id='main-graph', figure={}),
                                        radius="md",
                                                withBorder=True,
                                            ),
                                            span=8,
                                        ),
                                        dmc.GridCol(
                                            dmc.Stack(id='stats-container', gap="sm"),
                                            span=4,
                                        ),
                                    ], gutter="md"),
                                ], gap="md"),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)
