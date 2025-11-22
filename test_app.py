import dash
from dash import dcc, html, Input, Output, State, ctx, _dash_renderer
import dash_mantine_components as dmc
from dash_iconify import DashIconify
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
import tempfile
import os
import numpy as np
import math

try:
    from vmec_jax import VMECJaxProcessor
except ImportError:
    print("Warning: vmec_jax module not found. Ensure the file is in the directory.")
    VMECJaxProcessor = None

# 设置 React 版本 
_dash_renderer._set_react_version("18.2.0")

app = dash.Dash(
    __name__, 
    title="VMEC ProViz",
    suppress_callback_exceptions=True,
    external_stylesheets=dmc.styles.ALL
)
server = app.server

# -----------------------
# 1. UI Helpers
# -----------------------
def get_icon(icon, size=18):
    return DashIconify(icon=icon, width=size)

def create_stat_card(title, value, icon, color):
    return dmc.Paper(
        withBorder=True, shadow="xs", p="md", radius="md",
        children=[
            dmc.Group([
                dmc.Text(title, c="dimmed", size="xs", fw=700, tt="uppercase"),
                dmc.ThemeIcon(get_icon(icon), color=color, variant="light", size="sm")
            ], justify="space-between", mb="xs"),
            dmc.Text(value, fw=700, size="xl"),
        ]
    )

def create_nav_link(label, icon, id_val, description=None):
    return dmc.NavLink(
        id=id_val,
        label=label,
        leftSection=get_icon(icon),
        variant="light",
        n_clicks=0,
        description=description,
        style={"borderRadius": "6px", "marginBottom": "4px"}
    )


def build_select_data(options):
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


def unique_options(options):
    seen = set()
    unique = []
    for opt in options:
        value = opt.get("value")
        if value is None or value in seen:
            continue
        seen.add(value)
        unique.append(opt)
    return unique


def create_hero_stat(title, value, unit, icon, color, sub_text=None):
    return dmc.Paper(
        withBorder=True,
        radius="md",
        p="md",
        style={"backgroundColor": "var(--mantine-color-body)", "overflow": "hidden"},
        children=dmc.Group([
            dmc.ThemeIcon(
                get_icon(icon, size=24),
                size=48,
                radius="md",
                color=color,
                variant="light"
            ),
            dmc.Stack([
                dmc.Text(title, c="dimmed", fw=700, size="xs", tt="uppercase"),
                dmc.Group([
                    dmc.Text(value, fw=800, size="1.7rem", style={"lineHeight": 1}),
                    dmc.Text(unit, c="dimmed", fw=500, size="sm", style={"alignSelf": "flex-end", "marginBottom": "4px"})
                ], gap=4),
                dmc.Text(sub_text, size="xs", c=color) if sub_text else None
            ], gap=2)
        ], align="center")
    )


def create_spec_row(label, value, unit=None, icon=None):
    return dmc.Group([
        dmc.Group([
            get_icon(icon, size=16) if icon else None,
            dmc.Text(label, size="sm", c="dimmed"),
        ], gap="xs"),
        dmc.Box(style={"flexGrow": 1, "borderBottom": "1px dashed var(--mantine-color-dimmed)", "opacity": 0.3, "margin": "0 8px"}),
        dmc.Group([
            dmc.Text(value, fw=600, size="sm"),
            dmc.Text(unit, size="xs", c="dimmed") if unit else None
        ], gap=4)
    ], justify="space-between", mb=8)


def create_solver_badge(ftol=None, iterations=None):
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
        children=dmc.Group([
            dmc.ThemeIcon(get_icon(icon), color=color, variant="filled", size="sm", radius="xl"),
            dmc.Text("Solver Status:", size="xs", fw=700, c=color),
            dmc.Text(f"F_tol = {ftol:.2e}", size="xs", c="dimmed"),
            dmc.Divider(orientation="vertical"),
            dmc.Text(f"{iterations} Iterations", size="xs", c="dimmed")
        ], gap="sm")
    )

def create_detail_card(title, icon, items):
    rows = []
    for label, value, unit in items:
        rows.append(
            dmc.Group([
                dmc.Text(label, size="sm", c="dimmed"),
                dmc.Box(style={"flexGrow": 1, "borderBottom": "1px dashed var(--mantine-color-dimmed)", "opacity": 0.2, "margin": "0 8px"}),
                dmc.Group([
                    dmc.Text(value, fw=600, size="sm"),
                    dmc.Text(unit, size="xs", c="dimmed") if unit else None
                ], gap=4)
            ], justify="space-between", mb=6)
        )
    
    return dmc.Card(
        children=[
            dmc.Group([
                dmc.ThemeIcon(get_icon(icon), size="md", radius="xl", variant="light"),
                dmc.Text(title, fw=700, size="md")
            ], mb="md"),
            dmc.Stack(rows, gap=2)
        ],
        withBorder=True,
        radius="md",
        p="lg"
    )


def build_geometry_cross_section_figure(vmec, phi_angle, s_idx, geo_count, dark_mode, fig_template, paper_bg, plot_bg, reset_seed):
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

# -----------------------
# 2. Control Panels (Hidden Strategy)
# -----------------------

# A. Overview Controls
controls_overview = dmc.Paper(
    id="wrapper-overview",
    withBorder=True,
    shadow="xs",
    radius="md",
    p="md",
    style={"display": "block"},
    children=[
        dmc.Group([
            get_icon("mdi:information-outline", 20),
            dmc.Text("Summary dashboard highlights six canonical VMEC plots.", size="sm"),
        ], gap="xs"),
        dmc.Alert(
            "Displays the standard VMEC equilibrium summary (2x3 panel).",
            title="Overview Mode",
            color="gray",
            variant="light",
            icon=get_icon("mdi:view-dashboard-outline")
        ),
        dmc.Button(
            "Export Report",
            id="btn-export-report",
            variant="light",
            color="grape",
            fullWidth=True,
            leftSection=get_icon("mdi:file-document"),
            mt="md"
        )
    ]
)

# B. 1D Controls
controls_1d = dmc.Paper(
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
            children=dmc.RadioGroup(
                id="ctrl-1d-var",
                label="Profile Variable",
                value="iotaf",
                children=[],
                size="sm"
            )
        ),
        dmc.Alert(
            "Plots 1D flux surface averaged quantities vs normalized flux (s).",
            color="blue",
            variant="light",
            icon=get_icon("mdi:chart-bell-curve"),
        )
    ]
)

# C. 2D Controls
controls_2d = dmc.Paper(
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
            allowDeselect=False
        ),
        dmc.Select(
            id="ctrl-2d-var",
            label="Color Variable",
            value="geometry",
            data=[],
            mb="md",
            allowDeselect=False
        ),
        dmc.Stack([
            dmc.Text("Toroidal Angle (Phi)", size="sm", fw=500),
            dmc.Group([
                dmc.ActionIcon(get_icon("mdi:minus"), id="btn-phi-dec", variant="light", color="gray", size="lg"),
                html.Div(
                    dcc.Slider(
                        id="ctrl-phi",
                        min=0,
                        max=1,
                        step=0.01,
                        value=0,
                        marks={0: '0', 0.5: 'π', 1: '2π/N'},
                        tooltip={"placement": "bottom"},
                        updatemode='drag'
                    ),
                    style={"flexGrow": 1}
                ),
                dmc.ActionIcon(get_icon("mdi:plus"), id="btn-phi-inc", variant="light", color="gray", size="lg"),
            ], gap="xs"),

            dmc.Text("Flux Surface Index (s)", size="sm", fw=500, mt="sm"),
            dmc.Group([
                dmc.ActionIcon(get_icon("mdi:minus"), id="btn-s-dec", variant="light", color="gray", size="lg"),
                html.Div(
                    dcc.Slider(
                        id="ctrl-s-idx",
                        min=0,
                        max=10,
                        step=1,
                        value=10,
                        marks={0: 'Axis', 10: 'Edge'},
                        tooltip={"placement": "bottom"},
                        updatemode='drag'
                    ),
                    style={"flexGrow": 1}
                ),
                dmc.ActionIcon(get_icon("mdi:plus"), id="btn-s-inc", variant="light", color="gray", size="lg"),
            ], gap="xs"),

            # Geometry Stride Control
            dmc.Group([
                dmc.Text("Visible Surfaces (Count):", size="sm"),
                dmc.NumberInput(
                    id="ctrl-geo-stride",
                    value=15,
                    min=1,
                    max=200,
                    step=1,
                    w=80
                )
            ], id="group-geo-stride", style={"display": "none"}, mt="sm")
        ], gap="xs")
    ]
)

# D. 3D Controls
controls_3d = dmc.Paper(
    id="wrapper-3d",
    withBorder=True,
    shadow="xs",
    radius="md",
    p="md",
    style={"display": "none"},
    children=[
        dmc.Text("Rendering 3D surfaces using the accelerated JAX backend.", size="sm", c="dimmed", mb="sm"),
        dmc.Select(
            id="ctrl-3d-var",
            label="Surface Color",
            value="modB",
            data=[],
            mb="md",
            allowDeselect=False
        ),
        dmc.Stack([
            dmc.Text("Flux Surface (s)", size="sm", fw=500),
            dcc.Slider(
                id="ctrl-s3d-idx",
                min=0,
                max=10,
                step=1,
                value=10,
                marks={0: 'Axis', 10: 'Edge'},
                tooltip={"placement": "bottom"}
            ),
            dmc.Switch(
                id="ctrl-3d-bg",
                label="Coordinate-free Background",
                checked=True,
                color="cyan",
                mt="sm"
            )
        ], gap="xs")
    ]
)

# -----------------------
# 3. Main Layout
# -----------------------
app.layout = dmc.MantineProvider(
    id="theme-provider",
    forceColorScheme="dark",
    theme={
        "primaryColor": "cyan",
        "fontFamily": "'Inter', sans-serif",
        "defaultRadius": "md",
        "components": {
            "Card": {"defaultProps": {"withBorder": True}},
            "Paper": {"defaultProps": {"withBorder": True}}
        }
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
                # --- Header ---
                dmc.AppShellHeader(
                    px="lg",
                    children=dmc.Group(
                        justify="space-between",
                        align="center",
                        h="100%",
                        children=[
                            dmc.Group([
                                dmc.ThemeIcon(
                                    # get_icon("mdi:atom-variant"),
                                    get_icon("picon:infinity"),
                                    variant="gradient",
                                    gradient={"from": "cyan", "to": "indigo"},
                                    size="lg",
                                    radius="xl"
                                ),
                                dmc.Stack([
                                    dmc.Group([
                                        dmc.Text("VMEC ProViz", fw=700, size="xl", style={"letterSpacing": "-0.5px"}),
                                        dmc.Badge("Neo UI", color="gray", variant="outline")
                                    ], gap="xs"),
                                    dmc.Text("Interactive VMEC explorer", size="sm", c="dimmed")
                                ], gap=0)
                            ], gap="md"),
                            dmc.Group([
                                dmc.Text(id="header-filename", children="No File Loaded", size="sm", c="dimmed", fw=500),
                                dmc.Switch(
                                    id="toggle-theme",
                                    label="Dark Mode",
                                    checked=True,
                                    color="cyan",
                                    size="md"
                                ),
                                
                                # dmc.ActionIcon(get_icon("mdi:github"), variant="subtle", color="gray"),
                                # dmc.ActionIcon(get_icon("mdi:bell"), variant="subtle", color="gray"),
                                # dmc.Avatar(radius="xl", color="cyan", children="VM")
                            ], gap="xs")
                        ]
                    )
                ),

                # --- Navbar (Navigation Tree) ---
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
                                mb="sm"
                            ),
                            multiple=False
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
                            disabled=True
                        )
                    ])
                ),

                # --- Aside (Context Controls) ---
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
                                id="status-alert",
                                title="Status",
                                children="Ready",
                                color="gray",
                                variant="light",
                                hide=True,
                                icon=get_icon("mdi:information-outline")
                            ),
                            dmc.Button(
                                "Reset Camera",
                                id="btn-reset-camera",
                                variant="subtle",
                                color="gray",
                                leftSection=get_icon("mdi:axis-arrow"),
                                disabled=True
                            )
                        ], gap="md")
                    ]
                ),

                # --- Main Area ---
                dmc.AppShellMain(
                    children=dmc.Stack([
                        html.Div(id="stats-container"),
                        dmc.Card(
                            h="calc(100vh - 200px)",
                            p=0,
                            style={"overflow": "hidden", "position": "relative"},
                            children=[
                                html.Div(
                                    dmc.Badge("Interactive View", color="dark", radius="sm"),
                                    style={"position": "absolute", "top": 12, "left": 12, "zIndex": 10}
                                ),
                                dcc.Loading(
                                    custom_spinner=dmc.Loader(color="cyan", size="xl", type="bars"),
                                    children=dcc.Graph(
                                        id="main-graph",
                                        style={"height": "calc(100vh - 204px)", "width": "100%"},
                                        config={"displayModeBar": True, "displaylogo": False, "scrollZoom": True}
                                    )
                                )
                            ]
                        )
                    ], gap="md")
                )
            ]
        )
    ]
)

# -----------------------
# 4. Callbacks
# -----------------------

# 4.1 File Upload
@app.callback(
    [Output('stored-filepath', 'data'),
     Output('header-filename', 'children'),
     Output('vmec-meta', 'data')],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    prevent_initial_call=True
)
def handle_upload(contents, filename):
    if not contents: return dash.no_update
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        fd, path = tempfile.mkstemp(suffix=".nc")
        with os.fdopen(fd, 'wb') as f: f.write(decoded)
        
        vmec = VMECJaxProcessor.from_file(path)
        meta = {
            "ns": vmec.ns,
            "nfp": vmec.nfp,
            "profiles": vmec.available_profiles(),
            "fields": vmec.available_fields(),
            "summary_lines": vmec.get_summary_lines(),
        }
        return path, f"Active: {filename}", meta
    except Exception as e:
        return dash.no_update, f"Error: {str(e)}", dash.no_update


@app.callback(
    Output('download-report', 'data'),
    Input('btn-export-report', 'n_clicks'),
    State('stored-filepath', 'data'),
    prevent_initial_call=True
)
def export_report(n_clicks, filepath):
    if not n_clicks or not filepath:
        return dash.no_update
    try:
        vmec = VMECJaxProcessor.from_file(filepath)
        scalars = vmec.get_scalars()
        summary_lines = vmec.get_summary_lines()
        report_lines = ["VMEC Report", "==============", ""]
        for key, value in scalars.items():
            report_lines.append(f"{key}: {value}")
        report_lines.append("")
        report_lines.append("Summary")
        report_lines.extend(summary_lines)
        content = "\n".join(report_lines)
        return dict(content=content, filename="vmec_report.txt")
    except Exception as exc:
        print(f"Export error: {exc}")
        return dash.no_update

# 4.2 Populate Control Options
@app.callback(
    [Output('ctrl-1d-var', 'children'),
     Output('ctrl-2d-var', 'data'),
     Output('ctrl-3d-var', 'data'),
     Output('ctrl-s-idx', 'max'), Output('ctrl-s-idx', 'marks'), Output('ctrl-s-idx', 'value'),
     Output('ctrl-s3d-idx', 'max'), Output('ctrl-s3d-idx', 'marks'), Output('ctrl-s3d-idx', 'value')],
    Input('vmec-meta', 'data'),
    prevent_initial_call=True
)
def update_controls(meta):
    if not meta: return dash.no_update
    
    profiles_data = unique_options(build_select_data(meta.get('profiles', [])))
    profile_radios = dmc.Stack(
        [dmc.Radio(label=p['label'], value=p['value'], size="sm") for p in profiles_data],
        gap="xs"
    )

    fields = unique_options(build_select_data(meta.get('fields', [])))
    
    # 增加 Geometry 选项到 2D 颜色中（如果不在 fields 里）
    has_geometry = any(opt.get('value') == 'geometry' for opt in fields)
    fields_source = fields if has_geometry else [{"label": "Geometry Only (None)", "value": "geometry"}] + fields
    fields_2d = unique_options(fields_source)
    
    ns = meta.get('ns', 2)
    max_s = max(0, ns - 1)
    marks = {0: 'Axis', max_s: 'Edge'}
    
    return (
        profile_radios, 
        fields_2d, 
        fields, 
        max_s, marks, max_s, 
        max_s, marks, max_s
    )


@app.callback(
    Output('store-2d-data', 'data'),
    [Input('current-view', 'data'),
     Input('ctrl-2d-type', 'value'),
     Input('ctrl-2d-var', 'value'),
     Input('stored-filepath', 'data')],
    prevent_initial_call=True
)
def precompute_2d_slices(view, type_2d, var_2d, filepath):
    if view != '2d' or type_2d != 'cross_section' or not filepath or var_2d == 'geometry':
        return None
    try:
        vmec = VMECJaxProcessor.from_file(filepath)
        field_lookup = {opt["value"]: opt.get("label", opt["value"]) for opt in vmec.available_fields()}
        field_label = field_lookup.get(var_2d, var_2d)
        frames = []
        steps = np.linspace(0, 1, 21)
        for phi_frac in steps:
            phi = phi_frac * 2 * np.pi / vmec.nfp
            r, z, val = vmec.get_cross_section_grid(phi, var_2d, res_grid=140)
            if r is None or z is None or val is None:
                continue
            frames.append({
                "r": r.tolist(),
                "z": z.tolist(),
                "val": np.where(np.isnan(val), None, val).tolist()
            })
        if not frames:
            return None
        return {"frames": frames, "var_key": var_2d, "var_label": field_label}
    except Exception as exc:
        print(f"Precompute error: {exc}")
        return dash.no_update


@app.callback(
    [Output('ctrl-phi', 'disabled'),
     Output('group-geo-stride', 'style')],
    Input('ctrl-2d-type', 'value'),
    Input('ctrl-2d-var', 'value'),
    Input('store-2d-data', 'data')
)
def toggle_phi_slider(type_2d, var_name, data_store):
    is_cross = (type_2d == 'cross_section')
    is_geo = (var_name == 'geometry')
    stride_style = {"display": "flex"} if (is_cross and is_geo) else {"display": "none"}
    disabled = not is_cross
    return disabled, stride_style


@app.callback(
    Output('ctrl-phi', 'updatemode'),
    Input('ctrl-2d-type', 'value'),
    Input('ctrl-2d-var', 'value')
)
def adjust_phi_updatemode(type_2d, var_name):
    if type_2d == 'cross_section' and var_name == 'geometry':
        return 'drag'
    return 'mouseup'


@app.callback(
    Output('btn-reset-camera', 'disabled'),
    Input('current-view', 'data')
)
def toggle_reset_button(view):
    return view != '3d'


# 4.3 Navigation Logic (Visibility Toggles)
@app.callback(
    [Output('current-view', 'data'),
     Output('nav-overview', 'active'), Output('nav-1d', 'active'),
     Output('nav-2d', 'active'), Output('nav-3d', 'active'),
     Output('wrapper-overview', 'style'), Output('wrapper-1d', 'style'),
     Output('wrapper-2d', 'style'), Output('wrapper-3d', 'style')],
    [Input('nav-overview', 'n_clicks'),
     Input('nav-1d', 'n_clicks'),
     Input('nav-2d', 'n_clicks'),
     Input('nav-3d', 'n_clicks')],
    prevent_initial_call=True
)
def update_view(n1, n2, n3, n4):
    ctx_id = ctx.triggered_id or "nav-overview"
    view_map = {
        "nav-overview": "overview",
        "nav-1d": "1d",
        "nav-2d": "2d",
        "nav-3d": "3d"
    }
    view = view_map.get(ctx_id, "overview")
    
    # Active states
    is_ov = (view == "overview")
    is_1d = (view == "1d")
    is_2d = (view == "2d")
    is_3d = (view == "3d")
    
    # Styles
    show = {"display": "block"}
    hide = {"display": "none"}
    
    return (
        view, 
        is_ov, is_1d, is_2d, is_3d,
        show if is_ov else hide,
        show if is_1d else hide,
        show if is_2d else hide,
        show if is_3d else hide
    )

# 4.4 Main Plotting Logic (Restored from Original Code + Fixes)
@app.callback(
    [Output('main-graph', 'figure', allow_duplicate=True),
     Output('stats-container', 'children')],
    [Input('current-view', 'data'),
     Input('stored-filepath', 'data'),
     Input('ctrl-1d-var', 'value'),
     Input('ctrl-2d-type', 'value'),
     Input('ctrl-2d-var', 'value'),
     Input('ctrl-s-idx', 'value'),
     Input('ctrl-3d-var', 'value'),
     Input('ctrl-s3d-idx', 'value'),
     Input('ctrl-3d-bg', 'checked'),
     Input('btn-reset-camera', 'n_clicks'),
     Input('toggle-theme', 'checked'),
     Input('ctrl-geo-stride', 'value')],
    State('vmec-meta', 'data'),
    State('ctrl-phi', 'value'),
    prevent_initial_call=True
)
def update_visualization(view, filepath, var_1d, type_2d, var_2d, s_2d, var_3d, s_3d, bg_3d, reset_clicks, dark_mode, geo_count, meta, phi_state):
    # Skip heavy server updates when the clientside cross-section renderer owns the figure
    triggered = ctx.triggered_id
    if (
        filepath
        and view == '2d'
        and type_2d == 'cross_section'
        and var_2d != 'geometry'
        and triggered == 'ctrl-phi'
    ):
        return dash.no_update, dash.no_update

    dark_mode = True if dark_mode is None else bool(dark_mode)
    fig_template = 'plotly_dark' if dark_mode else 'plotly_white'
    paper_bg = 'rgba(0,0,0,0)' if dark_mode else 'white'
    plot_bg = 'rgba(0,0,0,0)'
    text_color = '#adb5bd' if dark_mode else '#495057'
    empty_fig = go.Figure()
    empty_fig.update_layout(
        template=fig_template,
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        xaxis={'visible': False},
        yaxis={'visible': False}
    )
    
    if not filepath:
        empty_fig.add_annotation(text="Please load a .nc file", showarrow=False, font=dict(size=20, color=text_color))
        return empty_fig, []

    try:
        vmec = VMECJaxProcessor.from_file(filepath)
        scalars = vmec.get_scalars()
        field_map = {opt['value']: opt.get('label', opt['value']) for opt in vmec.available_fields()}
        summary_lines = None
        if meta:
            summary_lines = meta.get('summary_lines')
        if not summary_lines and view == 'overview':
            summary_lines = vmec.get_summary_lines()
        reset_seed = reset_clicks or 0
        
        # --- FIX: Safe Access for Scalars ---
        beta = scalars.get('beta_total', 0.0)
        vol = scalars.get('volume', 0.0)
        curr = scalars.get('ctor', 0.0)
        
        # 安全计算 Aspect Ratio
        rmaj = scalars.get('Rmajor')
        amin = scalars.get('Aminor')
        if rmaj is not None and amin is not None and amin != 0:
            ar_str = f"{rmaj/amin:.2f}"
        else:
            ar_str = "N/A"
        
        def safe_fmt(value, pattern="{:.2f}", fallback="--"):
            try:
                if value is None or (isinstance(value, (float, int)) and math.isnan(value)):
                    return fallback
                return pattern.format(value)
            except Exception:
                return fallback

        base_cards = dmc.SimpleGrid(cols=4, children=[
            create_stat_card('Beta Total', f"{beta*100:.2f}%", 'mdi:percent', 'red'),
            create_stat_card('Volume', f"{vol:.2f} m³", 'mdi:cube-outline', 'blue'),
            create_stat_card('Aspect Ratio', ar_str, 'mdi:ratio', 'orange'),
            create_stat_card('Toroidal Current', f"{curr:.2f} A", 'mdi:current-ac', 'teal'),
        ])

        def render_summary_panels(lines):
            if not lines:
                return dmc.Center(dmc.Text('Summary unavailable', c='dimmed', size='sm'))
            panels = []
            for line in lines:
                fragments = [frag.strip() for frag in line.replace('|', ',').split(',') if frag.strip()]
                rows = []
                for frag in fragments:
                    if '=' in frag:
                        key, value = frag.split('=', 1)
                        rows.append(
                            dmc.Group([
                                dmc.Kbd(key.strip(), style={"minWidth": 70}),
                                dmc.Text(value.strip(), fw=600)
                            ], gap="xs", align="center")
                        )
                    else:
                        rows.append(dmc.Text(frag, fw=600))
                panels.append(
                    dmc.Paper(
                        dmc.Stack(rows, gap=4),
                        withBorder=True,
                        radius="sm",
                        shadow="xs",
                        p="sm"
                    )
                )
            return dmc.SimpleGrid(cols=2, spacing="sm", children=panels)

        # --------------------------
        # MODE: OVERVIEW (Restored)
        # --------------------------
        stats_ui = base_cards
        if view == 'overview':
            # 1. Hero Stats
            hero_cards = dmc.SimpleGrid(
                cols=3,
                spacing='md',
                children=[
                    create_hero_stat('Total Beta', f"{beta*100:.2f}", '%', 'mdi:percent', 'cyan', 'Global beta'),
                    create_hero_stat('Toroidal Current', f"{curr:.2f}", 'MA', 'mdi:current-ac', 'indigo', 'Plasma current'),
                    create_hero_stat('Volume', f"{vol:.2f}", 'm³', 'mdi:cube-outline', 'teal', 'Enclosed volume'),
                ]
            )

            # 2. Detailed Groups
            def fmt(val, f="{:.3f}"):
                return safe_fmt(val, f)

            # Geometry Group
            geo_items = [
                ("Major Radius (R0)", fmt(rmaj), "m"),
                ("Minor Radius (a)", fmt(amin), "m"),
                ("Aspect Ratio", fmt(scalars.get('aspect'), "{:.2f}"), ""),
                ("Volume", fmt(vol), "m³"),
            ]
            
            # Magnetics Group
            mag_items = [
                ("Magnetic Field (B0)", fmt(scalars.get('b0')), "T"),
                ("Toroidal Flux (RB_tor)", fmt(scalars.get('rbtor')), "T·m"),
                ("Iota (Axis)", fmt(scalars.get('iota_axis')), ""),
                ("Iota (Edge)", fmt(scalars.get('iota_edge')), ""),
                ("Safety Factor (q0)", fmt(scalars.get('q_axis')), ""),
                ("Safety Factor (qa)", fmt(scalars.get('q_edge')), ""),
                ("Shear (Edge)", fmt(scalars.get('shear_edge')), ""),
            ]

            # Plasma & Boundary Group
            mgrid = vmec.ds.attrs.get('mgrid_file', 'none')
            if isinstance(mgrid, bytes):
                mgrid = mgrid.decode('utf-8')
            if mgrid == 'none':
                lfreeb = vmec.ds.attrs.get('lfreeb__logical__', 0)
                if lfreeb:
                    bound_type = "Free Boundary (No mgrid)"
                else:
                    bound_type = "Fixed Boundary"
            else:
                bound_type = f"Free Boundary ({mgrid})"

            plasma_items = [
                ("Beta Poloidal", fmt(scalars.get('betapol')), ""),
                ("Beta Toroidal", fmt(scalars.get('betator')), ""),
                ("Pressure (Axis)", fmt(scalars.get('pressure_axis'), "{:.2e}"), "Pa"),
                ("Pressure (Edge)", fmt(scalars.get('pressure_edge'), "{:.2e}"), "Pa"),
                ("Boundary Type", bound_type, ""),
            ]

            details_grid = dmc.SimpleGrid(
                cols=3,
                spacing="md",
                children=[
                    create_detail_card("Geometry", "mdi:axis-arrow", geo_items),
                    create_detail_card("Magnetics", "mdi:magnet", mag_items),
                    create_detail_card("Plasma & Boundary", "mdi:fire", plasma_items),
                ]
            )

            stats_ui = dmc.Stack([hero_cards, details_grid], gap='md')

            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=(
                    'Rotational Transform (iota)', 'Safety Factor (q)', 'Pressure Profile',
                    'dP/ds', 'Volume Enclosed (Vp)', 'Flux Avg <B·B>'
                ),
                vertical_spacing=0.15, horizontal_spacing=0.08
            )

            def get_data_safe(key):
                try:
                    return vmec.get_1d_data(key)
                except Exception:
                    return [], []

            s_i, iota = get_data_safe('iotaf')
            s_q, q_prof = get_data_safe('q')
            s_p, pres = get_data_safe('presf')
            s_dp, dpds = get_data_safe('dpds')
            s_vp, vp = get_data_safe('vp')
            s_bb, bdotb = get_data_safe('bdotb')

            fig.add_trace(go.Scatter(x=s_i, y=iota, name='iota', line=dict(color='#22b8cf', width=3)), row=1, col=1)
            fig.add_trace(go.Scatter(x=s_q, y=q_prof, name='q', line=dict(color='#fd7e14', width=3)), row=1, col=2)
            fig.add_trace(go.Scatter(x=s_p, y=pres, name='pres', line=dict(color='#fa5252', width=3)), row=1, col=3)
            fig.add_trace(go.Scatter(x=s_dp, y=dpds, name='dP/ds', line=dict(color='#12b886', width=3)), row=2, col=1)
            fig.add_trace(go.Scatter(x=s_vp, y=vp, name='Vp', line=dict(color='#7950f2', width=3)), row=2, col=2)
            fig.add_trace(go.Scatter(x=s_bb, y=bdotb, name='<B.B>', line=dict(color='#0ca678', width=3)), row=2, col=3)

            fig.update_layout(
                title_text='Equilibrium Summary',
                template=fig_template,
                paper_bgcolor=paper_bg,
                plot_bgcolor=plot_bg,
                height=800,
                showlegend=False,
                uirevision=f"overview-{reset_seed}"
            )
            return fig, stats_ui

        # --------------------------
        # MODE: 1D PROFILES
        # --------------------------
        if view == '1d':
            var = var_1d or 'iotaf'
            s, y = vmec.get_1d_data(var)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=s, y=y, mode='lines', line=dict(color='#3bc9db', width=4)))
            fig.update_layout(
                title=f'Profile: {var}',
                xaxis_title='Normalized Flux (s)', 
                yaxis_title=var,
                template=fig_template,
                paper_bgcolor=paper_bg,
                plot_bgcolor=plot_bg,
                uirevision=f"1d-{reset_seed}"
            )
            return fig, stats_ui

        # --------------------------
        # MODE: 2D CROSS SECTION
        # --------------------------
        if view == '2d':
            phi_val = phi_state if phi_state is not None else 0.0
            phi_scale = 2 * np.pi / max(vmec.nfp, 1)
            phi_angle = phi_val * phi_scale
            s_idx = int(s_2d) if s_2d is not None else vmec.ns - 1
            field_label = field_map.get(var_2d, var_2d)

            if type_2d == 'cross_section':
                if var_2d == 'geometry':
                    fig = build_geometry_cross_section_figure(
                        vmec, phi_angle, s_idx, geo_count, dark_mode,
                        fig_template, paper_bg, plot_bg, reset_seed
                    )
                    return fig, stats_ui

                return dash.no_update, stats_ui

            fig = go.Figure()

            if type_2d == 'flux_surface':
                theta, zeta, val = vmec.get_flux_surface_data(s_idx, var_2d, res_u=128, res_v=128)
                if theta is not None:
                    fig.add_trace(
                        go.Contour(
                            x=zeta,
                            y=theta,
                            z=val,
                            colorscale='Viridis',
                            colorbar=dict(title=field_label),
                            contours=dict(coloring='fill')
                        )
                    )
                    fig.update_layout(
                        title=f"{field_label} on Flux Surface s={s_idx/(vmec.ns-1):.2f}",
                        xaxis_title='Zeta (toroidal) [rad]',
                        yaxis_title='Theta (poloidal) [rad]',
                        xaxis=dict(range=[0, 2*np.pi/vmec.nfp]),
                        yaxis=dict(range=[0, 2*np.pi])
                    )
                    fig.update_layout(
                        template=fig_template,
                        paper_bgcolor=paper_bg,
                        plot_bgcolor=plot_bg,
                        uirevision=f"2d-{reset_seed}"
                    )
                    return fig, stats_ui

            return dash.no_update, stats_ui

        # --------------------------
        # MODE: 3D GEOMETRY
        # --------------------------
        if view == '3d':
            s_val = int(s_3d) if s_3d is not None else vmec.ns - 1
            v_name = var_3d or 'modB'
            coord_free = True if bg_3d is None else bool(bg_3d)
            field_label = field_map.get(v_name, v_name)
            fig = go.Figure()
            
            x, y, z, val = vmec.compute_3d_surface(s_idx=s_val, var_name=v_name, resolution=100)
            
            # 根据变量是否为几何设置颜色
            if v_name == 'geometry':
                surf_color = np.full_like(z, 0.5)
                cscale = 'Greys'
            else:
                surf_color = val
                # cscale = 'Plasma'
                cscale = 'Jet'
            
            fig.add_trace(go.Surface(
                x=x, y=y, z=z, surfacecolor=surf_color,
                colorscale=cscale,
                colorbar=dict(title=field_label, len=0.6) if v_name != 'geometry' else None
            ))
            
            axis_color = '#dee2e6' if dark_mode else '#495057'
            grid_color = '#5c677d' if dark_mode else '#ced4da'
            axis_style = (
                dict(visible=False)
                if coord_free
                else dict(
                    visible=True,
                    backgroundcolor='rgba(0,0,0,0)',
                    gridcolor=grid_color,
                    zerolinecolor=grid_color,
                    color=axis_color,
                    title=dict(font=dict(color=axis_color)),
                    tickfont=dict(color=axis_color)
                )
            )
            title_txt = f"Surface s={s_val/(vmec.ns-1):.2f}"
            if v_name != 'geometry':
                title_txt += f" colored by {field_label}"
            fig.update_layout(
                title=title_txt,
                template=fig_template,
                paper_bgcolor=paper_bg,
                scene=dict(
                    bgcolor=plot_bg,
                    xaxis=axis_style,
                    yaxis=axis_style,
                    zaxis=axis_style,
                    aspectmode='data'
                ),
                margin=dict(l=0, r=0, t=30, b=0)
            )
            return fig, stats_ui

    except Exception as e:
        print(f"Detailed Error: {e}")
        import traceback
        traceback.print_exc()
        err_fig = go.Figure()
        err_fig.add_annotation(text=f"Error: {str(e)}", showarrow=False, font=dict(color="red", size=16))
        return err_fig, []
        
    return empty_fig, []


@app.callback(
    Output('theme-provider', 'forceColorScheme'),
    Input('toggle-theme', 'checked')
)
def toggle_theme(checked):
    return 'dark' if checked else 'light'


@app.callback(
    Output('btn-download', 'disabled'),
    Input('stored-filepath', 'data')
)
def toggle_download_button(filepath):
    return not bool(filepath)



app.clientside_callback(
    """
    function(phi_val, data_store, view, type_2d, var_2d) {
        if (phi_val === undefined || phi_val === null) {
            return window.dash_clientside.no_update;
        }
        if (view !== '2d' || type_2d !== 'cross_section' || var_2d === 'geometry') {
            return window.dash_clientside.no_update;
        }
        if (!data_store || !data_store.frames || !data_store.frames.length) {
            return window.dash_clientside.no_update;
        }
        if (!data_store.var_key || data_store.var_key !== var_2d) {
            return window.dash_clientside.no_update;
        }
        try {
            var total = data_store.frames.length;
            var idx = Math.round(phi_val * (total - 1));
            if (idx < 0) idx = 0;
            if (idx >= total) idx = total - 1;
            var frame = data_store.frames[idx];
            if (!frame || !frame.r || !frame.z || !frame.val) {
                return window.dash_clientside.no_update;
            }
            var displayLabel = data_store.var_label || data_store.var_key || 'Field';
            var fig_data = {
                type: 'contour',
                x: frame.r,
                y: frame.z,
                z: frame.val,
                colorscale: 'RdBu',
                colorbar: {title: displayLabel},
                contours: {coloring: 'heatmap'},
                ncontours: 50,
                line: {width: 0}
            };
            var layout = {
                title: displayLabel + ' on Cross-Section at φ=' + phi_val.toFixed(2),
                xaxis: {title: 'R [m]'},
                yaxis: {title: 'Z [m]', scaleanchor: 'x', scaleratio: 1},
                template: 'plotly_dark'
            };
            return {data: [fig_data], layout: layout};
        } catch (e) {
            console.error('Clientside callback error:', e);
            return window.dash_clientside.no_update;
        }
    }
    """,
    Output('main-graph', 'figure', allow_duplicate=True),
    Input('ctrl-phi', 'value'),
    Input('store-2d-data', 'data'),
    State('current-view', 'data'),
    State('ctrl-2d-type', 'value'),
    State('ctrl-2d-var', 'value'),
    prevent_initial_call=True
)


app.clientside_callback(
    """
    function(var_name, view, type_2d) {
        if (view === '2d' && type_2d === 'cross_section' && var_name !== 'geometry') {
            return [
                false,
                'Calculating ' + var_name + '...',
                'blue',
                {'height': 'calc(100vh - 204px)', 'opacity': 0.3, 'transition': 'opacity 0.5s'}
            ];
        }
        return [
            true,
            'Ready',
            'gray',
            {'height': 'calc(100vh - 204px)', 'opacity': 1}
        ];
    }
    """,
    Output('status-alert', 'hide'),
    Output('status-alert', 'children'),
    Output('status-alert', 'color'),
    Output('main-graph', 'style'),
    Input('ctrl-2d-var', 'value'),
    Input('current-view', 'data'),
    Input('ctrl-2d-type', 'value'),
    prevent_initial_call=True
)


app.clientside_callback(
    """
    function(data, var_name) {
        if (data && data.var_key === var_name) {
            return [
                false,
                'Rendered: ' + (data.var_label || var_name),
                'green',
                {'height': 'calc(100vh - 204px)', 'opacity': 1, 'transition': 'opacity 0.5s'}
            ];
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output('status-alert', 'hide', allow_duplicate=True),
    Output('status-alert', 'children', allow_duplicate=True),
    Output('status-alert', 'color', allow_duplicate=True),
    Output('main-graph', 'style', allow_duplicate=True),
    Input('store-2d-data', 'data'),
    State('ctrl-2d-var', 'value'),
    prevent_initial_call=True
)


app.clientside_callback(
    """
    function(n_clicks) {
        if (!n_clicks) {
            return window.dash_clientside.no_update;
        }
        var graphContainer = document.getElementById('main-graph');
        if (!graphContainer) {
            return window.dash_clientside.no_update;
        }
        var plot = graphContainer.querySelector('.js-plotly-plot');
        if (!plot) {
            return window.dash_clientside.no_update;
        }
        Plotly.downloadImage(plot, {
            format: 'png',
            width: 1400,
            height: 900,
            filename: 'vmec_viz'
        });
        return window.dash_clientside.no_update;
    }
    """,
    Output('btn-download', 'id'),
    Input('btn-download', 'n_clicks'),
    prevent_initial_call=True
)


@app.callback(
    Output('main-graph', 'figure', allow_duplicate=True),
    Input('ctrl-phi', 'value'),
    State('current-view', 'data'),
    State('ctrl-2d-type', 'value'),
    State('ctrl-2d-var', 'value'),
    State('ctrl-s-idx', 'value'),
    State('ctrl-geo-stride', 'value'),
    State('stored-filepath', 'data'),
    State('toggle-theme', 'checked'),
    prevent_initial_call=True
)
def update_geometry_phi(phi_val, view, type_2d, var_2d, s_idx, geo_count, filepath, dark_mode):
    if not (filepath and view == '2d' and type_2d == 'cross_section' and var_2d == 'geometry'):
        return dash.no_update
    phi_val = phi_val or 0.0
    try:
        vmec = VMECJaxProcessor.from_file(filepath)
        phi_scale = 2 * np.pi / max(vmec.nfp, 1)
        phi_angle = phi_val * phi_scale
        s_idx = int(s_idx) if s_idx is not None else vmec.ns - 1
        dark_mode = True if dark_mode is None else bool(dark_mode)
        fig_template = 'plotly_dark' if dark_mode else 'plotly_white'
        paper_bg = 'rgba(0,0,0,0)' if dark_mode else 'white'
        plot_bg = 'rgba(0,0,0,0)'
        fig = build_geometry_cross_section_figure(
            vmec, phi_angle, s_idx, geo_count, dark_mode,
            fig_template, paper_bg, plot_bg, reset_seed=0
        )
        return fig
    except Exception as exc:
        print(f"Geometry phi update error: {exc}")
        return dash.no_update

# 额外的按钮回调
@app.callback(
    Output('ctrl-phi', 'value', allow_duplicate=True),
    Input('btn-phi-dec', 'n_clicks'),
    Input('btn-phi-inc', 'n_clicks'),
    State('ctrl-phi', 'value'),
    prevent_initial_call=True
)
def update_phi_btn(n_dec, n_inc, val):
    ctx_id = ctx.triggered_id
    val = val or 0
    step = 0.01
    if ctx_id == 'btn-phi-dec':
        return max(0, val - step)
    elif ctx_id == 'btn-phi-inc':
        return min(1, val + step)
    return dash.no_update


@app.callback(
    Output('ctrl-s-idx', 'value', allow_duplicate=True),
    Input('btn-s-dec', 'n_clicks'),
    Input('btn-s-inc', 'n_clicks'),
    State('ctrl-s-idx', 'value'),
    State('ctrl-s-idx', 'max'),
    prevent_initial_call=True
)
def update_s_btn(n_dec, n_inc, val, max_val):
    ctx_id = ctx.triggered_id
    val = val or 0
    if ctx_id == 'btn-s-dec':
        return max(0, val - 1)
    elif ctx_id == 'btn-s-inc':
        return min(max_val, val + 1)
    return dash.no_update

if __name__ == '__main__':
    app.run(debug=True)
