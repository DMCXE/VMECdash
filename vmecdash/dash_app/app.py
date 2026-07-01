import base64
import os
import tempfile
from pathlib import Path

import dash
import dash_mantine_components as dmc
import numpy as np
from dash import Input, Output, State, ctx, dcc, html, _dash_renderer

from vmecdash import stats as vmec_stats
from vmecdash.core import VMECJaxProcessor
from vmecdash.dash_app.cards import (
    base_stat_cards,
    build_select_data,
    create_nav_link,
    get_icon,
    overview_stats_cards,
    unique_options,
)
from vmecdash.dash_app.controls import fieldline_controls, overview_controls, profiles_controls, three_d_controls, two_d_controls
from vmecdash.renderers import fieldline, profiles, three_d, two_d
from vmecdash.renderers.overview import overview_figure
from vmecdash.theme import build_theme, make_empty_figure

# 设置 React 版本 
_dash_renderer._set_react_version("18.2.0")

app = dash.Dash(
    __name__, 
    title="VMEC Dashboard",
    suppress_callback_exceptions=True,
    external_stylesheets=dmc.styles.ALL,
    assets_folder=str(Path(__file__).resolve().parent / "assets")
)
app._favicon = "icon_stell_circle.png"
server = app.server

# -----------------------
# 1. UI Helpers
# -----------------------
# Shared UI helpers now live in ui/components.py. Feature-specific helpers sit
# in the views/ modules to make it easy to add new plot types.

# -----------------------
# 2. Control Panels (Hidden Strategy)
# -----------------------
controls_overview = overview_controls()
controls_1d = profiles_controls()
controls_2d = two_d_controls()
controls_3d = three_d_controls()
controls_fieldline = fieldline_controls()

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
        dcc.Store(id='resize-ping'),
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
                                    variant="gradient",
                                    # gradient={"from": "cyan", "to": "indigo"},
                                    # gradient={"from": "slateblue", "to": "thistle"},
                                    gradient={"from": "skyblue", "to": "thistle"},
                                    size="lg",
                                    radius="xl",
                                    children=dmc.Avatar(
                                        src="/assets/icon_stell_noback.png",
                                        size=32,
                                        radius="xl",
                                        alt="VMEC Dashboard"
                                    )
                                ),
                                dmc.Stack([
                                    dmc.Group([
                                        dmc.Text("VMEC Dashboard", fw=700, size="xl", style={"letterSpacing": "-0.5px"}),
                                        dmc.Badge("USTCstellarator", color="gray", variant="outline")
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
                        create_nav_link("Field Lines", "mdi:vector-curve", "nav-fieldline", "Trace & |B|"),
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
                            controls_fieldline,
                            dmc.Alert(
                                id="status-alert",
                                title="Status",
                                children="Ready",
                                color="gray",
                                variant="light",
                                hide=True,
                                icon=get_icon("mdi:information-outline")
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
                                    type="default",
                                    color="cyan",
                                    children=dcc.Graph(
                                        id="main-graph",
                                        figure=make_empty_figure(build_theme(True, 0), "Please load a .nc file"),
                                        style={"height": "100%", "width": "100%"},
                                        config={"displayModeBar": True, "displaylogo": False, "scrollZoom": True, "responsive": True}
                                    )
                                ),
                                # Drag-and-drop overlay for initial load
                                dcc.Upload(
                                    id="upload-data-center",
                                    multiple=False,
                                    children=html.Div(
                                        id="upload-overlay",
                                        children=dmc.Center(
                                            dmc.Paper(
                                                withBorder=True,
                                                shadow="sm",
                                                radius="md",
                                                p="xl",
                                                style={"backgroundColor": "#2e2e2e"},
                                                children=dmc.Stack(
                                                    [
                                                        dmc.ThemeIcon(get_icon("mdi:upload"), size="xl", radius="xl", color="cyan", variant="light"),
                                                        dmc.Text("Drop or click to load a .nc file", fw=700, size="lg"),
                                                        dmc.Text("VMEC wout NetCDF files are supported", c="dimmed", size="sm"),
                                                    ],
                                                    align="center",
                                                    gap="xs",
                                                ),
                                            ),
                                        ),
                                        style={
                                            "position": "absolute",
                                            "top": 0,
                                            "left": 0,
                                            "right": 0,
                                            "bottom": 0,
                                            "zIndex": 30,
                                            "display": "flex",
                                            "alignItems": "center",
                                            "justifyContent": "center",
                                            "backgroundColor": "#2e2e2e",
                                        },
                                    ),
                                    style={
                                        "position": "absolute",
                                        "top": 0,
                                        "left": 0,
                                        "right": 0,
                                        "bottom": 0,
                                        "zIndex": 30,
                                        "width": "100%",
                                        "height": "100%",
                                    },
                                ),
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
    [Input('upload-data', 'contents'),
     Input('upload-data-center', 'contents')],
    [State('upload-data', 'filename'),
     State('upload-data-center', 'filename')],
    prevent_initial_call=True
)
def handle_upload(contents_nav, contents_center, fname_nav, fname_center):
    trigger = ctx.triggered_id
    if trigger == 'upload-data-center':
        contents, filename = contents_center, fname_center
    else:
        contents, filename = contents_nav, fname_nav
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
            "computed_profiles": vmec.available_computed_profiles(),
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
     Output('ctrl-s3d-idx', 'max'), Output('ctrl-s3d-idx', 'marks'), Output('ctrl-s3d-idx', 'value'),
     Output('ctrl-fl-s-idx', 'max'), Output('ctrl-fl-s-idx', 'marks'), Output('ctrl-fl-s-idx', 'value')],
    Input('vmec-meta', 'data'),
    prevent_initial_call=True
)
def update_controls(meta):
    if not meta: return dash.no_update
    
    profiles_data = unique_options(build_select_data(meta.get('profiles', [])))
    computed_profiles = unique_options(build_select_data(meta.get('computed_profiles', [])))
    all_profile_options = unique_options(profiles_data + computed_profiles)
    profile_radios = dmc.Stack(
        [dmc.Radio(label=p['label'], value=p['value'], size="sm") for p in all_profile_options],
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
    fl_min_s = 1 if max_s >= 1 else 0
    fl_marks = {fl_min_s: 'Near axis' if fl_min_s else 'Axis', max_s: 'Edge'}
    
    return (
        profile_radios, 
        fields_2d, 
        fields, 
        max_s, marks, max_s, 
        max_s, marks, max_s,
        max_s, fl_marks, max_s
    )


@app.callback(
    [Output('ctrl-phi', 'disabled'),
     Output('group-geo-stride', 'style')],
    Input('ctrl-2d-type', 'value'),
    Input('ctrl-2d-var', 'value')
)
def toggle_phi_slider(type_2d, var_name):
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


# 4.3 Navigation Logic (Visibility Toggles)
@app.callback(
    [Output('current-view', 'data'),
     Output('nav-overview', 'active'), Output('nav-1d', 'active'),
     Output('nav-2d', 'active'), Output('nav-3d', 'active'), Output('nav-fieldline', 'active'),
     Output('wrapper-overview', 'style'), Output('wrapper-1d', 'style'),
     Output('wrapper-2d', 'style'), Output('wrapper-3d', 'style'), Output('wrapper-fieldline', 'style')],
    [Input('nav-overview', 'n_clicks'),
     Input('nav-1d', 'n_clicks'),
     Input('nav-2d', 'n_clicks'),
     Input('nav-3d', 'n_clicks'),
     Input('nav-fieldline', 'n_clicks')],
    prevent_initial_call=True
)
def update_view(n1, n2, n3, n4, n5):
    ctx_id = ctx.triggered_id or "nav-overview"
    view_map = {
        "nav-overview": "overview",
        "nav-1d": "1d",
        "nav-2d": "2d",
        "nav-3d": "3d",
        "nav-fieldline": "fieldline"
    }
    view = view_map.get(ctx_id, "overview")
    
    # Active states
    is_ov = (view == "overview")
    is_1d = (view == "1d")
    is_2d = (view == "2d")
    is_3d = (view == "3d")
    is_fl = (view == "fieldline")
    
    # Styles
    show = {"display": "block"}
    hide = {"display": "none"}
    
    return (
        view, 
        is_ov, is_1d, is_2d, is_3d, is_fl,
        show if is_ov else hide,
        show if is_1d else hide,
        show if is_2d else hide,
        show if is_3d else hide,
        show if is_fl else hide
    )

# Hide upload overlay / disable click-to-upload once a file is loaded
@app.callback(
    [Output("upload-overlay", "style"), Output("upload-data-center", "style")],
    Input("stored-filepath", "data"),
)
def toggle_overlay(filepath):
    overlay_style = {
        "position": "absolute",
        "top": 0,
        "left": 0,
        "right": 0,
        "bottom": 0,
        "zIndex": 30,
        "display": "flex",
        "alignItems": "center",
        "justifyContent": "center",
        "backgroundColor": "#2e2e2e",
    }
    upload_style = {
        "position": "absolute",
        "top": 0,
        "left": 0,
        "right": 0,
        "bottom": 0,
        "zIndex": 30,
        "width": "100%",
        "height": "100%",
    }
    if filepath:
        overlay_style["display"] = "none"
        upload_style["display"] = "none"
        upload_style["pointerEvents"] = "none"
    return overlay_style, upload_style

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
     Input('toggle-theme', 'checked'),
     Input('ctrl-geo-stride', 'value'),
     Input('ctrl-fl-type', 'value'),
     Input('ctrl-fl-s-idx', 'value'),
     Input('ctrl-fl-nlines', 'value'),
     Input('ctrl-fl-transits', 'value'),
     Input('ctrl-fl-alpha0', 'value'),
     Input('ctrl-fl-res', 'value'),
     Input('ctrl-fl-zeta-shift', 'checked')],
    State('vmec-meta', 'data'),
    State('ctrl-phi', 'value'),
    prevent_initial_call=True
)
def update_visualization(view, filepath, var_1d, type_2d, var_2d, s_2d, var_3d, s_3d, bg_3d, dark_mode, geo_count, 
                         fl_type, fl_s_idx, fl_nlines, fl_transits, fl_alpha0, fl_res, fl_zeta_shift,
                         meta, phi_state):
    triggered = ctx.triggered_id
    if (
        filepath
        and view == "2d"
        and type_2d == "cross_section"
        and var_2d != "geometry"
        and triggered == "ctrl-phi"
    ):
        return dash.no_update, dash.no_update

    dark_mode = True if dark_mode is None else bool(dark_mode)
    reset_seed = 0
    theme = build_theme(dark_mode, reset_seed)
    if not filepath:
        empty_fig = make_empty_figure(theme, "Please load a .nc file")
        return empty_fig, []

    try:
        vmec = VMECJaxProcessor.from_file(filepath)
        scalars = vmec.get_scalars()
        stats_payload = vmec_stats.overview_stats(vmec, scalars)
        base_cards = base_stat_cards(stats_payload)
        field_map = {opt["value"]: opt.get("label", opt["value"]) for opt in vmec.available_fields()}
        summary_lines = meta.get("summary_lines") if meta else None
        if not summary_lines and view == "overview":
            summary_lines = vmec.get_summary_lines()

        # --------------------------
        # MODE: OVERVIEW
        # --------------------------
        if view == "overview":
            fig = overview_figure(vmec, theme)
            return fig, overview_stats_cards(stats_payload)

        # --------------------------
        # MODE: 1D PROFILES
        # --------------------------
        if view == "1d":
            fig = profiles.render_profile(vmec, var_1d, theme)
            return fig, base_cards

        # --------------------------
        # MODE: 2D CROSS SECTION
        # --------------------------
        if view == "2d":
            phi_val = phi_state if phi_state is not None else 0.0
            phi_scale = 2 * np.pi / max(vmec.nfp, 1)
            phi_angle = phi_val * phi_scale
            s_idx = int(s_2d) if s_2d is not None else vmec.ns - 1
            field_label = field_map.get(var_2d, var_2d)

            if type_2d == "cross_section":
                if var_2d == "geometry":
                    fig = two_d.build_geometry_cross_section_figure(
                        vmec, phi_angle, s_idx, geo_count, dark_mode, theme.fig_template, theme.paper_bg, theme.plot_bg, reset_seed
                    )
                    return fig, base_cards
                fig = two_d.render_cross_section_field(vmec, phi_angle, var_2d, field_label, theme)
                return fig, base_cards

            fig = two_d.render_flux_surface(vmec, s_idx, var_2d, field_label, theme)
            return fig, base_cards

        # --------------------------
        # MODE: 3D GEOMETRY
        # --------------------------
        if view == "3d":
            s_val = int(s_3d) if s_3d is not None else vmec.ns - 1
            v_name = var_3d or "modB"
            coord_free = True if bg_3d is None else bool(bg_3d)
            field_label = field_map.get(v_name, v_name)
            fig = three_d.render_3d(vmec, s_val, v_name, field_label, coord_free, theme)
            return fig, base_cards

        # --------------------------
        # MODE: FIELD LINE
        # --------------------------
        if view == "fieldline":
            res = fl_res if fl_res else 128
            if fl_type == "2d_modB":
                fig = fieldline.render_fieldline_heatmap(vmec, fl_s_idx, res, theme, shift_zeta=bool(fl_zeta_shift))
                return fig, base_cards
            if fl_type == "1d_lines":
                n_lines = fl_nlines if fl_nlines else 6
                fig = fieldline.render_fieldline_lines(vmec, fl_s_idx, n_lines, res, theme, shift_zeta=bool(fl_zeta_shift))
                return fig, base_cards
            if fl_type == "single_trace":
                fig = fieldline.render_single_trace(vmec, fl_s_idx, fl_transits, fl_alpha0, res, theme)
                return fig, base_cards
    except Exception as e:
        print(f"Detailed Error: {e}")
        import traceback
        traceback.print_exc()
        err_fig = make_empty_figure(theme, f"Error: {str(e)}")
        return err_fig, []

    return make_empty_figure(theme, "Please load a .nc file"), []


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
    function(var_name, view, type_2d) {
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

# Force a resize after a file is loaded so Plotly fills the container
app.clientside_callback(
    """
    function(filepath) {
        if (!filepath) {
            return window.dash_clientside.no_update;
        }
        setTimeout(function() {
            window.dispatchEvent(new Event('resize'));
        }, 50);
        return Date.now();
    }
    """,
    Output('resize-ping', 'data'),
    Input('stored-filepath', 'data'),
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
def update_cross_section_phi(phi_val, view, type_2d, var_2d, s_idx, geo_count, filepath, dark_mode):
    if not (filepath and view == '2d' and type_2d == 'cross_section'):
        return dash.no_update
    phi_val = phi_val or 0.0
    try:
        vmec = VMECJaxProcessor.from_file(filepath)
        phi_scale = 2 * np.pi / max(vmec.nfp, 1)
        phi_angle = phi_val * phi_scale
        s_idx = int(s_idx) if s_idx is not None else vmec.ns - 1
        dark_mode = True if dark_mode is None else bool(dark_mode)
        theme = build_theme(dark_mode, 0)
        if var_2d == 'geometry':
            fig = two_d.build_geometry_cross_section_figure(
                vmec, phi_angle, s_idx, geo_count, dark_mode, theme.fig_template, theme.paper_bg, theme.plot_bg, reset_seed=0
            )
            return fig
        field_map = {opt["value"]: opt.get("label", opt["value"]) for opt in vmec.available_fields()}
        field_label = field_map.get(var_2d, var_2d)
        return two_d.render_cross_section_field(vmec, phi_angle, var_2d, field_label, theme)
    except Exception as exc:
        print(f"Cross-section phi update error: {exc}")
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

@app.callback(
    [Output('group-fl-nlines', 'style'),
     Output('group-fl-transits', 'style'),
     Output('group-fl-alpha0', 'style'),
     Output('ctrl-fl-zeta-shift', 'style')],
    Input('ctrl-fl-type', 'value')
)
def toggle_fl_controls(fl_type):
    show = {"display": "flex"}
    hide = {"display": "none"}
    
    if fl_type == '1d_lines':
        return show, hide, hide, {"display": "flex"}
    elif fl_type == 'single_trace':
        return hide, show, show, {"display": "none"}
    else: # 2d_modB
        return hide, hide, hide, {"display": "flex"}

if __name__ == '__main__':
    app.run(debug=True)
