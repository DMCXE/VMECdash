import dash
from dash import dcc, html, Input, Output, State, ctx
import dash_mantine_components as dmc
from dash_iconify import DashIconify
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
import tempfile
import os
import numpy as np

# 引入 JAX 后端
from vmec_jax import VMECJaxProcessor

# -----------------------
# 1. App Configuration
# -----------------------
app = dash.Dash(
    __name__, 
    title="VMEC Viz",
    suppress_callback_exceptions=True
)
server = app.server

# -----------------------
# UI Helpers
# -----------------------
def get_icon(icon):
    return DashIconify(icon=icon, height=20)

def create_stat_card(title, value, icon_name, color):
    return dmc.Paper(
        children=[
            dmc.Group([
                dmc.Text(title, c="dimmed", size="xs", fw=700, tt="uppercase"),
                dmc.ThemeIcon(get_icon(icon_name), color=color, variant="light", size="sm")
            ], justify="space-between", mb="xs"),
            dmc.Text(value, fw=700, size="xl"),
        ],
        p="md",
        radius="md",
        withBorder=True,
        shadow="xs"
    )


def build_select_data(options):
    data = []
    for opt in options or []:
        category = opt.get("category")
        prefix = f"[{category}] " if category else ""
        data.append({"label": f"{prefix}{opt['label']}", "value": opt["value"]})
    return data

# -----------------------
# Control Panels
# -----------------------

# 1D Controls
controls_1d = html.Div(
    id="wrapper-1d",
    style={"display": "none"},
    children=[
        dmc.Select(
            id="control-var-1d", label="Variable", value="iotaf",
            data=[
                {"label": "Rotational Transform (iota)", "value": "iotaf"},
                {"label": "Safety Factor (q)", "value": "q"},
                {"label": "Pressure", "value": "presf"},
                {"label": "Enclosed Volume (Vp)", "value": "vp"},
                {"label": "<j^u> (Flux Avg)", "value": "jcuru"},
                {"label": "<B·B> (Flux Avg)", "value": "bdotb"},
            ],
            mb="md"
        )
    ]
)

# 2D Controls
controls_2d = html.Div(
    id="wrapper-2d",
    style={"display": "none"},
    children=[
        dmc.SegmentedControl(
            id="control-2d-type",
            value="cross_section",
            fullWidth=True,
            data=[
                {"value": "cross_section", "label": "Cross Section (R-Z)"},
                {"value": "flux_surface", "label": "Flux Surface (u-v)"},
            ],
            mb="md",
            color="cyan"
        ),
        dmc.Select(
            id="control-var-2d", label="Color Variable", value="geometry",
            data=[
                {"label": "Geometry Only (None)", "value": "geometry"},
                {"label": "|B| (Mod B)", "value": "modB"},
                {"label": "sqrt(g) (Jacobian)", "value": "jacobian"},
                {"label": "Lambda", "value": "lambda"},
                {"label": "B^u (Contravariant)", "value": "B^u"},
                {"label": "B^v (Contravariant)", "value": "B^v"},
                {"label": "j^u (Current)", "value": "j^u"},
                {"label": "j^v (Current)", "value": "j^v"},
            ],
            mb="md"
        ),
        dmc.Text("Toroidal Angle (Phi / Zeta)", size="sm", fw=500, mt="sm"),
        dcc.Slider(
            id="control-phi", min=0, max=1, step=0.05, value=0, 
            marks={0: '0', 0.5: '0.5', 1: '1'},
            tooltip={"placement": "bottom", "always_visible": True},
            updatemode='drag'
        ),
        dmc.Text("Flux Surface Index (s)", size="sm", fw=500, mt="sm"),
        dmc.Group([
            dmc.ActionIcon(get_icon("tabler:minus"), id="btn-s-dec", variant="light", color="indigo"),
            html.Div(
                dcc.Slider(
                    id="control-s-idx", min=0, max=127, step=1, value=127,
                    marks={0: 'Axis', 127: 'Edge'},
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
                style={"flex": 1}
            ),
            dmc.ActionIcon(get_icon("tabler:plus"), id="btn-s-inc", variant="light", color="indigo"),
        ], gap="xs", align="center"),
    ]
)

# 3D Controls
controls_3d = html.Div(
    id="wrapper-3d",
    style={"display": "block"},
    children=[
        dmc.Alert(
            "3D view rendered using JAX acceleration.",
            title="Info",
            color="indigo",
            variant="light",
            icon=get_icon("tabler:info-circle"),
            mt="md",
            mb="md"
        ),
        dmc.Select(
            id="control-var-3d", label="Color Variable", value="modB",
            data=[
                {"label": "Geometry Only (Solid)", "value": "geometry"},
                {"label": "|B| (Mod B)", "value": "modB"},
                {"label": "sqrt(g) (Jacobian)", "value": "jacobian"},
                {"label": "B^u", "value": "B^u"},
                {"label": "B^v", "value": "B^v"},
                {"label": "j^u", "value": "j^u"},
                {"label": "j^v", "value": "j^v"},
            ],
            mb="md"
        ),
        dmc.Text("Flux Surface Index (s)", size="sm", fw=500, mt="sm"),
        dmc.Group([
            dmc.ActionIcon(get_icon("tabler:minus"), id="btn-s3d-dec", variant="light", color="indigo"),
            html.Div(
                dcc.Slider(
                    id="control-s-idx-3d", min=0, max=127, step=1, value=127,
                    marks={0: 'Axis', 127: 'Edge'},
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
                style={"flex": 1}
            ),
            dmc.ActionIcon(get_icon("tabler:plus"), id="btn-s3d-inc", variant="light", color="indigo"),
        ], gap="xs", align="center"),
        dmc.Switch(
            id="control-3d-bg",
            label="Coordinate-free background",
            checked=True,
            mt="md",
            color="indigo"
        )
    ]
)

# Sidebar
sidebar_content = dmc.Stack([
    dmc.Group([
        # dmc.ThemeIcon(get_icon("mdi:atom-variant"), size="lg", radius="xl", color="indigo"),
        dmc.ThemeIcon(get_icon("picon:infinity"), size="lg", radius="xl", color="indigo"),
        dmc.Text("VMEC Viewer", size="xl", fw=700),
    ]),
    dmc.Divider(),
    dcc.Upload(
        id='upload-data',
        children=dmc.Container(
            [
                dmc.Center(dmc.ThemeIcon(get_icon("tabler:cloud-upload"), size="xl", color="gray", variant="outline")),
                dmc.Text("Drag & Drop wout file", size="sm", ta="center", mt="sm", c="dimmed"),
            ],
            p="xl",
            style={"border": "2px dashed #ced4da", "borderRadius": "8px", "cursor": "pointer"}
        ),
        multiple=False
    ),
    dmc.Text(id="filename-display", size="xs", c="dimmed", ta="center"),
    dmc.Divider(label="Visualization Mode", labelPosition="center"),
    dmc.SegmentedControl(
        id="plot-mode",
        value="3d",
        fullWidth=True,
        orientation="vertical",
        data=[
            {"value": "summary", "label": "Summary Dashboard"},
            {"value": "1d", "label": "1D Profiles"},
            {"value": "2d", "label": "2D Slices & Contours"},
            {"value": "3d", "label": "3D Geometry"},
        ],
        color="indigo",
        size="md"
    ),
    
    dmc.Divider(label="Controls", labelPosition="center"),
    html.Div([
        controls_1d,
        controls_2d,
        controls_3d
    ]),

    dmc.Space(h="xl"),
    dmc.Button(
        "Download Plot",
        id="btn-download",
        variant="outline",
        fullWidth=True,
        leftSection=get_icon("tabler:photo-down"),
        disabled=True
    ),
    dmc.Alert(
        id="status-alert",
        title="Status",
        children="Ready",
        color="gray",
        variant="light",
        hide=True,
        icon=get_icon("mdi:information-outline")
    )
], gap="md", style={"height": "100%"})

# Main Layout
app.layout = dmc.MantineProvider(
    theme={"fontFamily": "'Inter', sans-serif", "primaryColor": "indigo"},
    children=[
        dcc.Store(id='stored-filepath'),
        dcc.Store(id='vmec-meta'), # Store ns, nfp etc to update slider ranges
        dcc.Store(id='store-2d-data'), # Store pre-computed 2D slices
        dmc.Grid([
            dmc.GridCol(
                dmc.Paper(sidebar_content, p="md", withBorder=True, style={"height": "100vh", "borderRadius": 0, "overflowY": "auto"}),
                span=3
            ),
            dmc.GridCol(
                dmc.Container([
                    dmc.Stack([
                        html.Div(id="stats-grid"),
                        dmc.Card(
                            shadow="sm", radius="md", withBorder=True,
                            children=[
                                dcc.Loading(
                                    type="circle", 
                                    color="#4c6ef5",
                                    children=dcc.Graph(id="main-graph", style={"height": "85vh"}, config={"scrollZoom": True})
                                )
                            ]
                        )
                    ], gap="md")
                ], fluid=True, p="md"),
                span=9
            )
        ], gutter=0)
    ]
)

# -----------------------
# Callbacks
# -----------------------

app.clientside_callback(
    """
    function(phi_val, data_store, mode, type_2d, var_2d) {
        // Safety check for undefined inputs
        if (phi_val === undefined || phi_val === null) {
            return window.dash_clientside.no_update;
        }
        
        // Only update if we are in the correct mode
        if (mode !== '2d' || type_2d !== 'cross_section' || var_2d === 'geometry') {
            return window.dash_clientside.no_update;
        }

        if (!data_store || !data_store.frames) {
            return window.dash_clientside.no_update;
        }
        
        try {
            // Find closest frame
            // phi_val is 0 to 1
            // frames are indexed by step index 0 to 20
            var step = 0.05;
            var idx = Math.round(phi_val / step);
            if (idx < 0) idx = 0;
            if (idx >= data_store.frames.length) idx = data_store.frames.length - 1;
            
            var frame = data_store.frames[idx];
            
            if (!frame) {
                return window.dash_clientside.no_update;
            }
            
            var displayLabel = (data_store.var_label || data_store.var_key || 'Field');
            var fig_data = {
                type: 'contour',
                x: frame.r,
                y: frame.z,
                z: frame.val,
                colorscale: "Plasma",
                colorbar: {title: displayLabel},
                contours: {coloring: 'heatmap'},
                ncontours: 50,
                line: {width: 0}
            };
            
            var layout = {
                title: displayLabel + " on Cross-Section at phi=" + (phi_val).toFixed(2),
                xaxis: {title: "R [m]"},
                yaxis: {title: "Z [m]", scaleanchor: "x", scaleratio: 1},
                template: "plotly_white"
            };
            
            return {data: [fig_data], layout: layout};
        } catch (e) {
            console.error("Clientside callback error:", e);
            return window.dash_clientside.no_update;
        }
    }
    """,
    Output('main-graph', 'figure', allow_duplicate=True),
    Input('control-phi', 'value'),
    Input('store-2d-data', 'data'),
    State('plot-mode', 'value'),
    State('control-2d-type', 'value'),
    State('control-var-2d', 'value'),
    prevent_initial_call=True
)

app.clientside_callback(
    """
    function(n_clicks) {
        if (n_clicks) {
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
                width: 1200,
                height: 900,
                filename: 'vmec_plot'
            });
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output('btn-download', 'id'), # Dummy output
    Input('btn-download', 'n_clicks'),
    prevent_initial_call=True
)

@app.callback(
    Output('control-s-idx', 'value', allow_duplicate=True),
    [Input('btn-s-dec', 'n_clicks'),
     Input('btn-s-inc', 'n_clicks')],
    [State('control-s-idx', 'value'),
     State('control-s-idx', 'max')],
    prevent_initial_call=True
)
def update_s_idx(n_dec, n_inc, current_val, max_val):
    ctx_id = ctx.triggered_id
    if not current_val: current_val = 0
    if not max_val: max_val = 127
    
    if ctx_id == 'btn-s-dec':
        return max(0, current_val - 1)
    elif ctx_id == 'btn-s-inc':
        return min(max_val, current_val + 1)
    return dash.no_update

@app.callback(
    Output('control-s-idx-3d', 'value', allow_duplicate=True),
    [Input('btn-s3d-dec', 'n_clicks'),
     Input('btn-s3d-inc', 'n_clicks')],
    [State('control-s-idx-3d', 'value'),
     State('control-s-idx-3d', 'max')],
    prevent_initial_call=True
)
def update_s3d_idx(n_dec, n_inc, current_val, max_val):
    ctx_id = ctx.triggered_id
    if not current_val: current_val = 0
    if not max_val: max_val = 127
    
    if ctx_id == 'btn-s3d-dec':
        return max(0, current_val - 1)
    elif ctx_id == 'btn-s3d-inc':
        return min(max_val, current_val + 1)
    return dash.no_update

@app.callback(
    Output('store-2d-data', 'data'),
    [Input('plot-mode', 'value'),
     Input('control-2d-type', 'value'),
     Input('control-var-2d', 'value'),
     Input('stored-filepath', 'data')],
    prevent_initial_call=True
)
def precompute_2d_slices(mode, type_2d, var_2d, filepath):
    if mode != '2d' or type_2d != 'cross_section' or not filepath or var_2d == 'geometry':
        return dash.no_update
        
    try:
        # coord_free = True if coord_free_bg is None else coord_free_bg
        vmec = VMECJaxProcessor.from_file(filepath)
        field_lookup = {opt["value"]: opt["label"] for opt in vmec.available_fields()}
        field_label = field_lookup.get(var_2d, var_2d)
        frames = []
        # Pre-compute 21 frames (0 to 1 step 0.05)
        steps = np.linspace(0, 1, 21)
        
        for phi_frac in steps:
            phi = phi_frac * 2 * np.pi / vmec.nfp
            r, z, val = vmec.get_cross_section_grid(phi, var_2d, res_grid=100) # Lower res for speed/storage
            
            # Replace NaN with null for JSON
            val_list = np.where(np.isnan(val), None, val).tolist()
            
            frames.append({
                "r": r.tolist(),
                "z": z.tolist(),
                "val": val_list
            })
        return {"frames": frames, "var_key": var_2d, "var_label": field_label}
    except Exception as e:
        print(f"Precompute error: {e}")
        return dash.no_update

@app.callback(
    [Output("wrapper-1d", "style"),
     Output("wrapper-2d", "style"),
     Output("wrapper-3d", "style")],
    Input("plot-mode", "value")
)
def toggle_controls(mode):
    hide = {"display": "none"}
    show = {"display": "block"}
    
    if mode == "1d":
        return show, hide, hide
    elif mode == "2d":
        return hide, show, hide
    elif mode == "3d":
        return hide, hide, show
    else: # summary
        return hide, hide, hide

@app.callback(
    [Output('stored-filepath', 'data'), 
     Output('filename-display', 'children'),
     Output('vmec-meta', 'data'),
     Output('control-s-idx', 'max'),
     Output('control-s-idx', 'marks'),
     Output('control-s-idx-3d', 'max'),
     Output('control-s-idx-3d', 'marks'),
     Output('btn-download', 'disabled'),
     Output('control-s-idx', 'value', allow_duplicate=True),
     Output('control-s-idx-3d', 'value', allow_duplicate=True)],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    prevent_initial_call=True
)
def handle_upload(contents, filename):
    if not contents: return [dash.no_update] * 10
    
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    fd, path = tempfile.mkstemp(suffix=".nc")
    with os.fdopen(fd, 'wb') as f: f.write(decoded)
    
    try:
        vmec = VMECJaxProcessor.from_file(path)
        ns = vmec.ns
        meta = {
            "ns": ns,
            "nfp": vmec.nfp,
            "profiles": vmec.available_profiles(),
            "fields": vmec.available_fields(),
            "summary_lines": vmec.get_summary_lines(),
        }
        marks = {0: 'Axis', ns-1: 'Edge'}
        return path, f"Active: {filename}", meta, ns-1, marks, ns-1, marks, False, ns-1, ns-1
    except Exception as e:
        return dash.no_update, f"Error: {str(e)}", dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update


@app.callback(
    Output('control-var-1d', 'data'),
    Output('control-var-1d', 'value'),
    Output('control-var-2d', 'data'),
    Output('control-var-2d', 'value'),
    Output('control-var-3d', 'data'),
    Output('control-var-3d', 'value'),
    Input('vmec-meta', 'data'),
    prevent_initial_call=True
)
def populate_variable_options(meta):
    if not meta:
        raise dash.exceptions.PreventUpdate
    profile_data = build_select_data(meta.get('profiles'))
    field_data = build_select_data(meta.get('fields'))
    profile_value = profile_data[0]['value'] if profile_data else dash.no_update
    field_value = field_data[0]['value'] if field_data else dash.no_update
    field_data_3d = [dict(opt) for opt in field_data]
    return profile_data, profile_value, field_data, field_value, field_data_3d, field_value

@app.callback(
    [Output('main-graph', 'figure'), Output('stats-grid', 'children')],
    [Input('plot-mode', 'value'), 
     Input('stored-filepath', 'data'),
     Input('control-var-1d', 'value'), 
     Input('control-2d-type', 'value'),
     Input('control-var-2d', 'value'),
     Input('control-phi', 'value'), 
     Input('control-s-idx', 'value'),
     Input('control-var-3d', 'value'),
     Input('control-s-idx-3d', 'value'),
     Input('control-3d-bg', 'checked')],
    State('vmec-meta', 'data'),
    prevent_initial_call=True
)
def update_visualization(mode, filepath, var_1d, type_2d, var_2d, phi_val, s_idx, var_3d, s_idx_3d, coord_free_bg, meta):
    empty_fig = go.Figure()
    empty_fig.update_layout(template='plotly_white', xaxis={'visible': False}, yaxis={'visible': False})
    empty_fig.add_annotation(text='Please upload a wout.nc file', showarrow=False, font=dict(size=20))
    if not filepath:
        return empty_fig, []

    coord_free = True if coord_free_bg is None else bool(coord_free_bg)

    try:
        vmec = VMECJaxProcessor.from_file(filepath)
        scalars = vmec.get_scalars()
        profile_map = {opt['value']: opt['label'] for opt in vmec.available_profiles()}
        field_map = {opt['value']: opt['label'] for opt in vmec.available_fields()}

        def fmt(value, pattern='{:.2f}', fallback='--'):
            if value is None:
                return fallback
            try:
                if np.isnan(value):
                    return fallback
            except TypeError:
                pass
            return pattern.format(value)

        stats_cards = dmc.SimpleGrid(cols=4, spacing='md', children=[
            create_stat_card('Beta Total', f"{scalars['beta_total']*100:.2f}%", 'mdi:percent', 'red'),
            create_stat_card('Volume', f"{scalars['volume']:.1f} m3", 'mdi:cube-outline', 'blue'),
            create_stat_card('Edge q', fmt(scalars.get('q_edge')), 'mdi:chart-line', 'teal'),
            create_stat_card('Toroidal Current', f"{scalars['ctor']:.2f} MA", 'mdi:current-ac', 'orange'),
        ])
        stats_ui = stats_cards

        if mode == '2d' and type_2d == 'cross_section' and var_2d != 'geometry':
            return dash.no_update, stats_ui

        if mode == 'summary':
            summary_lines = vmec.get_summary_lines()

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
                                dmc.Group(
                                    [
                                        dmc.Kbd(key.strip(), style={"minWidth": 70}),
                                        dmc.Text(value.strip(), fw=600)
                                    ],
                                    gap="xs",
                                    align="center"
                                )
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

            summary_block = dmc.Card(
                withBorder=True,
                radius='md',
                shadow='xs',
                children=[
                    dmc.Group(
                        [
                            dmc.Text('Equilibrium snapshot', fw=600, size='sm'),
                            dmc.Badge('VMEC', color='indigo', variant='light')
                        ],
                        justify='space-between',
                        mb='sm'
                    ),
                    render_summary_panels(summary_lines)
                ]
            )
            stats_ui = dmc.Stack([stats_cards, summary_block], gap='sm')

            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=(
                    'Rotational Transform (iota)',
                    'Safety Factor (q)',
                    'Pressure Profile',
                    'dP/ds',
                    'Volume Enclosed (Vp)',
                    'Flux Avg <B·B>'
                ),
                vertical_spacing=0.15,
                horizontal_spacing=0.08
            )
            s_iota, iota = vmec.get_1d_data('iotaf')
            s_q, q_profile = vmec.get_1d_data('q')
            s_p, pres = vmec.get_1d_data('presf')
            s_dp, dpds = vmec.get_1d_data('dpds')
            s_vp, vp = vmec.get_1d_data('vp')
            s_b, bdotb = vmec.get_1d_data('bdotb')

            fig.add_trace(go.Scatter(x=s_iota, y=iota, name='iota', line=dict(color='#4c6ef5', width=3)), row=1, col=1)
            fig.add_trace(go.Scatter(x=s_q, y=q_profile, name='q', line=dict(color='#fd7e14', width=3)), row=1, col=2)
            fig.add_trace(go.Scatter(x=s_p, y=pres, name='pressure', line=dict(color='#fa5252', width=3)), row=1, col=3)
            fig.add_trace(go.Scatter(x=s_dp, y=dpds, name='dP/ds', line=dict(color='#12b886', width=3)), row=2, col=1)
            fig.add_trace(go.Scatter(x=s_vp, y=vp, name='Vp', line=dict(color='#7950f2', width=3)), row=2, col=2)
            fig.add_trace(go.Scatter(x=s_b, y=bdotb, name='<B·B>', line=dict(color='#0ca678', width=3)), row=2, col=3)

            fig.update_layout(
                title_text='Equilibrium Summary',
                template='plotly_white',
                height=900,
                showlegend=False
            )
            for axis_key in fig.layout:
                if axis_key.startswith('xaxis'):
                    fig.layout[axis_key].title = 'Normalized Flux (s)'

        elif mode == '1d':
            var = var_1d or 'iotaf'
            label = profile_map.get(var, var)
            s, y = vmec.get_1d_data(var)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=s, y=y, mode='lines', line=dict(color='#4c6ef5', width=3)))
            fig.update_layout(title=f'Profile: {label}', xaxis_title='s', yaxis_title=label, template='plotly_white')

        elif mode == '2d':
            phi_frac = phi_val if phi_val is not None else 0.0
            s_index = int(s_idx) if s_idx is not None else -1
            s_index = max(0, min(s_index, vmec.ns - 1))
            field_label = field_map.get(var_2d, var_2d)
            fig = go.Figure()

            if type_2d == 'cross_section':
                if var_2d == 'geometry':
                    surfaces = np.linspace(0, vmec.ns-1, 15, dtype=int)
                    if s_index not in surfaces:
                        surfaces = np.append(surfaces, s_index)
                    surfaces = np.sort(surfaces)
                    r_grid, z_grid, _ = vmec.get_cross_section_data(
                        phi_frac * 2 * np.pi / vmec.nfp, 'geometry', res_s=vmec.ns, res_u=160
                    )
                    for s_i in surfaces:
                        if s_i >= len(r_grid):
                            continue
                        r_line = np.append(r_grid[s_i], r_grid[s_i][0])
                        z_line = np.append(z_grid[s_i], z_grid[s_i][0])
                        color = 'red' if s_i == s_index else 'rgba(0,0,0,0.3)'
                        width = 3 if s_i == s_index else 1
                        fig.add_trace(go.Scatter(
                            x=r_line, y=z_line, mode='lines',
                            line=dict(color=color, width=width),
                            name=f's={s_i}',
                            hoverinfo='skip'
                        ))
                    fig.update_layout(
                        title=f'Flux surfaces at phi={phi_frac:.2f}',
                        xaxis_title='R [m]',
                        yaxis_title='Z [m]',
                        uirevision=f"{mode}-{type_2d}-{var_2d}-{phi_val}"
                    )
                else:
                    r_lin, z_lin, val_grid = vmec.get_cross_section_grid(
                        phi_frac * 2 * np.pi / vmec.nfp, var_2d, res_grid=220
                    )
                    fig.add_trace(go.Contour(
                        x=r_lin, y=z_lin, z=val_grid,
                        colorscale='Plasma',
                        colorbar=dict(title=field_label),
                        contours=dict(coloring='heatmap'),
                        ncontours=50,
                        line_width=0
                    ))
                    fig.update_layout(
                        title=f"{field_label} on cross-section at phi={phi_frac:.2f}",
                        xaxis_title='R [m]',
                        yaxis_title='Z [m]'
                    )
                fig.update_yaxes(scaleanchor='x', scaleratio=1)

            elif type_2d == 'flux_surface':
                theta, zeta, val = vmec.get_flux_surface_data(s_index, var_2d, res_u=128, res_v=128)
                if theta is not None:
                    fig.add_trace(go.Contour(
                        x=zeta,
                        y=theta,
                        z=val,
                        colorscale='Viridis',
                        colorbar=dict(title=field_label),
                        contours=dict(coloring='fill', showlines=True),
                        line_width=0.5,
                        ncontours=20
                    ))
                    fig.update_layout(
                        title=f"{field_label} on flux surface s={s_index/(vmec.ns-1):.2f}",
                        xaxis_title='Zeta (toroidal) [rad]',
                        yaxis_title='Theta (poloidal) [rad]',
                        xaxis=dict(range=[0, 2*np.pi/vmec.nfp]),
                        yaxis=dict(range=[0, 2*np.pi])
                    )

        elif mode == '3d':
            s_val = int(s_idx_3d) if s_idx_3d is not None else -1
            s_val = max(0, min(s_val, vmec.ns - 1))
            v_name = var_3d or 'modB'
            field_label = field_map.get(v_name, v_name)
            fig = go.Figure()
            x, y, z, val = vmec.compute_3d_surface(s_idx=s_val, var_name=v_name, resolution=110)
            if v_name == 'geometry':
                surface_color = np.full_like(z, 0.5)
                colorscale = 'Greys'
                show_scale = False
            else:
                surface_color = val
                colorscale = 'Jet'
                show_scale = True
            fig.add_trace(go.Surface(
                x=x, y=y, z=z,
                surfacecolor=surface_color,
                colorscale=colorscale,
                colorbar=dict(title=field_label, len=0.5) if show_scale else None,
                lighting=dict(ambient=0.6, roughness=0.1, specular=0.2)
            ))
            title_txt = f"Flux surface s={s_val/(vmec.ns-1):.2f}"
            if v_name != 'geometry':
                title_txt += f" colored by {field_label}"
            if coord_free:
                def hidden_axis():
                    return dict(
                        visible=False,
                        showgrid=False,
                        zeroline=False,
                        showbackground=False,
                        showticklabels=False
                    )

                scene_axes = dict(
                    xaxis=hidden_axis(),
                    yaxis=hidden_axis(),
                    zaxis=hidden_axis(),
                    bgcolor='rgba(0,0,0,0)',
                    aspectmode='data'
                )
            else:
                def axis_with(title):
                    return dict(
                        title=title,
                        visible=True,
                        showgrid=True,
                        zeroline=False,
                        showbackground=True,
                        backgroundcolor='rgba(245,245,245,1)',
                        showticklabels=True
                    )

                scene_axes = dict(
                    xaxis=axis_with('X'),
                    yaxis=axis_with('Y'),
                    zaxis=axis_with('Z'),
                    bgcolor='rgba(248,249,252,1)',
                    aspectmode='data'
                )

            fig.update_layout(
                title=title_txt,
                scene=scene_axes,
                margin=dict(l=0, r=0, t=40, b=0),
                uirevision=f"{mode}-{v_name}"
            )
        else:
            fig = empty_fig

        return fig, stats_ui

    except Exception as e:
        import traceback
        traceback.print_exc()
        err_fig = go.Figure()
        err_fig.add_annotation(text=f"Error: {str(e)}", showarrow=False, font=dict(color='red', size=16))
        return err_fig, []

app.clientside_callback(
    """
    function(var_name, mode, type_2d) {
        if (mode === '2d' && type_2d === 'cross_section' && var_name !== 'geometry') {
            return [
                false, 
                'Calculating ' + var_name + '...', 
                'blue',
                {'height': '85vh', 'opacity': 0.3, 'transition': 'opacity 0.5s'}
            ];
        }
        return [
            true, 
            'Ready', 
            'gray',
            {'height': '85vh', 'opacity': 1}
        ];
    }
    """,
    Output('status-alert', 'hide'),
    Output('status-alert', 'children'),
    Output('status-alert', 'color'),
    Output('main-graph', 'style'),
    Input('control-var-2d', 'value'),
    Input('plot-mode', 'value'),
    Input('control-2d-type', 'value'),
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
                {'height': '85vh', 'opacity': 1, 'transition': 'opacity 0.5s'}
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
    State('control-var-2d', 'value'),
    prevent_initial_call=True
)

if __name__ == '__main__':
    app.run(debug=True)
