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
from scipy.interpolate import griddata
from dash import Patch

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

# -----------------------
# Control Panels
# -----------------------

# 1D Controls
controls_1d = html.Div(
    id="wrapper-1d",
    style={"display": "none"},
    children=[
        dmc.Select(
            id="control-var-1d", label="Variable", value="iota",
            data=[
                {"label": "Rotational Transform (iota)", "value": "iota"},
                {"label": "Safety Factor (q)", "value": "q"},
                {"label": "Pressure", "value": "pressure"},
                {"label": "<B^u> (Flux Avg)", "value": "<Buco>"},
                {"label": "<B^v> (Flux Avg)", "value": "<Bvco>"},
                {"label": "<j^u> (Flux Avg)", "value": "<jcuru>"},
                {"label": "<j^v> (Flux Avg)", "value": "<jcurv>"},
                {"label": "<B·B> (Flux Avg)", "value": "<B.B>"},
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
                {"label": "|B| (Mod B)", "value": "|B|"},
                {"label": "sqrt(g) (Jacobian)", "value": "sqrt(g)"},
                {"label": "B^u (Contravariant)", "value": "B^u"},
                {"label": "B^v (Contravariant)", "value": "B^v"},
                {"label": "j^u (Current)", "value": "j^u"},
                {"label": "j^v (Current)", "value": "j^v"},
                {"label": "B_u (Covariant)", "value": "B_u"},
                {"label": "B_v (Covariant)", "value": "B_v"},
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
            id="control-var-3d", label="Color Variable", value="|B|",
            data=[
                {"label": "Geometry Only (Solid)", "value": "geometry"},
                {"label": "|B| (Mod B)", "value": "|B|"},
                {"label": "sqrt(g) (Jacobian)", "value": "sqrt(g)"},
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
    ]
)

# Sidebar
sidebar_content = dmc.Stack([
    dmc.Group([
        dmc.ThemeIcon(get_icon("mdi:atom-variant"), size="lg", radius="xl", color="indigo"),
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
            
            var fig_data = {
                type: 'contour',
                x: frame.r,
                y: frame.z,
                z: frame.val,
                colorscale: "Plasma",
                colorbar: {title: data_store.var_name},
                contours: {coloring: 'heatmap'},
                ncontours: 50,
                line: {width: 0}
            };
            
            var layout = {
                title: data_store.var_name + " on Cross-Section at phi=" + (phi_val).toFixed(2),
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
            var graph = document.getElementById('main-graph');
            Plotly.downloadImage(graph, {
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
        vmec = VMECJaxProcessor(filepath)
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
            
        vmec.close()
        return {"frames": frames, "var_name": var_2d}
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
    
    # Quick read to get metadata
    try:
        vmec = VMECJaxProcessor(path)
        ns = vmec.ns
        nfp = vmec.nfp
        vmec.close()
        
        marks = {0: 'Axis', ns-1: 'Edge'}
        return path, f"Active: {filename}", {"ns": ns, "nfp": nfp}, ns-1, marks, ns-1, marks, False, ns-1, ns-1
    except Exception as e:
        return dash.no_update, f"Error: {str(e)}", dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

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
     Input('control-s-idx-3d', 'value')],
    prevent_initial_call=True
)
def update_visualization(mode, filepath, var_1d, type_2d, var_2d, phi_val, s_idx, var_3d, s_idx_3d):
    # If we are in 2D Cross-Section Physics mode, let the client-side callback handle the figure update
    # We only update stats here
    if mode == '2d' and type_2d == 'cross_section' and var_2d != 'geometry':
        # We still need to return stats, but figure is no_update
        # But we need to calculate stats first
        pass 
    
    # Check if this is just a slider update for 2D Geometry
    ctx_id = ctx.triggered_id
    if ctx_id == 'control-s-idx' and mode == '2d' and type_2d == 'cross_section' and var_2d == 'geometry':
        # Use Patch for smooth update
        patched_fig = Patch()
        
        # We need to know which trace corresponds to which s_index
        # This is tricky because we don't have the previous figure state easily accessible here
        # But we can assume the traces are generated in a specific order or named
        # Actually, regenerating the figure is fast for geometry, the "flash" is the issue.
        # If we use Patch, we can't easily change data unless we know the index.
        # Let's stick to full redraw for now but optimize if possible.
        # Wait, the user specifically asked for smoother display.
        # If we can't use Patch easily without state, maybe we can just return the new figure.
        # The flash happens because the client clears the graph before rendering the new one.
        # Dash's default behavior.
        pass

    # 默认空状态
    empty_fig = go.Figure()
    empty_fig.update_layout(template="plotly_white", xaxis={"visible":False}, yaxis={"visible":False})
    empty_fig.add_annotation(text="Please upload a wout.nc file", showarrow=False, font=dict(size=20))
    
    if not filepath: return empty_fig, []

    try:
        vmec = VMECJaxProcessor(filepath)
        scalars = vmec.get_scalars()
        stats_ui = dmc.SimpleGrid(cols=4, spacing="md", children=[
            create_stat_card("Beta Total", f"{scalars['beta_total']*100:.2f}%", "mdi:percent", "red"),
            create_stat_card("Volume", f"{scalars['volume']:.1f} m³", "mdi:cube-outline", "blue"),
            create_stat_card("Aspect Ratio", f"{scalars['aspect']:.2f}", "mdi:ratio", "green"),
            create_stat_card("Field on Axis", f"{scalars['b0']:.2f} T", "mdi:magnet", "orange"),
        ])
        
        # Handover for 2D Physics Sliding
        if mode == '2d' and type_2d == 'cross_section' and var_2d != 'geometry':
            # If this was triggered by slider (phi), we return no_update for figure
            # If triggered by variable change, the client-side callback will pick up the new store data
            # So we can safely return no_update for figure here ALWAYS, 
            # because the store update will trigger the client-side callback.
            vmec.close()
            return dash.no_update, stats_ui

        if mode == "summary":
            # Create a 2x2 subplot summary
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Rotational Transform (iota)", "Pressure Profile", "Flux Average <B·B>", "Current Density <j·B>"),
                vertical_spacing=0.15
            )
            
            s, iota = vmec.get_1d_data('iota')
            s, pres = vmec.get_1d_data('pressure')
            s, bdotb = vmec.get_1d_data('<B.B>')
            s, jcuru = vmec.get_1d_data('<jcuru>') # Approximation for parallel current
            
            fig.add_trace(go.Scatter(x=s, y=iota, name="iota", line=dict(color="#4c6ef5", width=3)), row=1, col=1)
            fig.add_trace(go.Scatter(x=s, y=pres, name="pressure", line=dict(color="#fa5252", width=3), fill='tozeroy'), row=1, col=2)
            fig.add_trace(go.Scatter(x=s, y=bdotb, name="<B·B>", line=dict(color="#12b886", width=3)), row=2, col=1)
            fig.add_trace(go.Scatter(x=s, y=jcuru, name="<j^u>", line=dict(color="#be4bdb", width=3)), row=2, col=2)
            
            fig.update_layout(
                title_text="Equilibrium Summary", 
                template="plotly_white", 
                height=800,
                showlegend=False
            )
            fig.update_xaxes(title_text="Normalized Flux (s)")
            
        elif mode == "1d":
            var = var_1d if var_1d else 'iota'
            s, y = vmec.get_1d_data(var)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=s, y=y, mode='lines', fill='tozeroy', line_color="#4c6ef5"))
            fig.update_layout(title=f"Profile: {var}", xaxis_title="s", yaxis_title=var, template="plotly_white")
            
        elif mode == "2d":
            phi_frac = phi_val if phi_val is not None else 0.0
            s_index = int(s_idx) if s_idx is not None else -1
            # Safety clamp for index
            s_index = max(0, min(s_index, vmec.ns - 1))
            
            fig = go.Figure()
            
            if type_2d == "cross_section":
                # R-Z Plot
                if var_2d == 'geometry':
                    # Plot multiple flux surfaces as lines
                    surfaces = np.linspace(0, vmec.ns-1, 15, dtype=int)
                    # Ensure s_index is included and valid
                    if s_index not in surfaces: surfaces = np.append(surfaces, s_index)
                    surfaces = np.sort(surfaces)
                    
                    # Get data for all surfaces
                    # We use a high resolution for smooth lines
                    r_grid, z_grid, _ = vmec.get_cross_section_data(phi_frac * 2 * np.pi / vmec.nfp, 'geometry', res_s=vmec.ns, res_u=128)
                    
                    # Plot selected surfaces
                    for s_i in surfaces:
                        # Safety check
                        if s_i >= len(r_grid): continue
                        
                        # r_grid is (ns, nu)
                        # We need to close the loop
                        r_line = np.append(r_grid[s_i], r_grid[s_i][0])
                        z_line = np.append(z_grid[s_i], z_grid[s_i][0])
                        
                        color = "red" if s_i == s_index else "rgba(0,0,0,0.3)"
                        width = 3 if s_i == s_index else 1
                        
                        fig.add_trace(go.Scatter(
                            x=r_line, y=z_line, mode='lines', 
                            line=dict(color=color, width=width),
                            name=f"s={s_i}",
                            hoverinfo='skip' # Improve performance
                        ))
                    
                    fig.update_layout(
                        title=f"Flux Surfaces at phi={phi_frac:.2f}", 
                        xaxis_title="R [m]", 
                        yaxis_title="Z [m]",
                        uirevision=f"{mode}-{type_2d}-{var_2d}-{phi_val}" # Preserve zoom/state on slider change
                    )
                    
                else:
                    # Plot heatmap of variable using interpolation
                    # Use the new masked grid method for correct shape
                    r_lin, z_lin, val_grid = vmec.get_cross_section_grid(phi_frac * 2 * np.pi / vmec.nfp, var_2d, res_grid=200)
                    
                    fig.add_trace(go.Contour(
                        x=r_lin, y=z_lin, z=val_grid,
                        colorscale="Plasma",
                        colorbar=dict(title=var_2d),
                        contours=dict(coloring='heatmap'),
                        ncontours=50,
                        line_width=0 # No contour lines for heatmap look
                    ))
                    fig.update_layout(title=f"{var_2d} on Cross-Section at phi={phi_frac:.2f}", xaxis_title="R [m]", yaxis_title="Z [m]")

                fig.update_yaxes(scaleanchor="x", scaleratio=1)
                
            elif type_2d == "flux_surface":
                # Theta-Zeta Plot
                # Increase resolution as requested
                theta, zeta, val = vmec.get_flux_surface_data(s_index, var_2d, res_u=128, res_v=128)
                
                if theta is not None:
                    # Use Contour instead of Heatmap
                    # Axes: Zeta (0 to 2pi/nfp), Theta (0 to 2pi)
                    fig.add_trace(go.Contour(
                        x=zeta, 
                        y=theta, 
                        z=val,
                        colorscale="Viridis",
                        colorbar=dict(title=var_2d),
                        contours=dict(coloring='fill', showlines=True),
                        line_width=0.5,
                        ncontours=20
                    ))
                    fig.update_layout(
                        title=f"{var_2d} on Flux Surface s={s_index/(vmec.ns-1):.2f}",
                        xaxis_title="Zeta (Toroidal) [rad]",
                        yaxis_title="Theta (Poloidal) [rad]",
                        xaxis=dict(range=[0, 2*np.pi/vmec.nfp]),
                        yaxis=dict(range=[0, 2*np.pi]) # Removed scaleanchor to allow stretching
                    )
            
        elif mode == "3d":
            s_val = int(s_idx_3d) if s_idx_3d is not None else -1
            # Safety clamp
            s_val = max(0, min(s_val, vmec.ns - 1))
            
            v_name = var_3d if var_3d else '|B|'
            
            fig = go.Figure()
            
            x, y, z, val = vmec.compute_3d_surface(s_idx=s_val, var_name=v_name, resolution=100)
            
            # If geometry only, use a single color
            if v_name == 'geometry':
                surface_color = np.full_like(z, 0.5)
                colorscale = 'Greys'
                show_scale = False
            else:
                surface_color = val
                colorscale = 'Plasma'
                show_scale = True

            fig.add_trace(go.Surface(
                x=x, y=y, z=z, 
                surfacecolor=surface_color, 
                colorscale=colorscale,
                colorbar=dict(title=v_name, len=0.5) if show_scale else None,
                lighting=dict(ambient=0.6, roughness=0.1, specular=0.2)
            ))
            
            title_txt = f"Flux Surface s={s_val/(vmec.ns-1):.2f}"
            if v_name != 'geometry':
                title_txt += f" colored by {v_name}"
                
            fig.update_layout(
                title=title_txt,
                scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode='data'),
                margin=dict(l=0, r=0, t=40, b=0),
                uirevision=f"{mode}-{var_3d}" # Keep camera angle when changing s_idx
            )
        
        vmec.close()
        return fig, stats_ui

    except Exception as e:
        import traceback
        traceback.print_exc()
        err_fig = go.Figure()
        err_fig.add_annotation(text=f"Error: {str(e)}", showarrow=False, font=dict(color="red", size=16))
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
        if (data && data.var_name === var_name) {
            return [
                false,
                'Rendered: ' + var_name,
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
