import dash
from dash import dcc, html, Input, Output, State, ctx
import dash_mantine_components as dmc
from dash_iconify import DashIconify
import plotly.graph_objects as go
import base64
import tempfile
import os
import numpy as np

# 引入 JAX 后端
from vmec_jax copy import VMECJaxProcessor

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
            tooltip={"placement": "bottom", "always_visible": True}
        ),
        dmc.Text("Flux Surface Index (s)", size="sm", fw=500, mt="sm"),
        dcc.Slider(
            id="control-s-idx", min=0, max=127, step=1, value=127,
            marks={0: 'Axis', 127: 'Edge'},
            tooltip={"placement": "bottom", "always_visible": True}
        ),
    ]
)

# 3D Controls
controls_3d = html.Div(
    id="wrapper-3d",
    style={"display": "block"},
    children=[
        dmc.Alert(
            "3D view rendered using JAX acceleration. Shows LCFS (Last Closed Flux Surface).",
            title="Info",
            color="indigo",
            variant="light",
            icon=get_icon("tabler:info-circle"),
            mt="md"
        )
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
    )
], gap="md", style={"height": "100%"})

# Main Layout
app.layout = dmc.MantineProvider(
    theme={"fontFamily": "'Inter', sans-serif", "primaryColor": "indigo"},
    children=[
        dcc.Store(id='stored-filepath'),
        dcc.Store(id='vmec-meta'), # Store ns, nfp etc to update slider ranges
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
     Output('control-s-idx', 'marks')],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    prevent_initial_call=True
)
def handle_upload(contents, filename):
    if not contents: return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
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
        return path, f"Active: {filename}", {"ns": ns, "nfp": nfp}, ns-1, marks
    except Exception as e:
        return dash.no_update, f"Error: {str(e)}", dash.no_update, dash.no_update, dash.no_update

@app.callback(
    [Output('main-graph', 'figure'), Output('stats-grid', 'children')],
    [Input('plot-mode', 'value'), 
     Input('stored-filepath', 'data'),
     Input('control-var-1d', 'value'), 
     Input('control-2d-type', 'value'),
     Input('control-var-2d', 'value'),
     Input('control-phi', 'value'), 
     Input('control-s-idx', 'value')],
    prevent_initial_call=True
)
def update_visualization(mode, filepath, var_1d, type_2d, var_2d, phi_val, s_idx):
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

        fig = go.Figure()

        if mode == "summary":
            s, iota = vmec.get_1d_data('iota')
            s, pres = vmec.get_1d_data('pressure')
            fig.add_trace(go.Scatter(x=s, y=iota, name="iota", line_color="#4c6ef5"))
            fig.add_trace(go.Scatter(x=s, y=pres, name="pressure", yaxis="y2", line_color="#fa5252"))
            fig.update_layout(
                title="Quick Summary",
                xaxis=dict(title="Normalized Flux (s)"),
                yaxis=dict(title="iota", titlefont=dict(color="#4c6ef5"), tickfont=dict(color="#4c6ef5")),
                yaxis2=dict(title="Pressure", titlefont=dict(color="#fa5252"), tickfont=dict(color="#fa5252"), overlaying="y", side="right"),
                template="plotly_white",
                margin=dict(l=60, r=60, t=50, b=50)
            )
            
        elif mode == "1d":
            var = var_1d if var_1d else 'iota'
            s, y = vmec.get_1d_data(var)
            fig.add_trace(go.Scatter(x=s, y=y, mode='lines', fill='tozeroy', line_color="#4c6ef5"))
            fig.update_layout(title=f"Profile: {var}", xaxis_title="s", yaxis_title=var, template="plotly_white")
            
        elif mode == "2d":
            phi_frac = phi_val if phi_val is not None else 0.0
            s_index = int(s_idx) if s_idx is not None else -1
            
            if type_2d == "cross_section":
                # R-Z Plot
                # We plot contours of the variable on the R-Z plane
                # Or if variable is 'geometry', we plot flux surfaces
                
                if var_2d == 'geometry':
                    # Plot multiple flux surfaces
                    # We'll pick 10 surfaces evenly spaced
                    surfaces = np.linspace(0, vmec.ns-1, 10, dtype=int)
                    # Add the selected one too
                    if s_index not in surfaces: surfaces = np.append(surfaces, s_index)
                    surfaces = np.sort(surfaces)
                    
                    # Let's just use the grid function to get a nice contour plot of s
                    r_grid, z_grid, val_grid = vmec.get_cross_section_data(phi_frac * 2 * np.pi / vmec.nfp, 'geometry', res_s=40, res_u=128)
                    
                    # Plot s contours (flux surfaces)
                    fig.add_trace(go.Contour(
                        x=r_grid.flatten(), 
                        y=z_grid.flatten(), 
                        z=val_grid.flatten(),
                        colorscale="Greys",
                        contours=dict(
                            start=0,
                            end=1,
                            size=0.1,
                            showlabels=True,
                            coloring='lines'
                        ),
                        line_width=1
                    ))
                    
                    fig.update_layout(title=f"Flux Surfaces at phi={phi_frac:.2f}", xaxis_title="R [m]", yaxis_title="Z [m]")
                    
                else:
                    # Plot heatmap of variable
                    r_grid, z_grid, val_grid = vmec.get_cross_section_data(phi_frac * 2 * np.pi / vmec.nfp, var_2d, res_s=64, res_u=128)
                    
                    fig.add_trace(go.Contour(
                        x=r_grid.flatten(), 
                        y=z_grid.flatten(), 
                        z=val_grid.flatten(),
                        colorscale="Plasma",
                        colorbar=dict(title=var_2d),
                        contours=dict(coloring='heatmap')
                    ))
                    fig.update_layout(title=f"{var_2d} on Cross-Section at phi={phi_frac:.2f}", xaxis_title="R [m]", yaxis_title="Z [m]")

                fig.update_yaxes(scaleanchor="x", scaleratio=1)
                
            elif type_2d == "flux_surface":
                # Theta-Zeta Plot
                theta, zeta, val = vmec.get_flux_surface_data(s_index, var_2d)
                
                if theta is not None:
                    fig.add_trace(go.Heatmap(
                        x=np.degrees(zeta), 
                        y=np.degrees(theta), 
                        z=val,
                        colorscale="Viridis",
                        colorbar=dict(title=var_2d)
                    ))
                    fig.update_layout(
                        title=f"{var_2d} on Flux Surface s={s_index/(vmec.ns-1):.2f}",
                        xaxis_title="Zeta (Toroidal) [deg]",
                        yaxis_title="Theta (Poloidal) [deg]"
                    )
            
        elif mode == "3d":
            x, y, z, b = vmec.compute_3d_surface(resolution=120)
            fig.add_trace(go.Surface(
                x=x, y=y, z=z, surfacecolor=b, colorscale="Plasma",
                colorbar=dict(title="|B| [T]", len=0.5),
                lighting=dict(ambient=0.6, roughness=0.1, specular=0.2)
            ))
            fig.update_layout(
                title="LCFS Geometry (JAX Accelerated)",
                scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode='data'),
                margin=dict(l=0, r=0, t=40, b=0)
            )
        
        vmec.close()
        return fig, stats_ui

    except Exception as e:
        import traceback
        traceback.print_exc()
        err_fig = go.Figure()
        err_fig.add_annotation(text=f"Error: {str(e)}", showarrow=False, font=dict(color="red", size=16))
        return err_fig, []

if __name__ == '__main__':
    app.run(debug=True)
