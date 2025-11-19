import dash
from dash import dcc, html, Input, Output, State
import dash_mantine_components as dmc
from dash_iconify import DashIconify
import plotly.graph_objects as go
import base64
import tempfile
import os

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
# Layout Definition
# -----------------------

# 定义控件组（预先定义所有控件，通过回调控制显示/隐藏）
# 这样可以避免 "Nonexistent object" 错误
controls_1d = html.Div(
    id="wrapper-1d",
    style={"display": "none"}, # 默认隐藏
    children=[
        dmc.Select(
            id="control-var-1d", label="Variable", value="iota",
            data=[
                {"label": "Rotational Transform (iota)", "value": "iota"},
                {"label": "Pressure", "value": "pressure"},
                {"label": "Safety Factor (q)", "value": "q"},
            ]
        )
    ]
)

controls_2d = html.Div(
    id="wrapper-2d",
    style={"display": "none"}, # 默认隐藏
    children=[
        dmc.Stack([
            dmc.Text("Toroidal Angle (Phi)", size="sm", fw=500),
            dcc.Slider(
                id="control-phi", min=0, max=1, step=0.01, value=0, 
                marks={0: '0', 0.5: '0.5', 1: '1'}
            ),
            dmc.TextInput(id="control-surf-idx", label="Surface Index", value="-1")
        ], gap="xs")
    ]
)

controls_3d = html.Div(
    id="wrapper-3d",
    style={"display": "block"}, # 默认显示 (3d是默认模式)
    children=[
        dmc.Alert(
            "3D view rendered using JAX acceleration.",
            title="Performance",
            color="indigo",
            variant="light",
            icon=get_icon("tabler:bolt"),
            mt="md"
        )
    ]
)

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
    dmc.Divider(label="Visualization", labelPosition="center"),
    dmc.SegmentedControl(
        id="plot-mode",
        value="3d",
        fullWidth=True,
        data=[
            {"value": "summary", "label": "Sum"},
            {"value": "1d", "label": "1D"},
            {"value": "2d", "label": "2D"},
            {"value": "3d", "label": "3D"},
        ],
        color="indigo"
    ),
    
    # 这里放置所有的控件容器
    html.Div([
        controls_1d,
        controls_2d,
        controls_3d
    ]),

    dmc.Space(h="xl"),
    dmc.Button(
        "Download High-Res Plot",
        id="btn-download",
        variant="outline",
        fullWidth=True,
        leftSection=get_icon("tabler:photo-down")
    )
], gap="md", style={"height": "100%"})

app.layout = dmc.MantineProvider(
    theme={"fontFamily": "'Inter', sans-serif", "primaryColor": "indigo"},
    children=[
        dcc.Store(id='stored-filepath'),
        dmc.Grid([
            dmc.GridCol(
                dmc.Paper(sidebar_content, p="md", withBorder=True, style={"height": "100vh", "borderRadius": 0}),
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
                                    # 修复1: type改为 dcc.Loading 支持的值
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

# 修复2: 不再动态创建组件，而是切换 display 样式
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
    [Output('stored-filepath', 'data'), Output('filename-display', 'children')],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    prevent_initial_call=True
)
def handle_upload(contents, filename):
    if not contents: return dash.no_update
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    fd, path = tempfile.mkstemp(suffix=".nc")
    with os.fdopen(fd, 'wb') as f: f.write(decoded)
    return path, f"Active: {filename}"

@app.callback(
    [Output('main-graph', 'figure'), Output('stats-grid', 'children')],
    [Input('plot-mode', 'value'), Input('stored-filepath', 'data'),
     # 监听具体的控件值变化
     Input('control-var-1d', 'value'), Input('control-phi', 'value'), Input('control-surf-idx', 'value')],
    prevent_initial_call=True
)
def update_visualization(mode, filepath, var_1d, phi_val, surf_idx):
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
            idx = int(surf_idx) if surf_idx else -1
            r, z, s_val = vmec.compute_cross_section(phi_frac, idx)
            fig.add_trace(go.Scatter(x=r, y=z, mode='lines', fill='toself', fillcolor='rgba(76, 110, 245, 0.1)', line_color="#4c6ef5"))
            fig.update_layout(
                title=f"Flux Surface (s={s_val:.2f}) at phi={phi_frac:.2f}",
                xaxis_title="R [m]", yaxis_title="Z [m]",
                yaxis_scaleanchor="x", template="plotly_white"
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
        err_fig = go.Figure()
        err_fig.add_annotation(text=f"Error: {str(e)}", showarrow=False, font=dict(color="red", size=16))
        return err_fig, []

if __name__ == '__main__':
    app.run(debug=True)