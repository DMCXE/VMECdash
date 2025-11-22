import dash
from dash import dcc, html, _dash_renderer
import dash_mantine_components as dmc
from dash_iconify import DashIconify
import plotly.graph_objects as go
import numpy as np

# 设置 React 版本 (DMC 0.14+ 需要)
_dash_renderer._set_react_version("18.2.0")

app = dash.Dash(__name__)

# --- 辅助函数：生成图标 ---
def get_icon(icon, size=18):
    return DashIconify(icon=icon, width=size)

# --- 辅助函数：生成模拟的 3D 等离子体图 ---
def get_dummy_3d_plot():
    # 生成一个环形面包圈 (Torus) 模拟等离子体表面
    theta = np.linspace(0, 2.*np.pi, 50)
    phi = np.linspace(0, 2.*np.pi, 50)
    theta, phi = np.meshgrid(theta, phi)
    R, r = 3, 1
    x = (R + r*np.cos(theta)) * np.cos(phi)
    y = (R + r*np.cos(theta)) * np.sin(phi)
    z = r * np.sin(theta)
    
    fig = go.Figure(data=[go.Surface(x=x, y=y, z=z, colorscale="Viridis", showscale=False)])
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False))
    )
    return fig

# --- 辅助函数：侧边栏导航项 ---
def nav_link(label, icon, active=False, description=None):
    return dmc.NavLink(
        label=label,
        leftSection=get_icon(icon),
        active=active,
        variant="light",
        description=description,
        style={"borderRadius": "8px", "marginBottom": "4px"}
    )

# --- 布局设计 ---
app.layout = dmc.MantineProvider(
    forceColorScheme="dark", # 强制暗色模式，显得更专业
    theme={
        "primaryColor": "cyan", # 科技蓝青色
        "fontFamily": "'Inter', system-ui, sans-serif",
        "defaultRadius": "md"
    },
    children=dmc.AppShell(
        header={"height": 60},
        navbar={"width": 260, "breakpoint": "sm"},
        aside={"width": 300, "breakpoint": "md"},
        padding="md",
        children=[
            # 1. 顶部导航栏 (Header)
            dmc.AppShellHeader(
                px="md",
                children=dmc.Group(
                    justify="space-between",
                    h="100%",
                    children=[
                        dmc.Group([
                            dmc.ThemeIcon(get_icon("mdi:atom-variant"), variant="gradient", gradient={"from": "cyan", "to": "indigo"}, size="lg", radius="xl"),
                            dmc.Text("VMEC ProViz", fw=700, size="xl", style={"letterSpacing": "-0.5px"}),
                            dmc.Badge("v2.0 Beta", color="gray", variant="outline")
                        ]),
                        dmc.Group([
                            dmc.Text("wout_w7x_high_beta.nc", size="sm", c="dimmed", fw=500),
                            dmc.ActionIcon(get_icon("mdi:github"), variant="subtle", color="gray"),
                            dmc.ActionIcon(get_icon("mdi:bell"), variant="subtle", color="gray"),
                            dmc.Avatar(src=None, radius="xl", color="cyan", children="SCI")
                        ])
                    ]
                )
            ),

            # 2. 左侧导航栏 (Navbar) - 物理语义导航
            dmc.AppShellNavbar(
                p="md",
                children=dmc.ScrollArea([
                    dmc.Text("DATA SOURCE", size="xs", fw=700, c="dimmed", mb="sm"),
                    dmc.Button("Upload .nc File", leftSection=get_icon("mdi:upload"), variant="outline", fullWidth=True, mb="xl"),

                    dmc.Text("PHYSICS MODULES", size="xs", fw=700, c="dimmed", mb="sm"),
                    nav_link("Dashboard", "mdi:view-dashboard-outline", active=False),
                    nav_link("Global Scalars", "mdi:sigma", description="Beta, Volume, Energy"),
                    
                    dmc.Divider(my="sm", label="Topology", labelPosition="center"),
                    nav_link("1D Profiles", "mdi:chart-bell-curve", description="Iota, Pressure, Current"),
                    nav_link("2D Poincaré", "mdi:chart-scatter-plot", description="Flux Surfaces (R, Z)"),
                    
                    dmc.Divider(my="sm", label="Geometry", labelPosition="center"),
                    nav_link("3D Geometry", "mdi:shape-outline", active=True, description="Boundary, Mod-B"),
                    nav_link("Fourier Modes", "mdi:music-accidental-sharp", description="Rmn, Zmn Spectrum"),
                ])
            ),

            # 3. 右侧控制栏 (Aside) - 上下文相关控制
            dmc.AppShellAside(
                p="md",
                children=[
                    dmc.Stack([
                        dmc.Group([
                            get_icon("mdi:tune", 20),
                            dmc.Text("View Controls", fw=700)
                        ], mb="md"),

                        # 控制组 1: 渲染变量
                        dmc.Paper(p="sm", withBorder=True, children=[
                            dmc.Text("Surface Color", size="xs", fw=500, mb=5),
                            dmc.Select(
                                value="modB",
                                data=[
                                    {"label": "|B| (Mod B)", "value": "modB"},
                                    {"label": "Jacobian (sqrt g)", "value": "jac"},
                                    {"label": "Normal Curvature", "value": "kappa"},
                                ],
                                mb="md"
                            ),
                            dmc.Text("Colormap", size="xs", fw=500, mb=5),
                            dmc.SegmentedControl(
                                value="plasma",
                                fullWidth=True,
                                data=[
                                    {"label": "Plasma", "value": "plasma"},
                                    {"label": "Viridis", "value": "viridis"},
                                    {"label": "Jet", "value": "jet"},
                                ],
                                size="xs"
                            )
                        ]),

                        # 控制组 2: 切片与裁剪
                        dmc.Paper(p="sm", withBorder=True, mt="sm", children=[
                            dmc.Text("Clipping & Slicing", size="sm", fw=600, mb="xs"),
                            
                            dmc.Text("Toroidal Cut (Phi)", size="xs", c="dimmed"),
                            dmc.Slider(min=0, max=360, step=10, value=90, color="cyan", mb="sm", marks=[
                                {"value": 0, "label": "0°"},
                                {"value": 360, "label": "360°"},
                            ]),

                            dmc.Text("Flux Surface (s)", size="xs", c="dimmed"),
                            dmc.Group([
                                dmc.Slider(min=0, max=1, step=0.1, value=1, color="cyan", style={"flex": 1}),
                                dmc.Badge("s=1.0", variant="outline")
                            ])
                        ]),

                        # 控制组 3: 导出
                        dmc.Button("High-Res Screenshot", variant="light", color="gray", fullWidth=True, mt="xl", leftSection=get_icon("mdi:camera")),
                    ])
                ]
            ),

            # 4. 主内容区域 (Main)
            dmc.AppShellMain(
                children=[
                    # 顶部统计卡片行
                    dmc.SimpleGrid(cols=4, mb="md", children=[
                        dmc.Paper(p="md", withBorder=True, shadow="xs", children=[
                            dmc.Text("Beta Total", c="dimmed", size="xs", fw=700),
                            dmc.Group([
                                dmc.Text("4.25%", fw=700, size="xl", c="cyan"),
                                get_icon("mdi:arrow-up-bold", 16)
                            ])
                        ]),
                        dmc.Paper(p="md", withBorder=True, shadow="xs", children=[
                            dmc.Text("Stored Energy", c="dimmed", size="xs", fw=700),
                            dmc.Text("1.2 MJ", fw=700, size="xl")
                        ]),
                        dmc.Paper(p="md", withBorder=True, shadow="xs", children=[
                            dmc.Text("Magnetic Axis (R)", c="dimmed", size="xs", fw=700),
                            dmc.Text("5.52 m", fw=700, size="xl")
                        ]),
                        dmc.Paper(p="md", withBorder=True, shadow="xs", children=[
                            dmc.Text("Convergence", c="dimmed", size="xs", fw=700),
                            dmc.Badge("1.2e-11", color="green", variant="light")
                        ]),
                    ]),

                    # 主图表区域
                    dmc.Card(
                        h="calc(100vh - 180px)", # 动态计算高度
                        withBorder=True,
                        p=0,
                        style={"overflow": "hidden", "position": "relative"},
                        children=[
                            # 图表左上角的浮动标签
                            html.Div(
                                dmc.Badge("Interactive 3D View", color="dark", radius="sm"),
                                style={"position": "absolute", "top": 10, "left": 10, "zIndex": 10}
                            ),
                            dcc.Graph(
                                figure=get_dummy_3d_plot(),
                                style={"height": "100%", "width": "100%"},
                                config={"displayModeBar": True, "displaylogo": False}
                            )
                        ]
                    )
                ]
            )
        ]
    )
)

if __name__ == "__main__":
    app.run(debug=True)