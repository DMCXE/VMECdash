import dash
from dash import _dash_renderer
import dash_mantine_components as dmc

from callbacks import register_callbacks
from layout import layout

_d_ = _dash_renderer
_d_._set_react_version("18.2.0")

app = dash.Dash(
    __name__,
    title="VMEC ProViz",
    suppress_callback_exceptions=True,
    external_stylesheets=dmc.styles.ALL,
)
server = app.server

app.layout = layout
register_callbacks(app)


if __name__ == '__main__':
    app.run(debug=True)
