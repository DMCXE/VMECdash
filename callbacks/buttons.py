import dash
from dash import Input, Output, State, ctx


def register_callbacks(app):
    @app.callback(
        Output('ctrl-phi', 'value', allow_duplicate=True),
        Input('btn-phi-dec', 'n_clicks'),
        Input('btn-phi-inc', 'n_clicks'),
        State('ctrl-phi', 'value'),
        prevent_initial_call=True,
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
        prevent_initial_call=True,
    )
    def update_s_btn(n_dec, n_inc, val, max_val):
        ctx_id = ctx.triggered_id
        val = val or 0
        if ctx_id == 'btn-s-dec':
            return max(0, val - 1)
        elif ctx_id == 'btn-s-inc':
            return min(max_val, val + 1)
        return dash.no_update
