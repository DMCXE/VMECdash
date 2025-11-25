from dash import Input, Output, ctx
import dash


def register_callbacks(app):
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
        prevent_initial_call=True,
    )
    def update_view(n1, n2, n3, n4):
        ctx_id = ctx.triggered_id or "nav-overview"
        view_map = {
            "nav-overview": "overview",
            "nav-1d": "1d",
            "nav-2d": "2d",
            "nav-3d": "3d",
        }
        view = view_map.get(ctx_id, "overview")

        is_ov = (view == "overview")
        is_1d = (view == "1d")
        is_2d = (view == "2d")
        is_3d = (view == "3d")

        show = {"display": "block"}
        hide = {"display": "none"}

        return (
            view,
            is_ov, is_1d, is_2d, is_3d,
            show if is_ov else hide,
            show if is_1d else hide,
            show if is_2d else hide,
            show if is_3d else hide,
        )
