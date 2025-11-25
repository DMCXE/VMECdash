from dash import Input, Output


def register_callbacks(app):
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
