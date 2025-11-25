from . import buttons, clientside, controls, navigation, plots, theme, upload


def register_callbacks(app):
    upload.register_callbacks(app)
    controls.register_callbacks(app)
    navigation.register_callbacks(app)
    plots.register_callbacks(app)
    buttons.register_callbacks(app)
    theme.register_callbacks(app)
    clientside.register_callbacks(app)
