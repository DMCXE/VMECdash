"""Compatibility entrypoint for the standalone Dash app.

The implementation now lives in ``vmecdash.dash_app.app`` so the package can
also expose a Dash-free backend for the VS Code extension.
"""

from vmecdash.dash_app.app import app, server


if __name__ == "__main__":
    app.run(debug=True)

