import base64
import os
import tempfile

import dash
from dash import Input, Output, State

from vmec_jax import VMECJaxProcessor


def register_callbacks(app):
    @app.callback(
        [Output('stored-filepath', 'data'),
         Output('header-filename', 'children'),
         Output('vmec-meta', 'data')],
        Input('upload-data', 'contents'),
        State('upload-data', 'filename'),
        prevent_initial_call=True,
    )
    def handle_upload(contents, filename):
        if not contents:
            return dash.no_update
        try:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            fd, path = tempfile.mkstemp(suffix=".nc")
            with os.fdopen(fd, 'wb') as f:
                f.write(decoded)

            vmec = VMECJaxProcessor.from_file(path)
            meta = {
                "ns": vmec.ns,
                "nfp": vmec.nfp,
                "profiles": vmec.available_profiles(),
                "fields": vmec.available_fields(),
                "summary_lines": vmec.get_summary_lines(),
            }
            return path, f"Active: {filename}", meta
        except Exception as e:
            return dash.no_update, f"Error: {str(e)}", dash.no_update

    @app.callback(
        Output('download-report', 'data'),
        Input('btn-export-report', 'n_clicks'),
        State('stored-filepath', 'data'),
        prevent_initial_call=True,
    )
    def export_report(n_clicks, filepath):
        if not n_clicks or not filepath:
            return dash.no_update
        try:
            vmec = VMECJaxProcessor.from_file(filepath)
            scalars = vmec.get_scalars()
            summary_lines = vmec.get_summary_lines()
            report_lines = ["VMEC Report", "==============", ""]
            for key, value in scalars.items():
                report_lines.append(f"{key}: {value}")
            report_lines.append("")
            report_lines.append("Summary")
            report_lines.extend(summary_lines)
            content = "\n".join(report_lines)
            return dict(content=content, filename="vmec_report.txt")
        except Exception as exc:
            print(f"Export error: {exc}")
            return dash.no_update
