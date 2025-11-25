import numpy as np
import dash
from dash import Input, Output, State
import dash_mantine_components as dmc

from components.ui import build_select_data, unique_options
from vmec_jax import VMECJaxProcessor


def register_callbacks(app):
    @app.callback(
        [Output('ctrl-1d-var', 'children'),
         Output('ctrl-2d-var', 'data'),
         Output('ctrl-3d-var', 'data'),
         Output('ctrl-s-idx', 'max'), Output('ctrl-s-idx', 'marks'), Output('ctrl-s-idx', 'value'),
         Output('ctrl-s3d-idx', 'max'), Output('ctrl-s3d-idx', 'marks'), Output('ctrl-s3d-idx', 'value')],
        Input('vmec-meta', 'data'),
        prevent_initial_call=True,
    )
    def update_controls(meta):
        if not meta:
            return dash.no_update

        profiles_data = unique_options(build_select_data(meta.get('profiles', [])))
        profile_radios = dmc.Stack(
            [dmc.Radio(label=p['label'], value=p['value'], size="sm") for p in profiles_data],
            gap="xs",
        )

        fields = unique_options(build_select_data(meta.get('fields', [])))

        has_geometry = any(opt.get('value') == 'geometry' for opt in fields)
        fields_source = fields if has_geometry else [{"label": "Geometry Only (None)", "value": "geometry"}] + fields
        fields_2d = unique_options(fields_source)

        ns = meta.get('ns', 2)
        max_s = max(0, ns - 1)
        marks = {0: 'Axis', max_s: 'Edge'}

        return (
            profile_radios,
            fields_2d,
            fields,
            max_s, marks, max_s,
            max_s, marks, max_s,
        )

    @app.callback(
        Output('store-2d-data', 'data'),
        [Input('current-view', 'data'),
         Input('ctrl-2d-type', 'value'),
         Input('ctrl-2d-var', 'value'),
         Input('stored-filepath', 'data')],
        prevent_initial_call=True,
    )
    def precompute_2d_slices(view, type_2d, var_2d, filepath):
        if view != '2d' or type_2d != 'cross_section' or not filepath or var_2d == 'geometry':
            return None
        try:
            vmec = VMECJaxProcessor.from_file(filepath)
            field_lookup = {opt["value"]: opt.get("label", opt["value"]) for opt in vmec.available_fields()}
            field_label = field_lookup.get(var_2d, var_2d)
            frames = []
            steps = np.linspace(0, 1, 21)
            for phi_frac in steps:
                phi = phi_frac * 2 * np.pi / vmec.nfp
                r, z, val = vmec.get_cross_section_grid(phi, var_2d, res_grid=140)
                if r is None or z is None or val is None:
                    continue
                frames.append({
                    "r": r.tolist(),
                    "z": z.tolist(),
                    "val": np.where(np.isnan(val), None, val).tolist()
                })
            if not frames:
                return None
            return {"frames": frames, "var_key": var_2d, "var_label": field_label}
        except Exception as exc:
            print(f"Precompute error: {exc}")
            return dash.no_update

    @app.callback(
        [Output('ctrl-phi', 'disabled'),
         Output('group-geo-stride', 'style')],
        Input('ctrl-2d-type', 'value'),
        Input('ctrl-2d-var', 'value'),
        Input('store-2d-data', 'data'),
    )
    def toggle_phi_slider(type_2d, var_name, data_store):
        is_cross = (type_2d == 'cross_section')
        is_geo = (var_name == 'geometry')
        stride_style = {"display": "flex"} if (is_cross and is_geo) else {"display": "none"}
        disabled = not is_cross
        return disabled, stride_style

    @app.callback(
        Output('ctrl-phi', 'updatemode'),
        Input('ctrl-2d-type', 'value'),
        Input('ctrl-2d-var', 'value'),
    )
    def adjust_phi_updatemode(type_2d, var_name):
        if type_2d == 'cross_section' and var_name == 'geometry':
            return 'drag'
        return 'mouseup'

    @app.callback(
        Output('btn-reset-camera', 'disabled'),
        Input('current-view', 'data'),
    )
    def toggle_reset_button(view):
        return view != '3d'
