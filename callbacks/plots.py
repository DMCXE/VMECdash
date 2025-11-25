import math

import dash
from dash import Input, Output, State, ctx
import dash_mantine_components as dmc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

from components.ui import (
    build_geometry_cross_section_figure,
    create_detail_card,
    create_hero_stat,
    create_stat_card,
)
from vmec_jax import VMECJaxProcessor


def register_callbacks(app):
    @app.callback(
        [Output('main-graph', 'figure', allow_duplicate=True),
         Output('stats-container', 'children')],
        [Input('current-view', 'data'),
         Input('stored-filepath', 'data'),
         Input('ctrl-1d-var', 'value'),
         Input('ctrl-2d-type', 'value'),
         Input('ctrl-2d-var', 'value'),
         Input('ctrl-s-idx', 'value'),
         Input('ctrl-3d-var', 'value'),
         Input('ctrl-s3d-idx', 'value'),
         Input('ctrl-3d-bg', 'checked'),
         Input('btn-reset-camera', 'n_clicks'),
         Input('toggle-theme', 'checked'),
         Input('ctrl-geo-stride', 'value')],
        State('vmec-meta', 'data'),
        State('ctrl-phi', 'value'),
        prevent_initial_call=True,
    )
    def update_visualization(view, filepath, var_1d, type_2d, var_2d, s_2d, var_3d, s_3d, bg_3d, reset_clicks, dark_mode, geo_count, meta, phi_state):
        triggered = ctx.triggered_id
        if (
            filepath
            and view == '2d'
            and type_2d == 'cross_section'
            and var_2d != 'geometry'
            and triggered == 'ctrl-phi'
        ):
            return dash.no_update, dash.no_update

        dark_mode = True if dark_mode is None else bool(dark_mode)
        fig_template = 'plotly_dark' if dark_mode else 'plotly_white'
        paper_bg = 'rgba(0,0,0,0)' if dark_mode else 'white'
        plot_bg = 'rgba(0,0,0,0)'
        text_color = '#adb5bd' if dark_mode else '#495057'
        empty_fig = go.Figure()
        empty_fig.update_layout(
            template=fig_template,
            paper_bgcolor=paper_bg,
            plot_bgcolor=plot_bg,
            xaxis={'visible': False},
            yaxis={'visible': False}
        )

        if not filepath:
            empty_fig.add_annotation(text="Please load a .nc file", showarrow=False, font=dict(size=20, color=text_color))
            return empty_fig, []

        try:
            vmec = VMECJaxProcessor.from_file(filepath)
            scalars = vmec.get_scalars()
            field_map = {opt['value']: opt.get('label', opt['value']) for opt in vmec.available_fields()}
            summary_lines = None
            if meta:
                summary_lines = meta.get('summary_lines')
            if not summary_lines and view == 'overview':
                summary_lines = vmec.get_summary_lines()
            reset_seed = reset_clicks or 0

            beta = scalars.get('beta_total', 0.0)
            vol = scalars.get('volume', 0.0)
            curr = scalars.get('ctor', 0.0)

            rmaj = scalars.get('Rmajor')
            amin = scalars.get('Aminor')
            if rmaj is not None and amin is not None and amin != 0:
                ar_str = f"{rmaj/amin:.2f}"
            else:
                ar_str = "N/A"

            def safe_fmt(value, pattern="{:.2f}", fallback="--"):
                try:
                    if value is None or (isinstance(value, (float, int)) and math.isnan(value)):
                        return fallback
                    return pattern.format(value)
                except Exception:
                    return fallback

            base_cards = dmc.SimpleGrid(cols=4, children=[
                create_stat_card('Beta Total', f"{beta*100:.2f}%", 'mdi:percent', 'red'),
                create_stat_card('Volume', f"{vol:.2f} m³", 'mdi:cube-outline', 'blue'),
                create_stat_card('Aspect Ratio', ar_str, 'mdi:ratio', 'orange'),
                create_stat_card('Toroidal Current', f"{curr:.2f} A", 'mdi:current-ac', 'teal'),
            ])

            def render_summary_panels(lines):
                if not lines:
                    return dmc.Center(dmc.Text('Summary unavailable', c='dimmed', size='sm'))
                panels = []
                for line in lines:
                    fragments = [frag.strip() for frag in line.replace('|', ',').split(',') if frag.strip()]
                    rows = []
                    for frag in fragments:
                        if '=' in frag:
                            key, value = frag.split('=', 1)
                            rows.append(
                                dmc.Group([
                                    dmc.Kbd(key.strip(), style={"minWidth": 70}),
                                    dmc.Text(value.strip(), fw=600)
                                ], gap="xs", align="center")
                            )
                        else:
                            rows.append(dmc.Text(frag, fw=600))
                    panels.append(
                        dmc.Paper(
                            dmc.Stack(rows, gap=4),
                            withBorder=True,
                            radius="sm",
                            shadow="xs",
                            p="sm"
                        )
                    )
                return dmc.SimpleGrid(cols=2, spacing="sm", children=panels)

            stats_ui = base_cards
            if view == 'overview':
                hero_cards = dmc.SimpleGrid(
                    cols=3,
                    spacing='md',
                    children=[
                        create_hero_stat('Total Beta', f"{beta*100:.2f}", '%', 'mdi:percent', 'cyan', 'Global beta'),
                        create_hero_stat('Toroidal Current', f"{curr:.2f}", 'MA', 'mdi:current-ac', 'indigo', 'Plasma current'),
                        create_hero_stat('Volume', f"{vol:.2f}", 'm³', 'mdi:cube-outline', 'teal', 'Enclosed volume'),
                    ]
                )

                def fmt(val, f="{:.3f}"):
                    return safe_fmt(val, f)

                geo_items = [
                    ("Major Radius (R0)", fmt(rmaj), "m"),
                    ("Minor Radius (a)", fmt(amin), "m"),
                    ("Aspect Ratio", fmt(scalars.get('aspect'), "{:.2f}"), ""),
                    ("Volume", fmt(vol), "m³"),
                ]

                mag_items = [
                    ("Magnetic Field (B0)", fmt(scalars.get('b0')), "T"),
                    ("Toroidal Flux (RB_tor)", fmt(scalars.get('rbtor')), "T·m"),
                    ("Iota (Axis)", fmt(scalars.get('iota_axis')), ""),
                    ("Iota (Edge)", fmt(scalars.get('iota_edge')), ""),
                    ("Safety Factor (q0)", fmt(scalars.get('q_axis')), ""),
                    ("Safety Factor (qa)", fmt(scalars.get('q_edge')), ""),
                    ("Shear (Edge)", fmt(scalars.get('shear_edge')), ""),
                ]

                mgrid = vmec.ds.attrs.get('mgrid_file', 'none')
                if isinstance(mgrid, bytes):
                    mgrid = mgrid.decode('utf-8')
                if mgrid == 'none':
                    lfreeb = vmec.ds.attrs.get('lfreeb__logical__', 0)
                    if lfreeb:
                        bound_type = "Free Boundary (No mgrid)"
                    else:
                        bound_type = "Fixed Boundary"
                else:
                    bound_type = f"Free Boundary ({mgrid})"

                plasma_items = [
                    ("Beta Poloidal", fmt(scalars.get('betapol')), ""),
                    ("Beta Toroidal", fmt(scalars.get('betator')), ""),
                    ("Pressure (Axis)", fmt(scalars.get('pressure_axis'), "{:.2e}"), "Pa"),
                    ("Pressure (Edge)", fmt(scalars.get('pressure_edge'), "{:.2e}"), "Pa"),
                    ("Boundary Type", bound_type, ""),
                ]

                details_grid = dmc.SimpleGrid(
                    cols=3,
                    spacing="md",
                    children=[
                        create_detail_card("Geometry", "mdi:axis-arrow", geo_items),
                        create_detail_card("Magnetics", "mdi:magnet", mag_items),
                        create_detail_card("Plasma & Boundary", "mdi:fire", plasma_items),
                    ]
                )

                stats_ui = dmc.Stack([hero_cards, details_grid], gap='md')

                fig = make_subplots(
                    rows=2, cols=3,
                    subplot_titles=(
                        'Rotational Transform (iota)', 'Safety Factor (q)', 'Pressure Profile',
                        'dP/ds', 'Volume Enclosed (Vp)', 'Flux Avg <B·B>'
                    ),
                    vertical_spacing=0.15, horizontal_spacing=0.08
                )

                def get_data_safe(key):
                    try:
                        return vmec.get_1d_data(key)
                    except Exception:
                        return [], []

                s_i, iota = get_data_safe('iotaf')
                s_q, q_prof = get_data_safe('q')
                s_p, pres = get_data_safe('presf')
                s_dp, dpds = get_data_safe('dpds')
                s_vp, vp = get_data_safe('vp')
                s_bb, bdotb = get_data_safe('bdotb')

                fig.add_trace(go.Scatter(x=s_i, y=iota, name='iota', line=dict(color='#22b8cf', width=3)), row=1, col=1)
                fig.add_trace(go.Scatter(x=s_q, y=q_prof, name='q', line=dict(color='#fd7e14', width=3)), row=1, col=2)
                fig.add_trace(go.Scatter(x=s_p, y=pres, name='pres', line=dict(color='#fa5252', width=3)), row=1, col=3)
                fig.add_trace(go.Scatter(x=s_dp, y=dpds, name='dP/ds', line=dict(color='#12b886', width=3)), row=2, col=1)
                fig.add_trace(go.Scatter(x=s_vp, y=vp, name='Vp', line=dict(color='#7950f2', width=3)), row=2, col=2)
                fig.add_trace(go.Scatter(x=s_bb, y=bdotb, name='<B.B>', line=dict(color='#0ca678', width=3)), row=2, col=3)

                fig.update_layout(
                    title_text='Equilibrium Summary',
                    template=fig_template,
                    paper_bgcolor=paper_bg,
                    plot_bgcolor=plot_bg,
                    height=800,
                    showlegend=False,
                    uirevision=f"overview-{reset_seed}"
                )
                return fig, stats_ui

            if view == '1d':
                var = var_1d or 'iotaf'
                s, y = vmec.get_1d_data(var)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=s, y=y, mode='lines', line=dict(color='#3bc9db', width=4)))
                fig.update_layout(
                    title=f'Profile: {var}',
                    xaxis_title='s',
                    yaxis_title=field_map.get(var, var),
                    template=fig_template,
                    paper_bgcolor=paper_bg,
                    plot_bgcolor=plot_bg,
                    uirevision=f"1d-{reset_seed}"
                )
                return fig, stats_ui

            if view == '2d':
                phi_val = phi_state if phi_state is not None else 0.0
                phi_scale = 2 * np.pi / max(vmec.nfp, 1)
                phi_angle = phi_val * phi_scale
                s_idx = int(s_2d) if s_2d is not None else vmec.ns - 1
                field_label = field_map.get(var_2d, var_2d)

                if type_2d == 'cross_section':
                    if var_2d == 'geometry':
                        fig = build_geometry_cross_section_figure(
                            vmec, phi_angle, s_idx, geo_count, dark_mode,
                            fig_template, paper_bg, plot_bg, reset_seed
                        )
                        return fig, stats_ui

                    return dash.no_update, stats_ui

                fig = go.Figure()

                if type_2d == 'flux_surface':
                    theta, zeta, val = vmec.get_flux_surface_data(s_idx, var_2d, res_u=128, res_v=128)
                    if theta is not None:
                        fig.add_trace(
                            go.Contour(
                                x=zeta,
                                y=theta,
                                z=val,
                                colorscale='Viridis',
                                colorbar=dict(title=field_label),
                                contours=dict(coloring='fill')
                            )
                        )
                        fig.update_layout(
                            title=f"{field_label} on Flux Surface s={s_idx/(vmec.ns-1):.2f}",
                            xaxis_title='Zeta (toroidal) [rad]',
                            yaxis_title='Theta (poloidal) [rad]',
                            xaxis=dict(range=[0, 2*np.pi/vmec.nfp]),
                            yaxis=dict(range=[0, 2*np.pi])
                        )
                        fig.update_layout(
                            template=fig_template,
                            paper_bgcolor=paper_bg,
                            plot_bgcolor=plot_bg,
                            uirevision=f"2d-{reset_seed}"
                        )
                        return fig, stats_ui

                return dash.no_update, stats_ui

            if view == '3d':
                s_val = int(s_3d) if s_3d is not None else vmec.ns - 1
                v_name = var_3d or 'modB'
                coord_free = True if bg_3d is None else bool(bg_3d)
                field_label = field_map.get(v_name, v_name)
                fig = go.Figure()

                x, y, z, val = vmec.compute_3d_surface(s_idx=s_val, var_name=v_name, resolution=100)

                if v_name == 'geometry':
                    surf_color = np.full_like(z, 0.5)
                    cscale = 'Greys'
                else:
                    surf_color = val
                    cscale = 'Jet'

                fig.add_trace(go.Surface(
                    x=x, y=y, z=z, surfacecolor=surf_color,
                    colorscale=cscale,
                    colorbar=dict(title=field_label, len=0.6) if v_name != 'geometry' else None
                ))

                axis_color = '#dee2e6' if dark_mode else '#495057'
                grid_color = '#5c677d' if dark_mode else '#ced4da'
                axis_style = (
                    dict(visible=False)
                    if coord_free
                    else dict(
                        visible=True,
                        backgroundcolor='rgba(0,0,0,0)',
                        gridcolor=grid_color,
                        zerolinecolor=grid_color,
                        color=axis_color,
                        title=dict(font=dict(color=axis_color)),
                        tickfont=dict(color=axis_color)
                    )
                )
                title_txt = f"Surface s={s_val/(vmec.ns-1):.2f}"
                if v_name != 'geometry':
                    title_txt += f" colored by {field_label}"
                fig.update_layout(
                    title=title_txt,
                    template=fig_template,
                    paper_bgcolor=paper_bg,
                    scene=dict(
                        bgcolor=plot_bg,
                        xaxis=axis_style,
                        yaxis=axis_style,
                        zaxis=axis_style,
                        aspectmode='data'
                    ),
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                return fig, stats_ui

        except Exception as e:
            print(f"Detailed Error: {e}")
            import traceback
            traceback.print_exc()
            err_fig = go.Figure()
            err_fig.add_annotation(text=f"Error: {str(e)}", showarrow=False, font=dict(color="red", size=16))
            return err_fig, []

        return empty_fig, []

    @app.callback(
        Output('main-graph', 'figure', allow_duplicate=True),
        Input('ctrl-phi', 'value'),
        State('current-view', 'data'),
        State('ctrl-2d-type', 'value'),
        State('ctrl-2d-var', 'value'),
        State('ctrl-s-idx', 'value'),
        State('ctrl-geo-stride', 'value'),
        State('stored-filepath', 'data'),
        State('toggle-theme', 'checked'),
        prevent_initial_call=True,
    )
    def update_geometry_phi(phi_val, view, type_2d, var_2d, s_idx, geo_count, filepath, dark_mode):
        if not (filepath and view == '2d' and type_2d == 'cross_section' and var_2d == 'geometry'):
            return dash.no_update
        phi_val = phi_val or 0.0
        try:
            vmec = VMECJaxProcessor.from_file(filepath)
            phi_scale = 2 * np.pi / max(vmec.nfp, 1)
            phi_angle = phi_val * phi_scale
            s_idx = int(s_idx) if s_idx is not None else vmec.ns - 1
            dark_mode = True if dark_mode is None else bool(dark_mode)
            fig_template = 'plotly_dark' if dark_mode else 'plotly_white'
            paper_bg = 'rgba(0,0,0,0)' if dark_mode else 'white'
            plot_bg = 'rgba(0,0,0,0)'
            fig = build_geometry_cross_section_figure(
                vmec, phi_angle, s_idx, geo_count, dark_mode,
                fig_template, paper_bg, plot_bg, reset_seed=0
            )
            return fig
        except Exception as exc:
            print(f"Geometry phi update error: {exc}")
            return dash.no_update
