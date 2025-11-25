from dash import Input, Output, State


def register_callbacks(app):
    app.clientside_callback(
        """
        function(phi_val, data_store, view, type_2d, var_2d) {
            if (phi_val === undefined || phi_val === null) {
                return window.dash_clientside.no_update;
            }
            if (view !== '2d' || type_2d !== 'cross_section' || var_2d === 'geometry') {
                return window.dash_clientside.no_update;
            }
            if (!data_store || !data_store.frames || !data_store.frames.length) {
                return window.dash_clientside.no_update;
            }
            if (!data_store.var_key || data_store.var_key !== var_2d) {
                return window.dash_clientside.no_update;
            }
            try {
                var total = data_store.frames.length;
                var idx = Math.round(phi_val * (total - 1));
                if (idx < 0) idx = 0;
                if (idx >= total) idx = total - 1;
                var frame = data_store.frames[idx];
                if (!frame || !frame.r || !frame.z || !frame.val) {
                    return window.dash_clientside.no_update;
                }
                var displayLabel = data_store.var_label || data_store.var_key || 'Field';
                var fig_data = {
                    type: 'contour',
                    x: frame.r,
                    y: frame.z,
                    z: frame.val,
                    colorscale: 'RdBu',
                    colorbar: {title: displayLabel},
                    contours: {coloring: 'heatmap'},
                    ncontours: 50,
                    line: {width: 0}
                };
                var layout = {
                    title: displayLabel + ' on Cross-Section at φ=' + phi_val.toFixed(2),
                    xaxis: {title: 'R [m]'},
                    yaxis: {title: 'Z [m]', scaleanchor: 'x', scaleratio: 1},
                    template: 'plotly_dark'
                };
                return {data: [fig_data], layout: layout};
            } catch (e) {
                console.error('Clientside callback error:', e);
                return window.dash_clientside.no_update;
            }
        }
        """,
        Output('main-graph', 'figure', allow_duplicate=True),
        Input('ctrl-phi', 'value'),
        Input('store-2d-data', 'data'),
        State('current-view', 'data'),
        State('ctrl-2d-type', 'value'),
        State('ctrl-2d-var', 'value'),
        prevent_initial_call=True,
    )

    app.clientside_callback(
        """
        function(var_name, view, type_2d) {
            if (view === '2d' && type_2d === 'cross_section' && var_name !== 'geometry') {
                return [
                    false,
                    'Calculating ' + var_name + '...',
                    'blue',
                    {'height': 'calc(100vh - 204px)', 'opacity': 0.3, 'transition': 'opacity 0.5s'}
                ];
            }
            return [
                true,
                'Ready',
                'gray',
                {'height': 'calc(100vh - 204px)', 'opacity': 1}
            ];
        }
        """,
        Output('status-alert', 'hide'),
        Output('status-alert', 'children'),
        Output('status-alert', 'color'),
        Output('main-graph', 'style'),
        Input('ctrl-2d-var', 'value'),
        Input('current-view', 'data'),
        Input('ctrl-2d-type', 'value'),
        prevent_initial_call=True,
    )

    app.clientside_callback(
        """
        function(data, var_name) {
            if (data && data.var_key === var_name) {
                return [
                    false,
                    'Rendered: ' + (data.var_label || var_name),
                    'green',
                    {'height': 'calc(100vh - 204px)', 'opacity': 1, 'transition': 'opacity 0.5s'}
                ];
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output('status-alert', 'hide', allow_duplicate=True),
        Output('status-alert', 'children', allow_duplicate=True),
        Output('status-alert', 'color', allow_duplicate=True),
        Output('main-graph', 'style', allow_duplicate=True),
        Input('store-2d-data', 'data'),
        State('ctrl-2d-var', 'value'),
        prevent_initial_call=True,
    )

    app.clientside_callback(
        """
        function(n_clicks) {
            if (!n_clicks) {
                return window.dash_clientside.no_update;
            }
            var graphContainer = document.getElementById('main-graph');
            if (!graphContainer) {
                return window.dash_clientside.no_update;
            }
            var plot = graphContainer.querySelector('.js-plotly-plot');
            if (!plot) {
                return window.dash_clientside.no_update;
            }
            Plotly.downloadImage(plot, {
                format: 'png',
                width: 1400,
                height: 900,
                filename: 'vmec_viz'
            });
            return window.dash_clientside.no_update;
        }
        """,
        Output('btn-download', 'id'),
        Input('btn-download', 'n_clicks'),
        prevent_initial_call=True,
    )
