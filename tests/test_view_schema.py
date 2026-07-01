import json
import os
import sys
from pathlib import Path

import plotly.graph_objects as go

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-codex")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vmecdash import view_schema
from vmecdash.core import VMECJaxProcessor
from vmecdash.theme import build_theme
from vmecdash.vscode_backend import VmecDashBackend, figure_to_jsonable

EXAMPLE_WOUT = "example/wout_PO.nc"
EXPECTED_VIEWS = ["overview", "1d", "2d", "3d", "fieldline"]


def _vmec():
    return VMECJaxProcessor.from_file(EXAMPLE_WOUT)


def _field_map(vmec):
    return {opt["value"]: opt.get("label", opt["value"]) for opt in vmec.available_fields()}


def test_registry_covers_expected_views():
    assert list(view_schema.VIEWS) == EXPECTED_VIEWS


def test_render_view_smoke_for_every_view_with_default_controls():
    vmec = _vmec()
    theme = build_theme(True, 0)
    field_map = _field_map(vmec)
    for view_id in view_schema.VIEWS:
        fig = view_schema.render_view(view_id, vmec, {}, theme, field_map)
        assert isinstance(fig, go.Figure)


def test_build_ui_schema_lists_views_and_resolves_ns_tokens():
    vmec = _vmec()
    schema = view_schema.build_ui_schema(vmec)
    assert [v["id"] for v in schema["views"]] == EXPECTED_VIEWS

    twod = next(v for v in schema["views"] if v["id"] == "2d")
    s_idx = next(c for c in twod["controls"] if c["id"] == "sIdx")
    assert s_idx["max"] == vmec.ns - 1
    assert isinstance(s_idx["max"], int)
    assert s_idx["default"] == vmec.ns - 1

    fieldline = next(v for v in schema["views"] if v["id"] == "fieldline")
    fl_s = next(c for c in fieldline["controls"] if c["id"] == "fieldlineSIdx")
    assert fl_s["default"] == max(1, vmec.ns - 1)


def test_no_default_drift_between_schema_and_render_fallback():
    """The default shipped to the Webview must equal the render-time fallback (cv())."""
    vmec = _vmec()
    schema = view_schema.build_ui_schema(vmec)
    for view in schema["views"]:
        for control in view["controls"]:
            assert control["default"] == view_schema.cv({}, view["id"], control["id"], vmec)


def test_health_features_derived_from_registry():
    backend = VmecDashBackend()
    features = backend.health()["features"]
    assert features == list(view_schema.VIEWS) + ["exportReport"]


def test_render_view_serializes_without_nonfinite():
    vmec = _vmec()
    theme = build_theme(True, 0)
    field_map = _field_map(vmec)
    # lambda cross-section carries NaN/Inf at the axis; the wire writer forbids bare NaN/Inf.
    fig = view_schema.render_view("2d", vmec, {"type2d": "cross_section", "var2d": "lambda"}, theme, field_map)
    payload = figure_to_jsonable(fig)
    json.dumps(payload, allow_nan=False)  # must not raise


def test_single_trace_uses_no_webgl_trace():
    """Single Trace must render with SVG scatter, not Scattergl (webview WebGL-context cap)."""
    from vmecdash.renderers.fieldline import render_single_trace

    vmec = _vmec()
    theme = build_theme(True, 0)
    fig = render_single_trace(vmec, max(1, vmec.ns - 1), 20, 0.0, 64, theme)
    types = [trace.type for trace in fig.data]
    assert "scattergl" not in types
    assert "scatter" in types
