"""Colour policy, the shared figure template, and the LPK view.

The policy these cover is the one thing a reader cannot check by eye: a signed field on a
sequential colour scale looks perfectly plausible, it just hides the zero crossing.
"""

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-codex")

from vmecdash import view_schema  # noqa: E402
from vmecdash.core import VMECJaxProcessor  # noqa: E402
from vmecdash.renderers import two_d  # noqa: E402
from vmecdash.theme import (  # noqa: E402
    CONTOUR_LEVELS,
    DIVERGING_SCALE,
    SEQUENTIAL_SCALE,
    build_theme,
    resolve_colorscale,
)

EXAMPLE_WOUT = str(ROOT / "example" / "wout_PO.nc")

#: Verified against the example equilibrium: these fields change sign across a flux
#: surface, so they need a diverging scale centred on zero. B_phi stays single-sign
#: (the toroidal field does not reverse) and the force residual is a magnitude, so both
#: correctly stay sequential.
SIGNED_FIELDS = {"lambda", "B_s", "B_u", "B^u", "j^u", "j^v", "B_R", "B_Z", "j_para"}


def _vmec():
    return VMECJaxProcessor.from_file(EXAMPLE_WOUT)


def _field_map(vmec):
    return {opt["value"]: opt.get("label", opt["value"]) for opt in vmec.available_fields()}


# ------------------------------------------------------------------- colour policy


def test_signed_fields_get_a_diverging_scale_centred_on_zero():
    vmec = _vmec()
    for option in vmec.available_fields():
        key = option["value"]
        if key == "geometry":
            continue
        _, _, values = vmec.get_flux_surface_data(vmec.ns - 1, key, res_u=64, res_v=64)
        cs = resolve_colorscale(values)
        if key in SIGNED_FIELDS:
            assert cs.scale == DIVERGING_SCALE, key
            assert cs.zmin == -cs.zmax, key
            assert cs.reversescale is True, key  # blue negative, red positive
        else:
            assert cs.scale == SEQUENTIAL_SCALE, key
            assert cs.zmin < cs.zmax, key


def test_override_changes_the_scale_but_not_where_zero_sits():
    cs = resolve_colorscale(np.array([-3.0, 1.0]), "Cividis")
    assert cs.scale == "Cividis"
    assert cs.zmin == -cs.zmax  # still symmetric: the override must not move zero
    assert "forced" in cs.reason


def test_degenerate_and_empty_inputs_do_not_collapse_the_scale():
    flat = resolve_colorscale(np.array([1.5, 1.5]))
    assert flat.zmin < flat.zmax  # a zero-width range would make the colorbar unusable
    empty = resolve_colorscale(np.array([np.nan, np.inf]))
    assert empty.zmin < empty.zmax


def test_no_renderer_still_uses_jet():
    """3-D used to hardcode Jet, which is not perceptually uniform or colour-blind safe."""
    vmec = _vmec()
    theme = build_theme(True, 0)
    field_map = _field_map(vmec)
    for view, controls in (
        ("2d", {"type2d": "cross_section", "var2d": "modB", "sIdx": vmec.ns - 1}),
        ("2d", {"type2d": "flux_surface", "var2d": "j^u", "sIdx": vmec.ns - 1}),
        ("3d", {"var3d": "modB", "s3dIdx": vmec.ns - 1}),
        ("fieldline", {"fieldlineType": "2d_modB", "fieldlineSIdx": vmec.ns - 1, "fieldlineRes": "64"}),
    ):
        blob = view_schema.render_view(view, vmec, controls, theme, field_map).to_json().lower()
        assert '"jet"' not in blob, (view, controls)


def test_three_d_signed_field_is_diverging_and_symmetric():
    vmec = _vmec()
    fig = view_schema.render_view(
        "3d", vmec, {"var3d": "j^u", "s3dIdx": vmec.ns - 1}, build_theme(True, 0), _field_map(vmec)
    )
    surface = fig.data[0]
    assert surface.cmin == -surface.cmax


def test_template_is_composed_not_replaced():
    assert build_theme(True).fig_template == "plotly_dark+vmecdash_dark"
    assert build_theme(False).fig_template == "plotly_white+vmecdash_light"


def test_contour_level_count_is_shared():
    vmec = _vmec()
    theme = build_theme(True, 0)
    carpet = view_schema.render_view(
        "2d", vmec, {"type2d": "cross_section", "var2d": "modB"}, theme, _field_map(vmec)
    )
    contour = next(t for t in carpet.data if t.type == "contourcarpet")
    span = contour.contours.end - contour.contours.start
    assert np.isclose(span / contour.contours.size, CONTOUR_LEVELS)


# ----------------------------------------------------------------- availability


class _Axisymmetric:
    """Minimal stand-in: build_ui_schema only reads ns/nfp/ntor/lasym."""

    ns, nfp, ntor, lasym = 50, 1, 0, False


def test_lpk_is_offered_only_for_non_axisymmetric_equilibria():
    """It is a switch on the cross-section, dropped whole when every cut is identical."""

    def control_ids(vmec):
        schema = view_schema.build_ui_schema(vmec)
        two_d_view = next(v for v in schema["views"] if v["id"] == "2d")
        return [c["id"] for c in two_d_view["controls"]]

    assert "lpkMode" in control_ids(_vmec())
    assert "lpkMode" not in control_ids(_Axisymmetric())
    # And it is no longer masquerading as a third plot type.
    schema = view_schema.build_ui_schema(_vmec())
    two_d_view = next(v for v in schema["views"] if v["id"] == "2d")
    types = next(c for c in two_d_view["controls"] if c["id"] == "type2d")
    assert [o["value"] for o in types["options"]] == ["cross_section", "flux_surface"]


def test_availability_predicate_is_stripped_from_the_shipped_schema():
    schema = view_schema.build_ui_schema(_vmec())
    for view in schema["views"]:
        for control in view["controls"]:
            for option in control["options"] or []:
                assert "requires" not in option


def test_smooth_shading_is_offered_only_where_plotly_supports_it():
    """Contourcarpet has no "heatmap" coloring, so the R-Z carpet cannot smooth-shade."""
    schema = view_schema.build_ui_schema(_vmec())
    two_d_view = next(v for v in schema["views"] if v["id"] == "2d")
    style = next(c for c in two_d_view["controls"] if c["id"] == "contourStyle")
    smooth = next(opt for opt in style["options"] if opt["value"] == "smooth")
    assert smooth["visibleWhen"] == {"type2d": "flux_surface"}
    # ...and the field-line view, which draws a rectilinear Contour, offers it outright.
    fl_view = next(v for v in schema["views"] if v["id"] == "fieldline")
    fl_style = next(c for c in fl_view["controls"] if c["id"] == "contourStyle")
    assert all(opt.get("visibleWhen") is None for opt in fl_style["options"])


def test_colormap_shares_one_id_across_views_so_the_choice_follows_the_reader():
    owners = [vid for vid, spec in view_schema.VIEWS.items() if any(c.id == "colormap" for c in spec.controls)]
    assert set(owners) == {"2d", "3d", "fieldline"}


# ------------------------------------------------------------------------- LPK


def test_surface_curve_matches_the_batched_path_and_closes():
    vmec = _vmec()
    r, z = vmec.get_surface_curve(vmec.ns - 1, 0.0, res_u=160)
    r_grid, z_grid, _ = vmec.get_cross_section_data(0.0, "geometry", res_s=vmec.ns, res_u=160)
    # Not bit-exact: the single-surface kernel contracts a (1, nmodes) slice while the
    # batched one contracts (ns, nmodes), and XLA is free to order the reduction
    # differently. They agree to round-off, which is what matters.
    np.testing.assert_allclose(r, r_grid[vmec.ns - 1], rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(z, z_grid[vmec.ns - 1], rtol=1e-12, atol=1e-14)
    assert np.isclose(r[0], r[-1]) and np.isclose(z[0], z[-1])


def test_lpk_draws_three_curves_and_three_axis_markers():
    vmec = _vmec()
    fig = two_d.render_lpk(vmec, vmec.ns - 1, build_theme(True, 0))
    curves = [t for t in fig.data if getattr(t, "mode", None) == "lines"]
    markers = [t for t in fig.data if t.mode == "markers"]
    assert len(curves) == 3 and len(markers) == 3
    # Dash pattern carries the identity too, so the plot survives deuteranopia and print.
    assert len({c.line.dash for c in curves}) == 3
    assert len({c.line.color for c in curves}) == 3
    assert fig.layout.yaxis.scaleanchor == "x"


def test_lpk_cuts_sit_at_zero_quarter_and_half_of_one_field_period():
    """VMECplot's [0, pi/2, pi] are in its period-normalised angle - these are the same cuts."""
    vmec = _vmec()
    period = 2.0 * np.pi / vmec.nfp
    assert view_schema.two_d.LPK_FRACTIONS == (0.0, 0.25, 0.5)

    fig = two_d.render_lpk(vmec, vmec.ns - 1, build_theme(True, 0), res_u=240)
    curves = [t for t in fig.data if getattr(t, "mode", None) == "lines"]
    for frac, curve in zip(two_d.LPK_FRACTIONS, curves, strict=False):
        expected_r, expected_z = vmec.get_surface_curve(vmec.ns - 1, frac * period, res_u=240)
        np.testing.assert_allclose(np.asarray(curve.x, dtype=float), expected_r, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(np.asarray(curve.y, dtype=float), expected_z, rtol=0.0, atol=0.0)


def test_lpk_honours_the_colour_variable():
    """LPK is a viewpoint, not a plot type: it renders the currently selected field."""
    vmec = _vmec()
    theme = build_theme(True, 0)
    field_map = _field_map(vmec)

    def lpk(var):
        return view_schema.render_view(
            "2d", vmec, {"type2d": "cross_section", "lpkMode": True, "var2d": var}, theme, field_map
        )

    bare = lpk("geometry")
    field = lpk("j^u")
    assert bare.to_json() != field.to_json()
    # Geometry keeps the classic bare outlines: three curves, three axis markers, no fill.
    assert len(bare.data) == 6
    assert not any(t.type == "contourcarpet" for t in bare.data)
    # A field fills every cut exactly as the ordinary cross-section does...
    assert sum(1 for t in field.data if t.type == "contourcarpet") == 3
    # ...under one shared colorbar, or the overlap would blend unrelated scales.
    assert sum(1 for t in field.data if getattr(t, "showscale", False)) == 1
    # The fills are translucent; the identifying outlines on top are not.
    fills = [t for t in field.data if t.type == "contourcarpet"]
    assert all(f.opacity == two_d.LPK_FILL_OPACITY for f in fills)
    outlines = [t for t in field.data if getattr(t, "mode", None) == "lines"]
    assert len(outlines) == 3 and len({o.line.dash for o in outlines}) == 3


def test_lpk_does_not_leak_into_the_flux_surface_view():
    """A lpkMode left on while switching plot type must not hijack the other view."""
    vmec = _vmec()
    fig = view_schema.render_view(
        "2d", vmec, {"type2d": "flux_surface", "lpkMode": True, "var2d": "modB"},
        build_theme(True, 0), _field_map(vmec),
    )
    assert [t.type for t in fig.data] == ["contour"]


def test_background_grid_toggles_on_every_view():
    vmec = _vmec()
    theme = build_theme(True, 0)
    field_map = _field_map(vmec)
    cases = {
        "1d": {"profile": "iotaf"},
        "2d": {"type2d": "cross_section", "var2d": "modB"},
        "3d": {"var3d": "modB", "coordFree": False},
    }
    for view, controls in cases.items():
        for wanted in (False, True):
            fig = view_schema.render_view(view, vmec, {**controls, "showGrid": wanted}, theme, field_map)
            if view == "3d":
                assert fig.layout.scene.xaxis.showgrid is wanted, view
            else:
                assert fig.layout.xaxis.showgrid is wanted, view
                assert fig.layout.yaxis.showgrid is wanted, view


def test_grid_is_off_by_default():
    vmec = _vmec()
    fig = view_schema.render_view("1d", vmec, {"profile": "iotaf"}, build_theme(True, 0), _field_map(vmec))
    assert fig.layout.xaxis.showgrid is False


def test_diverging_scale_puts_red_at_the_top_whatever_the_data_sign():
    """RdBu must mean the same thing on |B| as it does on j^u."""
    from vmecdash.renderers.two_d import _sample_field_color

    signed = resolve_colorscale(np.array([-100.0, 100.0]))
    positive = resolve_colorscale(np.array([1.0, 5.0]), "RdBu")
    assert signed.reversescale is True and positive.reversescale is True
    # Plotly ships RdBu warm-to-cool, so reversing is what puts high at red.
    assert _sample_field_color(100.0, signed) == _sample_field_color(5.0, positive)
    assert "103" in _sample_field_color(100.0, signed)  # dark red end
    assert "48, 97" in _sample_field_color(-100.0, signed)  # dark blue end
    # A sequential scale is untouched by this rule.
    assert resolve_colorscale(np.array([1.0, 5.0])).reversescale is False


def test_no_shipped_condition_names_a_control_that_was_not_shipped():
    """The webview seeds only the controls it is sent; a condition on a missing one
    compares against undefined and can never match - that hid the phi slider on tokamaks."""
    for vmec in (_vmec(), _Axisymmetric()):
        for view in view_schema.build_ui_schema(vmec)["views"]:
            shipped = {control["id"] for control in view["controls"]}
            for control in view["controls"]:
                assert set(control["visibleWhen"] or {}) <= shipped, (view["id"], control["id"])


def test_a_tokamak_keeps_the_phi_slider_and_the_surface_count():
    schema = view_schema.build_ui_schema(_Axisymmetric())
    two_d_view = next(view for view in schema["views"] if view["id"] == "2d")
    by_id = {control["id"]: control for control in two_d_view["controls"]}
    assert "lpkMode" not in by_id
    assert by_id["phi"]["visibleWhen"] == {"type2d": "cross_section"}
    assert by_id["geoCount"]["visibleWhen"] == {"type2d": "cross_section", "var2d": "geometry"}


def test_a_stale_lpk_switch_never_draws_lpk_on_an_axisymmetric_equilibrium(monkeypatch):
    vmec = _vmec()
    # from_file hands back a shared cached instance: fake the tokamak via monkeypatch so
    # it is restored, or every later test would see ntor == 0.
    monkeypatch.setattr(vmec, "ntor", 0)
    fig = view_schema.render_view(
        "2d", vmec, {"type2d": "cross_section", "lpkMode": True, "var2d": "geometry"},
        build_theme(True, 0), _field_map(vmec),
    )
    assert "Evolution" not in (fig.layout.title.text or "")

