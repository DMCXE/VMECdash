import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-codex")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vmec_jax import VMECJaxProcessor


EXAMPLE_WOUT = "example/wout_PO.nc"
PS3_WOUT = Path("/Users/dmcxe/Downloads/wout_PS3_ms_dofs.nc")


def _eval_fourier_rows(cos_coeffs, sin_coeffs, xm, xn, theta, phi):
    theta = np.asarray(theta)
    angle = np.asarray(xm)[:, None] * theta[None, :] - np.asarray(xn)[:, None] * phi
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    return np.asarray(cos_coeffs) @ cos_angle + np.asarray(sin_coeffs) @ sin_angle


def _simsopt_full_grid_geometry(phi, theta):
    simsopt_mhd = pytest.importorskip("simsopt.mhd")
    vmec = simsopt_mhd.Vmec(EXAMPLE_WOUT)
    splines = simsopt_mhd.vmec_splines(vmec)
    return simsopt_mhd.vmec_compute_geometry(splines, vmec.s_full_grid, theta, np.array([phi]))


def test_cross_section_mesh_half_mesh_fields_match_simsopt_full_grid():
    """Half-mesh fields are spline-interpolated from VMEC half grid to full nodes."""
    vmec = VMECJaxProcessor.from_file(EXAMPLE_WOUT)
    phi = 0.37
    res_u = 32
    theta_nodes = np.linspace(0.0, 2 * np.pi, res_u + 1)
    geometry = _simsopt_full_grid_geometry(phi, theta_nodes)

    references = {
        "modB": geometry.modB[:, :, 0],
        "jacobian": geometry.sqrt_g_vmec[:, :, 0],
        "lambda": (geometry.theta_pest - geometry.theta_vmec)[:, :, 0],
        "B_u": geometry.B_sub_theta_vmec[:, :, 0],
        "B_v": geometry.B_sub_phi[:, :, 0],
        "B^u": geometry.B_sup_theta_vmec[:, :, 0],
        "B^v": geometry.B_sup_phi[:, :, 0],
    }

    for field, expected in references.items():
        r_nodes, z_nodes, values = vmec.get_cross_section_mesh(phi, field, res_u=res_u)

        assert r_nodes.shape == (vmec.ns, res_u + 1)
        assert z_nodes.shape == (vmec.ns, res_u + 1)
        assert values.shape == (vmec.ns, res_u + 1)
        np.testing.assert_allclose(values[1:], expected[1:], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(values[0], values[0, 0], rtol=0.0, atol=0.0)


def test_cross_section_mesh_full_mesh_evaluates_at_nodes():
    """Full-mesh fields remain direct Fourier values except for axis single-valuing."""
    vmec = VMECJaxProcessor.from_file(EXAMPLE_WOUT)
    phi = 0.4
    res_u = 24

    theta_nodes = np.linspace(0.0, 2 * np.pi, res_u + 1)
    for field in ("B_s", "j^u", "j^v"):
        _, _, values = vmec.get_cross_section_mesh(phi, field, res_u=res_u)
        assert values.shape == (vmec.ns, res_u + 1)

        payload = vmec._field_payload(field)
        cos_coeffs, sin_coeffs = payload["pair"]
        direct = _eval_fourier_rows(cos_coeffs, sin_coeffs, payload["xm"], payload["xn"], theta_nodes, phi)

        np.testing.assert_allclose(values[1:], direct[1:], rtol=1e-12, atol=1e-8)
        np.testing.assert_allclose(values[0], np.mean(direct[0, :-1]), rtol=1e-12, atol=1e-8)

    _, _, j_u_values = vmec.get_cross_section_mesh(phi, "j^u", res_u=res_u)
    assert np.max(np.abs(j_u_values[0] - j_u_values[1])) > 1.0


def test_cross_section_field_renderer_builds_carpet():
    if importlib.util.find_spec("dash_mantine_components") is None:
        pytest.skip("Dash UI dependencies are required for renderer tests")

    from views.shared import build_theme
    from views.two_d import render_cross_section_field

    vmec = VMECJaxProcessor.from_file(EXAMPLE_WOUT)
    fig = render_cross_section_field(vmec, 0.0, "modB", "|B| (Mod B)", build_theme(True, 0))

    trace_types = {trace.type for trace in fig.data}
    assert "carpet" in trace_types
    assert "contourcarpet" in trace_types
    assert len(fig.layout.images) == 0


def test_lambda_cross_section_uses_axis_fill_regularization():
    if importlib.util.find_spec("dash_mantine_components") is None:
        pytest.skip("Dash UI dependencies are required for renderer tests")

    from views.shared import build_theme
    from views.two_d import _carpet_colorscale, _sample_field_color, render_cross_section_field

    vmec = VMECJaxProcessor.from_file(EXAMPLE_WOUT)
    _, _, val_nodes = vmec.get_cross_section_mesh(0.0, "lambda", res_u=160)
    colorscale, reversescale, zmin, zmax = _carpet_colorscale(val_nodes)
    axis_value = float(np.nanmean(val_nodes[1, :-1]))
    expected_fill = _sample_field_color(axis_value, colorscale, reversescale, zmin, zmax)

    fig = render_cross_section_field(vmec, 0.0, "lambda", "Lambda", build_theme(True, 0))
    trace_types = [trace.type for trace in fig.data]

    assert trace_types == ["scatter", "carpet", "contourcarpet", "scatter"]
    assert fig.data[0].fill == "toself"
    assert fig.data[0].name == "Axis fill"
    assert fig.data[0].fillcolor == expected_fill
    assert np.min(np.asarray(fig.data[2].b, dtype=float)) > 0.0
    assert len(fig.layout.images) == 0


def test_modb_cross_section_keeps_degenerate_axis_in_carpet():
    if importlib.util.find_spec("dash_mantine_components") is None:
        pytest.skip("Dash UI dependencies are required for renderer tests")

    from views.shared import build_theme
    from views.two_d import render_cross_section_field

    vmec = VMECJaxProcessor.from_file(EXAMPLE_WOUT)
    fig = render_cross_section_field(vmec, 0.0, "modB", "|B| (Mod B)", build_theme(True, 0))
    trace_types = [trace.type for trace in fig.data]
    contour = next(trace for trace in fig.data if trace.type == "contourcarpet")

    assert trace_types == ["carpet", "contourcarpet", "scatter"]
    assert np.min(np.asarray(contour.b, dtype=float)) == 0.0


def test_ps3_lambda_cross_section_excludes_axis_from_contourcarpet():
    if importlib.util.find_spec("dash_mantine_components") is None:
        pytest.skip("Dash UI dependencies are required for renderer tests")
    if not PS3_WOUT.exists():
        pytest.skip("PS3 local smoke fixture is not available")

    from views.shared import build_theme
    from views.two_d import render_cross_section_field

    vmec = VMECJaxProcessor.from_file(str(PS3_WOUT))
    fig = render_cross_section_field(vmec, 0.0, "lambda", "Lambda", build_theme(True, 0))
    contour = next(trace for trace in fig.data if trace.type == "contourcarpet")

    assert fig.data[0].fill == "toself"
    assert np.min(np.asarray(contour.b, dtype=float)) > 0.0
