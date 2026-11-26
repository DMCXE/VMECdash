import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "vmecdash-mpl"))

from vmecdash.core import VMECJaxProcessor

EXAMPLE_WOUT = "example/wout_PO.nc"


class _Var:
    def __init__(self, values):
        self.values = values


def test_lambda_pair_keeps_lmns_for_stellarator_symmetric_file():
    vmec = VMECJaxProcessor.from_file(EXAMPLE_WOUT)
    lmnc, lmns = vmec._lambda_pair()

    assert not vmec.lasym
    assert np.max(np.abs(np.asarray(lmns))) > 0.0
    assert np.max(np.abs(np.asarray(lmnc))) == 0.0


def test_fieldline_axis_index_uses_first_half_mesh_surface():
    vmec = VMECJaxProcessor.from_file(EXAMPLE_WOUT)

    _, _, b_axis = vmec.compute_field_line_properties(0, alpha_points=4, zeta_points=8)
    _, _, b_first = vmec.compute_field_line_properties(1, alpha_points=4, zeta_points=8)

    assert np.max(np.abs(b_axis)) > 0.0
    np.testing.assert_allclose(b_axis, b_first, rtol=0.0, atol=0.0)


def test_asymmetric_lambda_and_b_sine_pairs_are_preserved():
    proc = object.__new__(VMECJaxProcessor)
    proc.lasym = True
    lmnc = np.array([[1.0, 2.0]])
    lmns = np.array([[3.0, 4.0]])
    bmnc = np.array([[5.0, 6.0]])
    bmns = np.array([[7.0, 8.0]])
    proc.ds = {"lmnc": _Var(lmnc), "lmns": _Var(lmns), "bmnc": _Var(bmnc), "bmns": _Var(bmns)}

    out_lmnc, out_lmns = proc._lambda_pair()
    out_bmnc, out_bmns = proc._coeff_pair("bmnc", "bmns")

    np.testing.assert_allclose(out_lmnc, lmnc)
    np.testing.assert_allclose(out_lmns, lmns)
    np.testing.assert_allclose(out_bmnc, bmnc)
    np.testing.assert_allclose(out_bmns, bmns)


def test_fieldline_modb_matches_simsopt_reference():
    if importlib.util.find_spec("simsopt") is None:
        import pytest

        pytest.skip("simsopt is required for the reference comparison")

    from simsopt.mhd import Vmec, vmec_fieldlines, vmec_splines

    s_idx = 60
    vmec = VMECJaxProcessor.from_file(EXAMPLE_WOUT)
    alpha, zeta, b_dash = vmec.compute_field_line_properties(
        s_idx, alpha_points=12, zeta_points=48, single_line=False
    )

    simsopt_vmec = Vmec(EXAMPLE_WOUT, verbose=False)
    splines = vmec_splines(simsopt_vmec)
    s_half = float(simsopt_vmec.s_half_grid[s_idx - 1])
    ref = vmec_fieldlines(splines, s_half, alpha, phi1d=zeta, phi_center=0, plot=False)

    np.testing.assert_allclose(b_dash, ref.modB[0], rtol=1e-10, atol=1e-10)


def test_newton_solver_zero_asymmetric_large_offset_and_folded_rejection():
    import jax.numpy as jnp

    from vmecdash.core.vmec_jax import (
        _check_surface_lambda_monotonicity,
        _solve_theta_star_verified,
        newton_solver,
        residual_fn,
    )

    xm = jnp.array([0.0, 1.0, 2.0, 3.0])
    xn = jnp.array([0.0, 0.0, 3.0, -3.0])
    iota = jnp.array(0.1)

    # A = 0: zero lambda returns theta* = alpha + iota * zeta directly
    th0 = newton_solver(0.7, 0.4, 0.0, iota, xm, xn, jnp.zeros(4), jnp.zeros(4))
    np.testing.assert_allclose(float(th0), 0.7 + 0.1 * 0.4, atol=1e-15)

    # Asymmetric lambda with large constant offset lmnc[0] = 50.0 rad
    lmns = jnp.array([0.0, 0.72, -0.09, 0.02])
    lmnc = jnp.array([50.0, -0.15, 0.04, -0.01])
    for alpha in (0.0, 1.1, 3.0, 5.9):
        for zeta in (0.0, 0.5, 1.7):
            th, ok = _solve_theta_star_verified(
                alpha, zeta, alpha + 0.1 * zeta, iota, xm, xn, lmnc, lmns
            )
            assert bool(ok) and np.isfinite(float(th))
            res = float(residual_fn(th, alpha, zeta, iota, xm, xn, lmnc, lmns))
            assert abs(res) < 1e-12

    # Singular root theta - sin(theta) = 0 at theta* = 0 -> rejected (NaN, valid=False)
    th_sing, ok_sing = _solve_theta_star_verified(
        0.0, 0.0, 0.0, 0.0, jnp.array([1.0]), jnp.array([0.0]), jnp.array([0.0]), jnp.array([-1.0])
    )
    assert not bool(ok_sing) and np.isnan(float(th_sing))

    # Native surface monotonicity check distinguishes folded, singular, grid_insufficient, monotone
    mono_folded = _check_surface_lambda_monotonicity(
        jnp.array([0.0]), jnp.array([1.25]), jnp.array([1.0]), jnp.array([0.0]), nfp=3
    )
    assert mono_folded["status"] == "folded" and mono_folded["min_dtheta_star_du"] < 0.0

    mono_sing = _check_surface_lambda_monotonicity(
        jnp.array([0.0]), jnp.array([-1.0]), jnp.array([1.0]), jnp.array([0.0]), nfp=3
    )
    assert mono_sing["status"] == "singular" and abs(mono_sing["min_dtheta_star_du"]) <= 1e-12

    mono_insuff = _check_surface_lambda_monotonicity(
        jnp.array([0.0]), jnp.array([0.99]), jnp.array([1.0]), jnp.array([0.0]), nfp=3, n_theta=16, n_zeta=8
    )
    assert mono_insuff["status"] == "grid_insufficient" and mono_insuff["min_dtheta_star_du"] > 0.0

    mono_ok = _check_surface_lambda_monotonicity(
        jnp.array([0.0]), jnp.array([0.1]), jnp.array([1.0]), jnp.array([0.0]), nfp=3, n_theta=96, n_zeta=96
    )
    assert mono_ok["status"] == "monotone"


def test_scan_bad_point_does_not_propagate_to_subsequent_points():
    import jax.numpy as jnp

    from vmecdash.core.vmec_jax import _compute_field_line_scan_with_mask

    params = {
        "nfp": 3,
        "iota": jnp.array(0.1),
        "xm_lambda": jnp.array([1.0]),
        "xn_lambda": jnp.array([0.0]),
        "lmnc": jnp.array([0.0]),
        "lmns": jnp.array([0.5]),
        "xm_b": jnp.array([0.0, 1.0]),
        "xn_b": jnp.array([0.0, 0.0]),
        "bmnc": jnp.array([1.0, 0.1]),
        "bmns": jnp.array([0.0, 0.0]),
    }
    # Inject a NaN zeta in the middle: point 2 fails, but points 3 and 4 must recover cleanly.
    zeta_grid = jnp.array([0.0, 0.2, jnp.nan, 0.6, 0.8])
    b_line, valid = _compute_field_line_scan_with_mask(0.3, zeta_grid, params)
    b_np = np.asarray(b_line)
    v_np = np.asarray(valid)
    assert list(v_np) == [True, True, False, True, True]
    assert np.isnan(b_np[2])
    assert np.all(np.isfinite(b_np[[0, 1, 3, 4]]))


def test_renderer_equidistant_lines_and_single_trace_annotations():
    from vmecdash.renderers.fieldline import (
        render_fieldline_heatmap,
        render_fieldline_lines,
        render_single_trace,
    )
    from vmecdash.theme import build_theme

    vmec = VMECJaxProcessor.from_file(EXAMPLE_WOUT)
    theme = build_theme(True, 0)
    s_idx = vmec.ns - 1
    expected_s = vmec.half_mesh_s(s_idx)

    # 1d_lines with default n_lines=6 must be exact 60 deg spacing (no 2.8125 deg endpoint overlap)
    fig_lines = render_fieldline_lines(vmec, s_idx=s_idx, n_lines=6, res=64, theme=theme)
    assert len(fig_lines.data) == 6
    alphas_deg = [float(tr.name.split("(")[1].split("°")[0]) for tr in fig_lines.data]
    np.testing.assert_allclose(alphas_deg, [0.0, 60.0, 120.0, 180.0, 240.0, 300.0], atol=0.15)
    assert f"s_idx={s_idx} (s={expected_s:.4f})" in fig_lines.layout.title.text
    assert all(tr.connectgaps is False for tr in fig_lines.data)

    # Heatmap shows half-mesh s_idx and s
    fig_hm = render_fieldline_heatmap(vmec, s_idx=s_idx, res=32, theme=theme)
    assert f"s_idx={s_idx} (s={expected_s:.4f})" in fig_hm.layout.title.text

    # Single trace shows folded segment-start label, toroidal turns, return period, and connectgaps=False
    fig_st = render_single_trace(vmec, s_idx=s_idx, transits=12, alpha0=0.0, res=32, theme=theme)
    assert fig_st.data[1].connectgaps is False
    sub0 = fig_st.layout.annotations[0].text
    assert "Folded" in sub0 and "toroidal turns" in sub0 and "return" in sub0
    assert f"s_idx={s_idx} (s={expected_s:.4f})" in fig_st.layout.title.text


def test_compact_wout_1200262026_fieldline_accuracy_against_brentq():
    import pytest
    from scipy.optimize import brentq

    from vmecdash.renderers.fieldline import render_fieldline_lines, render_single_trace
    from vmecdash.theme import build_theme

    compact_wout = Path(os.environ.get("VMECDASH_COMPACT_WOUT", ROOT / "example/wout_1200262026.nc"))
    if not compact_wout.is_file():
        pytest.skip("set VMECDASH_COMPACT_WOUT to the wout_1200262026.nc fixture")

    vmec = VMECJaxProcessor.from_file(str(compact_wout))
    s_idx = 63
    assert np.isclose(vmec.half_mesh_s(s_idx), 62.5 / 63.0, rtol=1e-14)

    # 2d_modB (128 & 256, shift=False/True) -> 0 NaN, 0 bad points
    for res in (128, 256):
        for shift in (False, True):
            zeta_offset = float(np.pi / vmec.nfp) if shift else 0.0
            alphas, zetas, b_grid, diag = vmec.compute_field_line_properties(
                s_idx=s_idx,
                alpha_points=res,
                zeta_points=res,
                single_line=False,
                zeta_offset=zeta_offset,
                return_diagnostics=True,
            )
            assert diag["n_invalid"] == 0
            assert np.all(np.isfinite(b_grid))
            assert diag["min_dtheta_star_du"] > 0.05

    # Pointwise |B| vs independent scipy brentq reference < 1e-11 T
    alphas, zetas, b_grid = vmec.compute_field_line_properties(
        s_idx=s_idx, alpha_points=32, zeta_points=32, single_line=False
    )
    iota = float(vmec.ds["iotas"].values[s_idx])
    xm_l = np.asarray(vmec.xm, float)
    xn_l = np.asarray(vmec.xn, float)
    lmns = np.asarray(vmec.ds["lmns"].values[s_idx], float)
    xm_b = np.asarray(vmec.xm_nyq, float)
    xn_b = np.asarray(vmec.xn_nyq, float)
    bmnc = np.asarray(vmec.ds["bmnc"].values[s_idx], float)
    amp = float(np.sum(np.abs(lmns))) + 1e-6

    max_err = 0.0
    for ia in range(0, 32, 4):
        for iz in range(0, 32, 4):
            th_s = alphas[ia] + iota * zetas[iz]
            u = brentq(
                lambda uu, zeta, theta_star: uu + np.sum(lmns * np.sin(xm_l * uu - xn_l * zeta)) - theta_star,
                th_s - amp,
                th_s + amp,
                args=(zetas[iz], th_s),
                xtol=1e-14,
            )
            b_ref = float(np.sum(bmnc * np.cos(xm_b * u - xn_b * zetas[iz])))
            max_err = max(max_err, abs(float(b_grid[ia, iz]) - b_ref))
    assert max_err < 1e-11

    # single_trace (transits=50, res=128) -> 0 NaN, near-rational return 30 periods, -1.74 deg
    zeta_st, alpha_segs, b_st, diag_st = vmec.compute_field_line_properties(
        s_idx=s_idx, zeta_points=128, n_transits=50, alpha0=0.0, single_line=True, return_diagnostics=True
    )
    assert diag_st["n_invalid"] == 0 and np.all(np.isfinite(b_st))
    theme = build_theme(True, 0)
    fig_st = render_single_trace(vmec, s_idx=s_idx, transits=50, alpha0=0.0, res=128, theme=theme)
    assert "30 periods" in fig_st.layout.annotations[0].text
    assert "-1.74°" in fig_st.layout.annotations[0].text
    assert "s_idx=63 (s=0.9921)" in fig_st.layout.title.text

    # 1d_lines n_lines=6 exact 60 deg spacing
    fig_1d = render_fieldline_lines(vmec, s_idx=s_idx, n_lines=6, res=128, theme=theme)
    assert len(fig_1d.data) == 6
    assert all(np.all(np.isfinite(tr.y)) for tr in fig_1d.data)
