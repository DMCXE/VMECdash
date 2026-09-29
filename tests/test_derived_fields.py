"""Cylindrical field components, parallel current, and the force residual.

The angular derivatives and the cylindrical components are checked against simsopt. The
current normalisation is checked against the wout file's own <J.B> profile, because every
equilibrium to hand is zero-beta and so cannot constrain force balance.
"""

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-codex")

import jax.numpy as jnp  # noqa: E402

from vmecdash.core import VMECJaxProcessor  # noqa: E402
from vmecdash.core.vmec_jax import (  # noqa: E402
    DERIVED_FIELD_SPECS,
    _jit_geometry_derivatives,
    _jit_pair_at_points,
    _jit_pair_on_grid,
)

EXAMPLE_WOUT = str(ROOT / "example" / "wout_PO.nc")

# Mid-radius: the half-mesh lift is least accurate at the axis and the edge.
MID = 60


def _vmec():
    return VMECJaxProcessor.from_file(EXAMPLE_WOUT)


def _grid(n_theta=8, n_phi=4, nfp=1):
    theta = np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False)
    phi = np.linspace(0.0, 2 * np.pi / nfp, n_phi, endpoint=False)
    tt, pp = np.meshgrid(theta, phi, indexing="ij")
    return theta, phi, tt, pp


def _simsopt_geometry(vmec, s_index, theta, phi):
    from simsopt.mhd import Vmec, vmec_compute_geometry

    return vmec_compute_geometry(Vmec(EXAMPLE_WOUT), np.array([s_index / (vmec.ns - 1)]), theta, phi)


def _needs_simsopt():
    if importlib.util.find_spec("simsopt") is None:
        import pytest

        pytest.skip("simsopt is not installed")


# ------------------------------------------------------------------ C1: derivatives


def test_angular_derivatives_match_simsopt():
    _needs_simsopt()
    vmec = _vmec()
    theta, phi, tt, pp = _grid(nfp=vmec.nfp)
    ref = _simsopt_geometry(vmec, MID, theta, phi)

    surfaces = np.array([MID])
    r, z, r_t, r_v, z_t, z_v = _jit_geometry_derivatives(
        vmec.rmnc[surfaces], vmec.rmns[surfaces], vmec.zmns[surfaces], vmec.zmnc[surfaces],
        vmec.xm, vmec.xn, jnp.asarray(tt.ravel()), jnp.asarray(pp.ravel()),
    )
    got = {
        "R": r, "Z": z,
        "d_R_d_theta_vmec": r_t, "d_R_d_phi": r_v,
        "d_Z_d_theta_vmec": z_t, "d_Z_d_phi": z_v,
    }
    for name, values in got.items():
        expected = getattr(ref, name)[0]
        actual = np.asarray(values)[0].reshape(tt.shape)
        # Same closed-form Fourier sum on both sides, so this must hold to round-off.
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12, err_msg=name)


# ---------------------------------------------------------------- C2: cylindrical B


def test_cylindrical_components_match_simsopt():
    _needs_simsopt()
    vmec = _vmec()
    theta, phi, tt, pp = _grid(nfp=vmec.nfp)
    ref = _simsopt_geometry(vmec, MID, theta, phi)

    def ours(key):
        return np.asarray(
            vmec.derived_values(key, np.array([MID]), tt.ravel(), pp.ravel())
        )[0].reshape(tt.shape)

    # simsopt reports Cartesian components; rotate into the cylindrical frame.
    cos_p, sin_p = np.cos(pp), np.sin(pp)
    expected = {
        "B_R": ref.B_X[0] * cos_p + ref.B_Y[0] * sin_p,
        "B_phi": -ref.B_X[0] * sin_p + ref.B_Y[0] * cos_p,
        "B_Z": ref.B_Z[0],
    }
    for key, want in expected.items():
        # Looser than the geometry check: B^u/B^v live on the half mesh, and our cubic
        # radial lift is not the same interpolation simsopt uses.
        rel = np.max(np.abs(ours(key) - want)) / np.max(np.abs(want))
        assert rel < 1e-3, f"{key} rel err {rel:.3e}"


def test_field_magnitude_from_components_matches_modb():
    _needs_simsopt()
    vmec = _vmec()
    theta, phi, tt, pp = _grid(nfp=vmec.nfp)
    ref = _simsopt_geometry(vmec, MID, theta, phi)

    parts = [
        np.asarray(vmec.derived_values(k, np.array([MID]), tt.ravel(), pp.ravel()))[0].reshape(tt.shape)
        for k in ("B_R", "B_phi", "B_Z")
    ]
    mod_b = np.sqrt(sum(p**2 for p in parts))
    rel = np.max(np.abs(mod_b - ref.modB[0])) / np.max(np.abs(ref.modB[0]))
    assert rel < 1e-3, f"|B| rel err {rel:.3e}"


def test_toroidal_component_is_r_times_bsupv():
    """B_phi is the physical component, not the contravariant one."""
    vmec = _vmec()
    theta, phi, tt, pp = _grid(nfp=vmec.nfp)
    flat_t, flat_p = jnp.asarray(tt.ravel()), jnp.asarray(pp.ravel())
    surfaces = np.array([MID])

    r, *_ = _jit_geometry_derivatives(
        vmec.rmnc[surfaces], vmec.rmns[surfaces], vmec.zmns[surfaces], vmec.zmnc[surfaces],
        vmec.xm, vmec.xn, flat_t, flat_p,
    )
    b_v_cos, b_v_sin = vmec._field_pair_full_mesh("B^v")
    b_sup_v = _jit_pair_at_points(
        b_v_cos[surfaces], b_v_sin[surfaces], vmec.xm_nyq, vmec.xn_nyq, flat_t, flat_p
    )
    expected = np.asarray(r * b_sup_v)[0]
    got = np.asarray(vmec.derived_values("B_phi", surfaces, tt.ravel(), pp.ravel()))[0]
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=0.0)


# ------------------------------------------------------- C3: current normalisation


def test_parallel_current_normalisation_matches_the_files_own_jdotb():
    """No mu0 or sqrt(g) factor hides in currumnc/currvmnc.

    A wrong normalisation would be off by 1/mu0 (~8e5), not by the O(1) spread that the
    half-mesh radial lift produces.
    """
    vmec = _vmec()
    n = 64
    theta = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    zeta = np.linspace(0.0, 2 * np.pi / vmec.nfp, n, endpoint=False)
    tt, zz = np.meshgrid(theta, zeta, indexing="ij")
    flat_t, flat_z = jnp.asarray(tt.ravel()), jnp.asarray(zz.ravel())

    reference = np.asarray(vmec._profile_arrays["jdotb"])
    for idx in (40, 60, 80):
        surfaces = np.array([idx])
        j_para = np.asarray(vmec.derived_values("j_para", surfaces, tt.ravel(), flat_z))[0]
        b_cos, b_sin = vmec._field_pair_full_mesh("B^u")
        v_cos, v_sin = vmec._field_pair_full_mesh("B^v")
        g_cos, g_sin = vmec._field_pair_full_mesh("jacobian")
        sqrt_g = np.asarray(
            _jit_pair_at_points(g_cos[surfaces], g_sin[surfaces], vmec.xm_nyq, vmec.xn_nyq, flat_t, flat_z)
        )[0]
        r, _, r_t, r_v, z_t, z_v = _jit_geometry_derivatives(
            vmec.rmnc[surfaces], vmec.rmns[surfaces], vmec.zmns[surfaces], vmec.zmnc[surfaces],
            vmec.xm, vmec.xn, flat_t, flat_z,
        )
        b_u = np.asarray(_jit_pair_at_points(b_cos[surfaces], b_sin[surfaces], vmec.xm_nyq, vmec.xn_nyq, flat_t, flat_z))[0]
        b_v = np.asarray(_jit_pair_at_points(v_cos[surfaces], v_sin[surfaces], vmec.xm_nyq, vmec.xn_nyq, flat_t, flat_z))[0]
        mod_b = np.sqrt(
            (b_u * np.asarray(r_t)[0] + b_v * np.asarray(r_v)[0]) ** 2
            + (np.asarray(r)[0] * b_v) ** 2
            + (b_u * np.asarray(z_t)[0] + b_v * np.asarray(z_v)[0]) ** 2
        )
        # <J.B> is the sqrt(g)-weighted flux-surface average.
        average = float(np.sum(j_para * mod_b * sqrt_g) / np.sum(sqrt_g))
        ratio = average / reference[idx]
        assert 0.5 < ratio < 2.0, f"idx {idx}: <J.B> ratio {ratio:.3f} — normalisation is off"


# ------------------------------------------------------------------ registry wiring


def test_derived_fields_are_offered_and_render_in_every_view():
    vmec = _vmec()
    offered = {opt["value"] for opt in vmec.available_fields()}
    assert set(DERIVED_FIELD_SPECS) <= offered

    for key in DERIVED_FIELD_SPECS:
        _, _, surface = vmec.get_flux_surface_data(MID, key, res_u=16, res_v=16)
        _, _, cross = vmec.get_cross_section_mesh(0.2, key, res_u=32)
        *_, volume = vmec.compute_3d_surface(s_idx=MID, var_name=key, n_theta=16, n_zeta_per_period=16)
        for name, values in (("surface", surface), ("cross-section", cross), ("3d", volume)):
            assert values is not None, f"{key} {name}"
            assert np.isfinite(np.asarray(values)).all(), f"{key} {name} has non-finite values"


def test_force_residual_is_a_magnitude():
    """It is reported raw in N/m^3, not divided by |grad p| - which is zero at beta = 0."""
    vmec = _vmec()
    assert float(np.max(np.asarray(vmec._profile_arrays["presf"]))) == 0.0
    _, _, values = vmec.get_flux_surface_data(MID, "force_residual", res_u=16, res_v=16)
    assert np.all(values >= 0.0)
    assert np.isfinite(values).all()


# ---------------------------------------------------------------- theta-zeta grid path


def test_grid_kernel_matches_the_direct_mode_sum():
    """The separable theta-zeta kernel must equal the direct O(M * T) evaluation."""
    vmec = _vmec()
    cos_c, sin_c = vmec._field_pairs["modB"]
    # Non-square grids, so a swapped theta/zeta axis cannot pass by symmetry.
    for n_theta, n_zeta in ((32, 16), (64, 48)):
        theta = np.linspace(0.0, 2 * np.pi, n_theta)
        zeta = np.linspace(0.0, 2 * np.pi / vmec.nfp, n_zeta)
        grid = np.asarray(
            _jit_pair_on_grid(cos_c[MID], sin_c[MID], vmec.xm_nyq, vmec.xn_nyq, jnp.asarray(theta), jnp.asarray(zeta))
        )[0]
        tt, zz = np.meshgrid(theta, zeta, indexing="ij")
        direct = np.asarray(
            _jit_pair_at_points(
                cos_c[MID][None], sin_c[MID][None], vmec.xm_nyq, vmec.xn_nyq,
                jnp.asarray(tt.ravel()), jnp.asarray(zz.ravel()),
            )
        )[0].reshape(tt.shape)
        np.testing.assert_allclose(grid, direct, rtol=1e-12, atol=1e-12)


def test_gridded_derived_values_match_the_validated_paired_path():
    """The theta-zeta view's derived fields inherit the paired path's simsopt validation."""
    vmec = _vmec()
    theta = np.linspace(0.0, 2 * np.pi, 20)
    zeta = np.linspace(0.0, 2 * np.pi / vmec.nfp, 12)
    tt, zz = np.meshgrid(theta, zeta, indexing="ij")
    for key in DERIVED_FIELD_SPECS:
        grid = np.asarray(vmec.derived_values_grid(key, np.array([MID]), theta, zeta))[0]
        paired = np.asarray(vmec.derived_values(key, np.array([MID]), tt.ravel(), zz.ravel()))[0].reshape(tt.shape)
        scale = max(1.0, float(np.nanmax(np.abs(paired))))
        np.testing.assert_allclose(grid, paired, rtol=1e-10, atol=1e-10 * scale, err_msg=key)


def test_theta_zeta_view_stays_bounded_at_the_highest_resolution():
    """At 960 x 960 the previous kernel needed ~3.8 GB for each intermediate array."""
    vmec = _vmec()
    _, _, values = vmec.get_flux_surface_data(MID, "modB", res_u=960, res_v=960)
    assert values.shape == (960, 960) and np.isfinite(values).all()
    _, _, values = vmec.get_flux_surface_data(MID, "B_R", res_u=480, res_v=480)
    assert values.shape == (480, 480) and np.isfinite(values).all()


def test_force_residual_is_not_offered_without_a_pressure_profile(monkeypatch):
    vmec = _vmec()
    assert "force_residual" in {opt["value"] for opt in vmec.available_fields()}
    trimmed = {k: v for k, v in vmec._profile_arrays.items() if k != "presf"}
    monkeypatch.setattr(vmec, "_profile_arrays", trimmed)
    monkeypatch.setattr(vmec, "_dp_ds", None)
    assert "force_residual" not in {opt["value"] for opt in vmec.available_fields()}
    # And asking for it anyway is a clean "no data", not an exception.
    assert vmec.derived_values("force_residual", np.array([MID]), np.zeros(3), np.zeros(3)) is None

