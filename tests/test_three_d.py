"""3-D surface view: sampling, the closed triangulation, and the resolution policy."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-codex")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vmecdash.core import VMECJaxProcessor  # noqa: E402
from vmecdash.renderers.three_d import (  # noqa: E402
    AUTO_MAX_VERTICES,
    AUTO_MIN_POINTS_PER_PERIOD,
    AUTO_POINTS,
    render_3d,
    resolve_3d_resolution,
    torus_triangles,
)
from vmecdash.theme import build_theme  # noqa: E402

EXAMPLE_WOUT = str(ROOT / "example" / "wout_PO.nc")


@pytest.fixture(scope="module")
def vmec():
    return VMECJaxProcessor.from_file(EXAMPLE_WOUT)


def _direct(cos_coeffs, sin_coeffs, xm, xn, theta, zeta):
    angle = np.asarray(xm)[:, None, None] * theta[None, :, None] - np.asarray(xn)[:, None, None] * zeta[None, None, :]
    return np.einsum("m,mtz->tz", np.asarray(cos_coeffs), np.cos(angle)) + np.einsum(
        "m,mtz->tz", np.asarray(sin_coeffs), np.sin(angle)
    )


def test_surface_matches_direct_fourier_sum_on_a_periodic_grid(vmec):
    idx, n_theta, per_period = vmec.ns - 1, 12, 8
    x, y, z, val = vmec.compute_3d_surface(idx, "modB", n_theta=n_theta, n_zeta_per_period=per_period)

    n_zeta = per_period * vmec.nfp
    assert x.shape == y.shape == z.shape == val.shape == (n_theta, n_zeta)

    # No repeated endpoint in either angle: the triangulation wraps instead.
    theta = np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False)
    zeta = np.linspace(0.0, 2 * np.pi, n_zeta, endpoint=False)
    r = _direct(vmec.rmnc[idx], vmec.rmns[idx], vmec.xm, vmec.xn, theta, zeta)
    payload = vmec._field_payload("modB")
    cos_b, sin_b = payload["pair"]
    b = _direct(cos_b[idx], sin_b[idx], payload["xm"], payload["xn"], theta, zeta)

    np.testing.assert_allclose(x, r * np.cos(zeta)[None, :], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(y, r * np.sin(zeta)[None, :], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(z, _direct(vmec.zmnc[idx], vmec.zmns[idx], vmec.xm, vmec.xn, theta, zeta), atol=1e-12)
    np.testing.assert_allclose(val, b, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("n_theta, n_zeta", [(3, 4), (8, 24)])
def test_triangulation_closes_the_torus(n_theta, n_zeta):
    i, j, k = torus_triangles(n_theta, n_zeta)
    assert len(i) == 2 * n_theta * n_zeta
    tris = np.stack([i, j, k], axis=1)
    assert tris.min() == 0 and tris.max() == n_theta * n_zeta - 1
    assert all(len(set(t)) == 3 for t in tris.tolist()), "degenerate triangle"

    # Closed 2-manifold: every edge is shared by exactly two triangles, so there is no
    # boundary - and so no seam for the shading to break along.
    edges = {}
    for t in tris.tolist():
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            key = (min(a, b), max(a, b))
            edges[key] = edges.get(key, 0) + 1
    assert set(edges.values()) == {2}
    # Euler characteristic of a torus: V - E + F = 0.
    assert n_theta * n_zeta - len(edges) + len(tris) == 0


def test_auto_resolution_samples_every_field_period_alike():
    assert resolve_3d_resolution("auto", 1) == (AUTO_POINTS, AUTO_POINTS)
    assert resolve_3d_resolution(None, 5) == (AUTO_POINTS, AUTO_POINTS)
    assert resolve_3d_resolution("96", 5) == (96, 96)

    # High-nfp devices are trimmed toroidally to stay within the vertex budget, never
    # below the floor.
    n_theta, per_period = resolve_3d_resolution("auto", 10)
    assert n_theta * per_period * 10 <= AUTO_MAX_VERTICES
    assert resolve_3d_resolution("auto", 1000)[1] == AUTO_MIN_POINTS_PER_PERIOD


def test_render_uses_a_closed_mesh_and_honours_the_resolution(vmec):
    theme = build_theme(True)
    fig = render_3d(vmec, vmec.ns - 1, "modB", "|B|", True, theme, colormap="RdBu", resolution="64")
    (trace,) = fig.data
    assert trace.type == "mesh3d"
    assert len(trace.x) == 64 * 64 * vmec.nfp
    assert len(trace.i) == 2 * len(trace.x)
    assert trace.cmin < trace.cmax
    assert trace.flatshading is False

    finer = render_3d(vmec, vmec.ns - 1, "modB", "|B|", True, theme, resolution="128")
    assert len(finer.data[0].x) == 4 * len(trace.x)


def test_geometry_is_drawn_in_a_flat_colour(vmec):
    fig = render_3d(vmec, vmec.ns - 1, "geometry", "Geometry", False, build_theme(False), resolution="64")
    (trace,) = fig.data
    assert trace.type == "mesh3d"
    assert trace.intensity is None
    assert trace.color is not None
