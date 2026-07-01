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

from vmec_jax import VMECJaxProcessor


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
