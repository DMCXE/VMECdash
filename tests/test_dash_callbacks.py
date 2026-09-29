"""Dash callbacks that decide what the controls panel shows and what 2-D draws."""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-codex")

dash = pytest.importorskip("dash")

from vmecdash.core import VMECJaxProcessor  # noqa: E402
from vmecdash.dash_app import app as dash_app  # noqa: E402
from vmecdash.theme import build_theme  # noqa: E402

EXAMPLE_WOUT = str(ROOT / "example" / "wout_PO.nc")
SHOW = {"display": "flex"}
HIDE = {"display": "none"}


def test_loading_a_tokamak_switches_a_stale_lpk_view_off():
    """Disabling the switch alone left a True in force that the user could not uncheck."""
    outputs = dash_app.update_controls({"ns": 50, "nfp": 1, "ntor": 0})
    disabled, checked = outputs[1], outputs[2]
    assert disabled is True
    assert checked is False


def test_loading_a_stellarator_leaves_the_lpk_choice_alone():
    outputs = dash_app.update_controls({"ns": 50, "nfp": 3, "ntor": 4})
    assert outputs[1] is False
    assert outputs[2] is dash.no_update


def test_lpk_hides_the_shared_shading_panel_only_while_2d_is_on_screen():
    *_, shading_in_2d = dash_app.toggle_phi_slider("cross_section", "modB", True, "2d")
    *_, shading_in_fieldline = dash_app.toggle_phi_slider("cross_section", "modB", True, "fieldline")
    assert shading_in_2d == HIDE
    assert shading_in_fieldline == SHOW


def test_render_2d_figure_ignores_lpk_on_an_axisymmetric_equilibrium(monkeypatch):
    vmec = VMECJaxProcessor.from_file(EXAMPLE_WOUT)
    monkeypatch.setattr(vmec, "ntor", 0)  # shared cached instance - restore after the test
    fig = dash_app.render_2d_figure(
        vmec, build_theme(True, 0),
        type_2d="cross_section", var_2d="geometry", phi_frac=0.0,
        s_idx=vmec.ns - 1, geo_count=15, lpk_mode=True,
    )
    assert "Evolution" not in (fig.layout.title.text or "")
