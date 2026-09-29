"""Cache bounds, refresh, and the render memo.

These cover the session lifecycle the other suites never exercise: eviction, mtime
invalidation, dispose, and the figure memo that sits in front of ``render``.
"""

import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-codex")

from vmecdash import view_schema  # noqa: E402
from vmecdash.core import VMECJaxProcessor  # noqa: E402
from vmecdash.theme import build_theme  # noqa: E402
from vmecdash.vscode_backend import BackendError, VmecDashBackend, _shrink_figure  # noqa: E402

EXAMPLE_WOUT = str(ROOT / "example" / "wout_PO.nc")


def _copy(tmp_path, name):
    target = tmp_path / name
    shutil.copyfile(EXAMPLE_WOUT, target)
    return str(target)


def _bump_mtime(path):
    # Coarse filesystem timestamps make a same-millisecond rewrite look unchanged.
    time.sleep(0.01)
    os.utime(path, None)


# --------------------------------------------------------------------------- cache


def test_cache_is_bounded_and_lru(tmp_path):
    saved = dict(VMECJaxProcessor._CACHE)
    try:
        VMECJaxProcessor._CACHE.clear()
        for index in range(VMECJaxProcessor._CACHE_MAX + 4):
            VMECJaxProcessor.from_file(_copy(tmp_path, f"wout_{index:02d}.nc"))
        assert len(VMECJaxProcessor._CACHE) == VMECJaxProcessor._CACHE_MAX
    finally:
        VMECJaxProcessor._CACHE.clear()
        VMECJaxProcessor._CACHE.update(saved)


def test_evicted_processor_stays_usable_while_referenced(tmp_path):
    """The correctness trap: eviction must not close a processor someone still holds."""
    pinned_path = _copy(tmp_path, "wout_pinned.nc")
    pinned = VMECJaxProcessor.from_file(pinned_path)

    for index in range(VMECJaxProcessor._CACHE_MAX + 2):
        VMECJaxProcessor.from_file(_copy(tmp_path, f"wout_filler_{index:02d}.nc"))

    assert pinned_path not in VMECJaxProcessor._CACHE  # dropped from the strong-ref LRU
    assert pinned.get_1d_data("iotaf")[1].shape[0] > 0  # ...but still fully alive
    assert VMECJaxProcessor.from_file(pinned_path) is pinned  # ...and still the one identity


def test_mtime_change_yields_new_instance_without_closing_the_old(tmp_path):
    path = _copy(tmp_path, "wout_touched.nc")
    old = VMECJaxProcessor.from_file(path)
    _bump_mtime(path)
    new = VMECJaxProcessor.from_file(path)

    assert new is not old
    # A live session may still be rendering from the old one until it refreshes.
    assert old.get_1d_data("iotaf")[1].shape[0] > 0


def test_close_releases_arrays_and_is_idempotent(tmp_path):
    processor = VMECJaxProcessor.from_file(_copy(tmp_path, "wout_closed.nc"))
    assert processor._field_pairs  # the bulk of the footprint

    processor.close()
    assert processor._field_pairs == {}
    assert processor._half_to_full_weights is None
    processor.close()  # must not raise


# ------------------------------------------------------------------- canonical keys


def test_canonical_controls_collapses_defaults_and_aliases():
    vmec = VMECJaxProcessor.from_file(EXAMPLE_WOUT)

    explicit = {"type2d": "cross_section", "var2d": "geometry", "phi": 0.0, "sIdx": vmec.ns - 1, "geoCount": 15}
    aliased = {"type": "cross_section", "var": "geometry"}
    assert view_schema.canonical_controls("2d", {}, vmec) == view_schema.canonical_controls("2d", explicit, vmec)
    assert view_schema.canonical_controls("2d", {}, vmec) == view_schema.canonical_controls("2d", aliased, vmec)

    # Numeric strings and numbers address the same figure.
    as_text = view_schema.canonical_controls("fieldline", {"fieldlineRes": "256"}, vmec)
    as_number = view_schema.canonical_controls("fieldline", {"fieldlineRes": 256}, vmec)
    assert as_text == as_number

    # A slider at 0 and an unchecked toggle are real values, not "missing".
    zeroed = view_schema.canonical_controls("3d", {"s3dIdx": 0, "coordFree": False}, vmec)
    assert dict(zeroed)["s3dIdx"] == 0.0
    assert dict(zeroed)["coordFree"] is False


def test_canonical_controls_differs_when_a_control_differs():
    vmec = VMECJaxProcessor.from_file(EXAMPLE_WOUT)
    base = view_schema.canonical_controls("3d", {}, vmec)
    assert base != view_schema.canonical_controls("3d", {"var3d": "geometry"}, vmec)


# -------------------------------------------------------------------- payload diet


def test_shrink_figure_preserves_structure_and_precision():
    vmec = VMECJaxProcessor.from_file(EXAMPLE_WOUT)
    theme = build_theme(True, 0)
    field_map = {opt["value"]: opt.get("label", opt["value"]) for opt in vmec.available_fields()}
    controls = {"var3d": "modB", "s3dIdx": vmec.ns - 1, "coordFree": True}

    figure = view_schema.render_view("3d", vmec, controls, theme, field_map)
    before_types = [trace.type for trace in figure.data]
    before_z = np.array(figure.data[0].z, dtype=float, copy=True)

    _shrink_figure(figure)

    assert [trace.type for trace in figure.data] == before_types
    np.testing.assert_allclose(np.asarray(figure.data[0].z, dtype=float), before_z, rtol=1e-6, atol=0.0)


def test_shrink_figure_handles_zero_and_nonfinite():
    from vmecdash.vscode_backend import _round_sig

    values = np.array([0.0, np.nan, np.inf, -np.inf, 1.0e-9, -1.2345678901e5])
    out = _round_sig(values, 7)

    assert out[0] == 0.0
    assert np.isnan(out[1]) and np.isposinf(out[2]) and np.isneginf(out[3])
    np.testing.assert_allclose(out[4:], values[4:], rtol=1e-6, atol=0.0)


# ------------------------------------------------------------------ backend memo


def test_render_memo_hits_and_clears_on_refresh(tmp_path, monkeypatch):
    path = _copy(tmp_path, "wout_memo.nc")
    backend = VmecDashBackend()
    meta = backend.open({"path": path})
    request = {"sessionId": meta["sessionId"], "view": "1d", "controls": {"profile": "iotaf"}, "theme": "dark"}

    renders = []
    real_render_view = view_schema.render_view

    def counting_render_view(*args, **kwargs):
        renders.append(args[0])
        return real_render_view(*args, **kwargs)

    monkeypatch.setattr(view_schema, "render_view", counting_render_view)

    first = backend.render(dict(request))
    assert backend._memo_bytes > 0
    assert backend.render(dict(request)) == first  # served from the memo...
    assert len(renders) == 1  # ...without rendering a second time

    # The alias spelling names the same figure, so it must neither render nor add an entry.
    entries = len(backend._memo)
    backend.render({**request, "controls": {"var1d": "iotaf"}})
    assert len(backend._memo) == entries
    assert len(renders) == 1

    _bump_mtime(path)
    assert backend.refresh({"sessionId": meta["sessionId"]})["changed"] is True
    assert backend._memo == {} and backend._memo_bytes == 0


def test_memo_budget_evicts_oldest(tmp_path):
    import vmecdash.vscode_backend as backend_module

    backend = VmecDashBackend()
    meta = backend.open({"path": _copy(tmp_path, "wout_budget.nc")})
    saved = backend_module.MEMO_BUDGET_BYTES
    try:
        backend_module.MEMO_BUDGET_BYTES = 1  # force eviction on every store
        for surface in (10, 20, 30):
            backend.render(
                {"sessionId": meta["sessionId"], "view": "3d", "theme": "dark", "controls": {"s3dIdx": surface}}
            )
        assert len(backend._memo) == 1  # the newest entry is always kept
    finally:
        backend_module.MEMO_BUDGET_BYTES = saved


# ----------------------------------------------------------------- refresh & refs


def test_refresh_reports_unchanged_and_returns_fresh_metadata(tmp_path):
    path = _copy(tmp_path, "wout_refresh.nc")
    backend = VmecDashBackend()
    meta = backend.open({"path": path})

    unchanged = backend.refresh({"sessionId": meta["sessionId"]})
    assert unchanged["changed"] is False
    assert unchanged["sessionId"] == meta["sessionId"]

    _bump_mtime(path)
    changed = backend.refresh({"sessionId": meta["sessionId"]})
    assert changed["changed"] is True
    assert changed["sessionId"] != meta["sessionId"]
    # Full metadata, not just a figure: the schema carries the resolved slider maxima.
    assert changed["schema"]["views"]
    assert backend.render({"sessionId": changed["sessionId"], "view": "1d", "controls": {}, "theme": "dark"})


def test_schema_slider_maxima_track_ns():
    vmec = VMECJaxProcessor.from_file(EXAMPLE_WOUT)
    schema = view_schema.build_ui_schema(vmec)
    three_d = next(view for view in schema["views"] if view["id"] == "3d")
    s_idx = next(control for control in three_d["controls"] if control["id"] == "s3dIdx")
    assert s_idx["max"] == vmec.ns - 1
    assert isinstance(s_idx["max"], int)


def test_two_holders_share_a_session_and_dispose_is_counted(tmp_path):
    path = _copy(tmp_path, "wout_shared.nc")
    backend = VmecDashBackend()
    first = backend.open({"path": path})
    second = backend.open({"path": path})
    assert first["sessionId"] == second["sessionId"]

    # One panel closing must not strand the other.
    assert backend.dispose({"sessionId": first["sessionId"]})["disposed"] is False
    assert backend.render({"sessionId": first["sessionId"], "view": "1d", "controls": {}, "theme": "dark"})

    assert backend.dispose({"sessionId": first["sessionId"]})["disposed"] is True
    assert backend.sessions == {}
    assert backend.dispose({"sessionId": first["sessionId"]})["disposed"] is False


def test_open_payload_serializes_without_nonfinite(tmp_path):
    """``_metadata`` sanitizes only ``scalars``; a NaN elsewhere would break stdio."""
    backend = VmecDashBackend()
    opened = backend.open({"path": _copy(tmp_path, "wout_json.nc")})
    json.dumps(opened, allow_nan=False)


def test_memo_holds_text_and_charges_its_exact_size(tmp_path):
    """The budget used to count JSON bytes while storing a parsed tree ~4.5x larger."""
    backend = VmecDashBackend()
    meta = backend.open({"path": _copy(tmp_path, "wout_text.nc")})
    for view, controls in (("1d", {"profile": "iotaf"}), ("2d", {"type2d": "cross_section", "var2d": "modB"})):
        backend.render({"sessionId": meta["sessionId"], "view": view, "controls": controls, "theme": "dark"})

    stored = [entry[0] for entry in backend._memo.values()]
    assert stored and all(isinstance(text, str) for text in stored)
    # Pure-ASCII text is one byte per character, so the charge is the real size.
    assert all(text.isascii() for text in stored)
    assert backend._memo_bytes == sum(len(text) for text in stored)


# ------------------------------------------------------------ files that change under us


def test_refresh_keeps_the_previous_data_when_the_new_file_is_unreadable(tmp_path):
    """A running VMEC rewrites its wout in place; a refresh can land half-way through."""
    path = _copy(tmp_path, "wout_halfwritten.nc")
    backend = VmecDashBackend()
    meta = backend.open({"path": path})
    request = {"sessionId": meta["sessionId"], "view": "2d", "controls": {"type2d": "cross_section", "var2d": "lambda"}, "theme": "dark"}

    Path(path).write_bytes(b"CDF\x01 half-written")  # truncated in place, same inode
    _bump_mtime(path)

    with pytest.raises(BackendError) as excinfo:
        backend.refresh({"sessionId": meta["sessionId"]})
    assert excinfo.value.code == "LOAD_FAILED"

    # The previous session survives, still owns the path, and - the part the message
    # promises - still renders, even for fields that are read after construction.
    assert meta["sessionId"] in backend.sessions
    assert backend.path_sessions[path] == meta["sessionId"]
    assert backend.render(dict(request))["figure"]["data"]


def test_open_keeps_a_stale_session_when_the_new_file_is_unreadable(tmp_path):
    path = _copy(tmp_path, "wout_stale.nc")
    backend = VmecDashBackend()
    meta = backend.open({"path": path})

    Path(path).write_bytes(b"not a netcdf file")
    _bump_mtime(path)

    with pytest.raises(BackendError) as excinfo:
        backend.open({"path": path})
    assert excinfo.value.code == "LOAD_FAILED"
    # The panel that already had the file open is left working.
    assert meta["sessionId"] in backend.sessions
    assert backend.render({"sessionId": meta["sessionId"], "view": "1d", "controls": {}, "theme": "dark"})


def test_refresh_with_two_panels_releases_the_new_session_once_both_close(tmp_path):
    """refresh() used to copy the old refcount, so the new session was never released."""
    path = _copy(tmp_path, "wout_two_panels.nc")
    backend = VmecDashBackend()
    panel_a = backend.open({"path": path})["sessionId"]
    panel_b = backend.open({"path": path})["sessionId"]
    assert panel_a == panel_b

    _bump_mtime(path)
    new_id = backend.refresh({"sessionId": panel_a})["sessionId"]
    assert new_id != panel_a

    # Panel B still holds the retired id; the extension answers SESSION_NOT_FOUND by
    # re-opening, which is how B registers on the new session.
    with pytest.raises(BackendError) as excinfo:
        backend.render({"sessionId": panel_b, "view": "1d", "controls": {}, "theme": "dark"})
    assert excinfo.value.code == "SESSION_NOT_FOUND"
    assert backend.open({"path": path})["sessionId"] == new_id

    backend.dispose({"sessionId": new_id})  # panel A closes
    assert new_id in backend.sessions
    assert backend.dispose({"sessionId": new_id})["disposed"] is True  # panel B closes
    assert backend.sessions == {}

