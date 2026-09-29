import builtins
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_WOUT = str(ROOT / "example" / "wout_PO.nc")


def test_renderers_import_without_dash(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "dash" or name.startswith("dash_") or name == "dash_mantine_components":
            raise AssertionError(f"Dash import leaked into backend path: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    for name in list(sys.modules):
        if name.startswith("vmecdash.renderers") or name == "vmecdash.vscode_backend":
            sys.modules.pop(name, None)
    __import__("vmecdash.renderers.profiles")
    __import__("vmecdash.renderers.two_d")
    __import__("vmecdash.renderers.three_d")
    __import__("vmecdash.renderers.fieldline")
    __import__("vmecdash.vscode_backend")


def test_backend_open_and_render_views():
    from vmecdash.vscode_backend import VmecDashBackend

    backend = VmecDashBackend()
    opened = backend.open({"path": EXAMPLE_WOUT})
    session_id = opened["sessionId"]

    requests = [
        ("overview", {}),
        ("1d", {"profile": "iotaf"}),
        ("2d", {"type2d": "cross_section", "var2d": "geometry", "phi": 0.0}),
        ("2d", {"type2d": "flux_surface", "var2d": "modB", "sIdx": opened["ns"] - 1}),
        ("3d", {"var3d": "modB", "s3dIdx": opened["ns"] - 1, "coordFree": True}),
        ("fieldline", {"fieldlineType": "1d_lines", "fieldlineSIdx": opened["ns"] - 1, "fieldlineRes": 64}),
    ]
    for view, controls in requests:
        rendered = backend.render({"sessionId": session_id, "view": view, "controls": controls, "theme": "dark"})
        assert "data" in rendered["figure"]
        assert "layout" in rendered["figure"]


def test_json_sanitize_rejects_bare_nan_tokens():
    from vmecdash.vscode_backend import sanitize

    payload = sanitize({"bad": float("nan"), "ok": 1.0})
    encoded = json.dumps(payload, allow_nan=False)
    assert "NaN" not in encoded
    assert json.loads(encoded) == {"bad": None, "ok": 1.0}


def test_backend_stdio_health_stdout_clean():
    proc = subprocess.run(
        [sys.executable, "-m", "vmecdash.vscode_backend", "--stdio"],
        input='{"id":1,"method":"health","params":{}}\n',
        text=True,
        capture_output=True,
        timeout=30,
        check=True,
    )
    stdout_lines = [line for line in proc.stdout.splitlines() if line.strip()]
    assert len(stdout_lines) == 1
    response = json.loads(stdout_lines[0])
    assert response["id"] == 1
    assert response["result"]["ok"] is True


def test_figure_serialization_round_trips_without_nan():
    from vmecdash.vscode_backend import VmecDashBackend

    backend = VmecDashBackend()
    opened = backend.open({"path": EXAMPLE_WOUT})
    rendered = backend.render(
        {
            "sessionId": opened["sessionId"],
            "view": "2d",
            "controls": {"type2d": "cross_section", "var2d": "lambda", "phi": 0.0},
            "theme": "dark",
        }
    )
    encoded = json.dumps(rendered, allow_nan=False)
    assert "NaN" not in encoded
    assert "Infinity" not in encoded
    assert json.loads(encoded)["figure"]["data"]


def _reject_bare_constant(token):
    raise AssertionError(f"bare {token} on the wire - JSON has no such literal")


def test_stdio_splices_memoised_render_text():
    """A render is serialized once and written as-is; a memo hit is byte-identical and
    carries exactly what the in-process API returns."""
    import os

    from vmecdash.vscode_backend import VmecDashBackend, _session_id

    path = os.path.abspath(EXAMPLE_WOUT)
    session_id = _session_id(path, os.path.getmtime(path))
    # Lambda's cross-section carries non-finite values at the axis: this exercises the
    # sanitizing that used to run on every response and now runs once, before storage.
    render = {"sessionId": session_id, "view": "2d", "controls": {"type2d": "cross_section", "var2d": "lambda"}, "theme": "dark"}
    requests = [
        {"id": 1, "method": "open", "params": {"path": path}},
        {"id": 2, "method": "render", "params": render},
        {"id": 3, "method": "render", "params": render},
    ]
    proc = subprocess.run(
        [sys.executable, "-m", "vmecdash.vscode_backend", "--stdio"],
        input="".join(json.dumps(request) + "\n" for request in requests),
        text=True,
        capture_output=True,
        timeout=180,
        check=True,
    )
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    assert len(lines) == 3
    responses = [json.loads(line, parse_constant=_reject_bare_constant) for line in lines]
    assert [response["id"] for response in responses] == [1, 2, 3]
    assert responses[0]["result"]["sessionId"] == session_id

    assert lines[1].split('"result":', 1)[1] == lines[2].split('"result":', 1)[1]
    backend = VmecDashBackend()
    backend.open({"path": path})
    assert responses[1]["result"] == backend.render(render)

