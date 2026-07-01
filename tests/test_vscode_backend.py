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
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
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
