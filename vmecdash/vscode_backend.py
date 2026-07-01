from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import math
import os
import sys
import traceback
from dataclasses import dataclass
from typing import Any

import numpy as np
import plotly
import plotly.io as pio

import vmecdash
from vmecdash import stats as vmec_stats
from vmecdash import view_schema
from vmecdash.core import VMECJaxProcessor
from vmecdash.theme import build_theme, make_empty_figure

LOGGER = logging.getLogger("vmecdash.vscode_backend")


@dataclass
class Session:
    session_id: str
    path: str
    mtime: float
    vmec: Any


class BackendError(Exception):
    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


class VmecDashBackend:
    def __init__(self):
        self.sessions: dict[str, Session] = {}
        self.path_sessions: dict[str, str] = {}

    def health(self, _params: dict[str, Any] | None = None) -> dict[str, Any]:
        import jax

        return {
            "ok": True,
            "protocol": 1,
            "vmecdashVersion": vmecdash.__version__,
            "jaxVersion": jax.__version__,
            "plotlyPythonVersion": plotly.__version__,
            "python": sys.executable,
            "features": list(view_schema.VIEWS) + ["exportReport"],
        }

    def open(self, params: dict[str, Any]) -> dict[str, Any]:
        path = os.path.abspath(params.get("path") or "")
        if not path:
            raise BackendError("INVALID_REQUEST", "Missing required path")
        if not os.path.exists(path):
            raise BackendError("FILE_NOT_FOUND", f"File not found: {path}", {"path": path})

        mtime = os.path.getmtime(path)
        existing_id = self.path_sessions.get(path)
        if existing_id:
            existing = self.sessions.get(existing_id)
            if existing and existing.mtime == mtime:
                return self._metadata(existing)
            self.sessions.pop(existing_id, None)
            self.path_sessions.pop(path, None)

        vmec = VMECJaxProcessor.from_file(path)
        session_id = _session_id(path, mtime)
        session = Session(session_id=session_id, path=path, mtime=mtime, vmec=vmec)
        self.sessions[session_id] = session
        self.path_sessions[path] = session_id
        return self._metadata(session)

    def render(self, params: dict[str, Any]) -> dict[str, Any]:
        session = self._session(params.get("sessionId"))
        controls = params.get("controls") or {}
        theme = _theme_from_param(params.get("theme"))
        view = params.get("view") or "overview"
        vmec = session.vmec
        scalars = vmec.get_scalars()
        stats_payload = vmec_stats.overview_stats(vmec, scalars)
        field_map = {opt["value"]: opt.get("label", opt["value"]) for opt in vmec.available_fields()}

        try:
            if view in view_schema.VIEWS:
                fig = view_schema.render_view(view, vmec, controls, theme, field_map)
                stats_mode = view_schema.VIEWS[view].stats
            else:
                fig = make_empty_figure(theme, f"Unknown view: {view}")
                stats_mode = "base"
        except Exception as exc:
            LOGGER.exception("Render failed")
            raise BackendError("RENDER_FAILED", str(exc), {"view": view})

        stats_payload_out = stats_payload if stats_mode == "full" else {"base": stats_payload["base"]}
        return {"figure": figure_to_jsonable(fig), "stats": stats_payload_out}

    def export_report(self, params: dict[str, Any]) -> dict[str, Any]:
        session = self._session(params.get("sessionId"))
        vmec = session.vmec
        scalars = vmec.get_scalars()
        summary_lines = vmec.get_summary_lines()
        report_lines = ["VMEC Report", "==============", ""]
        for key, value in scalars.items():
            report_lines.append(f"{key}: {value}")
        report_lines.append("")
        report_lines.append("Summary")
        report_lines.extend(summary_lines)
        return {"content": "\n".join(report_lines), "filename": "vmec_report.txt"}

    def dispose(self, params: dict[str, Any]) -> dict[str, Any]:
        session_id = params.get("sessionId")
        session = self.sessions.pop(session_id, None)
        if session:
            self.path_sessions.pop(session.path, None)
        return {"disposed": bool(session)}

    def dispatch(self, method: str, params: dict[str, Any] | None) -> dict[str, Any]:
        params = params or {}
        if method == "health":
            return self.health(params)
        if method == "open":
            return self.open(params)
        if method == "render":
            return self.render(params)
        if method == "exportReport":
            return self.export_report(params)
        if method == "dispose":
            return self.dispose(params)
        raise BackendError("METHOD_NOT_FOUND", f"Unknown method: {method}", {"method": method})

    def _session(self, session_id: str | None) -> Session:
        if not session_id or session_id not in self.sessions:
            raise BackendError("SESSION_NOT_FOUND", "Session not found", {"sessionId": session_id})
        return self.sessions[session_id]

    def _metadata(self, session: Session) -> dict[str, Any]:
        vmec = session.vmec
        return {
            "sessionId": session.session_id,
            "path": session.path,
            "mtime": session.mtime,
            "ns": vmec.ns,
            "nfp": vmec.nfp,
            "profiles": vmec.available_profiles(),
            "computedProfiles": vmec.available_computed_profiles(),
            "fields": vmec.available_fields(),
            "schema": view_schema.build_ui_schema(vmec),
            "scalars": sanitize(vmec.get_scalars()),
            "summaryLines": vmec.get_summary_lines(),
        }


def figure_to_jsonable(fig) -> dict[str, Any]:
    return sanitize(json.loads(pio.to_json(fig, validate=False)))


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    if isinstance(value, np.ndarray):
        return sanitize(value.tolist())
    if isinstance(value, np.generic):
        return sanitize(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _session_id(path: str, mtime: float) -> str:
    digest = hashlib.sha1(f"{path}:{mtime}".encode("utf-8")).hexdigest()
    return digest[:16]


def _theme_from_param(theme_param: Any):
    if isinstance(theme_param, dict):
        dark = bool(theme_param.get("dark", True))
    elif isinstance(theme_param, str):
        dark = theme_param != "light"
    else:
        dark = bool(theme_param) if theme_param is not None else True
    return build_theme(dark, 0)


def _error_response(request_id: Any, exc: Exception) -> dict[str, Any]:
    if isinstance(exc, BackendError):
        return {"id": request_id, "error": {"code": exc.code, "message": exc.message, "details": sanitize(exc.details)}}
    return {
        "id": request_id,
        "error": {
            "code": "INTERNAL_ERROR",
            "message": str(exc),
            "details": {"traceback": traceback.format_exc()},
        },
    }


def serve_stdio() -> int:
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    backend = VmecDashBackend()
    protocol_stdout = sys.stdout
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request_id = None
        try:
            request = json.loads(line)
            request_id = request.get("id")
            method = request.get("method")
            if not method:
                raise BackendError("INVALID_REQUEST", "Missing method")
            with contextlib.redirect_stdout(sys.stderr):
                result = backend.dispatch(method, request.get("params"))
            response = {"id": request_id, "result": sanitize(result)}
        except Exception as exc:
            LOGGER.exception("Request failed")
            response = _error_response(request_id, exc)
        protocol_stdout.write(json.dumps(response, allow_nan=False, separators=(",", ":")) + "\n")
        protocol_stdout.flush()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stdio", action="store_true", help="Run newline-delimited JSON-RPC over stdio")
    args = parser.parse_args(argv)
    if args.stdio:
        return serve_stdio()
    parser.print_help(sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
