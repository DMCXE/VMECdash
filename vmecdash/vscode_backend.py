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
from collections import OrderedDict
from dataclasses import dataclass, field
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

# Significant digits kept when serializing figure arrays. 7 matches float32 precision,
# which is far finer than any pixel a plot can resolve.
FIGURE_SIG_DIGITS = 7
# Total size of memoized render payloads. A single field-line figure can be ~5 MB, so the
# memo is bounded by bytes rather than entry count.
MEMO_BUDGET_BYTES = 64 * 1024 * 1024


@dataclass
class Session:
    session_id: str
    path: str
    mtime: float
    vmec: Any
    # How many editors hold this session. ``open`` returns the existing session when the
    # same file is opened twice, so without a count the first ``dispose`` would strand
    # the other holder with SESSION_NOT_FOUND.
    refs: int = field(default=1)


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
        # Memoized render payloads, newest last. Keyed by session so a refresh can drop
        # exactly the entries that went stale.
        self._memo: OrderedDict[tuple, tuple[str, int]] = OrderedDict()
        self._memo_bytes = 0

    def health(self, _params: dict[str, Any] | None = None) -> dict[str, Any]:
        import jax

        return {
            "ok": True,
            "protocol": 1,
            "vmecdashVersion": vmecdash.__version__,
            "jaxVersion": jax.__version__,
            "plotlyPythonVersion": plotly.__version__,
            "python": sys.executable,
            "features": list(view_schema.VIEWS) + ["exportReport", "refresh"],
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
                existing.refs += 1
                return self._metadata(existing)

        # Load before retiring anything: if the file is mid-write the load fails, and the
        # panels already holding the stale session must keep working.
        fresh = self._load_or_raise(path, mtime, "Could not read the file")
        if existing_id and existing_id != fresh.session_id:
            # The file moved on underneath us; retire the stale session entirely.
            self._drop_session(existing_id)
        return self._metadata(fresh)

    def refresh(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Re-read the session's file if it changed on disk.

        Returns full metadata on change, not just a figure: a re-run VMEC can change
        ``ns``, and the UI schema bakes the resolved slider maxima into that payload, so
        stale metadata would leave sliders addressing flux surfaces that no longer exist.
        """
        session = self._session(params.get("sessionId"))
        path = session.path
        if not os.path.exists(path):
            raise BackendError("FILE_NOT_FOUND", f"File not found: {path}", {"path": path})

        mtime = os.path.getmtime(path)
        if mtime == session.mtime:
            return {"changed": False, "sessionId": session.session_id, "mtime": mtime}

        # Load first and retire the old session only once the new one exists. A running
        # VMEC rewrites its output in place, so a refresh can land on a half-written file;
        # dropping first would leave every holder with no session at all.
        fresh = self._load_or_raise(
            path,
            mtime,
            "Could not read the updated file - it may still be being written. The previous data is still shown",
        )
        # The new session starts at one reference: only the panel asking holds its id.
        # Other holders still carry the old id; their next call gets SESSION_NOT_FOUND and
        # they re-register through open(), which counts them. Copying the old count here
        # counted them twice, so the session was never released.
        self._drop_session(session.session_id)
        metadata = self._metadata(fresh)
        metadata["changed"] = True
        return metadata

    def render(self, params: dict[str, Any]) -> dict[str, Any]:
        """Render a view and return it as a plain dict - the in-process API."""
        return json.loads(self.render_json(params))

    def render_json(self, params: dict[str, Any]) -> _PreSerialized:
        """Render a view as sanitized JSON text, memoised.

        The memo holds text rather than the parsed figure. A parsed figure occupies about
        4.5x its JSON size as Python objects, so a budget counted in JSON bytes admitted
        several times the memory it claimed; and a stored tree still had to be sanitized
        and re-serialized on every hit. Text is charged exactly, and ``serve_stdio``
        writes it out as-is.
        """
        session = self._session(params.get("sessionId"))
        controls = params.get("controls") or {}
        theme = _theme_from_param(params.get("theme"))
        view = params.get("view") or "overview"
        vmec = session.vmec

        key = None
        if view in view_schema.VIEWS:
            key = (
                session.session_id,
                view,
                view_schema.canonical_controls(view, controls, vmec),
                theme.dark_mode,
            )
            cached = self._memo.get(key)
            if cached is not None:
                self._memo.move_to_end(key)
                return cached[0]

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
            raise BackendError("RENDER_FAILED", str(exc), {"view": view}) from exc

        figure = _figure_payload(fig)[0]
        stats_payload_out = stats_payload if stats_mode == "full" else {"base": stats_payload["base"]}
        text = _PreSerialized(
            json.dumps(
                {"figure": figure, "stats": sanitize(stats_payload_out)},
                allow_nan=False,
                separators=(",", ":"),
            )
        )
        if key is not None:
            # ensure_ascii (the default) makes the text pure ASCII: one byte per character.
            self._memo_store(key, text, len(text))
        return text

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
        session = self.sessions.get(session_id)
        if session is None:
            return {"disposed": False}
        session.refs -= 1
        if session.refs > 0:
            return {"disposed": False, "remaining": session.refs}
        self._drop_session(session_id)
        return {"disposed": True}

    def dispatch(self, method: str, params: dict[str, Any] | None) -> dict[str, Any]:
        params = params or {}
        if method == "health":
            return self.health(params)
        if method == "open":
            return self.open(params)
        if method == "refresh":
            return self.refresh(params)
        if method == "render":
            return self.render_json(params)
        if method == "exportReport":
            return self.export_report(params)
        if method == "dispose":
            return self.dispose(params)
        raise BackendError("METHOD_NOT_FOUND", f"Unknown method: {method}", {"method": method})

    # ---- Session and memo bookkeeping -------------------------------------

    def _load_or_raise(self, path: str, mtime: float, message: str) -> Session:
        """Load a session, turning a read failure into an actionable LOAD_FAILED."""
        try:
            return self._load_session(path, mtime)
        except BackendError:
            raise
        except Exception as exc:
            raise BackendError("LOAD_FAILED", f"{message}. ({exc})", {"path": path}) from exc

    def _load_session(self, path: str, mtime: float) -> Session:
        vmec = VMECJaxProcessor.from_file(path)
        session = Session(session_id=_session_id(path, mtime), path=path, mtime=mtime, vmec=vmec)
        self.sessions[session.session_id] = session
        self.path_sessions[path] = session.session_id
        return session

    def _drop_session(self, session_id: str) -> None:
        """Tear a session down: forget its memo entries and release its processor."""
        session = self.sessions.pop(session_id, None)
        if session is None:
            return
        if self.path_sessions.get(session.path) == session_id:
            self.path_sessions.pop(session.path, None)
        self._memo_drop(session_id)
        session.vmec.close()

    def _memo_store(self, key: tuple, result: str, nbytes: int) -> None:
        self._memo[key] = (result, nbytes)
        self._memo_bytes += nbytes
        # Keep the entry we just stored even if it alone exceeds the budget; a single
        # oversized figure is still worth not recomputing.
        while self._memo_bytes > MEMO_BUDGET_BYTES and len(self._memo) > 1:
            _, (_, evicted) = self._memo.popitem(last=False)
            self._memo_bytes -= evicted

    def _memo_drop(self, session_id: str) -> None:
        for key in [k for k in self._memo if k[0] == session_id]:
            self._memo_bytes -= self._memo.pop(key)[1]

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


# Trace attributes that carry bulk float data worth rounding before serialization.
_SHRINK_ATTRS = ("x", "y", "z", "a", "b", "value", "surfacecolor")


def _round_sig(values: np.ndarray, sig: int) -> np.ndarray:
    """Round to ``sig`` significant digits, elementwise and vectorized."""
    out = np.array(values, dtype=np.float64, copy=True)
    mask = np.isfinite(out) & (out != 0.0)
    if mask.any():
        # Significant digits, not decimals: this tool plots pressures near 1e5 Pa beside
        # lambda values near 1e-9, and a fixed decimal count would wreck one or the other.
        scale = np.power(10.0, sig - 1 - np.floor(np.log10(np.abs(out[mask]))))
        out[mask] = np.round(out[mask] * scale) / scale
    return out


def _shrink_figure(fig, sig: int = FIGURE_SIG_DIGITS):
    """
    Round a figure's float arrays in place before it is serialized.

    Plotly writes every float with repr(), so full float64 coordinates cost up to 17
    characters each. At 7 significant digits - float32 precision, finer than any pixel a
    plot can resolve - payloads roughly halve and serializing gets *faster*, because
    there are fewer characters to emit and re-parse. Casting to float32 would do the
    opposite: repr(float(np.float32(0.1234567890123))) is '0.12345679104328156', longer
    than the float64 form.

    Only existing arrays are rewritten, never added or reordered, so trace structure is
    untouched.
    """
    for trace in fig.data:
        for attr in _SHRINK_ATTRS:
            value = getattr(trace, attr, None)
            if isinstance(value, np.ndarray) and value.dtype.kind == "f":
                trace[attr] = _round_sig(value, sig)
        marker = getattr(trace, "marker", None)
        color = getattr(marker, "color", None)
        if isinstance(color, np.ndarray) and color.dtype.kind == "f":
            marker.color = _round_sig(color, sig)
    return fig


def _figure_payload(fig) -> tuple[dict[str, Any], int]:
    """Return the JSON-able figure plus the byte size of its serialized form."""
    raw = pio.to_json(_shrink_figure(fig), validate=False)
    return sanitize(json.loads(raw)), len(raw)


def figure_to_jsonable(fig) -> dict[str, Any]:
    return _figure_payload(fig)[0]


class _PreSerialized(str):
    """A result that is already sanitized JSON text; ``serve_stdio`` splices it verbatim."""


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
    digest = hashlib.sha1(f"{path}:{mtime}".encode()).hexdigest()
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
            if isinstance(result, _PreSerialized):
                # Already sanitized JSON: splice it rather than parse, walk and re-encode
                # a multi-megabyte figure on every request, cache hits included.
                line_out = '{"id":' + json.dumps(request_id) + ',"result":' + result + "}"
            else:
                line_out = json.dumps(
                    {"id": request_id, "result": sanitize(result)}, allow_nan=False, separators=(",", ":")
                )
        except Exception as exc:
            LOGGER.exception("Request failed")
            line_out = json.dumps(_error_response(request_id, exc), allow_nan=False, separators=(",", ":"))
        protocol_stdout.write(line_out + "\n")
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
