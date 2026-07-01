"""Single source of truth for VMECdash views and their controls.

Each :class:`ViewSpec` declares one view (its label, nav icon, stat mode, and the
ordered list of :class:`Control` widgets it needs) together with a ``render`` callable.
This one registry drives three things that used to be hardcoded separately:

* ``build_ui_schema(vmec)`` produces a JSON-able description shipped to the Webview in
  the ``open`` metadata, so the frontend renders nav + controls generically (no per-view
  TypeScript/JS).
* ``render_view(...)`` dispatches a render request to the right view — replacing the
  if/elif chain that used to live in ``vscode_backend.render``.
* Control ``default`` values live here once and are consumed by both the Webview (via the
  schema) and the render functions (via :func:`cv`), so defaults never drift.

The module is pure: it imports only the (dash-free) renderers, numpy, and plotly types.
The per-view render bodies are transcribed verbatim from the previous
``vscode_backend.render`` branches — the numerics are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from vmecdash.renderers import fieldline, profiles, three_d, two_d
from vmecdash.renderers.overview import overview_figure


@dataclass(frozen=True)
class Control:
    """A single control widget declared once for both the UI and the backend."""

    id: str
    kind: str  # "select" | "slider" | "number" | "checkbox"
    label: str
    default: Any
    options: list[dict[str, str]] | None = None  # inline enum options [{value,label}]
    options_from: str | None = None  # "fields" | "profiles" -> resolved by the Webview from meta
    min: Any = None  # numbers/sliders; may be a dynamic token like "ns-1"
    max: Any = None
    step: Any = None
    unit: str | None = None  # optional suffix shown next to a slider's live value


@dataclass(frozen=True)
class ViewSpec:
    id: str
    label: str
    icon: str  # key into the Webview SVG icon map
    controls: list[Control]
    stats: str  # "full" (overview dashboard) | "base"
    render: Callable[..., Any]  # (vmec, controls, theme, field_map) -> go.Figure


# --------------------------------------------------------------------------------------
# Helpers: dynamic-token resolution + one-home default lookup
# --------------------------------------------------------------------------------------

def _resolve_token(value: Any, vmec: Any) -> Any:
    """Resolve dynamic schema tokens (e.g. flux-surface maxima) against the equilibrium."""
    if isinstance(value, str):
        if value == "ns-1":
            return vmec.ns - 1
        if value == "max(1,ns-1)":
            return max(1, vmec.ns - 1)
    return value


def _control(view_id: str, ctrl_id: str) -> Control:
    for control in VIEWS[view_id].controls:
        if control.id == ctrl_id:
            return control
    raise KeyError(f"Unknown control {ctrl_id!r} for view {view_id!r}")


def cv(controls: dict[str, Any], view_id: str, ctrl_id: str, vmec: Any) -> Any:
    """Return a control's value, falling back to its (single-source) schema default."""
    spec = _control(view_id, ctrl_id)
    value = controls.get(ctrl_id)
    if value is None:
        value = spec.default
    return _resolve_token(value, vmec)


# --------------------------------------------------------------------------------------
# Per-view render functions (bodies transcribed verbatim from the old dispatch)
# --------------------------------------------------------------------------------------

def overview_view(vmec, controls, theme, field_map):
    return overview_figure(vmec, theme)


def oned_view(vmec, controls, theme, field_map):
    var = controls.get("profile") or controls.get("var1d") or _control("1d", "profile").default
    return profiles.render_profile(vmec, var, theme)


def twod_view(vmec, controls, theme, field_map):
    type_2d = controls.get("type2d") or controls.get("type") or _control("2d", "type2d").default
    var_2d = controls.get("var2d") or controls.get("var") or _control("2d", "var2d").default
    phi_frac = float(cv(controls, "2d", "phi", vmec) or 0.0)
    phi_angle = phi_frac * (2 * np.pi / max(vmec.nfp, 1))
    s_idx = int(cv(controls, "2d", "sIdx", vmec))
    field_label = field_map.get(var_2d, var_2d)
    if type_2d == "cross_section":
        if var_2d == "geometry":
            return two_d.build_geometry_cross_section_figure(
                vmec,
                phi_angle,
                s_idx,
                cv(controls, "2d", "geoCount", vmec),
                theme.dark_mode,
                theme.fig_template,
                theme.paper_bg,
                theme.plot_bg,
                theme.reset_seed,
            )
        return two_d.render_cross_section_field(vmec, phi_angle, var_2d, field_label, theme)
    return two_d.render_flux_surface(vmec, s_idx, var_2d, field_label, theme)


def threed_view(vmec, controls, theme, field_map):
    s_val = int(cv(controls, "3d", "s3dIdx", vmec))
    var_3d = controls.get("var3d") or _control("3d", "var3d").default
    coord_free = bool(cv(controls, "3d", "coordFree", vmec))
    return three_d.render_3d(vmec, s_val, var_3d, field_map.get(var_3d, var_3d), coord_free, theme)


def fieldline_view(vmec, controls, theme, field_map):
    fl_type = controls.get("fieldlineType") or _control("fieldline", "fieldlineType").default
    s_idx = int(cv(controls, "fieldline", "fieldlineSIdx", vmec))
    res = int(cv(controls, "fieldline", "fieldlineRes", vmec) or 128)
    shift = bool(cv(controls, "fieldline", "fieldlineZetaShift", vmec))
    if fl_type == "1d_lines":
        n_lines = int(cv(controls, "fieldline", "fieldlineNLines", vmec) or 6)
        return fieldline.render_fieldline_lines(vmec, s_idx, n_lines, res, theme, shift_zeta=shift)
    if fl_type == "single_trace":
        transits = int(cv(controls, "fieldline", "fieldlineTransits", vmec) or 50)
        alpha0 = float(cv(controls, "fieldline", "fieldlineAlpha0", vmec) or 0.0)
        return fieldline.render_single_trace(vmec, s_idx, transits, alpha0, res, theme)
    return fieldline.render_fieldline_heatmap(vmec, s_idx, res, theme, shift_zeta=shift)


# --------------------------------------------------------------------------------------
# The registry — add a view or a control here and it flows to both UI and backend
# --------------------------------------------------------------------------------------

_TYPE_2D_OPTIONS = [
    {"value": "cross_section", "label": "Cross Section (R-Z)"},
    {"value": "flux_surface", "label": "Flux Surface (θ-ζ)"},
]
_FIELDLINE_TYPE_OPTIONS = [
    {"value": "2d_modB", "label": "2D |B|"},
    {"value": "1d_lines", "label": "1D Lines"},
    {"value": "single_trace", "label": "Single Trace"},
]
_FIELDLINE_RES_OPTIONS = [{"value": r, "label": r} for r in ("64", "128", "256", "512")]


VIEWS: dict[str, ViewSpec] = {
    "overview": ViewSpec(
        id="overview",
        label="Summary Dashboard",
        icon="overview",
        stats="full",
        controls=[],
        render=overview_view,
    ),
    "1d": ViewSpec(
        id="1d",
        label="1D Profiles",
        icon="profiles",
        stats="base",
        controls=[
            Control("profile", "select", "Profile Variable", default="iotaf", options_from="profiles"),
        ],
        render=oned_view,
    ),
    "2d": ViewSpec(
        id="2d",
        label="2D Cross-Section",
        icon="twod",
        stats="base",
        controls=[
            Control("type2d", "select", "Plot Type", default="cross_section", options=_TYPE_2D_OPTIONS),
            Control("var2d", "select", "Color Variable", default="geometry", options_from="fields"),
            Control("phi", "slider", "Toroidal Angle", default=0.0, min=0, max=1, step=0.01),
            Control("sIdx", "slider", "Flux Surface Index", default="ns-1", min=0, max="ns-1", step=1),
            Control("geoCount", "number", "Visible Geometry Surfaces", default=15, min=1, max=200, step=1),
        ],
        render=twod_view,
    ),
    "3d": ViewSpec(
        id="3d",
        label="3D Geometry",
        icon="threed",
        stats="base",
        controls=[
            Control("var3d", "select", "Surface Color", default="modB", options_from="fields"),
            Control("s3dIdx", "slider", "Flux Surface Index", default="ns-1", min=0, max="ns-1", step=1),
            Control("coordFree", "checkbox", "Coordinate-free Background", default=True),
        ],
        render=threed_view,
    ),
    "fieldline": ViewSpec(
        id="fieldline",
        label="Field Lines",
        icon="fieldline",
        stats="base",
        controls=[
            Control("fieldlineType", "select", "Plot Type", default="2d_modB", options=_FIELDLINE_TYPE_OPTIONS),
            Control("fieldlineSIdx", "slider", "Flux Surface Index", default="max(1,ns-1)", min=1, max="ns-1", step=1),
            Control("fieldlineNLines", "number", "Number of Lines", default=6, min=1, max=20, step=1),
            Control("fieldlineTransits", "number", "Transits", default=50, min=10, max=500, step=10),
            Control("fieldlineAlpha0", "number", "Alpha Start", default=0.0, step=0.1),
            Control("fieldlineRes", "select", "Resolution", default="128", options=_FIELDLINE_RES_OPTIONS),
            Control("fieldlineZetaShift", "checkbox", "Shift ζ by π/NFP", default=False),
        ],
        render=fieldline_view,
    ),
}


def build_ui_schema(vmec: Any) -> dict[str, Any]:
    """Describe every view + control for the Webview, with dynamic tokens resolved."""
    views = []
    for spec in VIEWS.values():
        controls = []
        for control in spec.controls:
            controls.append(
                {
                    "id": control.id,
                    "kind": control.kind,
                    "label": control.label,
                    "default": _resolve_token(control.default, vmec),
                    "options": control.options,
                    "optionsFrom": control.options_from,
                    "min": _resolve_token(control.min, vmec),
                    "max": _resolve_token(control.max, vmec),
                    "step": control.step,
                    "unit": control.unit,
                }
            )
        views.append(
            {
                "id": spec.id,
                "label": spec.label,
                "icon": spec.icon,
                "stats": spec.stats,
                "controls": controls,
            }
        )
    return {"views": views}


def render_view(view_id: str, vmec: Any, controls: dict[str, Any] | None, theme: Any, field_map: dict[str, str]):
    """Dispatch a render request to the view's render function."""
    spec = VIEWS[view_id]
    return spec.render(vmec, controls or {}, theme, field_map)
