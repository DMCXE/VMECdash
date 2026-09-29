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

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from vmecdash.renderers import fieldline, profiles, three_d, two_d
from vmecdash.renderers.overview import overview_figure
from vmecdash.theme import COLORMAP_OPTIONS


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
    visible_when: dict[str, Any] | None = None  # {sibling_ctrl_id: value_or_list}; AND across keys, list = any-of
    requires: str | None = None  # equilibrium predicate; the control is dropped when it fails


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


#: Equilibrium-level gates. ``visible_when`` can only test sibling control values; these
#: test the equilibrium itself, and are resolved once in build_ui_schema so an option that
#: would be degenerate never reaches the UI at all.
AVAILABILITY: dict[str, Callable[[Any], bool]] = {
    # Every toroidal cross-section of an axisymmetric device is identical, so a plot that
    # exists to compare them has nothing to show. VMECplot gates its LPK plot the same way.
    "nonaxisymmetric": lambda vmec: int(getattr(vmec, "ntor", 0)) > 0,
    "asymmetric": lambda vmec: bool(getattr(vmec, "lasym", False)),
}


def _available(requires: str | None, vmec: Any) -> bool:
    if not requires:
        return True
    predicate = AVAILABILITY.get(requires)
    return True if predicate is None else bool(predicate(vmec))


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


# Older spellings the render functions still accept, mapped onto the declared control id.
_CONTROL_ALIASES: dict[str, tuple[str, ...]] = {
    "profile": ("var1d",),
    "type2d": ("type",),
    "var2d": ("var",),
}


def effective_control(controls: dict[str, Any], view_id: str, ctrl_id: str, vmec: Any) -> Any:
    """Resolve one control the way the view's render function actually reads it."""
    spec = _control(view_id, ctrl_id)
    value = controls.get(ctrl_id)
    # Only "missing" falls through to an alias or the default. A slider sitting at 0 and
    # an unchecked toggle are real values, so test against None/"" rather than falsiness.
    if value is None or value == "":
        for alias in _CONTROL_ALIASES.get(ctrl_id, ()):
            alt = controls.get(alias)
            if alt is not None and alt != "":
                value = alt
                break
    if value is None or value == "":
        value = spec.default
    return _resolve_token(value, vmec)


def _hashable(value: Any) -> Any:
    """Collapse values that render identically onto one key (e.g. "128" and 128)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def canonical_controls(view_id: str, controls: dict[str, Any] | None, vmec: Any) -> tuple:
    """
    Return a hashable, fully-resolved control tuple for a view.

    Two requests that render the same figure produce the same tuple: absent controls are
    filled from the schema defaults, dynamic tokens like ``"ns-1"`` are resolved against
    the equilibrium, alias spellings collapse onto the declared id, and numeric strings
    collapse onto numbers. Controls the view does not declare are ignored, since they
    cannot affect its output. This is what makes the backend's figure memo safe to key on.
    """
    spec = VIEWS[view_id]
    controls = controls or {}
    return tuple(
        (control.id, _hashable(effective_control(controls, view_id, control.id, vmec)))
        for control in spec.controls
    )


# --------------------------------------------------------------------------------------
# Per-view render functions (bodies transcribed verbatim from the old dispatch)
# --------------------------------------------------------------------------------------

def overview_view(vmec, controls, theme, field_map):
    return overview_figure(vmec, theme)


def oned_view(vmec, controls, theme, field_map):
    var = controls.get("profile") or controls.get("var1d") or _control("1d", "profile").default
    return profiles.render_profile(vmec, var, theme)


def twod_view(vmec, controls, theme, field_map):
    type_2d = effective_control(controls, "2d", "type2d", vmec)
    var_2d = effective_control(controls, "2d", "var2d", vmec)
    s_idx = int(cv(controls, "2d", "sIdx", vmec))
    colormap = cv(controls, "2d", "colormap", vmec)
    field_label = field_map.get(var_2d, var_2d)
    res_2d = cv(controls, "2d", "res2d", vmec)
    res_u = None if res_2d in (None, "", "auto") else int(res_2d)
    style = cv(controls, "2d", "contourStyle", vmec)
    lines = bool(cv(controls, "2d", "contourLines", vmec))

    # LPK is a viewpoint on the cross-section, not a separate plot type, so it is only
    # consulted here - a stale True cannot leak into the flux-surface view.
    lpk_on = bool(cv(controls, "2d", "lpkMode", vmec)) and _available("nonaxisymmetric", vmec)
    if type_2d == "cross_section" and lpk_on:
        return two_d.render_lpk(
            vmec, s_idx, theme, var_name=var_2d, field_label=field_label,
            colormap=colormap, res_u=res_u, contour_lines=lines,
        )
    phi_frac = float(cv(controls, "2d", "phi", vmec) or 0.0)
    phi_angle = phi_frac * (2 * np.pi / max(vmec.nfp, 1))

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
        return two_d.render_cross_section_field(
            vmec,
            phi_angle,
            var_2d,
            field_label,
            theme,
            colormap=colormap,
            res_u=res_u,
            # The UI hides "smooth" here because a curvilinear carpet cannot be
            # smooth-shaded; coerce defensively in case a stale value survives a switch.
            contour_style="fill" if style == "smooth" else style,
            contour_lines=lines,
        )
    return two_d.render_flux_surface(
        vmec,
        s_idx,
        var_2d,
        field_label,
        theme,
        colormap=colormap,
        res_u=res_u,
        contour_style=style,
        contour_lines=lines,
    )


def threed_view(vmec, controls, theme, field_map):
    s_val = int(cv(controls, "3d", "s3dIdx", vmec))
    var_3d = controls.get("var3d") or _control("3d", "var3d").default
    coord_free = bool(cv(controls, "3d", "coordFree", vmec))
    return three_d.render_3d(
        vmec, s_val, var_3d, field_map.get(var_3d, var_3d), coord_free, theme,
        colormap=cv(controls, "3d", "colormap", vmec),
        resolution=cv(controls, "3d", "res3d", vmec),
    )


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
        return fieldline.render_single_trace(
            vmec, s_idx, transits, alpha0, res, theme, colormap=cv(controls, "fieldline", "colormap", vmec)
        )
    return fieldline.render_fieldline_heatmap(
        vmec,
        s_idx,
        res,
        theme,
        shift_zeta=shift,
        colormap=cv(controls, "fieldline", "colormap", vmec),
        contour_style=cv(controls, "fieldline", "contourStyle", vmec),
        contour_lines=bool(cv(controls, "fieldline", "contourLines", vmec)),
    )


# --------------------------------------------------------------------------------------
# The registry — add a view or a control here and it flows to both UI and backend
# --------------------------------------------------------------------------------------

_TYPE_2D_OPTIONS = [
    {"value": "cross_section", "label": "Cross Section (R-Z)"},
    {"value": "flux_surface", "label": "Flux Surface (θ-ζ)"},
]

RES_2D_OPTIONS = [{"value": "auto", "label": "Auto"}] + [
    {"value": r, "label": r} for r in ("120", "240", "480", "960")
]

# Points in theta, and in zeta per field period - so the mesh is equally fine on every
# device. Auto trims toroidal density only for high-nfp devices (see three_d).
RES_3D_OPTIONS = [{"value": "auto", "label": "Auto"}] + [
    {"value": r, "label": r} for r in ("64", "96", "128", "192", "256")
]

# Plotly's Contourcarpet has no "heatmap" coloring, so smooth shading is impossible on the
# curvilinear R-Z mesh. Hide the option there rather than quietly falling back to bands.
_CONTOUR_STYLE_2D = [
    {"value": "fill", "label": "Filled bands"},
    {"value": "smooth", "label": "Smooth", "visibleWhen": {"type2d": "flux_surface"}},
    {"value": "lines", "label": "Lines only"},
]
_CONTOUR_STYLE_PLAIN = [
    {"value": "fill", "label": "Filled bands"},
    {"value": "smooth", "label": "Smooth"},
    {"value": "lines", "label": "Lines only"},
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
        controls=[
            Control("showGrid", "checkbox", "Background Grid", default=False),
        ],
        render=overview_view,
    ),
    "1d": ViewSpec(
        id="1d",
        label="1D Profiles",
        icon="profiles",
        stats="base",
        controls=[
            Control("profile", "select", "Profile Variable", default="iotaf", options_from="profiles"),
            Control("showGrid", "checkbox", "Background Grid", default=False),
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
            Control(
                "lpkMode", "checkbox", "LPK view", default=False,
                visible_when={"type2d": "cross_section"},
                # Every toroidal cut of an axisymmetric device is the same curve.
                requires="nonaxisymmetric",
            ),
            Control(
                "phi", "slider", "Toroidal Angle", default=0.0, min=0, max=1, step=0.01,
                # LPK fixes its own three angles, so the slider has nothing to drive.
                visible_when={"type2d": "cross_section", "lpkMode": False},
            ),
            Control("sIdx", "slider", "Flux Surface Index", default="ns-1", min=0, max="ns-1", step=1),
            Control(
                "geoCount", "number", "Visible Geometry Surfaces", default=15, min=1, max=200, step=1,
                visible_when={"type2d": "cross_section", "var2d": "geometry", "lpkMode": False},
            ),
            Control("colormap", "select", "Colormap", default="auto", options=list(COLORMAP_OPTIONS)),
            # A select, not a slider: each change is a multi-megabyte transfer, and the
            # control's form should encode its cost.
            Control("res2d", "select", "Mesh Resolution", default="auto", options=RES_2D_OPTIONS),
            Control("contourStyle", "select", "Shading", default="fill", options=_CONTOUR_STYLE_2D),
            Control("contourLines", "checkbox", "Contour Lines", default=True),
            Control("showGrid", "checkbox", "Background Grid", default=False),
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
            Control("res3d", "select", "Mesh Resolution", default="auto", options=RES_3D_OPTIONS),
            Control("coordFree", "checkbox", "Coordinate-free Background", default=True),
            # Same control id as the 2-D and field-line views, so the choice follows the
            # reader across views: the webview keys control state by id, not by view.
            Control("colormap", "select", "Colormap", default="auto", options=list(COLORMAP_OPTIONS)),
            Control("showGrid", "checkbox", "Background Grid", default=False),
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
            Control(
                "fieldlineNLines", "number", "Number of Lines", default=6, min=1, max=20, step=1,
                visible_when={"fieldlineType": "1d_lines"},
            ),
            Control(
                "fieldlineTransits", "number", "Transits", default=50, min=10, max=500, step=10,
                visible_when={"fieldlineType": "single_trace"},
            ),
            Control(
                "fieldlineAlpha0", "number", "Alpha Start", default=0.0, step=0.1,
                visible_when={"fieldlineType": "single_trace"},
            ),
            Control("fieldlineRes", "select", "Resolution", default="128", options=_FIELDLINE_RES_OPTIONS),
            Control(
                "fieldlineZetaShift", "checkbox", "Shift ζ by π/NFP", default=False,
                visible_when={"fieldlineType": ["2d_modB", "1d_lines"]},
            ),
            Control("colormap", "select", "Colormap", default="auto", options=list(COLORMAP_OPTIONS)),
            # This view draws a rectilinear Contour, so smooth shading is available here
            # unconditionally, unlike the curvilinear R-Z carpet.
            Control(
                "contourStyle", "select", "Shading", default="fill", options=_CONTOUR_STYLE_PLAIN,
                visible_when={"fieldlineType": "2d_modB"},
            ),
            Control(
                "contourLines", "checkbox", "Contour Lines", default=True,
                visible_when={"fieldlineType": "2d_modB"},
            ),
            Control("showGrid", "checkbox", "Background Grid", default=False),
        ],
        render=fieldline_view,
    ),
}


def _available_options(options: list[dict[str, Any]] | None, vmec: Any) -> list[dict[str, Any]] | None:
    """Drop options whose equilibrium predicate fails, and strip the predicate itself."""
    if options is None:
        return None
    return [
        {key: value for key, value in option.items() if key != "requires"}
        for option in options
        if _available(option.get("requires"), vmec)
    ]


def _strip_references(condition: dict[str, Any] | None, dropped: set[str]) -> dict[str, Any] | None:
    """Remove conditions on controls this equilibrium does not ship.

    The webview only seeds controls it is sent. A sibling condition on an unshipped
    control compares against ``undefined`` and can never match, which silently hid the
    phi slider on every tokamak (``lpkMode`` is not offered when ntor == 0). An
    unavailable control has, in effect, its default value, and every condition on
    ``lpkMode`` is written for exactly that case, so dropping the term is correct.
    """
    if not condition:
        return condition
    kept = {key: value for key, value in condition.items() if key not in dropped}
    return kept or None


def build_ui_schema(vmec: Any) -> dict[str, Any]:
    """Describe every view + control for the Webview, with dynamic tokens resolved."""
    views = []
    for spec in VIEWS.values():
        dropped = {control.id for control in spec.controls if not _available(control.requires, vmec)}
        controls = []
        for control in spec.controls:
            if control.id in dropped:
                continue
            controls.append(
                {
                    "id": control.id,
                    "kind": control.kind,
                    "label": control.label,
                    "default": _resolve_token(control.default, vmec),
                    "options": _available_options(control.options, vmec),
                    "optionsFrom": control.options_from,
                    "min": _resolve_token(control.min, vmec),
                    "max": _resolve_token(control.max, vmec),
                    "step": control.step,
                    "unit": control.unit,
                    "visibleWhen": _strip_references(control.visible_when, dropped),
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


def apply_grid(fig: Any, show_grid: bool) -> Any:
    """Turn the background grid on or off across every axis of a figure.

    ``update_xaxes``/``update_yaxes`` reach every 2-D axis including subplots, and
    ``update_scenes`` covers the 3-D ones, so this single pass serves all five views. The
    carpet's own aaxis/baxis are not x/y axes and stay hidden either way.
    """
    # showgrid and zeroline are independent in Plotly, and the zero line defaults to on.
    # Left alone it kept drawing a bright line along Z = 0 on an R-Z cross-section with
    # the grid switched off - more visible here than in Plotly's own dark template,
    # because our template brightens zerolinecolor.
    fig.update_xaxes(showgrid=show_grid, zeroline=show_grid)
    fig.update_yaxes(showgrid=show_grid, zeroline=show_grid)
    fig.update_scenes(
        xaxis_showgrid=show_grid, yaxis_showgrid=show_grid, zaxis_showgrid=show_grid,
        xaxis_zeroline=show_grid, yaxis_zeroline=show_grid, zaxis_zeroline=show_grid,
    )
    return fig


def render_view(view_id: str, vmec: Any, controls: dict[str, Any] | None, theme: Any, field_map: dict[str, str]):
    """Dispatch a render request to the view's render function."""
    spec = VIEWS[view_id]
    controls = controls or {}
    fig = spec.render(vmec, controls, theme, field_map)
    return apply_grid(fig, bool(cv(controls, view_id, "showGrid", vmec)))
