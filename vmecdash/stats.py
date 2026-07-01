from __future__ import annotations

import math
import os
from typing import Any

import numpy as np


def _safe_fmt(value, pattern="{:.2f}", fallback="--"):
    try:
        if value is None or (isinstance(value, (float, int)) and np.isnan(value)):
            return fallback
        return pattern.format(value)
    except Exception:
        return fallback


def base_stats(scalars: dict[str, Any]) -> list[dict[str, str]]:
    beta = scalars.get("beta_total", 0.0)
    vol = scalars.get("volume", 0.0)
    curr = scalars.get("ctor", 0.0)
    aspect_ratio = scalars.get("Rmajor") / scalars.get("Aminor") if scalars.get("Aminor") else math.nan
    ar_str = f"{aspect_ratio:.2f}" if aspect_ratio == aspect_ratio else "N/A"
    return [
        {"title": "Beta Total", "value": f"{beta*100:.2f}%", "icon": "percent", "color": "red"},
        {"title": "Volume", "value": f"{vol:.2f} m^3", "icon": "cube", "color": "blue"},
        {"title": "Aspect Ratio", "value": ar_str, "icon": "torus", "color": "orange"},
        {"title": "Toroidal Current", "value": f"{curr:.2f} A", "icon": "current", "color": "teal"},
    ]


def overview_stats(vmec, scalars: dict[str, Any]) -> dict[str, Any]:
    beta = scalars.get("beta_total", 0.0)
    vol = scalars.get("volume", 0.0)
    curr = scalars.get("ctor", 0.0)
    rmaj = scalars.get("Rmajor")
    amin = scalars.get("Aminor")

    def fmt(val, f="{:.3f}"):
        return _safe_fmt(val, f)

    mgrid = vmec.ds.get("mgrid_file", None)
    mgrid_full = None
    if mgrid is not None:
        try:
            mgrid_val = mgrid.values
        except Exception:
            mgrid_val = mgrid
        try:
            mgrid_val = np.asarray(mgrid_val).item()
        except Exception:
            pass
        if isinstance(mgrid_val, (bytes, np.bytes_)):
            mgrid_val = mgrid_val.decode("utf-8", errors="ignore")
        mgrid_full = str(mgrid_val).strip()
        mgrid = mgrid_full.strip("()'\" ")
    lfreeb = vmec.ds.get("lfreeb__logical__", 0)
    if lfreeb is not None:
        try:
            lfreeb = bool(lfreeb.values)
        except Exception:
            lfreeb = False
    free_boundary = bool(lfreeb or (mgrid is not None and mgrid != "none"))
    mgrid_short = os.path.basename(mgrid.rstrip("/")) if (mgrid and mgrid != "none") else None
    bound_type = "Free boundary" if free_boundary else "Fixed boundary"
    mgrid_label = mgrid_short if free_boundary and mgrid_short else "None"

    hero = [
        {"title": "Total Beta", "value": f"{beta*100:.2f}", "unit": "%", "subText": "Global beta", "color": "cyan"},
        {"title": "Toroidal Current", "value": f"{curr:.2f}", "unit": "A", "subText": "Plasma current", "color": "indigo"},
        {"title": "Volume", "value": f"{vol:.2f}", "unit": "m^3", "subText": "Volume derivative (Vp)", "color": "teal"},
    ]
    details = [
        {
            "title": "Geometry",
            "items": [
                {"label": "Major Radius (R0)", "value": fmt(rmaj), "unit": "m"},
                {"label": "Minor Radius (a)", "value": fmt(amin), "unit": "m"},
                {"label": "Aspect Ratio", "value": fmt(scalars.get("aspect"), "{:.2f}"), "unit": ""},
                {"label": "Volume", "value": fmt(vol), "unit": "m^3"},
                {"label": "Stellarator Symmetry", "value": "True" if not vmec.lasym else "False", "unit": ""},
            ],
        },
        {
            "title": "Magnetics",
            "items": [
                {"label": "Magnetic Field (B0)", "value": fmt(scalars.get("b0")), "unit": "T"},
                {"label": "Toroidal Flux (RB_tor)", "value": fmt(scalars.get("rbtor")), "unit": "T.m"},
                {"label": "Iota (Axis)", "value": fmt(scalars.get("iota_axis")), "unit": ""},
                {"label": "Iota (Edge)", "value": fmt(scalars.get("iota_edge")), "unit": ""},
                {"label": "Safety Factor (q0)", "value": fmt(scalars.get("q_axis")), "unit": ""},
                {"label": "Safety Factor (qa)", "value": fmt(scalars.get("q_edge")), "unit": ""},
                {"label": "Shear (Edge)", "value": fmt(scalars.get("shear_edge")), "unit": ""},
            ],
        },
        {
            "title": "Plasma & Boundary",
            "items": [
                {"label": "Beta Poloidal", "value": fmt(scalars.get("betapol")), "unit": ""},
                {"label": "Beta Toroidal", "value": fmt(scalars.get("betator")), "unit": ""},
                {"label": "Pressure (Axis)", "value": fmt(scalars.get("pressure_axis"), "{:.2e}"), "unit": "Pa"},
                {"label": "Pressure (Edge)", "value": fmt(scalars.get("pressure_edge"), "{:.2e}"), "unit": "Pa"},
                {"label": "Boundary Type", "value": bound_type, "unit": ""},
                {"label": "MGRID file", "value": mgrid_label, "unit": ""},
            ],
        },
    ]
    return {"base": base_stats(scalars), "hero": hero, "details": details}

