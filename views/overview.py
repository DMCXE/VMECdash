from __future__ import annotations

import os

import dash_mantine_components as dmc
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ui.components import create_detail_card, create_hero_stat, get_icon
from views.shared import PlotTheme


def controls():
    return dmc.Paper(
        id="wrapper-overview",
        withBorder=True,
        shadow="xs",
        radius="md",
        p="md",
        style={"display": "block"},
        children=[
            dmc.Group([get_icon("mdi:information-outline", 20), dmc.Text("Summary dashboard highlights six canonical VMEC plots.", size="sm")], gap="xs"),
            dmc.Alert(
                "Displays the standard VMEC equilibrium summary (2x3 panel).",
                title="Overview Mode",
                color="gray",
                variant="light",
                icon=get_icon("mdi:view-dashboard-outline"),
            ),
            dmc.Button(
                "Export Report",
                id="btn-export-report",
                variant="light",
                color="grape",
                fullWidth=True,
                leftSection=get_icon("mdi:file-document"),
                mt="md",
            ),
        ],
    )


def _safe_fmt(value, pattern="{:.2f}", fallback="--"):
    try:
        if value is None or (isinstance(value, (float, int)) and np.isnan(value)):
            return fallback
        return pattern.format(value)
    except Exception:
        return fallback


def render_overview(vmec, scalars: dict, summary_lines: list[str] | None, theme: PlotTheme):
    beta = scalars.get("beta_total", 0.0)
    vol = scalars.get("volume", 0.0)
    curr = scalars.get("ctor", 0.0)
    rmaj = scalars.get("Rmajor")
    amin = scalars.get("Aminor")

    hero_cards = dmc.SimpleGrid(
        cols=3,
        spacing="md",
        children=[
            create_hero_stat("Total Beta", f"{beta*100:.2f}", "%", "mdi:percent", "cyan", "Global beta"),
            create_hero_stat("Toroidal Current", f"{curr:.2f}", "A", "mdi:current-ac", "indigo", "Plasma current"),
            create_hero_stat("Volume", f"{vol:.2f}", "m³", "mdi:cube-outline", "teal", "Volume derivative (Vp)"),
        ],
    )

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

    geo_items = [
        ("Major Radius (R0)", fmt(rmaj), "m"),
        ("Minor Radius (a)", fmt(amin), "m"),
        ("Aspect Ratio", fmt(scalars.get("aspect"), "{:.2f}"), ""),
        ("Volume", fmt(vol), "m³"),
        ("Stellarator Symmetry", "True" if not vmec.lasym else "False", ""),
    ]
    mag_items = [
        ("Magnetic Field (B0)", fmt(scalars.get("b0")), "T"),
        ("Toroidal Flux (RB_tor)", fmt(scalars.get("rbtor")), "T·m"),
        ("Iota (Axis)", fmt(scalars.get("iota_axis")), ""),
        ("Iota (Edge)", fmt(scalars.get("iota_edge")), ""),
        ("Safety Factor (q0)", fmt(scalars.get("q_axis")), ""),
        ("Safety Factor (qa)", fmt(scalars.get("q_edge")), ""),
        ("Shear (Edge)", fmt(scalars.get("shear_edge")), ""),
    ]
    plasma_items = [
        ("Beta Poloidal", fmt(scalars.get("betapol")), ""),
        ("Beta Toroidal", fmt(scalars.get("betator")), ""),
        ("Pressure (Axis)", fmt(scalars.get("pressure_axis"), "{:.2e}"), "Pa"),
        ("Pressure (Edge)", fmt(scalars.get("pressure_edge"), "{:.2e}"), "Pa"),
        ("Boundary Type", bound_type, ""),
        ("MGRID file", mgrid_label, ""),
    ]

    details_grid = dmc.SimpleGrid(
        cols=3,
        spacing="md",
        children=[
            create_detail_card("Geometry", "mdi:axis-arrow", geo_items),
            create_detail_card("Magnetics", "mdi:magnet", mag_items),
            create_detail_card("Plasma & Boundary", "mdi:fire", plasma_items),
        ],
    )

    stats_ui = dmc.Stack([hero_cards, details_grid], gap="md")

    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "Rotational Transform (iota)",
            "Safety Factor (q)",
            "Pressure Profile",
            "dP/ds",
            "Volume Derivative (Vp)",
            "Flux Avg <B·B>",
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.08,
    )

    def get_data_safe(key):
        try:
            return vmec.get_1d_data(key)
        except Exception:
            return [], []

    s_i, iota = get_data_safe("iotaf")
    s_q, q_prof = get_data_safe("q")
    s_p, pres = get_data_safe("presf")
    s_dp, dpds = get_data_safe("dpds")
    s_vp, vp = get_data_safe("vp")
    s_bb, bdotb = get_data_safe("bdotb")

    fig.add_trace(go.Scatter(x=s_i, y=iota, name="iota", line=dict(color="#22b8cf", width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=s_q, y=q_prof, name="q", line=dict(color="#fd7e14", width=3)), row=1, col=2)
    fig.add_trace(go.Scatter(x=s_p, y=pres, name="pres", line=dict(color="#fa5252", width=3)), row=1, col=3)
    fig.add_trace(go.Scatter(x=s_dp, y=dpds, name="dP/ds", line=dict(color="#12b886", width=3)), row=2, col=1)
    fig.add_trace(go.Scatter(x=s_vp, y=vp, name="Vp", line=dict(color="#7950f2", width=3)), row=2, col=2)
    fig.add_trace(go.Scatter(x=s_bb, y=bdotb, name="<B.B>", line=dict(color="#0ca678", width=3)), row=2, col=3)

    fig.update_layout(
        title_text="Equilibrium Summary",
        template=theme.fig_template,
        paper_bgcolor=theme.paper_bg,
        plot_bgcolor=theme.plot_bg,
        height=800,
        showlegend=False,
        uirevision=f"overview-{theme.reset_seed}",
    )

    stats_ui = dmc.Stack([hero_cards, details_grid], gap="md")
    return fig, stats_ui
