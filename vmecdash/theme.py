from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio


@dataclass(frozen=True)
class Palette:
    """Every colour VMECdash draws with, for one mode.

    Renderers read these off ``PlotTheme.palette`` instead of carrying their own
    literals, so the plot background and the surrounding page cannot drift apart.
    """

    surface: str  # figure paper and plot background
    text: str  # annotation text on an empty figure
    axis: str  # 3D axis labels and tick marks
    grid: str  # 3D grid lines and zero lines
    grid_soft: str  # 2D background grid, when the reader turns it on
    line: str  # default colour for a lone trace
    colorway: tuple[str, ...]  # summary dashboard panels, in order


# The six summary-dashboard hues. Shared by both modes: they sit on a mid-grey or white
# ground and stay legible on either.
_COLORWAY = ("#22b8cf", "#fd7e14", "#fa5252", "#12b886", "#7950f2", "#0ca678")

DARK_PALETTE = Palette(
    surface="#2e2e2e",
    text="#adb5bd",
    axis="#dee2e6",
    grid="#5c677d",
    grid_soft="rgba(255, 255, 255, 0.10)",
    line="#3bc9db",
    colorway=_COLORWAY,
)

LIGHT_PALETTE = Palette(
    surface="white",
    text="#495057",
    axis="#495057",
    grid="#ced4da",
    grid_soft="rgba(0, 0, 0, 0.08)",
    line="#1098ad",
    colorway=_COLORWAY,
)


# --------------------------------------------------------------------------------------
# Colour-scale policy
# --------------------------------------------------------------------------------------

#: Number of filled bands in every contour plot. One constant so the R-Z carpet, the
#: theta-zeta surface and the field-line map cannot drift apart again.
CONTOUR_LEVELS = 40

#: Isoline colour drawn on top of colormap fill. Semi-transparent black reads on both
#: a Viridis band and an RdBu one, in either page theme.
CONTOUR_LINE_COLOR = "rgba(0,0,0,0.35)"

SEQUENTIAL_SCALE = "Viridis"
DIVERGING_SCALE = "RdBu"

#: Plotly ships these diverging scales warm-to-cool - RdBu[0] is dark red - but the
#: reading convention is high = red, low = blue. Reverse them so that holds.
#:
#: The decision is made by scale identity rather than inside the signed/single-sign
#: branches: otherwise the same colormap would flip meaning depending on whether the data
#: happened to straddle zero, so RdBu on |B| would read high-blue while RdBu on j^u read
#: high-red.
WARM_LOW_SCALES = frozenset({"RdBu", "RdYlBu", "RdGy", "RdYlGn", "Spectral"})

#: Offered in the UI. "auto" is the default and the only value that guarantees a signed
#: field gets a diverging scale; everything else is a deliberate override.
COLORMAP_OPTIONS = (
    {"value": "auto", "label": "Auto"},
    {"value": "Viridis", "label": "Viridis"},
    {"value": "RdBu", "label": "RdBu"},
    {"value": "Cividis", "label": "Cividis"},
    {"value": "Plasma", "label": "Plasma"},
    {"value": "Greys", "label": "Greys"},
    {"value": "Jet", "label": "Jet"},
)


@dataclass(frozen=True)
class ColorScale:
    """A resolved colour mapping for one field: which scale, and over what limits."""

    scale: str
    reversescale: bool
    zmin: float
    zmax: float
    reason: str  # short note for the UI, e.g. "signed - diverging about 0"


def resolve_colorscale(values, override: str | None = None) -> ColorScale:
    """
    Pick the colour scale and limits for a field.

    Signed data gets a diverging scale with limits symmetric about zero, so the zero
    crossing lands on the neutral midpoint and the sign structure stays visible. Six of
    the ten VMEC fields VMECdash can draw are signed (lambda, B_s, B_u, B^u, j^u, j^v),
    and on a sequential scale their zero crossing is invisible.

    ``override`` replaces the scale but never the limits policy: choosing a sequential
    map for signed data is the user's call, silently moving where zero sits is not.
    """
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    chosen = override if override and override != "auto" else None

    if finite.size == 0:
        scale = chosen or SEQUENTIAL_SCALE
        return ColorScale(scale, scale in WARM_LOW_SCALES, 0.0, 1.0, "no finite data")

    vmin = float(np.min(finite))
    vmax = float(np.max(finite))

    if vmin < 0.0 < vmax:
        bound = max(abs(vmin), abs(vmax))
        scale = chosen or DIVERGING_SCALE
        reason = "signed - diverging about 0" if chosen is None else f"signed, forced {scale}"
        return ColorScale(scale, scale in WARM_LOW_SCALES, -bound, bound, reason)

    if np.isclose(vmin, vmax):
        pad = max(abs(vmin) * 0.01, 1e-9)
        vmin -= pad
        vmax += pad
    scale = chosen or SEQUENTIAL_SCALE
    reason = "single-sign - sequential" if chosen is None else f"single-sign, forced {scale}"
    return ColorScale(scale, scale in WARM_LOW_SCALES, vmin, vmax, reason)


# --------------------------------------------------------------------------------------
# Figure template
# --------------------------------------------------------------------------------------

def build_plotly_template(palette: Palette) -> go.layout.Template:
    """One place for the typography, spacing and colorbar geometry of every figure.

    Composed onto Plotly's own dark/light template rather than replacing it, so only the
    things VMECdash has an opinion about are overridden.
    """
    axis = dict(
        gridcolor=palette.grid_soft,
        gridwidth=1,
        griddash="dot",
        zerolinecolor=palette.grid,
        zerolinewidth=1,
        linecolor=palette.axis,
        ticks="outside",
        ticklen=4,
        tickfont=dict(size=11),
        title=dict(font=dict(size=12)),
    )
    return go.layout.Template(
        layout=dict(
            font=dict(family="Inter, Helvetica Neue, Helvetica, Arial, sans-serif", size=12, color=palette.text),
            title=dict(font=dict(size=15), x=0.02, xanchor="left"),
            paper_bgcolor=palette.surface,
            plot_bgcolor=palette.surface,
            colorway=list(palette.colorway),
            margin=dict(l=60, r=20, t=48, b=52),
            xaxis=axis,
            yaxis=axis,
            coloraxis=dict(colorbar=dict(thickness=12, outlinewidth=0, ticks="outside", ticklen=4)),
            legend=dict(font=dict(size=11), borderwidth=0),
        )
    )


@dataclass
class PlotTheme:
    fig_template: str
    paper_bg: str
    plot_bg: str
    dark_mode: bool
    text_color: str
    reset_seed: int
    palette: Palette = DARK_PALETTE


_TEMPLATES = {
    True: ("plotly_dark", "vmecdash_dark", DARK_PALETTE),
    False: ("plotly_white", "vmecdash_light", LIGHT_PALETTE),
}

for _base, _name, _palette in _TEMPLATES.values():
    pio.templates[_name] = build_plotly_template(_palette)


def build_theme(dark_mode: bool, reset_seed: int = 0) -> PlotTheme:
    base, name, palette = _TEMPLATES[bool(dark_mode)]
    # paper_bg/plot_bg/text_color are kept as their own fields because every renderer
    # already reads them; they are now views onto the palette rather than literals.
    return PlotTheme(
        fig_template=f"{base}+{name}",
        paper_bg=palette.surface,
        plot_bg=palette.surface,
        dark_mode=dark_mode,
        text_color=palette.text,
        reset_seed=reset_seed,
        palette=palette,
    )


def make_empty_figure(theme: PlotTheme, message: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        autosize=True,
        template=theme.fig_template,
        paper_bgcolor=theme.paper_bg,
        plot_bgcolor=theme.plot_bg,
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    fig.add_annotation(text=message, showarrow=False, font=dict(size=20, color=theme.text_color))
    return fig

