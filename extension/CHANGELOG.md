# Changelog — VMECdash Viewer

The Marketplace renders this file on the extension's Changelog tab. It covers the VS Code
extension only; the full project changelog lives at the
[repository root](https://github.com/DMCXE/VMECdash/blob/regularize/CHANGELOG.md).

## [0.2.0] - 2026-09-29

### Added

- **New quantities in every view**: cylindrical field components `B_R`, `B_phi`, `B_Z`,
  parallel current `j-parallel`, and a force-balance residual `|JxB - grad p|`.
- **Colormap selector** (`Auto / Viridis / RdBu / Cividis / Plasma / Greys / Jet`). `Auto`
  picks a diverging scale for signed fields. The choice follows you between views.
- **Mesh resolution selector** for 2-D plots. Raising it costs transfer size, not render
  time.
- **Mesh resolution selector** for the 3-D view (`Auto / 64 / 96 / 128 / 192 / 256`), in
  points along θ and along ζ per field period, so every device is sampled equally finely.
- **Shading** (filled bands / smooth / lines only) and a **contour lines** toggle. Smooth
  shading is offered only on the θ-ζ surface, because a curvilinear R-Z mesh cannot be
  smooth-shaded by Plotly.
- **Background grid** toggle, off by default.
- **LPK view** toggle on the R-Z cross-section — the same flux surface at φ = 0, ¼ and ½ of
  a field period on one axes, coloured by the current variable. Hidden for axisymmetric
  equilibria.
- **Reload from disk**, with a "File changed" badge when the wout changes underneath the
  preview, and a `vmecdash.autoReload` setting (off by default, since a running VMEC
  rewrites its output repeatedly). Reloading while the file is still being written keeps
  the previous data on screen.
- **Export the current figure as SVG or PNG**, from the camera button in the toolbar. The
  file is written through a save dialog.

### Changed

- Signed fields now use a diverging colour scale centred on zero in every view. The 3-D
  view previously used `Jet`, which put no neutral point at zero, so a field like `j^u`
  read as a magnitude.
- Diverging scales consistently place high values at red and low at blue.
- Figures share one template for typography, margins, colorbar geometry and contour level
  count.

### Fixed

- 3-D surfaces showed triangular facets and dark streaks along sharp bends, especially with
  `RdBu`, and did not respond to resolution. They are now drawn as a seamless, smooth-shaded
  triangle mesh sampled per field period.
- Changing the VS Code colour theme restyled the interface but left the plot on the old
  background until a control was touched.
- A bright line along Z = 0 remained on R-Z cross-sections with the grid switched off.

## [0.1.5]

- Responsive webview layout, schema-driven control visibility, smooth slider renders.

## [0.1.4]

- Force CPU JAX; attribute to USTC Stellarator Lab; MIT licence.
