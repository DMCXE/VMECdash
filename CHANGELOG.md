# Changelog

All notable changes to VMECdash are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The Python package (`vmecdash`), the standalone Dash app and the VS Code extension
(`vmecdash-viewer`) share a version number and are released together.

## [0.2.0] - 2026-09-29

New physical quantities and new controls, plus changes to how existing figures are
coloured.

### Added

#### New physical quantities

- **Cylindrical field components `B_R`, `B_phi`, `B_Z`.** These are what coil design,
  diagnostic placement and particle pushing actually need; previously only the covariant
  and contravariant components were available.
- **Parallel current `j-parallel`** (`J·B / |B|`).
- **Force-balance residual `|JxB - grad p|`**, reported as a raw magnitude in N/m³. It is
  deliberately *not* normalised by `|grad p|`, which vanishes identically on the zero-beta
  equilibria this tool is routinely pointed at.
- All of the above are available in the R-Z cross-section, the θ-ζ flux surface and the
  3-D surface views, and are validated against `simsopt` where a reference exists
  (see *Verification* below).

#### New controls

- **Colormap** selector (`Auto / Viridis / RdBu / Cividis / Plasma / Greys / Jet`). `Auto`
  is the default and the only setting that guarantees a signed field gets a diverging
  scale. Shares one control id across the 2-D, 3-D and field-line views, so the choice
  follows you between views.
- **Mesh Resolution** selector for 2-D plots (`Auto / 120 / 240 / 480 / 960`). Deliberately
  a select rather than a slider: raising it costs transfer size, not render time.
- **Mesh Resolution** selector for the 3-D view (`Auto / 64 / 96 / 128 / 192 / 256`), in
  points along θ and along ζ *per field period*, so every device is sampled equally finely.
  `Auto` is 128 × 128 per period, trimmed toroidally only for high-`nfp` devices so the mesh
  stays under 131 072 vertices.
- **Shading** selector (`Filled bands / Smooth / Lines only`) and a **Contour Lines**
  toggle. `Smooth` is offered only where Plotly supports it — `Contourcarpet` has no
  `heatmap` coloring, so a curvilinear R-Z mesh cannot be smooth-shaded, and the option is
  hidden there rather than silently falling back.
- **Background Grid** toggle, off by default, applying to every view.
- **LPK view** toggle on the R-Z cross-section: draws the same flux surface at
  φ = 0, ¼ and ½ of a field period on one axes, following VMECplot's `LPK Plot`. Unlike
  the MATLAB original it honours the current colour variable. Hidden entirely for
  axisymmetric equilibria, where every toroidal cut is the same curve.

#### Reloading

- **Reload from disk.** The VS Code preview notices when its wout file changes on disk and
  marks the reload button with a "File changed" badge. A new `vmecdash.autoReload` setting
  reloads automatically; it is off by default because a running VMEC rewrites its output
  repeatedly.
- **Reloading a file that is still being written keeps the previous data on screen**, with a
  message saying so, instead of losing the preview. Reload again once the run has finished.

#### Export

- **Vector export (SVG)** alongside PNG, in both front-ends. In the VS Code extension the
  image is saved through the extension host rather than a page download, which webviews
  block.

### Changed

- **Colour policy is now shared by every renderer.** Signed fields get a diverging scale
  with limits symmetric about zero; single-sign fields get a perceptually uniform
  sequential scale. Six of the ten VMEC fields are signed, and previously only the R-Z
  cross-section applied this rule.
- **`Jet` is no longer used anywhere by default.** The 3-D view hardcoded it; it is not
  perceptually uniform, not colour-blind safe, and places no neutral point at zero. It
  remains available as an explicit choice for comparing against legacy VMECplot figures.
- **Diverging scales put high values at red and low at blue**, consistently. The decision
  is made by colormap identity rather than per code path, so `RdBu` means the same thing
  on `|B|` as on `j^u`.
- **One figure template.** Typography, margins, colorbar geometry and contour level count
  are now defined once and composed onto Plotly's own dark/light template, instead of
  three different treatments of the same idea across the renderers.
- Contour level count unified to a single constant (previously 40, 50 and 50 in three
  places).
- **Views you have already rendered are served from memory** rather than recomputed, within
  a fixed memory budget. Returning to a figure, or toggling back to a previous setting, is
  immediate.
- Dash: the Display panel now sits below the physics controls and appears only in the
  views it can affect; it was above them and always visible, including on the 1-D profile
  and summary views where it does nothing.

### Fixed

- **3-D surfaces showed triangular facets and dark kinked streaks**, most visibly with
  `RdBu` along the sharp bends of a stellarator boundary, and raising the resolution could
  not help because the view ignored it. Three causes: the grid was a fixed 100 × 100 over
  the *whole torus* (33 toroidal points per period on a 3-period device); `go.Surface`
  estimates normals from its open parameter grid, which leaves streaks at high curvature
  and a one-sided seam at θ = ζ = 0; and the grid repeated its endpoints. The view now
  draws a `Mesh3d` on a closed, seamless triangulation of a periodic grid, as DESC's
  `plot_3d` does, with smooth shading and a user-selectable resolution.
- **Theme switching left the plot behind.** In the Dash app the file-drop overlay covering
  the plot area had its background hardcoded to the dark surface colour, so switching to
  light mode changed the text inside it but not its background. In the VS Code extension
  the webview sampled the theme only when a render was requested, so changing the VS Code
  colour theme restyled all the surrounding chrome while the figure kept its old
  background until some control was touched.
- **A bright line along Z = 0** persisted on R-Z cross-sections with the grid switched
  off. `showgrid` and `zeroline` are independent properties in Plotly and the zero line
  defaults to on; both now follow the grid toggle.
- **The background grid toggle did not reach the summary dashboard** in the Dash app.
- **The control panel could not be scrolled** in the Dash app, so a long control set
  required enlarging the window. Mantine's `AppShell.Aside` is `position: fixed` with
  `overflow: visible` and needs an explicit scroll container.
- Signed fields (`lambda`, `B_s`, `B_u`, `B^u`, `j^u`, `j^v`) rendered on a sequential
  colour scale in the θ-ζ and 3-D views, which hid the zero crossing — `j^u` spanning
  −2026…1767 read as if it were a magnitude.
- Dash: the 2-D figure silently reverted to default settings when the toroidal-angle
  slider was dragged, because the drag fast path was a second, separate copy of the render
  logic. Both callbacks now share one path.

### Internal

- `vmecdash/theme.py` owns the colour-scale policy (`resolve_colorscale`), the Plotly
  template and the shared colour tokens.
- `vmecdash/view_schema.py` gained `requires`, an equilibrium-level availability gate
  (`visible_when` can only test sibling control values), and option-level `visibleWhen`.
- New core APIs: `VMECJaxProcessor.get_surface_curve` (a single flux-surface outline at one
  toroidal angle) and `derived_values` (quantities assembled from several wout series).
- New JAX kernels for the angular derivatives of `R` and `Z`, which the cylindrical
  components and the force residual are built on.
- The θ-ζ flux-surface view evaluates each field separably (`cos(mθ−nζ)` factors into a θ
  part and a ζ part), so its memory grows with the resolution rather than its square. The
  direct form would have needed several gigabytes at the highest Mesh Resolution.
  `VMECJaxProcessor.derived_values_grid` is the gridded counterpart of `derived_values`.
- The 3-D view samples its surface with the same separable kernel, replacing
  `_jit_surface_3d`, whose `(modes, θ, ζ)` intermediates grew with the square of the
  resolution. `VMECJaxProcessor.compute_3d_surface` now takes `n_theta` and
  `n_zeta_per_period` instead of a single `resolution`.
- Figures are cached as serialized JSON, so the cache's memory budget is exact and a cached
  figure is written straight to the extension without being re-encoded.
- Continuous integration: tests on Python 3.10, 3.11 and 3.12, `ruff check`, and a type-check
  and build of the extension. See `CONTRIBUTING.md`.

### Verification

- 85 tests pass (1 skipped: an optional local fixture).
- The 3-D surface matches a direct Fourier sum to 1e-12, and its triangulation is a closed
  torus: every edge is shared by exactly two triangles and V − E + F = 0.
- Angular derivatives match `simsopt`'s `vmec_compute_geometry` to ~1e-15 relative.
- Cylindrical components match to ~1e-4 relative; the looser figure is the half-mesh
  radial lift, which is not the same interpolation `simsopt` uses.
- The current normalisation is checked against the wout file's own `<J·B>` profile, since
  every equilibrium available for testing is zero-beta and so cannot constrain force
  balance.

## [0.1.5]

- Responsive webview layout, schema-driven control visibility, smooth slider renders.

## [0.1.4]

- Force CPU JAX; attribute to USTC Stellarator Lab; MIT licence.
