# Contributing to VMECdash

Thanks for taking an interest. VMECdash is a visualisation tool for VMEC `wout` NetCDF
equilibria, so correctness of what a plot *shows* matters as much as whether the code runs.

## Getting set up

```bash
python -m pip install -e ".[dash,dev]"
python -m pytest -q
```

JAX is pinned to CPU. If you are on a platform without FP64 GPU support (Apple Metal, for
instance), `pip install jax[cpu]` is what you want.

The VS Code extension lives in `extension/`:

```bash
cd extension
npm install
npm run compile     # or: npm run watch
```

Press <kbd>F5</kbd> in VS Code to launch an Extension Development Host, then open any
`wout*.nc` file.

## Layout

| Path | What lives there |
| --- | --- |
| `vmecdash/core/` | Equilibrium loading and the JAX evaluation kernels |
| `vmecdash/renderers/` | Figure construction. **Dash-free** — shared by both front-ends |
| `vmecdash/theme.py` | Colour policy, the Plotly template, colour tokens |
| `vmecdash/view_schema.py` | The single registry of views and their controls |
| `vmecdash/vscode_backend.py` | The stdio backend the VS Code extension talks to |
| `vmecdash/dash_app/` | The standalone Dash application |
| `extension/` | The VS Code extension (TypeScript + a plain-JS webview) |

Two things are worth knowing before you change anything:

- **`vmecdash/renderers/` must not import Dash.** Both front-ends render through it, so a
  Dash import there would break the VS Code backend.
- **`view_schema.py` is the source of truth for views and controls.** A control declared
  there reaches the webview automatically. The Dash app currently declares its controls
  separately, so a new user-facing control has to be added in both places.

## Adding a control

1. Add a `Control(...)` to the relevant `ViewSpec` in `view_schema.py`. Use `visible_when`
   to depend on a sibling control's value, and `requires` to depend on a property of the
   equilibrium itself (for example `nonaxisymmetric`).
2. Read it in that view's render function via `cv(...)`, and thread it into the renderer.
3. Mirror it in `vmecdash/dash_app/controls.py` and `app.py`.
4. Update `tests/test_view_schema.py` if the control is conditional.

## Adding a plotted quantity

- Fields backed by their own Fourier series in the wout file go in `FIELD_SPECS`.
- Quantities assembled from several series — anything needing the geometry's angular
  derivatives — go in `DERIVED_FIELD_SPECS` and are evaluated by `derived_values`.
- Two things will silently corrupt a result if missed: several fields live on the **half
  radial mesh** and must be lifted before use, and field coefficients carry **Nyquist mode
  numbers** while `R`/`Z` do not, so they cannot share an angle matrix.

## Colour

Do not hardcode a colormap in a renderer. Call `resolve_colorscale(values, override)` from
`vmecdash/theme.py`. It gives signed data a diverging scale with limits symmetric about
zero so the zero crossing lands on the neutral midpoint — on a sequential scale that
structure is invisible, and roughly half the VMEC fields are signed.

## Tests

```bash
python -m pytest -q
```

New numerical work should be checked against something external rather than against
itself. `tests/test_derived_fields.py` compares against `simsopt`'s
`vmec_compute_geometry`, and against the wout file's own profiles where no external
reference exists. State the tolerance you chose and why — a looser tolerance is fine when
there is a reason for it, but the reason belongs in the test.

## Style

- Python targets 3.10+. Run `ruff check .` before opening a pull request; CI enforces it.
- The codebase is **not** `ruff format`-clean yet, and adopting it would touch almost every
  file. Please don't reformat files your change doesn't otherwise touch — a formatting
  sweep should be its own commit, agreed separately.
- Comments should explain *why*, especially where a non-obvious constraint forced a
  decision. Several already do; please keep that up.
