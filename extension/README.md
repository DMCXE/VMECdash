# VMECdash Viewer

Open VMEC `wout*.nc` equilibrium files directly in VS Code and explore them with interactive
Plotly figures — a summary dashboard, 1D profiles, 2D cross-sections, 3D geometry, and field
lines.

> **Requires a Python backend.** This extension is the UI only; all numerics run in Python
> (JAX). You must install the `vmecdash` package into a Python interpreter and point the
> extension at it. See **Requirements** below — without it a file will open but nothing renders.

## Features

- Custom editor for `**/wout*.nc` — just open the file.
- Views: Summary dashboard, 1D profiles, 2D cross-section (R–Z carpet / flux surface),
  3D geometry, and field lines.
- Per-view controls that adapt automatically (dropdowns, sliders with live value readouts,
  toggles), driven by a schema the backend ships — new physical quantities appear without a
  UI change.
- Light/dark aware; bundled Plotly.js (no CDN); works locally and over **Remote-SSH**.

## Requirements

1. **Python ≥ 3.10** with the backend installed:

   ```bash
   pip install vmecdash
   ```

   JAX is pulled in automatically; the CPU build works out of the box.

2. Tell the extension which interpreter to use. It resolves one in this order:
   1. the **`vmecdash.pythonPath`** setting — set it at **User** scope so it applies to every
      workspace (Command Palette → **“VMECdash: Select Python Path”** lets you pick User or
      Workspace), **or**
   2. the **`VMECDASH_PYTHON`** environment variable (handy for Remote-SSH / CI / terminal
      launches), **or**
   3. the interpreter selected by the Microsoft Python extension, **or**
   4. `python3` / `python` on your `PATH`.

   The chosen interpreter must have `vmecdash` installed. If it doesn't, the extension tells you
   *which* interpreter it used and offers a **Select Python Path** button — it won't fail silently.

3. Run **“VMECdash: Check Backend”** from the Command Palette to verify (`vmecdash` + `jax`
   versions).

On **Remote-SSH**, install `vmecdash` in the *remote* interpreter — the backend runs on the
remote host, so no localhost server or file upload is needed.

## Usage

- Open any `wout*.nc` file (or run **“VMECdash: Open Preview”**).
- Pick a view on the left; adjust its controls on the right.
- **“VMECdash: Export Report”** writes a text summary of scalars and metadata.

## Settings

| Setting | Description |
|---|---|
| `vmecdash.pythonPath` | Python executable that runs `python -m vmecdash.vscode_backend --stdio`. |
| `vmecdash.backendArgs` | Extra arguments passed to the backend. |

## Troubleshooting

If a `wout` file opens but no figure appears, the selected interpreter is almost certainly
missing `vmecdash` or `jax`. Run **“VMECdash: Check Backend”** for a diagnostic, then
`pip install vmecdash` into that interpreter and reload.

## Source

<https://github.com/DMCXE/VMECdash>
