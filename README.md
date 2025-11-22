# VMEC Viewer

An interactive Dash application for exploring VMEC `wout_*.nc` stellarator equilibria.  
It reconstructs magnetic surfaces with JAX, renders 1‑D/2‑D/3‑D plots, and exposes a summary dashboard similar to **VMECplot.m**.

---

## Features

- **File upload** for standard VMEC `wout` NetCDF output.
- **Summary dashboard** with key scalar diagnostics and highlighted metadata (free/fixed boundary, mgrid file, etc.).
- **1‑D profiles** for rotational transform, safety factor, pressure, enclosed volume, beta metrics, and more.
- **2‑D visualisations**:
  - R‑Z cross sections with geometry or interpolated scalar fields.
  - θ‑ζ flux-surface contours across a field period.
- **3‑D flux surfaces** with optional coordinate‑free background shading.
- **Client‑side download** button for exporting the current figure as PNG.
- **JAX acceleration** for geometry reconstruction while still integrating with SciPy/Plotly tooling.

---

## Requirements

Python 3.10+ is recommended. Install dependencies with pip:

```bash
python -m venv .venv
.\.venv\Scripts\activate           # Windows
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note for Windows users:** install the CPU version of JAX with  
> `pip install -r requirements.txt "jax[cpu]"`  
> GPU/TPU wheels require platform-specific instructions from the [JAX documentation](https://github.com/google/jax#pip-installation).

---

## Running the App

```bash
python app.py
```

Dash defaults to `http://127.0.0.1:8050/`. The layout is responsive, so you can resize the browser to focus on plots or the control sidebar.


---

## Usage Tips

1. **Upload** a VMEC NetCDF file (`wout_*.nc`) via the drag‑and‑drop area in the sidebar.
2. Use **Visualization Mode** to switch between:
   - **Summary Dashboard** (shows statistics + multi-panel plots),
   - **1D Profiles**, **2D Cross Sections**, **2D Flux Surfaces**, and **3D Geometry**.
3. Adjust **sliders** for toroidal angle (`phi`), flux surface index (`s`), and choose different physical quantities from the dropdowns.
4. Toggle **Coordinate-free background** in 3‑D mode for clean screenshots.
5. Click **Download Plot** to export the currently visible figure as a PNG.

The app caches equilibrium metadata, so switching modes or variables is fast.  
For heavy 2‑D physics overlays, a pre-computation step runs on the server while keeping the UI responsive.

---

## Repository Layout

| Path            | Description                                                      |
| --------------- | ---------------------------------------------------------------- |
| `app.py`        | Dash layout, callbacks, and UI logic.                            |
| `test_app.py`   | Dash layout, callbacks, and UI logic. Better than upper                          |
| `vmec_jax.py`   | JAX-powered data processor for VMEC equilibria.                  |
| `requirements.txt` | Python dependencies.                                          |
| `wout_PO.nc`    | Example VMEC equilibrium (use your own files for new cases).     |

---

## Troubleshooting

- **JAX import errors on Windows**: ensure you installed `jax[cpu]` and that no conflicting CUDA/TensorFlow packages are present.
- **Blank downloads**: make sure the plot has finished rendering before clicking the download button.
- **Large NetCDF files**: the processor uses lazy loading via `xarray`, but very high resolution VMEC outputs can still require significant RAM.

Feel free to open issues or pull requests to add new physical quantities, UI tweaks, or performance optimisations. Enjoy exploring your VMEC equilibria! 
