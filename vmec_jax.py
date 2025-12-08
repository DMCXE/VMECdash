from __future__ import annotations

import math
import numbers
import os
from dataclasses import dataclass
from functools import partial
from typing import Callable, Mapping

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from matplotlib.path import Path
from scipy.interpolate import griddata

# Enable double precision for stellarator equilibria
jax.config.update("jax_enable_x64", True)

EPS = 1e-12


# ---------------------------------------------------------------------------
# Definition helpers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ProfileSpec:
    """Definition of a flux-surface-averaged profile."""

    key: str
    label: str
    source: str
    category: str = "Profiles"
    transform: Callable | None = None
    requires_s: bool = False

    def option(self) -> dict:
        return {"value": self.key, "label": self.label, "category": self.category}


@dataclass(frozen=True)
class FieldSpec:
    """Definition of a 2D/3D field on (theta, zeta)."""

    key: str
    label: str
    cos_source: str | None
    sin_source: str | None = None
    category: str = "Fields"

    def option(self) -> dict:
        return {"value": self.key, "label": self.label, "category": self.category}


def _mode_from_name(var_name: str) -> str:
    return "sin" if var_name.endswith("mns") else "cos"


PROFILE_SPECS: Mapping[str, ProfileSpec] = {
    "iotaf": ProfileSpec("iotaf", "Rotational Transform (iota)", "iotaf", category="Core"),
    "q": ProfileSpec(
        "q",
        "Safety Factor (q)",
        "iotaf",
        category="Core",
        transform=lambda arr: 1.0 / (arr + EPS),
    ),
    "presf": ProfileSpec("presf", "Pressure", "presf", category="Pressure"),
    "betapol": ProfileSpec("betapol", "beta_pol", "betapol", category="Pressure"),
    "betator": ProfileSpec("betator", "beta_tor", "betator", category="Pressure"),
    "beta_vol": ProfileSpec("beta_vol", "beta_vol", "beta_vol", category="Pressure"),
    "phi": ProfileSpec("phi", "Toroidal Flux", "phi", category="Flux"),
    "phip": ProfileSpec("phip", "dPhi/ds", "phipf", category="Flux"),
    "vp": ProfileSpec("vp", "Volume Derivative (Vp)", "vp", category="Flux"),
    "overr": ProfileSpec("overr", "1/R", "over_r", category="Geometry"),
    "buco": ProfileSpec("buco", "<B^u>", "buco", category="Magnetic"),
    "bvco": ProfileSpec("bvco", "<B^v>", "bvco", category="Magnetic"),
    "jcuru": ProfileSpec("jcuru", "<j^u>", "jcuru", category="Current"),
    "jcurv": ProfileSpec("jcurv", "<j^v>", "jcurv", category="Current"),
    "bdotb": ProfileSpec("bdotb", "<B·B>", "bdotb", category="Magnetic"),
    "DMerc": ProfileSpec("DMerc", "Mercier D", "DMerc", category="Stability"),
    "DShear": ProfileSpec("DShear", "Shear D", "DShear", category="Stability"),
    "DWell": ProfileSpec("DWell", "Well D", "DWell", category="Stability"),
    "DCurr": ProfileSpec("DCurr", "Current D", "DCurr", category="Stability"),
    "DGeod": ProfileSpec("DGeod", "Geodesic D", "DGeod", category="Stability"),
    "jdotb": ProfileSpec("jdotb", "<J·B>", "jdotb", category="Current"),
    "bdotgradv": ProfileSpec("bdotgradv", "<B·∇v>", "bdotgradv", category="Magnetic"),
    "specw": ProfileSpec("specw", "Spectral Width", "specw", category="Diagnostics"),
    "dpds": ProfileSpec(
        "dpds",
        "dP/ds",
        "presf",
        category="Pressure",
        requires_s=True,
        transform=lambda arr, s=None: jnp.gradient(arr, s) if s is not None else arr,
    ),
}

FIELD_SPECS: Mapping[str, FieldSpec] = {
    "geometry": FieldSpec("geometry", "Geometry Only", None, category="Geometry"),
    "modB": FieldSpec("modB", "|B| (Mod B)", "bmnc", sin_source="bmns", category="Magnetic"),
    "jacobian": FieldSpec("jacobian", "sqrt(g)", "gmnc", sin_source="gmns", category="Metric"),
    "lambda": FieldSpec("lambda", "Lambda", "lmnc", sin_source="lmns", category="Metric"),
    "B_s": FieldSpec("B_s", "B_s (covariant)", "bsubsmnc", sin_source="bsubsmns", category="Magnetic"),
    "B_u": FieldSpec("B_u", "B_u (covariant)", "bsubumnc", sin_source="bsubumns", category="Magnetic"),
    "B_v": FieldSpec("B_v", "B_v (covariant)", "bsubvmnc", sin_source="bsubvmns", category="Magnetic"),
    "B^u": FieldSpec("B^u", "B^u (contravariant)", "bsupumnc", sin_source="bsupumns", category="Magnetic"),
    "B^v": FieldSpec("B^v", "B^v (contravariant)", "bsupvmnc", sin_source="bsupvmns", category="Magnetic"),
    "j^u": FieldSpec("j^u", "j^u", "currumnc", sin_source="currumns", category="Current"),
    "j^v": FieldSpec("j^v", "j^v", "currvmnc", sin_source="currvmns", category="Current"),
}


def _load_coefficients(ds: xr.Dataset, specs: Mapping[str, FieldSpec | ProfileSpec]) -> dict:
    arrays = {}
    for spec in specs.values():
        src = spec.source
        if src and src in ds and src not in arrays:
            arrays[src] = jnp.asarray(ds[src].values)
    return arrays


def _align_profile(s_axis: jnp.ndarray, arr: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Ensure the profile array matches the expected s-grid length."""
    arr = jnp.asarray(arr)
    if arr.ndim == 0 or arr.shape == ():
        arr = jnp.full_like(s_axis, arr)
        return s_axis, arr
    if arr.shape[0] == s_axis.shape[0]:
        return s_axis, arr
    if arr.shape[0] == s_axis.shape[0] - 1:
        return s_axis[1:], arr
    new_s = jnp.linspace(0.0, 1.0, arr.shape[0])
    return new_s, arr


# ---------------------------------------------------------------------------
# Public processing class
# ---------------------------------------------------------------------------
class VmecPostProcessor:
    """
    Post-processing helper for VMEC wout files.

    The API is intentionally small and documented so it can be used from the
    Dash app or directly in notebooks. The heavy kernels live at the bottom of
    this file and are shared across all public methods.
    """

    _CACHE: dict[str, "VmecPostProcessor"] = {}

    # ---- Construction -----------------------------------------------------
    @classmethod
    def from_file(cls, nc_path: str) -> "VmecPostProcessor":
        """
        Create (or reuse) a processor from a VMEC NetCDF file.

        Instances are cached by absolute path + mtime so repeated calls are
        fast but react to file changes.
        """
        path = os.path.abspath(nc_path)
        mtime = os.path.getmtime(path)
        cached = cls._CACHE.get(path)
        if cached and getattr(cached, "_mtime", None) == mtime:
            return cached
        inst = cls(path)
        cls._CACHE[path] = inst
        return inst

    def __init__(self, nc_path: str):
        self.path = os.path.abspath(nc_path)
        self._mtime = os.path.getmtime(self.path)
        self.ds = xr.open_dataset(self.path)
        self.ns = int(self.ds.sizes["radius"])
        self.nfp = int(np.asarray(self.ds["nfp"]).item())
        self.lasym = self.ds["lasym__logical__"].values.astype(bool).item()
        self.xm = jnp.asarray(self.ds["xm"].values)
        self.xn = jnp.asarray(self.ds["xn"].values)
        self.xm_nyq = jnp.asarray(self.ds["xm_nyq"].values)
        self.xn_nyq = jnp.asarray(self.ds["xn_nyq"].values)
        self.rmnc = jnp.asarray(self.ds["rmnc"].values)
        self.zmns = jnp.asarray(self.ds["zmns"].values)
        self.rmns = jnp.asarray(self.ds["rmns"].values) if "rmns" in self.ds else jnp.zeros_like(self.rmnc)
        self.zmnc = jnp.asarray(self.ds["zmnc"].values) if "zmnc" in self.ds else jnp.zeros_like(self.zmns)
        self._s_grid = jnp.linspace(0.0, 1.0, self.ns)

        self._profile_arrays = _load_coefficients(self.ds, PROFILE_SPECS)
        self._field_pairs: dict[str, tuple[jnp.ndarray, jnp.ndarray]] = {}
        self._load_field_pairs()
        self._profile_alias = {spec.label: key for key, spec in PROFILE_SPECS.items()}
        self._field_alias = {spec.label: key for key, spec in FIELD_SPECS.items()}
        self._theta_cache: dict[int, jnp.ndarray] = {}
        self._lcfs_cache: dict[tuple[float, int], tuple[jnp.ndarray, jnp.ndarray]] = {}
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self.ds.close()
        self._closed = True
        cached = self._CACHE.get(self.path)
        if cached is self:
            self._CACHE.pop(self.path, None)

    # ---- Coefficient helpers --------------------------------------------
    def _coeff_pair(self, cos_name: str, sin_name: str | None) -> tuple[jnp.ndarray, jnp.ndarray] | None:
        cos = jnp.asarray(self.ds[cos_name].values) if cos_name in self.ds else None
        sin = jnp.asarray(self.ds[sin_name].values) if (sin_name and sin_name in self.ds) else None
        if cos is None and sin is None:
            return None
        if cos is None:
            cos = jnp.zeros_like(sin)
        if sin is None:
            sin = jnp.zeros_like(cos)
        if not self.lasym:
            sin = jnp.zeros_like(cos)
        return cos, sin

    def _load_field_pairs(self) -> None:
        for key, spec in FIELD_SPECS.items():
            if key == "geometry":
                continue
            if spec.cos_source is None:
                continue
            pair = self._coeff_pair(spec.cos_source, spec.sin_source)
            if pair is not None:
                self._field_pairs[key] = pair

    # ---- Metadata / helpers ----------------------------------------------
    def _get_theta(self, resolution: int) -> jnp.ndarray:
        if resolution not in self._theta_cache:
            self._theta_cache[resolution] = jnp.linspace(0.0, 2 * jnp.pi, resolution)
        return self._theta_cache[resolution]

    def _sanitize_s(self, s_idx: int) -> int:
        if s_idx < 0:
            s_idx += self.ns
        return int(jnp.clip(s_idx, 0, self.ns - 1))

    def _sample_s_indices(self, res_s: int) -> jnp.ndarray:
        if res_s >= self.ns:
            return jnp.arange(self.ns, dtype=int)
        raw = jnp.linspace(0, self.ns - 1, res_s)
        return jnp.unique(jnp.asarray(jnp.round(raw), dtype=int))

    def _profile_key(self, var_name: str) -> str:
        return self._profile_alias.get(var_name, var_name)

    def _field_key(self, var_name: str) -> str:
        return self._field_alias.get(var_name, var_name)

    # ---- Available variables ---------------------------------------------
    def available_profiles(self) -> list[dict]:
        payload = []
        for key, spec in PROFILE_SPECS.items():
            src = spec.source
            if src and src in self._profile_arrays:
                payload.append(spec.option())
        return payload

    def available_fields(self) -> list[dict]:
        payload = []
        for key, spec in FIELD_SPECS.items():
            if key == "geometry":
                payload.append(spec.option())
                continue
            if key in self._field_pairs:
                payload.append(spec.option())
        return payload

    # ---- Scalars / summary information -----------------------------------
    def _scalar(self, key: str, default: float = jnp.nan) -> float:
        if key not in self.ds:
            return float(default)
        data = self.ds[key].values
        if isinstance(data, numbers.Number):
            return float(data)
        return float(jnp.asarray(data).ravel()[-1])

    def _logical(self, key: str, default: bool = False) -> bool:
        if key not in self.ds:
            return default
        data = self.ds[key].values
        try:
            return bool(np.asarray(data).item())
        except Exception:
            return default

    def _string(self, key: str) -> str:
        if key not in self.ds:
            return ""
        data = self.ds[key].values
        if isinstance(data, bytes):
            return data.decode("utf-8", errors="ignore").strip()
        arr = np.asarray(data)
        if arr.dtype.kind in {"S", "U"}:
            flat = "".join(arr.astype(str).tolist())
            return flat.strip()
        if arr.ndim == 0:
            return str(arr.item()).strip()
        return str(arr).strip()

    def get_scalars(self) -> dict[str, float]:
        scalars = {
            "beta_total": self._scalar("betatotal", 0.0),
            "betapol": self._scalar("betapol", 0.0),
            "betator": self._scalar("betator", 0.0),
            "volume": self._scalar("volume_p", 0.0),
            "aspect": self._scalar("aspect", 0.0),
            "b0": self._scalar("b0", 0.0),
            "Rmajor": self._scalar("Rmajor_p", jnp.nan),
            "Aminor": self._scalar("Aminor_p", jnp.nan),
            "rbtor": self._scalar("rbtor", jnp.nan),
            "ctor": self._scalar("ctor", 0.0),
        }
        iota = self._profile_arrays.get("iotaf")
        pres = self._profile_arrays.get("presf")
        if iota is not None:
            scalars["iota_axis"] = float(iota[0])
            scalars["iota_edge"] = float(iota[-1])
            scalars["q_axis"] = float(1.0 / (iota[0] + EPS))
            scalars["q_edge"] = float(1.0 / (iota[-1] + EPS))
            shear = np.gradient(iota, self._s_grid) # jnp.gradient causes issues here on windows
            scalars["shear_edge"] = float(jnp.asarray(shear)[-1])
        if pres is not None:
            scalars["pressure_axis"] = float(pres[0])
            scalars["pressure_edge"] = float(pres[-1])
        return scalars

    def get_summary_lines(self) -> list[str]:
        scalars = self.get_scalars()
        lines = []
        if "iota_axis" in scalars:
            lines.append(
                f"iota_axis={scalars['iota_axis']:.3f}, iota_edge={scalars['iota_edge']:.3f}, "
                f"q_axis={scalars['q_axis']:.3f}"
            )
        if "shear_edge" in scalars:
            pe = scalars.get("pressure_edge", jnp.nan)
            lines.append(f"edge shear={scalars['shear_edge']:.3f}, pressure_edge={pe:.3e}")
        lines.append(
            f"beta_total={scalars['beta_total']*100:.2f}% | volume={scalars['volume']:.3f} m^3 | B0={scalars['b0']:.3f} T"
        )
        if not math.isnan(scalars.get("Rmajor", jnp.nan)):
            lines.append(
                f"R_major={scalars['Rmajor']:.3f} m, a_minor={scalars['Aminor']:.3f} m, aspect={scalars['aspect']:.2f}"
            )
        boundary_free = self._logical("lfreeb__logical__", False)
        mgrid = self._string("mgrid_file")
        if boundary_free or mgrid:
            boundary_tag = "Free-boundary" if boundary_free else "Fixed-boundary"
            mgrid_txt = mgrid if mgrid else "no mgrid"
            lines.append(f"{boundary_tag} equilibrium (mgrid: {mgrid_txt})")
        lines.append(f"toroidal current={scalars['ctor']:.2f} MA, RB_tor={scalars['rbtor']:.3f} T·m")
        return lines

    # ---- 1D profiles ------------------------------------------------------
    def get_1d_data(self, var_name: str):
        key = self._profile_key(var_name)
        spec = PROFILE_SPECS.get(key)
        base = self._profile_arrays.get(spec.source) if spec else None
        if spec is None or base is None:
            s_np = np.asarray(self._s_grid)
            return s_np, np.zeros_like(s_np)
        data = base
        if spec.requires_s:
            data = spec.transform(data, self._s_grid) if spec.transform else data
        elif spec.transform:
            data = spec.transform(data)
        s_axis, data = _align_profile(self._s_grid, data)
        return np.asarray(s_axis[1:]), np.asarray(data[1:])

    # ---- 2D slices --------------------------------------------------------
    def _field_payload(self, var_name: str) -> dict | None:
        key = self._field_key(var_name)
        spec = FIELD_SPECS.get(key)
        if not spec:
            return None
        if key == "geometry":
            r_pair = self._coeff_pair("rmnc", "rmns")
            z_pair = self._coeff_pair("zmnc", "zmns")
            if r_pair is None or z_pair is None:
                return None
            return {"key": key, "label": spec.label, "r_pair": r_pair, "z_pair": z_pair}
        pair = self._field_pairs.get(key)
        if pair is None:
            return None
        cos_coeffs, _ = pair
        use_nyq = cos_coeffs.shape[-1] == self.xm_nyq.shape[0]
        return {
            "key": key,
            "label": spec.label,
            "pair": pair,
            "xm": self.xm_nyq if use_nyq else self.xm,
            "xn": self.xn_nyq if use_nyq else self.xn,
        }

    def get_flux_surface_data(self, s_idx: int, var_name: str, res_u: int = 128, res_v: int = 128):
        payload = self._field_payload(var_name)
        if payload is None or payload.get("pair") is None:
            return None, None, None
        idx = self._sanitize_s(s_idx)
        theta = self._get_theta(res_u)
        zeta = jnp.linspace(0.0, 2 * jnp.pi / self.nfp, res_v)
        cos_coeffs, sin_coeffs = payload["pair"]
        cos_slice = cos_coeffs[idx]
        sin_slice = sin_coeffs[idx]
        val = _jit_evaluate_2d_pair(cos_slice, sin_slice, payload["xm"], payload["xn"], theta, zeta)
        return np.asarray(theta), np.asarray(zeta), np.asarray(val)

    def get_cross_section_data(self, phi: float, var_name: str, res_s: int = 48, res_u: int = 160):
        theta = self._get_theta(res_u)
        s_indices = self._sample_s_indices(res_s)
        r_grid, z_grid = _jit_cross_section_batch(
            self.rmnc[s_indices],
            self.rmns[s_indices],
            self.zmns[s_indices],
            self.zmnc[s_indices],
            self.xm,
            self.xn,
            theta,
            phi,
        )
        payload = self._field_payload(var_name)
        if payload is None or payload.get("pair") is None or payload.get("key") == "geometry":
            val_grid = jnp.tile((s_indices / (self.ns - 1))[:, None], (1, theta.size))
        else:
            cos_coeffs, sin_coeffs = payload["pair"]
            cos_slice = cos_coeffs[s_indices]
            sin_slice = sin_coeffs[s_indices]
            if cos_slice.shape[0] > 1 and s_indices[0] == 0:
                cos_slice = cos_slice.at[0].set(cos_slice[1])
                sin_slice = sin_slice.at[0].set(sin_slice[1])
            val_grid = _jit_evaluate_line_pair(cos_slice, sin_slice, payload["xm"], payload["xn"], theta, phi)
        return np.asarray(r_grid), np.asarray(z_grid), np.asarray(val_grid)

    def _lcfs(self, phi: float, resolution: int = 256):
        key = (float(phi), resolution)
        if key not in self._lcfs_cache:
            theta = self._get_theta(resolution)
            r, z = _jit_cross_section_batch(
                self.rmnc[-1], self.rmns[-1], self.zmns[-1], self.zmnc[-1], self.xm, self.xn, theta, phi
            )
            self._lcfs_cache[key] = (r.reshape(-1), z.reshape(-1))
        return self._lcfs_cache[key]

    def get_cross_section_grid(self, phi: float, var_name: str, res_grid: int = 200):
        r_raw, z_raw, val_raw = self.get_cross_section_data(phi, var_name)
        r_lcfs, z_lcfs = self._lcfs(phi)
        points = np.column_stack((np.asarray(r_raw).reshape(-1), np.asarray(z_raw).reshape(-1)))
        values = np.asarray(val_raw).reshape(-1)
        r_min, r_max = float(jnp.min(r_lcfs)), float(jnp.max(r_lcfs))
        z_min, z_max = float(jnp.min(z_lcfs)), float(jnp.max(z_lcfs))
        pad_r = (r_max - r_min) * 0.05
        pad_z = (z_max - z_min) * 0.05
        r_lin = jnp.linspace(r_min - pad_r, r_max + pad_r, res_grid)
        z_lin = jnp.linspace(z_min - pad_z, z_max + pad_z, res_grid)
        r_mesh, z_mesh = jnp.meshgrid(r_lin, z_lin)
        r_mesh_np = np.asarray(r_mesh)
        z_mesh_np = np.asarray(z_mesh)
        val_grid = griddata(points, values, (r_mesh_np, z_mesh_np), method="linear")
        lcfs_poly = np.column_stack((np.asarray(r_lcfs), np.asarray(z_lcfs)))
        mask = Path(lcfs_poly).contains_points(np.column_stack((r_mesh_np.ravel(), z_mesh_np.ravel())))
        mask = mask.reshape(r_mesh_np.shape)
        val_grid[~mask] = np.nan
        return np.asarray(r_lin), np.asarray(z_lin), val_grid

    # ---- 3D geometry ------------------------------------------------------
    def compute_3d_surface(self, s_idx: int = -1, var_name: str = "modB", resolution: int = 128):
        idx = self._sanitize_s(s_idx)
        theta = self._get_theta(resolution)
        phi = self._get_theta(resolution)
        payload = self._field_payload(var_name)
        xm_var, xn_var = self.xm, self.xn
        var_cos = jnp.zeros_like(self.rmnc[idx])
        var_sin = jnp.zeros_like(self.rmnc[idx])
        if payload and payload.get("pair") is not None and payload.get("key") != "geometry":
            var_cos_full, var_sin_full = payload["pair"]
            var_cos = var_cos_full[idx]
            var_sin = var_sin_full[idx]
            xm_var, xn_var = payload["xm"], payload["xn"]
        x, y, z, val = _jit_surface_3d(
            self.rmnc[idx],
            self.rmns[idx],
            self.zmns[idx],
            self.zmnc[idx],
            var_cos,
            var_sin,
            self.xm,
            self.xn,
            xm_var,
            xn_var,
            theta,
            phi,
        )
        return np.asarray(x), np.asarray(y), np.asarray(z), np.asarray(val)

    # ---- Field line tracing ----------------------------------------------
    def compute_field_line_properties(
        self,
        s_idx: int,
        alpha_points: int = 256,
        zeta_points: int = 256,
        n_transits: int = 0,
        alpha0: float = 0.0,
        single_line: bool = False,
        zeta_offset: float = 0.0,
    ):
        """
        Compute |B| in (alpha, zeta) coordinates.

        If ``single_line`` is True, compute one field line over ``n_transits``.
        Otherwise compute a grid of alpha values on a single toroidal period.
        """
        idx = self._sanitize_s(s_idx)
        lambda_pair = self._coeff_pair("lmnc", "lmns")
        b_pair = self._coeff_pair("bmnc", "bmns")
        if lambda_pair is None or b_pair is None:
            return None, None, None
        lmnc, lmns = lambda_pair
        bmnc, bmns = b_pair
        iota = self._profile_arrays.get("iotaf")[idx]

        xm_l = self.xm if lmnc.shape[-1] == self.xm.shape[0] else self.xm_nyq
        xn_l = self.xn if lmnc.shape[-1] == self.xn.shape[0] else self.xn_nyq
        xm_b = self.xm if bmnc.shape[-1] == self.xm.shape[0] else self.xm_nyq
        xn_b = self.xn if bmnc.shape[-1] == self.xn.shape[0] else self.xn_nyq

        params = {
            "nfp": self.nfp,
            "iota": iota,
            "xm_b": xm_b,
            "xn_b": xn_b,
            "xm_lambda": xm_l,
            "xn_lambda": xn_l,
            "bmnc": bmnc[idx],
            "bmns": bmns[idx],
            "lmnc": lmnc[idx],
            "lmns": lmns[idx],
        }

        zeta_max = 2.0 * jnp.pi / self.nfp
        zeta_grid = jnp.linspace(0.0, zeta_max, zeta_points, endpoint=False) + float(zeta_offset)

        if single_line:
            n_transits = max(1, n_transits)
            alpha_segments = compute_shifted_alphas(alpha0, n_transits, params)
            b_data = compute_all_grid(alpha_segments, zeta_grid, params)
            return np.asarray(zeta_grid), np.asarray(alpha_segments), np.asarray(b_data)

        alpha_grid = jnp.linspace(0.0, 2.0 * jnp.pi, alpha_points, endpoint=False)
        b_data = compute_all_grid(alpha_grid, zeta_grid, params)
        return np.asarray(alpha_grid), np.asarray(zeta_grid), np.asarray(b_data)


# ---------------------------------------------------------------------------
# JAX kernels
# ---------------------------------------------------------------------------
@jax.jit
def _jit_cross_section_batch(rmnc, rmns, zmns, zmnc, xm, xn, theta, phi):
    if rmnc.ndim == 1:
        rmnc = rmnc[None, :]
        rmns = rmns[None, :]
        zmns = zmns[None, :]
        zmnc = zmnc[None, :]
    angle = jnp.outer(xm, theta) - jnp.outer(xn, jnp.full_like(theta, phi))
    cos_terms = jnp.cos(angle)
    sin_terms = jnp.sin(angle)
    r = (rmnc @ cos_terms) + (rmns @ sin_terms)
    z = (zmns @ sin_terms) + (zmnc @ cos_terms)
    return r, z


@jax.jit
def _jit_evaluate_line_pair(cos_coeffs, sin_coeffs, xm, xn, theta, phi):
    if cos_coeffs.ndim == 1:
        cos_coeffs = cos_coeffs[None, :]
        sin_coeffs = sin_coeffs[None, :]
    angle = jnp.outer(xm, theta) - jnp.outer(xn, jnp.full_like(theta, phi))
    cos_trig = jnp.cos(angle)
    sin_trig = jnp.sin(angle)
    return (cos_coeffs @ cos_trig) + (sin_coeffs @ sin_trig)


@jax.jit
def _jit_evaluate_2d_pair(cos_coeffs, sin_coeffs, xm, xn, theta, zeta):
    theta_grid, zeta_grid = jnp.meshgrid(theta, zeta, indexing="ij")
    angle = (xm[:, None, None] * theta_grid[None, :, :]) - (xn[:, None, None] * zeta_grid[None, :, :])
    cos_trig = jnp.cos(angle)
    sin_trig = jnp.sin(angle)
    return jnp.sum(cos_coeffs[:, None, None] * cos_trig + sin_coeffs[:, None, None] * sin_trig, axis=0)


@jax.jit
def _jit_surface_3d(r_cos, r_sin, z_sin, z_cos, var_cos, var_sin, xm, xn, xm_var, xn_var, theta, phi):
    theta_grid, phi_grid = jnp.meshgrid(theta, phi, indexing="ij")
    angle_geom = (xm[:, None, None] * theta_grid[None, :, :]) - (xn[:, None, None] * phi_grid[None, :, :])
    cos_geom = jnp.cos(angle_geom)
    sin_geom = jnp.sin(angle_geom)
    r = jnp.sum(r_cos[:, None, None] * cos_geom + r_sin[:, None, None] * sin_geom, axis=0)
    z_val = jnp.sum(z_sin[:, None, None] * sin_geom + z_cos[:, None, None] * cos_geom, axis=0)
    x_val = r * jnp.cos(phi_grid)
    y_val = r * jnp.sin(phi_grid)

    angle_var = (xm_var[:, None, None] * theta_grid[None, :, :]) - (xn_var[:, None, None] * phi_grid[None, :, :])
    cos_var = jnp.cos(angle_var)
    sin_var = jnp.sin(angle_var)
    val = jnp.sum(var_cos[:, None, None] * cos_var + var_sin[:, None, None] * sin_var, axis=0)
    return x_val, y_val, z_val, val


# ----------------------------------------------------------------------
# Field line tracing kernels (from plot_az_jax.py)
# ----------------------------------------------------------------------
@jax.jit
def eval_Lambda(theta, zeta, xm, xn, lmnc, lmns):
    arg = xm * theta - xn * zeta
    return jnp.sum(lmnc * jnp.cos(arg) + lmns * jnp.sin(arg))


@jax.jit
def eval_B(theta, zeta, xm, xn, bmnc, bmns):
    arg = xm * theta - xn * zeta
    return jnp.sum(bmnc * jnp.cos(arg) + bmns * jnp.sin(arg))


@jax.jit
def residual_fn(theta, alpha, zeta, iota, xm_l, xn_l, lmnc, lmns):
    lam = eval_Lambda(theta, zeta, xm_l, xn_l, lmnc, lmns)
    return theta + lam - iota * zeta - alpha


@partial(jax.jit, static_argnames=["max_iter"])
def newton_solver(alpha, zeta, theta_init, iota, xm_l, xn_l, lmnc, lmns, tol=1e-12, max_iter=100):
    def cond_fun(state):
        _, diff, count = state
        return (diff > tol) & (count < max_iter)

    def body_fun(state):
        theta, _, count = state
        f_val, d_f_dtheta = jax.value_and_grad(residual_fn, argnums=0)(
            theta, alpha, zeta, iota, xm_l, xn_l, lmnc, lmns
        )
        step = -f_val / d_f_dtheta
        theta_new = theta + step
        return theta_new, jnp.abs(step), count + 1

    init_state = (theta_init, 1.0, 0)
    final_state = jax.lax.while_loop(cond_fun, body_fun, init_state)
    return final_state[0]


@jax.jit
def compute_field_line_scan(alpha, zeta_grid, params):
    iota = params["iota"]
    xm_l = params["xm_lambda"]
    xn_l = params["xn_lambda"]
    lmnc = params["lmnc"]
    lmns = params["lmns"]
    xm_b = params["xm_b"]
    xn_b = params["xn_b"]
    bmnc = params["bmnc"]
    bmns = params["bmns"]

    theta_init = alpha + iota * zeta_grid[0]

    def scan_body(theta_prev, zeta):
        theta_sol = newton_solver(alpha, zeta, theta_prev, iota, xm_l, xn_l, lmnc, lmns, tol=1e-12, max_iter=100)
        b_val = eval_B(theta_sol, zeta, xm_b, xn_b, bmnc, bmns)
        return theta_sol, b_val

    _, b_line = jax.lax.scan(scan_body, theta_init, zeta_grid)
    return b_line


@jax.jit
def compute_all_grid(alpha_grid, zeta_grid, params):
    return jax.vmap(compute_field_line_scan, in_axes=(0, None, None))(alpha_grid, zeta_grid, params)


@partial(jax.jit, static_argnames=["n_transits"])
def compute_shifted_alphas(alpha0, n_transits, params):
    iota = params["iota"]
    nfp = params["nfp"]
    k_vals = jnp.arange(n_transits)
    delta_alpha = iota * (2.0 * jnp.pi / nfp)
    alphas = (alpha0 + k_vals * delta_alpha) % (2.0 * jnp.pi)
    return alphas


# Public alias preserved for the Dash app
VMECJaxProcessor = VmecPostProcessor
