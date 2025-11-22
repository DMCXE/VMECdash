import math
import numbers
import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from matplotlib.path import Path
from scipy.interpolate import griddata

# Enable double precision for stellarator equilibria
jax.config.update("jax_enable_x64", True)

EPS = 1e-12


PROFILE_DEFS = {
    "iotaf": {"label": "Rotational Transform (iota)", "source": "iotaf", "category": "Core"},
    "q": {
        "label": "Safety Factor (q)",
        "source": "iotaf",
        "transform": lambda arr: 1.0 / (arr + EPS),
        "category": "Core",
    },
    "presf": {"label": "Pressure", "source": "presf", "category": "Pressure"},
    "betapol": {"label": "beta_pol", "source": "betapol", "category": "Pressure"},
    "betator": {"label": "beta_tor", "source": "betator", "category": "Pressure"},
    "beta_vol": {"label": "beta_vol", "source": "beta_vol", "category": "Pressure"},
    "phi": {"label": "Toroidal Flux", "source": "phi", "category": "Flux"},
    "phip": {"label": "dPhi/ds", "source": "phipf", "category": "Flux"},
    "vp": {"label": "Enclosed Volume", "source": "vp", "category": "Flux"},
    "overr": {"label": "1/R", "source": "over_r", "category": "Geometry"},
    "buco": {"label": "<B^u>", "source": "buco", "category": "Magnetic"},
    "bvco": {"label": "<B^v>", "source": "bvco", "category": "Magnetic"},
    "jcuru": {"label": "<j^u>", "source": "jcuru", "category": "Current"},
    "jcurv": {"label": "<j^v>", "source": "jcurv", "category": "Current"},
    "bdotb": {"label": "<B·B>", "source": "bdotb", "category": "Magnetic"},
    "DMerc": {"label": "Mercier D", "source": "DMerc", "category": "Stability"},
    "DShear": {"label": "Shear D", "source": "DShear", "category": "Stability"},
    "DWell": {"label": "Well D", "source": "DWell", "category": "Stability"},
    "DCurr": {"label": "Current D", "source": "DCurr", "category": "Stability"},
    "DGeod": {"label": "Geodesic D", "source": "DGeod", "category": "Stability"},
    "jdotb": {"label": "<J·B>", "source": "jdotb", "category": "Current"},
    "bdotgradv": {"label": "<B·∇v>", "source": "bdotgradv", "category": "Magnetic"},
    "specw": {"label": "Spectral Width", "source": "specw", "category": "Diagnostics"},
    "dpds": {
        "label": "dP/ds",
        "source": "presf",
        "transform": lambda arr, s=None: jnp.gradient(arr, s) if s is not None else arr,
        "requires_s": True,
        "category": "Pressure",
    },
}


FIELD_DEFS = {
    "geometry": {"label": "Geometry Only", "category": "Geometry"},
    "modB": {"label": "|B| (Mod B)", "source": "bmnc", "category": "Magnetic"},
    "jacobian": {"label": "sqrt(g)", "source": "gmnc", "category": "Metric"},
    "lambda": {"label": "Lambda", "source": "lmns", "category": "Metric"},
    "B_s": {"label": "B_s (covariant)", "source": "bsubsmns", "category": "Magnetic"},
    "B_u": {"label": "B_u (covariant)", "source": "bsubumnc", "category": "Magnetic"},
    "B_v": {"label": "B_v (covariant)", "source": "bsubvmnc", "category": "Magnetic"},
    "B^u": {"label": "B^u (contravariant)", "source": "bsupumnc", "category": "Magnetic"},
    "B^v": {"label": "B^v (contravariant)", "source": "bsupvmnc", "category": "Magnetic"},
    "j^u": {"label": "j^u", "source": "currumnc", "category": "Current"},
    "j^v": {"label": "j^v", "source": "currvmnc", "category": "Current"},
}


def _mode_from_name(var_name: str) -> str:
    return "sin" if var_name.endswith("mns") else "cos"


class VMECJaxProcessor:
    """JAX accelerated helper around VMEC wout files."""

    _CACHE = {}

    @classmethod
    def from_file(cls, nc_path: str) -> "VMECJaxProcessor":
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
        self.nfp = int(self.ds["nfp"].values.item())
        self.xm = jnp.asarray(self.ds["xm"].values)
        self.xn = jnp.asarray(self.ds["xn"].values)
        self.xm_nyq = jnp.asarray(self.ds["xm_nyq"].values)
        self.xn_nyq = jnp.asarray(self.ds["xn_nyq"].values)
        self.rmnc = jnp.asarray(self.ds["rmnc"].values)
        self.zmns = jnp.asarray(self.ds["zmns"].values)
        self._profile_arrays = {}
        self._field_arrays = {}
        self._profile_alias = {spec["label"]: key for key, spec in PROFILE_DEFS.items()}
        self._field_alias = {spec["label"]: key for key, spec in FIELD_DEFS.items()}
        self._load_arrays()
        self._theta_cache = {}
        self._lcfs_cache = {}
        self._s_grid = jnp.linspace(0.0, 1.0, self.ns)

    def _load_arrays(self) -> None:
        for spec in PROFILE_DEFS.values():
            src = spec.get("source")
            if src and src in self.ds and src not in self._profile_arrays:
                self._profile_arrays[src] = jnp.asarray(self.ds[src].values)
        for spec in FIELD_DEFS.values():
            src = spec.get("source")
            if src and src in self.ds and src not in self._field_arrays:
                self._field_arrays[src] = jnp.asarray(self.ds[src].values)

    def close(self) -> None:
        if getattr(self, "_closed", False):
            return
        self.ds.close()
        self._closed = True
        cached = self._CACHE.get(self.path)
        if cached is self:
            self._CACHE.pop(self.path, None)

    # ------------------------------------------------------------------
    # Helpers / metadata
    # ------------------------------------------------------------------
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

    def _align_profile(self, s_axis: jnp.ndarray, arr: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
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

    def _profile_key(self, var_name: str) -> str:
        return self._profile_alias.get(var_name, var_name)

    def _field_key(self, var_name: str) -> str:
        return self._field_alias.get(var_name, var_name)

    def available_profiles(self) -> list[dict]:
        payload = []
        for key, spec in PROFILE_DEFS.items():
            src = spec.get("source")
            if src and src in self._profile_arrays:
                payload.append(
                    {"value": key, "label": spec["label"], "category": spec.get("category", "Profiles")}
                )
        return payload

    def available_fields(self) -> list[dict]:
        payload = []
        for key, spec in FIELD_DEFS.items():
            if key == "geometry":
                payload.append({"value": key, "label": spec["label"], "category": spec.get("category", "")})
                continue
            src = spec.get("source")
            if src and src in self._field_arrays:
                payload.append(
                    {"value": key, "label": spec["label"], "category": spec.get("category", "Fields")}
                )
        return payload

    def _field_payload(self, var_name: str) -> dict | None:
        key = self._field_key(var_name)
        spec = FIELD_DEFS.get(key)
        if not spec:
            return None
        if key == "geometry":
            return {"key": key, "label": spec["label"], "coeffs": None}
        src = spec.get("source")
        coeffs = self._field_arrays.get(src)
        if coeffs is None:
            return None
        mode = spec.get("mode", _mode_from_name(src))
        use_nyq = coeffs.shape[-1] == self.xm_nyq.shape[0]
        return {
            "key": key,
            "label": spec["label"],
            "coeffs": coeffs,
            "mode": mode,
            "xm": self.xm_nyq if use_nyq else self.xm,
            "xn": self.xn_nyq if use_nyq else self.xn,
        }

    # ------------------------------------------------------------------
    # Scalars / summary information
    # ------------------------------------------------------------------
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
            try:
                shear = jnp.gradient(iota, self._s_grid)
            except NotImplementedError:
                shear = np.gradient(np.asarray(iota), np.asarray(self._s_grid))
            scalars["shear_edge"] = float(np.asarray(shear)[-1])
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

    # ------------------------------------------------------------------
    # 1D profiles
    # ------------------------------------------------------------------
    def get_1d_data(self, var_name: str):
        key = self._profile_key(var_name)
        spec = PROFILE_DEFS.get(key)
        base = self._profile_arrays.get(spec.get("source")) if spec else None
        if spec is None or base is None:
            s_np = np.asarray(self._s_grid)
            return s_np, np.zeros_like(s_np)
        data = base
        if spec.get("requires_s"):
            data = spec["transform"](data, self._s_grid)
        elif "transform" in spec:
            data = spec["transform"](data)
        s_axis, data = self._align_profile(self._s_grid, data)
        return np.asarray(s_axis[1:]), np.asarray(data[1:])

    # ------------------------------------------------------------------
    # 2D slices
    # ------------------------------------------------------------------
    def get_flux_surface_data(self, s_idx: int, var_name: str, res_u: int = 128, res_v: int = 128):
        payload = self._field_payload(var_name)
        if payload is None or payload.get("coeffs") is None:
            return None, None, None
        idx = self._sanitize_s(s_idx)
        theta = self._get_theta(res_u)
        zeta = jnp.linspace(0.0, 2 * jnp.pi / self.nfp, res_v)
        coeffs = payload["coeffs"][idx]
        val = _jit_evaluate_2d(coeffs, payload["xm"], payload["xn"], theta, zeta, payload["mode"])
        return np.asarray(theta), np.asarray(zeta), np.asarray(val)

    def get_cross_section_data(self, phi: float, var_name: str, res_s: int = 48, res_u: int = 160):
        theta = self._get_theta(res_u)
        s_indices = self._sample_s_indices(res_s)
        r_grid, z_grid = _jit_cross_section_batch(
            self.rmnc[s_indices], self.zmns[s_indices], self.xm, self.xn, theta, phi
        )
        payload = self._field_payload(var_name)
        if payload is None or payload.get("coeffs") is None or payload.get("key") == "geometry":
            val_grid = jnp.tile((s_indices / (self.ns - 1))[:, None], (1, theta.size))
        else:
            coeffs = payload["coeffs"][s_indices]
            if coeffs.shape[0] > 1 and s_indices[0] == 0:
                coeffs = coeffs.at[0].set(coeffs[1])
            val_grid = _jit_evaluate_line_batch(
                coeffs, payload["xm"], payload["xn"], theta, phi, payload["mode"]
            )
        return np.asarray(r_grid), np.asarray(z_grid), np.asarray(val_grid)

    def _lcfs(self, phi: float, resolution: int = 256):
        key = (float(phi), resolution)
        if key not in self._lcfs_cache:
            theta = self._get_theta(resolution)
            r, z = _jit_cross_section_batch(self.rmnc[-1], self.zmns[-1], self.xm, self.xn, theta, phi)
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

    # ------------------------------------------------------------------
    # 3D geometry
    # ------------------------------------------------------------------
    def compute_3d_surface(self, s_idx: int = -1, var_name: str = "modB", resolution: int = 128):
        idx = self._sanitize_s(s_idx)
        theta = self._get_theta(resolution)
        phi = self._get_theta(resolution)
        payload = self._field_payload(var_name)
        coeffs = None
        xm_var, xn_var, mode = self.xm, self.xn, "cos"
        if payload and payload.get("coeffs") is not None and payload.get("key") != "geometry":
            coeffs = payload["coeffs"][idx]
            xm_var, xn_var, mode = payload["xm"], payload["xn"], payload["mode"]
        x, y, z, val = _jit_surface_3d(
            self.rmnc[idx],
            self.zmns[idx],
            coeffs if coeffs is not None else jnp.zeros_like(self.rmnc[idx]),
            self.xm,
            self.xn,
            xm_var,
            xn_var,
            theta,
            phi,
            mode,
        )
        return np.asarray(x), np.asarray(y), np.asarray(z), np.asarray(val)


# ----------------------------------------------------------------------
# JAX kernels
# ----------------------------------------------------------------------
@jax.jit
def _jit_cross_section_batch(rmnc, zmns, xm, xn, theta, phi):
    if rmnc.ndim == 1:
        rmnc = rmnc[None, :]
        zmns = zmns[None, :]
    angle = jnp.outer(xm, theta) - jnp.outer(xn, jnp.full_like(theta, phi))
    cos_terms = jnp.cos(angle)
    sin_terms = jnp.sin(angle)
    r = rmnc @ cos_terms
    z = zmns @ sin_terms
    return r, z


@partial(jax.jit, static_argnames=["mode"])
def _jit_evaluate_line_batch(coeffs, xm, xn, theta, phi, mode):
    if coeffs.ndim == 1:
        coeffs = coeffs[None, :]
    angle = jnp.outer(xm, theta) - jnp.outer(xn, jnp.full_like(theta, phi))
    trig = jnp.cos(angle) if mode == "cos" else jnp.sin(angle)
    return coeffs @ trig


@partial(jax.jit, static_argnames=["mode"])
def _jit_evaluate_2d(coeffs, xm, xn, theta, zeta, mode):
    theta_grid, zeta_grid = jnp.meshgrid(theta, zeta, indexing="ij")
    angle = (xm[:, None, None] * theta_grid[None, :, :]) - (xn[:, None, None] * zeta_grid[None, :, :])
    trig = jnp.cos(angle) if mode == "cos" else jnp.sin(angle)
    return jnp.sum(coeffs[:, None, None] * trig, axis=0)


@partial(jax.jit, static_argnames=["mode"])
def _jit_surface_3d(rmnc, zmns, coeffs, xm, xn, xm_var, xn_var, theta, phi, mode):
    theta_grid, phi_grid = jnp.meshgrid(theta, phi, indexing="ij")
    angle_geom = (xm[:, None, None] * theta_grid[None, :, :]) - (xn[:, None, None] * phi_grid[None, :, :])
    r = jnp.sum(rmnc[:, None, None] * jnp.cos(angle_geom), axis=0)
    z_val = jnp.sum(zmns[:, None, None] * jnp.sin(angle_geom), axis=0)
    x_val = r * jnp.cos(phi_grid)
    y_val = r * jnp.sin(phi_grid)
    angle_var = (xm_var[:, None, None] * theta_grid[None, :, :]) - (xn_var[:, None, None] * phi_grid[None, :, :])
    trig = jnp.cos(angle_var) if mode == "cos" else jnp.sin(angle_var)
    val = jnp.sum(coeffs[:, None, None] * trig, axis=0)
    return x_val, y_val, z_val, val
