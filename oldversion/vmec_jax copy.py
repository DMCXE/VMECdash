import jax
import jax.numpy as jnp
import xarray as xr
import numpy as np
from functools import partial

# 启用 64 位精度 (对磁流体平衡至关重要)
jax.config.update("jax_enable_x64", True)

class VMECJaxProcessor:
    def __init__(self, nc_path):
        # 使用 xarray 惰性加载，不占用显存
        self.ds = xr.open_dataset(nc_path)
        
        # Dimensions
        self.ns = self.ds.sizes['radius']
        self.nfp = self.ds['nfp'].values.item()
        
        # Coordinates
        self.xm = self.ds['xm'].values
        self.xn = self.ds['xn'].values
        self.xm_nyq = self.ds['xm_nyq'].values
        self.xn_nyq = self.ds['xn_nyq'].values

        # Geometry Coefficients (Full Grid)
        self.rmnc = self.ds['rmnc'].values # (ns, nmn)
        self.zmns = self.ds['zmns'].values # (ns, nmn)
        
        # Helper to safely get variable
        def load_var(name):
            if name in self.ds:
                return self.ds[name].values
            return None

        # 3D Field Coefficients
        self.bmnc = load_var('bmnc')
        self.gmnc = load_var('gmnc')
        self.bsupumnc = load_var('bsupumnc')
        self.bsupvmnc = load_var('bsupvmnc')
        self.currumnc = load_var('currumnc')
        self.currvmnc = load_var('currvmnc')
        self.bsubumnc = load_var('bsubumnc')
        self.bsubvmnc = load_var('bsubvmnc')
        
        # 1D Profiles
        self.iotaf = load_var('iotaf')
        self.presf = load_var('presf')
        self.buco = load_var('buco')
        self.bvco = load_var('bvco')
        self.jcuru = load_var('jcuru')
        self.jcurv = load_var('jcurv')
        self.bdotb = load_var('bdotb')

    def close(self):
        self.ds.close()

    def get_scalars(self):
        """获取标量概览"""
        def _safe_get(key):
            if key not in self.ds: return 0.0
            val = self.ds[key].values
            return val.item() if val.ndim == 0 else val[-1]
            
        return {
            "beta_total": _safe_get('betatotal'),
            "volume": _safe_get('volume'),
            "aspect": _safe_get('aspect'),
            "b0": _safe_get('b0')
        }

    def get_1d_data(self, var_name):
        """获取 1D 剖面数据"""
        s = np.linspace(0, 1, self.ns)
        
        # Mapping of UI names to internal variables
        if var_name == 'iota':
            val = self.iotaf
        elif var_name == 'q':
            val = 1.0 / (self.iotaf + 1e-9)
        elif var_name == 'pressure':
            val = self.presf
        elif var_name == '<Buco>':
            val = self.buco
        elif var_name == '<Bvco>':
            val = self.bvco
        elif var_name == '<jcuru>':
            val = self.jcuru
        elif var_name == '<jcurv>':
            val = self.jcurv
        elif var_name == '<B.B>':
            val = self.bdotb
        else:
            return s, np.zeros_like(s)
            
        if val is None:
            return s, np.zeros_like(s)
            
        # Handle potential shape mismatches (some might be on half-grid)
        # For visualization, simple interpolation or truncation is usually enough
        if len(val) == self.ns:
            return s, val
        elif len(val) == self.ns - 1:
            # Append last point or interpolate? Let's just prepend 0 or duplicate
            return s[1:], val
        
        return s, val

    def get_flux_surface_data(self, s_idx, var_name, res_u=64, res_v=64):
        """计算特定磁面上的 2D 分布 (Theta vs Zeta)"""
        if s_idx < 0: s_idx += self.ns
        s_idx = np.clip(s_idx, 0, self.ns - 1)

        theta = jnp.linspace(0, 2 * jnp.pi, res_u)
        zeta = jnp.linspace(0, 2 * jnp.pi / self.nfp, res_v) # One field period
        
        # Select coefficients based on variable
        coeffs, mode_type, use_nyq = self._get_coeffs(var_name, s_idx)
        
        if coeffs is None:
            return None, None, None

        xm = self.xm_nyq if use_nyq else self.xm
        xn = self.xn_nyq if use_nyq else self.xn
        
        val = _jit_evaluate_2d(coeffs, xm, xn, theta, zeta, mode_type)
        
        return np.array(theta), np.array(zeta), np.array(val)

    def get_cross_section_data(self, phi, var_name, res_s=30, res_u=120):
        """计算特定环向角截面上的分布 (R vs Z)"""
        # Grid: s (radial) x u (poloidal)
        s_indices = np.linspace(0, self.ns - 1, res_s).astype(int)
        theta = jnp.linspace(0, 2 * jnp.pi, res_u)
        
        # 1. Calculate Geometry (R, Z) for the grid
        # We need R(s, u, phi) and Z(s, u, phi)
        # Batch compute for all s indices
        
        r_grid = []
        z_grid = []
        val_grid = []
        
        # Pre-compile geometry function for single phi
        # We can reuse _jit_cross_section but we need it for multiple surfaces
        
        for s_idx in s_indices:
            # Geometry
            r_c = self.rmnc[s_idx, :]
            z_c = self.zmns[s_idx, :]
            r, z = _jit_cross_section(r_c, z_c, self.xm, self.xn, theta, phi)
            r_grid.append(r)
            z_grid.append(z)
            
            # Value
            if var_name == 'geometry':
                val_grid.append(np.full_like(r, s_idx/(self.ns-1)))
            else:
                coeffs, mode_type, use_nyq = self._get_coeffs(var_name, s_idx)
                if coeffs is not None:
                    xm = self.xm_nyq if use_nyq else self.xm
                    xn = self.xn_nyq if use_nyq else self.xn
                    # Evaluate at specific phi (zeta)
                    # Reuse _jit_evaluate_line (theta varies, zeta fixed)
                    val = _jit_evaluate_line(coeffs, xm, xn, theta, phi, mode_type)
                    val_grid.append(val)
                else:
                    val_grid.append(np.zeros_like(r))

        return np.array(r_grid), np.array(z_grid), np.array(val_grid)

    def _get_coeffs(self, var_name, s_idx):
        """Helper to select coefficients"""
        # Returns: (coeffs, mode_type='cos'/'sin', use_nyq=True/False)
        if var_name == '|B|': return self.bmnc[s_idx], 'cos', True
        if var_name == 'sqrt(g)': return self.gmnc[s_idx], 'cos', True
        if var_name == 'B^u': return self.bsupumnc[s_idx], 'cos', True
        if var_name == 'B^v': return self.bsupvmnc[s_idx], 'cos', True
        if var_name == 'j^u': return self.currumnc[s_idx], 'cos', True
        if var_name == 'j^v': return self.currvmnc[s_idx], 'cos', True
        if var_name == 'B_u': return self.bsubumnc[s_idx], 'cos', True
        if var_name == 'B_v': return self.bsubvmnc[s_idx], 'cos', True
        return None, 'cos', False

    def compute_3d_surface(self, resolution=128):
        """计算 3D LCFS"""
        # 取最外层系数
        r_c = self.rmnc[-1, :]
        z_c = self.zmns[-1, :]
        b_c = self.bmnc[-1, :] if self.bmnc is not None else np.zeros_like(self.xm_nyq)
        
        theta = jnp.linspace(0, 2 * jnp.pi, resolution)
        phi = jnp.linspace(0, 2 * jnp.pi, resolution)
        
        # 调用 JIT 编译函数
        x, y, z, b = _jit_surface_3d(
            r_c, z_c, b_c, 
            self.xm, self.xn, self.xm_nyq, self.xn_nyq,
            theta, phi
        )
        
        return np.array(x), np.array(y), np.array(z), np.array(b)


# ---------------------------------------------------------
# JAX JIT Compiled Kernels (纯函数，极速执行)
# ---------------------------------------------------------

@partial(jax.jit, static_argnames=['xm', 'xn'])
def _jit_cross_section(rmnc, zmns, xm, xn, theta, phi):
    # Broadcasting: (N_coeffs, N_theta)
    angle = jnp.outer(xm, theta) - jnp.outer(xn, jnp.full_like(theta, phi))
    
    # Summation
    r = jnp.sum(rmnc[:, None] * jnp.cos(angle), axis=0)
    z = jnp.sum(zmns[:, None] * jnp.sin(angle), axis=0)
    return r, z

@partial(jax.jit, static_argnames=['xm', 'xn', 'mode'])
def _jit_evaluate_2d(coeffs, xm, xn, theta, zeta, mode):
    # theta: (Nu,), zeta: (Nv,)
    # angle: (N_coeffs, Nu, Nv)
    theta_grid, zeta_grid = jnp.meshgrid(theta, zeta, indexing='ij')
    
    angle = (xm[:, None, None] * theta_grid[None, :, :]) - \
            (xn[:, None, None] * zeta_grid[None, :, :])
            
    if mode == 'cos':
        val = jnp.sum(coeffs[:, None, None] * jnp.cos(angle), axis=0)
    else:
        val = jnp.sum(coeffs[:, None, None] * jnp.sin(angle), axis=0)
    return val

@partial(jax.jit, static_argnames=['xm', 'xn', 'mode'])
def _jit_evaluate_line(coeffs, xm, xn, theta, phi, mode):
    # theta: (Nu,), phi: scalar
    angle = jnp.outer(xm, theta) - jnp.outer(xn, jnp.full_like(theta, phi))
    
    if mode == 'cos':
        val = jnp.sum(coeffs[:, None] * jnp.cos(angle), axis=0)
    else:
        val = jnp.sum(coeffs[:, None] * jnp.sin(angle), axis=0)
    return val

@jax.jit
def _jit_surface_3d(rmnc, zmns, bmnc, xm, xn, xm_nyq, xn_nyq, theta, phi):
    # Meshgrid in JAX
    theta_grid, phi_grid = jnp.meshgrid(theta, phi, indexing='ij')
    
    # 几何重建
    # Angle shape: (N_coeffs, Res, Res)
    # 利用 broadcasting 避免显式 meshgrid 展开系数，节省显存
    angle_geom = (xm[:, None, None] * theta_grid[None, :, :]) - \
                 (xn[:, None, None] * phi_grid[None, :, :])
                 
    r = jnp.sum(rmnc[:, None, None] * jnp.cos(angle_geom), axis=0)
    z_val = jnp.sum(zmns[:, None, None] * jnp.sin(angle_geom), axis=0)
    
    x_val = r * jnp.cos(phi_grid)
    y_val = r * jnp.sin(phi_grid)
    
    # 磁场重建 (使用 Nyquist 频率)
    angle_b = (xm_nyq[:, None, None] * theta_grid[None, :, :]) - \
              (xn_nyq[:, None, None] * phi_grid[None, :, :])
    b_val = jnp.sum(bmnc[:, None, None] * jnp.cos(angle_b), axis=0)
    
    return x_val, y_val, z_val, b_val
