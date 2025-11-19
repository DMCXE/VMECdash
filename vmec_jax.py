import jax
import jax.numpy as jnp
import xarray as xr
# import numpy as jnp
from functools import partial
from scipy.interpolate import griddata
from matplotlib.path import Path

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

    def get_cross_section_grid(self, phi, var_name, res_grid=200):
        """
        获取插值并掩膜后的规则网格数据，用于直接绘图。
        返回: r_lin, z_lin, val_grid (masked)
        """
        # 1. 获取原始散点数据 (用于插值)
        # 使用较高的径向分辨率以保证插值质量
        # 确保输入是 JAX 数组以避免隐式转换
        r_raw, z_raw, val_raw = self.get_cross_section_data(phi, var_name, res_s=64, res_u=128)
        
        # 2. 获取 LCFS 边界 (用于掩膜)
        # 直接计算边界 (s_idx = ns - 1)
        theta = jnp.linspace(0, 2 * jnp.pi, 256)
        r_c = self.rmnc[-1, :]
        z_c = self.zmns[-1, :]
        r_lcfs, z_lcfs = _jit_cross_section(r_c, z_c, self.xm, self.xn, theta, phi)
        
        # 3. 创建规则网格
        # griddata 需要 numpy 数组
        points = jnp.column_stack((jnp.array(r_raw).flatten(), jnp.array(z_raw).flatten()))
        values = jnp.array(val_raw).flatten()
        
        r_min, r_max = jnp.array(r_lcfs).min(), jnp.array(r_lcfs).max()
        z_min, z_max = jnp.array(z_lcfs).min(), jnp.array(z_lcfs).max()
        
        # 稍微扩大一点范围以免边界被切
        pad_r = (r_max - r_min) * 0.05
        pad_z = (z_max - z_min) * 0.05
        
        r_lin = jnp.linspace(r_min - pad_r, r_max + pad_r, res_grid)
        z_lin = jnp.linspace(z_min - pad_z, z_max + pad_z, res_grid)
        r_mesh, z_mesh = jnp.meshgrid(r_lin, z_lin)
        
        # 4. 插值
        val_grid = griddata(points, values, (r_mesh, z_mesh), method='linear')
        
        # 5. 掩膜 (Masking)
        # 创建 LCFS 多边形路径
        lcfs_poly = jnp.column_stack((r_lcfs, z_lcfs))
        path = Path(lcfs_poly)
        
        # 检查网格点是否在多边形内
        # flatten mesh points
        mesh_points = jnp.column_stack((r_mesh.flatten(), z_mesh.flatten()))
        mask = path.contains_points(mesh_points)
        mask = mask.reshape(r_mesh.shape)
        
        # 应用掩膜
        val_grid[~mask] = jnp.nan
        
        return r_lin, z_lin, val_grid

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
        s = jnp.linspace(0, 1, self.ns)
        
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
            return s, jnp.zeros_like(s)
            
        if val is None:
            return s, jnp.zeros_like(s)
            
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
        s_idx = jnp.clip(s_idx, 0, self.ns - 1)

        theta = jnp.linspace(0, 2 * jnp.pi, res_u)
        zeta = jnp.linspace(0, 2 * jnp.pi / self.nfp, res_v) # One field period
        
        # Select coefficients based on variable
        coeffs, mode_type, use_nyq = self._get_coeffs(var_name, s_idx)
        
        if coeffs is None:
            return None, None, None

        xm = self.xm_nyq if use_nyq else self.xm
        xn = self.xn_nyq if use_nyq else self.xn
        
        val = _jit_evaluate_2d(coeffs, xm, xn, theta, zeta, mode_type)
        
        return jnp.array(theta), jnp.array(zeta), jnp.array(val)

    def get_cross_section_data(self, phi, var_name, res_s=30, res_u=120):
        """计算特定环向角截面上的分布 (R vs Z)"""
        # Grid: s (radial) x u (poloidal)
        s_indices = jnp.linspace(0, self.ns - 1, res_s).astype(int)
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
                val_grid.append(jnp.full_like(r, s_idx/(self.ns-1)))
            else:
                # FIX: Handle axis singularity (s=0)
                # Use s=1 coefficients for value and average them to get a unique axis value
                is_axis = (s_idx == 0)
                calc_s_idx = 1 if (is_axis and self.ns > 1) else s_idx
                
                coeffs, mode_type, use_nyq = self._get_coeffs(var_name, calc_s_idx)
                
                if coeffs is not None:
                    xm = self.xm_nyq if use_nyq else self.xm
                    xn = self.xn_nyq if use_nyq else self.xn
                    # Evaluate at specific phi (zeta)
                    val = _jit_evaluate_line(coeffs, xm, xn, theta, phi, mode_type)
                    
                    if is_axis:
                        # Axis must be single-valued
                        val = jnp.full_like(val, jnp.mean(val))
                        
                    val_grid.append(val)
                else:
                    val_grid.append(jnp.zeros_like(r))

        return jnp.array(r_grid), jnp.array(z_grid), jnp.array(val_grid)

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

    def compute_3d_surface(self, s_idx=-1, var_name='|B|', resolution=128):
        """计算 3D 磁面几何及物理量分布"""
        # 处理索引
        if s_idx < 0: s_idx += self.ns
        s_idx = jnp.clip(s_idx, 0, self.ns - 1)

        # 获取几何系数
        r_c = self.rmnc[s_idx, :]
        z_c = self.zmns[s_idx, :]
        
        # 获取物理量系数
        if var_name == 'geometry':
            # 如果是纯几何，颜色可以设为常数或半径
            coeffs = None
            mode_type = 'cos'
            use_nyq = False
        else:
            coeffs, mode_type, use_nyq = self._get_coeffs(var_name, s_idx)
        
        theta = jnp.linspace(0, 2 * jnp.pi, resolution)
        phi = jnp.linspace(0, 2 * jnp.pi, resolution) # Full torus for 3D view usually
        
        # 准备物理量计算所需的频率数组
        if coeffs is not None:
            xm_var = self.xm_nyq if use_nyq else self.xm
            xn_var = self.xn_nyq if use_nyq else self.xn
        else:
            # Dummy
            xm_var = self.xm
            xn_var = self.xn
            coeffs = jnp.zeros_like(self.xm) # Zero value

        # 调用 JIT 编译函数
        x, y, z, val = _jit_surface_3d(
            r_c, z_c, coeffs, 
            self.xm, self.xn, xm_var, xn_var,
            theta, phi, mode_type
        )
        
        return jnp.array(x), jnp.array(y), jnp.array(z), jnp.array(val)


# ---------------------------------------------------------
# JAX JIT Compiled Kernels (纯函数，极速执行)
# ---------------------------------------------------------

@jax.jit
def _jit_cross_section(rmnc, zmns, xm, xn, theta, phi):
    # Broadcasting: (N_coeffs, N_theta)
    angle = jnp.outer(xm, theta) - jnp.outer(xn, jnp.full_like(theta, phi))
    
    # Summation
    r = jnp.sum(rmnc[:, None] * jnp.cos(angle), axis=0)
    z = jnp.sum(zmns[:, None] * jnp.sin(angle), axis=0)
    return r, z

@partial(jax.jit, static_argnames=['mode'])
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

@partial(jax.jit, static_argnames=['mode'])
def _jit_evaluate_line(coeffs, xm, xn, theta, phi, mode):
    # theta: (Nu,), phi: scalar
    angle = jnp.outer(xm, theta) - jnp.outer(xn, jnp.full_like(theta, phi))
    
    if mode == 'cos':
        val = jnp.sum(coeffs[:, None] * jnp.cos(angle), axis=0)
    else:
        val = jnp.sum(coeffs[:, None] * jnp.sin(angle), axis=0)
    return val

@partial(jax.jit, static_argnames=['mode'])
def _jit_surface_3d(rmnc, zmns, coeffs, xm, xn, xm_var, xn_var, theta, phi, mode):
    # Meshgrid in JAX
    theta_grid, phi_grid = jnp.meshgrid(theta, phi, indexing='ij')
    
    # 几何重建
    # Angle shape: (N_coeffs, Res, Res)
    angle_geom = (xm[:, None, None] * theta_grid[None, :, :]) - \
                 (xn[:, None, None] * phi_grid[None, :, :])
                 
    r = jnp.sum(rmnc[:, None, None] * jnp.cos(angle_geom), axis=0)
    z_val = jnp.sum(zmns[:, None, None] * jnp.sin(angle_geom), axis=0)
    
    x_val = r * jnp.cos(phi_grid)
    y_val = r * jnp.sin(phi_grid)
    
    # 物理量重建
    angle_var = (xm_var[:, None, None] * theta_grid[None, :, :]) - \
                (xn_var[:, None, None] * phi_grid[None, :, :])
    
    if mode == 'cos':
        val = jnp.sum(coeffs[:, None, None] * jnp.cos(angle_var), axis=0)
    else:
        val = jnp.sum(coeffs[:, None, None] * jnp.sin(angle_var), axis=0)
    
    return x_val, y_val, z_val, val
