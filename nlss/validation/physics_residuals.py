"""
BioFlo-CFD: Rigorous Physics Residuals & Conservation Suite
Evaluates physical consistency of neural predictions against Navier-Stokes constraints:
  1. Discrete Mass Conservation (Divergence: div(u) = 0)
  2. Small-Scale Rotational Coherence (Vorticity: omega = curl(u))
  3. Steady Navier-Stokes Momentum Residual: R_NS = (u . grad)u + 1/rho * grad(p) - nu * laplacian(u)
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch


def _finite_diff_gradient_3d(
    f: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes 3D spatial gradient (df/dx, df/dy, df/dz) using 2nd-order central differences
    in the interior and 1st-order one-sided differences at domain and fluid-solid boundaries.
    Grid axes: f shape is (D, H, W) where D is z, H is y, W is x.
    """
    df_dx = np.zeros_like(f)
    df_dy = np.zeros_like(f)
    df_dz = np.zeros_like(f)

    def _diff_axis(arr: np.ndarray, h: float, axis: int) -> np.ndarray:
        n = arr.shape[axis]
        out = np.zeros_like(arr)
        if n < 2:
            return out

        # Move target axis to 0 for uniform indexing
        a = np.moveaxis(arr, axis, 0)
        o = np.zeros_like(a)

        if mask is None:
            if n >= 3:
                o[1:-1] = (a[2:] - a[:-2]) / (2.0 * h)
            o[0] = (a[1] - a[0]) / h
            o[-1] = (a[-1] - a[-2]) / h
        else:
            m = np.moveaxis(mask.astype(bool), axis, 0)
            # Only apply one-sided fluid-wall stencils when masked region is actually zeroed out;
            # if fluid_mask is merely an interior evaluation ROI over a smooth field, keep 2nd-order central diffs.
            has_zeroed_solids = bool(np.any(~m) and np.all(a[~m] == 0.0))
            if has_zeroed_solids:
                if n >= 3:
                    both = m[:-2] & m[1:-1] & m[2:]
                    fwd = (~m[:-2]) & m[1:-1] & m[2:]
                    bwd = m[:-2] & m[1:-1] & (~m[2:])
                    o_mid = o[1:-1]
                    o_mid[both] = (a[2:][both] - a[:-2][both]) / (2.0 * h)
                    o_mid[fwd] = (a[2:][fwd] - a[1:-1][fwd]) / h
                    o_mid[bwd] = (a[1:-1][bwd] - a[:-2][bwd]) / h
                b0 = m[0] & m[1]
                o[0][b0] = (a[1][b0] - a[0][b0]) / h
                b_end = m[-1] & m[-2]
                o[-1][b_end] = (a[-1][b_end] - a[-2][b_end]) / h
            else:
                if n >= 3:
                    o[1:-1] = (a[2:] - a[:-2]) / (2.0 * h)
                o[0] = (a[1] - a[0]) / h
                o[-1] = (a[-1] - a[-2]) / h

        return np.moveaxis(o, 0, axis)

    df_dx = _diff_axis(f, dx, axis=2)
    df_dy = _diff_axis(f, dy, axis=1)
    df_dz = _diff_axis(f, dz, axis=0)

    return df_dx, df_dy, df_dz


def _finite_diff_laplacian_3d(
    f: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Computes 3D scalar Laplacian laplacian(f) = d2f/dx2 + d2f/dy2 + d2f/dz2 (mask-aware)."""
    def _d2_axis(arr: np.ndarray, h: float, axis: int) -> np.ndarray:
        n = arr.shape[axis]
        if n < 3:
            return np.zeros_like(arr)
        a = np.moveaxis(arr, axis, 0)
        o = np.zeros_like(a)
        if mask is None:
            o[1:-1] = (a[2:] - 2.0 * a[1:-1] + a[:-2]) / (h**2)
            o[0] = o[1]
            o[-1] = o[-2]
        else:
            m = np.moveaxis(mask.astype(bool), axis, 0)
            valid = m[:-2] & m[1:-1] & m[2:]
            o_mid = o[1:-1]
            o_mid[valid] = (a[2:][valid] - 2.0 * a[1:-1][valid] + a[:-2][valid]) / (h**2)
            # Zero out masked solid cells
            o[~m] = 0.0
        return np.moveaxis(o, 0, axis)

    return _d2_axis(f, dx, axis=2) + _d2_axis(f, dy, axis=1) + _d2_axis(f, dz, axis=0)


def compute_discrete_divergence(
    u_grid: Union[np.ndarray, torch.Tensor],
    dx: float = 1.0,
    dy: float = 1.0,
    dz: float = 1.0,
    fluid_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> Dict[str, float]:
    """
    Computes discrete divergence div(u) = dux/dx + duy/dy + duz/dz.
    u_grid: array of shape (3, D, H, W)
    """
    if isinstance(u_grid, torch.Tensor):
        u_np = u_grid.detach().cpu().numpy().astype(np.float64)
    else:
        u_np = np.asarray(u_grid, dtype=np.float64)

    mask: Optional[np.ndarray] = None
    if fluid_mask is not None:
        if isinstance(fluid_mask, torch.Tensor):
            mask = fluid_mask.detach().cpu().numpy().astype(bool)
        else:
            mask = np.asarray(fluid_mask, dtype=bool)

    ux, uy, uz = u_np[0], u_np[1], u_np[2]
    dux_dx, _, _ = _finite_diff_gradient_3d(ux, dx, dy, dz, mask=mask)
    _, duy_dy, _ = _finite_diff_gradient_3d(uy, dx, dy, dz, mask=mask)
    _, _, duz_dz = _finite_diff_gradient_3d(uz, dx, dy, dz, mask=mask)

    div = dux_dx + duy_dy + duz_dz

    if mask is not None:
        div = div[mask]

    abs_div = np.abs(div)
    return {
        "mean_abs_divergence": float(np.mean(abs_div)),
        "max_abs_divergence": float(np.max(abs_div)),
        "rmse_divergence": float(np.sqrt(np.mean(div**2))),
        "p95_abs_divergence": float(np.percentile(abs_div, 95)),
    }


def compute_discrete_vorticity(
    u_grid: Union[np.ndarray, torch.Tensor],
    dx: float = 1.0,
    dy: float = 1.0,
    dz: float = 1.0,
    fluid_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> np.ndarray:
    """
    Computes 3D vorticity vector field omega = curl(u) = [dw/dy - dv/dz, du/dz - dw/dx, dv/dx - du/dy].
    u_grid: array of shape (3, D, H, W)
    Returns: array of shape (3, D, H, W)
    """
    if isinstance(u_grid, torch.Tensor):
        u_np = u_grid.detach().cpu().numpy().astype(np.float64)
    else:
        u_np = np.asarray(u_grid, dtype=np.float64)

    mask: Optional[np.ndarray] = None
    if fluid_mask is not None:
        if isinstance(fluid_mask, torch.Tensor):
            mask = fluid_mask.detach().cpu().numpy().astype(bool)
        else:
            mask = np.asarray(fluid_mask, dtype=bool)

    ux, uy, uz = u_np[0], u_np[1], u_np[2]

    dux_dx, dux_dy, dux_dz = _finite_diff_gradient_3d(ux, dx, dy, dz, mask=mask)
    duy_dx, duy_dy, duy_dz = _finite_diff_gradient_3d(uy, dx, dy, dz, mask=mask)
    duz_dx, duz_dy, duz_dz = _finite_diff_gradient_3d(uz, dx, dy, dz, mask=mask)

    wx = duz_dy - duy_dz
    wy = dux_dz - duz_dx
    wz = duy_dx - dux_dy

    vort = np.stack([wx, wy, wz], axis=0)
    if mask is not None:
        vort[:, ~mask] = 0.0
    return vort


def compute_vorticity_metrics(
    pred_vel: Union[np.ndarray, torch.Tensor],
    true_vel: Union[np.ndarray, torch.Tensor],
    dx: float = 1.0,
    dy: float = 1.0,
    dz: float = 1.0,
    fluid_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> Dict[str, float]:
    """Computes error norms on the derived 3D vorticity field."""
    w_pred = compute_discrete_vorticity(pred_vel, dx, dy, dz, fluid_mask=fluid_mask)
    w_true = compute_discrete_vorticity(true_vel, dx, dy, dz, fluid_mask=fluid_mask)

    if fluid_mask is not None:
        if isinstance(fluid_mask, torch.Tensor):
            mask = fluid_mask.detach().cpu().numpy().astype(bool)
        else:
            mask = np.asarray(fluid_mask, dtype=bool)
        w_p_eval = w_pred[:, mask]
        w_t_eval = w_true[:, mask]
    else:
        w_p_eval = w_pred
        w_t_eval = w_true

    diff_norm = np.linalg.norm(w_p_eval - w_t_eval)
    base_norm = np.linalg.norm(w_t_eval)
    rel_l2 = float(diff_norm / (base_norm + 1e-12))
    rmse = float(np.sqrt(np.mean((w_p_eval - w_t_eval)**2)))
    mae = float(np.mean(np.abs(w_p_eval - w_t_eval)))

    return {
        "vorticity_rel_l2": rel_l2,
        "vorticity_rmse": rmse,
        "vorticity_mae": mae,
    }


def compute_momentum_residual_steady(
    vel_grid: Union[np.ndarray, torch.Tensor],
    p_grid: Union[np.ndarray, torch.Tensor],
    rho: float = 1000.0,
    nu: float = 1e-6,
    dx: float = 1.0,
    dy: float = 1.0,
    dz: float = 1.0,
    fluid_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
    kinematic_pressure: bool = False,
    du_dt: Optional[Union[np.ndarray, torch.Tensor]] = None,
    nu_t: Optional[Union[float, np.ndarray, torch.Tensor]] = None,
) -> Dict[str, float]:
    """
    Computes Navier-Stokes momentum residual:
      R_NS = du/dt + (u . grad)u + grad(p_kinematic) - nu_eff * laplacian(u)
    where p_kinematic = p if kinematic_pressure=True, else p / rho,
    and nu_eff = nu + nu_t (when turbulent eddy viscosity is provided).
    vel_grid: (3, D, H, W)
    p_grid: (1, D, H, W) or (D, H, W)
    """
    if isinstance(vel_grid, torch.Tensor):
        u_np = vel_grid.detach().cpu().numpy().astype(np.float64)
    else:
        u_np = np.asarray(vel_grid, dtype=np.float64)

    if isinstance(p_grid, torch.Tensor):
        p_np = p_grid.detach().cpu().numpy().astype(np.float64)
    else:
        p_np = np.asarray(p_grid, dtype=np.float64)

    if p_np.ndim == 4:
        p_np = p_np[0]

    mask: Optional[np.ndarray] = None
    if fluid_mask is not None:
        if isinstance(fluid_mask, torch.Tensor):
            mask = fluid_mask.detach().cpu().numpy().astype(bool)
        else:
            mask = np.asarray(fluid_mask, dtype=bool)

    ux, uy, uz = u_np[0], u_np[1], u_np[2]

    # Convective terms: (u . grad)u
    dux_dx, dux_dy, dux_dz = _finite_diff_gradient_3d(ux, dx, dy, dz, mask=mask)
    duy_dx, duy_dy, duy_dz = _finite_diff_gradient_3d(uy, dx, dy, dz, mask=mask)
    duz_dx, duz_dy, duz_dz = _finite_diff_gradient_3d(uz, dx, dy, dz, mask=mask)

    conv_x = ux * dux_dx + uy * dux_dy + uz * dux_dz
    conv_y = ux * duy_dx + uy * duy_dy + uz * duy_dz
    conv_z = ux * duz_dx + uy * duz_dy + uz * duz_dz

    # Pressure gradient: grad(p) for kinematic pressure, else (1/rho)*grad(p)
    dp_dx, dp_dy, dp_dz = _finite_diff_gradient_3d(p_np, dx, dy, dz, mask=mask)
    p_scale = 1.0 if kinematic_pressure else (1.0 / rho)
    grad_p_x = dp_dx * p_scale
    grad_p_y = dp_dy * p_scale
    grad_p_z = dp_dz * p_scale

    # Viscous diffusion: (nu + nu_t) * laplacian(u)
    nu_eff: Union[float, np.ndarray] = nu
    if nu_t is not None:
        if isinstance(nu_t, torch.Tensor):
            nu_eff = nu + nu_t.detach().cpu().numpy().astype(np.float64)
        else:
            nu_eff = nu + np.asarray(nu_t, dtype=np.float64)

    visc_x = nu_eff * _finite_diff_laplacian_3d(ux, dx, dy, dz, mask=mask)
    visc_y = nu_eff * _finite_diff_laplacian_3d(uy, dx, dy, dz, mask=mask)
    visc_z = nu_eff * _finite_diff_laplacian_3d(uz, dx, dy, dz, mask=mask)

    # Residuals
    rx = conv_x + grad_p_x - visc_x
    ry = conv_y + grad_p_y - visc_y
    rz = conv_z + grad_p_z - visc_z
    if du_dt is not None:
        du_dt_np = du_dt.detach().cpu().numpy().astype(np.float64) if isinstance(du_dt, torch.Tensor) else np.asarray(du_dt, dtype=np.float64)
        rx = rx + du_dt_np[0]
        ry = ry + du_dt_np[1]
        rz = rz + du_dt_np[2]

    r_total = np.stack([rx, ry, rz], axis=0)

    if mask is not None:
        r_eval = r_total[:, mask]
        conv_eval = np.stack([conv_x, conv_y, conv_z], axis=0)[:, mask]
        grad_p_eval = np.stack([grad_p_x, grad_p_y, grad_p_z], axis=0)[:, mask]
    else:
        r_eval = r_total
        conv_eval = np.stack([conv_x, conv_y, conv_z], axis=0)
        grad_p_eval = np.stack([grad_p_x, grad_p_y, grad_p_z], axis=0)

    r_norm = np.linalg.norm(r_eval)
    scale_norm = np.linalg.norm(conv_eval) + np.linalg.norm(grad_p_eval) + 1e-12

    return {
        "momentum_residual_norm": float(r_norm),
        "relative_momentum_residual": float(r_norm / scale_norm),
        "mean_momentum_residual_magnitude": float(np.mean(np.sqrt(np.sum(r_eval**2, axis=0)))),
        "max_momentum_residual_magnitude": float(np.max(np.sqrt(np.sum(r_eval**2, axis=0)))),
    }
