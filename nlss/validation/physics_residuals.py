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


def _finite_diff_gradient_3d(f: np.ndarray, dx: float, dy: float, dz: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes 3D spatial gradient (df/dx, df/dy, df/dz) using 2nd-order central differences
    in the interior and 1st-order one-sided differences at boundaries.
    Grid axes: f shape is (D, H, W) where D is z, H is y, W is x.
    """
    D, H, W = f.shape
    df_dx = np.zeros_like(f)
    df_dy = np.zeros_like(f)
    df_dz = np.zeros_like(f)

    # df/dx (axis 2 / W)
    df_dx[:, :, 1:-1] = (f[:, :, 2:] - f[:, :, :-2]) / (2.0 * dx)
    df_dx[:, :, 0] = (f[:, :, 1] - f[:, :, 0]) / dx
    df_dx[:, :, -1] = (f[:, :, -1] - f[:, :, -2]) / dx

    # df/dy (axis 1 / H)
    df_dy[:, 1:-1, :] = (f[:, 2:, :] - f[:, :-2, :]) / (2.0 * dy)
    df_dy[:, 0, :] = (f[:, 1, :] - f[:, 0, :]) / dy
    df_dy[:, -1, :] = (f[:, -1, :] - f[:, -2, :]) / dy

    # df/dz (axis 0 / D)
    df_dz[1:-1, :, :] = (f[2:, :, :] - f[:-2, :, :]) / (2.0 * dz)
    df_dz[0, :, :] = (f[1, :, :] - f[0, :, :]) / dz
    df_dz[-1, :, :] = (f[-1, :, :] - f[-2, :, :]) / dz

    return df_dx, df_dy, df_dz


def _finite_diff_laplacian_3d(f: np.ndarray, dx: float, dy: float, dz: float) -> np.ndarray:
    """Computes 3D scalar Laplacian laplacian(f) = d2f/dx2 + d2f/dy2 + d2f/dz2."""
    lap = np.zeros_like(f)

    # d2f/dx2
    d2x = np.zeros_like(f)
    d2x[:, :, 1:-1] = (f[:, :, 2:] - 2.0 * f[:, :, 1:-1] + f[:, :, :-2]) / (dx**2)
    d2x[:, :, 0] = d2x[:, :, 1]
    d2x[:, :, -1] = d2x[:, :, -2]

    # d2f/dy2
    d2y = np.zeros_like(f)
    d2y[:, 1:-1, :] = (f[:, 2:, :] - 2.0 * f[:, 1:-1, :] + f[:, :-2, :]) / (dy**2)
    d2y[:, 0, :] = d2y[:, 1, :]
    d2y[:, -1, :] = d2y[:, -2, :]

    # d2f/dz2
    d2z = np.zeros_like(f)
    d2z[1:-1, :, :] = (f[2:, :, :] - 2.0 * f[1:-1, :, :] + f[:-2, :, :]) / (dz**2)
    d2z[0, :, :] = d2z[1, :, :]
    d2z[-1, :, :] = d2z[-2, :, :]

    return d2x + d2y + d2z


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

    ux, uy, uz = u_np[0], u_np[1], u_np[2]
    dux_dx, _, _ = _finite_diff_gradient_3d(ux, dx, dy, dz)
    _, duy_dy, _ = _finite_diff_gradient_3d(uy, dx, dy, dz)
    _, _, duz_dz = _finite_diff_gradient_3d(uz, dx, dy, dz)

    div = dux_dx + duy_dy + duz_dz

    if fluid_mask is not None:
        if isinstance(fluid_mask, torch.Tensor):
            mask = fluid_mask.detach().cpu().numpy().astype(bool)
        else:
            mask = np.asarray(fluid_mask, dtype=bool)
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

    ux, uy, uz = u_np[0], u_np[1], u_np[2]

    dux_dx, dux_dy, dux_dz = _finite_diff_gradient_3d(ux, dx, dy, dz)
    duy_dx, duy_dy, duy_dz = _finite_diff_gradient_3d(uy, dx, dy, dz)
    duz_dx, duz_dy, duz_dz = _finite_diff_gradient_3d(uz, dx, dy, dz)

    wx = duz_dy - duy_dz
    wy = dux_dz - duz_dx
    wz = duy_dx - dux_dy

    return np.stack([wx, wy, wz], axis=0)


def compute_vorticity_metrics(
    pred_vel: Union[np.ndarray, torch.Tensor],
    true_vel: Union[np.ndarray, torch.Tensor],
    dx: float = 1.0,
    dy: float = 1.0,
    dz: float = 1.0,
    fluid_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> Dict[str, float]:
    """Computes error norms on the derived 3D vorticity field."""
    w_pred = compute_discrete_vorticity(pred_vel, dx, dy, dz)
    w_true = compute_discrete_vorticity(true_vel, dx, dy, dz)

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
) -> Dict[str, float]:
    """
    Computes steady Navier-Stokes momentum residual:
      R_NS = (u . grad)u + (1/rho)*grad(p) - nu*laplacian(u)
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

    ux, uy, uz = u_np[0], u_np[1], u_np[2]

    # Convective terms: (u . grad)u
    dux_dx, dux_dy, dux_dz = _finite_diff_gradient_3d(ux, dx, dy, dz)
    duy_dx, duy_dy, duy_dz = _finite_diff_gradient_3d(uy, dx, dy, dz)
    duz_dx, duz_dy, duz_dz = _finite_diff_gradient_3d(uz, dx, dy, dz)

    conv_x = ux * dux_dx + uy * dux_dy + uz * dux_dz
    conv_y = ux * duy_dx + uy * duy_dy + uz * duy_dz
    conv_z = ux * duz_dx + uy * duz_dy + uz * duz_dz

    # Pressure gradient: (1/rho)*grad(p)
    dp_dx, dp_dy, dp_dz = _finite_diff_gradient_3d(p_np, dx, dy, dz)
    grad_p_x = dp_dx / rho
    grad_p_y = dp_dy / rho
    grad_p_z = dp_dz / rho

    # Viscous diffusion: nu * laplacian(u)
    visc_x = nu * _finite_diff_laplacian_3d(ux, dx, dy, dz)
    visc_y = nu * _finite_diff_laplacian_3d(uy, dx, dy, dz)
    visc_z = nu * _finite_diff_laplacian_3d(uz, dx, dy, dz)

    # Residuals
    rx = conv_x + grad_p_x - visc_x
    ry = conv_y + grad_p_y - visc_y
    rz = conv_z + grad_p_z - visc_z
    r_total = np.stack([rx, ry, rz], axis=0)

    if fluid_mask is not None:
        if isinstance(fluid_mask, torch.Tensor):
            mask = fluid_mask.detach().cpu().numpy().astype(bool)
        else:
            mask = np.asarray(fluid_mask, dtype=bool)
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
