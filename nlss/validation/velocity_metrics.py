"""
BioFlo-CFD: Rigorous Velocity Validation Metrics Suite
Defines standardized scientific error metrics beyond global R²:
  - Relative L2, L1, Linf norms
  - RMSE and MAE
  - Velocity magnitude error
  - Component-wise errors (ux, uy, uz)
  - 3D Vector angular deviation (cosine alignment and degrees)
  - Coefficient of determination (R²)
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch


def to_numpy(arr: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Converts torch.Tensor or numpy array to float64 numpy array."""
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy().astype(np.float64)
    return np.asarray(arr, dtype=np.float64)


def relative_l2_error(pred: Union[np.ndarray, torch.Tensor], true: Union[np.ndarray, torch.Tensor], eps: float = 1e-12) -> float:
    """Computes E_L2 = ||u_pred - u_true||_2 / (||u_true||_2 + eps)."""
    p = to_numpy(pred)
    t = to_numpy(true)
    diff_norm = np.linalg.norm(p - t)
    base_norm = np.linalg.norm(t)
    return float(diff_norm / (base_norm + eps))


def relative_l1_error(pred: Union[np.ndarray, torch.Tensor], true: Union[np.ndarray, torch.Tensor], eps: float = 1e-12) -> float:
    """Computes E_L1 = ||u_pred - u_true||_1 / (||u_true||_1 + eps)."""
    p = to_numpy(pred)
    t = to_numpy(true)
    diff_l1 = np.sum(np.abs(p - t))
    base_l1 = np.sum(np.abs(t))
    return float(diff_l1 / (base_l1 + eps))


def relative_linf_error(pred: Union[np.ndarray, torch.Tensor], true: Union[np.ndarray, torch.Tensor], eps: float = 1e-12) -> float:
    """Computes E_inf = ||u_pred - u_true||_inf / (||u_true||_inf + eps)."""
    p = to_numpy(pred)
    t = to_numpy(true)
    diff_max = np.max(np.abs(p - t))
    base_max = np.max(np.abs(t))
    return float(diff_max / (base_max + eps))


def rmse_mae_metrics(pred: Union[np.ndarray, torch.Tensor], true: Union[np.ndarray, torch.Tensor]) -> Dict[str, float]:
    """Computes Root Mean Square Error (RMSE) and Mean Absolute Error (MAE)."""
    p = to_numpy(pred)
    t = to_numpy(true)
    diff = p - t
    rmse = float(np.sqrt(np.mean(diff**2)))
    mae = float(np.mean(np.abs(diff)))
    return {"rmse": rmse, "mae": mae}


def r2_score_field(pred: Union[np.ndarray, torch.Tensor], true: Union[np.ndarray, torch.Tensor], eps: float = 1e-12) -> float:
    """Computes Coefficient of Determination R² across all scalar components."""
    p = to_numpy(pred).ravel()
    t = to_numpy(true).ravel()
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - np.mean(t)) ** 2)
    if ss_tot < eps:
        return 1.0 if ss_res < eps else 0.0
    return float(1.0 - (ss_res / (ss_tot + eps)))


def component_wise_errors(
    pred_vel: Union[np.ndarray, torch.Tensor],
    true_vel: Union[np.ndarray, torch.Tensor],
    channel_dim: int = 1,
    eps: float = 1e-12,
) -> Dict[str, Dict[str, float]]:
    """
    Computes decomposed errors for ux, uy, uz and velocity magnitude ||u||.
    Expects arrays with 3 velocity components along channel_dim.
    """
    p = to_numpy(pred_vel)
    t = to_numpy(true_vel)

    # Bring channel_dim to front: (3, ...)
    p = np.moveaxis(p, channel_dim, 0)
    t = np.moveaxis(t, channel_dim, 0)

    if p.shape[0] < 3 or t.shape[0] < 3:
        raise ValueError(f"Expected at least 3 channels for 3D velocity, got shape {p.shape}")
    if p.ndim == 2 and p.shape[0] > 8:
        raise ValueError(
            f"channel_dim={channel_dim} has size {p.shape[0]} on a 2D array; "
            "reshape flattened arrays to (B, channels, N_active) before calling vector_angular_metrics."
        )

    if p.shape[0] < 3 or t.shape[0] < 3:
        raise ValueError(f"Expected at least 3 channels for 3D velocity, got shape {p.shape}")
    if p.ndim == 2 and p.shape[0] > 8:
        raise ValueError(
            f"channel_dim={channel_dim} has size {p.shape[0]} on a 2D array; "
            "reshape flattened arrays to (B, channels, N_active) before calling component_wise_errors."
        )

    ux_p, uy_p, uz_p = p[0], p[1], p[2]
    ux_t, uy_t, uz_t = t[0], t[1], t[2]

    mag_p = np.sqrt(ux_p**2 + uy_p**2 + uz_p**2)
    mag_t = np.sqrt(ux_t**2 + uy_t**2 + uz_t**2)

    components = {
        "ux": (ux_p, ux_t),
        "uy": (uy_p, uy_t),
        "uz": (uz_p, uz_t),
        "magnitude": (mag_p, mag_t),
    }

    results = {}
    for name, (c_p, c_t) in components.items():
        results[name] = {
            "rel_l2": relative_l2_error(c_p, c_t, eps=eps),
            "rel_l1": relative_l1_error(c_p, c_t, eps=eps),
            "rel_linf": relative_linf_error(c_p, c_t, eps=eps),
            "rmse": float(np.sqrt(np.mean((c_p - c_t)**2))),
            "mae": float(np.mean(np.abs(c_p - c_t))),
            "r2": r2_score_field(c_p, c_t, eps=eps),
        }
    return results


def vector_angular_metrics(
    pred_vel: Union[np.ndarray, torch.Tensor],
    true_vel: Union[np.ndarray, torch.Tensor],
    channel_dim: int = 1,
    threshold: float = 1e-4,
) -> Dict[str, float]:
    """
    Computes 3D vector alignment cosine and angular deviation in degrees.
    Masks out stagnant fluid (||u|| < threshold) to prevent division by zero.
    """
    p = to_numpy(pred_vel)
    t = to_numpy(true_vel)

    p = np.moveaxis(p, channel_dim, 0)
    t = np.moveaxis(t, channel_dim, 0)

    if p.shape[0] < 3 or t.shape[0] < 3:
        raise ValueError(f"Expected at least 3 channels for 3D velocity, got shape {p.shape}")
    if p.ndim == 2 and p.shape[0] > 8:
        raise ValueError(
            f"channel_dim={channel_dim} has size {p.shape[0]} on a 2D array; "
            "reshape flattened arrays to (B, channels, N_active) before calling vector_angular_metrics."
        )

    mag_p = np.sqrt(p[0]**2 + p[1]**2 + p[2]**2)
    mag_t = np.sqrt(t[0]**2 + t[1]**2 + t[2]**2)

    active = (mag_p > threshold) & (mag_t > threshold)
    if not np.any(active):
        return {"mean_cos_theta": 1.0, "mean_angle_deg": 0.0, "max_angle_deg": 0.0}

    dot = p[0][active] * t[0][active] + p[1][active] * t[1][active] + p[2][active] * t[2][active]
    denom = mag_p[active] * mag_t[active]
    cos_theta = np.clip(dot / np.maximum(denom, 1e-12), -1.0, 1.0)
    angles_deg = np.degrees(np.arccos(cos_theta))

    return {
        "mean_cos_theta": float(np.mean(cos_theta)),
        "mean_angle_deg": float(np.mean(angles_deg)),
        "max_angle_deg": float(np.max(angles_deg)),
        "p95_angle_deg": float(np.percentile(angles_deg, 95)),
    }


def compute_comprehensive_velocity_metrics(
    pred_vel: Union[np.ndarray, torch.Tensor],
    true_vel: Union[np.ndarray, torch.Tensor],
    channel_dim: int = 1,
) -> Dict[str, Union[float, Dict[str, float]]]:
    """Computes a complete, structured validation report for velocity fields."""
    p = to_numpy(pred_vel)
    t = to_numpy(true_vel)

    # Global norms across all entries
    rel_l2 = relative_l2_error(p, t)
    rel_l1 = relative_l1_error(p, t)
    rel_linf = relative_linf_error(p, t)
    r2 = r2_score_field(p, t)
    rmse_mae = rmse_mae_metrics(p, t)
    comp_metrics = component_wise_errors(p, t, channel_dim=channel_dim)
    angle_metrics = vector_angular_metrics(p, t, channel_dim=channel_dim)

    return {
        "global_rel_l2": rel_l2,
        "global_rel_l1": rel_l1,
        "global_rel_linf": rel_linf,
        "global_r2": r2,
        "rmse": rmse_mae["rmse"],
        "mae": rmse_mae["mae"],
        "components": comp_metrics,
        "angular_alignment": angle_metrics,
    }
