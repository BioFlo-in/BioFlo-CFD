"""
BioFlo-CFD: Reactor Region-of-Interest (ROI) Validation Suite
Enforces localized scientific evaluation across critical hydrodynamic zones:
  1. Impeller Discharge ROI (High-gradient radial jet, shear stress, power dissipation)
  2. Wall Baffle Boundary Layer ROI (Jet deflection, boundary separation, stagnation)
  3. Dished Bottom Sweeping ROI (Solids suspension, dead-zone scouring)
  4. Free Surface Boundary ROI (Liquid-gas interface shear, vortex draw-down)
  5. Bulk Recirculation ROI (Macroscopic toroidal circulation loops)
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from .velocity_metrics import relative_l2_error, relative_l1_error, relative_linf_error, vector_angular_metrics, to_numpy


def generate_bioreactor_rois(
    grid_coords: Tuple[np.ndarray, np.ndarray, np.ndarray],
    tank_radius: float = 0.25,
    impeller_radius: float = 0.083,
    impeller_elevation: float = 0.35,
    impeller_height: float = 0.035,
    baffle_width: float = 0.025,
    liquid_height: float = 1.0,
    bottom_dish_height: float = 0.10,
    fluid_mask: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Generates standard boolean 3D masks for critical bioreactor hydrodynamic zones.
    grid_coords: (GX, GY, GZ) each of shape (D, H, W)
    """
    GX, GY, GZ = grid_coords
    R = np.sqrt(GX**2 + GY**2)

    if fluid_mask is None:
        base_fluid = (R <= tank_radius) & (GZ >= -bottom_dish_height) & (GZ <= liquid_height)
    else:
        base_fluid = to_numpy(fluid_mask).astype(bool)

    # 1. Impeller Discharge Zone: High shear core
    impeller_roi = (
        base_fluid
        & (R <= impeller_radius * 1.35)
        & (GZ >= impeller_elevation - impeller_height * 2.0)
        & (GZ <= impeller_elevation + impeller_height * 2.0)
    )

    # 2. Wall Baffle Zone: Outer annulus where radial jet hits baffles
    baffle_roi = (
        base_fluid
        & (R >= tank_radius - baffle_width * 1.5)
        & (R <= tank_radius * 1.01)
    )

    # 3. Dished Bottom Sweeping Zone: Bottom suspension region
    bottom_roi = (
        base_fluid
        & (GZ <= bottom_dish_height * 1.5)
    )

    # 4. Free Surface Boundary Zone: Upper liquid-gas interface
    surface_roi = (
        base_fluid
        & (GZ >= liquid_height - 0.15 * liquid_height)
    )

    # 5. Bulk Recirculation: Everything else in the active fluid
    bulk_roi = base_fluid & (~impeller_roi) & (~baffle_roi) & (~bottom_roi) & (~surface_roi)

    return {
        "impeller_discharge": impeller_roi,
        "wall_baffles": baffle_roi,
        "dished_bottom": bottom_roi,
        "free_surface": surface_roi,
        "bulk_recirculation": bulk_roi,
    }


def evaluate_roi_metrics(
    pred_vel: Union[np.ndarray, torch.Tensor],
    true_vel: Union[np.ndarray, torch.Tensor],
    roi_masks: Dict[str, np.ndarray],
    channel_dim: int = 1,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluates relative L2, L1, Linf, and angular alignment within each localized ROI.
    Ensures a 1% global velocity error does not mask a 20% error in the impeller zone.
    """
    p = to_numpy(pred_vel)
    t = to_numpy(true_vel)

    p = np.moveaxis(p, channel_dim, 0)
    t = np.moveaxis(t, channel_dim, 0)

    results = {}
    for roi_name, mask in roi_masks.items():
        mask_bool = to_numpy(mask).astype(bool)
        n_cells = int(np.sum(mask_bool))

        if n_cells == 0:
            results[roi_name] = {
                "active_cells": 0,
                "rel_l2": 0.0,
                "rel_l1": 0.0,
                "rel_linf": 0.0,
                "rmse": 0.0,
                "mae": 0.0,
                "mean_angle_deg": 0.0,
            }
            continue

        p_roi = p[:, mask_bool]
        t_roi = t[:, mask_bool]

        l2_err = relative_l2_error(p_roi, t_roi)
        l1_err = relative_l1_error(p_roi, t_roi)
        linf_err = relative_linf_error(p_roi, t_roi)
        rmse = float(np.sqrt(np.mean((p_roi - t_roi)**2)))
        mae = float(np.mean(np.abs(p_roi - t_roi)))

        # Angular alignment in this ROI
        mag_p = np.sqrt(p_roi[0]**2 + p_roi[1]**2 + p_roi[2]**2)
        mag_t = np.sqrt(t_roi[0]**2 + t_roi[1]**2 + t_roi[2]**2)
        active_flow = (mag_p > 1e-4) & (mag_t > 1e-4)

        if np.any(active_flow):
            dot = np.sum(p_roi[:, active_flow] * t_roi[:, active_flow], axis=0)
            denom = np.maximum(mag_p[active_flow] * mag_t[active_flow], 1e-12)
            angles_deg = np.degrees(np.arccos(np.clip(dot / denom, -1.0, 1.0)))
            mean_angle = float(np.mean(angles_deg))
        else:
            mean_angle = 0.0

        results[roi_name] = {
            "active_cells": n_cells,
            "rel_l2": l2_err,
            "rel_l1": l1_err,
            "rel_linf": linf_err,
            "rmse": rmse,
            "mae": mae,
            "mean_angle_deg": mean_angle,
        }

    return results
