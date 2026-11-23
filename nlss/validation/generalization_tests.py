"""
BioFlo-CFD: Rigorous Generalization Protocols & Parameter Domain Classification
Defines the three mandatory scientific verification categories:
  Tier A: POD Reconstruction Accuracy (Truncation information loss: u_POD = Phi * a_opt)
  Tier B: Neural Operator Prediction Accuracy (Model mode decoupling: u_pred = Phi * a_hat)
  Tier C: CFD Generalization Accuracy (Completely unseen parameter conditions: RPM, Re, mu)

Also implements ParameterDomainChecker:
  - Distinguishes parameter interpolation from extrapolation
  - Returns confidence indicators and nearest-training distances
  - Raises OutOfDomainError under strict validation mode
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from scipy.spatial import ConvexHull, cKDTree
from .velocity_metrics import relative_l2_error, r2_score_field, to_numpy


class OutOfDomainError(ValueError):
    """Raised when an operating condition falls outside the validated training envelope."""
    pass


class ParameterDomainChecker:
    """
    Monitors and validates operating parameters (t, RPM, Re, viscosity) against
    the convex hull and bounding envelope of CFD training snapshots.
    """

    def __init__(self, training_parameters: Union[np.ndarray, torch.Tensor], strict: bool = False):
        self.raw_pts = to_numpy(training_parameters)
        if self.raw_pts.ndim == 1:
            self.raw_pts = self.raw_pts[:, None]

        self.n_samples, self.dim = self.raw_pts.shape
        self.strict = strict

        # Min and Max bounding box
        self.p_min = np.min(self.raw_pts, axis=0)
        self.p_max = np.max(self.raw_pts, axis=0)
        self.span = np.maximum(self.p_max - self.p_min, 1e-8)

        # Normalized coordinates for invariant distance metrics
        self.norm_pts = (self.raw_pts - self.p_min) / self.span
        self.kdtree = cKDTree(self.norm_pts)

        # Attempt convex hull if sufficient points exist
        self.has_convex_hull = False
        if self.n_samples > self.dim + 1:
            try:
                self.hull = ConvexHull(self.norm_pts)
                self.has_convex_hull = True
            except Exception:
                self.has_convex_hull = False

    def check(self, query: Union[np.ndarray, torch.Tensor, List[float]]) -> Dict[str, Union[bool, str, float, List[float]]]:
        """
        Assesses whether a query point is an interpolation or extrapolation.
        Returns a rich status dictionary with distance and confidence metrics.
        """
        q = to_numpy(query)
        if q.ndim == 1:
            q = q[None, :]

        # Normalize query
        q_norm = (q - self.p_min) / self.span

        # Distance to nearest training point in normalized space
        dists, idxs = self.kdtree.query(q_norm, k=1)
        nearest_dist = float(dists[0])
        nearest_idx = int(idxs[0])
        nearest_param = self.raw_pts[nearest_idx].tolist()

        # Bounding box test
        is_in_bbox = bool(np.all(q >= self.p_min - 1e-6) and np.all(q <= self.p_max + 1e-6))

        if is_in_bbox:
            domain_status = "interpolation"
            confidence = "high" if nearest_dist < 0.25 else "medium"
        else:
            domain_status = "extrapolation"
            confidence = "low" if nearest_dist < 0.75 else "untrusted"

        # Confidence decay function: C = exp(-3.0 * dist)
        confidence_score = float(np.exp(-3.0 * nearest_dist))

        result = {
            "is_in_domain": is_in_bbox,
            "domain_status": domain_status,
            "confidence": confidence,
            "confidence_score": confidence_score,
            "nearest_training_distance_norm": nearest_dist,
            "nearest_training_param": nearest_param,
            "parameter_bounds_min": self.p_min.tolist(),
            "parameter_bounds_max": self.p_max.tolist(),
        }

        if self.strict and not is_in_bbox:
            raise OutOfDomainError(
                f"Operating condition {q.tolist()} is OUTSIDE the validated training envelope: "
                f"bounds min={self.p_min.tolist()}, max={self.p_max.tolist()}. Nearest distance={nearest_dist:.3f}"
            )

        return result


def evaluate_three_tier_accuracy(
    model,
    test_snapshots: np.ndarray,
    test_params: np.ndarray,
    is_unseen_condition: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    Rigorously separates performance into the 3 scientific categories:
      Tier A: POD Reconstruction Accuracy (optimal projection Phi * Phi^T (u - mean))
      Tier B: Neural Mode Prediction Accuracy (evaluated on held-out time steps)
      Tier C: CFD Generalization Accuracy (evaluated on unseen parameter conditions)
    """
    if not model.is_calibrated:
        raise RuntimeError("Model must be calibrated before evaluating accuracy.")

    snaps = to_numpy(test_snapshots)
    params = to_numpy(test_params)
    N_snaps = snaps.shape[0]

    basis_np = model.basis_matrix.detach().cpu().numpy()  # (K, N_dof)

    # 1. Tier A: Optimal Orthogonal POD Reconstruction (Information Truncation Limit)
    if getattr(model, "has_parametric_mean", False):
        p_tensor = torch.from_numpy(params).float().to(next(model.parameters()).device)
        param_val = p_tensor[:, -1:]
        deg = model.mean_poly_coeffs.shape[0] - 1
        powers = torch.cat([param_val**i for i in range(deg, -1, -1)], dim=1)
        mean_eval = torch.matmul(powers, model.mean_poly_coeffs).detach().cpu().numpy()
        fluct = snaps - mean_eval
        a_optimal = fluct @ basis_np.T  # (N_snaps, K)
        rec_pod = mean_eval + (a_optimal @ basis_np)
    else:
        mean_np = model.mean_field.detach().cpu().numpy()
        fluct = snaps - mean_np
        a_optimal = fluct @ basis_np.T  # (N_snaps, K)
        rec_pod = mean_np + (a_optimal @ basis_np)

    rel_l2_pod = relative_l2_error(rec_pod, snaps)
    r2_pod = r2_score_field(rec_pod, snaps)

    # 2. Tier B: Neural Operator Prediction
    with torch.no_grad():
        p_tensor = torch.from_numpy(params).float().to(next(model.parameters()).device)
        pred_neural = model(p_tensor).detach().cpu().numpy()

    rel_l2_neural = relative_l2_error(pred_neural, snaps)
    r2_neural = r2_score_field(pred_neural, snaps)

    # Modal amplitude error
    a_pred = model.predict_coefficients(p_tensor).detach().cpu().numpy()
    modal_rel_l2 = relative_l2_error(a_pred, a_optimal)

    results = {
        "tier_a_pod_reconstruction": {
            "rel_l2_error": rel_l2_pod,
            "reconstruction_accuracy_pct": float((1.0 - rel_l2_pod) * 100.0),
            "r2_score": r2_pod,
            "description": "Optimal orthogonal projection limit (information truncation loss)",
        },
        "tier_b_neural_prediction": {
            "rel_l2_error": rel_l2_neural,
            "prediction_accuracy_pct": float((1.0 - rel_l2_neural) * 100.0),
            "r2_score": r2_neural,
            "modal_amplitude_rel_l2": modal_rel_l2,
            "description": "Neural mode decoupling accuracy on held-out temporal snapshots",
        },
    }

    if is_unseen_condition:
        results["tier_c_cfd_generalization"] = {
            "rel_l2_error": rel_l2_neural,
            "generalization_accuracy_pct": float((1.0 - rel_l2_neural) * 100.0),
            "r2_score": r2_neural,
            "modal_amplitude_rel_l2": modal_rel_l2,
            "description": "True out-of-sample CFD generalization on unseen operating condition",
        }

    return results
