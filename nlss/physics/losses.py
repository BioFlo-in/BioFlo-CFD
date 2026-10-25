"""
Physics-informed loss functions and metrics for Navier-Stokes CFD surrogate models.
Properly scales the learning objective with respect to the physical transition Delta S = S_{t+dt} - S_t,
ensuring initial losses start at O(1) (100% error) rather than falsely small O(10^-2) values.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_divergence_2d(u_field: torch.Tensor, dx: float = 1.0, dy: float = 1.0) -> torch.Tensor:
    """
    Computes numerical divergence div(u) = du/dx + dv/dy using central finite differences.
    u_field shape: (B, 2, H, W) or (B, >=2, H, W) where channel 0 is u, channel 1 is v.
    """
    u = u_field[:, 0:1, :, :]
    v = u_field[:, 1:2, :, :]

    du_dx = (u[:, :, :, 2:] - u[:, :, :, :-2]) / (2.0 * dx)
    dv_dy = (v[:, :, 2:, :] - v[:, :, :-2, :]) / (2.0 * dy)

    du_dx_inner = du_dx[:, :, 1:-1, :]
    dv_dy_inner = dv_dy[:, :, :, 1:-1]

    return du_dx_inner + dv_dy_inner


def compute_vorticity_2d(u_field: torch.Tensor, dx: float = 1.0, dy: float = 1.0) -> torch.Tensor:
    """Computes 2D vorticity omega = dv/dx - du/dy."""
    u = u_field[:, 0:1, :, :]
    v = u_field[:, 1:2, :, :]

    dv_dx = (v[:, :, :, 2:] - v[:, :, :, :-2]) / (2.0 * dx)
    du_dy = (u[:, :, 2:, :] - u[:, :, :-2, :]) / (2.0 * dy)

    dv_dx_inner = dv_dx[:, :, 1:-1, :]
    du_dy_inner = du_dy[:, :, :, 1:-1]

    return dv_dx_inner - du_dy_inner


def compute_cfd_importance_weight(
    target: torch.Tensor,
    input_state: Optional[torch.Tensor] = None,
    base_weight: float = 0.1,
    alpha_grad: float = 3.0,
    alpha_vort: float = 3.0,
    alpha_dyn: float = 4.0,
    power: float = 1.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Computes spatial importance map W(x, y) so that quiescent/unimportant areas
    do NOT artificially suppress the loss function.
    
    Assigns high importance weights (>10-100x) to:
      1. Dynamic transition activity / fluid acceleration (||S_{t+dt} - S_t||).
      2. High velocity shear & strain gradients (||nabla u||).
      3. Active vortex cores and rotational eddy structures (|omega|).
    
    Quiescent / dead zones receive small base weight (~0.1).
    Normalized such that spatial mean(W) = 1.0.
    """
    B, C, H, W = target.shape
    device = target.device

    # Base weight for quiescent zones
    weight = torch.full((B, 1, H, W), base_weight, device=device, dtype=target.dtype)

    # 1. Dynamic transition activity (regions where fluid is actually changing state)
    if input_state is not None:
        delta = torch.norm(target[:, :2] - input_state[:, :2], p=2, dim=1, keepdim=True)
        delta_max = torch.amax(delta, dim=(-2, -1), keepdim=True) + eps
        weight = weight + alpha_dyn * ((delta / delta_max) ** power)

    # 2. Velocity gradients & vorticity structures
    if C >= 2:
        u = target[:, 0:1]
        v = target[:, 1:2]
        du_dx = (u[:, :, :, 2:] - u[:, :, :, :-2]) / 2.0
        du_dy = (u[:, :, 2:, :] - u[:, :, :-2, :]) / 2.0
        dv_dx = (v[:, :, :, 2:] - v[:, :, :, :-2]) / 2.0
        dv_dy = (v[:, :, 2:, :] - v[:, :, :-2, :]) / 2.0

        pad = (1, 1, 1, 1)
        grad_mag = F.pad(
            torch.sqrt(
                du_dx[:, :, 1:-1, :] ** 2
                + du_dy[:, :, :, 1:-1] ** 2
                + dv_dx[:, :, 1:-1, :] ** 2
                + dv_dy[:, :, :, 1:-1] ** 2
                + eps
            ),
            pad,
        )
        vort_mag = F.pad(torch.abs(dv_dx[:, :, 1:-1, :] - du_dy[:, :, :, 1:-1]), pad)

        grad_max = torch.amax(grad_mag, dim=(-2, -1), keepdim=True) + eps
        vort_max = torch.amax(vort_mag, dim=(-2, -1), keepdim=True) + eps

        weight = (
            weight
            + alpha_grad * ((grad_mag / grad_max) ** power)
            + alpha_vort * ((vort_mag / vort_max) ** power)
        )

    # Normalize so mean weight over spatial domain is exactly 1.0
    weight_mean = torch.mean(weight, dim=(-2, -1), keepdim=True) + eps
    return weight / weight_mean


class DeltaTransitionLoss(nn.Module):
    """
    Relative loss computed on the physical state change Delta S = S_{t+dt} - S_t,
    with spatial importance weighting to prevent quiescent dead-zones
    from artificially suppressing the learning objective.
    """

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        input_state: Optional[torch.Tensor] = None,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input_state is not None:
            delta_pred = pred - input_state
            delta_target = target - input_state
        else:
            delta_pred = pred
            delta_target = target

        diff_sq = (delta_pred - delta_target) ** 2
        target_sq = delta_target ** 2

        if weight is not None:
            diff_norm = torch.sqrt(torch.sum(weight * diff_sq, dim=(-2, -1)) + self.eps)
            target_norm = torch.sqrt(torch.sum(weight * target_sq, dim=(-2, -1)) + self.eps)
        else:
            diff_norm = torch.norm(delta_pred - delta_target, p=2, dim=(-2, -1))
            target_norm = torch.norm(delta_target, p=2, dim=(-2, -1))

        return torch.mean(diff_norm / (target_norm + self.eps))


class NormalizedMSELoss(nn.Module):
    """
    Normalized Mean Squared Error scaled by target variance,
    with spatial importance weighting.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        input_state: Optional[torch.Tensor] = None,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input_state is not None:
            delta_pred = pred - input_state
            delta_target = target - input_state
        else:
            delta_pred = pred
            delta_target = target

        diff_sq = (delta_pred - delta_target) ** 2

        if weight is not None:
            weighted_mse = torch.mean(weight * diff_sq)
            target_mean = torch.mean(weight * delta_target, dim=(-2, -1), keepdim=True)
            weighted_var = torch.mean(weight * ((delta_target - target_mean) ** 2)) + self.eps
            return weighted_mse / weighted_var
        else:
            mse = torch.mean(diff_sq)
            var = torch.var(delta_target) + self.eps
            return mse / var


class PhysicsInformedCFDLoss(nn.Module):
    """
    Spatially-Adaptive Physics-Informed CFD surrogate loss.
    L = L_delta_weighted + lambda_nmse * L_nmse_weighted + lambda_div * L_div + lambda_vort * L_vort

    Uses spatial importance weighting W(x, y) to prevent quiescent / dead fluid regions
    from artificially suppressing the loss, focusing gradients on:
      - Active dynamic acceleration zones
      - High velocity gradient shear layers
      - Boundary layers and vortex cores
    """

    def __init__(
        self,
        lambda_nmse: float = 0.5,
        lambda_div: float = 0.1,
        lambda_vort: float = 0.05,
        dx: float = 1.0,
        dy: float = 1.0,
        use_importance_weighting: bool = True,
        base_weight: float = 0.1,
        alpha_grad: float = 3.0,
        alpha_vort: float = 3.0,
        alpha_dyn: float = 4.0,
    ):
        super().__init__()
        self.lambda_nmse = lambda_nmse
        self.lambda_div = lambda_div
        self.lambda_vort = lambda_vort
        self.dx = dx
        self.dy = dy
        self.use_importance_weighting = use_importance_weighting
        self.base_weight = base_weight
        self.alpha_grad = alpha_grad
        self.alpha_vort = alpha_vort
        self.alpha_dyn = alpha_dyn
        self.delta_loss = DeltaTransitionLoss()
        self.nmse_loss = NormalizedMSELoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        input_state: Optional[torch.Tensor] = None,
        weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Compute spatial importance weight if enabled and not provided
        if self.use_importance_weighting and weight is None:
            weight = compute_cfd_importance_weight(
                target=target,
                input_state=input_state,
                base_weight=self.base_weight,
                alpha_grad=self.alpha_grad,
                alpha_vort=self.alpha_vort,
                alpha_dyn=self.alpha_dyn,
            )

        # 1. Spatially-Weighted Delta Transition Loss
        l_delta = self.delta_loss(pred, target, input_state=input_state, weight=weight)

        # 2. Spatially-Weighted Normalized MSE
        l_nmse = self.nmse_loss(pred, target, input_state=input_state, weight=weight)

        total_loss = l_delta + self.lambda_nmse * l_nmse

        metrics = {
            "loss_delta": float(l_delta.item()),
            "loss_nmse": float(l_nmse.item()),
            "loss_data": float(total_loss.item()),
        }

        # 3. Spatially-Weighted Continuity (divergence-free) penalty
        if pred.shape[1] >= 2 and self.lambda_div > 0:
            div_pred = compute_divergence_2d(pred, dx=self.dx, dy=self.dy)
            if weight is not None:
                w_inner = weight[:, :, 1:-1, 1:-1]
                l_div = torch.mean(w_inner * (div_pred ** 2))
            else:
                l_div = torch.mean(div_pred ** 2)
            total_loss = total_loss + self.lambda_div * l_div
            metrics["loss_div"] = float(l_div.item())

        # 4. Spatially-Weighted Vorticity preservation
        if pred.shape[1] >= 2 and self.lambda_vort > 0:
            vort_pred = compute_vorticity_2d(pred, dx=self.dx, dy=self.dy)
            vort_target = compute_vorticity_2d(target, dx=self.dx, dy=self.dy)
            if weight is not None:
                w_inner = weight[:, :, 1:-1, 1:-1]
                l_vort = torch.mean(w_inner * ((vort_pred - vort_target) ** 2))
            else:
                l_vort = torch.mean((vort_pred - vort_target) ** 2)
            total_loss = total_loss + self.lambda_vort * l_vort
            metrics["loss_vort"] = float(l_vort.item())

        metrics["loss_total"] = float(total_loss.item())
        metrics["loss_rel_l2"] = float(l_delta.item())

        return total_loss, metrics
