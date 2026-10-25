"""
Autoregressive Rollout Engine and Acceleration Benchmarking for OpenFOAM CFD.
Executes iterative time-stepping with physics boundary enforcement and measures
speedup relative to numerical CFD solvers.
"""

import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from ..dataset.foam_dataset import FieldNormalizer
from ..io.foam_writer import export_simulation_state_to_openfoam
from ..physics.losses import compute_divergence_2d


class RolloutEngine:
    """
    Executes autoregressive CFD rollouts using a trained surrogate model.
    """

    def __init__(
        self,
        model: nn.Module,
        normalizer: FieldNormalizer,
        device: Optional[torch.device] = None,
        boundary_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.normalizer = normalizer
        self.boundary_fn = boundary_fn

    @torch.no_grad()
    def predict_next_state(self, current_state: torch.Tensor) -> torch.Tensor:
        """
        Takes current unnormalized physical state (1, C, H, W) or (C, H, W)
        and predicts next physical state S_{t+dt}.
        """
        if current_state.ndim == 3:
            state_batch = current_state.unsqueeze(0).to(self.device)
        else:
            state_batch = current_state.to(self.device)

        # 1. Normalize
        norm_in = self.normalizer.normalize(state_batch)

        # 2. Model forward pass
        norm_next = self.model(norm_in)

        # 3. Denormalize back to physical units
        next_phys = self.normalizer.denormalize(norm_next)

        # 4. Optional physical boundary condition enforcement
        if self.boundary_fn is not None:
            next_phys = self.boundary_fn(next_phys)

        return next_phys.squeeze(0)

    @torch.no_grad()
    def rollout(
        self,
        initial_state: Union[np.ndarray, torch.Tensor],
        num_steps: int,
    ) -> Tuple[torch.Tensor, float]:
        """
        Autoregressively rolls out predictions for num_steps.
        Returns:
            trajectory: Tensor of shape (num_steps + 1, C, H, W)
            elapsed_time_sec: Wall-clock inference time in seconds
        """
        if isinstance(initial_state, np.ndarray):
            curr = torch.from_numpy(initial_state).float().to(self.device)
        else:
            curr = initial_state.float().to(self.device)

        if curr.ndim == 4 and curr.shape[0] == 1:
            curr = curr.squeeze(0)

        history = [curr.clone()]

        # Synchronize for accurate GPU benchmarking
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        for _ in range(num_steps):
            curr = self.predict_next_state(curr)
            history.append(curr.clone())

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        trajectory = torch.stack(history, dim=0)
        return trajectory, elapsed

    def export_rollout_to_openfoam(
        self,
        trajectory: torch.Tensor,
        times: List[float],
        output_case_dir: Union[str, Path],
    ) -> Path:
        """
        Exports the entire AI predicted rollout trajectory into an OpenFOAM case directory.
        """
        case_path = Path(output_case_dir)
        case_path.mkdir(parents=True, exist_ok=True)
        traj_np = trajectory.cpu().numpy()
        T = traj_np.shape[0]

        for step_idx in range(T):
            t_val = times[step_idx] if step_idx < len(times) else float(step_idx)
            state = traj_np[step_idx]  # (3, H, W)
            ux, uy, p = state[0], state[1], state[2]

            u_vec = np.column_stack([ux.flatten(), uy.flatten()])
            p_scalar = p.flatten()

            export_simulation_state_to_openfoam(
                case_dir=case_path,
                time_val=t_val,
                u_pred=u_vec,
                p_pred=p_scalar,
            )

        return case_path


def benchmark_acceleration(
    rollout_engine: RolloutEngine,
    ground_truth_trajectory: torch.Tensor,
    cfd_solver_time_sec: float,
    num_steps: Optional[int] = None,
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Benchmarks AI surrogate acceleration factor and accuracy against ground truth OpenFOAM simulation.
    """
    total_steps = ground_truth_trajectory.shape[0] - 1
    if num_steps is None or num_steps > total_steps:
        num_steps = total_steps

    init_state = ground_truth_trajectory[0]
    gt_subset = ground_truth_trajectory[: num_steps + 1].cpu()

    # Run AI Rollout
    pred_traj, ai_time_sec = rollout_engine.rollout(init_state, num_steps=num_steps)
    pred_subset = pred_traj.cpu()

    # Metrics
    speedup = cfd_solver_time_sec / max(ai_time_sec, 1e-7)
    time_per_step_ai_ms = (ai_time_sec / num_steps) * 1000.0
    time_per_step_cfd_ms = (cfd_solver_time_sec / num_steps) * 1000.0

    # Relative L2 error per time step
    rel_l2_errors = []
    rmse_errors = []
    for t in range(num_steps + 1):
        diff = pred_subset[t] - gt_subset[t]
        err_l2 = torch.norm(diff) / (torch.norm(gt_subset[t]) + 1e-6)
        err_rmse = torch.sqrt(torch.mean(diff**2))
        rel_l2_errors.append(float(err_l2.item()))
        rmse_errors.append(float(err_rmse.item()))

    mean_rel_l2 = float(np.mean(rel_l2_errors[1:])) if len(rel_l2_errors) > 1 else 0.0
    mean_rmse = float(np.mean(rmse_errors[1:])) if len(rmse_errors) > 1 else 0.0

    return {
        "num_steps": num_steps,
        "ai_rollout_time_sec": ai_time_sec,
        "cfd_solver_time_sec": cfd_solver_time_sec,
        "speedup_factor": speedup,
        "time_per_step_ai_ms": time_per_step_ai_ms,
        "time_per_step_cfd_ms": time_per_step_cfd_ms,
        "mean_relative_l2": mean_rel_l2,
        "mean_rmse": mean_rmse,
        "relative_l2_trajectory": np.array(rel_l2_errors),
        "rmse_trajectory": np.array(rmse_errors),
    }
