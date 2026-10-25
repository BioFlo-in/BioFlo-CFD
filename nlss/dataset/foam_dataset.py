"""
PyTorch Dataset and Normalization utilities for OpenFOAM CFD simulation data.
Supports single-step prediction pairs (S_t -> S_{t+dt}) and multi-step rollout sequences.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset

from ..io.foam_reader import load_openfoam_simulation_series


class FieldNormalizer:
    """Normalizes CFD physical fields (velocity components, pressure) to zero-mean unit-variance."""

    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None

    def fit(self, data: torch.Tensor) -> "FieldNormalizer":
        # data shape: (N_samples, C, H, W) or (N_samples, N_cells, C)
        # Compute stats over samples and spatial dimensions
        if data.ndim == 4:
            self.mean = torch.mean(data, dim=(0, 2, 3), keepdim=True)
            self.std = torch.std(data, dim=(0, 2, 3), keepdim=True) + self.eps
        else:
            self.mean = torch.mean(data, dim=(0, 1), keepdim=True)
            self.std = torch.std(data, dim=(0, 1), keepdim=True) + self.eps
        return self

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            return x
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            return x
        return x * self.std.to(x.device) + self.mean.to(x.device)

    def to_dict(self) -> Dict[str, list]:
        return {
            "mean": self.mean.squeeze().tolist() if self.mean is not None else [],
            "std": self.std.squeeze().tolist() if self.std is not None else [],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, list]) -> "FieldNormalizer":
        norm = cls()
        if d.get("mean"):
            norm.mean = torch.tensor(d["mean"]).reshape(1, -1, 1, 1)
            norm.std = torch.tensor(d["std"]).reshape(1, -1, 1, 1)
        return norm


class CFDTrajectoryDataset(Dataset):
    """
    Dataset for training autoregressive surrogate models on CFD time trajectories.
    Supports either a single trajectory (T, C, H, W) or a list of trajectories [(T_i, C, H, W)].
    Generates training pairs: (S_t, S_{t+1}) or rollout sequences (S_t, S_{t+1}, ..., S_{t+K}).
    """

    def __init__(
        self,
        trajectories: Union[np.ndarray, torch.Tensor, List[Union[np.ndarray, torch.Tensor]]],
        rollout_steps: int = 1,
        normalizer: Optional[FieldNormalizer] = None,
    ):
        self.rollout_steps = rollout_steps
        if isinstance(trajectories, list):
            self.traj_list = [
                torch.from_numpy(t).float() if isinstance(t, np.ndarray) else t.float()
                for t in trajectories
            ]
        else:
            t_tensor = (
                torch.from_numpy(trajectories).float()
                if isinstance(trajectories, np.ndarray)
                else trajectories.float()
            )
            self.traj_list = [t_tensor]

        # Fit normalizer on all trajectories concatenated along time
        if normalizer is None:
            all_data = torch.cat(self.traj_list, dim=0)
            self.normalizer = FieldNormalizer().fit(all_data)
        else:
            self.normalizer = normalizer

        # Normalize trajectories
        self.norm_trajs = [self.normalizer.normalize(t) for t in self.traj_list]

        # Build index mapping: (traj_idx, step_idx)
        self.indices = []
        for t_idx, traj in enumerate(self.norm_trajs):
            T = traj.shape[0]
            for step in range(T - self.rollout_steps):
                self.indices.append((t_idx, step))

        # Backward compatibility attributes
        self.data = self.traj_list[0]
        self.normalized_data = self.norm_trajs[0]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t_idx, step = self.indices[idx]
        traj = self.norm_trajs[t_idx]
        inp = traj[step]
        if self.rollout_steps == 1:
            target = traj[step + 1]
        else:
            target = traj[step + 1 : step + 1 + self.rollout_steps]
        return inp, target


def load_dataset_from_openfoam_case(
    case_dir: Union[str, Path],
    grid_shape: Optional[Tuple[int, int]] = None,
    rollout_steps: int = 1,
) -> Tuple[CFDTrajectoryDataset, FieldNormalizer]:
    """
    Loads CFD simulation trajectory directly from an OpenFOAM case directory.
    If grid_shape=(H, W) is provided, reshapes flattened OpenFOAM cell lists into 2D field tensors.
    """
    case_data = load_openfoam_simulation_series(case_dir, fields=["U", "p"])
    u_traj = case_data["U"]  # (T, N_cells, 3) or (T, N_cells, 2)
    p_traj = case_data["p"]  # (T, N_cells) or (T, N_cells, 1)

    if isinstance(u_traj, list):
        try:
            u_traj = np.stack(u_traj, axis=0)
            p_traj = np.stack(p_traj, axis=0)
        except Exception:
            raise ValueError(f"Inconsistent cell dimensions across time steps in {case_dir}")

    T = len(case_data["times"])
    if grid_shape is not None:
        H, W = grid_shape
        if u_traj.shape[1] == H * W:
            # Extract u_x and u_y
            ux = u_traj[:, :, 0].reshape(T, H, W)
            uy = u_traj[:, :, 1].reshape(T, H, W)
            p = p_traj.reshape(T, H, W)
        else:
            # Resample general unstructured cell lists onto uniform (H, W) grid
            n_cells = u_traj.shape[1]
            orig_indices = np.linspace(0, 1, n_cells)
            target_indices = np.linspace(0, 1, H * W)
            ux = np.zeros((T, H, W), dtype=np.float32)
            uy = np.zeros((T, H, W), dtype=np.float32)
            p = np.zeros((T, H, W), dtype=np.float32)
            for t in range(T):
                ux[t] = np.interp(target_indices, orig_indices, u_traj[t, :, 0]).reshape(H, W)
                uy[t] = np.interp(target_indices, orig_indices, u_traj[t, :, 1]).reshape(H, W)
                p[t] = np.interp(target_indices, orig_indices, p_traj[t].flatten()).reshape(H, W)
        # Combine into (T, 3, H, W)
        stacked = np.stack([ux, uy, p], axis=1)
    else:
        # 1D cell list format (T, 3, N_cells)
        ux = u_traj[:, :, 0]
        uy = u_traj[:, :, 1]
        p = p_traj.squeeze(-1) if p_traj.ndim == 3 else p_traj
        stacked = np.stack([ux, uy, p], axis=1)

    normalizer = FieldNormalizer().fit(torch.from_numpy(stacked).float())
    dataset = CFDTrajectoryDataset(stacked, rollout_steps=rollout_steps, normalizer=normalizer)
    return dataset, normalizer
