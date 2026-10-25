"""
DenseCFDNet: High-Capacity Dense Multi-Scale Neural Operator for OpenFOAM CFD Acceleration.
Combines:
  1. Multi-scale Fourier Spectral convolutions (capturing global pressure & large vortices).
  2. Dense residual convolutional blocks with dilated kernels (capturing boundary layers and shear gradients).
  3. Residual velocity stepping: S_{t+dt} = S_t + Delta S.
  4. Hard Boundary Condition Enforcement layer (exact Dirichlet/no-slip wall constraints).
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .fno import SpectralConv2d


class DenseConvBlock(nn.Module):
    """Dense convolutional block with multi-dilation branches and GroupNorm."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        mid_features = out_features // 2

        # Branch 1: Standard 3x3 conv
        self.conv1 = nn.Conv2d(in_features, mid_features, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=min(8, mid_features), num_channels=mid_features)

        # Branch 2: Dilated 3x3 conv (dilation=2) for wider receptive field
        self.conv2 = nn.Conv2d(in_features, mid_features, kernel_size=3, padding=2, dilation=2, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=min(8, mid_features), num_channels=mid_features)

        # Fusion 1x1 conv
        self.fuse = nn.Conv2d(mid_features * 2, out_features, kernel_size=1, bias=False)
        self.gn_fuse = nn.GroupNorm(num_groups=min(8, out_features), num_channels=out_features)

        # Shortcut projection if channel dimension changes
        self.shortcut = (
            nn.Conv2d(in_features, out_features, kernel_size=1, bias=False)
            if in_features != out_features
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = F.gelu(self.gn1(self.conv1(x)))
        b2 = F.gelu(self.gn2(self.conv2(x)))
        cat = torch.cat([b1, b2], dim=1)
        out = self.gn_fuse(self.fuse(cat))
        return F.gelu(out + self.shortcut(x))


class DenseCFDNet(nn.Module):
    """
    Dense Multi-Scale Neural Operator for Fluid Dynamics State Stepping.
    Inputs:
        S_t: (B, 3, H, W) [u_x, u_y, p]
    Outputs:
        S_{t+dt}: (B, 3, H, W)
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        hidden_dim: int = 96,
        modes1: int = 20,
        modes2: int = 20,
        num_blocks: int = 4,
        enforce_boundary: bool = True,
        lid_velocity: float = 1.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.enforce_boundary = enforce_boundary
        self.lid_velocity = lid_velocity

        # Input lifting: takes [ux, uy, p, x_coord, y_coord] -> hidden_dim
        self.lifting = nn.Sequential(
            nn.Conv2d(in_channels + 2, hidden_dim, kernel_size=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
        )

        # Spectral Fourier layers
        self.spectral_convs = nn.ModuleList(
            [SpectralConv2d(hidden_dim, hidden_dim, modes1, modes2) for _ in range(num_blocks)]
        )

        # Dense Multi-scale spatial blocks
        self.dense_blocks = nn.ModuleList(
            [DenseConvBlock(hidden_dim, hidden_dim) for _ in range(num_blocks)]
        )

        # Norms and cross-branch feature fusion
        self.norms = nn.ModuleList([nn.GroupNorm(8, hidden_dim) for _ in range(num_blocks)])

        # Output projection predicting rate of change: Delta S_t
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim // 2),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, out_channels, kernel_size=1),
        )

    def get_grid(self, shape: Tuple[int, int, int, int], device: torch.device) -> torch.Tensor:
        """Normalized spatial grid coords [-1, 1]."""
        b, _, h, w = shape
        x = torch.linspace(0.0, 1.0, w, device=device)
        y = torch.linspace(0.0, 1.0, h, device=device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).repeat(b, 1, 1, 1)
        return grid

    def apply_boundary_conditions(
        self, state: torch.Tensor, input_state: Optional[torch.Tensor] = None, boundary_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        General Dirichlet boundary condition preservation:
        If boundary_mask is provided, preserves prescribed values where mask==1.
        Otherwise, preserves static physical boundaries from input_state (e.g. no-slip walls and inlets).
        """
        if input_state is None:
            return state

        clamped = state.clone()
        if boundary_mask is not None:
            clamped = torch.where(boundary_mask.bool(), input_state, clamped)
        else:
            # Preserve no-slip walls (velocity near zero at boundary remains zero)
            # Left & Right walls
            for wall_idx in [0, -1]:
                # If input had ~0 velocity, clamp output to 0 (preserving no-slip)
                wall_u = input_state[:, 0:2, :, wall_idx]
                is_no_slip = (torch.abs(wall_u) < 1e-3)
                clamped[:, 0:2, :, wall_idx] = torch.where(is_no_slip, wall_u, clamped[:, 0:2, :, wall_idx])

            # Bottom wall
            wall_b = input_state[:, 0:2, 0, :]
            is_no_slip_b = (torch.abs(wall_b) < 1e-3)
            clamped[:, 0:2, 0, :] = torch.where(is_no_slip_b, wall_b, clamped[:, 0:2, 0, :])

        return clamped

    def forward(
        self,
        x: torch.Tensor,
        boundary_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # x: (B, 3, H, W)
        identity = x
        grid = self.get_grid(x.shape, x.device)
        h = self.lifting(torch.cat([x, grid], dim=1))

        # Multi-scale spectral and dense processing
        for spec_conv, dense_blk, norm in zip(self.spectral_convs, self.dense_blocks, self.norms):
            h_spec = spec_conv(h)
            h_dense = dense_blk(h)
            h = norm(h + h_spec + h_dense)

        # Predict incremental rate of change
        delta = self.decoder(h)
        next_state = identity + delta

        if self.enforce_boundary:
            next_state = self.apply_boundary_conditions(next_state, input_state=identity, boundary_mask=boundary_mask)

        return next_state
