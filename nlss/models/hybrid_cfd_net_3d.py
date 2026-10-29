"""
HybridCFDNet3D: Full Volumetric 3D Neural Operator for Incompressible & Compressible CFD.
Architectural Hierarchy:
  1. 3D Fourier Spectral Convolutions (SpectralConv3d):
     - Parameterizes global Green's function / elliptic pressure Poisson operator in 3D Fourier space.
     - Discretization-invariant mapping across volumetric grids (D x H x W).
  2. 3D Multi-Scale Dilated Residual Convolutions (Mesh3DConvBlock):
     - Resolves 3D near-wall viscous boundary layers, turbulent shear stresses, and boundary wakes.
  3. DNN 3D Inter-Layer Linker:
     - Gated residual highway dynamically balancing global 3D spectral modes with local shear gradients.
  4. 3D Divergence & Mass Continuity Invariant:
     - Evaluates div_3D(u) = du_x/dx + du_y/dy + du_z/dz ~ 0.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv3d(nn.Module):
    """
    3D Fourier Spectral Convolution Layer.
    Computes 3D real FFT (torch.fft.rfftn over (z, y, x)), applies complex tensor contractions
    over low-frequency Fourier modes, and applies inverse 3D real FFT.
    Discretization-invariant in 3D continuous space.
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int, modes3: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Modes along Z (depth)
        self.modes2 = modes2  # Modes along Y (height)
        self.modes3 = modes3  # Modes along X (width)

        self.scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat)
        )

    def compl_mul3d(self, input_tensor: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # (B, in_c, z, y, x) * (in_c, out_c, z, y, x) -> (B, out_c, z, y, x)
        return torch.einsum("bizyx,iozyx->bozyx", input_tensor, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D, H, W)
        B, C, D, H, W = x.shape
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        out_ft = torch.zeros(
            B, self.out_channels, D, H, W // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )

        m1 = min(self.modes1, D // 2)
        m2 = min(self.modes2, H // 2)
        m3 = min(self.modes3, W // 2 + 1)

        out_ft[:, :, :m1, :m2, :m3] = self.compl_mul3d(
            x_ft[:, :, :m1, :m2, :m3], self.weights1[:, :, :m1, :m2, :m3]
        )
        out_ft[:, :, -m1:, :m2, :m3] = self.compl_mul3d(
            x_ft[:, :, -m1:, :m2, :m3], self.weights2[:, :, -m1:, :m2, :m3]
        )

        return torch.fft.irfftn(out_ft, s=(D, H, W), dim=[-3, -2, -1])


class Mesh3DConvBlock(nn.Module):
    """
    Multi-Scale 3D Residual Convolutional Block.
    Captures localized viscous shear stresses, boundary layer friction, and baffle wakes.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.conv1 = nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2)
        self.norm = nn.GroupNorm(num_groups=min(4, hidden_dim), num_channels=hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        h = F.gelu(self.conv1(x))
        h = self.conv2(h)
        return F.gelu(self.norm(res + h))


class DNN3DInterLayerLinker(nn.Module):
    """
    Channel-wise 3D MLP linker with LayerNorm, GELU, and learnable gating highway.
    """

    def __init__(self, dim: int, expansion: int = 2):
        super().__init__()
        self.dim = dim
        hidden_dim = dim * expansion
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.norm = nn.LayerNorm(dim)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, f_spectral: torch.Tensor, f_conv: torch.Tensor) -> torch.Tensor:
        # f_spectral, f_conv: (B, D, H, W, C)
        combined = torch.cat([f_spectral, f_conv], dim=-1)
        linked = self.mlp(combined)
        g = torch.sigmoid(self.gate)
        return self.norm(f_spectral + g * linked + (1.0 - g) * f_conv)


class HybridCFDNet3D(nn.Module):
    """
    Full Volumetric 3D Neural Operator for CFD.
    State tensor: S_t = (u_x, u_y, u_z, p) in R^(4 x D x H x W).
    Predicts S_{t+dt} autoregressively with physical mass continuity preservation.
    """

    PRESETS = {
        "small": {"hidden_dim": 24, "modes": 6, "num_layers": 2},
        "medium": {"hidden_dim": 36, "modes": 8, "num_layers": 3},
        "large": {"hidden_dim": 48, "modes": 10, "num_layers": 4},
    }

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        hidden_dim: int = 36,
        modes: int = 8,
        num_layers: int = 3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.modes = modes
        self.num_layers = num_layers

        # 1. 3D Lifting Layer
        self.lifting = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=1),
        )

        # 2. Hybrid 3D Operator Layers
        self.spectral_layers = nn.ModuleList([
            SpectralConv3d(hidden_dim, hidden_dim, modes, modes, modes)
            for _ in range(num_layers)
        ])
        self.conv_layers = nn.ModuleList([
            Mesh3DConvBlock(hidden_dim)
            for _ in range(num_layers)
        ])
        self.linkers = nn.ModuleList([
            DNN3DInterLayerLinker(hidden_dim)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.GroupNorm(num_groups=min(4, hidden_dim), num_channels=hidden_dim)
            for _ in range(num_layers)
        ])

        # 3. 3D Projection Layer
        self.projection = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(hidden_dim, out_channels, kernel_size=1),
        )

    @classmethod
    def from_preset(cls, preset_name: str, in_channels: int = 4, out_channels: int = 4) -> "HybridCFDNet3D":
        if preset_name not in cls.PRESETS:
            raise ValueError(f"Unknown preset '{preset_name}'. Available: {list(cls.PRESETS.keys())}")
        cfg = cls.PRESETS[preset_name]
        return cls(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_dim=cfg["hidden_dim"],
            modes=cfg["modes"],
            num_layers=cfg["num_layers"],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 4, D, H, W)
        h = self.lifting(x)

        for spec, conv, linker, norm in zip(
            self.spectral_layers, self.conv_layers, self.linkers, self.norms
        ):
            f_spec = spec(h)
            f_conv = conv(h)

            # Permute to (B, D, H, W, C) for channel-wise linking
            f_spec_p = f_spec.permute(0, 2, 3, 4, 1).contiguous()
            f_conv_p = f_conv.permute(0, 2, 3, 4, 1).contiguous()
            f_linked = linker(f_spec_p, f_conv_p)
            f_out = f_linked.permute(0, 4, 1, 2, 3).contiguous()

            h = F.gelu(norm(h + f_out))

        delta = self.projection(h)
        return x + delta  # Residual update


def compute_divergence_3d(u: torch.Tensor, dx: float = 1.0, dy: float = 1.0, dz: float = 1.0) -> torch.Tensor:
    """
    Computes 3D discrete mass continuity residual:
      div_3d(u) = du/dx + dv/dy + dw/dz
    u shape: (B, 3, D, H, W) where components are (ux, uy, uz)
    """
    ux = u[:, 0]
    uy = u[:, 1]
    uz = u[:, 2]

    du_dx = torch.zeros_like(ux)
    dv_dy = torch.zeros_like(uy)
    dw_dz = torch.zeros_like(uz)

    du_dx[:, :, :, 1:-1] = (ux[:, :, :, 2:] - ux[:, :, :, :-2]) / (2.0 * dx)
    dv_dy[:, :, 1:-1, :] = (uy[:, :, 2:, :] - uy[:, :, :-2, :]) / (2.0 * dy)
    dw_dz[:, 1:-1, :, :] = (uz[:, 2:, :, :] - uz[:, :-2, :, :]) / (2.0 * dz)

    return du_dx + dv_dy + dw_dz


class ScaleInvariantRelativeLoss3D(nn.Module):
    """
    Scale-Invariant Multi-Channel Relative L2 Loss + 3D Mass Continuity Penalty.
    Ensures high-speed jets and low-speed recirculation zones receive uniform gradient magnitude.
    """

    def __init__(self, eps: float = 1e-4, div_weight: float = 0.05):
        super().__init__()
        self.eps = eps
        self.div_weight = div_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        # pred, target: (B, 4, D, H, W) -> [ux, uy, uz, p]
        pred_vel = pred[:, :3]
        targ_vel = target[:, :3]
        pred_p = pred[:, 3:]
        targ_p = target[:, 3:]

        # Vector Relative L2 for velocity field (integrated over all 3 vector components and volume)
        vel_diff = torch.norm(pred_vel - targ_vel, p=2, dim=(1, 2, 3, 4))
        vel_base = torch.norm(targ_vel, p=2, dim=(1, 2, 3, 4)) + self.eps
        loss_vel = torch.mean(vel_diff / vel_base)

        # Scalar Relative L2 for pressure field
        p_diff = torch.norm(pred_p - targ_p, p=2, dim=(1, 2, 3, 4))
        p_base = torch.norm(targ_p, p=2, dim=(1, 2, 3, 4)) + self.eps
        loss_p = torch.mean(p_diff / p_base)

        # 3D Incompressibility constraint div(u) ~ 0
        div_pred = compute_divergence_3d(pred_vel)
        loss_div = torch.mean(torch.abs(div_pred))

        total_loss = loss_vel + 0.5 * loss_p + self.div_weight * loss_div

        metrics = {
            "loss_total": float(total_loss.item()),
            "loss_vel": float(loss_vel.item()),
            "loss_p": float(loss_p.item()),
            "loss_div": float(loss_div.item()),
        }
        return total_loss, metrics
