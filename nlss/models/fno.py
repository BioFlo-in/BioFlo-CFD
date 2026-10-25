"""
Fourier Neural Operator (FNO-2D) for CFD simulation acceleration.
Learns resolution-invariant operator mapping from state S_t = (u, v, p) to S_{t+dt}.
Based on spectral convolutions in Fourier frequency domain.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """
    2D Fourier layer. Computes FFT, linear transform on low Fourier modes, and inverse FFT.
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes along dimension 1
        self.modes2 = modes2  # Number of Fourier modes along dimension 2

        self.scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, input_tensor: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Complex multiplication along channels: (batch, in_c, x, y) * (in_c, out_c, x, y) -> (batch, out_c, x, y)"""
        return torch.einsum("bixy,ioxy->boxy", input_tensor, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, channels, H, W)
        batchsize, _, H, W = x.shape

        # Compute 2D Real FFT
        x_ft = torch.fft.rfft2(x)

        # Output Fourier representation initialized to zeros
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            H,
            W // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        m1 = min(self.modes1, H // 2)
        m2 = min(self.modes2, W // 2 + 1)

        # Multiply relevant Fourier modes
        out_ft[:, :, :m1, :m2] = self.compl_mul2d(
            x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2]
        )
        out_ft[:, :, -m1:, :m2] = self.compl_mul2d(
            x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2]
        )

        # Inverse FFT back to spatial domain
        x_out = torch.fft.irfft2(out_ft, s=(H, W))
        return x_out


class FNO2d(nn.Module):
    """
    Fourier Neural Operator (FNO-2D) for OpenFOAM CFD State-Space Stepping.
    Inputs:
        Tensor of shape (B, in_channels, H, W)
        e.g., in_channels = 3 for (u_x, u_y, p)
    Outputs:
        Tensor of shape (B, out_channels, H, W)
        predicted state S_{t+dt} (or delta S_t when residual=True)
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        modes1: int = 16,
        modes2: int = 16,
        width: int = 48,
        num_layers: int = 4,
        residual_connection: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.residual_connection = residual_connection

        # Lifting projection: maps physical state channels (+ optional spatial coordinates) to latent width
        self.fc0 = nn.Linear(in_channels + 2, width)

        # Spectral convolution blocks
        self.conv_layers = nn.ModuleList(
            [SpectralConv2d(width, width, modes1, modes2) for _ in range(num_layers)]
        )
        self.w_layers = nn.ModuleList(
            [nn.Conv2d(width, width, kernel_size=1) for _ in range(num_layers)]
        )

        # Decoding projection: maps latent width to physical state
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def get_grid(self, shape: torch.Size, device: torch.device) -> torch.Tensor:
        """Constructs normalized (x, y) spatial coordinate grid for geometry conditioning."""
        batchsize, _, size_x, size_y = shape
        gridx = torch.linspace(0, 1, size_x, dtype=torch.float32, device=device)
        gridx = gridx.reshape(1, 1, size_x, 1).repeat([batchsize, 1, 1, size_y])
        gridy = torch.linspace(0, 1, size_y, dtype=torch.float32, device=device)
        gridy = gridy.reshape(1, 1, 1, size_y).repeat([batchsize, 1, size_x, 1])
        return torch.cat((gridx, gridy), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        identity = x
        grid = self.get_grid(x.shape, x.device)
        x_with_grid = torch.cat((x, grid), dim=1)

        # Lift channels: (B, C+2, H, W) -> (B, H, W, width)
        x_perm = x_with_grid.permute(0, 2, 3, 1)
        x_latent = self.fc0(x_perm).permute(0, 3, 1, 2)

        for conv, w in zip(self.conv_layers, self.w_layers):
            x1 = conv(x_latent)
            x2 = w(x_latent)
            x_latent = F.gelu(x1 + x2)

        # Decode back: (B, width, H, W) -> (B, H, W, width) -> (B, H, W, out_channels)
        x_latent = x_latent.permute(0, 2, 3, 1)
        x_out = self.fc1(x_latent)
        x_out = F.gelu(x_out)
        out = self.fc2(x_out).permute(0, 3, 1, 2)

        if self.residual_connection:
            # Predict residual update: S_{t+dt} = S_t + delta S
            return identity + out
        return out
