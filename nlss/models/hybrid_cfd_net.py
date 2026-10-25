"""
HybridCFDNet: Unified Multi-Scale Neural Operator + MeshGraphNet + Spatiotemporal Transformer Head.
Architectural Hierarchy:
  1. Multi-Scale Neural Operator (Fourier Spectral Convolutions + Multi-Dilation Residual Convolutions):
     - Parameterizes global integral kernel in Fourier frequency domain for non-local pressure waves.
     - Multi-scale dilated convolutions for local viscous boundary layer shear and near-wall gradients.
  2. Mesh GraphNet (GNN Cell-Face Message Passing):
     - Operates directly on computational mesh cell topology and face connectivity.
     - Computes directional face fluxes and aggregates cell-centroid messages, respecting physical conservation laws.
  3. DNN Inter-Layer Linker (Cross-Modality Deep Highway):
     - Deep Neural Network linking and bridging Spectral, Convolutional, and Graph representations between layers.
     - Adaptive gating mechanism dynamically balancing global spectral modes with localized graph cell fluxes.
  4. Spatiotemporal Transformer Head:
     - Applies Causal Multi-Head Self-Attention over historical state sequences [S_{t-K+1}, ..., S_t].
     - Multi-head spatial attention across grid tokens for long-range velocity-pressure coupling.
     - Predicts strictly physical state rate of change Delta S_{t+1}.
  5. Hard Boundary Condition Enforcement:
     - Guarantees exact Dirichlet no-slip wall constraints and lid velocities.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .fno import SpectralConv2d
from .dense_cfd_net import DenseConvBlock
from .transformer_stepper import CausalTemporalSelfAttention


class DNNInterLayerLinker(nn.Module):
    """
    Deep Neural Network (DNN) Cross-Modality Linker between layers.
    Bridges and fuses:
      1. Global Spectral Fourier representations
      2. Multi-scale convolutional local gradients
      3. Mesh Graph cell-face flux aggregations
    Features:
      - Channel-wise MLP with LayerNorm, GELU, and residual highway connection
      - Adaptive gating parameter (alpha) to dynamically balance spectral vs graph signals
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
        # Learnable gating parameter initialized near 0 for smooth residual highway learning
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, f_spectral: torch.Tensor, f_graph: torch.Tensor) -> torch.Tensor:
        """
        f_spectral: (B, H, W, C)
        f_graph:    (B, H, W, C)
        Returns:    (B, H, W, C) fused representation
        """
        combined = torch.cat([f_spectral, f_graph], dim=-1)
        linked = self.mlp(combined)
        g = torch.sigmoid(self.gate)
        # Gated residual highway fusion
        out = self.norm(f_spectral + g * linked + (1.0 - g) * f_graph)
        return out


class MeshGridGraphBlock(nn.Module):
    """
    Mesh Graph Message-Passing Block for computational CFD grids and meshes.
    Computes flux exchange across cell faces (North, South, East, West).
    Computes:
      m_{ij} = EdgeMLP([h_i, h_j, e_ij])
      h_i'   = NodeMLP([h_i, sum_{j in N(i)} m_ij]) + h_i
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Edge message generator (takes source, destination, and face orientation vector)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Node update MLP (aggregates incoming face flux messages)
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, H, W, C) cell-centered feature tensor
        """
        B, H, W, C = x.shape
        device = x.device

        # Cardinal directional neighbor shifts (Face connectivity)
        # North (i-1), South (i+1), East (j+1), West (j-1)
        x_north = torch.roll(x, shifts=-1, dims=1)
        x_south = torch.roll(x, shifts=1, dims=1)
        x_east = torch.roll(x, shifts=-1, dims=2)
        x_west = torch.roll(x, shifts=1, dims=2)

        # Unit normal vectors for cell faces
        n_north = torch.tensor([0.0, 1.0], device=device).view(1, 1, 1, 2).expand(B, H, W, 2)
        n_south = torch.tensor([0.0, -1.0], device=device).view(1, 1, 1, 2).expand(B, H, W, 2)
        n_east = torch.tensor([1.0, 0.0], device=device).view(1, 1, 1, 2).expand(B, H, W, 2)
        n_west = torch.tensor([-1.0, 0.0], device=device).view(1, 1, 1, 2).expand(B, H, W, 2)

        # Compute edge flux messages across each face
        m_north = self.edge_mlp(torch.cat([x, x_north, n_north], dim=-1))
        m_south = self.edge_mlp(torch.cat([x, x_south, n_south], dim=-1))
        m_east = self.edge_mlp(torch.cat([x, x_east, n_east], dim=-1))
        m_west = self.edge_mlp(torch.cat([x, x_west, n_west], dim=-1))

        # Aggregate incoming face messages (scatter sum)
        agg_messages = m_north + m_south + m_east + m_west

        # Node update with residual connection
        updated_node = self.node_mlp(torch.cat([x, agg_messages], dim=-1))
        return self.norm(x + updated_node)


class HybridOperatorLayer(nn.Module):
    """
    Single Hybrid Layer combining:
      1. Multi-Scale Neural Operator (Fourier Spectral Conv + Dense Multi-Dilation Conv)
      2. Mesh GraphNet Message Passing across cell faces
      3. Inter-Layer DNN Linker fusing both representations
    """

    def __init__(self, hidden_dim: int, modes1: int = 16, modes2: int = 16):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Branch 1: Spectral Fourier Operator
        self.spectral_conv = SpectralConv2d(hidden_dim, hidden_dim, modes1, modes2)

        # Branch 1: Multi-scale Dilated Residual Convolution
        self.dense_conv = DenseConvBlock(hidden_dim, hidden_dim)

        # Branch 2: Mesh GraphNet Cell-Face Message Passing
        self.mesh_gnn = MeshGridGraphBlock(hidden_dim)

        # Branch 3: DNN Inter-Layer Linker
        self.dnn_linker = DNNInterLayerLinker(hidden_dim, expansion=2)

        self.norm = nn.GroupNorm(num_groups=min(8, hidden_dim), num_channels=hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        """
        # --- Branch 1: Multi-Scale Neural Operator ---
        # Spectral Fourier conv (global modes)
        f_spec = self.spectral_conv(x)
        # Dense multi-dilation conv (local shear & wall gradients)
        f_dense = self.dense_conv(x)
        f_operator = F.gelu(f_spec + f_dense)  # (B, C, H, W)

        # --- Branch 2: Mesh GraphNet ---
        # Permute to (B, H, W, C) for node-edge message passing
        x_perm = x.permute(0, 2, 3, 1).contiguous()
        f_graph = self.mesh_gnn(x_perm)  # (B, H, W, C)

        # --- Branch 3: DNN Inter-Layer Linker ---
        f_operator_perm = f_operator.permute(0, 2, 3, 1).contiguous()
        f_linked = self.dnn_linker(f_operator_perm, f_graph)  # (B, H, W, C)

        # Permute back to (B, C, H, W) with residual connection
        f_out = f_linked.permute(0, 3, 1, 2).contiguous()
        return F.gelu(self.norm(x + f_out))


class SpatiotemporalTransformerHead(nn.Module):
    """
    Spatiotemporal Transformer Head with Spatial Patch Embedding.
    Processes feature tokens with:
      1. Spatial Patch Compression (downsamples H, W to patch grid e.g. 16x16 = 256 tokens)
      2. Causal Temporal Attention across historical context states
      3. Spatial Multi-Head Self-Attention over patch tokens (O(P^2) rather than O((HW)^2))
      4. Transformer Decoder with skip connection back to full resolution (H, W)
      5. Generates physical state update Delta S_{t+1}
    """

    def __init__(self, hidden_dim: int, num_heads: int = 8, out_channels: int = 3, patch_size: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.temporal_attn = CausalTemporalSelfAttention(embed_dim=hidden_dim, num_heads=num_heads)

        # Patch downsampler for compact attention tokens (e.g. 64x64 -> 16x16 = 256 tokens)
        self.patch_down = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
        )

        # Spatial token projection & attention on compact patch tokens
        self.spatial_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Decoder upsampler back to full resolution
        self.patch_up = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
        )

        # Final projection decoder (combines full-res skip with patch attention)
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W) or (B, T, C, H, W)
        Returns: (B, out_channels, H, W)
        """
        if x.dim() == 5:
            # Spatiotemporal sequence: (B, T, C, H, W)
            B, T, C, H, W = x.shape
            skip = x[:, -1]  # Full resolution skip from latest time step
            # Downsample to patch tokens: (B*T, C, H_p, W_p)
            x_flat = x.view(B * T, C, H, W)
            patches = self.patch_down(x_flat)
            _, _, Hp, Wp = patches.shape
            # Causal temporal attention over time steps
            tokens = patches.permute(0, 2, 3, 1).contiguous().view(B, T, Hp * Wp, C)
            tokens = tokens.permute(0, 2, 1, 3).contiguous().view(B * Hp * Wp, T, C)
            tokens = tokens + self.temporal_attn(tokens)
            last_tokens = tokens[:, -1, :].view(B, Hp * Wp, C)
        else:
            # Single state: (B, C, H, W)
            B, C, H, W = x.shape
            skip = x
            patches = self.patch_down(x)
            _, _, Hp, Wp = patches.shape
            last_tokens = patches.permute(0, 2, 3, 1).contiguous().view(B, Hp * Wp, C)

        # Spatial self-attention over compact patch tokens (FlashAttention with need_weights=False)
        proj = self.norm1(last_tokens)
        attn_out, _ = self.spatial_attn(proj, proj, proj, need_weights=False)
        spatial_features = self.norm2(last_tokens + attn_out)  # (B, Hp*Wp, C)

        # Reshape to (B, C, Hp, Wp) and upsample back to full resolution
        patch_map = spatial_features.view(B, Hp, Wp, C).permute(0, 3, 1, 2).contiguous()
        up_features = self.patch_up(patch_map)

        # Combine with full-resolution skip connection and decode
        fused = torch.cat([skip, up_features], dim=1)
        return self.decoder(fused)


class HybridCFDNet(nn.Module):
    """
    Unified Hybrid Architecture:
      [Multi-Scale Neural Operator] + [Mesh GraphNet] + [DNN Inter-Layer Linker] + [Spatiotemporal Transformer Head]

    Inputs:
      S_t: (B, 3, H, W) or window (B, K, 3, H, W)
    Outputs:
      S_{t+1}: (B, 3, H, W) satisfying physical boundary conditions
    """

    PRESETS = {
        "base": dict(hidden_dim=32, modes1=12, modes2=12, num_layers=2, num_heads=4),       # ~705K params
        "medium": dict(hidden_dim=48, modes1=14, modes2=14, num_layers=3, num_heads=6),     # ~3.0M params
        "large": dict(hidden_dim=64, modes1=16, modes2=16, num_layers=3, num_heads=8),      # ~6.8M params
        "xlarge": dict(hidden_dim=96, modes1=16, modes2=16, num_layers=4, num_heads=8),     # ~20.3M params
        "xxlarge": dict(hidden_dim=128, modes1=20, modes2=20, num_layers=4, num_heads=8),   # ~55.0M params
    }

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        hidden_dim: int = 64,
        modes1: int = 16,
        modes2: int = 16,
        num_layers: int = 3,
        num_heads: int = 8,
        enforce_boundary: bool = True,
        lid_velocity: float = 1.0,
        use_checkpointing: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.enforce_boundary = enforce_boundary
        self.lid_velocity = lid_velocity
        self.use_checkpointing = use_checkpointing

        # 1. Input Lifting: [ux, uy, p, x_coord, y_coord] -> hidden_dim
        self.lifting = nn.Sequential(
            nn.Conv2d(in_channels + 2, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )

        # 2. Stacked Hybrid Operator Layers with DNN Linkers
        self.hybrid_layers = nn.ModuleList(
            [
                HybridOperatorLayer(hidden_dim=hidden_dim, modes1=modes1, modes2=modes2)
                for _ in range(num_layers)
            ]
        )

        # 3. Spatiotemporal Transformer Head
        self.transformer_head = SpatiotemporalTransformerHead(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            out_channels=out_channels,
        )

    @classmethod
    def from_preset(cls, preset_name: str, **overrides) -> "HybridCFDNet":
        """Instantiates HybridCFDNet from a standard scaling preset ('base', 'medium', 'large', 'xlarge', 'xxlarge')."""
        preset_name = preset_name.lower()
        if preset_name not in cls.PRESETS:
            valid = list(cls.PRESETS.keys())
            raise ValueError(f"Unknown preset '{preset_name}'. Valid presets are: {valid}")
        kwargs = dict(cls.PRESETS[preset_name])
        kwargs.update(overrides)
        return cls(**kwargs)

    @property
    def num_parameters(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _build_grid_coords(self, shape: Tuple[int, int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Constructs normalized (x, y) spatial grid coordinates in [0, 1]."""
        ny, nx = shape
        gx = torch.linspace(0.0, 1.0, nx, device=device, dtype=dtype)
        gy = torch.linspace(0.0, 1.0, ny, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")
        return torch.stack([grid_x, grid_y], dim=0)

    def _enforce_physical_boundaries(self, state: torch.Tensor, input_ref: torch.Tensor) -> torch.Tensor:
        """
        Hard boundary condition enforcement:
        Preserves Dirichlet wall velocities, moving lids, and no-slip boundaries.
        """
        out = state.clone()
        # Enforce no-slip wall boundaries:
        # Left wall (x = 0)
        out[:, 0, :, 0] = input_ref[:, 0, :, 0]
        out[:, 1, :, 0] = input_ref[:, 1, :, 0]
        # Right wall (x = -1)
        out[:, 0, :, -1] = input_ref[:, 0, :, -1]
        out[:, 1, :, -1] = input_ref[:, 1, :, -1]
        # Bottom wall (y = 0)
        out[:, 0, 0, :] = input_ref[:, 0, 0, :]
        out[:, 1, 0, :] = input_ref[:, 1, 0, :]
        # Top wall (y = -1, lid or moving boundary)
        out[:, 0, -1, :] = input_ref[:, 0, -1, :]
        out[:, 1, -1, :] = input_ref[:, 1, -1, :]
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward step:
          x: (B, 3, H, W) or (B, T, 3, H, W)
          returns: S_{t+1} (B, 3, H, W)
        """
        if x.dim() == 5:
            # Context window: (B, T, 3, H, W)
            B, T, C, H, W = x.shape
            current_state = x[:, -1]
            ref_input = current_state
            # Process each timestep through lifting + hybrid layers
            grid = self._build_grid_coords((H, W), x.device, x.dtype)
            grid_b = grid.unsqueeze(0).expand(B * T, -1, -1, -1)
            x_flat = x.view(B * T, C, H, W)
            h = self.lifting(torch.cat([x_flat, grid_b], dim=1))
            for layer in self.hybrid_layers:
                if self.use_checkpointing and self.training:
                    h = checkpoint(layer, h, use_reentrant=False)
                else:
                    h = layer(h)
            h_seq = h.view(B, T, self.hidden_dim, H, W)
            delta_s = self.transformer_head(h_seq)
        else:
            # Single state: (B, 3, H, W)
            B, C, H, W = x.shape
            current_state = x
            ref_input = current_state
            grid = self._build_grid_coords((H, W), x.device, x.dtype)
            grid_b = grid.unsqueeze(0).expand(B, -1, -1, -1)
            h = self.lifting(torch.cat([x, grid_b], dim=1))
            for layer in self.hybrid_layers:
                if self.use_checkpointing and self.training:
                    h = checkpoint(layer, h, use_reentrant=False)
                else:
                    h = layer(h)
            delta_s = self.transformer_head(h)

        # Residual state update: S_{t+1} = S_t + Delta S
        s_next = current_state + delta_s

        if self.enforce_boundary:
            s_next = self._enforce_physical_boundaries(s_next, ref_input)

        return s_next
