"""
FluidGPT: Autoregressive Next-State CFD Transformer.
Adopts LLM-style next-token / next-step prediction for fluid dynamics:
  1. Accepts a causal context window of K previous simulation states [S_{t-K+1}, ..., S_t].
  2. Applies Causal Multi-Head Temporal Self-Attention to extract velocity, acceleration, and vortex rotation.
  3. Predicts strictly ONE simulation step into the future: S_{t+1}.
  4. Generates long rollouts autoregressively (just like an LLM generates text token-by-token).
"""

from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalTemporalSelfAttention(nn.Module):
    """Causal Multi-Head Self-Attention over the time-step sequence dimension."""

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.05):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, Seq_Len, embed_dim)
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product with causal mask
        scale = 1.0 / (self.head_dim ** 0.5)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Upper triangular causal mask (prevents looking into future timesteps)
        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


class SpatialEncoder(nn.Module):
    """Encodes (C=3, H, W) fluid state into compact feature tokens."""

    def __init__(self, in_channels: int = 3, embed_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),  # (64, 32, 32)
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),          # (128, 16, 16)
            nn.GELU(),
            nn.Conv2d(128, embed_dim, kernel_size=4, stride=2, padding=1),   # (embed_dim, 8, 8)
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class SpatialDecoder(nn.Module):
    """Decodes token features back into full physical fluid state (C=3, H, W)."""

    def __init__(self, embed_dim: int = 256, out_channels: int = 3):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 128, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class FluidGPT(nn.Module):
    """
    LLM-Style Autoregressive Next-State Fluid Dynamics Transformer.
    Takes context window: [S_{t-K+1}, ..., S_t] (shape: B, K, 3, H, W)
    Predicts single next state: S_{t+1} (shape: B, 3, H, W)
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        embed_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        context_window: int = 4,
    ):
        super().__init__()
        self.context_window = context_window
        self.embed_dim = embed_dim

        # 1. Spatial Tokenizer
        self.spatial_encoder = SpatialEncoder(in_channels, embed_dim)
        self.spatial_decoder = SpatialDecoder(embed_dim, out_channels)

        # 2. Positional Encoding for Timesteps
        self.time_pos_embed = nn.Parameter(torch.randn(1, context_window, embed_dim, 1, 1) * 0.02)

        # 3. Transformer Temporal Attention Blocks
        self.attn_layers = nn.ModuleList([
            CausalTemporalSelfAttention(embed_dim=embed_dim, num_heads=num_heads)
            for _ in range(num_layers)
        ])
        self.mlp_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.GELU(),
                nn.Linear(embed_dim * 2, embed_dim),
            )
            for _ in range(num_layers)
        ])
        self.norm1 = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """
        history: Sequence of previous K states: (B, Seq_Len, 3, H, W)
                 If single state (B, 3, H, W), automatically wrapped to (B, 1, 3, H, W).
        Returns:
            next_state: Single 1-step prediction S_{t+1}: (B, 3, H, W)
        """
        if history.ndim == 4:
            history = history.unsqueeze(1)

        B, T, C, H, W = history.shape
        last_state = history[:, -1]  # S_t

        # 1. Encode each time frame spatially: (B * T, embed_dim, h_feat, w_feat)
        flat_hist = history.view(B * T, C, H, W)
        feat_maps = self.spatial_encoder(flat_hist)
        _, d, h_feat, w_feat = feat_maps.shape

        # Reshape to (B, T, d, h_feat, w_feat)
        feat_seq = feat_maps.view(B, T, d, h_feat, w_feat)

        # Add temporal positional embedding
        feat_seq = feat_seq + self.time_pos_embed[:, :T]

        # Flatten spatial feature grid into tokens for temporal attention: (B * N_tokens, T, d)
        tokens = feat_seq.permute(0, 3, 4, 1, 2).contiguous().view(B * h_feat * w_feat, T, d)

        # 2. Causal Temporal Attention across timesteps
        for attn, mlp, n1, n2 in zip(self.attn_layers, self.mlp_layers, self.norm1, self.norm2):
            tokens = tokens + attn(n1(tokens))
            tokens = tokens + mlp(n2(tokens))

        # 3. Extract the last token (at step t) to predict the next step t+1 (LLM style!)
        next_tokens = tokens[:, -1]  # (B * h_feat * w_feat, d)
        next_feat = next_tokens.view(B, h_feat, w_feat, d).permute(0, 3, 1, 2).contiguous()

        # 4. Decode next feature into spatial state increment Delta S
        delta = self.spatial_decoder(next_feat)

        # 5. Residual 1-step update: S_{t+1} = S_t + Delta S
        next_state = last_state + delta
        return next_state
