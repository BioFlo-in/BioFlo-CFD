"""
Graph Neural Network (MeshGNN) for unstructured OpenFOAM computational meshes.
Operates directly on cell centers and face connectivity (owner/neighbour graph),
enabling surrogate modeling on complex geometries and unstructured polyhedral meshes.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class MeshMessagePassingBlock(nn.Module):
    """
    Edge message passing and node state aggregation block.
    """

    def __init__(self, hidden_dim: int, edge_dim: int = 3):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        node_features: (N_cells, hidden_dim)
        edge_index: (2, N_edges)
        edge_attr: (N_edges, edge_dim)
        """
        src, dst = edge_index[0], edge_index[1]

        # Gather node features for edges
        src_feat = node_features[src]
        dst_feat = node_features[dst]

        if edge_attr is not None:
            edge_inputs = torch.cat([src_feat, dst_feat, edge_attr], dim=-1)
        else:
            default_edge = torch.zeros(src_feat.shape[0], self.edge_mlp[0].in_features - 2 * src_feat.shape[1], device=src_feat.device)
            edge_inputs = torch.cat([src_feat, dst_feat, default_edge], dim=-1)

        # Message computation
        messages = self.edge_mlp(edge_inputs)  # (N_edges, hidden_dim)

        # Message aggregation (scatter sum)
        aggregated = torch.zeros_like(node_features)
        aggregated.index_add_(0, dst, messages)

        # Node update with residual connection
        updated = self.node_mlp(torch.cat([node_features, aggregated], dim=-1))
        return self.norm(node_features + updated)


class MeshGNN(nn.Module):
    """
    MeshGraphNet for Unstructured OpenFOAM CFD State Prediction.
    Maps S_t -> S_{t+dt} on unstructured cell grids.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        edge_dim: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 4,
        residual_connection: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.residual_connection = residual_connection

        # Node encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Message passing layers
        self.layers = nn.ModuleList(
            [MeshMessagePassingBlock(hidden_dim, edge_dim) for _ in range(num_layers)]
        )

        # Node decoder
        self.node_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_channels),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        node_features: (N_cells, in_channels)
        edge_index: (2, N_edges)
        edge_attr: (N_edges, edge_dim)
        """
        identity = node_features
        h = self.node_encoder(node_features)

        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)

        delta = self.node_decoder(h)

        if self.residual_connection:
            return identity + delta
        return delta
