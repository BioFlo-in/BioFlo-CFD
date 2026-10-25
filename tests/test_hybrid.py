import pytest
import torch
from nlss.models.hybrid_cfd_net import (
    HybridCFDNet,
    DNNInterLayerLinker,
    MeshGridGraphBlock,
    HybridOperatorLayer,
    SpatiotemporalTransformerHead,
)
from nlss.train import create_model


def test_dnn_inter_layer_linker():
    B, H, W, C = 2, 16, 16, 32
    linker = DNNInterLayerLinker(dim=C, expansion=2)
    f_spec = torch.randn(B, H, W, C)
    f_graph = torch.randn(B, H, W, C)
    out = linker(f_spec, f_graph)
    assert out.shape == (B, H, W, C)


def test_mesh_grid_graph_block():
    B, H, W, C = 2, 16, 16, 32
    gnn_block = MeshGridGraphBlock(hidden_dim=C)
    x = torch.randn(B, H, W, C)
    out = gnn_block(x)
    assert out.shape == (B, H, W, C)


def test_hybrid_operator_layer():
    B, C, H, W = 2, 32, 16, 16
    layer = HybridOperatorLayer(hidden_dim=C, modes1=8, modes2=8)
    x = torch.randn(B, C, H, W)
    out = layer(x)
    assert out.shape == (B, C, H, W)


def test_hybrid_cfd_net_forward_single_step():
    model = create_model(
        model_type="hybrid",
        in_channels=3,
        out_channels=3,
        hidden_dim=32,
        modes1=8,
        modes2=8,
        num_layers=2,
        num_heads=4,
    )
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 3, 32, 32)


def test_hybrid_cfd_net_scaling_presets():
    # Verify presets scale parameters predictably
    base = HybridCFDNet.from_preset("base", enforce_boundary=False)
    large = HybridCFDNet.from_preset("large", enforce_boundary=False)
    assert base.num_parameters > 500_000
    assert large.num_parameters > 6_000_000
    assert large.num_parameters > base.num_parameters * 8



def test_hybrid_cfd_net_forward_sequence():
    model = HybridCFDNet(
        in_channels=3,
        out_channels=3,
        hidden_dim=32,
        modes1=8,
        modes2=8,
        num_layers=2,
        num_heads=4,
    )
    # Context window of 3 steps
    x_seq = torch.randn(2, 3, 3, 32, 32)
    out = model(x_seq)
    assert out.shape == (2, 3, 32, 32)
