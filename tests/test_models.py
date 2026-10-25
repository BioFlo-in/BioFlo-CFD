import pytest
import torch

from nlss.models.fno import FNO2d
from nlss.models.unet import ResUNetCFD
from nlss.models.gnn import MeshGNN


def test_fno2d_forward():
    # Batch=2, Channels=3 (ux, uy, p), H=32, W=32
    x = torch.randn(2, 3, 32, 32)
    model = FNO2d(in_channels=3, out_channels=3, modes1=8, modes2=8, width=32, num_layers=3)
    out = model(x)
    assert out.shape == (2, 3, 32, 32)

    # Test backward pass
    loss = out.sum()
    loss.backward()
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None


def test_res_unet_forward():
    x = torch.randn(2, 3, 32, 32)
    model = ResUNetCFD(in_channels=3, out_channels=3, base_features=16)
    out = model(x)
    assert out.shape == (2, 3, 32, 32)

    loss = out.sum()
    loss.backward()
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None


def test_mesh_gnn_forward():
    # Unstructured graph with 50 cells and 120 edges
    n_cells = 50
    node_feat = torch.randn(n_cells, 3)
    # Random edge index
    src = torch.randint(0, n_cells, (120,))
    dst = torch.randint(0, n_cells, (120,))
    edge_index = torch.stack([src, dst], dim=0)

    model = MeshGNN(in_channels=3, out_channels=3, edge_dim=3, hidden_dim=32, num_layers=2)
    out = model(node_feat, edge_index)
    assert out.shape == (n_cells, 3)

    loss = out.sum()
    loss.backward()
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None


if __name__ == "__main__":
    test_fno2d_forward()
    test_res_unet_forward()
    test_mesh_gnn_forward()
    print("All model tests passed!")
