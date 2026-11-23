#!/usr/bin/env python3
"""
Multi-Domain 3D CFD Foundation Surrogate Training Pipeline (BioFlo-CFD):
Unifies disparate fluid geometries into a single 3D Neural Operator:
  - Domain 1: 3D Rushton-Turbine CSTR Bioreactor (Multiphase Swirling Flow)
  - Domain 2: 3D Cylindrical Pipe / Channel Flow (Axial Shear Flow)
Conditioned via Continuous Euclidean Signed Distance Functions (SDF) & Fluid Domain Masks.

Inputs (8 channels):  [ux, uy, uz, p, alpha, SDF, fluid_mask, domain_id]
Outputs (5 channels): [ux, uy, uz, p, alpha] at t + dt
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nlss.models.hybrid_cfd_net_3d import HybridCFDNet3D, compute_divergence_3d


def compute_sdf_from_mask(mask: np.ndarray, dx: float = 1.0) -> np.ndarray:
    """
    Computes exact Euclidean Signed Distance Function (SDF).
    Negative inside fluid, zero at wall boundary, positive in solid/exterior.
    """
    dist_out = ndi.distance_transform_edt(~mask, sampling=dx)
    dist_in = ndi.distance_transform_edt(mask, sampling=dx)
    return (dist_out - dist_in).astype(np.float32)


def compute_vorticity_3d(vel: torch.Tensor, dx: float = 1.0) -> torch.Tensor:
    """
    vel: (B, 3, D, H, W) [ux, uy, uz]
    omega = curl(u) = [duz/dy - duy/dz, dux/dz - duz/dx, duy/dx - dux/dy]
    """
    ux, uy, uz = vel[:, 0:1], vel[:, 1:2], vel[:, 2:3]
    duz_dy = (torch.roll(uz, -1, dims=3) - torch.roll(uz, 1, dims=3)) / (2 * dx)
    duy_dz = (torch.roll(uy, -1, dims=2) - torch.roll(uy, 1, dims=2)) / (2 * dx)
    omega_x = duz_dy - duy_dz

    dux_dz = (torch.roll(ux, -1, dims=2) - torch.roll(ux, 1, dims=2)) / (2 * dx)
    duz_dx = (torch.roll(uz, -1, dims=4) - torch.roll(uz, 1, dims=4)) / (2 * dx)
    omega_y = dux_dz - duz_dx

    duy_dx = (torch.roll(uy, -1, dims=4) - torch.roll(uy, 1, dims=4)) / (2 * dx)
    dux_dy = (torch.roll(ux, -1, dims=3) - torch.roll(ux, 1, dims=3)) / (2 * dx)
    omega_z = duy_dx - dux_dy

    return torch.cat([omega_x, omega_y, omega_z], dim=1)


class MultiDomainCFDDataset(Dataset):
    """
    Dataset serving (X, Y, mask, domain_id) pairs across multiple geometric domains.
    X: (8, D, H, W) -> [ux, uy, uz, p, alpha, SDF, fluid_mask, domain_id]
    Y: (5, D, H, W) -> [ux, uy, uz, p, alpha]
    """

    def __init__(self, x_tensor: torch.Tensor, y_tensor: torch.Tensor, domain_ids: torch.Tensor, masks: torch.Tensor):
        self.x = x_tensor
        self.y = y_tensor
        self.domain_ids = domain_ids
        self.masks = masks

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx], self.masks[idx], self.domain_ids[idx]


def prepare_multi_domain_data(grid_dim: int = 32) -> Tuple[MultiDomainCFDDataset, MultiDomainCFDDataset, Dict]:
    print("=" * 80)
    print("🌐 Assembling Multi-Domain 3D CFD Datasets (Massive CSTR Suite + Pipe Flow)...")
    print("=" * 80)

    # 1. Domain 1: Massive 3D CSTR Multiphase Suite (210 snapshots across 6 regimes)
    cstr_path = "data/cstr_massive_multiphase_dataset.pt"
    if not os.path.exists(cstr_path):
        cstr_path = "data/cstr_3d_real_openfoam_dataset.pt"

    cstr_data = torch.load(cstr_path, map_location="cpu", weights_only=False)

    # CSTR geometry: cylinder R <= 0.25
    lin_cstr = np.linspace(-0.262, 0.262, grid_dim)
    Z_c, Y_c, X_c = np.meshgrid(np.linspace(-0.099, 1.299, grid_dim), lin_cstr, lin_cstr, indexing="ij")
    r_cstr = np.sqrt(X_c**2 + Y_c**2)
    mask_cstr = (r_cstr <= 0.25)
    dx_cstr = 0.524 / (grid_dim - 1)
    sdf_cstr = compute_sdf_from_mask(mask_cstr, dx=dx_cstr)

    sdf_cstr_t = torch.from_numpy(sdf_cstr).float().unsqueeze(0)  # (1, D, H, W)
    mask_cstr_t = torch.from_numpy(mask_cstr.astype(np.float32)).unsqueeze(0)
    id_cstr_t = torch.full((1, grid_dim, grid_dim, grid_dim), fill_value=1.0, dtype=torch.float32)

    x_cstr_tr_list, y_cstr_tr_list = [], []
    x_cstr_va_list, y_cstr_va_list = [], []

    if "cases" in cstr_data:
        for case in cstr_data["cases"]:
            states = case["states"]  # (T, 5, D, H, W)
            T = states.shape[0]
            if T < 2:
                continue
            x_pairs = states[:-1]
            y_pairs = states[1:]
            n_pairs = x_pairs.shape[0]

            geom = torch.cat([
                sdf_cstr_t.repeat(n_pairs, 1, 1, 1, 1),
                mask_cstr_t.repeat(n_pairs, 1, 1, 1, 1),
                id_cstr_t.repeat(n_pairs, 1, 1, 1, 1),
            ], dim=1)
            x_8ch = torch.cat([x_pairs, geom], dim=1)

            # 80/20 train/val split per case
            n_tr = max(1, int(0.8 * n_pairs))
            x_cstr_tr_list.append(x_8ch[:n_tr])
            y_cstr_tr_list.append(y_pairs[:n_tr])
            x_cstr_va_list.append(x_8ch[n_tr:])
            y_cstr_va_list.append(y_pairs[n_tr:])
            print(f"   ✓ {case['name']:<30}: {n_tr} train, {n_pairs - n_tr} val pairs")
    else:
        x_all = cstr_data["input_pairs"]
        y_all = cstr_data["target_pairs"]
        n_pairs = x_all.shape[0]
        geom = torch.cat([
            sdf_cstr_t.repeat(n_pairs, 1, 1, 1, 1),
            mask_cstr_t.repeat(n_pairs, 1, 1, 1, 1),
            id_cstr_t.repeat(n_pairs, 1, 1, 1, 1),
        ], dim=1)
        x_8ch = torch.cat([x_all, geom], dim=1)
        n_tr = int(0.8 * n_pairs)
        x_cstr_tr_list.append(x_8ch[:n_tr])
        y_cstr_tr_list.append(y_all[:n_tr])
        x_cstr_va_list.append(x_8ch[n_tr:])
        y_cstr_va_list.append(y_all[n_tr:])

    x_cstr_train = torch.cat(x_cstr_tr_list, dim=0)
    y_cstr_train = torch.cat(y_cstr_tr_list, dim=0)
    x_cstr_val = torch.cat(x_cstr_va_list, dim=0)
    y_cstr_val = torch.cat(y_cstr_va_list, dim=0)
    print(f"Total CSTR Dataset: {x_cstr_train.shape[0]} train pairs, {x_cstr_val.shape[0]} val pairs")

    # 2. Domain 2: 3D Pipe Flow & Analytical Benchmarks
    pipe_data = torch.load("data/cfd_3d_dataset.pt", map_location="cpu", weights_only=False)
    x_pipe_tr = pipe_data["x_train"]  # (106, 4, 32, 32, 32)
    y_pipe_tr = pipe_data["y_train"]  # (106, 4, 32, 32, 32)
    x_pipe_va = pipe_data["x_val"]    # (19, 4, 32, 32, 32)
    y_pipe_va = pipe_data["y_val"]    # (19, 4, 32, 32, 32)

    # Pipe geometry: cylinder R <= 0.45 in X, Y
    lin_pipe = np.linspace(-0.5, 0.5, grid_dim)
    Z_p, Y_p, X_p = np.meshgrid(np.linspace(0.0, 1.0, grid_dim), lin_pipe, lin_pipe, indexing="ij")
    r_pipe = np.sqrt(X_p**2 + Y_p**2)
    mask_pipe = (r_pipe <= 0.45)
    dx_pipe = 1.0 / (grid_dim - 1)
    sdf_pipe = compute_sdf_from_mask(mask_pipe, dx=dx_pipe)

    sdf_pipe_t = torch.from_numpy(sdf_pipe).float().unsqueeze(0)
    mask_pipe_t = torch.from_numpy(mask_pipe.astype(np.float32)).unsqueeze(0)
    id_pipe_t = torch.full((1, grid_dim, grid_dim, grid_dim), fill_value=2.0, dtype=torch.float32)

    # Add single-phase alpha = 1.0 for pipe flow
    def make_pipe_8ch(x_4ch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        N = x_4ch.shape[0]
        alpha = mask_pipe_t.repeat(N, 1, 1, 1, 1)  # 1.0 in fluid, 0.0 outside
        geom = torch.cat([
            sdf_pipe_t.repeat(N, 1, 1, 1, 1),
            mask_pipe_t.repeat(N, 1, 1, 1, 1),
            id_pipe_t.repeat(N, 1, 1, 1, 1),
        ], dim=1)
        x_8ch = torch.cat([x_4ch, alpha, geom], dim=1)
        return x_8ch, alpha

    x_pipe_tr_8ch, alpha_pipe_tr = make_pipe_8ch(x_pipe_tr)
    y_pipe_tr_5ch = torch.cat([y_pipe_tr, alpha_pipe_tr], dim=1)

    x_pipe_va_8ch, alpha_pipe_va = make_pipe_8ch(x_pipe_va)
    y_pipe_va_5ch = torch.cat([y_pipe_va, alpha_pipe_va], dim=1)

    # 3. Concatenate into Unified Multi-Domain Sets
    x_train = torch.cat([x_cstr_train, x_pipe_tr_8ch], dim=0)
    y_train = torch.cat([y_cstr_train, y_pipe_tr_5ch], dim=0)
    masks_train = torch.cat([
        mask_cstr_t.repeat(x_cstr_train.shape[0], 1, 1, 1, 1),
        mask_pipe_t.repeat(x_pipe_tr.shape[0], 1, 1, 1, 1),
    ], dim=0)
    domain_ids_train = torch.cat([
        torch.full((x_cstr_train.shape[0],), 1.0),
        torch.full((x_pipe_tr.shape[0],), 2.0),
    ], dim=0)

    x_val = torch.cat([x_cstr_val, x_pipe_va_8ch], dim=0)
    y_val = torch.cat([y_cstr_val, y_pipe_va_5ch], dim=0)
    masks_val = torch.cat([
        mask_cstr_t.repeat(x_cstr_val.shape[0], 1, 1, 1, 1),
        mask_pipe_t.repeat(x_pipe_va.shape[0], 1, 1, 1, 1),
    ], dim=0)
    domain_ids_val = torch.cat([
        torch.full((x_cstr_val.shape[0],), 1.0),
        torch.full((x_pipe_va.shape[0],), 2.0),
    ], dim=0)

    # Shuffle training set
    perm = torch.randperm(x_train.shape[0])
    x_train, y_train = x_train[perm], y_train[perm]
    masks_train, domain_ids_train = masks_train[perm], domain_ids_train[perm]

    train_ds = MultiDomainCFDDataset(x_train, y_train, domain_ids_train, masks_train)
    val_ds = MultiDomainCFDDataset(x_val, y_val, domain_ids_val, masks_val)

    meta = {
        "cstr_val_slice": slice(0, x_cstr_val.shape[0]),
        "pipe_val_slice": slice(x_cstr_val.shape[0], x_val.shape[0]),
        "x_cstr_val": x_cstr_val,
        "y_cstr_val": y_cstr_val,
        "x_pipe_val": x_pipe_va_8ch,
        "y_pipe_val": y_pipe_va_5ch,
        "mask_cstr": mask_cstr_t,
        "mask_pipe": mask_pipe_t,
        "sdf_cstr": sdf_cstr_t,
        "sdf_pipe": sdf_pipe_t,
    }

    print(f"✓ Training Samples:   {len(train_ds)} (CSTR: {x_cstr_train.shape[0]}, Pipe: {x_pipe_tr.shape[0]})")
    print(f"✓ Validation Samples: {len(val_ds)} (CSTR: {x_cstr_val.shape[0]}, Pipe: {x_pipe_va.shape[0]})")
    print(f"✓ Input Shape:        {x_train[0].shape} (8 channels)")
    print(f"✓ Target Shape:       {y_train[0].shape} (5 channels)")
    return train_ds, val_ds, meta


class MultiDomainCFDLoss(nn.Module):
    """
    Physics-Informed Loss for Multi-Domain 3D Neural Operator:
      - Domain-Masked Relative L2 for Velocity & Pressure
      - Multiphase Interface Loss (BCE + L1)
      - 3D Vorticity / Enstrophy Preservation
      - 3D Incompressibility Constraint: ||div(u)||_1
      - Solid Wall No-Slip Penalty: ||u * (1 - mask)||^2
    """

    def __init__(
        self,
        w_vel: float = 1.0,
        w_p: float = 0.5,
        w_alpha: float = 0.8,
        w_vort: float = 0.1,
        w_div: float = 0.05,
        w_wall: float = 0.5,
    ):
        super().__init__()
        self.w_vel = w_vel
        self.w_p = w_p
        self.w_alpha = w_alpha
        self.w_vort = w_vort
        self.w_div = w_div
        self.w_wall = w_wall

    def forward(
        self,
        pred: torch.Tensor,
        targ: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # pred, targ: (B, 5, D, H, W) [ux, uy, uz, p, alpha]
        # mask: (B, 1, D, H, W)
        pred_vel, targ_vel = pred[:, :3], targ[:, :3]
        pred_p, targ_p = pred[:, 3:4], targ[:, 3:4]
        pred_alpha, targ_alpha = pred[:, 4:5], targ[:, 4:5]

        # 1. Masked Velocity Relative L2
        vel_diff = (pred_vel - targ_vel) * mask
        vel_norm = torch.norm(targ_vel * mask, p=2) + 1e-5
        loss_vel = torch.norm(vel_diff, p=2) / vel_norm

        # 2. Masked Pressure Relative L2
        p_diff = (pred_p - targ_p) * mask
        p_norm = torch.norm(targ_p * mask, p=2) + 1e-5
        loss_p = torch.norm(p_diff, p=2) / p_norm

        # 3. Multiphase Alpha Loss (BCE + L1)
        clamped_pred_a = torch.clamp(pred_alpha, 1e-4, 1.0 - 1e-4)
        clamped_targ_a = torch.clamp(targ_alpha, 0.0, 1.0)
        bce_a = F.binary_cross_entropy(clamped_pred_a * mask, clamped_targ_a * mask)
        l1_a = F.l1_loss(pred_alpha * mask, clamped_targ_a * mask)
        loss_alpha = bce_a + l1_a

        # 4. 3D Vorticity Preservation
        pred_vort = compute_vorticity_3d(pred_vel)
        targ_vort = compute_vorticity_3d(targ_vel)
        vort_norm = torch.norm(targ_vort * mask, p=2) + 1e-5
        loss_vort = torch.norm((pred_vort - targ_vort) * mask, p=2) / vort_norm

        # 5. 3D Incompressibility Constraint
        div_pred = compute_divergence_3d(pred_vel)
        loss_div = torch.mean(torch.abs(div_pred * mask))

        # 6. Wall No-Slip Penalty (outside fluid domain)
        loss_wall = torch.mean((pred_vel * (1.0 - mask)) ** 2)

        total_loss = (
            self.w_vel * loss_vel
            + self.w_p * loss_p
            + self.w_alpha * loss_alpha
            + self.w_vort * loss_vort
            + self.w_div * loss_div
            + self.w_wall * loss_wall
        )

        metrics = {
            "loss_total": float(total_loss.item()),
            "loss_vel": float(loss_vel.item()),
            "loss_p": float(loss_p.item()),
            "loss_alpha": float(loss_alpha.item()),
            "loss_vort": float(loss_vort.item()),
            "loss_div": float(loss_div.item()),
            "loss_wall": float(loss_wall.item()),
        }
        return total_loss, metrics


def train_multi_domain_foundation(
    epochs: int = 35,
    batch_size: int = 2,
    lr: float = 1e-3,
    hidden_dim: int = 32,
    modes: int = 8,
    num_layers: int = 3,
    save_path: str = "models/multi_domain_foundation_net_3d_best.pt",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 80)
    print(f"🚀 Training Multi-Domain 3D Foundation Neural Operator on: {device}")
    print(f"   Architecture: HybridCFDNet3D (in=8, out=5, hidden={hidden_dim}, modes={modes}, layers={num_layers})")
    print(f"   Epochs: {epochs} | Batch Size: {batch_size} (Grad Accum: 2) | LR: {lr}")
    print("=" * 80)

    train_ds, val_ds, meta = prepare_multi_domain_data()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = HybridCFDNet3D(
        in_channels=8,
        out_channels=5,
        hidden_dim=hidden_dim,
        modes=modes,
        num_layers=num_layers,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Trainable Parameters: {total_params:,}")

    criterion = MultiDomainCFDLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_loss = float("inf")
    os.makedirs(Path(save_path).parent, exist_ok=True)

    history = {
        "train_loss": [], "val_loss": [],
        "cstr_vel_err": [], "pipe_vel_err": [],
        "cstr_r": [], "pipe_r": [],
    }

    t0_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        # -----------------------------
        # Training Phase
        # -----------------------------
        model.train()
        train_loss_accum = 0.0
        train_batches = 0
        optimizer.zero_grad()

        for step, (x_b, y_b, mask_b, id_b) in enumerate(train_loader):
            x_b = x_b.to(device)
            y_b = y_b.to(device)
            mask_b = mask_b.to(device)

            pred_b = model(x_b)
            loss, _ = criterion(pred_b, y_b, mask_b)
            (loss / 2.0).backward()  # gradient accumulation = 2

            if (step + 1) % 2 == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            train_loss_accum += loss.item()
            train_batches += 1

        scheduler.step()
        avg_train_loss = train_loss_accum / max(1, train_batches)

        # -----------------------------
        # Validation Phase & Domain Stratification
        # -----------------------------
        model.eval()
        val_loss_accum = 0.0
        val_batches = 0

        # Domain-specific error accumulators
        cstr_u_diff, cstr_u_norm = 0.0, 0.0
        pipe_u_diff, pipe_u_norm = 0.0, 0.0

        cstr_preds, cstr_targs = [], []
        pipe_preds, pipe_targs = [], []

        with torch.no_grad():
            for x_b, y_b, mask_b, id_b in val_loader:
                x_b, y_b, mask_b = x_b.to(device), y_b.to(device), mask_b.to(device)
                pred_b = model(x_b)
                loss, _ = criterion(pred_b, y_b, mask_b)
                val_loss_accum += loss.item()
                val_batches += 1

                for i in range(x_b.shape[0]):
                    d_id = int(id_b[i].item())
                    m = mask_b[i]
                    p_vel = pred_b[i, :3]
                    t_vel = y_b[i, :3]

                    diff = torch.norm((p_vel - t_vel) * m).item()
                    norm = torch.norm(t_vel * m).item() + 1e-5

                    if d_id == 1:  # CSTR
                        cstr_u_diff += diff
                        cstr_u_norm += norm
                        cstr_preds.append((p_vel * m).cpu().numpy().flatten())
                        cstr_targs.append((t_vel * m).cpu().numpy().flatten())
                    else:  # Pipe
                        pipe_u_diff += diff
                        pipe_u_norm += norm
                        pipe_preds.append((p_vel * m).cpu().numpy().flatten())
                        pipe_targs.append((t_vel * m).cpu().numpy().flatten())

        avg_val_loss = val_loss_accum / max(1, val_batches)
        cstr_rel_err = (cstr_u_diff / max(1e-5, cstr_u_norm)) * 100.0
        pipe_rel_err = (pipe_u_diff / max(1e-5, pipe_u_norm)) * 100.0

        # Pearson correlations
        r_cstr = 0.0
        if cstr_preds:
            p_flat = np.concatenate(cstr_preds)
            t_flat = np.concatenate(cstr_targs)
            r_cstr = float(np.corrcoef(p_flat, t_flat)[0, 1])

        r_pipe = 0.0
        if pipe_preds:
            p_flat = np.concatenate(pipe_preds)
            t_flat = np.concatenate(pipe_targs)
            r_pipe = float(np.corrcoef(p_flat, t_flat)[0, 1])

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["cstr_vel_err"].append(cstr_rel_err)
        history["pipe_vel_err"].append(pipe_rel_err)
        history["cstr_r"].append(r_cstr)
        history["pipe_r"].append(r_pipe)

        # Checkpointing
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "cstr_rel_err": cstr_rel_err,
                "pipe_rel_err": pipe_rel_err,
                "r_cstr": r_cstr,
                "r_pipe": r_pipe,
                "config": {
                    "in_channels": 8, "out_channels": 5,
                    "hidden_dim": hidden_dim, "modes": modes, "num_layers": num_layers,
                },
            }, save_path)
            mark = "⭐ (Best)"
        else:
            mark = ""

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs or is_best:
            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] "
                f"Loss: {avg_train_loss:.4f} / Val: {avg_val_loss:.4f} | "
                f"CSTR Err: {cstr_rel_err:5.2f}% (r={r_cstr:.4f}) | "
                f"Pipe Err: {pipe_rel_err:5.2f}% (r={r_pipe:.4f}) {mark}"
            )

    total_time = time.perf_counter() - t0_start
    print("=" * 80)
    print(f"✅ Foundation Training Finished in {total_time:.2f}s ({total_time/epochs:.2f}s/epoch)")
    print(f"   Best Val Loss: {best_val_loss:.4f}")
    print(f"   Saved Checkpoint: {save_path}")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Cross-Domain Rollout & Comprehensive Figure Generation
    # -------------------------------------------------------------------------
    generate_foundation_evaluation_figure(model, meta, device)


def generate_foundation_evaluation_figure(model: nn.Module, meta: Dict, device: torch.device):
    print("\n📊 Generating Multi-Domain Foundation Evaluation Figure...")
    model.eval()

    x_cstr_val = meta["x_cstr_val"].to(device)
    y_cstr_val = meta["y_cstr_val"].to(device)
    mask_cstr = meta["mask_cstr"].to(device)

    x_pipe_val = meta["x_pipe_val"].to(device)
    y_pipe_val = meta["y_pipe_val"].to(device)
    mask_pipe = meta["mask_pipe"].to(device)

    with torch.no_grad():
        pred_cstr = model(x_cstr_val[:1])
        pred_pipe = model(x_pipe_val[:1])

    # Convert slices to numpy for visualization
    # 1. CSTR Velocity Magnitude at Impeller plane (z = 10 / 32)
    z_imp = 10
    gt_u_cstr = torch.norm(y_cstr_val[0, :3, z_imp], dim=0).cpu().numpy()
    pr_u_cstr = (torch.norm(pred_cstr[0, :3, z_imp], dim=0) * mask_cstr[0, z_imp]).cpu().numpy()
    err_u_cstr = np.abs(pr_u_cstr - gt_u_cstr)

    # 2. CSTR Multiphase Alpha at Axial plane (x = 16 / 32)
    x_mid = 16
    gt_a_cstr = y_cstr_val[0, 4, :, :, x_mid].cpu().numpy()
    pr_a_cstr = (pred_cstr[0, 4, :, :, x_mid] * mask_cstr[0, :, :, x_mid]).cpu().numpy()
    err_a_cstr = np.abs(pr_a_cstr - gt_a_cstr)

    # 3. Pipe Velocity Profile (axial Uz) along centerline (x = 16)
    gt_u_pipe = y_pipe_val[0, 2, :, :, x_mid].cpu().numpy()
    pr_u_pipe = (pred_pipe[0, 2, :, :, x_mid] * mask_pipe[0, :, :, x_mid]).cpu().numpy()
    err_u_pipe = np.abs(pr_u_pipe - gt_u_pipe)

    # 4. Pipe Pressure Drop along duct
    gt_p_pipe = y_pipe_val[0, 3, :, :, x_mid].cpu().numpy()
    pr_p_pipe = (pred_pipe[0, 3, :, :, x_mid] * mask_pipe[0, :, :, x_mid]).cpu().numpy()
    err_p_pipe = np.abs(pr_p_pipe - gt_p_pipe)

    # Setup 4-row x 3-column figure
    fig, axes = plt.subplots(4, 3, figsize=(15, 16))
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    cases = [
        ("Domain 1 (CSTR Bioreactor): Impeller Swirl |U| (XY Plane)", gt_u_cstr, pr_u_cstr, err_u_cstr, "viridis", "m/s"),
        ("Domain 1 (CSTR Bioreactor): Multiphase Alpha (XZ Plane)", gt_a_cstr, pr_a_cstr, err_a_cstr, "coolwarm", "-"),
        ("Domain 2 (Pipe Flow): Axial Velocity Uz (YZ Plane)", gt_u_pipe, pr_u_pipe, err_u_pipe, "plasma", "m/s"),
        ("Domain 2 (Pipe Flow): Pressure Distribution P (YZ Plane)", gt_p_pipe, pr_p_pipe, err_p_pipe, "cividis", "Pa"),
    ]

    for row_idx, (title, gt, pr, err, cmap, unit) in enumerate(cases):
        vmin = min(gt.min(), pr.min())
        vmax = max(gt.max(), pr.max())

        im0 = axes[row_idx, 0].imshow(gt, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
        axes[row_idx, 0].set_title(f"OpenFOAM Truth\n{title.split(':')[1].strip()}", fontsize=11, fontweight="bold")
        fig.colorbar(im0, ax=axes[row_idx, 0], fraction=0.046, pad=0.04, label=unit)

        im1 = axes[row_idx, 1].imshow(pr, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
        axes[row_idx, 1].set_title(f"Foundation Surrogate (Ours)\n{title.split(':')[0]}", fontsize=11, fontweight="bold", color="darkblue")
        fig.colorbar(im1, ax=axes[row_idx, 1], fraction=0.046, pad=0.04, label=unit)

        im2 = axes[row_idx, 2].imshow(err, cmap="hot", origin="lower")
        max_err = err.max()
        axes[row_idx, 2].set_title(f"Absolute Residual Error\n(Max: {max_err:.4f} {unit})", fontsize=11, fontweight="bold", color="darkred")
        fig.colorbar(im2, ax=axes[row_idx, 2], fraction=0.046, pad=0.04, label=unit)

    out_fig = "outputs/figures/multi_domain_foundation_surrogate_evaluation.png"
    plt.suptitle("BioFlo 3D CFD Foundation Model: Cross-Geometry Multi-Domain Verification", fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Multi-Domain Figure saved to: {out_fig}")


if __name__ == "__main__":
    train_multi_domain_foundation()
