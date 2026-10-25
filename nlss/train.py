"""
Training engine for CFD surrogate models (FNO-2D, ResUNet, MeshGNN).
Includes physics-informed loss supervision, multi-step unrolling, and checkpointing.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset.foam_dataset import CFDTrajectoryDataset, FieldNormalizer
from .models.fno import FNO2d
from .models.unet import ResUNetCFD
from .physics.losses import PhysicsInformedCFDLoss


def create_model(
    model_type: str = "fno",
    in_channels: int = 3,
    out_channels: int = 3,
    **kwargs,
) -> nn.Module:
    """Factory function for CFD surrogate model architectures."""
    model_type = model_type.lower()
    if model_type == "fno":
        modes1 = kwargs.get("modes1", 16)
        modes2 = kwargs.get("modes2", 16)
        width = kwargs.get("width", 48)
        num_layers = kwargs.get("num_layers", 4)
        return FNO2d(
            in_channels=in_channels,
            out_channels=out_channels,
            modes1=modes1,
            modes2=modes2,
            width=width,
            num_layers=num_layers,
        )
    elif model_type == "unet":
        base_features = kwargs.get("base_features", 32)
        return ResUNetCFD(
            in_channels=in_channels,
            out_channels=out_channels,
            base_features=base_features,
        )
    elif model_type in {"dense", "dense_cfd_net", "dense_fno"}:
        hidden_dim = kwargs.get("hidden_dim", 96)
        modes1 = kwargs.get("modes1", 20)
        modes2 = kwargs.get("modes2", 20)
        num_blocks = kwargs.get("num_blocks", 4)
        enforce_boundary = kwargs.get("enforce_boundary", True)
        lid_velocity = kwargs.get("lid_velocity", 1.0)
        from .models.dense_cfd_net import DenseCFDNet
        return DenseCFDNet(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_dim=hidden_dim,
            modes1=modes1,
            modes2=modes2,
            num_blocks=num_blocks,
            enforce_boundary=enforce_boundary,
            lid_velocity=lid_velocity,
        )
    elif model_type in {"rk4", "neural_rk4", "neuralrk4"}:
        hidden_dim = kwargs.get("hidden_dim", 96)
        modes = kwargs.get("modes", 24)
        from .models.neural_rk4 import NeuralRK4Operator
        return NeuralRK4Operator(
            channels=in_channels,
            hidden_dim=hidden_dim,
            modes=modes,
            enforce_leray=kwargs.get("enforce_leray", True),
        )
    elif model_type in {"hybrid", "hybrid_cfd_net", "hybrid_net", "graph_operator_transformer"}:
        hidden_dim = kwargs.get("hidden_dim", 48)
        modes1 = kwargs.get("modes1", 12)
        modes2 = kwargs.get("modes2", 12)
        num_layers = kwargs.get("num_layers", 2)
        num_heads = kwargs.get("num_heads", 4)
        from .models.hybrid_cfd_net import HybridCFDNet
        return HybridCFDNet(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_dim=hidden_dim,
            modes1=modes1,
            modes2=modes2,
            num_layers=num_layers,
            num_heads=num_heads,
            enforce_boundary=kwargs.get("enforce_boundary", True),
            lid_velocity=kwargs.get("lid_velocity", 1.0),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Supported: 'fno', 'unet', 'dense', 'neural_rk4', 'hybrid'")


def train_surrogate_model(
    model: nn.Module,
    train_dataset: CFDTrajectoryDataset,
    val_dataset: Optional[CFDTrajectoryDataset] = None,
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: Optional[torch.device] = None,
    checkpoint_dir: Optional[Union[str, Path]] = None,
    lambda_div: float = 0.05,
    lambda_vort: float = 0.02,
    log_interval: int = 10,
) -> Dict[str, list]:
    """
    Trains surrogate model to predict next simulation state S_{t+dt} from S_t.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = (
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        if val_dataset is not None
        else None
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = PhysicsInformedCFDLoss(lambda_div=lambda_div, lambda_vort=lambda_vort)

    history: Dict[str, list] = {
        "epoch": [],
        "train_loss": [],
        "train_rel_l2": [],
        "val_loss": [],
        "val_rel_l2": [],
    }

    best_val_loss = float("inf")
    ckpt_path = Path(checkpoint_dir) if checkpoint_dir else None
    if ckpt_path:
        ckpt_path.mkdir(parents=True, exist_ok=True)

    print(f"\n🚀 Training {model.__class__.__name__} on device: {device}")
    print(f"Total training pairs: {len(train_dataset)}, Epochs: {epochs}, Batch size: {batch_size}\n")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        epoch_rel_l2 = []

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            if y_batch.ndim == 5:
                # Multi-step unrolled pushforward training
                K = y_batch.shape[1]
                curr_x = x_batch
                total_loss = 0.0
                accum_rel_l2 = 0.0
                for k in range(K):
                    pred_k = model(curr_x)
                    target_k = y_batch[:, k]
                    step_loss, step_metrics = criterion(pred_k, target_k, input_state=curr_x)
                    total_loss = total_loss + (0.8 ** k) * step_loss
                    accum_rel_l2 += step_metrics["loss_rel_l2"]
                    curr_x = pred_k
                loss = total_loss / K
                metrics = {"loss_total": float(loss.item()), "loss_rel_l2": accum_rel_l2 / K}
            else:
                pred = model(x_batch)
                loss, metrics = criterion(pred, y_batch, input_state=x_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses.append(metrics["loss_total"])
            epoch_rel_l2.append(metrics["loss_rel_l2"])

        scheduler.step()

        mean_train_loss = float(np.mean(epoch_losses))
        mean_train_rel = float(np.mean(epoch_rel_l2))

        history["epoch"].append(epoch)
        history["train_loss"].append(mean_train_loss)
        history["train_rel_l2"].append(mean_train_rel)

        # Validation
        val_loss_val = None
        val_rel_val = None
        if val_loader:
            model.eval()
            val_losses = []
            val_rels = []
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx = vx.to(device)
                    vy = vy.to(device)
                    if vy.ndim == 5:
                        K = vy.shape[1]
                        curr_vx = vx
                        tot_vloss = 0.0
                        tot_vrel = 0.0
                        for k in range(K):
                            vpred_k = model(curr_vx)
                            vtarget_k = vy[:, k]
                            s_loss, s_metrics = criterion(vpred_k, vtarget_k, input_state=curr_vx)
                            tot_vloss = tot_vloss + (0.8 ** k) * s_loss
                            tot_vrel += s_metrics["loss_rel_l2"]
                            curr_vx = vpred_k
                        val_losses.append(float((tot_vloss / K).item()))
                        val_rels.append(tot_vrel / K)
                    else:
                        vpred = model(vx)
                        vloss, vmetrics = criterion(vpred, vy, input_state=vx)
                        val_losses.append(vmetrics["loss_total"])
                        val_rels.append(vmetrics["loss_rel_l2"])

            val_loss_val = float(np.mean(val_losses))
            val_rel_val = float(np.mean(val_rels))
            history["val_loss"].append(val_loss_val)
            history["val_rel_l2"].append(val_rel_val)

            # Checkpoint best model
            if val_loss_val < best_val_loss and ckpt_path:
                best_val_loss = val_loss_val
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "normalizer": train_dataset.normalizer.to_dict(),
                        "history": history,
                    },
                    ckpt_path / "best_model.pt",
                )

        if epoch % log_interval == 0 or epoch == 1 or epoch == epochs:
            val_str = f" | Val Loss: {val_loss_val:.4e} (RelL2: {val_rel_val:.4e})" if val_loss_val else ""
            print(
                f"Epoch [{epoch:03d}/{epochs:03d}] "
                f"Train Loss: {mean_train_loss:.4e} (RelL2: {mean_train_rel:.4e})"
                f"{val_str} | LR: {scheduler.get_last_lr()[0]:.2e}"
            )

    # Save final model
    if ckpt_path:
        torch.save(
            {
                "epoch": epochs,
                "model_state_dict": model.state_dict(),
                "normalizer": train_dataset.normalizer.to_dict(),
                "history": history,
            },
            ckpt_path / "final_model.pt",
        )
        print(f"\n✅ Checkpoint saved to: {ckpt_path / 'final_model.pt'}")

    return history
