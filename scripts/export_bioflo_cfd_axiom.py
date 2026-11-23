#!/usr/bin/env python3
"""
Export BioFlo-CFD HybridCFDNet to a single browser-consumable binary artifact.

Why not ONNX/wasm: SpectralConv2d uses complex64 parameters and torch.fft.rfft2 /
irfft2, which do not survive ONNX export (confirmed 2026-10-27). The Axiom web
engine therefore re-implements the forward pass in TypeScript and loads this flat
float32 artifact.

Spectral weight handling: complex weights are folded into real-valued Fourier
multiplier tensors (`*_real` / `*_imag`) that preserve the exact circulant
(periodic convolution) structure of SpectralConv2d on the finite Z_H x Z_W grid.
Expanding them into dense H x W spatial kernels is mathematically equivalent
(verified below) but ~50x larger, so the compact real/imag form ships instead.
This script verifies both numpy routes against PyTorch before writing.

Outputs (default dir: outputs/axiom_export/):
  bioflo_cfd_axiom_v1.bin : magic "BFX1" + JSON header + float32 payload
  golden_trace.npz        : PyTorch reference I/O + per-module intermediates
                            for TypeScript engine regression tests

Usage:
    python scripts/export_bioflo_cfd_axiom.py [--weights PATH] [--out-dir DIR]
"""

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import nlss  # noqa: E402

MAGIC = b"BFX1"


# ----------------------------------------------------------------------------- #
# numpy reference routes for SpectralConv2d                                     #
# ----------------------------------------------------------------------------- #

def hermitian_complete(y_half: np.ndarray, h: int, w: int) -> np.ndarray:
    """Expand an rfft2-style half spectrum (..., H, W//2+1) to the full (..., H, W)
    spectrum with the conjugate symmetry irfft2 implicitly assumes for real signals."""
    wq = w // 2 + 1
    rows = np.concatenate([[0], np.arange(h - 1, 0, -1)])      # i -> (H - i) % H
    cols = np.arange(w - wq, 0, -1)                            # j -> W - j, descending
    full = np.empty(y_half.shape[:-1] + (w,), dtype=np.complex128)
    full[..., :wq] = y_half
    full[..., wq:] = np.conj(y_half[..., rows, :][..., cols])
    return full


def spectral_conv_mode_block(x, w1, w2, modes1, modes2):
    """Route A (shipped in the web engine): reproduce torch SpectralConv2d exactly -
    rfft2 bins, two masked mode blocks, Hermitian completion, inverse FFT."""
    b, c, h, w = x.shape
    m1 = min(modes1, h // 2)
    m2 = min(modes2, w // 2 + 1)
    x_ft = np.fft.rfft2(x)
    y_ft = np.zeros((b, c, h, w // 2 + 1), dtype=np.complex128)
    y_ft[:, :, :m1, :m2] = np.einsum("bixy,ioxy->boxy", x_ft[:, :, :m1, :m2], w1)
    y_ft[:, :, -m1:, :m2] = np.einsum("bixy,ioxy->boxy", x_ft[:, :, -m1:, :m2], w2)
    return np.fft.ifft2(hermitian_complete(y_ft, h, w)).real


def build_circulant_kernels(w1, w2, h, w, modes1, modes2):
    """Fold the band-limited complex weights into real-valued circulant convolution
    kernels k[i, o] = ifft2(M_full), one H x W kernel per (in, out) channel pair."""
    i_c, o_c, m1, m2 = w1.shape
    kernels = np.zeros((i_c, o_c, h, w), dtype=np.float64)
    for i in range(i_c):
        for o in range(o_c):
            m = np.zeros((h, w // 2 + 1), dtype=np.complex128)
            m[:m1, :m2] = w1[i, o]
            m[h - m1:, :m2] = w2[i, o]
            kernels[i, o] = np.fft.ifft2(hermitian_complete(m, h, w)).real
    return kernels


def spectral_conv_circulant(x, kernels):
    """Route B (verification only): plain circular convolution with the real kernels."""
    k_hat = np.fft.fft2(kernels, axes=(-2, -1))
    x_hat = np.fft.fft2(x, axes=(-2, -1))
    y_hat = np.einsum("bixy,ioxy->boxy", x_hat, k_hat)
    return np.fft.ifft2(y_hat, axes=(-2, -1)).real


# ----------------------------------------------------------------------------- #
# artifact writer                                                               #
# ----------------------------------------------------------------------------- #

def flatten_f32(arr: np.ndarray) -> bytes:
    return np.ascontiguousarray(arr, dtype=np.float32).tobytes(order="C")


def split_complex(sd):
    out = {}
    for k, v in sd.items():
        if torch.is_complex(v):
            out[f"{k}_real"] = v.real.detach().cpu().numpy()
            out[f"{k}_imag"] = v.imag.detach().cpu().numpy()
        else:
            out[k] = v.detach().cpu().numpy()
    return out


def write_artifact(path, header, tensors):
    payload = b""
    table = []
    for name, arr in tensors.items():
        blob = flatten_f32(arr)
        pad = (-len(payload)) % 4
        payload += b"\x00" * pad
        table.append({"name": name, "shape": list(arr.shape), "count": int(arr.size),
                      "offset": len(payload), "bytes": len(blob)})
        payload += blob
    header = dict(header)
    header["tensors"] = table
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", len(header_bytes)))
        f.write(header_bytes)
        f.write(payload)
    return len(header_bytes) + len(payload) + 8


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


# ----------------------------------------------------------------------------- #
# main                                                                          #
# ----------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default="pretrained/bioflo_cfd_hybrid_net.pt")
    ap.add_argument("--out-dir", default="outputs/axiom_export")
    ap.add_argument("--grid", type=int, default=64, help="golden-trace inference grid")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    out_dir = Path(args.out_dir) if Path(args.out_dir).is_absolute() else root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    weights_path = Path(args.weights) if Path(args.weights).is_absolute() else root / args.weights
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    model, normalizer = nlss.load_pretrained_model(device="cpu", weights_path=weights_path)
    preset = ckpt["preset"]
    cfg = model.PRESETS[preset]
    h = w = args.grid
    modes1, modes2 = cfg["modes1"], cfg["modes2"]
    print(f"Loaded HybridCFDNet preset '{preset}': hidden={cfg['hidden_dim']}, "
          f"modes=({modes1},{modes2}), layers={cfg['num_layers']}, heads={cfg['num_heads']}")

    # ---- verify both numpy routes equal PyTorch SpectralConv2d ----------------- #
    torch.manual_seed(args.seed)
    x = torch.randn(1, cfg["hidden_dim"], h, w, dtype=torch.float32)
    worst_a = worst_b = 0.0
    for layer_idx, layer in enumerate(model.hybrid_layers):
        spec = layer.spectral_conv
        ref = spec(x).detach().numpy()
        w1 = spec.weights1.detach().numpy().astype(np.complex128)
        w2 = spec.weights2.detach().numpy().astype(np.complex128)
        xn = x.detach().numpy().astype(np.float64)
        err_a = np.max(np.abs(spectral_conv_mode_block(xn, w1, w2, modes1, modes2) - ref))
        kernels = build_circulant_kernels(w1, w2, h, w, modes1, modes2)
        err_b = np.max(np.abs(spectral_conv_circulant(xn, kernels) - ref))
        print(f"  spectral_conv layer {layer_idx}: mode-block route max|d|={err_a:.3e}, "
              f"circulant-kernel route max|d|={err_b:.3e}")
        worst_a, worst_b = max(worst_a, err_a), max(worst_b, err_b)
    if worst_a > 1e-5 or worst_b > 1e-5:
        raise SystemExit(f"FAIL: numpy routes deviate from PyTorch ({worst_a:.2e}, {worst_b:.2e})")
    print("  numpy circulant / mode-block routes match PyTorch within float tolerance")

    # ---- golden trace ---------------------------------------------------------- #
    sim = nlss.NavierStokes2DSimulator(nx=h, ny=w, nu=0.01, dt=0.005)
    for _ in range(20):
        sim.step(is_cavity=True, u_lid=1.0)
    s_t = np.stack([sim.u, sim.v, sim.p], axis=0)
    sim.step(is_cavity=True, u_lid=1.0)
    s_next_gt = np.stack([sim.u, sim.v, sim.p], axis=0)

    x_phys = torch.from_numpy(s_t).float().unsqueeze(0)
    x_norm = normalizer.normalize(x_phys)
    golden = {"input_phys": s_t, "input_norm": x_norm.numpy().squeeze(0),
              "gt_next_phys": s_next_gt}

    trace_names = [
        "lifting",
        "hybrid_layers.0.spectral_conv", "hybrid_layers.0.dense_conv",
        "hybrid_layers.0.mesh_gnn", "hybrid_layers.0.dnn_linker",
        "hybrid_layers.0.norm",
        "hybrid_layers.1.spectral_conv", "hybrid_layers.1.dense_conv",
        "hybrid_layers.1.mesh_gnn", "hybrid_layers.1.dnn_linker",
        "hybrid_layers.1.norm",
        "transformer_head",
    ]
    handles = []
    for name in trace_names:
        module = model
        for part in name.split("."):
            module = module[int(part)] if part.isdigit() else getattr(module, part)
        def hook(_m, _inp, out, name=name):
            golden[f"trace.{name}"] = out.detach().numpy().squeeze(0)
        handles.append(module.register_forward_hook(hook))
    with torch.no_grad():
        y_norm = model(x_norm)
    for hd in handles:
        hd.remove()
    y_phys = normalizer.denormalize(y_norm)
    golden["output_norm"] = y_norm.numpy().squeeze(0)
    golden["output_phys"] = y_phys.numpy().squeeze(0)

    golden_path = out_dir / "golden_trace.npz"
    np.savez(golden_path, **golden)
    print(f"  golden trace -> {golden_path} ({len(golden)} arrays)")

    # ---- single binary artifact ------------------------------------------------- #
    tensors = split_complex(model.state_dict())
    tensors["__normalizer_mean"] = normalizer.mean.detach().cpu().numpy()
    tensors["__normalizer_std"] = normalizer.std.detach().cpu().numpy()

    header = {
        "format": "bioflo-cfd-axiom/1",
        "model": "HybridCFDNet",
        "preset": preset,
        "in_channels": 3,
        "out_channels": 3,
        "hidden_dim": cfg["hidden_dim"],
        "modes1": modes1,
        "modes2": modes2,
        "num_layers": cfg["num_layers"],
        "num_heads": cfg["num_heads"],
        "patch_size": 4,
        "groupnorm_groups": 8,
        "reference_grid": [h, w],
        "enforce_boundary": False,
        "temporal_path": "single-step (4D input only; CausalTemporalSelfAttention unused)",
        "spectral_layout": "weights1/2 split to *_real/*_imag float32, shape (in,out,modes1,modes2); "
                           "apply on rfft2 bins [0:m1] and [H-m1:H] x [0:m2], Hermitian-complete, inverse FFT",
        "normalizer": {"mean": normalizer.mean.detach().cpu().numpy().flatten().tolist(),
                       "std": normalizer.std.detach().cpu().numpy().flatten().tolist()},
        "num_parameters": int(model.num_parameters),
        "checkpoint_sha256": sha256(weights_path),
    }
    bin_path = out_dir / "bioflo_cfd_axiom_v1.bin"
    size = write_artifact(bin_path, header, tensors)
    print(f"  artifact -> {bin_path} ({size / 1e6:.2f} MB, {len(tensors)} float32 tensors)")
    print(f"  sha256: {sha256(bin_path)}")


if __name__ == "__main__":
    main()
