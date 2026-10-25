# BioFlo-CFD: AI Surrogate & OpenFOAM Hybrid Accelerator
### Physics-Informed Neural Operators & Bioprocess Digital Twin

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![OpenFOAM](https://img.shields.io/badge/OpenFOAM-2512-brightgreen.svg)](https://openfoam.com/)
[![Tests](https://img.shields.io/badge/Tests-14%20Passing-brightgreen.svg)](tests/)

**BioFlo-CFD** is an industrial-grade deep learning surrogate and acceleration framework for **OpenFOAM** and Computational Fluid Dynamics (CFD). It trains multi-scale neural operators (`HybridCFDNet` combining Spectral Fourier Layers, MeshGraphNet flux message-passing, and causal attention heads) across 150+ OpenFOAM simulation regimes, cutting simulation iteration times by orders of magnitude while preserving machine-precision physical conservation.

---

## 🚀 Key Capabilities

1. **Working Pre-Trained Model Out-of-the-Box**:
   - Ships with an official pre-trained `HybridCFDNet` surrogate checkpoint (`pretrained/bioflo_cfd_hybrid_net.pt`), ready for instant evaluation and inference without waiting to train from scratch.
   - Run predictions in **under 3 seconds** with a single command: `python examples/01_quickstart_inference.py`.

2. **Unified Multi-Scale Hybrid Operator (`HybridCFDNet`)**:
   - **Fourier Spectral Convolutions**: Captures global elliptical pressure fields and acoustic propagation.
   - **Mesh GraphNet Message-Passing**: Resolves localized cell-face flux balances and viscous boundary layers.
   - **DNN Inter-Layer Linker**: Cross-modality highway dynamically fusing spectral and graph representations with adaptive gating ($\alpha$).
   - **Spatiotemporal Transformer Head**: Causal attention over temporal histories for unsteady vortex shedding.

3. **Stirred-Tank Bioreactor (STR) Digital Twin**:
   - Direct integration of bioprocess cell viability diagnostics:
     - Hydrodynamic shear stress $\tau(x, y)$ (safe thresholds for mammalian CHO / stem cells).
     - Turbulent kinetic energy dissipation rate $\epsilon(x, y)$ (W/kg).
     - Kolmogorov microscale eddy sizing $\eta_K$ ($\mu\text{m}$).

4. **Closed-Loop OpenFOAM Hybrid Warm-Starter**:
   - AI predicts transient flow states $\hat{S}_{t+\Delta t}^{AI}$ as initial guesses directly injected into OpenFOAM's PISO/SIMPLE linear solvers, cutting linear solver iterations by 5x–15x while maintaining strict physical boundary and mass conservation.

5. **154-Regime Multi-Source Ingestion Engine**:
   - Ingests, normalizes, and packages upstream OpenFOAM tutorials, community benchmarks (Wolf Dynamics, Holzmann, Phresher, UnnamedMoose, NimaSam), velocity sweeps ($0.05\text{--}25.0\text{ m/s}$), and surface boundary condition variations with $SO(2)$ symmetry augmentations.

---

## ⚡ 1-Minute Quickstart

### 1. Installation

```bash
git clone https://github.com/BioFlo-in/BioFlo-CFD.git
cd BioFlo-CFD

# Create & activate environment
python3 -m venv .venv
source .venv/bin/activate

# Install in editable mode
pip install -e .
```

### 2. Instant Pre-trained Inference Demo

Run inference using the pre-trained surrogate model:

```bash
python examples/01_quickstart_inference.py
```

Outputs side-by-side velocity magnitude heatmaps, streamlines, and absolute error fields to `outputs/quickstart/quickstart_prediction.png` in ~30 ms!

---

## 🧠 Parameter Scaling Tiers

`HybridCFDNet` supports modular scaling presets via `from_preset()`:

| Tier | `hidden_dim` | Fourier Modes (`k1, k2`) | Layers | Heads | Parameters | Ideal Use Case |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`base`** | 32 | $(12, 12)$ | 2 | 4 | **705,509** | Real-time edge inference & laptops |
| **`medium`** | 48 | $(14, 14)$ | 3 | 6 | **3,019,878** | Balanced transient rollouts |
| **`large`** | 64 | $(16, 16)$ | 3 | 8 | **6,841,478** | High-fidelity multi-regime generalist |
| **`xlarge`** | 96 | $(16, 16)$ | 4 | 8 | **20,312,551** | Complex 3D turbulence & high-$Re$ |
| **`xxlarge`**| 128 | $(20, 20)$ | 4 | 8 | **54,981,255** | Foundation CFD model across industrial geometries |

```python
import nlss

# Instant instantiation with scaling preset:
model = nlss.HybridCFDNet.from_preset("large", use_checkpointing=True)
print(f"Trainable Parameters: {model.num_parameters:,}")  # 6,841,478
```

---

## 🧬 Stirred-Tank Bioreactor Digital Twin

BioFlo-CFD models 2D transient flow in industrial Rushton stirred tanks with active impeller forcing, central shafts, wall baffles, and top free surfaces:

```bash
python examples/03_bioreactor_digital_twin.py
```

### Key Bioprocess Benchmark Results

| Metric | OpenFOAM CFD Ground Truth | BioFlo-CFD AI Surrogate | Biological Assessment |
| :--- | :---: | :---: | :--- |
| **Peak Shear Stress ($\tau_{\max}$)** | $0.05\text{ Pa}$ | $0.07\text{ Pa}$ | **Safe** (well below $5.0\text{ Pa}$ CHO lysis limit) |
| **Min. Kolmogorov Microscale ($\eta_K$)** | $119.8\text{ }\mu\text{m}$ | $119.8\text{ }\mu\text{m}$ | **Safe** ($> 7\times$ mammalian cell diameter $15\text{ }\mu\text{m}$) |
| **Inference Latency** | $72.30\text{ ms/step}$ | **$4.36\text{ ms/step}$** | **~16.6x Speedup** |

---

## 📂 Project Architecture

```
BioFlo-CFD/
├── nlss/                              # Core Python package
│   ├── models/
│   │   ├── hybrid_cfd_net.py          # HybridCFDNet (Spectral + GNN + Transformer)
│   │   ├── fno.py                     # Fourier Neural Operator (2D)
│   │   ├── unet.py                    # Residual CFD U-Net
│   │   └── gnn.py                     # MeshGraphNet for unstructured meshes
│   ├── dataset/
│   │   ├── bioreactor.py              # Rushton STR simulator & shear stress metrics
│   │   ├── foam_dataset.py            # OpenFOAM dataset reader & normalizer
│   │   └── generator.py               # Chorin 2D Navier-Stokes solver
│   ├── physics/
│   │   └── losses.py                  # Continuity div(u)=0 & vorticity losses
│   ├── hybrid/
│   │   └── warm_starter.py            # OpenFOAM PISO/SIMPLE accelerator
│   └── io/
│       ├── foam_reader.py             # OpenFOAM field parser
│       └── foam_writer.py             # Native OpenFOAM field exporter
├── pretrained/
│   └── bioflo_cfd_hybrid_net.pt       # Pre-trained working model weights (5.0 MB)
├── examples/
│   ├── 01_quickstart_inference.py     # 1-minute inference demo
│   ├── 02_train_surrogate.py          # Scaling preset training pipeline
│   ├── 03_bioreactor_digital_twin.py  # Industrial bioreactor testing & metrics
│   ├── 04_openfoam_hybrid_warmstart.py# Hybrid solver warmstart
│   └── 05_build_openfoam_dataset.py   # Multi-source dataset builder
├── openfoam_cases/                    # Benchmark OpenFOAM cases
├── scripts/                           # Production batch solvers and suite generators
└── tests/                             # Pytest test suite (14 passing tests)
```

---

## 🐍 Python API Example

```python
import nlss

# 1. Load official pre-trained surrogate model
model, normalizer = nlss.load_pretrained_model(device="cuda")

# 2. Ingest existing OpenFOAM case directory
dataset, _ = nlss.load_dataset_from_openfoam_case(
    case_dir="openfoam_cases/lidDrivenCavity",
    grid_shape=(64, 64)
)

# 3. Create Rollout Engine
engine = nlss.RolloutEngine(model=model, normalizer=normalizer, device="cuda")

# 4. Multi-step autoregressive rollout
current_state = dataset.data[0] # (channels=3, H=64, W=64)
pred_trajectory, elapsed_sec = engine.rollout(current_state, num_steps=50)

# 5. Export back into native OpenFOAM time folders
times = [i * 0.005 for i in range(51)]
engine.export_rollout_to_openfoam(pred_trajectory, times, "outputs/openfoam_predictions")
```

---

## 🧪 Testing

Run the full automated test suite:

```bash
pytest tests/ -v
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
