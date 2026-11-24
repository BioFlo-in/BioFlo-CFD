# BioFlo-NLSS: Comprehensive Research & Engineering Notes

**Project**: Nonlinear State-Space (NLSS) Learned Physics Engine  
**Target Application**: Stirred-Tank Bioreactor Digital Twins (`bioflo.in` & `workbench.bioflo.in`)  
**Standard**: Bounded Numerical Equivalence to OpenFOAM Incompressible Solvers  
**Date**: November 2026  

---

## Table of Contents
1. [Executive Summary & The Strategic Paradigm Shift](#1-executive-summary--the-strategic-paradigm-shift)
2. [The Scientific Audit & Methodological Integrity (E0–E6)](#2-the-scientific-audit--methodological-integrity-e0e6)
3. [10-Point Adaptive Data Acquisition Diagnostics](#3-10-point-adaptive-data-acquisition-diagnostics)
4. [Derived Bioprocess Engineering & Scale-Up Validation Layer](#4-derived-bioprocess-engineering--scale-up-validation-layer)
5. [OpenFOAM Solver Teardown & Hybrid Backbone Blueprint](#5-openfoam-solver-teardown--hybrid-backbone-blueprint)
6. [Implementation of the Learned Evolution Operator](#6-implementation-of-the-learned-evolution-operator)
7. [Current Repository Status & Validation Verification](#7-current-repository-status--validation-verification)
8. [Strategic Upstream Integration: BiRD (BioReactorDesign - NREL/DOE)](#8-strategic-upstream-integration-bird-bioreactordesign---nreldoe)
9. [Immediate Execution Roadmap](#9-immediate-execution-roadmap)

---

## 1. Executive Summary & The Strategic Paradigm Shift

### The Two Objectives
In computational fluid dynamics (CFD) and scientific machine learning (SciML), AI models are typically deployed in one of two paradigms:

```
+----------------------------------------------------------------------------------------------------+
|                                    Surrogate vs. Autonomous Operator                               |
+----------------------------------------------------------------------------------------------------+
| Objective A: Accelerating OpenFOAM (Surrogate Assistant)                                            |
|                                                                                                    |
|   X_t ----[ Neural Network ]----> X*_{t+1} ----[ OpenFOAM PIMPLE Solve ]----> X_{t+1}               |
|                                                                                                    |
|   - The neural network acts as a warm-start or guess.                                              |
|   - OpenFOAM executes the momentum equations, Poisson solves, and boundary projections.           |
|   - Errors are forgiven because numerical convergence cleans up predictions.                       |
+----------------------------------------------------------------------------------------------------+
| Objective B: Autonomous Learned Physical State-Transition Operator                                 |
|                                                                                                    |
|   X_{t+1} = F_theta(X_t, X_{t-1}, dt, mu)                                                          |
|                                                                                                    |
|   - The model replaces the numerical evolution engine entirely.                                    |
|   - Must autonomously conserve mass, momentum, and turbulent kinetic energy over long horizons.    |
|   - Standard: Bounded Numerical Equivalence to OpenFOAM without solver intervention.               |
+----------------------------------------------------------------------------------------------------+
```

### Why Naive Deep Learning Fails Objective B
Prior surrogate efforts (e.g., standard FNO, 3D UNet, dense MLPs) failed when unrolled autonomously because:
1. **Capacity $\neq$ Physical Constraint**: Increasing parameters or mode rank without physical structure leads to spectral leakage and high-frequency noise accumulation.
2. **Error Cascading in Autoregressive Rollouts**: A $1\%$ single-step error compounded over 100 timesteps leads to exponential divergence or unphysical numerical diffusion.
3. **Representation Geometry**: A single global linear subspace cannot efficiently span topologically distinct hydrodynamic regimes (e.g., transitional radial jet discharge vs fully developed turbulent circulation).

BioFlo-NLSS solves this through a **Hybridized Learned Evolution Engine**:
- Retains mathematical invariants as architectural priors (exact solenoidal projection $\nabla \cdot \mathbf{u} = 0$, wall boundary masks, local manifold basis).
- Replaces iterative bottlenecks (such as the Krylov Poisson solve) with $\mathcal{O}(N \log N)$ exact discrete spectral projections and parameter-conditioned residual state transitions.

---

## 2. The Scientific Audit & Methodological Integrity (E0–E6)

A rigorous audit of the controlled ablation study resolved a critical methodological distinction that previously masked surrogate generalization limits.

### The E4 Methodological Boundary: E4a vs. E4b
In reduced-order modeling, reporting projection error onto a basis as surrogate generalization is a catastrophic methodological error.

- **E4a: Optimal Local Subspace Representation Bound (Tier A)**:
  $$\mathbf{a}_{\text{opt}}(t, \boldsymbol{\mu}^*) = \boldsymbol{\Phi}_{\text{loc}}^T (\mathbf{u}_{\text{CFD}}(t, \boldsymbol{\mu}^*) - \bar{\mathbf{u}})$$
  $$\mathbf{u}_{\text{POD}}(\mathbf{x}, t) = \bar{\mathbf{u}}(\mathbf{x}) + \boldsymbol{\Phi}_{\text{loc}} \mathbf{a}_{\text{opt}}(t, \boldsymbol{\mu}^*)$$
  *Achieved **$79.45\%$ accuracy ($E_{L_2} = 20.55\%, R^2 = 0.9576$)** on unseen $450\text{ RPM}$ using only 20 modes.*
  *Significance*: Proves that the local subspace spanned by $\{300, 600\}\text{ RPM}$ contains sufficient spatial geometry to represent $450\text{ RPM}$ with $<12\%$ impeller ROI error and $<3\%$ $k_L a$ error.

- **E4b: Blind Neural Dynamic Generalization (Tier C)**:
  $$\hat{\mathbf{a}}(t, \boldsymbol{\mu}^*) = f_\theta(t, \boldsymbol{\mu}^*)$$
  $$\hat{\mathbf{u}}(\mathbf{x}, t) = \bar{\mathbf{u}}(\mathbf{x}) + \boldsymbol{\Phi}_{\text{loc}} \hat{\mathbf{a}}(t, \boldsymbol{\mu}^*)$$
  *Achieved **$54.50\%$ accuracy ($E_{L_2} = 45.50\%, R^2 = 0.7919$)** when blindly predicting unseen $450\text{ RPM}$.*
  *Significance*: Isolates the true scientific bottleneck. **Spatial representation is tractable ($79.45\%$), but parameter-to-trajectory dynamic evolution ($54.50\%$) is where the error resides.**

### Controlled Ablation Matrix (E0–E6)

| Exp ID | Description | Modes | Tier A Accuracy | Tier C Accuracy | Global $R^2$ | Impeller Zone Error | $k_L a$ Error | Vector Alignment ($\cos \theta$) | Key Scientific Finding |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **E0** | Baseline Global POD | 25 | 80.10% | 53.38% | 0.7816 | 45.82% | 29.80% | 0.8124 | Baseline generalization floor. |
| **E1** | Global POD (Double Rank) | 50 | 80.10% | 50.56% | 0.7543 | 47.90% | 31.40% | 0.7980 | Rank expansion degrades generalization. |
| **E2** | Global POD (Quad Rank) | 100 | 80.10% | 48.84% | 0.7369 | 49.23% | 32.43% | 0.7810 | High modes fit high-frequency CFD noise. |
| **E3** | Imbalanced Sampling (115 snaps) | 25 | 78.40% | **31.60%** | **0.5298** | **56.91%** | **41.20%** | **0.6840** | Dominant 300 RPM data collapsed the decoder. |
| **E4a** | Local POD Subspace Bound | 20 | **79.45%** | [Bound] | **0.9576** | **11.71%** | **2.95%** | **0.8931** | Optimal local manifold geometry with fewer modes. |
| **E4b** | Local POD Blind Neural | 20 | — | **54.50%** | **0.7919** | **45.34%** | **28.10%** | **0.8210** | Dynamic stepper bottleneck isolated. |
| **E5** | High-Viscosity Broth ($\nu=10^{-5}$) | 20 | **83.54%** | [Bound] | **0.9728** | **9.84%** | **2.40%** | **0.9120** | Viscous damping increases manifold compactness. |
| **E6** | Parameter-Aware Scaled POD | 25 | 76.50% | 46.60% | 0.7134 | 51.10% | 34.50% | 0.7650 | Naive velocity-scale normalization distorts POD. |

---

## 3. 10-Point Adaptive Data Acquisition Diagnostics

Before executing expensive CFD runs, a comprehensive diagnostic study was conducted across the operational space:

1. **Parameter Space Geometry & Coverage**:
   - Operational envelope: $[300, 600]\text{ RPM}$ ($\text{Re} \approx 6.7 \times 10^4 \text{ to } 1.34 \times 10^5$).
   - Current sampled points: $300, 450, 600\text{ RPM}$.
   - Max parameter gap: $150\text{ RPM}$ ($25\%$ normalized coverage density).

2. **Temporal Snapshot Imbalance**:
   - 300 RPM: 80 snapshots ($69.6\%$ of dataset).
   - 450 RPM: 24 snapshots ($20.9\%$).
   - 600 RPM: 11 snapshots ($9.6\%$).
   - Imbalance ratio: **$7.27\times$**, directly responsible for the E3 failure mode.

3. **Subspace Angle (Grassmannian Distance)**:
   - Principal subspace angle between $300\text{ RPM}$ and $600\text{ RPM}$: **$\theta_{300 \to 600} = 83.0^\circ$** (near orthogonal).
   - Subspace angle between $450\text{ RPM}$ and $600\text{ RPM}$: **$\theta_{450 \to 600} = 5.9^\circ$** (nearly parallel).
   - *Discovery*: $450\text{ RPM}$ and $600\text{ RPM}$ occupy the same hydrodynamic regime; the critical regime transition occurs entirely between $300\text{ RPM}$ and $450\text{ RPM}$.

4. **Regime Transition Gradient**:
   - Gradient $\frac{dD}{d(\text{RPM})}$ in $[300, 450]$: **$0.00826\text{ per RPM}$**.
   - Gradient $\frac{dD}{d(\text{RPM})}$ in $[450, 600]$: **$0.00168\text{ per RPM}$**.
   - Transition rate is **$4.9\times$ higher** in the lower corridor!

5. **Impeller Discharge Jet Topology**:
   - Normalized ROI displacement between $300$ and $450\text{ RPM}$: **$64.2\%$**.
   - Vector angular rotation in discharge stream: **$54.1^\circ$**.

6. **Targeted Adaptive CFD Run Priorities (Stage 1)**:
   - **425 RPM** (Priority Score: **221.3**): Captures the upper boundary of the transition corridor.
   - **400 RPM** (Priority Score: **177.1**): Mid-point non-linear inflection in vortex shedding.
   - **375 RPM** (Priority Score: **132.8**): Lower inception of turbulent jet detachment.

---

## 4. Derived Bioprocess Engineering & Scale-Up Validation Layer

To validate that the model reproduces real-world bioprocess physics rather than just superficial velocity fields, a specialized engineering validation suite was implemented ([`nlss/validation/bioprocess_metrics.py`](file:///home/ajit/Documents/git/NLSS/nlss/validation/bioprocess_metrics.py)):

### Governing Formulations
1. **Strain Rate Tensor & Shear Rate**:
   $$\mathbf{S} = \frac{1}{2}\left(\nabla \mathbf{u} + (\nabla \mathbf{u})^T\right), \quad \dot{\gamma} = \sqrt{2 \mathbf{S} : \mathbf{S}} = \sqrt{2 \sum_{i,j} S_{ij} S_{ij}}$$

2. **Viscous Energy Dissipation & Kolmogorov Microscale**:
   $$\epsilon(\mathbf{x}) = 2\nu (\mathbf{S} : \mathbf{S}) = \nu \dot{\gamma}^2$$
   $$\eta_K(\mathbf{x}) = \left(\frac{\nu^3}{\epsilon(\mathbf{x})}\right)^{1/4}$$
   *Reported metrics*: Mean $\eta_K$, median $\eta_K$, peak-shear minimum $\eta_{K, \min}$, and 10th-percentile $\eta_{K, 10\%}$.

3. **Specific Power Dissipation vs. Impeller Power**:
   $$(P/V)_{\text{diss}} = \frac{\rho \int_{\Omega} \epsilon(\mathbf{x})\, d\Omega}{V_{\text{liquid}}}, \quad P_{\text{imp}} = N_p \rho N^3 D^5$$

4. **Volumetric Mass Transfer Coefficient ($k_L a$)**:
   $$k_L a = C \cdot \left(\frac{P}{V}\right)^{0.45} \cdot v_s^{0.50}$$

5. **Turbulent Blend Time & Circulation Time**:
   $$\tau_{95} = \frac{5.4}{N N_p^{1/3}} \left(\frac{T}{D}\right)^2, \quad \tau_{\text{circ}} = \frac{V_{\text{liquid}}}{Q_{\text{imp}}}$$

6. **Cell Shear Exposure Ratios**:
   - Mammalian cell ($d = 15\,\mu\text{m}$): Ratio $= d_{\text{cell}} / \eta_{K, \min}$ (Safe when $< 1.0$).
   - Microcarrier ($d = 150\,\mu\text{m}$): Ratio $= d_{\text{carrier}} / \eta_{K, \min}$ (Shear damage risk when $> 1.0$).

---

## 5. OpenFOAM Solver Teardown & Hybrid Backbone Blueprint

An in-depth code audit of OpenFOAM `pimpleFoam` / `rhoPimpleFoam` produced the comprehensive architectural blueprint ([`docs/NLSS_SOLVER_TEARDOWN_BLUEPRINT.md`](file:///home/ajit/Documents/git/NLSS/docs/NLSS_SOLVER_TEARDOWN_BLUEPRINT.md)):

### Computational Bottlenecks Identified
- **The Pressure Poisson Solve accounts for 70% to 85% of solver runtime**:
  $$\nabla \cdot \left(\frac{1}{A} \nabla p\right) = \nabla \cdot \left(\frac{\mathbf{H}}{A}\right)$$
  Inverting this elliptic system requires 50–200 Krylov / GAMG sweeps per timestep.
- **The Momentum Equation ($\text{UEqn}$)** is diagonally dominant and computationally cheap (10–15% of runtime).
- **Face Flux Interpolation ($\phi_f = \mathbf{S}_f \cdot \mathbf{U}_f$)** and Rhie-Chow stabilization prevent checkerboard pressure decoupling.

### Solver Replacement Taxonomy

| CFD Operation | NLSS Architecture Decision | Implementation Mechanism |
| :--- | :--- | :--- |
| **Continuity $\nabla \cdot \mathbf{u} = 0$** | **Retain as Exact Prior** | `LerayProjection3D` discrete trigonometric projection ($\mathcal{O}(N \log N)$). |
| **Poisson Inversion** | **Replace with Neural Preconditioner** | Discrete spectral solver reducing Krylov iterations from 50+ to 1–2 sweeps. |
| **Convection-Diffusion Time Stepping** | **Replace with Learned State Stepper** | `AutoregressiveStateEvolutionOperator` with two-step memory $[\mathbf{a}_n, \mathbf{a}_{n-1}]$. |
| **Turbulence Eddy Viscosity $\nu_t$** | **Learn as Implicit Prior** | Absorbed directly into the learned latent evolution dynamics. |
| **Iterative PIMPLE Corrections** | **Eliminate Completely** | Direct single-pass state-transition $X_{n+1} = \mathcal{F}_\theta(X_n, X_{n-1}, \Delta t, \boldsymbol{\mu})$. |

---

## 6. Implementation of the Learned Evolution Operator

The learned evolution operator has been implemented in [`nlss/models/learned_evolution_operator.py`](file:///home/ajit/Documents/git/NLSS/nlss/models/learned_evolution_operator.py) and registered in [`nlss/models/__init__.py`](file:///home/ajit/Documents/git/NLSS/nlss/models/__init__.py).

### Core Components
1. **`LerayProjection3D`**:
   - Solves discrete divergence-free projection on 3D grids matching central-difference stencils:
     $$s_x = \sin(2\pi k_x / W), \quad s_y = \sin(2\pi k_y / H), \quad s_z = \sin(2\pi k_z / D)$$
     $$\hat{\mathbf{u}}_{\text{sol}} = \hat{\mathbf{u}} - \frac{\mathbf{s} (\mathbf{s} \cdot \hat{\mathbf{u}})}{s_x^2 + s_y^2 + s_z^2 + \epsilon}$$
   - **Verification**: Reduces grid discrete divergence from $77.72$ to $8.45 \times 10^{-6}$ (a factor of $9.2 \times 10^6$ reduction).

2. **`AutoregressiveStateEvolutionOperator`**:
   - Computes latent velocity: $\mathbf{v}_n = (\mathbf{a}_n - \mathbf{a}_{n-1}) / \Delta t$.
   - Modulates state evolution using FiLM layers conditioned on physical parameters $\boldsymbol{\mu} = [\text{RPM}, \nu]$ and timestep $\Delta t$.
   - Includes dissipative damping $(1 - \alpha \Delta t)$ to enforce Lyapunov boundedness over long rollout horizons.

3. **`LearnedNumericalEvolutionEngine`**:
   - Master operator coupling latent dynamics, POD decoding, 3D Leray incompressibility projection, and basis re-projection.
   - Provides autonomous multi-step `rollout()` tracking kinetic energy and divergence traces.

4. **`MultiStepRolloutLoss`**:
   - Enforces $K$-step forward unrolling:
     $$\mathcal{L} = \sum_{k=1}^K \gamma^k \left( \|\hat{\mathbf{a}}_{n+k} - \mathbf{a}_{n+k}\|_2^2 + \lambda_E \left| \|\hat{\mathbf{a}}_{n+k}\|_2^2 - \|\mathbf{a}_{n+k}\|_2^2 \right| + \lambda_v \|\Delta \hat{\mathbf{a}} - \Delta \mathbf{a}\|_2^2 \right)$$

---

## 7. Current Repository Status & Validation Verification

- **Repository Branch**: `main` (clean working tree).
- **Unit & Physics Test Suite**: **56 passed in 4.63s** via `pytest`:
  * `tests/test_audit_verification.py` (9 tests)
  * `tests/test_benchmark.py` (2 tests)
  * `tests/test_bioprocess_metrics.py` (8 tests)
  * `tests/test_bioreactor.py` (3 tests)
  * `tests/test_bird_mesh_adapter.py` (4 tests)
  * `tests/test_end_to_end.py` (1 test)
  * `tests/test_generalization.py` (3 tests)
  * `tests/test_hybrid.py` (6 tests)
  * `tests/test_io.py` (1 test)
  * `tests/test_learned_evolution.py` (5 tests)
  * `tests/test_models.py` (3 tests)
  * `tests/test_neural_pod_operator.py` (6 tests)
  * `tests/test_physics_validation.py` (5 tests)

---

## 8. Strategic Upstream Integration: BiRD (BioReactorDesign - NREL/DOE)

A technical audit of **BiRD (`NatLabRockies/BioReactorDesign`)**, authored by researchers at the National Laboratory of the Rockies (NLR) / NREL under U.S. DOE funding (Software Record SWR 24-35), identified profound synergies with NLSS. Full report preserved at [`docs/BIRD_BIOREACTOR_INVESTIGATION.md`](file:///home/ajit/Documents/git/NLSS/docs/BIRD_BIOREACTOR_INVESTIGATION.md).

### How BiRD Directly Resolves NLSS Architectural Limitations
1. **Resolution of `LIM-02` (Geometry Lock / Fixed Mesh Coordinate Dependency)**:
   - *Current Limitation*: NLSS POD basis operates on fixed Cartesian/cylindrical grids. Altering tank diameter $D$, liquid height $H$, or baffle layout requires manual CAD redesign.
   - *BiRD Solution*: BiRD provides parametric Python generators (`stirred_tank_mesh.py`, `block_cyl_mesh.py`, `stl_patch/`) that automate structured `blockMesh` topology generation.
   - *NLSS Implementation*: Created [`nlss/dataset/bird_mesh_adapter.py`](file:///home/ajit/Documents/git/NLSS/nlss/dataset/bird_mesh_adapter.py) with `StirredTankSpec` and dimensionless coordinate normalization:
     $$\xi = \frac{x}{D_t/2} \in [-1, 1], \quad \eta = \frac{y}{D_t/2} \in [-1, 1], \quad \zeta = \frac{z - Z_0}{H} \in [0, 1]$$
     enabling scale-invariant transfer across vessel volumes (from 2L to 20,000L).
2. **Resolution of `LIM-05` (Multiphase Moving Free-Surface & Gibbs Ringing)**:
   - *Current Limitation*: Sharp liquid-gas interface tracking (VOF) causes severe modal ringing ($\alpha < 0$ or $\alpha > 1$) under linear POD superposition.
   - *BiRD Solution*: BiRD uses **dispersed Eulerian-Eulerian volume fraction fields ($\alpha_g$)** coupled with Population Balance Modeling (PBM bubble number density distributions $d_{32}$). This continuous field formulation eliminates discontinuous surface shocks and is ideally suited for smooth neural operator learning.
3. **First-Principles Bioprocess Physics Upgrade (Phase 2 Implementations)**:
   - Ported and verified in [`nlss/validation/bioprocess_metrics.py`](file:///home/ajit/Documents/git/NLSS/nlss/validation/bioprocess_metrics.py):
     * **Higbie Penetration Mass Transfer (`compute_higbie_mass_transfer`)**:
       $$u_{\text{rel}} = \|\mathbf{u}_g - \mathbf{u}_l\|_2, \quad a = \frac{6\alpha_g}{d_{32}}, \quad k_L = \frac{2}{\sqrt{\pi}} \sqrt{D_L \frac{u_{\text{rel}}}{d_{32}}}, \quad k_L a = k_L \cdot a$$
     * **Tracer Homogenization Mixing Time (`compute_tracer_mixing_time_t95`)**:
       Evaluates spatial $\text{CoV}(t) = \sigma_c(t) / \bar{c}(t)$ and identifies $t_{95}$ when $\text{CoV}(t) \le 0.05$.
     * **Dimensionless Bioprocess Numbers (`compute_dimensionless_bioprocess_numbers`)**:
       Impeller $Re, Fr, We$, bubble $Re_b, We_b$, and transport numbers $Sc, Sh, Mo$.
4. **Multiphase Neural POD Operator Benchmark Results**:
   - Evaluated in [`scripts/train_multiphase_bird_operator.py`](file:///home/ajit/Documents/git/NLSS/scripts/train_multiphase_bird_operator.py) on continuous 4-channel fields $[u_x, u_y, u_z, \alpha_g]$:
     * **Tier C Velocity Relative $L_2$ Error**: **$18.77\%$ ($R^2 = 0.9646$)** on unobserved $450\text{ RPM}$.
     * **Vector Cosine Alignment**: **$0.9141$** ($>91\%$ directional parity).
     * **Tier C Gas Holdup ($\alpha_g$) Error**: **$18.44\%$**.
     * **Volume-Averaged $k_L a$ Accuracy**: CFD $= 167.81\text{ h}^{-1}$, Surrogate $= 183.70\text{ h}^{-1}$ (**$9.47\%$ error**).
     * **Mean Gas Holdup Parity**: CFD $= 3.41\%$, Surrogate $= 3.41\%$ (**$0.0\%$ discrepancy**).
     * **Full Report**: [`reports/multiphase_bird_operator_report.json`](file:///home/ajit/Documents/git/NLSS/reports/multiphase_bird_operator_report.json).

---

## 9. Immediate Execution Roadmap

```
+----------------------------------------------------------------------------------------------------+
|                                    Next Execution Milestones                                       |
+----------------------------------------------------------------------------------------------------+
|  Phase 1: Adaptive Data Generation (OpenFOAM)                                                      |
|  - Run 3 targeted transition conditions: 375 RPM, 400 RPM, 425 RPM in openfoam_cases/              |
|  - Ingest with equal-allocation temporal stratification (30 snapshots each)                       |
|  - Expand dataset to balanced multi-regime archive (total ~150 balanced snapshots)                 |
+----------------------------------------------------------------------------------------------------+
|  Phase 2: Level 2 Autoregressive Training                                                          |
|  - Train LearnedNumericalEvolutionEngine on 5-step unrolling windows                              |
|  - Benchmark Tier C blind generalization on held-out 450 RPM trajectory                           |
|  - Compare single-step vs multi-step rollout drift against pure OpenFOAM time series              |
+----------------------------------------------------------------------------------------------------+
|  Phase 3: Long-Horizon Autonomous Stability (100+ Timesteps)                                       |
|  - Verify kinetic energy stability over 100 autonomous unrolled steps                             |
|  - Confirm divergence remains bounded <= 10^-5 across entire rollout                               |
|  - Generate comparative visualization videos and export digital twin package                       |
+----------------------------------------------------------------------------------------------------+
```
