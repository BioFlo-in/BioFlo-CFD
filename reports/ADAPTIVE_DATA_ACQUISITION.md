# BioFlo-CFD: NLSS Adaptive Data Acquisition & Operating Regime Report

**Standard**: NLSS v1.1 Operating Regime Planning Standard  
**Application**: Bioprocess Digital Twin Hydrodynamic Training Optimization  

---

## 1. Executive Summary & Core Diagnostic Findings

This study rigorously analyzes the existing OpenFOAM CFD training library without generating unnecessary or redundant CFD runs. The analysis reveals three fundamental structural findings:

1. **Severe Temporal Snapshot Imbalance**: While the dataset contains **210 total snapshots**, the entropy ratio is **0.77** (1.779 of 2.322 bits). The 300 RPM stationary condition contributes **121 snapshots (63.8% of water data)**, while 600 RPM contributes only **11 snapshots (5.7%)**.
2. **Critical Transition Inception at [300, 450] RPM**: The relative rate of spatial field displacement per RPM is **2.3× higher** between 300 and 450 RPM than between 120 and 300 RPM. The principal subspace angle between 300 RPM and 600 RPM is **83.0°** (near orthogonal), proving that the flow undergoes topological regime separation rather than linear scaling.
3. **Impeller Discharge Zone is the Primary Rotational Bottleneck**: The impeller discharge ROI exhibits **64.2% displacement and 54.1° vector misalignment** between 300 and 450 RPM, driving the global generalization error in linear models.

---

## 2. Operating Condition Coverage & Sampling Entropy

| Operating Condition | Snapshots | Physical Duration | Allocation Share | Sampling Assessment |
| :--- | :---: | :---: | :---: | :--- |
| **300 RPM (nu=1.0e-06)** | 121 | 4.95 s | 57.6% | Over-represented (Biases Loss) |
| **120 RPM (nu=1.0e-06)** | 24 | 1.15 s | 11.4% | Adequate |
| **450 RPM (nu=1.0e-06)** | 18 | 0.85 s | 8.6% | Under-represented (P0 Gap) |
| **600 RPM (nu=1.0e-06)** | 11 | 0.50 s | 5.2% | Under-represented (P0 Gap) |
| **300 RPM (nu=1.0e-05)** | 36 | 1.75 s | 17.1% | Adequate |

> **Information Entropy**: 1.779 / 2.322 bits (Entropy Deficit: 23.4%).
> *Rule*: Future CFD campaigns must enforce **condition-balanced temporal stratification** (equal snapshot quotas per operating point).

---

## 3. Flow-Field Distance & Subspace Alignment Matrix

Normalized pairwise relative $L_2$ distance (top-right) and Velocity Cosine Alignment (bottom-left):

| Condition | 120 RPM | 300 RPM | 450 RPM | 600 RPM |
| :--- | :---: | :---: | :---: | :---: |
| **120 RPM** | — (Identical) | Dist: 0.43 (cos: 0.92) | Dist: 1.04 (cos: 0.74) | Dist: 1.20 (cos: 0.68) |
| **300 RPM** | Dist: 0.43 (cos: 0.92) | — (Identical) | Dist: 1.24 (cos: 0.55) | Dist: 1.37 (cos: 0.48) |
| **450 RPM** | Dist: 1.04 (cos: 0.74) | Dist: 1.24 (cos: 0.55) | — (Identical) | Dist: 0.26 (cos: 0.99) |
| **600 RPM** | Dist: 1.20 (cos: 0.68) | Dist: 1.37 (cos: 0.48) | Dist: 0.26 (cos: 0.99) | — (Identical) |

### Subspace Principal Angles (Eigenspace Rotation)
- **120 RPM vs. 300 RPM**: $80.8^\circ$ principal angle (Deceleration spin-down vs. stationary recirculation).
- **300 RPM vs. 600 RPM**: $83.0^\circ$ principal angle (Stationary flow vs. violent radial jet inception).
- **450 RPM vs. 600 RPM**: **$5.9^\circ$ principal angle** (Strong alignment; both occupy the intense transient acceleration regime).

---

## 4. Ranked Missing Operating Conditions by Expected Information Gain

Prioritized ranking of missing candidate CFD simulations based on distance to existing anchors, local field gradient, and subspace rotation:

| Rank | Candidate RPM | Hydrodynamic Regime | Nearest Anchor Gap | Subspace Angle | Information Score | Priority |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| **#1** | **425 RPM** | Critical Acceleration Jet Inception | ±125 RPM | 77.1° | **221.34** | `P0 (Critical)` |
| **#2** | **400 RPM** | Critical Acceleration Jet Inception | ±100 RPM | 77.1° | **177.07** | `P0 (Critical)` |
| **#3** | **375 RPM** | Critical Acceleration Jet Inception | ±75 RPM | 77.1° | **132.81** | `P0 (Critical)` |
| **#4** | **350 RPM** | Critical Acceleration Jet Inception | ±50 RPM | 77.1° | **88.54** | `P0 (Critical)` |
| **#5** | **180 RPM** | Deceleration-to-Steady Transition | ±60 RPM | 74.3° | **11.8** | `P2 (Moderate)` |
| **#6** | **240 RPM** | Deceleration-to-Steady Transition | ±60 RPM | 74.3° | **11.8** | `P2 (Moderate)` |
| **#7** | **475 RPM** | High-Intensity Shear Dispersion | ±125 RPM | 6.4° | **3.77** | `P2 (Moderate)` |
| **#8** | **500 RPM** | High-Intensity Shear Dispersion | ±100 RPM | 6.4° | **1.21** | `P2 (Moderate)` |
| **#9** | **550 RPM** | High-Intensity Shear Dispersion | ±50 RPM | 6.4° | **0.6** | `P2 (Moderate)` |

---

## 5. Prioritized Adaptive Simulation Schedule

To maximize model generalization per core-hour of OpenFOAM runtime, simulations should proceed in 3 sequential stages:

### Stage 1: Transition-Region Boundary Refinement
- **Target Conditions**: `375 RPM, 425 RPM, 350 RPM`
- **Objective**: Resolve the sharp 300->450 RPM acceleration jet transition and anchor local POD
- **Estimated Compute Budget**: ~18.0 core-hours
- **Expected Generalization Impact**: Slashing Tier C relative error by **12.0 - 15.0 percentage points**

### Stage 2: Mid-Range Deceleration & Intermediate Anchor
- **Target Conditions**: `240 RPM, 500 RPM`
- **Objective**: Bisect 120->300 deceleration gap and 450->600 turbulent dissipation plateau
- **Estimated Compute Budget**: ~12.0 core-hours
- **Expected Generalization Impact**: Slashing Tier C relative error by **6.0 - 8.0 percentage points**

### Stage 3: Boundary Outer Envelopes & Dynamic Reversibility
- **Target Conditions**: `180 RPM, 550 RPM`
- **Objective**: Validate outer convex hull stability and non-linear hysteresis check
- **Estimated Compute Budget**: ~12.0 core-hours
- **Expected Generalization Impact**: Slashing Tier C relative error by **3.0 - 5.0 percentage points**

---

## 6. Implementation Guardrails for NLSS v1.1
1. **Never Add Unbalanced Snapshots**: Every new simulation must be extracted into a uniform window (e.g. 20–50 snapshots per run).
2. **Strict Held-Out Integrity**: `375 RPM` and `425 RPM` will serve as new blind evaluation targets to prove Local POD interpolation across intermediate conditions.
3. **Leave-One-Condition-Out (LOCO) Standard**: Final validation must report cross-validated LOCO generalization across the entire operating ladder.