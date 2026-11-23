# NLSS Solver Teardown Blueprint: From OpenFOAM FVM to a Learned Numerical Evolution Engine

**Document ID**: `NLSS-ENG-BLUEPRINT-01`  
**Target Application**: Stirred-Tank Bioreactor Hydrodynamic Simulation & Scale-Up Digital Twins  
**Reference CFD Solver**: OpenFOAM v2406 `pimpleFoam` / `interFoam` / `rhoPimpleFoam`  
**Status**: Architectural Specification & Implementation Roadmap for NLSS Level 2/3  

---

## Executive Overview: The Two Objectives

In computational fluid dynamics (CFD) acceleration and neural operator design, two fundamentally different objectives exist:

```
Objective A: Surrogate Acceleration Assistant
    X_t ──► [ AI Model ] ──► X*_{t+1} ──► [ OpenFOAM PIMPLE Corrections ] ──► X_{t+1}
    (Feasible, but remains tethered to OpenFOAM computational overhead)

Objective B: Autonomous Learned Physical Evolution Engine
    [ X_t, X_{t-1}, Δt, μ ] ──► [ NLSS Neural Numerical Operator F_θ ] ──► X_{t+1}
    (Autonomous, eliminates CPU solver bottlenecks, enforces discrete conservation)
```

To achieve **Objective B** with numerical equivalence to OpenFOAM, NLSS must not be a naive black-box feed-forward regressor. It must be built around the **underlying mathematical and topological structure of the Finite Volume Method (FVM)**.

This blueprint dissects every component of OpenFOAM's transient solver into six parts:
1. **The Finite-Volume Framework** (Mesh, metrics, cell-vs-face fields, discrete operators).
2. **Momentum Equation Assembly** (Temporal inertia, linearized convection, turbulent diffusion, MRF sources, the $\mathbf{H}$ vector).
3. **Pressure–Velocity Coupling & The Poisson Projection** (Rhie–Chow interpolation, elliptic projection, velocity correction).
4. **Linear Solvers & Sparse Matrix Inversion** (LDU format, GAMG multigrid, Krylov methods, preconditioning).
5. **Temporal Integration & Turbulence Closures** (Courant limits, $k$-$\epsilon$/SST transport, eddy viscosity, wall functions).
6. **The PIMPLE Iterative Loop & Unified Neural Blueprint** (Outer/inner loops, synthesis, classification table).

---

## Part 1: The Finite-Volume Framework Teardown

OpenFOAM discretizes continuous partial differential equations over arbitrary convex polyhedral cells.

```
       Cell P (Owner)                       Cell N (Neighbour)
   ┌───────────────────────┐            ┌───────────────────────┐
   │                       │            │                       │
   │       • x_P           │     S_f    │       • x_N           │
   │      (Volume V_P)     │────► ────► │      (Volume V_N)     │
   │                       │     Face f │                       │
   └───────────────────────┴────────────┴───────────────────────┘
                           ◄────────────►
                                 d
```

### 1.1 Mesh Representation & Data Structures (`polyMesh`)
- **Points** ($\mathbf{x}_i \in \mathbb{R}^3$): Spatial vertices defining domain boundaries.
- **Faces** ($f$): Ordered planar point lists with Face Area Vector $\mathbf{S}_f = \mathbf{n}_f A_f$ directed from Owner cell $P$ to Neighbour cell $N$.
- **Cells** ($P$): Polyhedra with cell volume calculated via Gauss's theorem:
  $$V_P = \frac{1}{3} \sum_{f \in P} (\mathbf{x}_f - \mathbf{x}_P) \cdot \mathbf{S}_{f, P}$$
- **Owner–Neighbour Addressing**: Faces stored once in contiguous LDU ordering (`owner[f] < neighbour[f]`).

### 1.2 Cell-Centered vs. Face-Centered Fields
- **`volScalarField` / `volVectorField`**: Stored at cell centroids $\mathbf{x}_P$ (velocity $\mathbf{u}_P$, pressure $p_P$, dissipation $\epsilon_P$).
- **`surfaceScalarField`**: Stored at face centroids $\mathbf{x}_f$ (volumetric face flux $\phi_f = \mathbf{u}_f \cdot \mathbf{S}_f$).

> **Core Principle**: In OpenFOAM, mass conservation is enforced strictly on face fluxes:
> $$\sum_{f \in P} \phi_f = 0 \quad \forall P$$
> A neural network predicting only cell-centered $\mathbf{u}_P$ fails discrete continuity. NLSS must predict or constrain face fluxes $\phi_f$.

### 1.3 Discrete Differential Operators (Gauss Theorem)
1. **Gradient ($\nabla \phi$)**:
   $$[\nabla \phi]_P = \frac{1}{V_P} \sum_{f \in P} \mathbf{S}_f \phi_f$$
2. **Divergence of a Convected Flux ($\nabla \cdot (\phi \mathbf{u})$)**:
   $$[\nabla \cdot (\phi \mathbf{u})]_P = \frac{1}{V_P} \sum_{f \in P} \phi_f \mathbf{u}_f$$
3. **Laplacian Diffusion ($\nabla \cdot (\Gamma \nabla \phi)$)**:
   $$[\nabla \cdot (\Gamma \nabla \phi)]_P = \frac{1}{V_P} \sum_{f \in P} \Gamma_f (\nabla \phi)_f \cdot \mathbf{S}_f \approx \frac{1}{V_P} \sum_{f \in P} \Gamma_f \frac{\phi_N - \phi_P}{\|\mathbf{d}\|} \|\mathbf{S}_f\| + \text{explicit non-orthogonal correction}$$

### 1.4 Interpolation & Limiters
- **Linear (Central)**: $\phi_f = f_x \phi_P + (1 - f_x) \phi_N$ (2nd-order, oscillatory at $\text{Pe} > 2$).
- **Upwind**: $\phi_f = \phi_P$ if $\phi > 0$ else $\phi_N$ (1st-order, unconditionally bounded, highly diffusive).
- **TVD / NVD Limiters (`vanLeer`, `linearUpwind`)**: $\phi_f = \phi_{\text{upwind}} + \frac{1}{2}\psi(r)(\phi_{\text{downwind}} - \phi_{\text{upwind}})$.

---

## Part 2: Momentum Equation Assembly Teardown

In `UEqn.H`, OpenFOAM solves the incompressible Navier–Stokes momentum balance:

$$\frac{\partial \mathbf{u}}{\partial t} + \nabla \cdot (\mathbf{u} \mathbf{u}) = -\frac{1}{\rho}\nabla p + \nabla \cdot \left[ \nu_{\text{eff}} \left( \nabla \mathbf{u} + (\nabla \mathbf{u})^T \right) \right] + \mathbf{S}_{\text{MRF}} + \mathbf{S}_{\text{body}}$$

```cpp
tmp<fvVectorMatrix> tUEqn
(
    fvm::ddt(U)
  + fvm::div(phi, U)
  + MRF.DDt(U)
  + turbulence->divDevReff(U)
 ==
    fvOptions(U)
);
```

### 2.1 Term-by-Term Discretization
1. **Temporal Acceleration (`fvm::ddt(U)`)**:
   - Backward 2nd-order implicit: $\frac{3\mathbf{u}^n - 4\mathbf{u}^{n-1} + \mathbf{u}^{n-2}}{2\Delta t}$.
   - Contributes $\frac{3V_P}{2\Delta t}$ to `diag[P]` and historical inertia to source $\mathbf{b}_P$.
   - **Requirement**: Requires two-step temporal history $[\mathbf{u}^{n-1}, \mathbf{u}^{n-2}]$ to infer acceleration.
2. **Convection Linearization (`fvm::div(phi, U)`)**:
   - Linearized using known flux $\phi_f = \mathbf{u}_f \cdot \mathbf{S}_f$.
   - Diagonal dominance maintained by implicit upwind coefficients in $[A]$; higher-order TVD terms handled via explicit deferred correction in source vector $\mathbf{b}$.
3. **Turbulent Viscous Shear (`divDevReff(U)`)**:
   - Implicit Laplacian $\nabla \cdot (\nu_{\text{eff}} \nabla \mathbf{u})$ populates off-diagonals.
   - Deviatoric transpose $\nabla \cdot [\nu_{\text{eff}} (\nabla \mathbf{u})^T] = (\nabla \mathbf{u}) \cdot \nabla \nu_{\text{eff}}$ evaluated explicitly into $\mathbf{b}$.
4. **Moving Reference Frame (`MRF.DDt(U)`)**:
   - Centrifugal acceleration $\mathbf{f}_{\text{cent}} = \Omega^2 \mathbf{r}_\perp$ (exact analytical body force).
   - Coriolis acceleration $\mathbf{f}_{\text{Cor}} = 2\boldsymbol{\Omega} \times \mathbf{u}$ (couples $u_x \leftrightarrow u_y$).

### 2.2 The Fundamental $\mathbf{H}$ Vector
Splitting matrix $[A]$ into diagonal $A_D$ and off-diagonal neighbor operator $A_H$:

$$A_D \mathbf{u} + A_H \mathbf{u} = \mathbf{b} - \nabla p \implies A_D \mathbf{u} = \underbrace{\mathbf{b} - A_H \mathbf{u}}_{\mathbf{H}} - \nabla p$$

$$\boxed{\mathbf{u}^* = \frac{\mathbf{H}}{A_D} - \frac{1}{A_D} \nabla p}$$

$\mathbf{H}$ contains all non-pressure momentum physics: temporal inertia, neighbor convection-diffusion, MRF rotation, and turbulence sources. The intermediate velocity predictor solves:
$$[A] \mathbf{u}^* = -\nabla p^n$$

---

## Part 3: Pressure–Velocity Coupling & The Poisson Projection

Pressure in incompressible flow is not thermodynamic; it is the **Lagrange multiplier enforcing kinematic incompressibility**:

$$\nabla \cdot \mathbf{u} = 0$$

```cpp
volScalarField rAU(1.0/UEqn.A());
volVectorField HbyA(constrainHbyA(rAU*UEqn.H(), U, p));
surfaceScalarField phiHbyA("phiHbyA", fvc::flux(HbyA));

while (pimple.correctNonOrthogonal())
{
    fvScalarMatrix pEqn(fvm::laplacian(rAU, p) == fvc::div(phiHbyA));
    pEqn.solve(...);
    phi = phiHbyA - pEqn.flux();
}
U = HbyA - rAU*fvc::grad(p);
```

### 3.1 Derivation of the Pressure Poisson Equation
Substituting $\mathbf{u} = HbyA - rAU \nabla p$ into $\nabla \cdot \mathbf{u} = 0$:

$$\boxed{\nabla \cdot (rAU \nabla p) = \nabla \cdot (HbyA)}$$

Over cell $P$:
$$\sum_{f \in P} (rAU)_f (\nabla p)_f \cdot \mathbf{S}_f = \sum_{f \in P} \phi_{HbyA, f}$$
The right-hand side is the **net mass imbalance (dilation)** of the unconstrained momentum state.

### 3.2 Rhie–Chow Face Flux Interpolation
To eliminate checkerboard pressure decoupling on collocated grids:
$$\phi_f = \phi_{HbyA, f} - (rAU)_f (\nabla p)_f \cdot \mathbf{S}_f$$
Directly couples adjacent cell pressures $p_P$ and $p_N$.

### 3.3 The Incompressible Velocity Projection
Once pressure $p^{n+1}$ is converged:
$$\mathbf{u}^{n+1} = HbyA - rAU \nabla p^{n+1}$$
Guarantees $\sum_{f \in P} \phi_f^{n+1} = 0$ to machine precision.

---

## Part 4: Linear Solvers & Sparse Matrix Inversion

The pressure Poisson equation accounts for **75–85% of execution time** due to its global elliptic nature.

### 4.1 OpenFOAM LDU Matrix Format
OpenFOAM stores sparse coefficients in contiguous arrays:
- `diag`: Size $N_{\text{cells}}$
- `upper`: Size $N_{\text{internal\_faces}}$ (stores $a_N$ where $P = \text{owner}[f]$)
- `lower`: Size $N_{\text{internal\_faces}}$ (symmetric alias for symmetric Laplacian)

### 4.2 Linear Solvers in `fvSolution`
1. **GAMG (Geometric-Algebraic Multigrid)**:
   - Agglomerates adjacent polyhedral cells into coarse levels ($N \to N/4 \to N/16$).
   - Damps high-frequency errors on fine grids via Gauss–Seidel smoothers.
   - Solves low-frequency global modes on the coarsest level.
2. **PCG / PBiCGStab (Preconditioned Conjugate Gradient)**:
   - Krylov subspace iteration with Diagonal Incomplete Cholesky (DIC) preconditioning.
   - Requires global scalar dot-product reductions across MPI ranks at every iteration.

### 4.3 Neural Preconditioning Opportunity
A learned multi-scale neural operator (e.g. Graph FNO) can predict an approximate inverse $[M]^{-1} \approx [A_p]^{-1}$. Using the neural output as an initial guess $p_0 = p_{\text{neural}}$ reduces required Krylov iterations from **50+ down to 1–2 sweeps**.

---

## Part 5: Temporal Integration & Turbulence Closures

### 5.1 Courant Number Constraints
$$\text{Co} = \frac{\|\mathbf{u}\| \Delta t}{\Delta x} \le \text{Co}_{\max}$$
- **PISO (Pure Implicit)**: Strictly requires $\text{Co} \le 1.0$ for temporal accuracy.
- **PIMPLE (Merged PISO-SIMPLE)**: Uses outer iterations to relax the Courant constraint, allowing $\text{Co} \le 3.0 \dots 5.0$.

### 5.2 Two-Equation Turbulence Modeling ($k$-$\epsilon$ and $k$-$\omega$ SST)
1. **Turbulent Kinetic Energy ($k$)**:
   $$\frac{\partial k}{\partial t} + \nabla \cdot (\mathbf{u} k) = \nabla \cdot \left[ \left(\nu + \frac{\nu_t}{\sigma_k}\right) \nabla k \right] + P_k - \epsilon$$
2. **Turbulent Dissipation Rate ($\epsilon$)**:
   $$\frac{\partial \epsilon}{\partial t} + \nabla \cdot (\mathbf{u} \epsilon) = \nabla \cdot \left[ \left(\nu + \frac{\nu_t}{\sigma_\epsilon}\right) \nabla \epsilon \right] + C_{1\epsilon}\frac{\epsilon}{k} P_k - C_{2\epsilon}\frac{\epsilon^2}{k}$$
3. **Eddy Viscosity Closure**:
   $$\nu_t = C_\mu \frac{k^2}{\epsilon}, \quad C_\mu = 0.09$$
   Added directly to molecular viscosity: $\nu_{\text{eff}} = \nu + \nu_t$.

---

## Part 6: The PIMPLE Algorithm & Neural Architecture Synthesis

### 6.1 The Iterative PIMPLE Loop Structure
PIMPLE nests two loops:
- **Outer Loop (`nOuterCorrectors` = 2–3)**: Resolves non-linear convection-diffusion coupling and updates $\nu_t$.
- **Inner Loop (`nCorrectors` = 1–2)**: Solves pressure Poisson equation and corrects $\phi_f$ and $\mathbf{u}$.
- **Non-Orthogonal Loop (`nNonOrthogonalCorrectors` = 0–1)**: Recalculates explicit cross-diffusion fluxes.

```
                              Start Time Step t^{n+1}
                                         │
                   ┌─────────────────────┴─────────────────────┐
                   │                                           ▼
                   │                                Momentum Predictor:
                   │                              Assemble & Solve UEqn
                   │                                           │
                   ▼                                           ▼
          Outer Corrector Loop                       Calculate H and rAU:
         (Non-linear Convection,                    H = b - A_H u, rAU = 1/A_D
          Turbulence k-ε Update)                               │
                   ▲                                           ▼
                   │                               Pressure Poisson Loop:
                   │                           Solve ∇·(rAU ∇p) = ∇·(H/A_D)
                   │                                           │
                   │                                           ▼
                   │                                Correct Face Fluxes:
                   │                                φ = phiHbyA - rAU ∇p · S_f
                   │                                           │
                   │                                           ▼
                   │                                Explicit Velocity Update:
                   │                                 u = HbyA - rAU ∇p
                   │                                           │
                   └────────────────── Not Converged ──────────┤
                                                               │ Converged
                                                               ▼
                                                       Advance to t^{n+2}
```

---

## Part 7: The Master Blueprint: NLSS Neural Numerical Operator

We classify every operation in the CFD solver into five implementation strategies:

| Solver Subsystem | Specific Operation | NLSS Strategy | Neural Implementation |
| :--- | :--- | :---: | :--- |
| **Mesh Geometry** | Cell volume $V_P$, face vector $\mathbf{S}_f$, distance $\mathbf{d}$ | **Retain Mathematically** | Pre-computed static tensor buffers. |
| **Mesh Topology** | Owner / Neighbour cell connectivity | **Retain Mathematically** | Graph adjacency indices (`edge_index`). |
| **Temporal Inertia** | Acceleration $A_D = V_P / \Delta t$ | **Retain Mathematically** | Exact scalar division. |
| **Coriolis / Centrifugal** | Rotating frame forces $\mathbf{S}_{\text{MRF}}$ | **Retain Mathematically** | Exact analytical kinematic formulation. |
| **Convective Fluxes** | Non-linear transport $\sum_f \phi_f \mathbf{u}_f$ | **Replace with Neural Operator** | Graph Attention / Spatial Operator predicting neighbor momentum transfer $A_H \mathbf{u}$. |
| **Flux Limiters** | TVD / NVD limiter $\psi(r)$ | **Learn as Residual Correction** | Non-linear edge network predicting deferred flux correction. |
| **Pressure Poisson Inversion** | $[A_p]^{-1} \mathbf{b}_p \to p$ | **Replace with Neural Multigrid / FNO** | Multi-resolution neural operator mapping dilation $\nabla \cdot (HbyA) \to p$ in $\mathcal{O}(1)$ time. |
| **Pressure Residual Polish** | High-frequency residual elimination | **Hybridize** | Neural output initializes 1–2 Krylov sweeps. |
| **Velocity Correction** | Projection $\mathbf{u} = HbyA - rAU \nabla p$ | **Retain Mathematically** | Exact analytical gradient subtraction. |
| **Turbulence Closure** | Eddy viscosity $\nu_t(\mathbf{x})$ | **Hybridize** | Neural network predicts local $\nu_t$ field; diffusion evaluated via discrete Laplacian. |
| **Rollout Optimization** | Long-term temporal stability | **Trajectory Rollout Loss** | Unrolled multi-step training ($K=5$ steps) with divergence penalty. |

---

## Verification & Execution Protocol

1. **State-Transition Interface**:
   $$X_{t+1} = \mathcal{F}_{\text{NLSS}}(X_t, X_{t-1}; \Delta t, \text{RPM}, \nu)$$
2. **Trajectory Loss Formulation**:
   $$\mathcal{L} = \sum_{k=1}^K \gamma^k \|\hat{\mathbf{u}}_{t+k} - \mathbf{u}_{t+k}\|_2^2 + \lambda_{\text{div}} \|\nabla \cdot \hat{\mathbf{u}}_{t+k}\|_2^2 + \lambda_{\text{NS}} \|\mathbf{R}_{NS}(\hat{\mathbf{u}}_{t+k})\|_2^2$$
3. **Bioprocess Metric Layer**:
   Directly computes $(P/V)_{\text{diss}}$, $k_L a$, shear strain rate $\dot{\gamma}$, blend time $\tau_{95}$, and Kolmogorov microscale distribution ($\eta_{K, \text{mean}}, \eta_{K, \min}$).
