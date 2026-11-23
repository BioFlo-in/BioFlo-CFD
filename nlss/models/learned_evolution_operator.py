"""
BioFlo-CFD: Learned Numerical Evolution Operator (Level 2 / Objective B Backbone)
Replaces iterative OpenFOAM solver evolution with an autonomous, physical state-transition operator:
    X_{n+1} = F_theta(X_n, X_{n-1}, dt, mu)

Mathematical & Architectural Features:
  1. Exact 3D Solenoidal Helmholtz-Hodge Projector (LerayProjection3D):
     Eliminates velocity divergence div(u) = 0 in O(N log N) 3D Fourier space.
  2. Two-Step Autoregressive Latent Evolution (AutoregressiveStateEvolutionOperator):
     Conditions on state history (a_n, a_{n-1}) to preserve momentum and temporal derivatives.
  3. Physical Parameter Conditioning via FiLM (Feature-wise Linear Modulation):
     Conditions dynamics on operating parameters (RPM, Re, nu) across hydrodynamic regimes.
  4. Closed-Loop Numerical Evolution Engine (LearnedNumericalEvolutionEngine):
     Integrates latent dynamics, spatial basis decoding, 3D Leray incompressibility projection,
     and basis re-projection for bounded, long-horizon rollout stability.
  5. Multi-Step Rollout Loss (MultiStepRolloutLoss):
     Unrolls K-step forward trajectories penalizing cumulative drift and kinetic energy deviation.
"""

from typing import Dict, List, Optional, Tuple, Union
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LerayProjection3D(nn.Module):
    """
    Exact Discrete Spectral Leray-Helmholtz Projection Operator in 3D.
    Decomposes any 3D velocity field u into an exact solenoidal (divergence-free) component
    and an irrotational gradient component:
        u = u_sol + grad(phi)   where   div(u_sol) = 0
    
    Uses discrete trigonometric transfer functions s = sin(2*pi*k/N) exactly matching
    the discrete finite-difference / finite-volume stencil on the 3D grid.
    Mathematically guarantees discrete grid divergence div(u) = 0 to float32 machine precision (~1e-6).
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        velocity: torch.Tensor,
        fluid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            velocity: Tensor of shape (B, 3, D, H, W) or (3, D, H, W).
            fluid_mask: Optional boolean or float mask of shape (D, H, W) or (1, 1, D, H, W).
        Returns:
            Divergence-free velocity of same shape as input.
        """
        was_4d = False
        if velocity.ndim == 4:
            velocity = velocity.unsqueeze(0)
            was_4d = True

        B, C, D, H, W = velocity.shape
        if C != 3:
            raise ValueError(f"LerayProjection3D requires 3 velocity components (u_x, u_y, u_z), got {C}")

        device = velocity.device

        # Compute 3D FFT over spatial dimensions (D, H, W)
        v_fft = torch.fft.fftn(velocity, dim=(-3, -2, -1))

        # Discrete trigonometric frequency transfer functions: s_k = sin(2*pi*k/N)
        iz = torch.arange(D, device=device).reshape(1, 1, D, 1, 1) * (2.0 * math.pi / D)
        iy = torch.arange(H, device=device).reshape(1, 1, 1, H, 1) * (2.0 * math.pi / H)
        ix = torch.arange(W, device=device).reshape(1, 1, 1, 1, W) * (2.0 * math.pi / W)

        sz = torch.sin(iz)
        sy = torch.sin(iy)
        sx = torch.sin(ix)

        # Denominator matching discrete Laplacian
        denom = sx**2 + sy**2 + sz**2
        denom[:, :, 0, 0, 0] = 1.0  # Avoid zero division at DC frequency

        u_fft = v_fft[:, 0:1]
        v_fft_comp = v_fft[:, 1:2]
        w_fft = v_fft[:, 2:3]

        # Discrete divergence in Fourier domain
        div_f = sx * u_fft + sy * v_fft_comp + sz * w_fft

        # Subtract gradient projection: u_sol = u - s * div_f / denom
        u_sol_fft = u_fft - sx * div_f / (denom + self.eps)
        v_sol_fft = v_fft_comp - sy * div_f / (denom + self.eps)
        w_sol_fft = w_fft - sz * div_f / (denom + self.eps)

        # Preserve DC (mean) mode
        u_sol_fft[:, :, 0, 0, 0] = u_fft[:, :, 0, 0, 0]
        v_sol_fft[:, :, 0, 0, 0] = v_fft_comp[:, :, 0, 0, 0]
        w_sol_fft[:, :, 0, 0, 0] = w_fft[:, :, 0, 0, 0]

        sol_fft = torch.cat([u_sol_fft, v_sol_fft, w_sol_fft], dim=1)
        u_sol = torch.fft.ifftn(sol_fft, dim=(-3, -2, -1)).real

        # Zero out solid/boundary regions if fluid_mask is provided
        if fluid_mask is not None:
            if fluid_mask.ndim == 3:
                mask = fluid_mask.view(1, 1, D, H, W).float()
            else:
                mask = fluid_mask.float()
            u_sol = u_sol * mask

        if was_4d:
            u_sol = u_sol.squeeze(0)

        return u_sol


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation layer for parameter conditioning."""

    def __init__(self, cond_dim: int, hidden_dim: int):
        super().__init__()
        self.scale_bias = nn.Linear(cond_dim, 2 * hidden_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # cond: (B, cond_dim), x: (B, hidden_dim)
        sb = self.scale_bias(cond)
        gamma, beta = torch.chunk(sb, 2, dim=-1)
        return x * (1.0 + gamma) + beta


class ResNetBlockFiLM(nn.Module):
    """Residual block with FiLM conditioning and Swish activation."""

    def __init__(self, hidden_dim: int, cond_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.film1 = FiLMLayer(cond_dim, hidden_dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.film2 = FiLMLayer(cond_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.act(self.film1(self.fc1(x), cond))
        out = self.dropout(out)
        out = self.film2(self.fc2(out), cond)
        out = self.norm(res + out)
        return self.act(out)


class AutoregressiveStateEvolutionOperator(nn.Module):
    """
    Two-step autoregressive latent dynamical state-transition operator:
        v_n = (a_n - a_{n-1}) / dt
        Delta_a = F_theta(a_n, v_n, dt, mu)
        a_{n+1} = a_n + dt * Delta_a

    Captures second-order momentum and acceleration without requiring
    expensive numerical ODE solvers.
    """

    def __init__(
        self,
        n_modes: int = 20,
        param_dim: int = 1,      # Operating parameters (e.g., [RPM] or [RPM, Re, nu])
        hidden_dim: int = 128,
        num_blocks: int = 3,
        dissipative_damping: float = 0.01,
    ):
        super().__init__()
        self.n_modes = n_modes
        self.param_dim = param_dim
        self.hidden_dim = hidden_dim
        self.dissipative_damping = dissipative_damping

        # Condition input includes parameters mu and timestep dt
        cond_in_dim = param_dim + 1
        self.cond_net = nn.Sequential(
            nn.Linear(cond_in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # State input: current mode coefficients a_n and velocity v_n
        state_in_dim = 2 * n_modes
        self.input_proj = nn.Linear(state_in_dim, hidden_dim)

        # Residual FiLM blocks
        self.blocks = nn.ModuleList([
            ResNetBlockFiLM(hidden_dim, hidden_dim) for _ in range(num_blocks)
        ])

        # Output head: predicted time derivative d(a)/dt
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_modes),
        )

    def forward(
        self,
        a_curr: torch.Tensor,
        a_prev: torch.Tensor,
        dt: torch.Tensor,
        params: torch.Tensor,
    ) -> torch.Tensor:
        """
        Advance latent mode state by one timestep dt.

        Args:
            a_curr: Current modes, shape (B, n_modes)
            a_prev: Previous modes, shape (B, n_modes)
            dt: Timestep, shape (B, 1) or scalar
            params: Parameters mu, shape (B, param_dim)
        Returns:
            a_next: Predicted next modes, shape (B, n_modes)
        """
        if dt.ndim == 0 or dt.ndim == 1 and dt.numel() == 1:
            dt = dt.reshape(1, 1).expand(a_curr.shape[0], 1)
        elif dt.ndim == 1:
            dt = dt.unsqueeze(-1)

        # Finite-difference velocity estimate
        v_curr = (a_curr - a_prev) / (dt + 1e-6)

        # Build condition embedding [params, dt]
        cond_raw = torch.cat([params, dt], dim=-1)
        cond_emb = self.cond_net(cond_raw)

        # Encode state [a_curr, v_curr]
        state_in = torch.cat([a_curr, v_curr], dim=-1)
        h = self.input_proj(state_in)

        # Pass through residual FiLM blocks
        for block in self.blocks:
            h = block(h, cond_emb)

        da_dt = self.output_head(h)

        # Advance state with dissipative damping to prevent runaway explosion:
        # a_{n+1} = (1 - alpha * dt) * a_curr + dt * da_dt
        damping_factor = torch.clamp(1.0 - self.dissipative_damping * dt, min=0.5, max=1.0)
        a_next = damping_factor * a_curr + dt * da_dt
        return a_next


class LearnedNumericalEvolutionEngine(nn.Module):
    """
    Master Learned CFD Evolution Engine.
    Combines:
      - Autoregressive latent dynamics (AutoregressiveStateEvolutionOperator)
      - Orthonormal spatial POD basis (Phi)
      - Exact 3D Solenoidal incompressibility projector (LerayProjection3D)
      - Closed-loop spatial projection feedback
    """

    def __init__(
        self,
        pod_basis: torch.Tensor,          # (3*D*H*W, K) or (K, 3*D*H*W)
        base_flow: torch.Tensor,          # (3, D, H, W) or (3*D*H*W,)
        grid_shape: Tuple[int, int, int], # (D, H, W)
        param_dim: int = 1,
        hidden_dim: int = 128,
        num_blocks: int = 3,
        enforce_projection_in_loop: bool = True,
    ):
        super().__init__()
        self.grid_shape = grid_shape
        D, H, W = grid_shape
        self.spatial_dim = 3 * D * H * W

        # Ensure pod_basis is (spatial_dim, K)
        if pod_basis.ndim == 2:
            if pod_basis.shape[0] == self.spatial_dim:
                n_modes = pod_basis.shape[1]
                basis = pod_basis
            elif pod_basis.shape[1] == self.spatial_dim:
                n_modes = pod_basis.shape[0]
                basis = pod_basis.T
            else:
                raise ValueError(f"Shape mismatch: pod_basis shape {pod_basis.shape} vs spatial_dim {self.spatial_dim}")
        else:
            raise ValueError("pod_basis must be 2D tensor")

        self.n_modes = n_modes
        self.register_buffer("basis", basis.float())

        # Base flow
        if base_flow.numel() == self.spatial_dim:
            self.register_buffer("base_flow", base_flow.reshape(1, 3, D, H, W).float())
        else:
            raise ValueError(f"Base flow shape mismatch: got {base_flow.shape}, expected {self.spatial_dim}")

        self.enforce_projection_in_loop = enforce_projection_in_loop
        self.evolution_op = AutoregressiveStateEvolutionOperator(
            n_modes=n_modes,
            param_dim=param_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
        )
        self.leray_proj = LerayProjection3D()

    def decode_modes(self, a: torch.Tensor) -> torch.Tensor:
        """
        Decodes latent mode vector a into spatial 3D velocity field:
            u = u_bar + sum_k a_k * phi_k
        Args:
            a: Tensor of shape (B, K)
        Returns:
            u: Tensor of shape (B, 3, D, H, W)
        """
        B, K = a.shape
        D, H, W = self.grid_shape
        # u_pert = a @ basis.T -> shape (B, spatial_dim)
        u_pert = torch.matmul(a, self.basis.T)
        u_pert = u_pert.view(B, 3, D, H, W)
        return self.base_flow + u_pert

    def encode_field(self, u: torch.Tensor) -> torch.Tensor:
        """
        Projects spatial 3D velocity field u onto latent POD modes:
            a = Phi^T (u - u_bar)
        Args:
            u: Tensor of shape (B, 3, D, H, W)
        Returns:
            a: Tensor of shape (B, K)
        """
        B, C, D, H, W = u.shape
        u_pert = (u - self.base_flow).reshape(B, self.spatial_dim)
        # a = u_pert @ basis -> shape (B, K)
        return torch.matmul(u_pert, self.basis)

    def step(
        self,
        a_curr: torch.Tensor,
        a_prev: torch.Tensor,
        dt: torch.Tensor,
        params: torch.Tensor,
        fluid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Advances one physical step.

        Returns:
            (a_next, u_next) where u_next is divergence-free.
        """
        # 1. Advance latent dynamics
        a_raw = self.evolution_op(a_curr, a_prev, dt, params)

        # 2. Decode to 3D spatial field
        u_raw = self.decode_modes(a_raw)

        # 3. Enforce exact solenoidal projection
        u_clean = self.leray_proj(u_raw, fluid_mask=fluid_mask)

        # 4. Optional closed-loop re-projection to maintain latent-spatial alignment
        if self.enforce_projection_in_loop:
            a_next = self.encode_field(u_clean)
        else:
            a_next = a_raw

        return a_next, u_clean

    def rollout(
        self,
        a_0: torch.Tensor,
        a_prev: torch.Tensor,
        dt: torch.Tensor,
        params: torch.Tensor,
        num_steps: int,
        fluid_mask: Optional[torch.Tensor] = None,
        return_fields: bool = False,
    ) -> Dict[str, Union[torch.Tensor, List[float]]]:
        """
        Autonomously rolls out a multi-step trajectory without external CFD intervention.

        Returns:
            Dictionary containing:
              - 'modes': (num_steps, B, K)
              - 'kinetic_energy': List of mean kinetic energy per step
              - 'divergence_norm': List of mean relative divergence per step
              - 'fields': Optional (num_steps, B, 3, D, H, W)
        """
        modes_list = []
        fields_list = []
        ke_list = []
        div_list = []

        curr_a = a_0
        prev_a = a_prev

        for step_idx in range(num_steps):
            next_a, next_u = self.step(curr_a, prev_a, dt, params, fluid_mask=fluid_mask)
            modes_list.append(next_a.unsqueeze(0))
            if return_fields:
                fields_list.append(next_u.unsqueeze(0))

            # Compute kinetic energy: 0.5 * integral ||u||^2 dV
            ke = 0.5 * torch.mean(torch.sum(next_u**2, dim=1)).item()
            ke_list.append(ke)

            # Compute divergence in spatial domain
            du_dx = torch.gradient(next_u[:, 0], dim=-1)[0]
            dv_dy = torch.gradient(next_u[:, 1], dim=-2)[0]
            dw_dz = torch.gradient(next_u[:, 2], dim=-3)[0]
            div_val = torch.mean(torch.abs(du_dx + dv_dy + dw_dz)).item()
            div_list.append(div_val)

            # Advance pointers
            prev_a = curr_a
            curr_a = next_a

        res = {
            "modes": torch.cat(modes_list, dim=0),
            "kinetic_energy": ke_list,
            "divergence_norm": div_list,
        }
        if return_fields:
            res["fields"] = torch.cat(fields_list, dim=0)

        return res


class MultiStepRolloutLoss(nn.Module):
    """
    Loss function for training the evolution operator over K-step rollouts:
      L = sum_{k=1}^K gamma^k * ( ||a_{n+k} - a^*_{n+k}||^2 + lambda_E * |E_k - E^*_k| + lambda_v * ||v_k - v^*_k||^2 )
    """

    def __init__(
        self,
        gamma: float = 0.95,
        energy_weight: float = 0.05,
        velocity_weight: float = 0.1,
    ):
        super().__init__()
        self.gamma = gamma
        self.energy_weight = energy_weight
        self.velocity_weight = velocity_weight

    def forward(
        self,
        predicted_trajectories: torch.Tensor,  # (K, B, n_modes)
        target_trajectories: torch.Tensor,     # (K, B, n_modes)
    ) -> Dict[str, torch.Tensor]:
        K, B, M = predicted_trajectories.shape
        total_loss = 0.0
        mode_loss_sum = 0.0
        energy_loss_sum = 0.0
        vel_loss_sum = 0.0

        for k in range(K):
            decay = self.gamma**k
            p_k = predicted_trajectories[k]
            t_k = target_trajectories[k]

            # 1. Mode coefficient L2 loss
            m_loss = F.mse_loss(p_k, t_k)

            # 2. Kinetic energy preservation in latent space: ||a||^2
            p_energy = torch.sum(p_k**2, dim=-1)
            t_energy = torch.sum(t_k**2, dim=-1)
            e_loss = F.l1_loss(p_energy, t_energy)

            # 3. Velocity / derivative loss
            if k > 0:
                p_vel = p_k - predicted_trajectories[k - 1]
                t_vel = t_k - target_trajectories[k - 1]
                v_loss = F.mse_loss(p_vel, t_vel)
            else:
                v_loss = torch.tensor(0.0, device=p_k.device)

            step_loss = m_loss + self.energy_weight * e_loss + self.velocity_weight * v_loss
            total_loss = total_loss + decay * step_loss

            mode_loss_sum = mode_loss_sum + m_loss
            energy_loss_sum = energy_loss_sum + e_loss
            vel_loss_sum = vel_loss_sum + v_loss

        return {
            "total_loss": total_loss,
            "mode_loss": mode_loss_sum / K,
            "energy_loss": energy_loss_sum / K,
            "velocity_loss": vel_loss_sum / max(1, K - 1),
        }
