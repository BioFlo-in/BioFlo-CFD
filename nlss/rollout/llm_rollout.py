"""
LLM-Style 1-Step Autoregressive Rollout Engine.
Adopts next-token generation mechanics for fluid dynamics simulations:
  - Maintains sliding context window of past physical states [S_{t-K+1}, ..., S_t].
  - Predicts strictly 1 state into the future at a time: S_{t+1}.
  - Appends predicted state to context buffer and iterates.
"""

import time
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from ..dataset.foam_dataset import FieldNormalizer


class LLMStyleRolloutEngine:
    """
    Executes 1-step-at-a-time autoregressive rollout using a temporal context window.
    """

    def __init__(
        self,
        model: nn.Module,
        normalizer: FieldNormalizer,
        context_window: int = 4,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.normalizer = normalizer
        self.context_window = context_window

    @torch.no_grad()
    def predict_next_step(self, context_states: List[torch.Tensor]) -> torch.Tensor:
        """
        Takes context window of K previous states and predicts strictly the single next step.
        """
        # Stack into (1, K, C, H, W)
        k_states = context_states[-self.context_window:]
        stacked = torch.stack([s.to(self.device) for s in k_states], dim=0).unsqueeze(0)

        # Normalize context tokens
        B, K, C, H, W = stacked.shape
        flat = stacked.view(B * K, C, H, W)
        norm_flat = self.normalizer.normalize(flat)
        norm_seq = norm_flat.view(B, K, C, H, W)

        # 1-step forward pass through Transformer
        norm_next = self.model(norm_seq)

        # Denormalize next state
        next_phys = self.normalizer.denormalize(norm_next).squeeze(0)
        return next_phys

    @torch.no_grad()
    def rollout(
        self,
        prompt_states: Union[List[torch.Tensor], torch.Tensor],
        num_steps: int,
    ) -> Tuple[torch.Tensor, float]:
        """
        Autoregressively rolls out num_steps into the future, 1 step at a time.
        """
        if isinstance(prompt_states, torch.Tensor):
            if prompt_states.ndim == 3:
                history = [prompt_states.clone().to(self.device)]
            else:
                history = [prompt_states[i].clone().to(self.device) for i in range(prompt_states.shape[0])]
        else:
            history = [s.clone().to(self.device) for s in prompt_states]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        for step in range(num_steps):
            next_step = self.predict_next_step(history)
            history.append(next_step)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        trajectory = torch.stack(history, dim=0)
        return trajectory, elapsed
