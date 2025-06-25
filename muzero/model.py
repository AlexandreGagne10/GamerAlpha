"""Neural network components for MuZero."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class MuZeroNetwork(nn.Module):
    """Minimal MuZero network with representation, dynamics and prediction parts."""

    def __init__(
        self,
        observation_shape: Tuple[int, ...],
        action_space_size: int,
        hidden_dim: int = 128,
        latent_dim: int = 64,
    ) -> None:
        """Initialize the network architecture."""

        super().__init__()
        obs_size = observation_shape[0]
        self.action_space = action_space_size
        # Representation network
        self.representation = nn.Sequential(
            nn.Linear(obs_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        # Dynamics network: input latent + action -> next latent and reward
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + action_space_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim + 1),
        )
        # Prediction network: latent -> policy logits and value
        self.policy_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_space_size)
        )
        self.value_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def initial_inference(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute initial latent state, policy and value from an observation."""

        latent = self.representation(observation)
        policy_logits = self.policy_head(latent)
        value = torch.tanh(self.value_head(latent))
        reward = torch.zeros_like(value)
        return latent, value, reward, policy_logits

    def recurrent_inference(
        self, latent: torch.Tensor, action_one_hot: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict next latent state and outputs conditioned on an action."""

        x = torch.cat([latent, action_one_hot], dim=1)
        out = self.dynamics(x)
        next_latent = F.relu(out[:, :-1])
        reward = out[:, -1:]
        policy_logits = self.policy_head(next_latent)
        value = torch.tanh(self.value_head(next_latent))
        return next_latent, value, reward, policy_logits
