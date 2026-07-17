"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        self.network = []
        for dim in hidden_dims:
            self.network.append(nn.Linear(state_dim, dim))
            self.network.append(nn.ReLU())
            state_dim = dim
        self.network.append(nn.Linear(state_dim, action_dim * chunk_size))
        self.model = nn.Sequential(*self.network)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        pred_action_chunk = self.sample_actions(state)
        batch_size = state.shape[0]
        return nn.functional.mse_loss(pred_action_chunk, action_chunk,reduction='sum') / batch_size

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        return self.model(state).reshape(-1, self.chunk_size, self.action_dim)


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        input_dim = state_dim + action_dim * chunk_size + 1
        self.network = []
        for dim in hidden_dims:
            self.network.append(nn.Linear(input_dim, dim))
            self.network.append(nn.ReLU())
            input_dim = dim
        self.network.append(nn.Linear(input_dim, action_dim * chunk_size))
        self.model = nn.Sequential(*self.network)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        noise_action_chunk = torch.randn_like(action_chunk)
        batch_size = state.shape[0]
        t = torch.rand(batch_size, 1, device=state.device)[:,None,:]
        interpolated_action_chunk = (1 - t) * noise_action_chunk + t * action_chunk
        model_input = torch.cat(
                [state,
                interpolated_action_chunk.reshape(-1, self.chunk_size * self.action_dim),
                t.reshape(-1,1)],
                dim=1,
            )
        pred_velocity = self.model(model_input).reshape(-1, self.chunk_size, self.action_dim)
        expert_action_chunk = action_chunk - noise_action_chunk
        return nn.functional.mse_loss(pred_velocity, expert_action_chunk,reduction='sum') / batch_size

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        action_chunk = torch.randn(state.shape[0], self.chunk_size, self.action_dim,device=state.device)
        for t in range(num_steps):
            model_input = torch.cat(
                [state,
                action_chunk.reshape(-1, self.chunk_size * self.action_dim),
                (t / num_steps) * torch.ones(state.shape[0], 1, device=state.device)],
                dim=1,
            )
            velocity = self.model(model_input).reshape(-1, self.chunk_size, self.action_dim)
            action_chunk = action_chunk + (velocity / num_steps)
        return action_chunk.reshape(-1, self.chunk_size, self.action_dim)


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
