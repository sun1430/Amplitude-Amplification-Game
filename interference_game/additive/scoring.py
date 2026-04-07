from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from interference_game.config import ModelConfig


@dataclass(slots=True)
class DistributionEvaluationResult:
    influence_distribution: torch.Tensor
    utilities: torch.Tensor
    costs: torch.Tensor
    competitive_penalties: torch.Tensor
    latent_state: torch.Tensor | None = None


class DistributionScoringMixin:
    def __init__(self, config: ModelConfig, targets: torch.Tensor):
        self.config = config
        requested_device = config.device
        if requested_device == "cpu" and torch.cuda.is_available():
            requested_device = "cuda"
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            requested_device = "cpu"
        self.device = torch.device(requested_device)
        self.real_dtype = torch.float64
        self.complex_dtype = torch.complex128
        self.targets = targets.to(self.device, dtype=self.real_dtype)
        self.budgets = torch.tensor(config.budgets(), dtype=self.real_dtype, device=self.device)
        self.lambdas = torch.tensor(config.lambda_list(), dtype=self.real_dtype, device=self.device)
        self.gammas = torch.tensor(config.gamma_list(), dtype=self.real_dtype, device=self.device)

    def _as_action_tensor(self, actions: torch.Tensor | np.ndarray | list[list[float]]) -> torch.Tensor:
        tensor = torch.as_tensor(actions, dtype=self.real_dtype, device=self.device)
        expected = (self.config.num_agents, self.config.state_dim)
        if tensor.shape != expected:
            raise ValueError(f"Actions must have shape {expected}, got {tuple(tensor.shape)}.")
        return tensor

    def project_actions(self, actions: torch.Tensor | np.ndarray | list[list[float]]) -> torch.Tensor:
        tensor = self._as_action_tensor(actions).clone()
        norms = torch.linalg.norm(tensor, dim=1)
        radii = torch.sqrt(self.budgets)
        scales = torch.minimum(torch.ones_like(norms), radii / torch.clamp(norms, min=1e-12))
        return tensor * scales.unsqueeze(1)

    def _score_distribution(
        self,
        distribution: torch.Tensor,
        actions: torch.Tensor,
        latent_state: torch.Tensor | None = None,
    ) -> DistributionEvaluationResult:
        projected = self.project_actions(actions)
        normalized = torch.clamp(distribution.to(self.device, dtype=self.real_dtype), min=0.0)
        normalized = normalized / torch.clamp(normalized.sum(), min=1e-12)
        distances = (self.targets - normalized.unsqueeze(0)).square().sum(dim=1)
        costs = self.lambdas * projected.square().sum(dim=1)
        alignments = torch.clamp(1.0 - distances, min=0.0)
        if self.config.use_competitive_extension and self.config.num_agents > 1:
            penalties = []
            for agent_idx in range(self.config.num_agents):
                others = torch.cat((alignments[:agent_idx], alignments[agent_idx + 1 :]))
                penalties.append(self.gammas[agent_idx] * others.mean())
            competitive_penalties = torch.stack(penalties)
        else:
            competitive_penalties = torch.zeros_like(distances)
        utilities = -distances - costs - competitive_penalties
        return DistributionEvaluationResult(
            influence_distribution=normalized,
            utilities=utilities,
            costs=costs,
            competitive_penalties=competitive_penalties,
            latent_state=latent_state,
        )
