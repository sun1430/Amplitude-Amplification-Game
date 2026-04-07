from __future__ import annotations

import numpy as np
import torch

from interference_game.additive.activations import entmax15
from interference_game.additive.scoring import DistributionEvaluationResult, DistributionScoringMixin
from interference_game.config import ModelConfig


class ClassicalGroundTruthGame(DistributionScoringMixin):
    def __init__(self, config: ModelConfig, targets: torch.Tensor):
        super().__init__(config, targets)
        self.initial_distribution = torch.full(
            (self.config.state_dim,),
            1.0 / float(self.config.state_dim),
            dtype=self.real_dtype,
            device=self.device,
        )
        self.transition_matrix, self.agent_influences = self._build_dynamics()

    def _build_dynamics(self) -> tuple[torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng(self.config.mixer_seed)
        matrix = rng.standard_normal((self.config.state_dim, self.config.state_dim)) / np.sqrt(self.config.state_dim)
        matrix += 0.75 * np.eye(self.config.state_dim)
        spectral_norm = np.linalg.norm(matrix, ord=2)
        matrix = matrix / max(spectral_norm, 1.0)
        influences = rng.standard_normal((self.config.num_agents, self.config.state_dim, self.config.state_dim)) / np.sqrt(
            self.config.state_dim
        )
        return (
            torch.tensor(matrix, dtype=self.real_dtype, device=self.device),
            torch.tensor(influences, dtype=self.real_dtype, device=self.device),
        )

    def _joint_influence(self, projected: torch.Tensor) -> torch.Tensor:
        return torch.einsum("asd,ad->s", self.agent_influences, projected)

    def _simulate_distribution(self, projected: torch.Tensor) -> torch.Tensor:
        state = self.initial_distribution.clone()
        steps = max(self.config.mixing_depth, 1)
        influence = self._joint_influence(projected)
        for _ in range(steps):
            logits = self.transition_matrix @ state + influence
            state = entmax15(logits, dim=0)
        return state

    def evaluate(self, actions: torch.Tensor | np.ndarray | list[list[float]]) -> DistributionEvaluationResult:
        projected = self.project_actions(actions)
        distribution = self._simulate_distribution(projected)
        return self._score_distribution(distribution, projected, latent_state=distribution)
