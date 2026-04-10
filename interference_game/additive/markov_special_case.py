from __future__ import annotations

import numpy as np
import torch

from interference_game.additive.classical_game import ClassicalGroundTruthGame
from interference_game.additive.config import AdditiveExperimentConfig
from interference_game.additive.quantum_game import QuantumEncodedGame
from interference_game.additive.scoring import DistributionEvaluationResult, DistributionScoringMixin
from interference_game.config import ModelConfig


class ReversibleSparseMarkovGame(DistributionScoringMixin):
    def __init__(self, config: ModelConfig, targets: torch.Tensor):
        super().__init__(config, targets)
        self.initial_distribution = torch.full(
            (self.config.state_dim,),
            1.0 / float(self.config.state_dim),
            dtype=self.real_dtype,
            device=self.device,
        )
        self.transition_matrix, self.stationary_distribution = self._build_transition_matrix()
        self.source_bias, self.agent_influences = self._build_source_map()

    def _build_transition_matrix(self) -> tuple[torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng(self.config.mixer_seed)
        state_dim = self.config.state_dim
        degree = max(1, min(self.config.markov_graph_degree, max(state_dim - 1, 1)))
        adjacency = np.zeros((state_dim, state_dim), dtype=np.float64)
        for node in range(state_dim):
            for offset in range(1, degree + 1):
                neighbor = (node + offset) % state_dim
                weight = 0.5 + rng.random()
                adjacency[node, neighbor] = max(adjacency[node, neighbor], weight)
                adjacency[neighbor, node] = adjacency[node, neighbor]
        adjacency += float(self.config.markov_self_loop) * np.eye(state_dim, dtype=np.float64)
        degree_vector = adjacency.sum(axis=1, keepdims=True)
        transition = adjacency / np.clip(degree_vector, 1e-12, None)
        stationary = degree_vector[:, 0] / np.clip(degree_vector.sum(), 1e-12, None)
        return (
            torch.tensor(transition.T, dtype=self.real_dtype, device=self.device),
            torch.tensor(stationary, dtype=self.real_dtype, device=self.device),
        )

    def _build_source_map(self) -> tuple[torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng(self.config.mixer_seed + 7919)
        base = rng.standard_normal(self.config.state_dim) / np.sqrt(self.config.state_dim)
        influences = rng.standard_normal((self.config.num_agents, self.config.state_dim, self.config.state_dim)) / np.sqrt(
            self.config.state_dim
        )
        return (
            torch.tensor(base, dtype=self.real_dtype, device=self.device),
            torch.tensor(influences, dtype=self.real_dtype, device=self.device),
        )

    def _joint_influence(self, projected: torch.Tensor) -> torch.Tensor:
        return torch.einsum("asd,ad->s", self.agent_influences, projected)

    def _source_distribution(self, projected: torch.Tensor) -> torch.Tensor:
        logits = self.source_bias + self._joint_influence(projected)
        return ClassicalGroundTruthGame.apply_activation(self, logits)

    def _simulate_distribution(self, projected: torch.Tensor) -> torch.Tensor:
        state = self._source_distribution(projected)
        inertia = float(np.clip(self.config.markov_inertia, 0.0, 1.0))
        steps = max(self.config.mixing_depth, 1)
        for _ in range(steps):
            state = (1.0 - inertia) * (self.transition_matrix @ state) + inertia * self.stationary_distribution
            state = state / torch.clamp(state.sum(), min=1e-12)
        return state

    def evaluate(self, actions: torch.Tensor | np.ndarray | list[list[float]]) -> DistributionEvaluationResult:
        projected = self.project_actions(actions)
        distribution = self._simulate_distribution(projected)
        return self._score_distribution(distribution, projected, latent_state=distribution)


class QuantumMarkovEncodedGame(DistributionScoringMixin):
    def __init__(self, reference_game: ReversibleSparseMarkovGame, experiment_config: AdditiveExperimentConfig):
        super().__init__(reference_game.config, reference_game.targets)
        self.reference_game = reference_game
        self.experiment_config = experiment_config
        self.initial_distribution = reference_game.initial_distribution
        self.initial_state = torch.sqrt(self.initial_distribution).to(dtype=self.complex_dtype)
        self.transition_matrix = reference_game.transition_matrix
        self.stationary_distribution = reference_game.stationary_distribution
        self.agent_influences = reference_game.agent_influences

    def _normalize_state(self, psi: torch.Tensor) -> torch.Tensor:
        norm = torch.clamp(torch.linalg.norm(psi), min=1e-12)
        return psi / norm

    def _simulate_state(self, projected: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        distribution = self.reference_game._source_distribution(projected)
        psi = torch.sqrt(torch.clamp(distribution, min=0.0)).to(dtype=self.complex_dtype)
        steps = max(self.config.mixing_depth, 1)
        total_signal = self.reference_game._joint_influence(projected)
        inertia = float(np.clip(self.config.markov_inertia, 0.0, 1.0))
        for _ in range(steps):
            phase = total_signal / float(max(steps, 1))
            psi = self._normalize_state(psi * torch.exp(1j * phase.to(dtype=self.complex_dtype)))
            distribution = (1.0 - inertia) * (self.transition_matrix @ distribution) + inertia * self.stationary_distribution
            distribution = distribution / torch.clamp(distribution.sum(), min=1e-12)
            psi = torch.sqrt(torch.clamp(distribution, min=0.0)).to(dtype=self.complex_dtype) * torch.exp(1j * torch.angle(psi))
            psi = self._normalize_state(psi)
        return distribution, psi

    def sample_from_distribution(self, distribution: torch.Tensor, num_draws: int, seed: int) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        draws = torch.multinomial(distribution.cpu(), num_draws, replacement=True, generator=generator)
        counts = torch.bincount(draws, minlength=self.config.state_dim).to(dtype=self.real_dtype, device=self.device)
        return counts / float(num_draws)

    def estimate_observable_expectations(self, distribution: torch.Tensor, num_qubits: int | None = None) -> torch.Tensor:
        return QuantumEncodedGame.estimate_observable_expectations(self, distribution, num_qubits=num_qubits)

    def evaluate(self, actions: torch.Tensor | np.ndarray | list[list[float]]) -> DistributionEvaluationResult:
        projected = self.project_actions(actions)
        distribution, psi = self._simulate_state(projected)
        exact_expectations = self.observables @ distribution.to(self.device, dtype=self.real_dtype)
        return self._score_distribution(
            distribution,
            projected,
            latent_state=psi,
            observable_expectations=exact_expectations,
        )
