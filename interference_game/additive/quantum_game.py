from __future__ import annotations

import numpy as np
import torch

from interference_game.additive.config import AdditiveExperimentConfig
from interference_game.additive.activations import entmax15
from interference_game.additive.classical_game import ClassicalGroundTruthGame
from interference_game.additive.scoring import DistributionEvaluationResult, DistributionScoringMixin


class QuantumEncodedGame(DistributionScoringMixin):
    def __init__(self, reference_game: ClassicalGroundTruthGame, experiment_config: AdditiveExperimentConfig):
        super().__init__(reference_game.config, reference_game.targets)
        self.reference_game = reference_game
        self.experiment_config = experiment_config
        self.initial_distribution = reference_game.initial_distribution
        self.initial_state = torch.sqrt(self.initial_distribution).to(dtype=self.complex_dtype)
        self.transition_matrix = reference_game.transition_matrix
        self.agent_influences = reference_game.agent_influences
        self.mixer = self._build_unitary(reference_game.transition_matrix)

    def _build_unitary(self, transition_matrix: torch.Tensor) -> torch.Tensor:
        real = transition_matrix.to(dtype=torch.float64)
        complex_matrix = real.to(dtype=self.complex_dtype) + 1j * real.T.to(dtype=self.complex_dtype)
        q, r = torch.linalg.qr(complex_matrix)
        diagonal = torch.diagonal(r)
        phases = diagonal / torch.clamp(diagonal.abs(), min=1e-12)
        return q * phases.conj().unsqueeze(0)

    def _normalize_state(self, psi: torch.Tensor) -> torch.Tensor:
        norm = torch.clamp(torch.linalg.norm(psi), min=1e-12)
        return psi / norm

    def _joint_signal(self, projected: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        per_agent = torch.einsum("asd,ad->as", self.agent_influences, projected)
        return per_agent, per_agent.sum(dim=0)

    def _simulate_state(self, projected: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        distribution = self.initial_distribution.clone()
        psi = self.initial_state.clone()
        steps = max(self.config.mixing_depth, 1)
        agent_signal, total_signal = self._joint_signal(projected)
        for _ in range(steps):
            for agent_idx in range(self.config.num_agents):
                phase = agent_signal[agent_idx] / float(steps)
                psi = psi * torch.exp(1j * phase.to(dtype=self.complex_dtype))
                psi = self._normalize_state(psi)
                if self.config.mixing_depth > 0:
                    psi = self._normalize_state(self.mixer @ psi)
            logits = self.transition_matrix @ distribution + total_signal
            distribution = entmax15(logits, dim=0)
            phases = torch.angle(psi)
            psi = torch.sqrt(torch.clamp(distribution, min=0.0)).to(dtype=self.complex_dtype) * torch.exp(1j * phases)
            psi = self._normalize_state(psi)
        return distribution, psi

    def sample_from_distribution(self, distribution: torch.Tensor, num_draws: int, seed: int) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        draws = torch.multinomial(distribution.cpu(), num_draws, replacement=True, generator=generator)
        counts = torch.bincount(draws, minlength=self.config.state_dim).to(dtype=self.real_dtype, device=self.device)
        return counts / float(num_draws)

    def estimate_observable_expectations(self, distribution: torch.Tensor, num_qubits: int | None = None) -> torch.Tensor:
        exact = self.observables @ distribution.to(self.device, dtype=self.real_dtype)
        resolved_qubits = self.experiment_config.amplitude_estimation_qubits if num_qubits is None else num_qubits
        num_qubits = max(int(resolved_qubits), 1)
        grid = 2**num_qubits
        amplitudes = torch.sqrt(torch.clamp(exact, min=0.0, max=1.0))
        angles = torch.arcsin(torch.clamp(amplitudes, min=0.0, max=1.0))
        grid_index = torch.round(grid * angles / np.pi)
        grid_index = torch.clamp(grid_index, min=0, max=grid // 2)
        estimated = torch.sin(np.pi * grid_index / grid).square()
        return torch.clamp(estimated, min=0.0, max=1.0)

    def evaluate(self, actions: torch.Tensor | np.ndarray | list[list[float]]) -> DistributionEvaluationResult:
        projected = self.project_actions(actions)
        distribution, psi = self._simulate_state(projected)
        expectations = self.estimate_observable_expectations(distribution)
        return self._score_distribution(
            distribution,
            projected,
            latent_state=psi,
            observable_expectations=expectations,
        )
