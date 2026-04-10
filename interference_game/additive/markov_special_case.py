from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from interference_game.additive.classical_game import ClassicalGroundTruthGame
from interference_game.additive.config import AdditiveExperimentConfig
from interference_game.additive.scoring import DistributionEvaluationResult, DistributionScoringMixin
from interference_game.config import ModelConfig


@dataclass(slots=True)
class QuantumWalkStructure:
    row_transition: torch.Tensor
    reverse_row_transition: torch.Tensor
    column_operator: torch.Tensor
    stationary_distribution: torch.Tensor
    sqrt_stationary: torch.Tensor
    neighbor_indices: torch.Tensor
    neighbor_probabilities: torch.Tensor
    neighbor_counts: torch.Tensor
    discriminant: torch.Tensor
    eigenvalues: torch.Tensor
    eigenvectors: torch.Tensor
    walk_unitary: torch.Tensor

    @property
    def max_degree(self) -> int:
        return int(self.neighbor_indices.shape[1])

    def neighbor_probability_oracle(self, node: int, slot: int) -> tuple[int, float]:
        count = int(self.neighbor_counts[node].item())
        if slot < 0 or slot >= count:
            raise IndexError(f"Neighbor slot {slot} is out of range for node {node} with degree {count}.")
        neighbor = int(self.neighbor_indices[node, slot].item())
        probability = float(self.neighbor_probabilities[node, slot].item())
        return neighbor, probability

    def prepare_row_state(self, node: int) -> torch.Tensor:
        return torch.sqrt(torch.clamp(self.row_transition[node], min=0.0))

    def prepare_column_state(self, node: int) -> torch.Tensor:
        return torch.sqrt(torch.clamp(self.reverse_row_transition[node], min=0.0))


class ReversibleSparseMarkovGame(DistributionScoringMixin):
    def __init__(self, config: ModelConfig, targets: torch.Tensor):
        super().__init__(config, targets)
        self.initial_distribution = torch.full(
            (self.config.state_dim,),
            1.0 / float(self.config.state_dim),
            dtype=self.real_dtype,
            device=self.device,
        )
        (
            self.row_transition,
            self.transition_matrix,
            self.stationary_distribution,
            self.walk_structure,
        ) = self._build_transition_matrix()
        self.source_bias, self.agent_influences = self._build_source_map()

    def _build_transition_matrix(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, QuantumWalkStructure]:
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
        transition = adjacency / np.clip(degree_vector, 1e-12, None)  # Row-stochastic.
        stationary = degree_vector[:, 0] / np.clip(degree_vector.sum(), 1e-12, None)
        inertia = float(np.clip(self.config.markov_inertia, 0.0, 1.0))
        if inertia > 0.0:
            transition = inertia * np.eye(state_dim, dtype=np.float64) + (1.0 - inertia) * transition

        row_transition = torch.tensor(transition, dtype=self.real_dtype, device=self.device)
        stationary_tensor = torch.tensor(stationary, dtype=self.real_dtype, device=self.device)
        walk_structure = self._build_walk_structure(row_transition, stationary_tensor)
        return (
            row_transition,
            walk_structure.column_operator,
            stationary_tensor,
            walk_structure,
        )

    def _build_walk_structure(self, row_transition: torch.Tensor, stationary: torch.Tensor) -> QuantumWalkStructure:
        state_dim = row_transition.shape[0]
        row_nnz = (row_transition > 1e-12).sum(dim=1)
        max_degree = int(torch.max(row_nnz).item())
        neighbor_indices = torch.full((state_dim, max_degree), -1, dtype=torch.int64, device=self.device)
        neighbor_probabilities = torch.zeros((state_dim, max_degree), dtype=self.real_dtype, device=self.device)
        for node in range(state_dim):
            neighbors = torch.nonzero(row_transition[node] > 1e-12, as_tuple=False).flatten()
            count = int(neighbors.numel())
            neighbor_indices[node, :count] = neighbors.to(dtype=torch.int64)
            neighbor_probabilities[node, :count] = row_transition[node, neighbors]

        reverse_row = (stationary.unsqueeze(1) * row_transition).T / torch.clamp(stationary.unsqueeze(1), min=1e-12)
        column_operator = row_transition.T
        sqrt_stationary = torch.sqrt(torch.clamp(stationary, min=1e-12))
        discriminant = torch.diag(sqrt_stationary) @ row_transition @ torch.diag(1.0 / sqrt_stationary)
        discriminant = 0.5 * (discriminant + discriminant.T)
        eigenvalues, eigenvectors = torch.linalg.eigh(discriminant)
        walk_unitary = self._build_szegedy_walk(row_transition, reverse_row)
        return QuantumWalkStructure(
            row_transition=row_transition,
            reverse_row_transition=reverse_row,
            column_operator=column_operator,
            stationary_distribution=stationary,
            sqrt_stationary=sqrt_stationary,
            neighbor_indices=neighbor_indices,
            neighbor_probabilities=neighbor_probabilities,
            neighbor_counts=row_nnz.to(dtype=torch.int64),
            discriminant=discriminant,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            walk_unitary=walk_unitary,
        )

    def _build_szegedy_walk(self, row_transition: torch.Tensor, reverse_row: torch.Tensor) -> torch.Tensor:
        state_dim = row_transition.shape[0]
        hilbert_dim = state_dim * state_dim
        row_isometry = torch.zeros((hilbert_dim, state_dim), dtype=self.complex_dtype, device=self.device)
        column_isometry = torch.zeros((hilbert_dim, state_dim), dtype=self.complex_dtype, device=self.device)
        for node in range(state_dim):
            row_state = torch.sqrt(torch.clamp(row_transition[node], min=0.0)).to(dtype=self.complex_dtype)
            row_isometry[node * state_dim : (node + 1) * state_dim, node] = row_state
            column_state = torch.sqrt(torch.clamp(reverse_row[node], min=0.0)).to(dtype=self.complex_dtype)
            column_isometry[node::state_dim, node] = column_state
        identity = torch.eye(hilbert_dim, dtype=self.complex_dtype, device=self.device)
        reflector_a = 2.0 * (row_isometry @ row_isometry.conj().T) - identity
        reflector_b = 2.0 * (column_isometry @ column_isometry.conj().T) - identity
        return reflector_b @ reflector_a

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
        steps = max(int(self.config.mixing_depth), 0)
        for _ in range(steps):
            state = self.transition_matrix @ state
            state = state / torch.clamp(state.sum(), min=1e-12)
        return state

    def neighbor_probability_oracle(self, node: int, slot: int) -> tuple[int, float]:
        return self.walk_structure.neighbor_probability_oracle(node, slot)

    def prepare_row_state(self, node: int) -> torch.Tensor:
        return self.walk_structure.prepare_row_state(node)

    def prepare_column_state(self, node: int) -> torch.Tensor:
        return self.walk_structure.prepare_column_state(node)

    def evaluate(self, actions: torch.Tensor | np.ndarray | list[list[float]]) -> DistributionEvaluationResult:
        projected = self.project_actions(actions)
        distribution = self._simulate_distribution(projected)
        return self._score_distribution(distribution, projected, latent_state=distribution)

    def sample_terminal_distribution(
        self,
        actions: torch.Tensor | np.ndarray | list[list[float]],
        num_draws: int,
        seed: int,
    ) -> torch.Tensor:
        projected = self.project_actions(actions)
        source_distribution = self._source_distribution(projected)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)

        current_states = torch.multinomial(
            source_distribution.detach().cpu(),
            num_samples=num_draws,
            replacement=True,
            generator=generator,
        )
        steps = max(int(self.config.mixing_depth), 0)
        row_transition_cpu = self.row_transition.detach().cpu()
        for _ in range(steps):
            transition_rows = row_transition_cpu[current_states]
            current_states = torch.multinomial(transition_rows, num_samples=1, replacement=True, generator=generator).squeeze(1)

        counts = torch.bincount(current_states, minlength=self.config.state_dim).to(dtype=self.real_dtype, device=self.device)
        return counts / float(num_draws)

    def sample_observable_expectations(
        self,
        actions: torch.Tensor | np.ndarray | list[list[float]],
        num_draws: int,
        seed: int,
    ) -> torch.Tensor:
        sampled_distribution = self.sample_terminal_distribution(actions, num_draws=num_draws, seed=seed)
        return self.observables @ sampled_distribution


class QuantumMarkovEncodedGame(DistributionScoringMixin):
    def __init__(self, reference_game: ReversibleSparseMarkovGame, experiment_config: AdditiveExperimentConfig):
        super().__init__(reference_game.config, reference_game.targets)
        self.reference_game = reference_game
        self.experiment_config = experiment_config
        self.initial_distribution = reference_game.initial_distribution
        self.transition_matrix = reference_game.transition_matrix
        self.stationary_distribution = reference_game.stationary_distribution
        self.agent_influences = reference_game.agent_influences
        self.walk_structure = reference_game.walk_structure
        self.walk_unitary = reference_game.walk_structure.walk_unitary

    def _normalize_state(self, psi: torch.Tensor) -> torch.Tensor:
        norm = torch.clamp(torch.linalg.norm(psi), min=1e-12)
        return psi / norm

    def _prepare_signal(self, projected: torch.Tensor) -> torch.Tensor:
        source_distribution = self.reference_game._source_distribution(projected)
        # Similarity-transform the source distribution into the reversible-chain signal
        # basis so one-step evolution is induced by the single-step transition operator,
        # not by a precomputed terminal-state oracle.
        return source_distribution / torch.clamp(self.walk_structure.sqrt_stationary, min=1e-12)

    def _fast_forward_signal(self, signal: torch.Tensor) -> torch.Tensor:
        steps = max(int(self.config.mixing_depth), 0)
        if steps == 0:
            return signal
        # This is a classical simulator for applying a polynomial/spectral function of the
        # single-step discriminant derived from P. It is intentionally not an oracle that
        # maps actions directly to the terminal distribution.
        coeffs = self.walk_structure.eigenvectors.T @ signal
        evolved = self.walk_structure.eigenvectors @ ((self.walk_structure.eigenvalues**steps) * coeffs)
        return evolved.to(dtype=self.real_dtype)

    def _decode_distribution(self, signal: torch.Tensor) -> torch.Tensor:
        decoded = self.walk_structure.sqrt_stationary * signal
        decoded = torch.clamp(decoded, min=0.0)
        return decoded / torch.clamp(decoded.sum(), min=1e-12)

    def exact_expectations_from_signal(self, signal: torch.Tensor) -> torch.Tensor:
        observable_signal = self.observables * self.walk_structure.sqrt_stationary.unsqueeze(0)
        exact = observable_signal @ signal.to(self.device, dtype=self.real_dtype)
        return torch.clamp(exact, min=0.0, max=1.0)

    def _simulate_state(self, projected: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        source_signal = self._prepare_signal(projected)
        terminal_signal = self._fast_forward_signal(source_signal)
        distribution = self._decode_distribution(terminal_signal)
        return distribution, terminal_signal

    def sample_from_distribution(self, distribution: torch.Tensor, num_draws: int, seed: int) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        draws = torch.multinomial(distribution.cpu(), num_draws, replacement=True, generator=generator)
        counts = torch.bincount(draws, minlength=self.config.state_dim).to(dtype=self.real_dtype, device=self.device)
        return counts / float(num_draws)

    def estimate_observable_expectations_from_signal(
        self,
        signal: torch.Tensor,
        num_qubits: int | None = None,
    ) -> torch.Tensor:
        exact = self.exact_expectations_from_signal(signal)
        resolved_qubits = self.experiment_config.amplitude_estimation_qubits if num_qubits is None else num_qubits
        num_qubits = max(int(resolved_qubits), 1)
        grid = 2**num_qubits
        amplitudes = torch.sqrt(torch.clamp(exact, min=0.0, max=1.0))
        angles = torch.arcsin(torch.clamp(amplitudes, min=0.0, max=1.0))
        grid_index = torch.round(grid * angles / np.pi)
        grid_index = torch.clamp(grid_index, min=0, max=grid // 2)
        estimated = torch.sin(np.pi * grid_index / grid).square()
        return torch.clamp(estimated, min=0.0, max=1.0)

    def estimate_observable_expectations(self, signal_or_distribution: torch.Tensor, num_qubits: int | None = None) -> torch.Tensor:
        tensor = signal_or_distribution.to(self.device, dtype=self.real_dtype)
        is_distribution = bool(
            torch.all(tensor >= -1e-10)
            and torch.isclose(tensor.sum(), torch.tensor(1.0, dtype=tensor.dtype, device=tensor.device), atol=1e-8)
        )
        if is_distribution:
            exact = self.observables @ (tensor / torch.clamp(tensor.sum(), min=1e-12))
            resolved_qubits = self.experiment_config.amplitude_estimation_qubits if num_qubits is None else num_qubits
            num_qubits = max(int(resolved_qubits), 1)
            grid = 2**num_qubits
            amplitudes = torch.sqrt(torch.clamp(exact, min=0.0, max=1.0))
            angles = torch.arcsin(torch.clamp(amplitudes, min=0.0, max=1.0))
            grid_index = torch.round(grid * angles / np.pi)
            grid_index = torch.clamp(grid_index, min=0, max=grid // 2)
            estimated = torch.sin(np.pi * grid_index / grid).square()
            return torch.clamp(estimated, min=0.0, max=1.0)
        return self.estimate_observable_expectations_from_signal(tensor, num_qubits=num_qubits)

    def evaluate(self, actions: torch.Tensor | np.ndarray | list[list[float]]) -> DistributionEvaluationResult:
        projected = self.project_actions(actions)
        distribution, signal = self._simulate_state(projected)
        exact_expectations = self.exact_expectations_from_signal(signal)
        return self._score_distribution(
            distribution,
            projected,
            latent_state=signal.to(dtype=self.complex_dtype),
            observable_expectations=exact_expectations,
        )

    def evaluate_with_estimation(
        self,
        actions: torch.Tensor | np.ndarray | list[list[float]],
        num_qubits: int | None = None,
    ) -> DistributionEvaluationResult:
        projected = self.project_actions(actions)
        distribution, signal = self._simulate_state(projected)
        estimated_expectations = self.estimate_observable_expectations_from_signal(signal, num_qubits=num_qubits)
        return self._score_distribution(
            distribution,
            projected,
            latent_state=signal.to(dtype=self.complex_dtype),
            observable_expectations=estimated_expectations,
        )
