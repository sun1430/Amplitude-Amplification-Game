from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from interference_game.config import ModelConfig


ForwardMode = Literal["exact", "no_mixing", "aggregate", "mean_field"]


@dataclass(slots=True)
class EvaluationResult:
    psi: torch.Tensor
    outcome_distribution: torch.Tensor
    fidelities: torch.Tensor
    utilities: torch.Tensor
    costs: torch.Tensor
    competitive_penalties: torch.Tensor
    intermediate_states: list[torch.Tensor] | None = None


def _complex_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "complex64":
        return torch.complex64
    return torch.complex128


def _sample_unitary(state_dim: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((state_dim, state_dim)) + 1j * rng.standard_normal((state_dim, state_dim))
    tensor = torch.tensor(matrix, dtype=torch.complex128)
    q, r = torch.linalg.qr(tensor)
    diagonal = torch.diagonal(r)
    phases = diagonal / torch.clamp(diagonal.abs(), min=1e-12)
    return q * phases.conj().unsqueeze(0)


class ExactInterferenceGame:
    def __init__(self, config: ModelConfig, targets: torch.Tensor):
        self.config = config
        self.device = torch.device(config.device)
        self.real_dtype = torch.float64
        self.complex_dtype = _complex_dtype(config.dtype)
        self.targets = targets.to(self.device, dtype=self.complex_dtype)
        self.budgets = torch.tensor(config.budgets(), dtype=self.real_dtype, device=self.device)
        self.lambdas = torch.tensor(config.lambda_list(), dtype=self.real_dtype, device=self.device)
        self.gammas = torch.tensor(config.gamma_list(), dtype=self.real_dtype, device=self.device)
        self.initial_state = self._build_initial_state()
        self.mixers = self._build_mixers()

    def _build_initial_state(self) -> torch.Tensor:
        amplitude = 1.0 / np.sqrt(self.config.state_dim)
        state = torch.full(
            (self.config.state_dim,),
            complex(amplitude, 0.0),
            dtype=self.complex_dtype,
            device=self.device,
        )
        return state

    def _build_mixers(self) -> list[torch.Tensor]:
        return [
            _sample_unitary(self.config.state_dim, self.config.mixer_seed + offset).to(self.device, dtype=self.complex_dtype)
            for offset in range(self.config.mixing_depth)
        ]

    def _as_action_tensor(self, actions: torch.Tensor | np.ndarray | list[list[float]]) -> torch.Tensor:
        tensor = torch.as_tensor(actions, dtype=self.real_dtype, device=self.device)
        if tensor.shape != (self.config.num_agents, self.config.state_dim):
            raise ValueError(
                f"Actions must have shape {(self.config.num_agents, self.config.state_dim)}, got {tuple(tensor.shape)}."
            )
        return tensor

    def wrap_phases(self, actions: torch.Tensor) -> torch.Tensor:
        return torch.remainder(actions + np.pi, 2.0 * np.pi) - np.pi

    def project_actions(self, actions: torch.Tensor | np.ndarray | list[list[float]]) -> torch.Tensor:
        tensor = self._as_action_tensor(actions).clone()
        norms = torch.linalg.norm(tensor, dim=1)
        radii = torch.sqrt(self.budgets)
        scales = torch.minimum(torch.ones_like(norms), radii / torch.clamp(norms, min=1e-12))
        tensor = tensor * scales.unsqueeze(1)
        return self.wrap_phases(tensor)

    def _apply_phase(self, state: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        return state * torch.exp(1j * theta.to(self.complex_dtype))

    def _simulate_state(self, actions: torch.Tensor, mode: ForwardMode, return_intermediate: bool = False) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        state = self.initial_state.clone()
        history: list[torch.Tensor] | None = [state.clone()] if return_intermediate else None
        rounds = max(self.config.mixing_depth, 1)
        per_round_actions = actions / rounds

        if mode == "aggregate":
            phase_template = per_round_actions.mean(dim=0, keepdim=True)
        elif mode == "mean_field":
            phase_template = per_round_actions.mean(dim=0, keepdim=True).repeat(self.config.num_agents, 1)
        else:
            phase_template = per_round_actions

        for round_idx in range(rounds):
            if mode == "aggregate":
                state = self._apply_phase(state, phase_template[0])
                if self.config.mixing_depth > 0:
                    state = self.mixers[round_idx] @ state
                if history is not None:
                    history.append(state.clone())
                continue

            for agent_idx in range(self.config.num_agents):
                state = self._apply_phase(state, phase_template[agent_idx])
                if mode != "no_mixing" and self.config.mixing_depth > 0:
                    state = self.mixers[round_idx] @ state
                if history is not None:
                    history.append(state.clone())

        return state, history

    def _fidelities_to_utilities(self, fidelities: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        costs = self.lambdas * actions.pow(2).sum(dim=1)
        if self.config.use_competitive_extension and self.config.num_agents > 1:
            penalty_terms = []
            for agent_idx in range(self.config.num_agents):
                others = torch.cat((fidelities[:agent_idx], fidelities[agent_idx + 1 :]))
                penalty_terms.append(self.gammas[agent_idx] * others.mean())
            penalties = torch.stack(penalty_terms)
        else:
            penalties = torch.zeros_like(fidelities)
        utilities = fidelities - costs - penalties
        return costs, penalties, utilities

    def evaluate_mode(
        self,
        actions: torch.Tensor | np.ndarray | list[list[float]],
        mode: ForwardMode = "exact",
        return_intermediate: bool = False,
    ) -> EvaluationResult:
        projected = self.project_actions(actions)
        psi, history = self._simulate_state(projected, mode=mode, return_intermediate=return_intermediate)
        probabilities = psi.abs().pow(2).real
        fidelities = torch.abs(self.targets.conj() @ psi) ** 2
        costs, competitive_penalties, utilities = self._fidelities_to_utilities(fidelities.real, projected)
        return EvaluationResult(
            psi=psi,
            outcome_distribution=probabilities,
            fidelities=fidelities.real,
            utilities=utilities.real,
            costs=costs.real,
            competitive_penalties=competitive_penalties.real,
            intermediate_states=history,
        )

    def evaluate(
        self,
        actions: torch.Tensor | np.ndarray | list[list[float]],
        return_intermediate: bool = False,
    ) -> EvaluationResult:
        return self.evaluate_mode(actions, mode="exact", return_intermediate=return_intermediate)

    def evaluate_sampling(
        self,
        actions: torch.Tensor | np.ndarray | list[list[float]],
        num_draws: int,
        seed: int,
    ) -> EvaluationResult:
        projected = self.project_actions(actions)
        exact = self.evaluate(projected, return_intermediate=False)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        draws = torch.multinomial(exact.outcome_distribution.cpu(), num_draws, replacement=True, generator=generator)
        counts = torch.bincount(draws, minlength=self.config.state_dim).to(dtype=self.real_dtype, device=self.device)
        sampled_distribution = counts / float(num_draws)
        sample_probabilities = torch.clamp(exact.outcome_distribution[draws.to(self.device)], min=1e-12)
        sampled_psi = exact.psi[draws.to(self.device)]
        target_samples = self.targets[:, draws.to(self.device)]
        overlap_estimates = (target_samples.conj() * sampled_psi.unsqueeze(0) / sample_probabilities.unsqueeze(0).to(self.complex_dtype)).mean(dim=1)
        fidelities = overlap_estimates.abs().pow(2).real
        costs, penalties, utilities = self._fidelities_to_utilities(fidelities, projected)
        return EvaluationResult(
            psi=exact.psi,
            outcome_distribution=sampled_distribution,
            fidelities=fidelities.real,
            utilities=utilities.real,
            costs=costs.real,
            competitive_penalties=penalties.real,
            intermediate_states=None,
        )
