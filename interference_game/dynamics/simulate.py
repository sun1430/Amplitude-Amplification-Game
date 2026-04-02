from __future__ import annotations

from dataclasses import dataclass

import torch

from interference_game.config import DynamicsConfig
from interference_game.utils.metrics import detect_cycle, random_feasible_actions, trajectory_variance


@dataclass(slots=True)
class DynamicsRunResult:
    method: str
    seed: int
    converged: bool
    cycle_detected: bool
    time_to_stability: int | None
    action_history: list[torch.Tensor]
    utility_history: list[torch.Tensor]
    final_actions: torch.Tensor
    final_utilities: torch.Tensor
    trajectory_variance: float


def _project_for_game(game, actions: torch.Tensor) -> torch.Tensor:
    if hasattr(game, "project_actions"):
        return game.project_actions(actions)
    return game.exact_game.project_actions(actions)


def _config_for_game(game):
    if hasattr(game, "config"):
        return game.config
    return game.exact_game.config


def _utility_gradient(game, actions: torch.Tensor, agent_idx: int, epsilon: float = 1e-4) -> torch.Tensor:
    supports_autograd = not getattr(game, "mode", None) == "sampling"
    if supports_autograd:
        try:
            candidate = actions.clone().detach().requires_grad_(True)
            utility = game.evaluate(candidate).utilities[agent_idx]
            utility.backward()
            gradient = candidate.grad[agent_idx]
            if gradient is not None:
                return gradient.detach()
        except RuntimeError:
            pass

    gradient = torch.zeros_like(actions[agent_idx])
    for coord_idx in range(actions.shape[1]):
        plus = actions.clone()
        minus = actions.clone()
        plus[agent_idx, coord_idx] += epsilon
        minus[agent_idx, coord_idx] -= epsilon
        plus = _project_for_game(game, plus)
        minus = _project_for_game(game, minus)
        plus_utility = game.evaluate(plus).utilities[agent_idx]
        minus_utility = game.evaluate(minus).utilities[agent_idx]
        gradient[coord_idx] = (plus_utility - minus_utility) / (2.0 * epsilon)
    return gradient.detach()


def _joint_gradient(game, actions: torch.Tensor) -> torch.Tensor:
    return torch.stack([_utility_gradient(game, actions, agent_idx) for agent_idx in range(actions.shape[0])], dim=0)


def _approximate_best_response(game, actions: torch.Tensor, agent_idx: int, config: DynamicsConfig, seed: int) -> torch.Tensor:
    game_config = _config_for_game(game)
    restarts = [actions[agent_idx].clone()]
    if config.br_restarts > 1:
        extra = random_feasible_actions(
            num_samples=config.br_restarts - 1,
            num_agents=1,
            state_dim=game_config.state_dim,
            budgets=[game_config.budgets()[agent_idx]],
            seed=seed,
        )
        restarts.extend(extra[:, 0, :].to(actions.device))

    best_theta = restarts[0].clone()
    best_utility = float("-inf")
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    for restart_idx, initial_theta in enumerate(restarts):
        theta = initial_theta.clone()
        m = torch.zeros_like(theta)
        v = torch.zeros_like(theta)
        for step_idx in range(1, config.br_inner_steps + 1):
            candidate = actions.clone()
            candidate[agent_idx] = theta
            grad = _utility_gradient(game, candidate, agent_idx)
            m = beta1 * m + (1.0 - beta1) * grad
            v = beta2 * v + (1.0 - beta2) * grad.pow(2)
            m_hat = m / (1.0 - beta1**step_idx)
            v_hat = v / (1.0 - beta2**step_idx)
            theta = theta + config.step_size * m_hat / (torch.sqrt(v_hat) + eps)
            candidate[agent_idx] = theta
            theta = _project_for_game(game, candidate)[agent_idx]

        candidate = actions.clone()
        candidate[agent_idx] = theta
        utility = float(game.evaluate(candidate).utilities[agent_idx].item())
        if utility > best_utility:
            best_utility = utility
            best_theta = theta.clone()

    return best_theta


def simulate_dynamics(game, dynamics_config: DynamicsConfig, seed: int) -> DynamicsRunResult:
    game_config = _config_for_game(game)
    initial = random_feasible_actions(
        num_samples=1,
        num_agents=game_config.num_agents,
        state_dim=game_config.state_dim,
        budgets=game_config.budgets(),
        seed=seed,
    )[0]
    actions = _project_for_game(game, initial)
    action_history = [actions.clone()]
    utility_history = [game.evaluate(actions).utilities.detach().cpu()]
    stable_counter = 0
    time_to_stability: int | None = None
    cycle_flag = False

    for step in range(1, dynamics_config.max_steps + 1):
        if dynamics_config.method == "best_response":
            updated = actions.clone()
            for agent_idx in range(game_config.num_agents):
                updated[agent_idx] = _approximate_best_response(game, updated, agent_idx, dynamics_config, seed + step + agent_idx)
            next_actions = _project_for_game(game, updated)
        elif dynamics_config.method == "extra_gradient":
            grad = _joint_gradient(game, actions)
            lookahead = _project_for_game(game, actions + dynamics_config.step_size * dynamics_config.extragradient_lookahead * grad)
            lookahead_grad = _joint_gradient(game, lookahead)
            next_actions = _project_for_game(game, actions + dynamics_config.step_size * lookahead_grad)
        else:
            grad = _joint_gradient(game, actions)
            next_actions = _project_for_game(game, actions + dynamics_config.step_size * grad)

        delta = torch.max(torch.abs(next_actions - actions)).item()
        if delta < dynamics_config.tolerance:
            stable_counter += 1
            if stable_counter >= dynamics_config.stable_window and time_to_stability is None:
                time_to_stability = step
        else:
            stable_counter = 0

        actions = next_actions
        action_history.append(actions.clone())
        utility_history.append(game.evaluate(actions).utilities.detach().cpu())
        cycle_flag = detect_cycle(action_history, dynamics_config.cycle_window, dynamics_config.cycle_tolerance)
        if cycle_flag or stable_counter >= dynamics_config.stable_window:
            break

    final_utilities = utility_history[-1]
    return DynamicsRunResult(
        method=dynamics_config.method,
        seed=seed,
        converged=stable_counter >= dynamics_config.stable_window,
        cycle_detected=cycle_flag,
        time_to_stability=time_to_stability,
        action_history=action_history,
        utility_history=utility_history,
        final_actions=action_history[-1],
        final_utilities=final_utilities,
        trajectory_variance=trajectory_variance(action_history),
    )
