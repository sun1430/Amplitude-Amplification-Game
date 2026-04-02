from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import pandas as pd
import torch

from interference_game.config import EquilibriumConfig


@dataclass(slots=True)
class EquilibriumSummary:
    records: pd.DataFrame
    pure_nash: pd.DataFrame
    epsilon_nash: pd.DataFrame


def build_individual_action_space(state_dim: int, phase_grid: list[float]) -> list[torch.Tensor]:
    return [torch.tensor(values, dtype=torch.float64) for values in product(phase_grid, repeat=state_dim)]


def enumerate_equilibria(game, equilibrium_config: EquilibriumConfig) -> EquilibriumSummary:
    game_config = game.config if hasattr(game, "config") else game.exact_game.config
    phase_grid = equilibrium_config.phase_grid()
    individual_actions = build_individual_action_space(game_config.state_dim, phase_grid)
    num_actions = len(individual_actions)
    total_profiles = num_actions ** game_config.num_agents
    if total_profiles > equilibrium_config.max_profiles:
        raise ValueError(
            f"Joint action space has {total_profiles} profiles, exceeding max_profiles={equilibrium_config.max_profiles}."
        )

    cache: dict[tuple[int, ...], torch.Tensor] = {}
    records: list[dict[str, object]] = []

    for profile_indices in product(range(num_actions), repeat=game_config.num_agents):
        joint_action = torch.stack([individual_actions[index] for index in profile_indices], dim=0)
        utility = game.evaluate(joint_action).utilities.detach().cpu()
        cache[profile_indices] = utility

    for profile_indices, utility in cache.items():
        regrets = []
        for agent_idx in range(game_config.num_agents):
            best_deviation_utility = utility[agent_idx]
            for alternative_idx in range(num_actions):
                deviated_profile = list(profile_indices)
                deviated_profile[agent_idx] = alternative_idx
                candidate_utility = cache[tuple(deviated_profile)][agent_idx]
                if candidate_utility > best_deviation_utility:
                    best_deviation_utility = candidate_utility
            regrets.append(float((best_deviation_utility - utility[agent_idx]).item()))

        row = {
            "profile_key": "|".join(str(index) for index in profile_indices),
            "profile_indices": list(profile_indices),
            "max_regret": max(regrets),
            "is_pure_nash": max(regrets) <= 1e-12,
            "is_epsilon_nash": max(regrets) <= equilibrium_config.epsilon,
        }
        for agent_idx, value in enumerate(utility.tolist()):
            row[f"utility_agent_{agent_idx}"] = value
        for agent_idx, value in enumerate(regrets):
            row[f"regret_agent_{agent_idx}"] = value
        records.append(row)

    frame = pd.DataFrame(records).sort_values(["max_regret", "profile_key"]).reset_index(drop=True)
    pure = frame[frame["is_pure_nash"]].reset_index(drop=True)
    epsilon = frame[frame["is_epsilon_nash"]].reset_index(drop=True)
    return EquilibriumSummary(records=frame, pure_nash=pure, epsilon_nash=epsilon)


def compare_equilibrium_sets(exact_summary: EquilibriumSummary, approx_summary: EquilibriumSummary) -> dict[str, float]:
    exact_pure = set(exact_summary.pure_nash["profile_key"].tolist())
    approx_pure = set(approx_summary.pure_nash["profile_key"].tolist())
    exact_eps = set(exact_summary.epsilon_nash["profile_key"].tolist())
    approx_eps = set(approx_summary.epsilon_nash["profile_key"].tolist())

    pure_overlap = len(exact_pure & approx_pure)
    epsilon_overlap = len(exact_eps & approx_eps)
    pure_union = max(len(exact_pure | approx_pure), 1)
    epsilon_union = max(len(exact_eps | approx_eps), 1)

    approx_regrets_in_exact = exact_summary.records[
        exact_summary.records["profile_key"].isin(approx_summary.epsilon_nash["profile_key"])
    ]
    mean_rechecked_regret = float(approx_regrets_in_exact["max_regret"].mean()) if not approx_regrets_in_exact.empty else 0.0

    return {
        "exact_pure_count": float(len(exact_pure)),
        "approx_pure_count": float(len(approx_pure)),
        "exact_epsilon_count": float(len(exact_eps)),
        "approx_epsilon_count": float(len(approx_eps)),
        "pure_overlap": float(pure_overlap),
        "epsilon_overlap": float(epsilon_overlap),
        "pure_jaccard": pure_overlap / pure_union,
        "epsilon_jaccard": epsilon_overlap / epsilon_union,
        "mean_exact_regret_of_approx_epsilon": mean_rechecked_regret,
    }
