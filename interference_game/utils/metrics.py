from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch


@dataclass(slots=True)
class BestResponseRecord:
    agent_idx: int
    exact_index: int
    approx_index: int
    match: bool
    rank_correlation: float


def absolute_relative_error(exact: torch.Tensor, approx: torch.Tensor, eps: float = 1e-12) -> tuple[torch.Tensor, torch.Tensor]:
    absolute = torch.abs(exact - approx)
    relative = absolute / torch.clamp(torch.abs(exact), min=eps)
    return absolute, relative


def kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> float:
    p_safe = torch.clamp(p, min=eps)
    q_safe = torch.clamp(q, min=eps)
    return float(torch.sum(p_safe * torch.log(p_safe / q_safe)).item())


def l2_distance(p: torch.Tensor, q: torch.Tensor) -> float:
    return float(torch.linalg.norm(p - q).item())


def random_feasible_actions(num_samples: int, num_agents: int, state_dim: int, budgets: list[float], seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((num_samples, num_agents, state_dim))
    budgets_array = np.sqrt(np.asarray(budgets, dtype=float))
    norms = np.linalg.norm(raw, axis=2, keepdims=True)
    safe_norms = np.maximum(norms, 1e-12)
    radii = rng.random((num_samples, num_agents, 1)) ** (1.0 / state_dim)
    scaled = raw / safe_norms * radii * budgets_array.reshape(1, num_agents, 1)
    wrapped = ((scaled + np.pi) % (2.0 * np.pi)) - np.pi
    return torch.tensor(wrapped, dtype=torch.float64)


def build_candidate_actions(num_candidates: int, state_dim: int, budget: float, seed: int) -> torch.Tensor:
    samples = random_feasible_actions(
        num_samples=num_candidates,
        num_agents=1,
        state_dim=state_dim,
        budgets=[budget],
        seed=seed,
    )
    return samples[:, 0, :]


def spearman_rank_correlation(values_a: list[float], values_b: list[float]) -> float:
    array_a = np.asarray(values_a, dtype=float)
    array_b = np.asarray(values_b, dtype=float)
    if array_a.size < 2 or array_b.size < 2:
        return 1.0
    if np.allclose(array_a, array_a[0]) and np.allclose(array_b, array_b[0]):
        return 1.0
    if np.std(array_a) < 1e-12 or np.std(array_b) < 1e-12:
        return 0.0
    try:
        from scipy.stats import rankdata

        rank_a = rankdata(array_a, method="average")
        rank_b = rankdata(array_b, method="average")
    except Exception:
        rank_a = pd.Series(array_a).rank(method="average").to_numpy(dtype=float)
        rank_b = pd.Series(array_b).rank(method="average").to_numpy(dtype=float)
    corr = np.corrcoef(rank_a, rank_b)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(corr)


def best_response_preservation(exact_game, approx_game, joint_action: torch.Tensor, candidate_seed: int, num_candidates: int) -> list[BestResponseRecord]:
    records: list[BestResponseRecord] = []
    projected = exact_game.project_actions(joint_action)
    for agent_idx in range(exact_game.config.num_agents):
        candidates = build_candidate_actions(
            num_candidates=num_candidates,
            state_dim=exact_game.config.state_dim,
            budget=exact_game.config.budgets()[agent_idx],
            seed=candidate_seed + agent_idx,
        )
        exact_scores = []
        approx_scores = []
        for candidate in candidates:
            candidate_profile = projected.clone()
            candidate_profile[agent_idx] = candidate.to(projected.device)
            exact_scores.append(float(exact_game.evaluate(candidate_profile).utilities[agent_idx].item()))
            approx_scores.append(float(approx_game.evaluate(candidate_profile).utilities[agent_idx].item()))
        exact_index = int(np.argmax(exact_scores))
        approx_index = int(np.argmax(approx_scores))
        records.append(
            BestResponseRecord(
                agent_idx=agent_idx,
                exact_index=exact_index,
                approx_index=approx_index,
                match=exact_index == approx_index,
                rank_correlation=spearman_rank_correlation(exact_scores, approx_scores),
            )
        )
    return records


def payoff_metric_record(exact_result, approx_result) -> dict[str, float]:
    abs_error, rel_error = absolute_relative_error(exact_result.utilities, approx_result.utilities)
    return {
        "mean_abs_utility_error": float(abs_error.mean().item()),
        "mean_rel_utility_error": float(rel_error.mean().item()),
        "mean_abs_fidelity_error": float(torch.abs(exact_result.fidelities - approx_result.fidelities).mean().item()),
        "kl_divergence": kl_divergence(exact_result.outcome_distribution, approx_result.outcome_distribution),
        "l2_distribution_error": l2_distance(exact_result.outcome_distribution, approx_result.outcome_distribution),
    }


def summarize_frame(frame: pd.DataFrame, group_cols: list[str], value_cols: list[str]) -> pd.DataFrame:
    grouped = frame.groupby(group_cols, dropna=False)[value_cols].agg(["mean", "std"]).reset_index()
    grouped.columns = [
        "_".join([piece for piece in column if piece]).strip("_") if isinstance(column, tuple) else column
        for column in grouped.columns
    ]
    return grouped


def trajectory_variance(action_history: list[torch.Tensor]) -> float:
    if len(action_history) < 2:
        return 0.0
    stacked = torch.stack(action_history)
    return float(stacked.var(dim=0).mean().item())


def detect_cycle(action_history: list[torch.Tensor], window: int, tolerance: float) -> bool:
    if len(action_history) <= window:
        return False
    current = action_history[-1]
    for candidate in action_history[-window - 1 : -1]:
        if torch.max(torch.abs(current - candidate)).item() < tolerance:
            return True
    return False
