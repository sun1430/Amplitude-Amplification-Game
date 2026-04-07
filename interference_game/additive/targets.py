from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from interference_game.config import TargetConfig
from interference_game.models.targets import CONFLICT_TO_RHO


@dataclass(slots=True)
class DistributionTargetBundle:
    targets: torch.Tensor
    overlap_matrix: np.ndarray
    rho: float
    conflict_level: str


def _resolve_rho(config: TargetConfig) -> float:
    if config.rho is not None:
        return float(config.rho)
    if config.conflict_level not in CONFLICT_TO_RHO:
        raise ValueError(f"Unknown conflict level: {config.conflict_level}")
    return CONFLICT_TO_RHO[config.conflict_level]


def _random_distribution(size: int, rng: np.random.Generator) -> np.ndarray:
    sample = rng.gamma(shape=1.5, scale=1.0, size=size)
    total = sample.sum()
    if total <= 0:
        return _random_distribution(size, rng)
    return sample / total


def generate_distribution_targets(config: TargetConfig, num_agents: int, state_dim: int) -> DistributionTargetBundle:
    rng = np.random.default_rng(config.seed)
    rho = _resolve_rho(config)
    anchor = _random_distribution(state_dim, rng)

    if config.shared_target:
        targets = np.repeat(anchor[None, :], num_agents, axis=0)
    else:
        target_list = []
        for _ in range(num_agents):
            noise = _random_distribution(state_dim, rng)
            target = rho * anchor + (1.0 - rho) * noise
            target = target / np.clip(target.sum(), 1e-12, None)
            target_list.append(target)
        targets = np.stack(target_list, axis=0)

    overlap = targets @ targets.T
    tensor = torch.tensor(targets, dtype=torch.float64)
    return DistributionTargetBundle(targets=tensor, overlap_matrix=overlap, rho=rho, conflict_level=config.conflict_level)
