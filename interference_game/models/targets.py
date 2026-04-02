from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from interference_game.config import TargetConfig


CONFLICT_TO_RHO = {
    "low": 0.85,
    "medium": 0.5,
    "high": 0.15,
}


@dataclass(slots=True)
class TargetBundle:
    targets: torch.Tensor
    overlap_matrix: np.ndarray
    rho: float
    conflict_level: str


def _random_complex_vector(size: int, rng: np.random.Generator) -> np.ndarray:
    vector = rng.standard_normal(size) + 1j * rng.standard_normal(size)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return _random_complex_vector(size, rng)
    return vector / norm


def _resolve_rho(config: TargetConfig) -> float:
    if config.rho is not None:
        return float(config.rho)
    if config.conflict_level not in CONFLICT_TO_RHO:
        raise ValueError(f"Unknown conflict level: {config.conflict_level}")
    return CONFLICT_TO_RHO[config.conflict_level]


def _orthogonal_noise(anchor: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    noise = _random_complex_vector(anchor.shape[0], rng)
    noise = noise - np.vdot(anchor, noise) * anchor
    norm = np.linalg.norm(noise)
    if norm < 1e-12:
        return _orthogonal_noise(anchor, rng)
    return noise / norm


def generate_targets(config: TargetConfig, num_agents: int, state_dim: int) -> TargetBundle:
    rng = np.random.default_rng(config.seed)
    rho = _resolve_rho(config)
    anchor = _random_complex_vector(state_dim, rng)

    if config.shared_target:
        targets = np.repeat(anchor[None, :], num_agents, axis=0)
    else:
        target_list = []
        for _ in range(num_agents):
            noise = _orthogonal_noise(anchor, rng)
            target = np.sqrt(rho) * anchor + np.sqrt(1.0 - rho) * noise
            target = target / np.linalg.norm(target)
            target_list.append(target)
        targets = np.stack(target_list, axis=0)

    overlap = np.abs(targets @ np.conjugate(targets.T)) ** 2
    tensor = torch.tensor(targets, dtype=torch.complex128)
    return TargetBundle(targets=tensor, overlap_matrix=overlap, rho=rho, conflict_level=config.conflict_level)
