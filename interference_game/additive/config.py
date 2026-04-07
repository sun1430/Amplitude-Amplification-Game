from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from interference_game.config import EquilibriumConfig, ModelConfig, TargetConfig, get_scenarios, load_yaml, merge_nested


@dataclass(slots=True)
class AdditiveExperimentConfig:
    name: str
    output_root: str
    seed: int = 0
    num_profiles: int = 16
    best_response_candidates: int = 8
    train_samples: int = 128
    val_samples: int = 32
    batch_size: int = 32
    epochs: int = 40
    learning_rate: float = 5e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 6
    hidden_dim: int = 64
    max_regret_candidates: int = 8
    amplitude_estimation_qubits: int = 8
    estimation_budgets: list[int] = field(default_factory=lambda: [8, 16, 32, 64, 128, 256])
    mc_repetitions: int = 1

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AdditiveExperimentConfig":
        return cls(**data)


def build_additive_configs(
    config: dict[str, Any],
    scenario: dict[str, Any],
) -> tuple[AdditiveExperimentConfig, ModelConfig, TargetConfig, EquilibriumConfig]:
    merged = merge_nested(config, scenario)
    experiment = AdditiveExperimentConfig.from_dict(merged.get("experiment", {}))
    model = ModelConfig.from_dict(merged.get("model", {}))
    target = TargetConfig.from_dict(merged.get("target", {}))
    equilibrium = EquilibriumConfig.from_dict(merged.get("equilibrium", {}))
    return experiment, model, target, equilibrium


def load_additive_yaml(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw = load_yaml(path)
    return raw, get_scenarios(raw)
