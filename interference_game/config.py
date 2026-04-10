from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _ensure_list(value: Any, length: int, name: str) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)] * length
    if isinstance(value, list) and len(value) == length:
        return [float(item) for item in value]
    raise ValueError(f"{name} must be a scalar or a list of length {length}.")


def _normalize_phase_grid(values: list[float] | None) -> list[float]:
    if not values:
        return [0.0, 1.5707963267948966, 3.141592653589793, 4.71238898038469]
    return [float(value) for value in values]


@dataclass(slots=True)
class ModelConfig:
    num_agents: int
    state_dim: int
    mixing_depth: int
    action_budget: float | list[float]
    lambdas: float | list[float]
    gammas: float | list[float] = 0.0
    mixer_seed: int = 0
    use_competitive_extension: bool = False
    device: str = "cpu"
    dtype: str = "complex128"
    activation_family: str = "entmax"
    activation_alpha: float = 1.5
    activation_beta: float = 1.0
    activation_tau: float = 0.0
    activation_gamma: float = 8.0
    activation_iterations: int = 50
    markov_graph_degree: int = 2
    markov_self_loop: float = 0.2
    markov_inertia: float = 0.3

    def budgets(self) -> list[float]:
        return _ensure_list(self.action_budget, self.num_agents, "action_budget")

    def lambda_list(self) -> list[float]:
        return _ensure_list(self.lambdas, self.num_agents, "lambdas")

    def gamma_list(self) -> list[float]:
        return _ensure_list(self.gammas, self.num_agents, "gammas")

    def activation_name(self) -> str:
        family = self.activation_family.lower()
        if family == "softmax":
            return f"softmax(beta={self.activation_beta:.2f})"
        if family == "sparsemax":
            return "sparsemax"
        if family in {"bounded_confidence", "smooth_bounded_confidence"}:
            return f"bounded_confidence(gamma={self.activation_gamma:.2f}, tau={self.activation_tau:.2f})"
        return f"entmax(alpha={self.activation_alpha:.2f})"

    def activation_slug(self) -> str:
        family = self.activation_family.lower()
        if family == "softmax":
            suffix = f"b{self.activation_beta:.2f}"
        elif family == "sparsemax":
            suffix = None
        elif family in {"bounded_confidence", "smooth_bounded_confidence"}:
            suffix = f"g{self.activation_gamma:.2f}_t{self.activation_tau:.2f}"
        else:
            suffix = f"a{self.activation_alpha:.2f}"
        raw = family if suffix is None else f"{family}_{suffix}"
        return raw.replace("-", "_").replace(".", "p").replace("(", "_").replace(")", "").replace("=", "_")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        return cls(**data)


@dataclass(slots=True)
class TargetConfig:
    seed: int = 0
    conflict_level: str = "medium"
    rho: float | None = None
    shared_target: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TargetConfig":
        return cls(**data)


@dataclass(slots=True)
class EquilibriumConfig:
    discrete_phases: list[float] = field(default_factory=list)
    epsilon: float = 1e-3
    max_profiles: int = 500000

    def phase_grid(self) -> list[float]:
        return _normalize_phase_grid(self.discrete_phases)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EquilibriumConfig":
        return cls(**data)


@dataclass(slots=True)
class DynamicsConfig:
    method: str = "projected_gradient"
    step_size: float = 0.2
    max_steps: int = 50
    tolerance: float = 1e-4
    stable_window: int = 5
    num_initializations: int = 8
    br_restarts: int = 3
    br_inner_steps: int = 25
    extragradient_lookahead: float = 1.0
    cycle_window: int = 8
    cycle_tolerance: float = 1e-3

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DynamicsConfig":
        return cls(**data)


@dataclass(slots=True)
class ExperimentConfig:
    name: str
    output_root: str
    seed: int = 0
    num_profiles: int = 32
    sampling_draws: int = 256
    best_response_candidates: int = 12
    perturbation_delta: float = 0.2
    perturbation_points: int = 25
    save_format: str = "csv"
    methods: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        return cls(**data)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def merge_nested(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_nested(merged[key], value)
        else:
            merged[key] = value
    return merged


def get_scenarios(config: dict[str, Any]) -> list[dict[str, Any]]:
    scenarios = config.get("scenarios")
    if not scenarios:
        return [{"name": config["experiment"]["name"]}]
    return scenarios


def build_configs(config: dict[str, Any], scenario: dict[str, Any]) -> tuple[ExperimentConfig, ModelConfig, TargetConfig, EquilibriumConfig, DynamicsConfig]:
    merged = merge_nested(config, scenario)
    experiment = ExperimentConfig.from_dict(merged.get("experiment", {}))
    model = ModelConfig.from_dict(merged.get("model", {}))
    target = TargetConfig.from_dict(merged.get("target", {}))
    equilibrium = EquilibriumConfig.from_dict(merged.get("equilibrium", {}))
    dynamics = DynamicsConfig.from_dict(merged.get("dynamics", {}))
    return experiment, model, target, equilibrium, dynamics
