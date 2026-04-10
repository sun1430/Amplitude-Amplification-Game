from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from interference_game.additive.config import AdditiveExperimentConfig, build_additive_configs, load_additive_yaml
from interference_game.additive.markov_special_case import QuantumMarkovEncodedGame, ReversibleSparseMarkovGame
from interference_game.additive.surrogate import ResidualMLPSurrogate
from interference_game.additive.targets import DistributionTargetBundle, generate_distribution_targets
from interference_game.config import EquilibriumConfig, ModelConfig, TargetConfig
from interference_game.utils.io import ensure_dir


@dataclass(slots=True)
class MarkovExperimentCase:
    scenario_name: str
    scenario_index: int
    experiment_config: AdditiveExperimentConfig
    model_config: ModelConfig
    target_config: TargetConfig
    equilibrium_config: EquilibriumConfig
    target_bundle: DistributionTargetBundle
    ground_truth_game: ReversibleSparseMarkovGame
    quantum_game: QuantumMarkovEncodedGame
    sota_model: ResidualMLPSurrogate

    def metadata(self) -> dict[str, Any]:
        overlap = self.target_bundle.overlap_matrix
        budgets = self.model_config.budgets()
        return {
            "scenario": self.scenario_name,
            "num_agents": self.model_config.num_agents,
            "state_dim": self.model_config.state_dim,
            "mixing_depth": self.model_config.mixing_depth,
            "activation_name": self.model_config.activation_name(),
            "markov_graph_degree": self.model_config.markov_graph_degree,
            "markov_inertia": self.model_config.markov_inertia,
            "mean_action_budget": float(sum(budgets) / len(budgets)),
            "conflict_level": self.target_bundle.conflict_level,
            "rho": self.target_bundle.rho,
            "mean_target_overlap": float(overlap.mean()),
        }

    def model_map(self) -> dict[str, Any]:
        return {
            "ground_truth": self.ground_truth_game,
            "quantum_encoded": self.quantum_game,
            "residual_mlp": self.sota_model,
        }


def output_dir(base_root: str | Path, scenario_name: str | None = None) -> Path:
    root = ensure_dir(base_root)
    if scenario_name is None:
        return root
    return ensure_dir(root / scenario_name)


def build_case(raw_config: dict[str, Any], scenario_override: dict[str, Any], scenario_index: int) -> MarkovExperimentCase:
    experiment_config, model_config, target_config, equilibrium_config = build_additive_configs(raw_config, scenario_override)
    scenario_name = scenario_override.get("name", experiment_config.name)
    target_bundle = generate_distribution_targets(target_config, model_config.num_agents, model_config.state_dim)
    ground_truth = ReversibleSparseMarkovGame(model_config, target_bundle.targets)
    quantum = QuantumMarkovEncodedGame(ground_truth, experiment_config)
    surrogate = ResidualMLPSurrogate(ground_truth, experiment_config)
    artifact_name = (
        f"{scenario_name}__depth{model_config.mixing_depth}"
        f"__dim{model_config.state_dim}__{model_config.activation_slug()}"
        f"__h{experiment_config.hidden_dim}__tr{experiment_config.train_samples}"
    )
    artifact_dir = output_dir(experiment_config.output_root, artifact_name) / "_artifacts" / "residual_mlp"
    surrogate.fit(artifact_dir, seed=experiment_config.seed + scenario_index)
    return MarkovExperimentCase(
        scenario_name=scenario_name,
        scenario_index=scenario_index,
        experiment_config=experiment_config,
        model_config=model_config,
        target_config=target_config,
        equilibrium_config=equilibrium_config,
        target_bundle=target_bundle,
        ground_truth_game=ground_truth,
        quantum_game=quantum,
        sota_model=surrogate,
    )


def load_cases(config_path: str | Path) -> list[MarkovExperimentCase]:
    raw, scenarios = load_additive_yaml(config_path)
    return [build_case(raw, scenario, idx) for idx, scenario in enumerate(scenarios)]
