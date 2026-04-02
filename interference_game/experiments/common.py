from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from interference_game.config import build_configs, get_scenarios, load_yaml
from interference_game.models.baselines import build_baselines
from interference_game.models.exact_game import ExactInterferenceGame
from interference_game.models.targets import TargetBundle, generate_targets
from interference_game.utils.io import ensure_dir


@dataclass(slots=True)
class ExperimentCase:
    scenario_name: str
    scenario_index: int
    experiment_config: Any
    model_config: Any
    target_config: Any
    equilibrium_config: Any
    dynamics_config: Any
    target_bundle: TargetBundle
    exact_game: ExactInterferenceGame
    baselines: dict[str, Any]

    def metadata(self) -> dict[str, Any]:
        overlap = self.target_bundle.overlap_matrix
        budgets = self.model_config.budgets()
        return {
            "scenario": self.scenario_name,
            "num_agents": self.model_config.num_agents,
            "state_dim": self.model_config.state_dim,
            "mixing_depth": self.model_config.mixing_depth,
            "mean_action_budget": float(sum(budgets) / len(budgets)),
            "conflict_level": self.target_bundle.conflict_level,
            "rho": self.target_bundle.rho,
            "shared_target": self.target_config.shared_target,
            "competitive_extension": self.model_config.use_competitive_extension,
            "mean_target_overlap": float(overlap.mean()),
        }


def load_cases(config_path: str | Path) -> list[ExperimentCase]:
    raw = load_yaml(config_path)
    cases: list[ExperimentCase] = []
    for idx, scenario in enumerate(get_scenarios(raw)):
        experiment_config, model_config, target_config, equilibrium_config, dynamics_config = build_configs(raw, scenario)
        scenario_name = scenario.get("name", experiment_config.name)
        target_bundle = generate_targets(target_config, model_config.num_agents, model_config.state_dim)
        exact_game = ExactInterferenceGame(model_config, target_bundle.targets)
        baselines = build_baselines(exact_game, sampling_draws=experiment_config.sampling_draws, seed=experiment_config.seed + idx)
        cases.append(
            ExperimentCase(
                scenario_name=scenario_name,
                scenario_index=idx,
                experiment_config=experiment_config,
                model_config=model_config,
                target_config=target_config,
                equilibrium_config=equilibrium_config,
                dynamics_config=dynamics_config,
                target_bundle=target_bundle,
                exact_game=exact_game,
                baselines=baselines,
            )
        )
    return cases


def output_dir(base_root: str | Path, scenario_name: str | None = None) -> Path:
    root = ensure_dir(base_root)
    if scenario_name is None:
        return root
    return ensure_dir(root / scenario_name)
