from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from interference_game.dynamics.simulate import simulate_dynamics
from interference_game.experiments.common import load_cases, output_dir
from interference_game.experiments.plot_dynamics import plot_results
from interference_game.utils.io import write_frame


def run_from_config(config_path: str | Path) -> Path:
    cases = load_cases(config_path)
    results_dir = output_dir(cases[0].experiment_config.output_root)
    raw = []
    trajectories = []

    for case in cases:
        methods = case.experiment_config.methods
        if not methods:
            methods = ["best_response", "projected_gradient", "extra_gradient"]

        for model_name, model in {"exact": case.exact_game, **case.baselines}.items():
            for method in methods:
                for init_idx in range(case.dynamics_config.num_initializations):
                    config = replace(case.dynamics_config, method=method)
                    result = simulate_dynamics(model, config, seed=case.experiment_config.seed + case.scenario_index * 100 + init_idx)
                    raw.append(
                        {
                            **case.metadata(),
                            "model_name": model_name,
                            "method": method,
                            "seed": result.seed,
                            "converged": int(result.converged),
                            "cycle_detected": int(result.cycle_detected),
                            "time_to_stability": result.time_to_stability if result.time_to_stability is not None else -1,
                            "trajectory_variance": result.trajectory_variance,
                            "final_mean_utility": float(result.final_utilities.mean().item()),
                        }
                    )
                    for step_idx, utilities in enumerate(result.utility_history):
                        for agent_idx, utility in enumerate(utilities.tolist()):
                            trajectories.append(
                                {
                                    "scenario": case.scenario_name,
                                    "model_name": model_name,
                                    "method": method,
                                    "seed": result.seed,
                                    "step": step_idx,
                                    "agent_idx": agent_idx,
                                    "utility": utility,
                                }
                            )

    write_frame(results_dir / "summary.csv", pd.DataFrame(raw))
    write_frame(results_dir / "trajectories.csv", pd.DataFrame(trajectories))
    plot_results(results_dir)
    return results_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_from_config(args.config)


if __name__ == "__main__":
    main()
