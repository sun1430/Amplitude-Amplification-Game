from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from interference_game.dynamics.simulate import simulate_dynamics
from interference_game.experiments.common import load_cases, output_dir
from interference_game.experiments.plot_ablation import plot_results
from interference_game.utils.io import write_frame
from interference_game.utils.metrics import payoff_metric_record, random_feasible_actions


def run_from_config(config_path: str | Path) -> Path:
    cases = load_cases(config_path)
    results_dir = output_dir(cases[0].experiment_config.output_root)
    rows = []

    for case in cases:
        samples = random_feasible_actions(
            num_samples=case.experiment_config.num_profiles,
            num_agents=case.model_config.num_agents,
            state_dim=case.model_config.state_dim,
            budgets=case.model_config.budgets(),
            seed=case.experiment_config.seed + case.scenario_index,
        )
        aggregate_errors = []
        mean_field_errors = []
        sampling_errors = []
        for joint_action in samples:
            exact = case.exact_game.evaluate(joint_action)
            aggregate_errors.append(payoff_metric_record(exact, case.baselines["aggregate"].evaluate(joint_action))["mean_abs_utility_error"])
            mean_field_errors.append(payoff_metric_record(exact, case.baselines["mean_field"].evaluate(joint_action))["mean_abs_utility_error"])
            sampling_errors.append(payoff_metric_record(exact, case.baselines["sampling"].evaluate(joint_action))["mean_abs_utility_error"])

        dynamics_config = replace(case.dynamics_config, method="projected_gradient")
        runs = [
            simulate_dynamics(case.exact_game, dynamics_config, seed=case.experiment_config.seed + case.scenario_index * 100 + run_idx)
            for run_idx in range(case.dynamics_config.num_initializations)
        ]
        rows.append(
            {
                **case.metadata(),
                "aggregate_error": float(sum(aggregate_errors) / len(aggregate_errors)),
                "mean_field_error": float(sum(mean_field_errors) / len(mean_field_errors)),
                "sampling_error": float(sum(sampling_errors) / len(sampling_errors)),
                "cycle_rate": float(sum(run.cycle_detected for run in runs) / len(runs)),
                "convergence_rate": float(sum(run.converged for run in runs) / len(runs)),
            }
        )

    write_frame(results_dir / "summary.csv", pd.DataFrame(rows))
    plot_results(results_dir)
    return results_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_from_config(args.config)


if __name__ == "__main__":
    main()
