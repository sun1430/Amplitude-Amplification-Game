from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch

from interference_game.additive.experiments.common import load_cases, output_dir
from interference_game.utils.io import write_frame
from interference_game.utils.metrics import random_feasible_actions


def _utility_from_expectations(game, expectations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    projected = game.project_actions(actions)
    costs = game.lambdas * projected.square().sum(dim=1)
    if game.config.use_competitive_extension and game.config.num_agents > 1:
        penalties = []
        for agent_idx in range(game.config.num_agents):
            others = torch.cat((expectations[:agent_idx], expectations[agent_idx + 1 :]))
            penalties.append(game.gammas[agent_idx] * others.mean())
        competitive_penalties = torch.stack(penalties)
    else:
        competitive_penalties = torch.zeros_like(expectations)
    return expectations - costs - competitive_penalties


def run_from_config(config_path: str | Path) -> Path:
    raw_rows: list[dict[str, float | int | str]] = []
    cases = load_cases(config_path)
    results_dir = output_dir(cases[0].experiment_config.output_root)

    for case in cases:
        budgets = sorted(set(int(value) for value in case.experiment_config.estimation_budgets if int(value) > 0))
        samples = random_feasible_actions(
            num_samples=case.experiment_config.num_profiles,
            num_agents=case.model_config.num_agents,
            state_dim=case.model_config.state_dim,
            budgets=case.model_config.budgets(),
            seed=case.experiment_config.seed + case.scenario_index,
        )

        for sample_idx, joint_action in enumerate(samples):
            gt_result = case.ground_truth_game.evaluate(joint_action)
            exact_expectations = gt_result.observable_expectations
            assert exact_expectations is not None

            quantum_distribution = case.quantum_game.evaluate(joint_action).influence_distribution
            for query_budget in budgets:
                num_qubits = max(int(round(math.log2(query_budget))), 1)
                ae_expectations = case.quantum_game.estimate_observable_expectations(quantum_distribution, num_qubits=num_qubits)
                ae_utilities = _utility_from_expectations(case.ground_truth_game, ae_expectations, joint_action)
                raw_rows.append(
                    {
                        **case.metadata(),
                        "sample_idx": sample_idx,
                        "method": "amplitude_estimation",
                        "query_budget": query_budget,
                        "observable_error": float(torch.mean(torch.abs(ae_expectations - exact_expectations)).item()),
                        "utility_error": float(torch.mean(torch.abs(ae_utilities - gt_result.utilities)).item()),
                    }
                )

                mc_observable_errors = []
                mc_utility_errors = []
                for rep in range(case.experiment_config.mc_repetitions):
                    sampled_distribution = case.quantum_game.sample_from_distribution(
                        quantum_distribution,
                        num_draws=query_budget,
                        seed=case.experiment_config.seed + case.scenario_index * 10000 + sample_idx * 100 + query_budget + rep,
                    )
                    mc_expectations = case.ground_truth_game.observables @ sampled_distribution
                    mc_utilities = _utility_from_expectations(case.ground_truth_game, mc_expectations, joint_action)
                    mc_observable_errors.append(float(torch.mean(torch.abs(mc_expectations - exact_expectations)).item()))
                    mc_utility_errors.append(float(torch.mean(torch.abs(mc_utilities - gt_result.utilities)).item()))

                raw_rows.append(
                    {
                        **case.metadata(),
                        "sample_idx": sample_idx,
                        "method": "monte_carlo",
                        "query_budget": query_budget,
                        "observable_error": float(sum(mc_observable_errors) / len(mc_observable_errors)),
                        "utility_error": float(sum(mc_utility_errors) / len(mc_utility_errors)),
                    }
                )

    raw = pd.DataFrame(raw_rows)
    summary = (
        raw.groupby(["scenario", "method", "query_budget"], dropna=False)[["observable_error", "utility_error"]]
        .mean()
        .reset_index()
        .sort_values(["scenario", "method", "query_budget"])
        .reset_index(drop=True)
    )
    aggregate = (
        raw.groupby(["method", "query_budget"], dropna=False)[["observable_error", "utility_error"]]
        .mean()
        .reset_index()
        .sort_values(["method", "query_budget"])
        .reset_index(drop=True)
    )
    write_frame(results_dir / "raw.csv", raw)
    write_frame(results_dir / "summary.csv", summary)
    write_frame(results_dir / "aggregate.csv", aggregate)
    return results_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_from_config(args.config)


if __name__ == "__main__":
    main()
