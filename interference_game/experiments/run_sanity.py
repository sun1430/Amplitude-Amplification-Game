from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from interference_game.experiments.common import load_cases, output_dir
from interference_game.experiments.plot_sanity import plot_results
from interference_game.models.exact_game import ExactInterferenceGame
from interference_game.models.targets import generate_targets
from interference_game.utils.io import write_frame, write_text
from interference_game.utils.metrics import random_feasible_actions


def run_from_config(config_path: str | Path) -> Path:
    case = load_cases(config_path)[0]
    results_dir = output_dir(case.experiment_config.output_root)

    reference_actions = random_feasible_actions(
        num_samples=1,
        num_agents=case.model_config.num_agents,
        state_dim=case.model_config.state_dim,
        budgets=case.model_config.budgets(),
        seed=case.experiment_config.seed,
    )[0]
    exact = case.exact_game.evaluate(reference_actions, return_intermediate=True)
    normalization = pd.DataFrame(
        {
            "step": list(range(len(exact.intermediate_states or []))),
            "norm": [float(torch.linalg.norm(state).item()) for state in exact.intermediate_states or []],
        }
    )
    write_frame(results_dir / "normalization.csv", normalization)

    small_action = reference_actions * 0.5
    large_action = reference_actions
    utility_small = case.exact_game.evaluate(small_action).utilities
    utility_large = case.exact_game.evaluate(large_action).utilities

    l0_model = replace(case.model_config, mixing_depth=0)
    l0_game = ExactInterferenceGame(l0_model, case.target_bundle.targets)
    l0_eval = l0_game.evaluate(reference_actions)
    l0_nomixing = l0_game.evaluate_mode(reference_actions, mode="no_mixing")

    shared_target_config = replace(case.target_config, shared_target=True)
    shared_targets = generate_targets(shared_target_config, case.model_config.num_agents, case.model_config.state_dim)
    shared_game = ExactInterferenceGame(case.model_config, shared_targets.targets)
    shared_eval = shared_game.evaluate(reference_actions)

    checks = pd.DataFrame(
        [
            {"check": "normalization_close_to_one", "passed": bool(np.allclose(normalization["norm"], 1.0, atol=1e-8))},
            {"check": "cost_term_monotonic", "passed": bool(torch.all(case.exact_game.evaluate(large_action).costs >= case.exact_game.evaluate(small_action).costs).item())},
            {"check": "no_mixing_matches_l0", "passed": bool(torch.allclose(l0_eval.psi, l0_nomixing.psi, atol=1e-10))},
            {"check": "shared_target_equal_fidelity", "passed": bool(torch.allclose(shared_eval.fidelities, shared_eval.fidelities[0].expand_as(shared_eval.fidelities), atol=1e-10))},
            {"check": "larger_action_changes_utility", "passed": bool(not torch.allclose(utility_small, utility_large, atol=1e-8))},
        ]
    )
    write_frame(results_dir / "checks.csv", checks)
    passed = bool(checks["passed"].all())
    note_lines = [
        f"Sanity checks for scenario '{case.scenario_name}'",
        f"All checks passed: {passed}",
        "",
        "Checks:",
    ]
    note_lines.extend(f"- {row.check}: {row.passed}" for row in checks.itertuples())
    write_text(results_dir / "sanity_note.txt", "\n".join(note_lines) + "\n")

    deltas = np.linspace(-case.experiment_config.perturbation_delta, case.experiment_config.perturbation_delta, case.experiment_config.perturbation_points)
    perturbation_rows = []
    for delta in deltas:
        perturbed = reference_actions.clone()
        perturbed[0, 0] += float(delta)
        result = case.exact_game.evaluate(perturbed)
        for agent_idx, utility in enumerate(result.utilities.tolist()):
            perturbation_rows.append({"delta": float(delta), "agent_idx": agent_idx, "utility": utility})
    write_frame(results_dir / "perturbation.csv", pd.DataFrame(perturbation_rows))

    plot_results(results_dir)
    return results_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_from_config(args.config)


if __name__ == "__main__":
    main()
