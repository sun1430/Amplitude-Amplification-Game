from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from interference_game.additive.experiments.common import load_cases, output_dir
from interference_game.utils.io import write_frame
from interference_game.utils.metrics import build_candidate_actions, random_feasible_actions


def run_from_config(config_path: str | Path) -> Path:
    raw_rows: list[dict[str, float | int | str]] = []
    cases = load_cases(config_path)
    results_dir = output_dir(cases[0].experiment_config.output_root)

    for case in cases:
        samples = random_feasible_actions(
            num_samples=case.experiment_config.num_profiles,
            num_agents=case.model_config.num_agents,
            state_dim=case.model_config.state_dim,
            budgets=case.model_config.budgets(),
            seed=case.experiment_config.seed + case.scenario_index,
        )
        exact_game = case.ground_truth_game
        model_map = case.model_map()

        for sample_idx, joint_action in enumerate(samples):
            projected = exact_game.project_actions(joint_action)
            for agent_idx in range(case.model_config.num_agents):
                candidates = build_candidate_actions(
                    num_candidates=case.experiment_config.max_regret_candidates,
                    state_dim=case.model_config.state_dim,
                    budget=case.model_config.budgets()[agent_idx],
                    seed=case.experiment_config.seed + case.scenario_index * 1000 + sample_idx * 100 + agent_idx,
                ).to(projected.device, dtype=projected.dtype)
                exact_scores = []
                model_scores = {name: [] for name in model_map}
                for candidate in candidates:
                    candidate_profile = projected.clone()
                    candidate_profile[agent_idx] = candidate
                    exact_scores.append(float(exact_game.evaluate(candidate_profile).utilities[agent_idx].item()))
                    for model_name, model in model_map.items():
                        model_scores[model_name].append(float(model.evaluate(candidate_profile).utilities[agent_idx].item()))

                exact_best_value = max(exact_scores)
                exact_best_index = int(torch.tensor(exact_scores).argmax().item())
                for model_name, scores in model_scores.items():
                    model_index = int(torch.tensor(scores).argmax().item())
                    chosen_exact_value = exact_scores[model_index]
                    raw_rows.append(
                        {
                            **case.metadata(),
                            "sample_idx": sample_idx,
                            "agent_idx": agent_idx,
                            "model_name": model_name,
                            "regret": exact_best_value - chosen_exact_value,
                            "exact_best_index": exact_best_index,
                            "model_best_index": model_index,
                        }
                    )

    raw = pd.DataFrame(raw_rows)
    summary = (
        raw.groupby(["scenario", "model_name"], dropna=False)["regret"]
        .mean()
        .reset_index()
        .rename(columns={"regret": "mean_regret"})
        .sort_values(["scenario", "model_name"])
        .reset_index(drop=True)
    )
    write_frame(results_dir / "raw.csv", raw)
    write_frame(results_dir / "summary.csv", summary)
    return results_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_from_config(args.config)


if __name__ == "__main__":
    main()
