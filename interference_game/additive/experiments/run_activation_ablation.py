from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from interference_game.additive.experiments.common import AdditiveExperimentCase, build_case, output_dir
from interference_game.config import get_scenarios, load_yaml, merge_nested
from interference_game.equilibrium.discrete_enumeration import enumerate_equilibria
from interference_game.utils.io import write_frame
from interference_game.utils.metrics import build_candidate_actions, random_feasible_actions


def _case_metadata(case: AdditiveExperimentCase, activation_label: str) -> dict[str, float | int | str]:
    return {**case.metadata(), "activation_label": activation_label}


def _distribution_entropy(distribution: torch.Tensor) -> float:
    clipped = torch.clamp(distribution, min=1e-12)
    return float((-clipped * torch.log(clipped)).sum().item())


def _support_size(distribution: torch.Tensor) -> float:
    return float(torch.count_nonzero(distribution > 1e-8).item())


def _load_ablation_cases(config_path: str | Path) -> list[tuple[str, AdditiveExperimentCase]]:
    raw = load_yaml(config_path)
    scenarios = get_scenarios(raw)
    activation_ablations = raw.get("activation_ablations") or [{"name": "default", "model": {}}]
    cases: list[tuple[str, AdditiveExperimentCase]] = []
    case_index = 0
    for scenario in scenarios:
        for activation in activation_ablations:
            override = merge_nested(scenario, {"model": activation.get("model", {})})
            cases.append((activation.get("name", f"activation_{case_index}"), build_case(raw, override, case_index)))
            case_index += 1
    return cases


def run_from_config(config_path: str | Path) -> Path:
    strategy_rows: list[dict[str, float | int | str]] = []
    regret_rows: list[dict[str, float | int | str]] = []
    epsilon_rows: list[dict[str, float | int | str]] = []
    diagnostics_rows: list[dict[str, float | int | str]] = []
    cases = _load_ablation_cases(config_path)
    results_dir = output_dir(cases[0][1].experiment_config.output_root)

    for activation_label, case in cases:
        metadata = _case_metadata(case, activation_label)
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
            gt_distribution = exact_game.evaluate(projected).influence_distribution
            diagnostics_rows.append(
                {
                    **metadata,
                    "sample_idx": sample_idx,
                    "entropy": _distribution_entropy(gt_distribution),
                    "support_size": _support_size(gt_distribution),
                }
            )
            for agent_idx in range(case.model_config.num_agents):
                candidates = build_candidate_actions(
                    num_candidates=max(case.experiment_config.best_response_candidates, case.experiment_config.max_regret_candidates),
                    state_dim=case.model_config.state_dim,
                    budget=case.model_config.budgets()[agent_idx],
                    seed=case.experiment_config.seed + case.scenario_index * 1000 + sample_idx * 100 + agent_idx,
                ).to(projected.device, dtype=projected.dtype)
                exact_scores: list[float] = []
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
                    strategy_rows.append(
                        {
                            **metadata,
                            "sample_idx": sample_idx,
                            "agent_idx": agent_idx,
                            "model_name": model_name,
                            "accuracy": float(model_index == exact_best_index),
                            "exact_best_index": exact_best_index,
                            "model_best_index": model_index,
                        }
                    )
                    regret_rows.append(
                        {
                            **metadata,
                            "sample_idx": sample_idx,
                            "agent_idx": agent_idx,
                            "model_name": model_name,
                            "regret": exact_best_value - chosen_exact_value,
                            "exact_best_index": exact_best_index,
                            "model_best_index": model_index,
                        }
                    )

        exact_summary = enumerate_equilibria(case.ground_truth_game, case.equilibrium_config)
        exact_regrets = exact_summary.records.set_index("profile_key")["max_regret"]
        for model_name, model in case.model_map().items():
            approx_summary = enumerate_equilibria(model, case.equilibrium_config)
            approx_candidates = approx_summary.epsilon_nash
            if approx_candidates.empty:
                epsilon_rows.append({**metadata, "model_name": model_name, "profile_key": None, "gt_epsilon": float("nan")})
                continue
            for row in approx_candidates.itertuples():
                epsilon_rows.append(
                    {
                        **metadata,
                        "model_name": model_name,
                        "profile_key": row.profile_key,
                        "gt_epsilon": float(exact_regrets.loc[row.profile_key]),
                    }
                )

    strategy_raw = pd.DataFrame(strategy_rows)
    regret_raw = pd.DataFrame(regret_rows)
    epsilon_raw = pd.DataFrame(epsilon_rows)
    diagnostics_raw = pd.DataFrame(diagnostics_rows)

    group_keys = ["scenario", "activation_label", "activation_name", "activation_family", "activation_slug", "model_name"]
    summary_keys = ["scenario", "activation_label", "activation_name", "activation_family", "activation_slug"]

    strategy_summary = (
        strategy_raw.groupby(group_keys, dropna=False)["accuracy"]
        .mean()
        .reset_index()
        .sort_values(group_keys)
        .reset_index(drop=True)
    )
    regret_summary = (
        regret_raw.groupby(group_keys, dropna=False)["regret"]
        .mean()
        .reset_index()
        .rename(columns={"regret": "mean_regret"})
        .sort_values(group_keys)
        .reset_index(drop=True)
    )
    epsilon_summary = (
        epsilon_raw.groupby(group_keys, dropna=False)["gt_epsilon"]
        .mean()
        .reset_index()
        .rename(columns={"gt_epsilon": "mean_epsilon"})
        .sort_values(group_keys)
        .reset_index(drop=True)
    )
    diagnostics_summary = (
        diagnostics_raw.groupby(summary_keys, dropna=False)[["entropy", "support_size"]]
        .mean()
        .reset_index()
        .rename(columns={"entropy": "avg_entropy", "support_size": "avg_support_size"})
        .sort_values(summary_keys)
        .reset_index(drop=True)
    )

    summary = strategy_summary.merge(regret_summary, on=group_keys, how="outer").merge(epsilon_summary, on=group_keys, how="outer")
    summary = summary.merge(diagnostics_summary, on=summary_keys, how="left").sort_values(group_keys).reset_index(drop=True)

    write_frame(results_dir / "strategy_raw.csv", strategy_raw)
    write_frame(results_dir / "regret_raw.csv", regret_raw)
    write_frame(results_dir / "epsilon_raw.csv", epsilon_raw)
    write_frame(results_dir / "diagnostics_raw.csv", diagnostics_raw)
    write_frame(results_dir / "strategy_summary.csv", strategy_summary)
    write_frame(results_dir / "regret_summary.csv", regret_summary)
    write_frame(results_dir / "epsilon_summary.csv", epsilon_summary)
    write_frame(results_dir / "diagnostics_summary.csv", diagnostics_summary)
    write_frame(results_dir / "summary.csv", summary)
    return results_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_from_config(args.config)


if __name__ == "__main__":
    main()
