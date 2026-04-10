from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch

from interference_game.additive.config import load_additive_yaml
from interference_game.additive.experiments.markov_common import build_case, load_cases, output_dir
from interference_game.config import merge_nested
from interference_game.equilibrium.discrete_enumeration import enumerate_equilibria
from interference_game.utils.io import write_frame
from interference_game.utils.metrics import build_candidate_actions, random_feasible_actions


def _distribution_entropy(distribution: torch.Tensor) -> float:
    clipped = torch.clamp(distribution, min=1e-12)
    return float((-clipped * torch.log(clipped)).sum().item())


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


def _benchmark_case(raw_config: dict, scenarios: list[dict], horizon: int, scenario_index: int):
    candidates = [scenario for scenario in scenarios if scenario.get("target", {}).get("conflict_level") == "medium"]
    base = candidates[0] if candidates else scenarios[0]
    override = merge_nested(base, {"name": f"{base.get('name', 'benchmark')}_h{horizon}", "model": {"mixing_depth": horizon}})
    return build_case(raw_config, override, scenario_index)


def run_from_config(config_path: str | Path) -> Path:
    strategy_rows: list[dict[str, float | int | str]] = []
    regret_rows: list[dict[str, float | int | str]] = []
    epsilon_rows: list[dict[str, float | int | str]] = []
    diagnostics_rows: list[dict[str, float | int | str]] = []
    horizon_rows: list[dict[str, float | int | str]] = []
    observable_rows: list[dict[str, float | int | str]] = []
    estimation_strategy_rows: list[dict[str, float | int | str]] = []

    raw_config, scenarios = load_additive_yaml(config_path)
    cases = load_cases(config_path)
    results_dir = output_dir(cases[0].experiment_config.output_root)

    for case in cases:
        metadata = case.metadata()
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
                    "support_size": float(torch.count_nonzero(gt_distribution > 1e-8).item()),
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
                    strategy_rows.append(
                        {
                            **metadata,
                            "sample_idx": sample_idx,
                            "agent_idx": agent_idx,
                            "model_name": model_name,
                            "accuracy": float(model_index == exact_best_index),
                        }
                    )
                    regret_rows.append(
                        {
                            **metadata,
                            "sample_idx": sample_idx,
                            "agent_idx": agent_idx,
                            "model_name": model_name,
                            "regret": exact_best_value - exact_scores[model_index],
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

    benchmark_seed_offset = 10000
    max_case_index = len(cases)
    base_experiment = cases[0].experiment_config
    for horizon_index, horizon in enumerate(base_experiment.horizon_values):
        bench_case = _benchmark_case(raw_config, scenarios, int(horizon), max_case_index + horizon_index)
        metadata = bench_case.metadata()
        samples = random_feasible_actions(
            num_samples=bench_case.experiment_config.num_profiles,
            num_agents=bench_case.model_config.num_agents,
            state_dim=bench_case.model_config.state_dim,
            budgets=bench_case.model_config.budgets(),
            seed=bench_case.experiment_config.seed + benchmark_seed_offset + horizon_index,
        )
        for sample_idx, joint_action in enumerate(samples):
            projected = bench_case.ground_truth_game.project_actions(joint_action)
            query_budgets = sorted(set(int(value) for value in bench_case.experiment_config.estimation_budgets if int(value) > 0))
            for agent_idx in range(bench_case.model_config.num_agents):
                candidates = build_candidate_actions(
                    num_candidates=bench_case.experiment_config.best_response_candidates,
                    state_dim=bench_case.model_config.state_dim,
                    budget=bench_case.model_config.budgets()[agent_idx],
                    seed=bench_case.experiment_config.seed + benchmark_seed_offset + horizon_index * 1000 + sample_idx * 100 + agent_idx,
                ).to(projected.device, dtype=projected.dtype)
                exact_scores: list[float] = []
                quantum_scores: list[float] = []
                mlp_scores: list[float] = []
                estimation_scores = {("amplitude_estimation", budget): [] for budget in query_budgets}
                estimation_scores.update({("monte_carlo", budget): [] for budget in query_budgets})
                for candidate_idx, candidate in enumerate(candidates):
                    candidate_profile = projected.clone()
                    candidate_profile[agent_idx] = candidate
                    exact_scores.append(float(bench_case.ground_truth_game.evaluate(candidate_profile).utilities[agent_idx].item()))
                    quantum_scores.append(float(bench_case.quantum_game.evaluate(candidate_profile).utilities[agent_idx].item()))
                    mlp_scores.append(float(bench_case.sota_model.evaluate(candidate_profile).utilities[agent_idx].item()))
                    for query_budget in query_budgets:
                        num_qubits = max(int(round(math.log2(query_budget))), 1)
                        ae_result = bench_case.quantum_game.evaluate_with_estimation(candidate_profile, num_qubits=num_qubits)
                        estimation_scores[("amplitude_estimation", query_budget)].append(float(ae_result.utilities[agent_idx].item()))
                        mc_expectations = bench_case.ground_truth_game.sample_observable_expectations(
                            candidate_profile,
                            num_draws=query_budget,
                            seed=(
                                bench_case.experiment_config.seed
                                + benchmark_seed_offset
                                + horizon_index * 100000
                                + sample_idx * 1000
                                + agent_idx * 100
                                + candidate_idx * 10
                                + query_budget
                            ),
                        )
                        mc_utilities = _utility_from_expectations(bench_case.ground_truth_game, mc_expectations, candidate_profile)
                        estimation_scores[("monte_carlo", query_budget)].append(float(mc_utilities[agent_idx].item()))
                exact_best_value = max(exact_scores)
                exact_best_index = int(torch.tensor(exact_scores).argmax().item())
                quantum_index = int(torch.tensor(quantum_scores).argmax().item())
                mlp_index = int(torch.tensor(mlp_scores).argmax().item())
                horizon_rows.extend(
                    [
                        {
                            "horizon": int(horizon),
                            "model_name": "quantum_encoded",
                            "accuracy": float(quantum_index == exact_best_index),
                            "regret": exact_best_value - exact_scores[quantum_index],
                        },
                        {
                            "horizon": int(horizon),
                            "model_name": "residual_mlp",
                            "accuracy": float(mlp_index == exact_best_index),
                            "regret": exact_best_value - exact_scores[mlp_index],
                        },
                    ]
                )
                for (method_name, query_budget), scores in estimation_scores.items():
                    model_index = int(torch.tensor(scores).argmax().item())
                    estimation_strategy_rows.append(
                        {
                            "horizon": int(horizon),
                            "method": method_name,
                            "query_budget": int(query_budget),
                            "accuracy": float(model_index == exact_best_index),
                            "regret": exact_best_value - exact_scores[model_index],
                        }
                    )

            gt_result = bench_case.ground_truth_game.evaluate(projected)
            exact_expectations = gt_result.observable_expectations
            assert exact_expectations is not None
            quantum_result = bench_case.quantum_game.evaluate(projected)
            quantum_signal = quantum_result.latent_state
            assert quantum_signal is not None
            for query_budget in query_budgets:
                num_qubits = max(int(round(math.log2(query_budget))), 1)
                ae_expectations = bench_case.quantum_game.estimate_observable_expectations_from_signal(
                    quantum_signal.real,
                    num_qubits=num_qubits,
                )
                ae_utilities = _utility_from_expectations(bench_case.ground_truth_game, ae_expectations, projected)
                observable_rows.append(
                    {
                        **metadata,
                        "horizon": int(horizon),
                        "sample_idx": sample_idx,
                        "method": "amplitude_estimation",
                        "query_budget": query_budget,
                        "observable_error": float(torch.mean(torch.abs(ae_expectations - exact_expectations)).item()),
                        "utility_error": float(torch.mean(torch.abs(ae_utilities - gt_result.utilities)).item()),
                    }
                )

                mc_expectations = bench_case.ground_truth_game.sample_observable_expectations(
                    projected,
                    num_draws=query_budget,
                    seed=bench_case.experiment_config.seed + benchmark_seed_offset + horizon_index * 100000 + sample_idx * 100 + query_budget,
                )
                mc_utilities = _utility_from_expectations(bench_case.ground_truth_game, mc_expectations, projected)
                observable_rows.append(
                    {
                        **metadata,
                        "horizon": int(horizon),
                        "sample_idx": sample_idx,
                        "method": "monte_carlo",
                        "query_budget": query_budget,
                        "observable_error": float(torch.mean(torch.abs(mc_expectations - exact_expectations)).item()),
                        "utility_error": float(torch.mean(torch.abs(mc_utilities - gt_result.utilities)).item()),
                    }
                )

    strategy_raw = pd.DataFrame(strategy_rows)
    regret_raw = pd.DataFrame(regret_rows)
    epsilon_raw = pd.DataFrame(epsilon_rows)
    diagnostics_raw = pd.DataFrame(diagnostics_rows)
    horizon_raw = pd.DataFrame(horizon_rows)
    observable_raw = pd.DataFrame(observable_rows)
    estimation_strategy_raw = pd.DataFrame(estimation_strategy_rows)

    group_keys = ["scenario", "model_name"]
    strategy_summary = strategy_raw.groupby(group_keys, dropna=False)["accuracy"].mean().reset_index()
    regret_summary = regret_raw.groupby(group_keys, dropna=False)["regret"].mean().reset_index().rename(columns={"regret": "mean_regret"})
    epsilon_summary = (
        epsilon_raw.groupby(group_keys, dropna=False)["gt_epsilon"].mean().reset_index().rename(columns={"gt_epsilon": "mean_epsilon"})
    )
    diagnostics_summary = (
        diagnostics_raw.groupby(["scenario"], dropna=False)[["entropy", "support_size"]]
        .mean()
        .reset_index()
        .rename(columns={"entropy": "avg_entropy", "support_size": "avg_support_size"})
    )
    horizon_summary = (
        horizon_raw.groupby(["horizon", "model_name"], dropna=False)[["accuracy", "regret"]]
        .mean()
        .reset_index()
        .rename(columns={"regret": "mean_regret"})
        .sort_values(["horizon", "model_name"])
        .reset_index(drop=True)
    )
    observable_summary = (
        observable_raw.groupby(["horizon", "method", "query_budget"], dropna=False)[["observable_error", "utility_error"]]
        .mean()
        .reset_index()
        .sort_values(["horizon", "method", "query_budget"])
        .reset_index(drop=True)
    )
    observable_aggregate = (
        observable_raw.groupby(["method", "query_budget"], dropna=False)[["observable_error", "utility_error"]]
        .mean()
        .reset_index()
        .sort_values(["method", "query_budget"])
        .reset_index(drop=True)
    )
    estimation_strategy_summary = (
        estimation_strategy_raw.groupby(["horizon", "method", "query_budget"], dropna=False)[["accuracy", "regret"]]
        .mean()
        .reset_index()
        .rename(columns={"regret": "mean_regret"})
        .sort_values(["horizon", "method", "query_budget"])
        .reset_index(drop=True)
    )
    summary = strategy_summary.merge(regret_summary, on=group_keys).merge(epsilon_summary, on=group_keys).merge(diagnostics_summary, on="scenario")

    write_frame(results_dir / "strategy_raw.csv", strategy_raw)
    write_frame(results_dir / "regret_raw.csv", regret_raw)
    write_frame(results_dir / "epsilon_raw.csv", epsilon_raw)
    write_frame(results_dir / "diagnostics_raw.csv", diagnostics_raw)
    write_frame(results_dir / "horizon_raw.csv", horizon_raw)
    write_frame(results_dir / "observable_raw.csv", observable_raw)
    write_frame(results_dir / "estimation_strategy_raw.csv", estimation_strategy_raw)
    write_frame(results_dir / "strategy_summary.csv", strategy_summary)
    write_frame(results_dir / "regret_summary.csv", regret_summary)
    write_frame(results_dir / "epsilon_summary.csv", epsilon_summary)
    write_frame(results_dir / "diagnostics_summary.csv", diagnostics_summary)
    write_frame(results_dir / "horizon_summary.csv", horizon_summary)
    write_frame(results_dir / "observable_summary.csv", observable_summary)
    write_frame(results_dir / "observable_aggregate.csv", observable_aggregate)
    write_frame(results_dir / "estimation_strategy_summary.csv", estimation_strategy_summary)
    write_frame(results_dir / "summary.csv", summary.sort_values(["scenario", "model_name"]).reset_index(drop=True))
    return results_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_from_config(args.config)


if __name__ == "__main__":
    main()
