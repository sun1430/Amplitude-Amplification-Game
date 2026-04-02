from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from interference_game.experiments.common import load_cases, output_dir
from interference_game.experiments.plot_payoff_distortion import plot_results
from interference_game.utils.io import write_frame
from interference_game.utils.metrics import best_response_preservation, payoff_metric_record, random_feasible_actions, summarize_frame


def run_from_config(config_path: str | Path) -> Path:
    rows: list[dict[str, float | int | str | bool]] = []
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
        for sample_idx, joint_action in enumerate(samples):
            exact = case.exact_game.evaluate(joint_action)
            for baseline_name, baseline_game in case.baselines.items():
                approx = baseline_game.evaluate(joint_action)
                metrics = payoff_metric_record(exact, approx)
                br_records = best_response_preservation(
                    case.exact_game,
                    baseline_game,
                    joint_action,
                    candidate_seed=case.experiment_config.seed + sample_idx + case.scenario_index * 1000,
                    num_candidates=case.experiment_config.best_response_candidates,
                )
                rows.append(
                    {
                        **case.metadata(),
                        "sample_idx": sample_idx,
                        "baseline": baseline_name,
                        **metrics,
                        "br_match_rate": sum(record.match for record in br_records) / len(br_records),
                        "spearman_rank_correlation": sum(record.rank_correlation for record in br_records) / len(br_records),
                    }
                )

    raw = pd.DataFrame(rows)
    write_frame(results_dir / "raw.csv", raw)
    summary = summarize_frame(
        raw,
        group_cols=[
            "scenario",
            "baseline",
            "num_agents",
            "state_dim",
            "mixing_depth",
            "mean_action_budget",
            "conflict_level",
            "competitive_extension",
        ],
        value_cols=[
            "mean_abs_utility_error",
            "mean_rel_utility_error",
            "mean_abs_fidelity_error",
            "kl_divergence",
            "l2_distribution_error",
            "br_match_rate",
            "spearman_rank_correlation",
        ],
    )
    write_frame(results_dir / "summary.csv", summary)
    plot_results(results_dir)
    return results_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_from_config(args.config)


if __name__ == "__main__":
    main()
