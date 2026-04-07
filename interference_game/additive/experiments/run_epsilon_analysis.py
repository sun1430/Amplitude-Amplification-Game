from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from interference_game.additive.experiments.common import load_cases, output_dir
from interference_game.equilibrium.discrete_enumeration import enumerate_equilibria
from interference_game.utils.io import write_frame


def run_from_config(config_path: str | Path) -> Path:
    raw_rows: list[dict[str, float | int | str]] = []
    summary_rows: list[dict[str, float | str]] = []
    cases = load_cases(config_path)
    results_dir = output_dir(cases[0].experiment_config.output_root)

    for case in cases:
        exact_summary = enumerate_equilibria(case.ground_truth_game, case.equilibrium_config)
        exact_regrets = exact_summary.records.set_index("profile_key")["max_regret"]
        for model_name, model in case.model_map().items():
            approx_summary = enumerate_equilibria(model, case.equilibrium_config)
            approx_candidates = approx_summary.epsilon_nash
            if approx_candidates.empty:
                summary_rows.append(
                    {
                        "scenario": case.scenario_name,
                        "model_name": model_name,
                        "mean_epsilon": float("nan"),
                    }
                )
                continue

            candidate_epsilons = []
            for row in approx_candidates.itertuples():
                gt_epsilon = float(exact_regrets.loc[row.profile_key])
                candidate_epsilons.append(gt_epsilon)
                raw_rows.append(
                    {
                        **case.metadata(),
                        "model_name": model_name,
                        "profile_key": row.profile_key,
                        "gt_epsilon": gt_epsilon,
                    }
                )

            summary_rows.append(
                {
                    "scenario": case.scenario_name,
                    "model_name": model_name,
                    "mean_epsilon": float(sum(candidate_epsilons) / len(candidate_epsilons)),
                }
            )

    raw = pd.DataFrame(raw_rows, columns=["scenario", "model_name", "profile_key", "gt_epsilon"])
    summary = pd.DataFrame(summary_rows).sort_values(["scenario", "model_name"]).reset_index(drop=True)
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
