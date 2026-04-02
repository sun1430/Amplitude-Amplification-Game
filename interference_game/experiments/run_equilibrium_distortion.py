from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from interference_game.equilibrium.discrete_enumeration import compare_equilibrium_sets, enumerate_equilibria
from interference_game.experiments.common import load_cases, output_dir
from interference_game.experiments.plot_equilibrium_distortion import plot_results
from interference_game.utils.io import write_frame


def run_from_config(config_path: str | Path) -> Path:
    cases = load_cases(config_path)
    results_dir = output_dir(cases[0].experiment_config.output_root)
    comparison_rows = []
    profile_rows = []

    for case in cases:
        exact_summary = enumerate_equilibria(case.exact_game, case.equilibrium_config)
        exact_records = exact_summary.records.copy()
        exact_records["scenario"] = case.scenario_name
        exact_records["model_name"] = "exact"
        profile_rows.append(exact_records)

        for baseline_name, baseline_game in case.baselines.items():
            approx_summary = enumerate_equilibria(baseline_game, case.equilibrium_config)
            baseline_records = approx_summary.records.copy()
            baseline_records["scenario"] = case.scenario_name
            baseline_records["model_name"] = baseline_name
            profile_rows.append(baseline_records)
            comparison_rows.append(
                {
                    **case.metadata(),
                    "baseline": baseline_name,
                    **compare_equilibrium_sets(exact_summary, approx_summary),
                }
            )

    write_frame(results_dir / "profiles.csv", pd.concat(profile_rows, ignore_index=True))
    write_frame(results_dir / "summary.csv", pd.DataFrame(comparison_rows))
    plot_results(results_dir)
    return results_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_from_config(args.config)


if __name__ == "__main__":
    main()
