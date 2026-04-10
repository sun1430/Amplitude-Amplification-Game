from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from interference_game.utils.io import read_frame


SCENARIO_COLORS = {
    "shallow_low_conflict": "#2a9d8f",
    "medium_conflict": "#e9c46a",
    "deep_high_conflict": "#e76f51",
}


def plot_results(results_dir: str | Path) -> Path:
    results_path = Path(results_dir)
    summary = read_frame(results_path / "summary.csv")
    diagnostics = read_frame(results_path / "diagnostics_summary.csv")
    mlp = summary[summary["model_name"] == "residual_mlp"].copy()
    activation_order = list(mlp["activation_label"].drop_duplicates())
    scenarios = list(mlp["scenario"].drop_duplicates())

    figure, axes = plt.subplots(2, 2, figsize=(13.5, 8.5))
    panels = [
        (axes[0, 0], "accuracy", "Residual MLP Accuracy"),
        (axes[0, 1], "mean_regret", "Residual MLP Mean Regret"),
        (axes[1, 0], "mean_epsilon", "Residual MLP Mean Epsilon"),
    ]

    for axis, metric, title in panels:
        for scenario in scenarios:
            group = mlp[mlp["scenario"] == scenario].set_index("activation_label").reindex(activation_order)
            axis.plot(
                activation_order,
                group[metric],
                marker="o",
                linewidth=2.0,
                color=SCENARIO_COLORS.get(scenario, "#264653"),
                label=scenario,
            )
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=18)
        axis.grid(alpha=0.25, linewidth=0.6)

    diagnostics_axis = axes[1, 1]
    for scenario in scenarios:
        group = diagnostics[diagnostics["scenario"] == scenario].set_index("activation_label").reindex(activation_order)
        diagnostics_axis.plot(
            activation_order,
            group["avg_support_size"],
            marker="o",
            linewidth=2.0,
            color=SCENARIO_COLORS.get(scenario, "#264653"),
            label=scenario,
        )
    diagnostics_axis.set_title("GT Average Support Size")
    diagnostics_axis.tick_params(axis="x", rotation=18)
    diagnostics_axis.grid(alpha=0.25, linewidth=0.6)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=len(scenarios), frameon=False, bbox_to_anchor=(0.5, 1.02))
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    output = results_path / "activation_ablation_mlp.png"
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()
    plot_results(args.results_dir)


if __name__ == "__main__":
    main()
