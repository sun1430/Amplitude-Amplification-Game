from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from interference_game.additive.experiments.plot_strategy_comparison import MODEL_COLORS, MODEL_LABELS, MODEL_ORDER
from interference_game.utils.io import read_frame


METHOD_COLORS = {
    "amplitude_estimation": "#e76f51",
    "monte_carlo": "#2a9d8f",
}


def plot_results(results_dir: str | Path) -> tuple[Path, Path]:
    results_path = Path(results_dir)
    summary = read_frame(results_path / "summary.csv")
    horizon = read_frame(results_path / "horizon_summary.csv")
    observable = read_frame(results_path / "observable_aggregate.csv")
    scenarios = list(summary["scenario"].drop_duplicates())
    positions = range(len(scenarios))
    width = 0.24

    figure, axes = plt.subplots(2, 2, figsize=(13.5, 8.0))
    metric_panels = [
        (axes[0, 0], "accuracy", "Scenario Accuracy"),
        (axes[0, 1], "mean_regret", "Scenario Mean Regret"),
        (axes[1, 0], "mean_epsilon", "Scenario Mean Epsilon"),
    ]
    for axis, metric, title in metric_panels:
        for offset, model_name in enumerate(MODEL_ORDER):
            group = summary[summary["model_name"] == model_name].set_index("scenario").reindex(scenarios)
            x = [position + (offset - 1) * width for position in positions]
            axis.bar(x, group[metric], width=width, color=MODEL_COLORS[model_name], label=MODEL_LABELS[model_name])
        axis.set_xticks(list(positions))
        axis.set_xticklabels(scenarios, rotation=12, ha="right")
        axis.set_title(title)
        axis.grid(alpha=0.2, axis="y")

    horizon_axis = axes[1, 1]
    for model_name in ["quantum_encoded", "residual_mlp"]:
        group = horizon[horizon["model_name"] == model_name].sort_values("horizon")
        horizon_axis.plot(
            group["horizon"],
            group["mean_regret"],
            marker="o",
            linewidth=2.2,
            color=MODEL_COLORS[model_name],
            label=MODEL_LABELS[model_name],
        )
    horizon_axis.set_title("Horizon Sweep: Mean Regret")
    horizon_axis.set_xlabel("Mixing Depth")
    horizon_axis.grid(alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    metrics_output = results_path / "markov_special_case_metrics.png"
    figure.savefig(metrics_output, dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure2, axes2 = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for axis, metric, title in [
        (axes2[0], "observable_error", "Observable Error vs Query Budget"),
        (axes2[1], "utility_error", "Utility Error vs Query Budget"),
    ]:
        for method, group in observable.groupby("method"):
            ordered = group.sort_values("query_budget")
            axis.plot(
                ordered["query_budget"],
                ordered[metric],
                marker="o",
                linewidth=2.0,
                color=METHOD_COLORS[method],
                label=method.replace("_", " ").title(),
            )
        axis.set_xscale("log", base=2)
        axis.set_xlabel("Query Budget")
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axes2[0].legend(frameon=False)
    figure2.tight_layout()
    observable_output = results_path / "markov_special_case_observable.png"
    figure2.savefig(observable_output, dpi=180, bbox_inches="tight")
    plt.close(figure2)

    estimation_strategy = read_frame(results_path / "estimation_strategy_summary.csv")
    estimation_aggregate = (
        estimation_strategy.groupby(["method", "query_budget"], dropna=False)[["accuracy", "mean_regret"]]
        .mean()
        .reset_index()
        .sort_values(["method", "query_budget"])
        .reset_index(drop=True)
    )
    figure3, axes3 = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for axis, metric, title in [
        (axes3[0], "accuracy", "Strategy Accuracy vs Query Budget"),
        (axes3[1], "mean_regret", "Strategy Regret vs Query Budget"),
    ]:
        for method, group in estimation_aggregate.groupby("method"):
            ordered = group.sort_values("query_budget")
            axis.plot(
                ordered["query_budget"],
                ordered[metric],
                marker="o",
                linewidth=2.0,
                color=METHOD_COLORS[method],
                label=method.replace("_", " ").title(),
            )
        axis.set_xscale("log", base=2)
        axis.set_xlabel("Query Budget")
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axes3[0].legend(frameon=False)
    figure3.tight_layout()
    estimation_output = results_path / "markov_special_case_estimation_strategy.png"
    figure3.savefig(estimation_output, dpi=180, bbox_inches="tight")
    plt.close(figure3)
    return metrics_output, observable_output, estimation_output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()
    plot_results(args.results_dir)


if __name__ == "__main__":
    main()
