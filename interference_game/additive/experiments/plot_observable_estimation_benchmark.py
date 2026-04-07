from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from interference_game.utils.io import read_frame


METHOD_LABELS = {
    "amplitude_estimation": "Amplitude Estimation",
    "monte_carlo": "Monte Carlo",
}

METHOD_COLORS = {
    "amplitude_estimation": "#e76f51",
    "monte_carlo": "#2a9d8f",
}


def plot_results(results_dir: str | Path) -> Path:
    results_path = Path(results_dir)
    aggregate = read_frame(results_path / "aggregate.csv")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for method, group in aggregate.groupby("method"):
        ordered = group.sort_values("query_budget")
        for axis, metric, title in [
            (axes[0], "observable_error", "Observable Error vs Query Budget"),
            (axes[1], "utility_error", "Utility Error vs Query Budget"),
        ]:
            axis.plot(
                ordered["query_budget"],
                ordered[metric],
                marker="o",
                linewidth=2.0,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )
            axis.set_xscale("log", base=2)
            axis.set_yscale("log")
            axis.set_xlabel("query budget")
            axis.set_ylabel(metric.replace("_", " "))
            axis.set_title(title)

    axes[0].legend(frameon=False)
    figure.tight_layout()
    output = results_path / "observable_estimation_tradeoff.png"
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
