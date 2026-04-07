from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from interference_game.utils.io import read_frame
from interference_game.additive.experiments.plot_strategy_comparison import MODEL_COLORS, MODEL_LABELS, MODEL_ORDER


def plot_results(results_dir: str | Path) -> Path:
    results_path = Path(results_dir)
    summary = read_frame(results_path / "summary.csv")
    scenarios = list(summary["scenario"].drop_duplicates())
    positions = range(len(scenarios))
    width = 0.24

    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    for offset, model_name in enumerate(MODEL_ORDER):
        group = summary[summary["model_name"] == model_name].set_index("scenario").reindex(scenarios)
        x = [position + (offset - 1) * width for position in positions]
        axis.bar(
            x,
            group["mean_regret"],
            width=width,
            color=MODEL_COLORS[model_name],
            label=MODEL_LABELS[model_name],
        )

    axis.set_xticks(list(positions))
    axis.set_xticklabels(scenarios, rotation=15, ha="right")
    axis.set_ylabel("Mean Regret")
    axis.set_title("Ground-Truth Regret of Model Recommendations")
    axis.legend(frameon=False)
    figure.tight_layout()
    output = results_path / "regret_analysis.png"
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
