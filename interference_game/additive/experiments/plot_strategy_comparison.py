from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from interference_game.utils.io import read_frame


MODEL_ORDER = ["ground_truth", "residual_mlp", "quantum_encoded"]
MODEL_LABELS = {
    "ground_truth": "Ground Truth",
    "residual_mlp": "Residual MLP",
    "quantum_encoded": "Quantum Encoded",
}
MODEL_COLORS = {
    "ground_truth": "#264653",
    "residual_mlp": "#2a9d8f",
    "quantum_encoded": "#e76f51",
}


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
            group["accuracy"],
            width=width,
            color=MODEL_COLORS[model_name],
            label=MODEL_LABELS[model_name],
        )

    axis.set_xticks(list(positions))
    axis.set_xticklabels(scenarios, rotation=15, ha="right")
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Accuracy")
    axis.set_title("Best-Response Accuracy by Scenario")
    axis.legend(frameon=False)
    figure.tight_layout()
    output = results_path / "strategy_comparison.png"
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
