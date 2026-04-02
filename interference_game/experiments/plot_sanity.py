from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from interference_game.utils.io import read_frame


def plot_results(results_dir: str | Path) -> Path:
    results_path = Path(results_dir)
    perturbation = read_frame(results_path / "perturbation.csv")
    normalization = read_frame(results_path / "normalization.csv")
    checks = pd.read_csv(results_path / "checks.csv")

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for agent_idx, group in perturbation.groupby("agent_idx"):
        axes[0].plot(group["delta"], group["utility"], label=f"agent {agent_idx}")
    axes[0].set_title("Utility vs. Controlled Phase Perturbation")
    axes[0].set_xlabel("phase delta")
    axes[0].set_ylabel("utility")
    axes[0].legend()

    axes[1].plot(normalization["step"], normalization["norm"], marker="o")
    axes[1].set_title("State Normalization Trace")
    axes[1].set_xlabel("intermediate step")
    axes[1].set_ylabel("L2 norm")
    text = "\n".join(f"{row.check}: {row.passed}" for row in checks.itertuples())
    axes[1].text(1.05, 0.95, text, transform=axes[1].transAxes, va="top")

    figure.tight_layout()
    output = results_path / "sanity_summary.png"
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()
    plot_results(args.results_dir)
