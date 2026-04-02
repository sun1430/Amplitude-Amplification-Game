from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from interference_game.utils.io import read_frame


def plot_results(results_dir: str | Path) -> Path:
    results_path = Path(results_dir)
    summary = read_frame(results_path / "summary.csv")
    trajectories = read_frame(results_path / "trajectories.csv")

    rates = (
        summary.groupby(["model_name", "method"], dropna=False)[["converged", "cycle_detected"]]
        .mean()
        .reset_index()
    )
    representative = trajectories[
        (trajectories["model_name"] == "exact") & (trajectories["method"] == "projected_gradient")
    ]

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    labels = [f"{row.model_name}\n{row.method}" for row in rates.itertuples()]
    axes[0].bar(labels, rates["converged"], label="converged")
    axes[0].bar(labels, rates["cycle_detected"], bottom=rates["converged"], label="cycle detected")
    axes[0].set_title("Dynamics Outcome Rates")
    axes[0].set_ylabel("fraction of runs")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].legend()

    if representative.empty:
        representative = trajectories
    for agent_idx, group in representative.groupby("agent_idx"):
        axes[1].plot(group["step"], group["utility"], label=f"agent {agent_idx}")
    axes[1].set_title("Representative Utility Trajectory")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("utility")
    axes[1].legend()

    figure.tight_layout()
    output = results_path / "dynamics_summary.png"
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()
    plot_results(args.results_dir)
