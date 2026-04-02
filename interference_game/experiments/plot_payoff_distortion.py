from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from interference_game.utils.io import read_frame


def plot_results(results_dir: str | Path) -> Path:
    results_path = Path(results_dir)
    summary = read_frame(results_path / "summary.csv")

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for baseline, group in summary.groupby("baseline"):
        group = group.sort_values("mixing_depth")
        axes[0].plot(group["mixing_depth"], group["mean_abs_utility_error_mean"], marker="o", label=baseline)
        axes[1].plot(group["mixing_depth"], group["br_match_rate_mean"], marker="o", label=baseline)
        axes[2].plot(group["mixing_depth"], group["spearman_rank_correlation_mean"], marker="o", label=baseline)

    axes[0].set_title("Payoff Distortion vs. Mixing Depth")
    axes[0].set_xlabel("mixing depth")
    axes[0].set_ylabel("mean absolute utility error")
    axes[1].set_title("Best-Response Preservation")
    axes[1].set_xlabel("mixing depth")
    axes[1].set_ylabel("match rate")
    axes[1].set_ylim(0.0, 1.05)
    axes[2].set_title("Candidate Rank Correlation")
    axes[2].set_xlabel("mixing depth")
    axes[2].set_ylabel("Spearman correlation")
    axes[2].set_ylim(-1.05, 1.05)
    axes[0].legend()
    axes[1].legend()
    axes[2].legend()

    figure.tight_layout()
    output = results_path / "payoff_distortion.png"
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()
    plot_results(args.results_dir)
