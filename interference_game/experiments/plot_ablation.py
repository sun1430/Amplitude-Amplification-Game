from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from interference_game.utils.io import read_frame


def plot_results(results_dir: str | Path) -> Path:
    results_path = Path(results_dir)
    summary = read_frame(results_path / "summary.csv")

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(summary["scenario"], summary["aggregate_error"])
    axes[0].set_title("Aggregate Approximation Error by Ablation")
    axes[0].set_ylabel("mean abs. utility error")
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].bar(summary["scenario"], summary["cycle_rate"])
    axes[1].set_title("Exact Dynamics Cycle Rate by Ablation")
    axes[1].set_ylabel("cycle rate")
    axes[1].tick_params(axis="x", rotation=45)

    figure.tight_layout()
    output = results_path / "ablation_summary.png"
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()
    plot_results(args.results_dir)
