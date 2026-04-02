from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from interference_game.utils.io import read_frame


def plot_results(results_dir: str | Path) -> Path:
    results_path = Path(results_dir)
    summary = read_frame(results_path / "summary.csv")

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for baseline, group in summary.groupby("baseline"):
        axes[0].bar(
            group["scenario"] + "\n" + baseline,
            group["pure_jaccard"],
            label=baseline,
        )
        axes[1].bar(
            group["scenario"] + "\n" + baseline,
            group["mean_exact_regret_of_approx_epsilon"],
            label=baseline,
        )

    axes[0].set_title("Pure-Equilibrium Overlap")
    axes[0].set_ylabel("Jaccard overlap")
    axes[0].tick_params(axis="x", rotation=45)
    axes[1].set_title("Approx. Equilibria Regret in Exact Game")
    axes[1].set_ylabel("mean regret")
    axes[1].tick_params(axis="x", rotation=45)

    figure.tight_layout()
    output = results_path / "equilibrium_distortion.png"
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()
    plot_results(args.results_dir)
