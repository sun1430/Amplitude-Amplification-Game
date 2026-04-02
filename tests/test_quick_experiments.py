from __future__ import annotations

from pathlib import Path

import pandas as pd

from interference_game.experiments.run_ablation import run_from_config as run_ablation
from interference_game.experiments.run_dynamics import run_from_config as run_dynamics
from interference_game.experiments.run_equilibrium_distortion import run_from_config as run_equilibrium
from interference_game.experiments.run_payoff_distortion import run_from_config as run_payoff
from interference_game.experiments.run_sanity import run_from_config as run_sanity


ROOT = Path(__file__).resolve().parents[1]


def assert_artifacts(path: Path, png_name: str) -> None:
    assert path.exists()
    assert any(item.suffix == ".csv" for item in path.iterdir())
    assert (path / png_name).exists()


def test_quick_sanity() -> None:
    result_dir = run_sanity(ROOT / "configs" / "quick" / "sanity.yaml")
    assert_artifacts(result_dir, "sanity_summary.png")
    assert (result_dir / "sanity_note.txt").exists()


def test_quick_payoff() -> None:
    result_dir = run_payoff(ROOT / "configs" / "quick" / "payoff.yaml")
    assert_artifacts(result_dir, "payoff_distortion.png")
    summary = pd.read_csv(result_dir / "summary.csv")
    assert "spearman_rank_correlation_mean" in summary.columns


def test_quick_equilibrium() -> None:
    result_dir = run_equilibrium(ROOT / "configs" / "quick" / "equilibrium.yaml")
    assert_artifacts(result_dir, "equilibrium_distortion.png")


def test_quick_dynamics() -> None:
    result_dir = run_dynamics(ROOT / "configs" / "quick" / "dynamics.yaml")
    assert_artifacts(result_dir, "dynamics_summary.png")


def test_quick_ablation() -> None:
    result_dir = run_ablation(ROOT / "configs" / "quick" / "ablation.yaml")
    assert_artifacts(result_dir, "ablation_summary.png")
