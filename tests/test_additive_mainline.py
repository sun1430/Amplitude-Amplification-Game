from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from interference_game.additive.activations import entmax15
from interference_game.additive.experiments.common import load_cases
from interference_game.additive.experiments.run_epsilon_analysis import run_from_config as run_epsilon
from interference_game.additive.experiments.run_regret_analysis import run_from_config as run_regret
from interference_game.additive.experiments.run_strategy_comparison import run_from_config as run_strategy


ROOT = Path(__file__).resolve().parents[1]


def _simplex(tensor: torch.Tensor) -> bool:
    return bool(torch.all(tensor >= -1e-10) and torch.isclose(tensor.sum(), torch.tensor(1.0, dtype=tensor.dtype), atol=1e-8))


def test_entmax15_produces_sparse_simplex_outputs() -> None:
    logits = torch.tensor([3.0, 1.0, -2.0], dtype=torch.float64)
    output = entmax15(logits, dim=0)
    assert _simplex(output)
    assert torch.count_nonzero(output == 0.0) >= 1


def test_additive_models_return_valid_distributions_and_scores() -> None:
    cases = load_cases(ROOT / "configs" / "additive" / "quick" / "strategy.yaml")
    case = cases[0]
    actions = torch.tensor([[0.3, -0.1, 0.2], [0.1, 0.2, -0.2]], dtype=torch.float64)

    gt_result = case.ground_truth_game.evaluate(actions)
    quantum_result = case.quantum_game.evaluate(actions)
    mlp_result = case.sota_model.evaluate(actions)

    assert _simplex(gt_result.influence_distribution)
    assert _simplex(quantum_result.influence_distribution)
    assert _simplex(mlp_result.influence_distribution)
    quantum_distribution = quantum_result.latent_state.abs().square().real
    quantum_distribution = quantum_distribution / quantum_distribution.sum()
    assert torch.allclose(quantum_result.influence_distribution, quantum_distribution, atol=1e-8)
    assert torch.allclose(quantum_result.influence_distribution, gt_result.influence_distribution, atol=1e-8)
    assert torch.allclose(quantum_result.utilities, gt_result.utilities, atol=1e-8)
    larger_action = actions * 1.5
    assert torch.all(case.ground_truth_game.evaluate(larger_action).costs >= gt_result.costs)


def test_residual_mlp_artifacts_and_activation() -> None:
    case = load_cases(ROOT / "configs" / "additive" / "quick" / "strategy.yaml")[0]
    artifacts = case.sota_model.training_artifacts
    assert artifacts is not None
    assert artifacts.checkpoint_path.exists()
    assert artifacts.metadata_path.exists()
    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
    assert "history" in metadata
    actions = torch.tensor([[0.2, -0.2, 0.1], [0.1, 0.0, -0.1]], dtype=torch.float64)
    projected = case.sota_model.project_actions(actions)
    logits = case.sota_model.network(projected.reshape(1, -1)).squeeze(0)
    distribution = entmax15(logits, dim=0)
    assert _simplex(distribution)


def test_additive_quick_experiments() -> None:
    strategy_dir = run_strategy(ROOT / "configs" / "additive" / "quick" / "strategy.yaml")
    regret_dir = run_regret(ROOT / "configs" / "additive" / "quick" / "regret.yaml")
    epsilon_dir = run_epsilon(ROOT / "configs" / "additive" / "quick" / "epsilon.yaml")

    for result_dir, column in [
        (strategy_dir, "accuracy"),
        (regret_dir, "mean_regret"),
        (epsilon_dir, "mean_epsilon"),
    ]:
        summary = pd.read_csv(result_dir / "summary.csv")
        assert set(summary["model_name"]) == {"ground_truth", "quantum_encoded", "residual_mlp"}
        assert column in summary.columns
        assert (result_dir / "raw.csv").exists()
