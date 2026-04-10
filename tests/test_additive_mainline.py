from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from interference_game.additive.activations import apply_simplex_activation, entmax15
from interference_game.additive.experiments.common import load_cases
from interference_game.additive.experiments.markov_common import load_cases as load_markov_cases
from interference_game.additive.experiments.run_activation_ablation import run_from_config as run_activation_ablation
from interference_game.additive.experiments.run_markov_special_case import run_from_config as run_markov_special_case
from interference_game.additive.experiments.run_observable_estimation_benchmark import run_from_config as run_observable_benchmark
from interference_game.additive.experiments.run_epsilon_analysis import run_from_config as run_epsilon
from interference_game.additive.experiments.run_regret_analysis import run_from_config as run_regret
from interference_game.additive.experiments.run_strategy_comparison import run_from_config as run_strategy
from interference_game.additive.markov_special_case import ReversibleSparseMarkovGame


ROOT = Path(__file__).resolve().parents[1]
ADDITIVE_ROOT = ROOT / "additive"


def _simplex(tensor: torch.Tensor) -> bool:
    return bool(torch.all(tensor >= -1e-10) and torch.isclose(tensor.sum(), torch.tensor(1.0, dtype=tensor.dtype), atol=1e-8))


def test_entmax15_produces_sparse_simplex_outputs() -> None:
    logits = torch.tensor([3.0, 1.0, -2.0], dtype=torch.float64)
    output = entmax15(logits, dim=0)
    assert _simplex(output)
    assert torch.count_nonzero(output == 0.0) >= 1


def test_simplex_activation_family_outputs() -> None:
    logits = torch.tensor([2.0, 0.5, -1.0], dtype=torch.float64)
    for family, kwargs in [
        ("softmax", {"beta": 1.0}),
        ("sparsemax", {}),
        ("entmax", {"alpha": 1.2}),
        ("entmax", {"alpha": 1.8}),
        ("bounded_confidence", {"gamma": 10.0, "tau": 0.05}),
    ]:
        output = apply_simplex_activation(logits, family=family, dim=0, **kwargs)
        assert _simplex(output)
        assert torch.all(output >= -1e-10)


def test_additive_models_return_valid_distributions_and_scores() -> None:
    cases = load_cases(ADDITIVE_ROOT / "configs" / "quick" / "strategy.yaml")
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
    assert quantum_result.observable_expectations is not None
    assert gt_result.observable_expectations is not None
    assert torch.max(torch.abs(quantum_result.observable_expectations - gt_result.observable_expectations)).item() < 0.02
    assert torch.max(torch.abs(quantum_result.utilities - gt_result.utilities)).item() < 0.02
    larger_action = actions * 1.5
    assert torch.all(case.ground_truth_game.evaluate(larger_action).costs >= gt_result.costs)


def test_markov_special_case_returns_valid_distributions() -> None:
    case = load_cases(ADDITIVE_ROOT / "configs" / "quick" / "strategy.yaml")[0]
    markov_game = ReversibleSparseMarkovGame(case.model_config, case.target_bundle.targets)
    actions = torch.tensor([[0.25, -0.05, 0.15], [0.1, 0.2, -0.15]], dtype=torch.float64)
    result = markov_game.evaluate(actions)
    assert _simplex(result.influence_distribution)
    assert result.observable_expectations is not None
    sampled_distribution = markov_game.sample_terminal_distribution(actions, num_draws=64, seed=11)
    sampled_expectations = markov_game.sample_observable_expectations(actions, num_draws=64, seed=11)
    assert _simplex(sampled_distribution)
    assert torch.all(sampled_expectations >= -1e-10)
    assert torch.all(sampled_expectations <= 1.0 + 1e-10)


def test_residual_mlp_artifacts_and_activation() -> None:
    case = load_cases(ADDITIVE_ROOT / "configs" / "quick" / "strategy.yaml")[0]
    artifacts = case.sota_model.training_artifacts
    assert artifacts is not None
    assert artifacts.checkpoint_path.exists()
    assert artifacts.metadata_path.exists()
    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8-sig"))
    assert "history" in metadata
    actions = torch.tensor([[0.2, -0.2, 0.1], [0.1, 0.0, -0.1]], dtype=torch.float64)
    distribution = case.sota_model.evaluate(actions).influence_distribution
    assert _simplex(distribution)


def test_markov_quantum_encoding_respects_single_step_transition_structure() -> None:
    case = load_markov_cases(ADDITIVE_ROOT / "configs" / "quick" / "markov_special_case.yaml")[0]
    actions = torch.tensor([[0.25, -0.05, 0.15], [0.1, 0.2, -0.15]], dtype=torch.float64)
    projected = case.ground_truth_game.project_actions(actions)

    classical_result = case.ground_truth_game.evaluate(projected)
    quantum_distribution, signal = case.quantum_game._simulate_state(projected)
    signal_expectations = case.quantum_game.exact_expectations_from_signal(signal)

    assert torch.allclose(quantum_distribution, classical_result.influence_distribution, atol=1e-8)
    assert classical_result.observable_expectations is not None
    assert torch.allclose(signal_expectations, classical_result.observable_expectations, atol=1e-8)

    for node in range(case.model_config.state_dim):
        row_state = case.ground_truth_game.prepare_row_state(node)
        column_state = case.ground_truth_game.prepare_column_state(node)
        assert torch.isclose(row_state.square().sum(), torch.tensor(1.0, dtype=row_state.dtype, device=row_state.device), atol=1e-8)
        assert torch.isclose(column_state.square().sum(), torch.tensor(1.0, dtype=column_state.dtype, device=column_state.device), atol=1e-8)

        neighbor_count = int(case.ground_truth_game.walk_structure.neighbor_counts[node].item())
        recovered = torch.zeros(case.model_config.state_dim, dtype=row_state.dtype, device=row_state.device)
        for slot in range(neighbor_count):
            neighbor, probability = case.ground_truth_game.neighbor_probability_oracle(node, slot)
            recovered[neighbor] = probability
        assert torch.allclose(recovered, case.ground_truth_game.row_transition[node], atol=1e-8)


def test_additive_quick_experiments() -> None:
    strategy_dir = run_strategy(ADDITIVE_ROOT / "configs" / "quick" / "strategy.yaml")
    regret_dir = run_regret(ADDITIVE_ROOT / "configs" / "quick" / "regret.yaml")
    epsilon_dir = run_epsilon(ADDITIVE_ROOT / "configs" / "quick" / "epsilon.yaml")
    benchmark_dir = run_observable_benchmark(ADDITIVE_ROOT / "configs" / "quick" / "observable_estimation.yaml")
    activation_dir = run_activation_ablation(ADDITIVE_ROOT / "configs" / "quick" / "activation_ablation.yaml")

    for result_dir, column in [
        (strategy_dir, "accuracy"),
        (regret_dir, "mean_regret"),
        (epsilon_dir, "mean_epsilon"),
    ]:
        summary = pd.read_csv(result_dir / "summary.csv")
        assert set(summary["model_name"]) == {"ground_truth", "quantum_encoded", "residual_mlp"}
        assert column in summary.columns
        assert (result_dir / "raw.csv").exists()

    benchmark_summary = pd.read_csv(benchmark_dir / "summary.csv")
    benchmark_aggregate = pd.read_csv(benchmark_dir / "aggregate.csv")
    assert set(benchmark_summary["method"]) == {"amplitude_estimation", "monte_carlo"}
    assert set(benchmark_aggregate["method"]) == {"amplitude_estimation", "monte_carlo"}
    assert {"query_budget", "observable_error", "utility_error"}.issubset(benchmark_aggregate.columns)

    activation_summary = pd.read_csv(activation_dir / "summary.csv")
    assert {"scenario", "activation_label", "model_name", "accuracy", "mean_regret", "mean_epsilon"}.issubset(activation_summary.columns)
    assert set(activation_summary["activation_label"]) == {"entmax_1p5", "sparsemax"}

    markov_dir = run_markov_special_case(ADDITIVE_ROOT / "configs" / "quick" / "markov_special_case.yaml")
    markov_summary = pd.read_csv(markov_dir / "summary.csv")
    markov_horizon = pd.read_csv(markov_dir / "horizon_summary.csv")
    markov_observable = pd.read_csv(markov_dir / "observable_aggregate.csv")
    markov_estimation = pd.read_csv(markov_dir / "estimation_strategy_summary.csv")
    assert set(markov_summary["model_name"]) == {"ground_truth", "quantum_encoded", "residual_mlp"}
    assert {"horizon", "model_name", "accuracy", "mean_regret"}.issubset(markov_horizon.columns)
    assert set(markov_observable["method"]) == {"amplitude_estimation", "monte_carlo"}
    assert set(markov_estimation["method"]) == {"amplitude_estimation", "monte_carlo"}
    assert {"horizon", "query_budget", "accuracy", "mean_regret"}.issubset(markov_estimation.columns)
