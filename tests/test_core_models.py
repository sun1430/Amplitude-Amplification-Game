from __future__ import annotations

import torch

from interference_game.config import EquilibriumConfig, ModelConfig, TargetConfig
from interference_game.equilibrium.discrete_enumeration import enumerate_equilibria
from interference_game.models.exact_game import ExactInterferenceGame
from interference_game.models.targets import generate_targets


def build_game(*, mixing_depth: int = 2, shared_target: bool = False) -> ExactInterferenceGame:
    model = ModelConfig(
        num_agents=2,
        state_dim=4,
        mixing_depth=mixing_depth,
        action_budget=4.0,
        lambdas=0.08,
        mixer_seed=17,
    )
    targets = generate_targets(TargetConfig(seed=5, conflict_level="medium", shared_target=shared_target), 2, 4)
    return ExactInterferenceGame(model, targets.targets)


def test_state_normalization_and_projection_wrap() -> None:
    game = build_game(mixing_depth=2)
    actions = torch.tensor([[10.0, -10.0, 8.0, -8.0], [0.5, 0.4, 0.3, 0.2]], dtype=torch.float64)
    projected = game.project_actions(actions)
    result = game.evaluate(projected, return_intermediate=True)

    assert torch.linalg.norm(projected[0]).item() <= 2.0 + 1e-8
    assert torch.all(projected <= torch.pi + 1e-8)
    assert torch.all(projected >= -torch.pi - 1e-8)
    assert result.intermediate_states is not None
    for state in result.intermediate_states:
        assert torch.allclose(torch.linalg.norm(state), torch.tensor(1.0, dtype=state.abs().dtype), atol=1e-8)


def test_l0_and_identity_mixers_match_no_mixing() -> None:
    l0_game = build_game(mixing_depth=0)
    actions = torch.tensor([[0.2, 0.1, -0.3, 0.4], [0.5, -0.2, 0.1, -0.4]], dtype=torch.float64)
    exact_l0 = l0_game.evaluate(actions)
    nomixing_l0 = l0_game.evaluate_mode(actions, mode="no_mixing")
    assert torch.allclose(exact_l0.psi, nomixing_l0.psi, atol=1e-10)

    identity_game = build_game(mixing_depth=3)
    identity = torch.eye(identity_game.config.state_dim, dtype=identity_game.complex_dtype)
    identity_game.mixers = [identity.clone() for _ in range(identity_game.config.mixing_depth)]
    exact_identity = identity_game.evaluate(actions)
    nomixing_identity = identity_game.evaluate_mode(actions, mode="no_mixing")
    assert torch.allclose(exact_identity.psi, nomixing_identity.psi, atol=1e-10)


def test_target_overlap_and_shared_target_common_structure() -> None:
    bundle = generate_targets(TargetConfig(seed=3, conflict_level="high"), 2, 4)
    assert bundle.overlap_matrix.shape == (2, 2)
    assert (abs(bundle.overlap_matrix.diagonal() - 1.0) < 1e-10).all()

    shared_game = build_game(mixing_depth=1, shared_target=True)
    actions = torch.tensor([[0.2, 0.2, -0.1, 0.1], [0.2, 0.2, -0.1, 0.1]], dtype=torch.float64)
    result = shared_game.evaluate(actions)
    assert torch.allclose(result.fidelities, result.fidelities[0].expand_as(result.fidelities), atol=1e-10)
    assert torch.allclose(result.utilities, result.utilities[0].expand_as(result.utilities), atol=1e-10)


def test_small_equilibrium_enumeration_returns_regrets() -> None:
    model = ModelConfig(
        num_agents=2,
        state_dim=2,
        mixing_depth=0,
        action_budget=2.0,
        lambdas=0.05,
        mixer_seed=13,
    )
    targets = generate_targets(TargetConfig(seed=4, shared_target=True), 2, 2)
    game = ExactInterferenceGame(model, targets.targets)
    summary = enumerate_equilibria(game, equilibrium_config=EquilibriumConfig(epsilon=1e-3, max_profiles=256))
    assert len(summary.records) == 256
    assert (summary.records["max_regret"] >= 0).all()
    assert not summary.pure_nash.empty
