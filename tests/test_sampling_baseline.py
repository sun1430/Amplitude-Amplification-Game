from __future__ import annotations

import torch

from interference_game.config import ModelConfig, TargetConfig
from interference_game.models.baselines import ApproximateGame
from interference_game.models.exact_game import ExactInterferenceGame
from interference_game.models.targets import generate_targets


def test_sampling_baseline_improves_with_more_draws() -> None:
    model = ModelConfig(
        num_agents=2,
        state_dim=4,
        mixing_depth=2,
        action_budget=4.0,
        lambdas=0.08,
        mixer_seed=21,
    )
    targets = generate_targets(TargetConfig(seed=6, conflict_level="high"), 2, 4)
    game = ExactInterferenceGame(model, targets.targets)
    actions = torch.tensor([[0.4, -0.3, 0.2, -0.1], [0.1, 0.3, -0.2, 0.4]], dtype=torch.float64)
    exact = game.evaluate(actions)
    low_errors = []
    high_errors = []
    for seed in range(7, 13):
        low = ApproximateGame(exact_game=game, mode="sampling", sampling_draws=64, seed=seed).evaluate(actions)
        high = ApproximateGame(exact_game=game, mode="sampling", sampling_draws=4096, seed=seed).evaluate(actions)
        low_errors.append(torch.abs(exact.utilities - low.utilities).mean().item())
        high_errors.append(torch.abs(exact.utilities - high.utilities).mean().item())

    assert sum(high_errors) / len(high_errors) <= sum(low_errors) / len(low_errors)
