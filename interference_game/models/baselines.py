from __future__ import annotations

from dataclasses import dataclass

from interference_game.models.exact_game import EvaluationResult, ExactInterferenceGame


BASELINE_NAMES = ("no_mixing", "aggregate", "mean_field", "sampling")


@dataclass(slots=True)
class ApproximateGame:
    exact_game: ExactInterferenceGame
    mode: str
    sampling_draws: int = 256
    seed: int = 0

    def evaluate(self, actions, return_intermediate: bool = False) -> EvaluationResult:
        if self.mode == "sampling":
            return self.exact_game.evaluate_sampling(actions, num_draws=self.sampling_draws, seed=self.seed)
        return self.exact_game.evaluate_mode(actions, mode=self.mode, return_intermediate=return_intermediate)


def build_baselines(exact_game: ExactInterferenceGame, sampling_draws: int, seed: int) -> dict[str, ApproximateGame]:
    return {
        "no_mixing": ApproximateGame(exact_game=exact_game, mode="no_mixing"),
        "aggregate": ApproximateGame(exact_game=exact_game, mode="aggregate"),
        "mean_field": ApproximateGame(exact_game=exact_game, mode="mean_field"),
        "sampling": ApproximateGame(exact_game=exact_game, mode="sampling", sampling_draws=sampling_draws, seed=seed),
    }
