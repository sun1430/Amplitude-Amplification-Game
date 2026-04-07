from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from interference_game.additive.activations import entmax15
from interference_game.additive.classical_game import ClassicalGroundTruthGame
from interference_game.additive.config import AdditiveExperimentConfig
from interference_game.additive.scoring import DistributionEvaluationResult, DistributionScoringMixin
from interference_game.utils.io import ensure_dir, write_json
from interference_game.utils.metrics import random_feasible_actions


@dataclass(slots=True)
class SurrogateTrainingArtifacts:
    checkpoint_path: Path
    metadata_path: Path


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.GELU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        x = self.norm(inputs)
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        return residual + x


class ResidualMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.block1 = ResidualBlock(hidden_dim)
        self.block2 = ResidualBlock(hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.GELU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.input(inputs))
        x = self.block1(x)
        x = self.block2(x)
        return self.output(x)


class ResidualMLPSurrogate(DistributionScoringMixin):
    def __init__(self, reference_game: ClassicalGroundTruthGame, experiment_config: AdditiveExperimentConfig):
        super().__init__(reference_game.config, reference_game.targets)
        self.reference_game = reference_game
        self.experiment_config = experiment_config
        self.network = ResidualMLP(
            input_dim=self.config.num_agents * self.config.state_dim,
            hidden_dim=experiment_config.hidden_dim,
            output_dim=self.config.state_dim,
        ).to(device=self.device, dtype=self.real_dtype)
        self.training_artifacts: SurrogateTrainingArtifacts | None = None

    def _forward_distribution(self, projected: torch.Tensor) -> torch.Tensor:
        logits = self.network(projected.reshape(1, -1)).squeeze(0)
        return entmax15(logits, dim=0)

    def _build_dataset(self, num_samples: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        actions = random_feasible_actions(
            num_samples=num_samples,
            num_agents=self.config.num_agents,
            state_dim=self.config.state_dim,
            budgets=self.config.budgets(),
            seed=seed,
        ).to(dtype=self.real_dtype, device=self.device)
        targets = torch.stack(
            [self.reference_game.evaluate(sample).influence_distribution.detach() for sample in actions],
            dim=0,
        )
        return actions, targets

    def fit(self, artifact_dir: str | Path, seed: int) -> SurrogateTrainingArtifacts:
        artifact_root = ensure_dir(artifact_dir)
        checkpoint_path = artifact_root / "checkpoint.pt"
        metadata_path = artifact_root / "metadata.json"

        if checkpoint_path.exists() and metadata_path.exists():
            try:
                state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            except TypeError:
                state_dict = torch.load(checkpoint_path, map_location=self.device)
            self.network.load_state_dict(state_dict)
            self.network.eval()
            artifacts = SurrogateTrainingArtifacts(checkpoint_path=checkpoint_path, metadata_path=metadata_path)
            self.training_artifacts = artifacts
            return artifacts

        torch.manual_seed(seed)
        train_actions, train_targets = self._build_dataset(self.experiment_config.train_samples, seed=seed)
        val_actions, val_targets = self._build_dataset(self.experiment_config.val_samples, seed=seed + 1)
        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.experiment_config.learning_rate,
            weight_decay=self.experiment_config.weight_decay,
        )

        best_state = None
        best_val = float("inf")
        patience = 0
        history: list[dict[str, float | int]] = []
        batch_size = min(self.experiment_config.batch_size, train_actions.shape[0])

        for epoch in range(self.experiment_config.epochs):
            self.network.train()
            order = torch.randperm(train_actions.shape[0], device=self.device)
            epoch_loss = 0.0
            for start in range(0, train_actions.shape[0], batch_size):
                batch_indices = order[start : start + batch_size]
                batch_actions = train_actions[batch_indices].reshape(batch_indices.shape[0], -1)
                batch_targets = train_targets[batch_indices]
                optimizer.zero_grad(set_to_none=True)
                logits = self.network(batch_actions)
                predictions = entmax15(logits, dim=-1)
                loss = torch.mean((predictions - batch_targets).square())
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item()) * batch_indices.shape[0]

            self.network.eval()
            with torch.no_grad():
                val_logits = self.network(val_actions.reshape(val_actions.shape[0], -1))
                val_predictions = entmax15(val_logits, dim=-1)
                val_loss = torch.mean((val_predictions - val_targets).square()).item()

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": epoch_loss / float(train_actions.shape[0]),
                    "val_loss": float(val_loss),
                }
            )

            if val_loss + 1e-9 < best_val:
                best_val = val_loss
                best_state = {key: value.detach().cpu().clone() for key, value in self.network.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= self.experiment_config.early_stopping_patience:
                    break

        if best_state is not None:
            self.network.load_state_dict(best_state)
        self.network.eval()
        torch.save(self.network.state_dict(), checkpoint_path)
        write_json(
            metadata_path,
            {
                "experiment_config": asdict(self.experiment_config),
                "seed": seed,
                "train_seed": seed,
                "val_seed": seed + 1,
                "train_samples": self.experiment_config.train_samples,
                "val_samples": self.experiment_config.val_samples,
                "history": history,
                "best_val_loss": best_val,
            },
        )
        artifacts = SurrogateTrainingArtifacts(checkpoint_path=checkpoint_path, metadata_path=metadata_path)
        self.training_artifacts = artifacts
        return artifacts

    def evaluate(self, actions: torch.Tensor | list[list[float]]) -> DistributionEvaluationResult:
        projected = self.project_actions(actions)
        with torch.no_grad():
            distribution = self._forward_distribution(projected)
        return self._score_distribution(distribution, projected, latent_state=distribution)
