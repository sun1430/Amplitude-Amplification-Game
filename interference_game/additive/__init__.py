"""Additive distribution-scoring mainline models and experiments."""

from interference_game.additive.classical_game import ClassicalGroundTruthGame
from interference_game.additive.quantum_game import QuantumEncodedGame
from interference_game.additive.surrogate import ResidualMLPSurrogate

__all__ = [
    "ClassicalGroundTruthGame",
    "QuantumEncodedGame",
    "ResidualMLPSurrogate",
]
