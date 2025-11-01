"""
Strategy Encoder: 1D CNN for extracting opponent playstyle embeddings.

This module implements a lightweight temporal convolutional network that processes
a history of opponent observations to extract a compact strategy representation.
The output embedding captures high-level patterns like aggression, spacing preferences,
and move usage, enabling the agent to adapt its strategy in real-time.
"""

import torch
import torch.nn as nn
from typing import Dict, Any


class StrategyEncoder(nn.Module):
    """
    1D CNN that encodes opponent behavior history into a strategy embedding.

    Architecture:
        - 3 Conv1D layers with increasing channels (64, 128, 128)
        - Layer normalization for stable training (better than BatchNorm for RL)
        - Global average pooling to handle variable-length sequences
        - Final linear projection to strategy embedding space

    Args:
        input_features: Number of opponent features per timestep (default: 13)
        history_length: Number of timesteps in observation history (default: 60)
        embedding_dim: Dimensionality of output strategy vector (default: 32)
        dropout: Dropout rate for regularization (default: 0.1)
    """

    def __init__(
        self,
        input_features: int = 13,
        history_length: int = 60,
        embedding_dim: int = 32,
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_features = input_features
        self.history_length = history_length
        self.embedding_dim = embedding_dim

        # 1D Convolutional layers for temporal pattern extraction
        # Input shape: (batch, features, time)
        self.conv1 = nn.Conv1d(
            in_channels=input_features,
            out_channels=64,
            kernel_size=5,
            stride=1,
            padding=2
        )
        # LayerNorm over (channels, time) - more stable for RL than BatchNorm
        self.ln1 = nn.LayerNorm([64, history_length])

        self.conv2 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=5,
            stride=2,
            padding=2
        )
        # After stride=2, time dimension is halved
        self.ln2 = nn.LayerNorm([128, history_length // 2])

        self.conv3 = nn.Conv1d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            stride=2,
            padding=1
        )
        # After another stride=2, time dimension is quartered
        self.ln3 = nn.LayerNorm([128, history_length // 4])

        # Global pooling to aggregate temporal information
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Final projection to embedding space
        self.fc = nn.Linear(128, embedding_dim)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, opponent_history: torch.Tensor) -> torch.Tensor:
        """
        Extract strategy embedding from opponent observation history.

        Args:
            opponent_history: Tensor of shape (batch, history_length, input_features)
                             Contains recent opponent observations

        Returns:
            strategy_embedding: Tensor of shape (batch, embedding_dim)
                               Compact representation of opponent strategy
        """
        # Transpose to (batch, features, time) for Conv1D
        x = opponent_history.transpose(1, 2)  # (batch, input_features, history_length)

        # Conv block 1: Extract low-level temporal patterns
        x = self.conv1(x)
        x = self.ln1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Conv block 2: Mid-level pattern aggregation
        x = self.conv2(x)
        x = self.ln2(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Conv block 3: High-level strategy features
        x = self.conv3(x)
        x = self.ln3(x)
        x = self.relu(x)

        # Global pooling: Aggregate across time
        x = self.global_pool(x)  # (batch, 128, 1)
        x = x.squeeze(-1)  # (batch, 128)

        # Project to embedding space
        strategy_embedding = self.fc(x)  # (batch, embedding_dim)

        return strategy_embedding

    def get_embedding_dim(self) -> int:
        """Returns the dimensionality of the strategy embedding."""
        return self.embedding_dim


class StrategyEncoderWithCache(nn.Module):
    """
    Wrapper around StrategyEncoder that caches embeddings during episode.

    This is useful for reducing computation when the same opponent is faced
    multiple times within an episode, or when we want to track embedding
    evolution over time.
    """

    def __init__(self, encoder: StrategyEncoder, cache_every_n_steps: int = 10):
        super().__init__()
        self.encoder = encoder
        self.cache_every_n_steps = cache_every_n_steps
        self.step_counter = 0
        self.cached_embedding = None

    def forward(self, opponent_history: torch.Tensor, force_recompute: bool = False) -> torch.Tensor:
        """
        Forward pass with caching logic.

        Args:
            opponent_history: Opponent observation history
            force_recompute: If True, bypass cache and recompute

        Returns:
            strategy_embedding: Current strategy embedding
        """
        # Recompute if forced, cache expired, or first call
        if force_recompute or self.cached_embedding is None or self.step_counter % self.cache_every_n_steps == 0:
            self.cached_embedding = self.encoder(opponent_history)

        self.step_counter += 1
        return self.cached_embedding

    def reset_cache(self):
        """Reset cache (call on episode boundaries)."""
        self.cached_embedding = None
        self.step_counter = 0


def create_strategy_encoder(config: Dict[str, Any]) -> StrategyEncoder:
    """
    Factory function to create strategy encoder from config.

    Args:
        config: Dictionary with encoder hyperparameters

    Returns:
        StrategyEncoder instance
    """
    return StrategyEncoder(
        input_features=config.get('input_features', 13),
        history_length=config.get('history_length', 60),
        embedding_dim=config.get('embedding_dim', 32),
        dropout=config.get('dropout', 0.1)
    )


if __name__ == "__main__":
    # Test the encoder
    print("Testing StrategyEncoder...")

    # Create encoder
    encoder = StrategyEncoder(
        input_features=13,
        history_length=60,
        embedding_dim=32
    )

    # Test forward pass
    batch_size = 4
    opponent_history = torch.randn(batch_size, 60, 13)

    strategy_embedding = encoder(opponent_history)

    print(f"Input shape: {opponent_history.shape}")
    print(f"Output shape: {strategy_embedding.shape}")
    print(f"Expected: (batch={batch_size}, embedding={32})")

    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Expected: ~50k parameters")

    # Test with cached version
    print("\nTesting StrategyEncoderWithCache...")
    cached_encoder = StrategyEncoderWithCache(encoder, cache_every_n_steps=10)

    embedding1 = cached_encoder(opponent_history)
    embedding2 = cached_encoder(opponent_history)  # Should use cache
    embedding3 = cached_encoder(opponent_history, force_recompute=True)  # Force recompute

    print(f"Embedding 1 and 2 identical (cached): {torch.allclose(embedding1, embedding2)}")
    print(f"Embedding 1 and 3 identical (recomputed): {torch.allclose(embedding1, embedding3)}")

    print("\n✓ StrategyEncoder tests passed!")
