"""
Opponent-Conditioned Policy: RecurrentPPO policy that conditions on opponent strategy.

This module extends the standard RecurrentPPO policy to include strategy encoding.
The policy processes both agent observations (through standard feature extraction)
and opponent history (through a strategy encoder), then concatenates them before
feeding into the LSTM.
"""

import sys
from pathlib import Path

# Add project root to path for imports
if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import gymnasium as gym
from typing import Dict, List, Tuple, Type, Union
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import get_flattened_obs_dim

from user.models.strategy_encoder import StrategyEncoder


class OpponentConditionedFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor that processes both agent observations and opponent history.

    Architecture:
        - Agent observations → Base feature extractor (ResidualMLP) → 512D
        - Opponent history → Strategy encoder (1D CNN) → 32D
        - Concatenate → 544D output

    Args:
        observation_space: Gymnasium observation space
        base_extractor_class: Class for base feature extraction (e.g., WarehouseFeatureExtractor)
        base_extractor_kwargs: Kwargs for base extractor
        strategy_encoder_config: Config dict for strategy encoder
        features_dim: Output dimension (default: 544 = 512 + 32)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        base_extractor_class: Type[BaseFeaturesExtractor],
        base_extractor_kwargs: Dict,
        strategy_encoder_config: Dict,
        features_dim: int = 544,
    ):
        # The output dimension is base_features + strategy_embedding
        super().__init__(observation_space, features_dim)

        self.base_feature_dim = base_extractor_kwargs.get('feature_dim', 512)
        self.strategy_embedding_dim = strategy_encoder_config.get('embedding_dim', 32)

        # Verify output dimension matches
        assert features_dim == self.base_feature_dim + self.strategy_embedding_dim, \
            f"features_dim ({features_dim}) must equal base_feature_dim ({self.base_feature_dim}) + " \
            f"strategy_embedding_dim ({self.strategy_embedding_dim})"

        # Calculate dimensions
        # We expect observation to be: [agent_obs, flattened_opponent_history]
        self.opponent_history_length = strategy_encoder_config.get('history_length', 60)
        self.opponent_feature_dim = strategy_encoder_config.get('input_features', 13)
        self.opponent_history_flat_dim = self.opponent_history_length * self.opponent_feature_dim

        total_obs_dim = get_flattened_obs_dim(observation_space)
        self.agent_obs_dim = total_obs_dim - self.opponent_history_flat_dim

        print(f"  OpponentConditionedFeatureExtractor:")
        print(f"    - Agent obs dim: {self.agent_obs_dim}")
        print(f"    - Opponent history dim: {self.opponent_history_flat_dim} "
              f"({self.opponent_history_length} × {self.opponent_feature_dim})")
        print(f"    - Base features: {self.base_feature_dim}D")
        print(f"    - Strategy embedding: {self.strategy_embedding_dim}D")
        print(f"    - Total output: {features_dim}D")

        # Create base feature extractor for agent observations
        # We need to create a dummy observation space for just the agent obs
        agent_obs_space = gym.spaces.Box(
            low=-float('inf'),
            high=float('inf'),
            shape=(self.agent_obs_dim,),
            dtype=observation_space.dtype
        )
        self.base_extractor = base_extractor_class(
            observation_space=agent_obs_space,
            **base_extractor_kwargs
        )

        # Create strategy encoder for opponent history
        self.strategy_encoder = StrategyEncoder(
            input_features=self.opponent_feature_dim,
            history_length=self.opponent_history_length,
            embedding_dim=self.strategy_embedding_dim,
            dropout=strategy_encoder_config.get('dropout', 0.1)
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Extract features from observations.

        Args:
            observations: Tensor of shape (batch, total_obs_dim)
                         where total_obs_dim = agent_obs_dim + opponent_history_flat_dim

        Returns:
            features: Tensor of shape (batch, features_dim)
        """
        # Split observation into agent obs and opponent history
        agent_obs = observations[:, :self.agent_obs_dim]
        opponent_history_flat = observations[:, self.agent_obs_dim:]

        # Process agent observations through base extractor
        agent_features = self.base_extractor(agent_obs)  # (batch, base_feature_dim)

        # Reshape opponent history for strategy encoder
        # From (batch, history_length * feature_dim) to (batch, history_length, feature_dim)
        batch_size = opponent_history_flat.shape[0]
        opponent_history = opponent_history_flat.view(
            batch_size,
            self.opponent_history_length,
            self.opponent_feature_dim
        )

        # Extract strategy embedding
        strategy_embedding = self.strategy_encoder(opponent_history)  # (batch, strategy_embedding_dim)

        # Concatenate features
        combined_features = torch.cat([agent_features, strategy_embedding], dim=1)

        return combined_features


class WarehouseFeatureExtractorWrapper(BaseFeaturesExtractor):
    """
    Wrapper to make existing WarehouseFeatureExtractor compatible.
    This is needed because we create it with a modified observation space.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        feature_dim: int = 512,
        num_residual_blocks: int = 5,
        dropout: float = 0.08,
    ):
        super().__init__(observation_space, feature_dim)

        obs_dim = get_flattened_obs_dim(observation_space)

        # Initial projection with LayerNorm
        self.input_proj = nn.Sequential(
            nn.LayerNorm(obs_dim),
            nn.Linear(obs_dim, feature_dim),
            nn.GELU(),
        )

        # Stack of residual blocks
        self.residual_stack = nn.ModuleList([
            ResidualMLPBlock(feature_dim, expansion=3, dropout=dropout)
            for _ in range(num_residual_blocks)
        ])

        # Output normalization
        self.output_norm = nn.LayerNorm(feature_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(observations)
        for block in self.residual_stack:
            x = block(x)
        return self.output_norm(x)


class ResidualMLPBlock(nn.Module):
    """Residual MLP block with LayerNorm and GELU activation."""

    def __init__(self, dim: int, expansion: int = 3, dropout: float = 0.08):
        super().__init__()
        hidden_dim = dim * expansion
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


def create_opponent_conditioned_policy_kwargs(
    base_extractor_kwargs: Dict,
    strategy_encoder_config: Dict,
    lstm_hidden_size: int = 512,
    n_lstm_layers: int = 3,
    net_arch: Dict = None,
    **other_policy_kwargs
) -> Dict:
    """
    Create policy_kwargs for RecurrentPPO with opponent conditioning.

    Args:
        base_extractor_kwargs: Config for base feature extractor (WarehouseFeatureExtractor)
        strategy_encoder_config: Config for strategy encoder
        lstm_hidden_size: LSTM hidden units
        n_lstm_layers: Number of LSTM layers
        net_arch: Architecture for actor/critic heads
        **other_policy_kwargs: Additional policy kwargs (optimizer, activation, etc.)

    Returns:
        policy_kwargs dict ready for RecurrentPPO
    """
    base_feature_dim = base_extractor_kwargs.get('feature_dim', 512)
    strategy_embedding_dim = strategy_encoder_config.get('embedding_dim', 32)
    combined_feature_dim = base_feature_dim + strategy_embedding_dim

    if net_arch is None:
        net_arch = dict(pi=[512, 256], vf=[512, 256])

    policy_kwargs = {
        'features_extractor_class': OpponentConditionedFeatureExtractor,
        'features_extractor_kwargs': {
            'base_extractor_class': WarehouseFeatureExtractorWrapper,
            'base_extractor_kwargs': base_extractor_kwargs,
            'strategy_encoder_config': strategy_encoder_config,
            'features_dim': combined_feature_dim,
        },
        'lstm_hidden_size': lstm_hidden_size,
        'n_lstm_layers': n_lstm_layers,
        'net_arch': net_arch,
        'shared_lstm': False,
        'enable_critic_lstm': True,
        'share_features_extractor': True,
    }

    # Add other kwargs (optimizer, activation, etc.)
    policy_kwargs.update(other_policy_kwargs)

    return policy_kwargs


if __name__ == "__main__":
    print("Testing OpponentConditionedFeatureExtractor...")

    # Test configuration
    agent_obs_dim = 52  # Example: WarehouseBrawl has ~52 features for agent+opponent
    history_length = 60
    opponent_feature_dim = 13
    total_obs_dim = agent_obs_dim + (history_length * opponent_feature_dim)

    # Create observation space
    obs_space = gym.spaces.Box(
        low=-float('inf'),
        high=float('inf'),
        shape=(total_obs_dim,),
        dtype='float32'
    )

    # Create extractor
    base_extractor_kwargs = {
        'feature_dim': 512,
        'num_residual_blocks': 5,
        'dropout': 0.08,
    }

    strategy_encoder_config = {
        'input_features': opponent_feature_dim,
        'history_length': history_length,
        'embedding_dim': 32,
        'dropout': 0.1,
    }

    extractor = OpponentConditionedFeatureExtractor(
        observation_space=obs_space,
        base_extractor_class=WarehouseFeatureExtractorWrapper,
        base_extractor_kwargs=base_extractor_kwargs,
        strategy_encoder_config=strategy_encoder_config,
        features_dim=544,
    )

    # Test forward pass
    batch_size = 4
    test_obs = torch.randn(batch_size, total_obs_dim)

    features = extractor(test_obs)

    print(f"\n✓ Forward pass successful!")
    print(f"  Input shape: {test_obs.shape}")
    print(f"  Output shape: {features.shape}")
    print(f"  Expected output: (batch={batch_size}, features=544)")

    # Count parameters
    total_params = sum(p.numel() for p in extractor.parameters())
    print(f"\n  Total parameters: {total_params:,}")

    print("\n✓ OpponentConditionedFeatureExtractor tests passed!")
