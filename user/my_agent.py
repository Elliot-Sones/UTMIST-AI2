# SUBMISSION: Strategy Encoder Agent
# This agent uses a RecurrentPPO model with opponent strategy conditioning.
# All required classes are defined within this file per submission requirements.

import os
import gdown
from typing import Optional, Literal, Dict, Type
from environment.agent import Agent
from stable_baselines3 import PPO, A2C
from sb3_contrib import RecurrentPPO
from stable_baselines3.common import utils as sb3_utils
import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import get_flattened_obs_dim

# Compatibility for older SB3 checkpoints
if not hasattr(sb3_utils, "FloatSchedule"):
    class FloatSchedule:
        def __init__(self, value):
            self.value = float(value)

        def __call__(self, _: float) -> float:
            return self.value

    sb3_utils.FloatSchedule = FloatSchedule


class RuleBasedAgent(Agent):
    """Deterministic fallback agent driven by simple heuristics."""

    def __init__(self):
        super().__init__(file_path=None)
        self._ticks = 0
        self._stage_half_width = 10.67 / 2  # keep players inside the arena

    def _gdown(self) -> Optional[str]:
        # No assets required for this agent.
        return None

    def _initialize(self) -> None:
        # Rule agent keeps no additional state.
        return

    def predict(self, obs):
        self._ticks += 1
        action = self.act_helper.zeros()

        # Pull structured data from the observation helper.
        my_pos = self.obs_helper.get_section(obs, "player_pos")
        opp_pos = self.obs_helper.get_section(obs, "opponent_pos")
        opp_state = self.obs_helper.get_section(obs, "opponent_state")

        # If we're drifting off stage, prioritise recovering.
        if my_pos[0] > self._stage_half_width:
            action = self.act_helper.press_keys(["a"], action)
        elif my_pos[0] < -self._stage_half_width:
            action = self.act_helper.press_keys(["d"], action)
        else:
            opponent_down = opp_state in (5, 11)
            if not opponent_down:
                horiz_gap = opp_pos[0] - my_pos[0]
                action = self.act_helper.press_keys(["d" if horiz_gap > 0 else "a"], action)

        # Soft jump cadence to contest air or recover from drops.
        should_jump = my_pos[1] > 1.6 or opp_pos[1] > my_pos[1]
        if should_jump and self._ticks % 8 == 0:
            action = self.act_helper.press_keys(["space"], action)

        # Throw quick attacks when in melee range.
        if abs(opp_pos[0] - my_pos[0]) < 1.5 and abs(opp_pos[1] - my_pos[1]) < 1.0:
            action = self.act_helper.press_keys(["j"], action)

        return action


# ============================================================================
# STRATEGY ENCODER CLASSES (copied for submission compliance)
# ============================================================================

class StrategyEncoder(nn.Module):
    """
    1D CNN that encodes opponent behavior history into a strategy embedding.
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
        self.conv1 = nn.Conv1d(input_features, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        # Global pooling to aggregate temporal information
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Final projection to embedding space
        self.fc = nn.Linear(128, embedding_dim)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, opponent_history: torch.Tensor) -> torch.Tensor:
        """
        Extract strategy embedding from opponent observation history.
        """
        # Transpose to (batch, features, time) for Conv1D
        x = opponent_history.transpose(1, 2)

        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        # Global pooling
        x = self.global_pool(x)
        x = x.squeeze(-1)

        # Project to embedding space
        strategy_embedding = self.fc(x)
        return strategy_embedding


class WarehouseFeatureExtractorWrapper(BaseFeaturesExtractor):
    """
    Wrapper for base feature extraction with residual blocks.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        feature_dim: int = 512,
        num_residual_blocks: int = 5,
        dropout: float = 0.08,
    ):
        super().__init__(observation_space, feature_dim)

        self.feature_dim = feature_dim
        self.num_residual_blocks = num_residual_blocks

        # Initial projection
        self.input_proj = nn.Linear(observation_space.shape[0], feature_dim)

        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(feature_dim, feature_dim),
            )
            for _ in range(num_residual_blocks)
        ])

        self.final_norm = nn.LayerNorm(feature_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(observations)

        for block in self.residual_blocks:
            residual = x
            x = block(x)
            x = x + residual  # Residual connection

        x = self.final_norm(x)
        return x


class OpponentConditionedFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor that processes both agent observations and opponent history.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        base_extractor_class: Type[BaseFeaturesExtractor],
        base_extractor_kwargs: Dict,
        strategy_encoder_config: Dict,
        features_dim: int = 544,
    ):
        super().__init__(observation_space, features_dim)

        self.base_feature_dim = base_extractor_kwargs.get('feature_dim', 512)
        self.strategy_embedding_dim = strategy_encoder_config.get('embedding_dim', 32)

        # Calculate dimensions
        self.opponent_history_length = strategy_encoder_config.get('history_length', 60)
        self.opponent_feature_dim = strategy_encoder_config.get('input_features', 13)
        self.opponent_history_flat_dim = self.opponent_history_length * self.opponent_feature_dim

        total_obs_dim = get_flattened_obs_dim(observation_space)
        self.agent_obs_dim = total_obs_dim - self.opponent_history_flat_dim

        # Create base feature extractor for agent observations
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
        """
        # Split observation into agent obs and opponent history
        agent_obs = observations[:, :self.agent_obs_dim]
        opponent_history_flat = observations[:, self.agent_obs_dim:]

        # Process agent observations through base extractor
        agent_features = self.base_extractor(agent_obs)

        # Reshape opponent history for strategy encoder
        batch_size = opponent_history_flat.shape[0]
        opponent_history = opponent_history_flat.view(
            batch_size,
            self.opponent_history_length,
            self.opponent_feature_dim
        )

        # Extract strategy embedding
        strategy_embedding = self.strategy_encoder(opponent_history)

        # Concatenate features
        combined_features = torch.cat([agent_features, strategy_embedding], dim=1)

        return combined_features


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

    policy_kwargs.update(other_policy_kwargs)
    return policy_kwargs


class SubmittedAgent(Agent):
    """
    Strategy Encoder Agent for UTMIST AI² Tournament.

    This agent uses RecurrentPPO with opponent strategy conditioning to adapt
    its behavior based on detected opponent patterns. Falls back to rule-based
    behavior if model loading fails.
    """

    def __init__(
        self,
        file_path: Optional[str] = None,
        mode: Literal["rl", "rules"] = "rl",
    ):
        self.mode = mode.lower()
        if self.mode not in {"rl", "rules"}:
            raise ValueError(f"Unsupported agent mode '{mode}'")

        super().__init__(file_path if self.mode == "rl" else None)

        # Model handles (can be RecurrentPPO or PPO)
        self.model = None  # Type: Optional[Union[RecurrentPPO, PPO]]
        self._rule_agent: Optional[RuleBasedAgent] = None

        # Strategy encoder configuration
        self._strategy_config = {
            'input_features': 13,     # opponent features tracked
            'history_length': 60,     # 2 seconds at 30 FPS
            'embedding_dim': 32,      # strategy embedding size
            'dropout': 0.1,
        }

        self._base_config = {
            'feature_dim': 512,
            'num_residual_blocks': 5,
            'dropout': 0.08,
        }

    def _initialize(self) -> None:
        """Initialize the agent with environment info."""
        if self.mode == "rules":
            self._rule_agent = RuleBasedAgent()
            self._rule_agent.get_env_info(self.env)
            return

        if self.file_path:
            try:
                self.model = self._load_trained_model(self.file_path)
                print("✓ Model loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load model from {self.file_path}: {e}")
                print("🔄 Falling back to rule-based agent...")
                self.mode = "rules"
                if self.env:
                    self._rule_agent = RuleBasedAgent()
                    self._rule_agent.get_env_info(self.env)
                return
        else:
            # Create untrained model (for training from scratch)
            self.model = self._build_default_model()

    def _build_default_model(self) -> RecurrentPPO:
        """Create a RecurrentPPO model with strategy encoder."""
        assert self.env is not None, "Environment must be set before building the model."

        policy_kwargs = create_opponent_conditioned_policy_kwargs(
            base_extractor_kwargs=self._base_config,
            strategy_encoder_config=self._strategy_config,
            lstm_hidden_size=512,
            n_lstm_layers=3,
            net_arch=dict(pi=[512, 256], vf=[512, 256]),
        )

        model = RecurrentPPO(
            policy="MlpLstmPolicy",
            env=self.env,
            policy_kwargs=policy_kwargs,
            n_steps=4096,
            batch_size=1024,
            n_epochs=6,
            learning_rate=3e-4,
            ent_coef=0.02,
            gamma=0.995,
            gae_lambda=0.98,
            max_grad_norm=1.5,
            vf_coef=0.5,
            verbose=0,
        )

        return model

    def _load_trained_model(self, file_path: str) -> RecurrentPPO:
        """Load a trained model, trying different formats."""
        try:
            # First try RecurrentPPO (for strategy encoder models)
            print(f"Attempting to load as RecurrentPPO: {file_path}")
            return RecurrentPPO.load(file_path)
        except Exception as e1:
            print(f"RecurrentPPO load failed: {e1}")
            try:
                # Fall back to regular PPO
                print(f"Attempting to load as regular PPO: {file_path}")
                model = PPO.load(file_path)
                print("Warning: Loaded regular PPO model instead of RecurrentPPO")
                return model
            except Exception as e2:
                print(f"Regular PPO load also failed: {e2}")
                print("Model loading failed completely")
                raise e2

    def _gdown(self) -> Optional[str]:
        """Return path to trained model for testing."""
        if self.mode == "rules":
            return None

        # Use working checkpoint from test_run directory
        model_path = "checkpoints/test_run/rl_model_15003_steps.zip"
        if os.path.isfile(model_path):
            print(f"Using trained model: {model_path}")
            return model_path
        else:
            print(f"Model not found: {model_path}")
            print("Falling back to rule-based agent...")
            self.mode = "rules"
            return None

    def predict(self, obs):
        """Choose an action for the current observation."""
        if self.mode == "rules" or self.model is None:
            if self._rule_agent is None:
                self._rule_agent = RuleBasedAgent()
                if hasattr(self, 'env'):
                    self._rule_agent.get_env_info(self.env)
            return self._rule_agent.predict(obs)

        try:
            action, _ = self.model.predict(obs, deterministic=True)
            return action
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            print("🔄 Falling back to rule-based agent for this prediction...")
            if self._rule_agent is None:
                self._rule_agent = RuleBasedAgent()
                if hasattr(self, 'env'):
                    self._rule_agent.get_env_info(self.env)
            return self._rule_agent.predict(obs)

    def save(self, file_path: str) -> None:
        """Persist the model."""
        if self.mode == "rules":
            raise RuntimeError("Rule-based agent does not support saving checkpoints.")
        self.model.save(file_path)

    def learn(self, env, total_timesteps, log_interval: int = 4):
        """Train the model."""
        if self.mode == "rules":
            raise RuntimeError("Rule-based agent cannot be trained.")

        self.model.set_env(env)
        self.model.learn(total_timesteps=total_timesteps, log_interval=log_interval)
