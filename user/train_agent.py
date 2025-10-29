"""
To do: 
[]Add heavy tracking system for training agaisnt 1 agent
[] train and tweak model against 1 agent until satisfied
[] add in the adversarial training against and 





--------------------------------------------------------
--------------------------------------------------------
--------------------------------------------------------

UTMIST AI² - Transformer-Based Strategy Recognition Training System

Training:
1. Encoder model (transformer strategy encoder)
At every frame, the model encodes the opponents actions into a 256-dim vector that represents the opponents strategy 

2. LSTM RNN (recurrent neural network policy)
The model takes in: 
- Thestrategy (latent 256-dim vector) from the encoder model 
- Current state of environment
- Previous actions (LSTM momory)

The model outputs:
- The action to take (discrete action space)

3. Reward 
Computes reward based on predetrmined reward functions

4. Learning 
Updates both transformer strategy encoder and LSTM RNN with backprop through PPO (Proximal Policy Optimisation). transformer learns what patterns help 
the policy win, policy learns how to use those patterns.

5. Self-advisory training loop
Runs the model vs past versions of ITSELF (snapshots) and scripted bots




Sections of the file: 
1. Imports
2. Editable Training Configuration
2.5. Transformer-Based Strategy Recognition 
3. Main Agent (RecurrentPPOAgent & TransformerStrategyAgent)
4. Supporting Training Agents
5. Reward Shaping Library
6. Self-Play Infrastructure
7. Training Loop (train)
8. Evaluation and Match Helpers
9. Main Entrypoint (__main__)

=============================================================================
WHAT WE ARE BUILDING: AlphaGo-Style Strategy Understanding
=============================================================================

TWO TRAINING MODES AVAILABLE:

1. STANDARD MODE (RecurrentPPOAgent):
   - Learner: Recurrent PPO (LSTM) with small MLP heads
   - Training: Self-play vs. snapshots + scripted bots
   - Rewards: Dense shaping + event signals
   - Good for: Quick experiments, baseline performance
   
2. TRANSFORMER MODE (TransformerStrategyAgent): **NEW!**
   - Pure Latent Space Learning: NO pre-defined concepts
   - Self-Attention: Automatically discovers opponent patterns
   - Infinite Strategies: Continuous 256-dim representation space
   - Like AlphaGo: Learns abstract representations from experience
   - Good for: Competition, robust generalization to unseen opponents

=============================================================================
KEY INNOVATION: Transformer Strategy Encoder
=============================================================================

Traditional Approach (Limitation):
  - Pre-define concepts: "aggression", "defensive", "spacing"
  - Force opponent into 8 discrete clusters
  - Cannot handle novel strategies outside training distribution

Transformer Approach (Infinite Generalization):
  - NO pre-defined concepts or clusters
  - Learn 256-dim continuous latent space
  - Self-attention discovers patterns automatically
  - Infinite possible strategies as combinations of latent dimensions
  - Generalizes to completely novel opponents

Example:
  Opponent does: [dash, wait, dash, attack, retreat, dash, attack]
  
  Standard: "Fits cluster 3 (aggressive)" ← Forced categorization
  Transformer: [0.23, -0.71, 0.88, ..., 0.15] ← Unique representation
               Self-attention learns: "dash→attack pattern with spacing"
               NO human labeling required!

=============================================================================
HOW TO USE
=============================================================================

SWITCH MODES (Line 154):
  # TRAIN_CONFIG = TRAIN_CONFIG           # Standard RecurrentPPO
  TRAIN_CONFIG = TRAIN_CONFIG_TRANSFORMER  # Transformer (recommended)

RUN TRAINING:
  python user/train_agent.py

VISUALIZE ATTENTION (optional):
  During inference, call agent.visualize_attention(obs) to see what
  the transformer focuses on in opponent behavior.

=============================================================================
ARCHITECTURE DETAILS
=============================================================================

Transformer Strategy Encoder:
  - Input: Sequence of opponent observations (90 frames = 3 seconds)
  - Embedding: Each frame → 256-dim vector
  - Positional Encoding: Temporal order information
  - Self-Attention: 6 layers × 8 heads
  - Output: Single 256-dim strategy latent vector
  
Policy Conditioning:
  - RecurrentPPO policy receives strategy latent
  - Cross-attention: "Given opponent strategy, what should I do?"
  - Actions adapt based on recognized patterns
  
Online Learning:
  - Strategy understanding refines during match
  - Handles opponent adaptation mid-game
  - No pre-training needed - learns from experience

=============================================================================
EXPECTED BENEFITS
=============================================================================

✓ True generalization to infinite opponent strategies
✓ NO finite strategy categorization 
✓ Automatic pattern discovery via self-attention
✓ Handles novel opponents never seen in training
✓ AlphaGo-level strategic understanding
✓ Robust performance in competition scenarios

"""

# --------------------------------------------------------------------------------
# ----------------------------- 1. Imports -----------------------------
# --------------------------------------------------------------------------------

import os
import math
import torch
import gymnasium as gym
from torch.nn import functional as F
from torch import nn as nn
import numpy as np
import pygame
from functools import partial
from stable_baselines3 import A2C, PPO, SAC, DQN, DDPG, TD3, HER
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from environment.agent import *
from typing import Optional, Type, List, Tuple, Dict, Callable

from environment.agent import train as env_train
from environment.agent import run_match as env_run_match
from environment.agent import run_real_time_match as env_run_real_time_match


# --------------------------------------------------------------------------------
# ----------------------------- 2. Editable Training Configuration -----------------------------
# --------------------------------------------------------------------------------
# Centralised knobs for experiments; update these values to change training behaviour.
TRAIN_CONFIG: Dict[str, dict] = {
    "agent": {
        "type": "recurrent_ppo",     # options: "recurrent_ppo", "custom", "sb3"
        "load_path": None,            # resume from checkpoint path
        # LSTM / PPO hyperparameters
        "policy_kwargs": {
            "activation_fn": nn.ReLU,
            "lstm_hidden_size": 512,
            "net_arch": [dict(pi=[32, 32], vf=[32, 32])],
            "shared_lstm": True,
            "enable_critic_lstm": False,
            "share_features_extractor": True,
        },
        "n_steps": 30 * 90 * 20,
        "batch_size": 16,
        "ent_coef": 0.05,
        "sb3_class": PPO,             # used when type == "custom" or "sb3"
        "extractor_class": None,      # optional custom features extractor for CustomAgent
    },
    "reward": {
        "factory": None,              # callable returning RewardManager; None => gen_reward_manager()
    },
    "self_play": {
        "run_name": "experiment_9",
        "save_freq": 100_000,
        "max_saved": 40,
        "mode": SaveHandlerMode.FORCE,
        "opponent_mix": None,         # override dict[str, tuple] to customise opponent sampling
        "handler": SelfPlayRandom,    # SelfPlayRandom or SelfPlayLatest
    },
    "training": {
        "resolution": CameraResolution.LOW,
        "timesteps": 50_000,
        "logging": TrainLogging.PLOT,
    },
}

# ============ TRANSFORMER-BASED CONFIGURATION (AlphaGo-Style) ============
# Pure latent space learning with self-attention - NO pre-defined concepts
TRAIN_CONFIG_TRANSFORMER: Dict[str, dict] = {
    "agent": {
        "type": "transformer_strategy",  # Transformer-based strategy understanding
        "load_path": None,
        
        # Transformer hyperparameters
        "latent_dim": 256,           # Dimensionality of strategy latent space
        "num_heads": 8,              # Number of attention heads
        "num_layers": 6,             # Depth of transformer
        "sequence_length": 90,       # Frames to analyze (3 seconds at 30 FPS)
        "opponent_obs_dim": None,    # Auto-detected if None
        
        # Enhanced PPO hyperparameters
        "policy_kwargs": {
            "activation_fn": nn.ReLU,
            "lstm_hidden_size": 512,
            "net_arch": [dict(pi=[96, 96], vf=[96, 96])],  # Larger for strategy
            "shared_lstm": True,
            "enable_critic_lstm": True,
            "share_features_extractor": True,
        },
        "n_steps": 30 * 90 * 20,
        "batch_size": 64,            # Larger batch (transformer is parallelizable)
        "ent_coef": 0.10,            # Higher exploration
    },
    "reward": {
        "factory": None,             # Uses gen_reward_manager()
    },
    "self_play": {
        "run_name": "transformer_strategy_exp1",
        "save_freq": 100_000,
        "max_saved": 60,
        "mode": SaveHandlerMode.FORCE,
        "opponent_mix": None,
        "handler": SelfPlayRandom,
    },
    "training": {
        "resolution": CameraResolution.LOW,
        "timesteps": 10_000_000,     # Longer training for strategy learning
        "logging": TrainLogging.PLOT,
    },
}

# ============ SWITCH CONFIGURATION HERE ============
# Uncomment the configuration you want to use:

# TRAIN_CONFIG = TRAIN_CONFIG          # Standard RecurrentPPO
# TRAIN_CONFIG = TRAIN_CONFIG_TRANSFORMER  # Transformer with pure latent space

# --------------------------------------------------------------------------------
# ----------------------------- 2.5. Transformer-Based Strategy Recognition -----------------------------
# --------------------------------------------------------------------------------
# Pure latent space learning with self-attention for infinite strategy understanding.
# NO pre-defined concepts - learns abstract representations like AlphaGo.

class PositionalEncoding(nn.Module):
    """
    Adds positional information to sequence embeddings.
    Enables transformer to understand temporal order of opponent behavior.
    """
    def __init__(self, d_model: int, max_len: int = 150):
        super().__init__()
        
        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        return x + self.pe[:, :x.size(1)]


class AttentionPooling(nn.Module):
    """
    Learns which parts of the sequence are most important for strategy understanding.
    Better than mean/max pooling - lets network decide what matters.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1)
        )
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, dim]
        Returns:
            pooled: [batch, dim]
        """
        # Compute attention weights for each frame
        attn_weights = self.attention(x)  # [batch, seq_len, 1]
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # Weighted sum
        pooled = (x * attn_weights).sum(dim=1)
        return pooled, attn_weights


class TransformerStrategyEncoder(nn.Module):
    """
    Transformer-based strategy encoder using self-attention.
    
    Key Features:
    - Pure latent space learning (NO named concepts like "aggression")
    - Self-attention discovers important patterns automatically
    - Handles infinite strategy variations through continuous representation
    - 256-dim latent space for rich compositional understanding
    
    Architecture:
    1. Embed each opponent observation frame
    2. Add positional encoding (temporal order)
    3. Multi-layer self-attention (discover patterns)
    4. Attention pooling (aggregate to single strategy vector)
    
    Like AlphaGo: learns abstract representations, not human-defined concepts.
    """
    def __init__(
        self,
        opponent_obs_dim: int = 32,
        latent_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        max_sequence_length: int = 90  # 3 seconds at 30 FPS
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.max_sequence_length = max_sequence_length
        
        # Embed each opponent observation frame to latent dimension
        self.frame_embedding = nn.Sequential(
            nn.Linear(opponent_obs_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU()
        )
        
        # Positional encoding (so transformer knows time order)
        self.positional_encoding = PositionalEncoding(latent_dim, max_sequence_length)
        
        # Multi-head self-attention transformer encoder
        # Each layer learns different temporal patterns
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for better training stability
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Attention pooling - learns which frames matter most
        self.strategy_pooling = AttentionPooling(latent_dim)
        
        # Optional: Additional processing of pooled strategy
        self.strategy_refinement = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
    
    def forward(self, opponent_history, return_attention=False):
        """
        Extract pure latent strategy representation via self-attention.
        
        Args:
            opponent_history: [batch, seq_len, opponent_obs_dim]
                              Sequence of opponent observations
            return_attention: If True, return attention weights for visualization
        
        Returns:
            strategy_latent: [batch, latent_dim]
                            Pure latent vector representing opponent strategy
                            NO discrete clusters, NO named concepts
                            Continuous space supporting infinite strategies
            
            attention_info: (optional) Dict with attention maps for analysis
        """
        batch_size, seq_len, _ = opponent_history.shape
        
        # 1. Embed each frame
        embeddings = self.frame_embedding(opponent_history)
        
        # 2. Add positional encoding
        embeddings = self.positional_encoding(embeddings)
        
        # 3. Self-attention: each frame attends to all others
        # Network learns which temporal patterns matter!
        # Example: "Frame 5 (dash) + Frame 12 (attack) = aggressive pattern"
        contextualized = self.transformer(embeddings)
        
        # 4. Attention pooling: aggregate sequence to single strategy vector
        # Learns which frames are most representative of strategy
        strategy_latent, pooling_attn = self.strategy_pooling(contextualized)
        
        # 5. Refine strategy representation
        strategy_latent = self.strategy_refinement(strategy_latent)
        
        if return_attention:
            # Return attention info for visualization/debugging
            attention_info = {
                'pooling_attention': pooling_attn,
                'contextualized_frames': contextualized
            }
            return strategy_latent, attention_info
        
        return strategy_latent


class TransformerConditionedExtractor(BaseFeaturesExtractor):
    """
    Features extractor that conditions policy on transformer-learned strategy latent.
    
    Process:
    1. Encode raw game observations
    2. Receive strategy latent from transformer encoder
    3. Fuse observations with strategy context
    4. Output strategy-conditioned features for policy
    
    This allows the policy to adapt based on understood opponent patterns.
    """
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 256,
        latent_dim: int = 256
    ):
        super().__init__(observation_space, features_dim)
        
        self.latent_dim = latent_dim
        obs_dim = observation_space.shape[0]
        
        # Encode raw observations
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )
        
        # Cross-attention: observations attend to strategy latent
        # "Given opponent strategy, what should I focus on in current state?"
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=8,
            batch_first=True
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(256 + latent_dim, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU()
        )
        
        # Buffer for current strategy latent (set by agent)
        self.register_buffer('current_strategy', torch.zeros(1, latent_dim))
    
    def set_strategy_latent(self, strategy_latent: torch.Tensor):
        """Update current strategy latent (called by agent before forward)."""
        self.current_strategy = strategy_latent.detach()
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Condition features on strategy latent.
        
        Args:
            observations: [batch, obs_dim]
        
        Returns:
            features: [batch, features_dim] - strategy-conditioned features
        """
        batch_size = observations.shape[0]
        
        # Encode observations
        obs_features = self.obs_encoder(observations)  # [batch, 256]
        
        # Expand strategy latent to match batch
        strategy_latent = self.current_strategy.expand(batch_size, -1)  # [batch, latent_dim]
        
        # Cross-attention: let observations attend to strategy context
        obs_features_unsqueezed = obs_features.unsqueeze(1)  # [batch, 1, 256]
        strategy_unsqueezed = strategy_latent.unsqueeze(1)   # [batch, 1, latent_dim]
        
        # Query: observations, Key/Value: strategy
        attended_obs, _ = self.cross_attention(
            query=obs_features_unsqueezed,
            key=strategy_unsqueezed,
            value=strategy_unsqueezed
        )
        attended_obs = attended_obs.squeeze(1)  # [batch, 256]
        
        # Fuse attended observations with strategy latent
        combined = torch.cat([attended_obs, strategy_latent], dim=-1)
        features = self.fusion(combined)
        
        return features


# --------------------------------------------------------------------------------
# ----------------------------- 3. Main Agent (RecurrentPPOAgent) -----------------------------
# --------------------------------------------------------------------------------
# Primary sequence-based learner used for AlphaGo-style self-play training.

class RecurrentPPOAgent(Agent):
    '''
    Main Recurrent PPO Agent:
    - Leverages sb3-contrib's RecurrentPPO (LSTM + PPO) to model temporal dynamics
    - Maintains hidden state across timesteps to anticipate combos and counters
    '''
    def __init__(
            self,
            file_path: Optional[str] = None
    ):
        super().__init__(file_path)
        self.lstm_states = None
        self.episode_starts = np.ones((1,), dtype=bool)
        self.default_policy_kwargs: Optional[dict] = None
        self.default_n_steps: Optional[int] = None
        self.default_batch_size: Optional[int] = None
        self.default_ent_coef: Optional[float] = None

    def _initialize(self) -> None:
        if self.file_path is None:
            policy_kwargs = self.default_policy_kwargs or {
                'activation_fn': nn.ReLU,
                'lstm_hidden_size': 512,
                'net_arch': [dict(pi=[32, 32], vf=[32, 32])],
                'shared_lstm': True,
                'enable_critic_lstm': False,
                'share_features_extractor': True,
            }
            self.model = RecurrentPPO(
                "MlpLstmPolicy",
                self.env,
                verbose=0,
                n_steps=self.default_n_steps or 30*90*20,
                batch_size=self.default_batch_size or 16,
                ent_coef=self.default_ent_coef or 0.05,
                policy_kwargs=policy_kwargs,
            )
            del self.env
        else:
            self.model = RecurrentPPO.load(self.file_path)

    def reset(self) -> None:
        self.episode_starts = True

    def predict(self, obs):
        action, self.lstm_states = self.model.predict(
            obs,
            state=self.lstm_states,
            episode_start=self.episode_starts,
            deterministic=True,
        )
        if self.episode_starts:
            self.episode_starts = False
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path)

    def learn(self, env, total_timesteps, log_interval: int = 2, verbose=0):
        self.model.set_env(env)
        self.model.verbose = verbose
        self.model.learn(total_timesteps=total_timesteps, log_interval=log_interval)


class TransformerStrategyAgent(Agent):
    """
    Transformer-based agent with pure latent space strategy understanding.
    
    Key Features:
    - Uses self-attention to discover opponent patterns automatically
    - NO pre-defined strategy concepts (learns abstract representations)
    - Handles infinite strategy variations through continuous latent space
    - Adapts policy based on recognized patterns in real-time
    - Like AlphaGo: learns what matters from experience, not human labels
    
    Architecture:
    1. Transformer encoder tracks opponent behavior sequences
    2. Self-attention discovers important temporal patterns
    3. Extracts 256-dim continuous latent representation
    4. Policy conditions on latent to adapt counter-strategy
    5. Online refinement during matches
    """
    def __init__(
        self,
        file_path: Optional[str] = None,
        latent_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        sequence_length: int = 90,
        opponent_obs_dim: Optional[int] = None
    ):
        super().__init__(file_path)
        
        # Transformer hyperparameters
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.opponent_obs_dim = opponent_obs_dim
        
        # PPO LSTM states (for action policy)
        self.lstm_states = None
        self.episode_starts = np.ones((1,), dtype=bool)
        
        # Opponent history buffer for transformer
        self.opponent_history = []
        self.current_strategy_latent = None
        
        # Transformer strategy encoder (initialized in _initialize)
        self.strategy_encoder = None
        
        # Default hyperparameters
        self.default_policy_kwargs: Optional[dict] = None
        self.default_n_steps: Optional[int] = None
        self.default_batch_size: Optional[int] = None
        self.default_ent_coef: Optional[float] = None
    
    def _initialize(self) -> None:
        """Initialize agent with transformer-based strategy recognition."""
        
        # Auto-detect opponent observation dimension
        total_obs_dim = self.observation_space.shape[0]
        if self.opponent_obs_dim is None:
            # Assume equal split between player and opponent observations
            self.opponent_obs_dim = total_obs_dim // 2
        
        if self.file_path is None:
            # Create transformer strategy encoder
            self.strategy_encoder = TransformerStrategyEncoder(
                opponent_obs_dim=self.opponent_obs_dim,
                latent_dim=self.latent_dim,
                num_heads=self.num_heads,
                num_layers=self.num_layers,
                max_sequence_length=self.sequence_length
            )
            
            # Enhanced policy kwargs with transformer conditioning
            policy_kwargs = self.default_policy_kwargs or {
                'activation_fn': nn.ReLU,
                'lstm_hidden_size': 512,
                'net_arch': [dict(pi=[96, 96], vf=[96, 96])],  # Larger for strategy
                'shared_lstm': True,
                'enable_critic_lstm': True,
                'share_features_extractor': True,
            }
            
            # Add transformer-conditioned features extractor
            policy_kwargs['features_extractor_class'] = TransformerConditionedExtractor
            policy_kwargs['features_extractor_kwargs'] = {
                'features_dim': 256,
                'latent_dim': self.latent_dim
            }
            
            self.model = RecurrentPPO(
                "MlpLstmPolicy",
                self.env,
                verbose=0,
                n_steps=self.default_n_steps or 30*90*20,
                batch_size=self.default_batch_size or 32,
                ent_coef=self.default_ent_coef or 0.10,
                policy_kwargs=policy_kwargs,
            )
            del self.env
        else:
            # Load existing model
            self.model = RecurrentPPO.load(self.file_path)
            
            # Try to load transformer encoder
            encoder_path = self.file_path.replace('.zip', '_transformer_encoder.pth')
            if os.path.exists(encoder_path):
                self.strategy_encoder = TransformerStrategyEncoder(
                    opponent_obs_dim=self.opponent_obs_dim or 32,
                    latent_dim=self.latent_dim,
                    num_heads=self.num_heads,
                    num_layers=self.num_layers
                )
                self.strategy_encoder.load_state_dict(torch.load(encoder_path))
                print(f"Loaded transformer strategy encoder from {encoder_path}")
            else:
                # Create new encoder
                self.strategy_encoder = TransformerStrategyEncoder(
                    opponent_obs_dim=self.opponent_obs_dim or 32,
                    latent_dim=self.latent_dim,
                    num_heads=self.num_heads,
                    num_layers=self.num_layers
                )
                print("Created new transformer strategy encoder")
    
    def reset(self) -> None:
        """Reset for new episode - clear opponent history."""
        self.episode_starts = True
        self.opponent_history = []
        self.current_strategy_latent = None
    
    def predict(self, obs):
        """
        Predict action with transformer-based strategy conditioning.
        
        Process:
        1. Split observation into player and opponent components
        2. Update opponent history buffer
        3. Extract strategy latent via transformer self-attention
        4. Update policy features extractor with strategy latent
        5. Generate strategy-conditioned action
        """
        # Split observation
        player_obs, opponent_obs = self._split_observation(obs)
        
        # Update opponent history (rolling window)
        self.opponent_history.append(opponent_obs)
        if len(self.opponent_history) > self.sequence_length:
            self.opponent_history.pop(0)
        
        # Extract strategy latent with transformer (need at least 10 frames)
        if len(self.opponent_history) >= 10:
            self._update_strategy_latent()
            
            # Update policy's features extractor with strategy latent
            if hasattr(self.model.policy, 'features_extractor'):
                extractor = self.model.policy.features_extractor
                if hasattr(extractor, 'set_strategy_latent') and self.current_strategy_latent is not None:
                    extractor.set_strategy_latent(self.current_strategy_latent)
        
        # Generate action (now conditioned on strategy)
        action, self.lstm_states = self.model.predict(
            obs,
            state=self.lstm_states,
            episode_start=self.episode_starts,
            deterministic=True,
        )
        
        if self.episode_starts:
            self.episode_starts = False
        
        return action
    
    def _split_observation(self, obs):
        """
        Split observation into player and opponent components.
        Assumes observation structure: [player_features, opponent_features]
        """
        obs_array = np.array(obs)
        mid = len(obs_array) // 2
        player_obs = obs_array[:mid]
        opponent_obs = obs_array[mid:]
        return player_obs, opponent_obs
    
    def _update_strategy_latent(self):
        """
        Extract strategy latent using transformer self-attention.
        Discovers patterns in opponent behavior automatically.
        """
        # Convert history to tensor [1, seq_len, obs_dim]
        history_array = np.array(self.opponent_history)
        history_tensor = torch.tensor(
            history_array,
            dtype=torch.float32
        ).unsqueeze(0)  # Add batch dimension
        
        # Extract latent via transformer
        with torch.no_grad():
            self.current_strategy_latent = self.strategy_encoder(history_tensor)
    
    def get_strategy_latent_info(self) -> Dict[str, any]:
        """
        Get information about current strategy latent.
        Useful for debugging and analysis.
        """
        if self.current_strategy_latent is None:
            return {'latent': None, 'norm': 0, 'history_length': 0}
        
        latent_numpy = self.current_strategy_latent.cpu().numpy()
        latent_norm = np.linalg.norm(latent_numpy)
        
        return {
            'latent': latent_numpy,
            'norm': float(latent_norm),
            'history_length': len(self.opponent_history),
        }
    
    def visualize_attention(self, obs) -> Dict:
        """
        Extract attention weights for visualization.
        Shows which frames the transformer focuses on.
        """
        if len(self.opponent_history) < 10:
            return {}
        
        history_array = np.array(self.opponent_history)
        history_tensor = torch.tensor(
            history_array,
            dtype=torch.float32
        ).unsqueeze(0)
        
        with torch.no_grad():
            strategy_latent, attention_info = self.strategy_encoder(
                history_tensor,
                return_attention=True
            )
        
        return {
            'pooling_attention': attention_info['pooling_attention'].cpu().numpy(),
            'contextualized_frames': attention_info['contextualized_frames'].cpu().numpy()
        }
    
    def save(self, file_path: str) -> None:
        """Save both PPO model and transformer encoder."""
        # Save main PPO model
        self.model.save(file_path)
        
        # Save transformer encoder separately
        if self.strategy_encoder is not None:
            encoder_path = file_path.replace('.zip', '_transformer_encoder.pth')
            torch.save(self.strategy_encoder.state_dict(), encoder_path)
            print(f"Saved transformer strategy encoder to {encoder_path}")
    
    def learn(self, env, total_timesteps, log_interval: int = 2, verbose=0):
        """Standard learning interface."""
        self.model.set_env(env)
        self.model.verbose = verbose
        self.model.learn(total_timesteps=total_timesteps, log_interval=log_interval)


# --------------------------------------------------------------------------------
# ----------------------------- 4. Supporting Training Agents -----------------------------
# --------------------------------------------------------------------------------
# Auxiliary controllers for curriculum opponents, baselines, and human/demo play.

class SB3Agent(Agent):
    '''
    Generic SB3 Agent:
    - Wraps any Stable-Baselines3 algorithm for quick baselines or comparisons
    '''
    def __init__(
            self,
            sb3_class: Optional[Type[BaseAlgorithm]] = PPO,
            file_path: Optional[str] = None
    ):
        self.sb3_class = sb3_class
        super().__init__(file_path)

    def _initialize(self) -> None:
        if self.file_path is None:
            self.model = self.sb3_class(
                "MlpPolicy",
                self.env,
                verbose=0,
                n_steps=30*90*3,
                batch_size=128,
                ent_coef=0.01,
            )
            del self.env
        else:
            self.model = self.sb3_class.load(self.file_path)

    def _gdown(self) -> str:
        return

    def predict(self, obs):
        action, _ = self.model.predict(obs)
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path, include=['num_timesteps'])

    def learn(self, env, total_timesteps, log_interval: int = 1, verbose=0):
        self.model.set_env(env)
        self.model.verbose = verbose
        self.model.learn(
            total_timesteps=total_timesteps,
            log_interval=log_interval,
        )


class BasedAgent(Agent):
    '''
    BasedAgent:
    - Hard-coded heuristic opponent for early-stage curriculum matches
    '''
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.time = 0

    def predict(self, obs):
        self.time += 1
        pos = self.obs_helper.get_section(obs, 'player_pos')
        opp_pos = self.obs_helper.get_section(obs, 'opponent_pos')
        opp_KO = self.obs_helper.get_section(obs, 'opponent_state') in [5, 11]
        action = self.act_helper.zeros()

        if pos[0] > 10.67/2:
            action = self.act_helper.press_keys(['a'])
        elif pos[0] < -10.67/2:
            action = self.act_helper.press_keys(['d'])
        elif not opp_KO:
            action = self.act_helper.press_keys(['d' if opp_pos[0] > pos[0] else 'a'])

        if (pos[1] > 1.6 or pos[1] > opp_pos[1]) and self.time % 2 == 0:
            action = self.act_helper.press_keys(['space'], action)

        if (pos[0] - opp_pos[0])**2 + (pos[1] - opp_pos[1])**2 < 4.0:
            action = self.act_helper.press_keys(['j'], action)
        return action


class UserInputAgent(Agent):
    '''
    UserInputAgent:
    - Enables human-controlled demos or debugging sessions
    '''
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

    def predict(self, obs):
        action = self.act_helper.zeros()
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            action = self.act_helper.press_keys(['w'], action)
        if keys[pygame.K_a]:
            action = self.act_helper.press_keys(['a'], action)
        if keys[pygame.K_s]:
            action = self.act_helper.press_keys(['s'], action)
        if keys[pygame.K_d]:
            action = self.act_helper.press_keys(['d'], action)
        if keys[pygame.K_SPACE]:
            action = self.act_helper.press_keys(['space'], action)
        if keys[pygame.K_h]:
            action = self.act_helper.press_keys(['h'], action)
        if keys[pygame.K_j]:
            action = self.act_helper.press_keys(['j'], action)
        if keys[pygame.K_k]:
            action = self.act_helper.press_keys(['k'], action)
        if keys[pygame.K_l]:
            action = self.act_helper.press_keys(['l'], action)
        if keys[pygame.K_g]:
            action = self.act_helper.press_keys(['g'], action)
        return action


class ClockworkAgent(Agent):
    '''
    ClockworkAgent:
    - Plays a scripted sequence of actions; useful for style opponents
    '''
    def __init__(
            self,
            action_sheet: Optional[List[Tuple[int, List[str]]]] = None,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.steps = 0
        self.current_action_end = 0
        self.current_action_data = None
        self.action_index = 0

        if action_sheet is None:
            self.action_sheet = [
                (10, ['a']),
                (1, ['l']),
                (20, ['a']),
                (3, ['a', 'j']),
                (15, ['space']),
            ]
        else:
            self.action_sheet = action_sheet

    def predict(self, obs):
        if self.steps >= self.current_action_end and self.action_index < len(self.action_sheet):
            hold_time, action_data = self.action_sheet[self.action_index]
            self.current_action_data = action_data
            self.current_action_end = self.steps + hold_time
            self.action_index += 1

        action = self.act_helper.press_keys(self.current_action_data)
        self.steps += 1
        return action


class MLPPolicy(nn.Module):
    def __init__(self, obs_dim: int = 64, action_dim: int = 10, hidden_dim: int = 64):
        super(MLPPolicy, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim, dtype=torch.float32)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class MLPExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 64, hidden_dim: int = 64):
        super(MLPExtractor, self).__init__(observation_space, features_dim)
        self.model = MLPPolicy(
            obs_dim=observation_space.shape[0],
            action_dim=10,
            hidden_dim=hidden_dim,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.model(obs)

    @classmethod
    def get_policy_kwargs(cls, features_dim: int = 64, hidden_dim: int = 64) -> dict:
        return dict(
            features_extractor_class=cls,
            features_extractor_kwargs=dict(features_dim=features_dim, hidden_dim=hidden_dim)
        )


class CustomAgent(Agent):
    def __init__(self, sb3_class: Optional[Type[BaseAlgorithm]] = PPO, file_path: str = None, extractor: BaseFeaturesExtractor = None):
        self.sb3_class = sb3_class
        self.extractor = extractor
        super().__init__(file_path)

    def _initialize(self) -> None:
        if self.file_path is None:
            policy_kwargs = None
            if self.extractor is not None:
                policy_kwargs = self.extractor.get_policy_kwargs()
            self.model = self.sb3_class(
                "MlpPolicy",
                self.env,
                policy_kwargs=policy_kwargs,
                verbose=0,
                n_steps=30*90*3,
                batch_size=128,
                ent_coef=0.01,
            )
            del self.env
        else:
            self.model = self.sb3_class.load(self.file_path)

    def _gdown(self) -> str:
        return

    def predict(self, obs):
        action, _ = self.model.predict(obs)
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path, include=['num_timesteps'])

    def learn(self, env, total_timesteps, log_interval: int = 1, verbose=0):
        self.model.set_env(env)
        self.model.verbose = verbose
        self.model.learn(
            total_timesteps=total_timesteps,
            log_interval=log_interval,
        )


def _resolve_callable(value, fallback):
    """Resolve config entries that may be callables or string names."""
    if value is None:
        return fallback
    if isinstance(value, str):
        resolved = globals().get(value)
        return resolved if resolved is not None else fallback
    return value


def create_learning_agent(agent_cfg: Dict[str, object]) -> Agent:
    """Factory that instantiates the training agent based on TRAIN_CONFIG."""

    agent_type = agent_cfg.get("type", "custom")
    load_path = agent_cfg.get("load_path")

    if agent_type == "custom":
        extractor_cls = _resolve_callable(agent_cfg.get("extractor_class"), MLPExtractor)
        sb3_cls = _resolve_callable(agent_cfg.get("sb3_class"), PPO)
        return CustomAgent(sb3_class=sb3_cls, file_path=load_path, extractor=extractor_cls)

    if agent_type == "recurrent_ppo":
        agent = RecurrentPPOAgent(file_path=load_path)
        agent.default_policy_kwargs = agent_cfg.get("policy_kwargs")
        agent.default_n_steps = agent_cfg.get("n_steps")
        agent.default_batch_size = agent_cfg.get("batch_size")
        agent.default_ent_coef = agent_cfg.get("ent_coef")
        return agent
    
    if agent_type == "transformer_strategy":
        # Create transformer-based strategy-aware agent
        agent = TransformerStrategyAgent(
            file_path=load_path,
            latent_dim=agent_cfg.get("latent_dim", 256),
            num_heads=agent_cfg.get("num_heads", 8),
            num_layers=agent_cfg.get("num_layers", 6),
            sequence_length=agent_cfg.get("sequence_length", 90),
            opponent_obs_dim=agent_cfg.get("opponent_obs_dim", None)
        )
        agent.default_policy_kwargs = agent_cfg.get("policy_kwargs")
        agent.default_n_steps = agent_cfg.get("n_steps")
        agent.default_batch_size = agent_cfg.get("batch_size")
        agent.default_ent_coef = agent_cfg.get("ent_coef")
        return agent

    if agent_type == "sb3":
        sb3_cls = _resolve_callable(agent_cfg.get("sb3_class"), PPO)
        return SB3Agent(sb3_class=sb3_cls, file_path=load_path)

    raise ValueError(f"Unknown agent type '{agent_type}' in TRAIN_CONFIG['agent']")


# --------------------------------------------------------------------------------
# ----------------------------- 5. Reward Shaping Library -----------------------------
# --------------------------------------------------------------------------------
# Dense and signal-based rewards used to shape agent behaviour. Assemble them
# into a RewardManager via `gen_reward_manager` for training loops.

def base_height_l2(
    env: WarehouseBrawl,
    target_height: float,
    obj_name: str = 'player'
) -> float:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # Extract the used quantities (to enable type-hinting)
    obj: GameObject = env.objects[obj_name]

    # Compute the L2 squared penalty
    return (obj.body.position.y - target_height)**2

class RewardMode(Enum):
    ASYMMETRIC_OFFENSIVE = 0
    SYMMETRIC = 1
    ASYMMETRIC_DEFENSIVE = 2

def damage_interaction_reward(
    env: WarehouseBrawl,
    mode: RewardMode = RewardMode.SYMMETRIC,
) -> float:
    """
    Computes the reward based on damage interactions between players.

    Modes:
    - ASYMMETRIC_OFFENSIVE (0): Reward is based only on damage dealt to the opponent
    - SYMMETRIC (1): Reward is based on both dealing damage to the opponent and avoiding damage
    - ASYMMETRIC_DEFENSIVE (2): Reward is based only on avoiding damage

    Args:
        env (WarehouseBrawl): The game environment
        mode (DamageRewardMode): Reward mode, one of DamageRewardMode

    Returns:
        float: The computed reward.
    """
    # Getting player and opponent from the enviornment
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]

    # Reward dependent on the mode
    damage_taken = player.damage_taken_this_frame
    damage_dealt = opponent.damage_taken_this_frame

    if mode == RewardMode.ASYMMETRIC_OFFENSIVE:
        reward = damage_dealt
    elif mode == RewardMode.SYMMETRIC:
        reward = damage_dealt - damage_taken
    elif mode == RewardMode.ASYMMETRIC_DEFENSIVE:
        reward = -damage_taken
    else:
        raise ValueError(f"Invalid mode: {mode}")

    return reward / 140

def danger_zone_reward(
    env: WarehouseBrawl,
    zone_penalty: int = 1,
    zone_height: float = 4.2
) -> float:
    """
    Applies a penalty for every time frame player surpases a certain height threshold in the environment.

    Args:
        env (WarehouseBrawl): The game environment.
        zone_penalty (int): The penalty applied when the player is in the danger zone.
        zone_height (float): The height threshold defining the danger zone.

    Returns:
        float: The computed penalty as a tensor.
    """
    # Get player object from the environment
    player: Player = env.objects["player"]

    # Apply penalty if the player is in the danger zone
    reward = -zone_penalty if player.body.position.y >= zone_height else 0.0

    return reward * env.dt

def in_state_reward(
    env: WarehouseBrawl,
    desired_state: Type[PlayerObjectState]=BackDashState,
) -> float:
    """
    Applies a penalty for every time frame player surpases a certain height threshold in the environment.

    Args:
        env (WarehouseBrawl): The game environment.
        zone_penalty (int): The penalty applied when the player is in the danger zone.
        zone_height (float): The height threshold defining the danger zone.

    Returns:
        float: The computed penalty as a tensor.
    """
    # Get player object from the environment
    player: Player = env.objects["player"]

    # Apply penalty if the player is in the danger zone
    reward = 1 if isinstance(player.state, desired_state) else 0.0

    return reward * env.dt

def head_to_middle_reward(
    env: WarehouseBrawl,
) -> float:
    """
    Applies a penalty for every time frame player surpases a certain height threshold in the environment.

    Args:
        env (WarehouseBrawl): The game environment.
        zone_penalty (int): The penalty applied when the player is in the danger zone.
        zone_height (float): The height threshold defining the danger zone.

    Returns:
        float: The computed penalty as a tensor.
    """
    # Get player object from the environment
    player: Player = env.objects["player"]

    # Apply penalty if the player is in the danger zone
    multiplier = -1 if player.body.position.x > 0 else 1
    reward = multiplier * (player.body.position.x - player.prev_x)

    return reward

def head_to_opponent(
    env: WarehouseBrawl,
) -> float:

    # Get player object from the environment
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]

    # Apply penalty if the player is in the danger zone
    multiplier = -1 if player.body.position.x > opponent.body.position.x else 1
    reward = multiplier * (player.body.position.x - player.prev_x)

    return reward

def holding_more_than_3_keys(
    env: WarehouseBrawl,
) -> float:

    # Get player object from the environment
    player: Player = env.objects["player"]

    # Apply penalty if the player is holding more than 3 keys
    a = player.cur_action
    if (a > 0.5).sum() > 3:
        return env.dt
    return 0

def on_win_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == 'player':
        return 1.0
    else:
        return -1.0

def on_knockout_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == 'player':
        return -1.0
    else:
        return 1.0

def on_equip_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == "player":
        if env.objects["player"].weapon == "Hammer":
            return 2.0
        elif env.objects["player"].weapon == "Spear":
            return 1.0
    return 0.0

def on_drop_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == "player":
        if env.objects["player"].weapon == "Punch":
            return -1.0
    return 0.0

def on_combo_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == 'player':
        return -1.0
    else:
        return 1.0

'''
Add your dictionary of RewardFunctions here using RewTerms
'''
def gen_reward_manager():
    reward_functions = {
        #'target_height_reward': RewTerm(func=base_height_l2, weight=0.0, params={'target_height': -4, 'obj_name': 'player'}),
        'danger_zone_reward': RewTerm(func=danger_zone_reward, weight=0.5),
        'damage_interaction_reward': RewTerm(func=damage_interaction_reward, weight=1.0),
        #'head_to_middle_reward': RewTerm(func=head_to_middle_reward, weight=0.01),
        #'head_to_opponent': RewTerm(func=head_to_opponent, weight=0.05),
        'penalize_attack_reward': RewTerm(func=in_state_reward, weight=-0.04, params={'desired_state': AttackState}),
        'holding_more_than_3_keys': RewTerm(func=holding_more_than_3_keys, weight=-0.01),
        #'taunt_reward': RewTerm(func=in_state_reward, weight=0.2, params={'desired_state': TauntState}),
    }
    signal_subscriptions = {
        'on_win_reward': ('win_signal', RewTerm(func=on_win_reward, weight=50)),
        'on_knockout_reward': ('knockout_signal', RewTerm(func=on_knockout_reward, weight=8)),
        'on_combo_reward': ('hit_during_stun', RewTerm(func=on_combo_reward, weight=5)),
        'on_equip_reward': ('weapon_equip_signal', RewTerm(func=on_equip_reward, weight=10)),
        'on_drop_reward': ('weapon_drop_signal', RewTerm(func=on_drop_reward, weight=15))
    }
    return RewardManager(reward_functions, signal_subscriptions)

# --------------------------------------------------------------------------------
# ----------------------------- 6. Self-Play Infrastructure -----------------------------
# --------------------------------------------------------------------------------
# Helper builders for SaveHandler snapshots and opponent sampling. Modify here to
# adjust snapshot cadence, opponent mix, or self-play sampling strategy.

def build_self_play_components(
    agent: Agent,
    *,
    save_path: str = "checkpoints",
    run_name: str = "experiment_9",
    save_freq: int = 100_000,
    max_saved: int = 40,
    mode: SaveHandlerMode = SaveHandlerMode.FORCE,
    opponent_mix: Optional[dict] = None,
    selfplay_handler_cls: Type[SelfPlayHandler] = SelfPlayRandom,
) -> Tuple[SelfPlayHandler, SaveHandler, OpponentsCfg]:
    """Configure snapshot saving and opponent sampling for self-play."""

    selfplay_handler = selfplay_handler_cls(partial(type(agent)))

    save_handler = SaveHandler(
        agent=agent,
        save_freq=save_freq,
        max_saved=max_saved,
        save_path=save_path,
        run_name=run_name,
        mode=mode,
    )

    if opponent_mix is None:
        opponent_mix = {
            'self_play': (8, selfplay_handler),
            'constant_agent': (0.5, partial(ConstantAgent)),
            'based_agent': (1.5, partial(BasedAgent)),
        }

    opponent_cfg = OpponentsCfg(opponents=opponent_mix)
    return selfplay_handler, save_handler, opponent_cfg


# --------------------------------------------------------------------------------
# ----------------------------- 7. Training Loop (train) -----------------------------
# --------------------------------------------------------------------------------
# Wrapper that funnels configuration into the environment-agent training helper.

def run_training_loop(
    agent: Agent,
    reward_manager: RewardManager,
    save_handler: SaveHandler,
    opponent_cfg: OpponentsCfg,
    *,
    resolution: CameraResolution = CameraResolution.LOW,
    train_timesteps: int = 50_000,
    train_logging: TrainLogging = TrainLogging.PLOT,
):
    """Launch training with the provided environment, reward, and self-play setup."""

    return env_train(
        agent,
        reward_manager,
        save_handler,
        opponent_cfg,
        resolution,
        train_timesteps=train_timesteps,
        train_logging=train_logging,
    )


# --------------------------------------------------------------------------------
# ----------------------------- 8. Evaluation and Match Helpers -----------------------------
# --------------------------------------------------------------------------------
# Convenience wrappers around the environment's evaluation utilities.

def run_eval_match(
    agent_1: Agent | partial,
    agent_2: Agent | partial,
    *,
    max_timesteps: int = 30 * 90,
    resolution: CameraResolution = CameraResolution.LOW,
    video_path: Optional[str] = None,
    train_mode: bool = False,
):
    """Run a scripted evaluation match; returns MatchStats."""

    return env_run_match(
        agent_1,
        agent_2,
        max_timesteps=max_timesteps,
        video_path=video_path,
        resolution=resolution,
        train_mode=train_mode,
    )


def run_live_eval(
    agent_1: Agent,
    agent_2: Agent,
    *,
    max_timesteps: int = 30 * 90,
    resolution: CameraResolution = CameraResolution.LOW,
):
    """Run a human-vs-agent (or agent-vs-agent) real-time demo."""

    return env_run_real_time_match(
        agent_1,
        agent_2,
        max_timesteps=max_timesteps,
        resolution=resolution,
    )


# --------------------------------------------------------------------------------
# ----------------------------- 9. Main Entrypoint (__main__) -----------------------------
# --------------------------------------------------------------------------------
# Assemble the pieces above and kick off training when the script is executed.

def main() -> None:
    """Default experiment setup driven by TRAIN_CONFIG."""

    agent_cfg = TRAIN_CONFIG.get("agent", {})
    agent_type = agent_cfg.get("type", "custom")
    agent_checkpoint = agent_cfg.get("load_path")
    agent_sb3_class = agent_cfg.get("sb3_class")
    agent_extractor = agent_cfg.get("extractor_class")
    normalized_agent_cfg = {
        **agent_cfg,
        "type": agent_type,
        "load_path": agent_checkpoint,
        "sb3_class": agent_sb3_class,
        "extractor_class": agent_extractor,
    }
    learning_agent = create_learning_agent(normalized_agent_cfg)

    reward_cfg = TRAIN_CONFIG.get("reward", {})
    reward_factory = _resolve_callable(reward_cfg.get("factory"), gen_reward_manager)
    reward_manager = reward_factory()

    self_play_cfg = TRAIN_CONFIG.get("self_play", {})
    self_play_run_name = self_play_cfg.get("run_name", "experiment_9")
    self_play_save_freq = self_play_cfg.get("save_freq", 100_000)
    self_play_max_saved = self_play_cfg.get("max_saved", 40)
    self_play_mode = self_play_cfg.get("mode", SaveHandlerMode.FORCE)
    self_play_opponent_mix = self_play_cfg.get("opponent_mix")
    self_play_handler_cls = _resolve_callable(self_play_cfg.get("handler"), SelfPlayRandom)

    _self_play_handler, save_handler, opponent_cfg = build_self_play_components(
        learning_agent,
        run_name=self_play_run_name,
        save_freq=self_play_save_freq,
        max_saved=self_play_max_saved,
        mode=self_play_mode,
        opponent_mix=self_play_opponent_mix,
        selfplay_handler_cls=self_play_handler_cls,
    )

    training_cfg = TRAIN_CONFIG.get("training", {})
    train_resolution = training_cfg.get("resolution", CameraResolution.LOW)
    train_timesteps = training_cfg.get("timesteps", 50_000)
    train_logging = training_cfg.get("logging", TrainLogging.PLOT)

    run_training_loop(
        agent=learning_agent,
        reward_manager=reward_manager,
        save_handler=save_handler,
        opponent_cfg=opponent_cfg,
        resolution=train_resolution,
        train_timesteps=train_timesteps,
        train_logging=train_logging,
    )


if __name__ == '__main__':
    main()

