"""
Version: new optimisation
--------------------------------------------------------
🚀 QUICK START FOR GOOGLE COLAB
--------------------------------------------------------

For quick experiments (< 2 hours) with auto-download to desktop:
1. See COLAB_QUICK_START.md for step-by-step guide
2. Or copy-paste COLAB_ONE_CELL_SETUP.py into a Colab cell
3. Results auto-download to your Downloads folder every 15 minutes!

For long training (> 2 hours):
- Remove QUICK_EXPERIMENT flag to use Google Drive storage
- Files persist even if Colab disconnects

--------------------------------------------------------
Logic overview
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



=============================================================================
WHAT WE ARE BUILDING: AlphaGo-Style Strategy Understanding
=============================================================================

TRAINING MODE:

• TRANSFORMER MODE (TransformerStrategyAgent):
  - Pure Latent Space Learning: NO pre-defined concepts
  - Self-Attention: Automatically discovers opponent patterns
  - Infinite Strategies: Continuous 256-dim representation space
  - Like AlphaGo: Learns abstract representations from experience
  - Good for: Competition, robust generalization to unseen opponents



"""

# --------------------------------------------------------------------------------
# ----------------------------- 1. Imports -----------------------------
# --------------------------------------------------------------------------------
import os
import sys

# CRITICAL: Set MPS fallback BEFORE importing torch
# This enables CPU fallback for unsupported MPS operations (like linalg.qr)
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import math
import logging
import random
import torch

# --------------------------------------------------------------------------------
# ----------------------------- Google Colab Auto-Setup -----------------------------
# --------------------------------------------------------------------------------
def setup_colab_environment():
    """
    Automatically detects Google Colab and sets up storage.
    
    Storage Strategy:
    - Quick experiments (<2 hours): Local storage for speed, auto-download CSVs
    - Long training (>2 hours): Google Drive for persistence across disconnects
    
    Returns:
        checkpoint_path: Path to use for checkpoints
    
    Environment Variables:
        QUICK_EXPERIMENT=1  : Use local storage (fast, no Drive needed)
        USE_DRIVE=1        : Force Google Drive storage (persistent)
    
    Note:
        google.colab only exists in Colab environment - linter warnings are expected
        when running locally. The try/except handles this gracefully.
    """
    try:
        # Check if running in Colab
        import google.colab  # type: ignore  # Only available in Colab environment
        IN_COLAB = True
        print("=" * 70)
        print("🔍 Google Colab detected!")
        print("=" * 70)
    except ImportError:
        IN_COLAB = False
    
    if IN_COLAB:
        # Check for quick experiment mode
        use_quick_mode = os.environ.get('QUICK_EXPERIMENT', '0') == '1'
        force_drive = os.environ.get('USE_DRIVE', '0') == '1'
        
        if use_quick_mode and not force_drive:
            # Quick experiment mode - use local storage
            print("⚡ QUICK EXPERIMENT MODE")
            print("  ℹ️  Using local storage (fast, no Drive needed)")
            print("  ℹ️  Perfect for experiments < 2 hours")
            print("  ℹ️  Use colab_helper.py to auto-download results")
            print("  ⚠️  Checkpoints will be lost if Colab disconnects")
            checkpoint_path = "/tmp/checkpoints"
            os.makedirs(checkpoint_path, exist_ok=True)
            print(f"  ✓ Checkpoints saving to: {checkpoint_path}")
            print("=" * 70 + "\n")
            return checkpoint_path
        
        # Long training mode - use Google Drive
        print("📁 Setting up Google Drive for persistent checkpoint storage...")
        
        # Check if Drive is already mounted
        if not os.path.exists('/content/drive/MyDrive'):
            try:
                from google.colab import drive  # type: ignore  # Only available in Colab
                print("  ⏳ Mounting Google Drive (you may need to authorize)...")
                drive.mount('/content/drive')
                print("  ✓ Google Drive mounted successfully!")
            except Exception as e:
                print(f"  ⚠️  Warning: Could not mount Drive: {e}")
                print("  ℹ️  Falling back to local storage")
                checkpoint_path = "/tmp/checkpoints"
                os.makedirs(checkpoint_path, exist_ok=True)
                return checkpoint_path
        else:
            print("  ✓ Google Drive already mounted!")
        
        # Use Drive path for checkpoints
        checkpoint_path = "/content/drive/MyDrive/UTMIST-AI2-Checkpoints"
        os.makedirs(checkpoint_path, exist_ok=True)
        
        print(f"  ✓ Checkpoints will auto-save to: {checkpoint_path}")
        print(f"  ✓ Your checkpoints are safe even if Colab disconnects!")
        print("=" * 70 + "\n")
        
        return checkpoint_path
    else:
        # Local machine - use relative path
        return "checkpoints"

# Auto-setup for Colab (runs once at import)
CHECKPOINT_BASE_PATH = setup_colab_environment()
import gymnasium as gym
from torch.nn import functional as F
from torch import nn as nn
import numpy as np
import pygame
from functools import partial
from pathlib import Path
from stable_baselines3 import A2C, PPO, SAC, DQN, DDPG, TD3, HER
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# --------------------------------------------------------------------------------
# ----------------------------- GPU Device Configuration -----------------------------
# --------------------------------------------------------------------------------
# Configure PyTorch to use the best available GPU for training
# Optimized for NVIDIA T4 GPU (16GB VRAM) and Apple Silicon MPS
def get_torch_device():
    """
    Automatically detects and returns the best available device for PyTorch.
    Priority: CUDA (NVIDIA) > MPS (Apple Silicon) > CPU
    
    T4 GPU Optimization:
    - 16GB VRAM available
    - Excellent for transformer + LSTM training
    - Supports mixed precision training (FP16)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ Using NVIDIA CUDA GPU: {gpu_name}")
        print(f"  ℹ️  Total VRAM: {total_memory:.1f} GB")
        print(f"  ℹ️  CUDA Version: {torch.version.cuda}")
        
        # CUDA-specific optimizations
        torch.backends.cudnn.benchmark = True  # Enable cuDNN autotuner
        torch.set_default_dtype(torch.float32)
        
        return device
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"✓ Using Apple Silicon MPS GPU for acceleration")
        print(f"  ℹ️  MPS fallback enabled for unsupported operations")
        
        # MPS-specific optimizations
        torch.set_default_dtype(torch.float32)  # MPS works best with float32
        
        return device
    else:
        device = torch.device("cpu")
        print(f"⚠ No GPU detected, using CPU (training will be VERY slow)")
        return device

# Global device for all PyTorch operations
TORCH_DEVICE = get_torch_device()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environment.agent import *
from typing import Optional, Type, List, Tuple, Dict, Callable

from environment.agent import train as env_train
from environment.agent import run_match as env_run_match
from environment.agent import run_real_time_match as env_run_real_time_match


# --------------------------------------------------------------------------------
# ----------------------------- Debug Controls -----------------------------
# --------------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

DEBUG_FLAGS: Dict[str, bool] = {
    "config": True,
    "agent_init": True,
    "strategy_latent": False,
    "opponent_selection": True,
    "reward_terms": False,
    "training_loop": True,
}


def debug_log(flag: str, message: str) -> None:
    if DEBUG_FLAGS.get(flag, False):
        logging.info("[%s] %s", flag, message)


def _to_float(value) -> float:
    if isinstance(value, (float, int)):
        return float(value)
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def attach_reward_debug(manager: RewardManager, *, steps: int = 5) -> RewardManager:
    """Patch RewardManager.process to emit per-term debugging information."""

    manager._debug_steps_remaining = steps  # type: ignore[attr-defined]
    original_reset = manager.reset

    def debug_process(self: RewardManager, env, dt):
        reward_buffer = 0.0
        breakdown: List[str] = []
        if self.reward_functions is not None:
            for name, term_cfg in self.reward_functions.items():
                if term_cfg.weight == 0.0:
                    continue
                value = term_cfg.func(env, **term_cfg.params) * term_cfg.weight
                reward_buffer += value
                if DEBUG_FLAGS.get("reward_terms", False) and self._debug_steps_remaining > 0:  # type: ignore[attr-defined]
                    breakdown.append(f"{name}={_to_float(value):.4f}")

        signal_reward = self.collected_signal_rewards
        reward = reward_buffer + signal_reward
        self.collected_signal_rewards = 0.0
        self.total_reward += reward

        log = env.logger[0]
        log['reward'] = f'{reward_buffer:.3f}'
        log['total_reward'] = f'{self.total_reward:.3f}'
        env.logger[0] = log

        if DEBUG_FLAGS.get("reward_terms", False) and self._debug_steps_remaining > 0:  # type: ignore[attr-defined]
            breakdown.append(f"signals={_to_float(signal_reward):.4f}")
            breakdown.append(f"total={_to_float(reward):.4f}")
            debug_log("reward_terms", ", ".join(breakdown))
            self._debug_steps_remaining -= 1  # type: ignore[attr-defined]
        return reward

    manager.process = debug_process.__get__(manager, manager.__class__)  # type: ignore[method-assign]
    def debug_reset(self: RewardManager):
        original_reset()
        self._debug_steps_remaining = steps  # type: ignore[attr-defined]

    manager.reset = debug_reset.__get__(manager, manager.__class__)  # type: ignore[method-assign]
    return manager


def attach_opponent_debug(opponent_cfg: OpponentsCfg) -> OpponentsCfg:
    """Patch OpponentsCfg to log opponent sampling decisions."""

    def debug_on_env_reset(self: OpponentsCfg) -> Agent:
        agent_name = random.choices(
            list(self.opponents.keys()),
            weights=[
                prob if isinstance(prob, float) else prob[0]
                for prob in self.opponents.values()
            ],
        )[0]

        debug_log("opponent_selection", f"Selected opponent '{agent_name}'")
        if agent_name == "self_play":
            selfplay_handler: SelfPlayHandler = self.opponents[agent_name][1]
            opponent = selfplay_handler.get_opponent()
            debug_log("opponent_selection", "Loaded self-play opponent snapshot")
        else:
            opponent = self.opponents[agent_name][1]()
            debug_log(
                "opponent_selection",
                f"Instantiated scripted opponent '{type(opponent).__name__}'",
            )

        opponent.get_env_info(self.env)
        return opponent

    opponent_cfg.on_env_reset = debug_on_env_reset.__get__(opponent_cfg, opponent_cfg.__class__)  # type: ignore[method-assign]
    return opponent_cfg


# --------------------------------------------------------------------------------
# ----------------------------- 2. Editable Training Configuration -----------------------------
# --------------------------------------------------------------------------------
# Centralised knobs for experiments; update these values to change training behaviour.

# ============================================================================
# TRAINING CONFIGURATIONS - FOLLOWING SCALING LAWS
# ============================================================================
# All configurations use IDENTICAL hyperparameters for proper scaling behavior
# Only timesteps and checkpoint frequencies differ between test and full training

# ------------ SHARED HYPERPARAMETERS (DO NOT MODIFY INDIVIDUALLY) ------------
# These hyperparameters are identical across all configs to ensure scaling laws hold
_SHARED_AGENT_CONFIG = {
        "type": "transformer_strategy",  # Transformer-based strategy understanding
        "load_path": None,
        
    # Transformer hyperparameters (optimized for T4 GPU - 16GB VRAM)
        "latent_dim": 256,           # Dimensionality of strategy latent space
    "num_heads": 8,              # Number of attention heads (divisor of latent_dim)
    "num_layers": 6,             # Depth of transformer encoder
        "sequence_length": 90,       # Frames to analyze (3 seconds at 30 FPS)
    "opponent_obs_dim": None,    # Auto-detected from observation space
        
    # RecurrentPPO hyperparameters (optimized for T4 GPU)
        "policy_kwargs": {
            "activation_fn": nn.ReLU,
        "lstm_hidden_size": 512,              # LSTM hidden state size
        "net_arch": dict(pi=[96, 96], vf=[96, 96]),  # Actor/Critic network sizes
        "shared_lstm": True,                  # Share LSTM between actor and critic
        "enable_critic_lstm": False,          # Critic uses MLP only
        "share_features_extractor": True,     # Share feature extraction
    },
    
    # PPO training hyperparameters
    "n_steps": 2048,             # 🔧 FIXED: Was 54,000 (too large for 50k training!)
                                 # Now 2048 = standard PPO rollout size
                                 # Allows ~24 learning updates during 50k training
    "batch_size": 128,           # Batch size (safe for T4 16GB VRAM)
    "n_epochs": 10,              # Gradient epochs per rollout
    "ent_coef": 0.10,            # Entropy coefficient (exploration)
    "learning_rate": 2.5e-4,     # Learning rate (standard PPO)
}

_SHARED_REWARD_CONFIG = {
    "factory": None,  # Uses gen_reward_manager() by default
}

# ============ 10M CONFIGURATION (FULL TRAINING ON T4 GPU) ============
# Full training with self-play, curriculum, and adversarial learning
# Estimated time on T4: ~10-12 hours for 10M timesteps
TRAIN_CONFIG_10M: Dict[str, dict] = {
    "agent": _SHARED_AGENT_CONFIG.copy(),
    "reward": _SHARED_REWARD_CONFIG.copy(),
    "self_play": {
        "run_name": "transformer_10M_t4",  # Experiment name
        "save_freq": 100_000,              # Save checkpoint every 100k steps
        "max_saved": 100,                  # Keep last 100 checkpoints (10M worth)
        "mode": SaveHandlerMode.FORCE,     # Always save at intervals
        
        # Self-play opponent mix (with self-play enabled)
        "opponent_mix": {
            "self_play": (8.0, None),              # 80% self-play vs past snapshots
            "based_agent": (1.5, partial(BasedAgent)),  # 15% scripted opponent
            "constant_agent": (0.5, partial(ConstantAgent)),  # 5% random baseline
        },
        "handler": SelfPlayRandom,  # Randomly sample from past snapshots
    },
    "training": {
        "resolution": CameraResolution.LOW,
        "timesteps": 10_000_000,   # 10 million timesteps (~10-12 hours on T4)
        "logging": TrainLogging.PLOT,
        "enable_debug": False,     # Disable debug mode for full training
    },
}

# ============ 50K TEST CONFIGURATION (SCALING LAW TEST) ============
# Quick test run following 1:200 scaling ratio to validate configuration
# Estimated time on T4: ~15 minutes
# If this works well, 10M training will work identically (just 200x longer)
TRAIN_CONFIG_TEST: Dict[str, dict] = {
    "agent": _SHARED_AGENT_CONFIG.copy(),
    "reward": _SHARED_REWARD_CONFIG.copy(),
    "self_play": {
        "run_name": "test_50k_t4",         # Separate folder for test runs
        "save_freq": 5_000,                # Save every 5k steps (10 checkpoints total)
        "max_saved": 10,                   # Keep last 10 checkpoints
        "mode": SaveHandlerMode.FORCE,
        
        # Same opponent mix as 10M (CRITICAL for scaling laws)
        "opponent_mix": {
            "self_play": (8.0, None),              # 80% self-play vs past snapshots
            "based_agent": (1.5, partial(BasedAgent)),  # 15% scripted opponent
            "constant_agent": (0.5, partial(ConstantAgent)),  # 5% random baseline
        },
        "handler": SelfPlayRandom,
    },
    "training": {
        "resolution": CameraResolution.LOW,
        "timesteps": 50_000,       # 50k timesteps (1:200 ratio, ~15 minutes)
        "logging": TrainLogging.PLOT,
        
        # Enhanced debugging for test runs
        "enable_debug": True,      # Enable reward tracking, behavior metrics, etc.
        "eval_freq": 10_000,       # Evaluate every 10k steps (5 times total)
        "eval_episodes": 3,        # Run 3 matches per evaluation
    },
}

# ============ SWITCH CONFIGURATION HERE ============
# Toggle between test and full training
# TRAIN_CONFIG = TRAIN_CONFIG_10M    # For full 10M training on T4 GPU
TRAIN_CONFIG = TRAIN_CONFIG_TEST   # For fast 50k test runs



# --------------------------------------------------------------------------------
# ----------------------------- 3. Encoder-Based Strategy Recognition -----------------------------
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
        max_sequence_length: int = 90,  # 3 seconds at 30 FPS
        device: torch.device = None
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.max_sequence_length = max_sequence_length
        # Use provided device or fall back to global TORCH_DEVICE
        self.device = device if device is not None else TORCH_DEVICE
        
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
        
        # Move entire model to the target device (MPS/CUDA/CPU)
        self.to(self.device)
    
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
        # Ensure input is on the correct device (important for MPS)
        if not isinstance(opponent_history, torch.Tensor):
            opponent_history = torch.tensor(opponent_history, dtype=torch.float32, device=self.device)
        elif opponent_history.device != self.device:
            opponent_history = opponent_history.to(self.device)
        
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
        latent_dim: int = 256,
        device: torch.device = None
    ):
        super().__init__(observation_space, features_dim)
        
        self.latent_dim = latent_dim
        # Use provided device or fall back to global TORCH_DEVICE
        self.device = device if device is not None else TORCH_DEVICE
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
        # Initialize on the correct device
        self.register_buffer('current_strategy', torch.zeros(1, latent_dim, device=self.device))
        
        # Move entire model to the target device (MPS/CUDA/CPU)
        self.to(self.device)
    
    def set_strategy_latent(self, strategy_latent: torch.Tensor):
        """Update current strategy latent (called by agent before forward)."""
        # Ensure strategy latent is on the correct device
        if strategy_latent.device != self.device:
            strategy_latent = strategy_latent.to(self.device)
        self.current_strategy = strategy_latent.detach()
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Condition features on strategy latent.
        
        Args:
            observations: [batch, obs_dim]
        
        Returns:
            features: [batch, features_dim] - strategy-conditioned features
        """
        # Ensure observations are on the correct device (important for MPS)
        if observations.device != self.device:
            observations = observations.to(self.device)
        
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
# ----------------------------- 3. Main Agent (TransformerStrategyAgent) -----------------------------
# --------------------------------------------------------------------------------
# Primary sequence-based learner used for AlphaGo-style self-play training.

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
        debug_log(
            "agent_init",
            (
                "TransformerStrategyAgent _initialize "
                f"(total_obs_dim={total_obs_dim}, opponent_dim={self.opponent_obs_dim}, "
                f"sequence_length={self.sequence_length})"
            ),
        )
        
        if self.file_path is None:
            # Create transformer strategy encoder with device support (MPS/CUDA/CPU)
            self.strategy_encoder = TransformerStrategyEncoder(
                opponent_obs_dim=self.opponent_obs_dim,
                latent_dim=self.latent_dim,
                num_heads=self.num_heads,
                num_layers=self.num_layers,
                max_sequence_length=self.sequence_length,
                device=TORCH_DEVICE
            )
            debug_log("agent_init", f"Transformer encoder initialized on device: {TORCH_DEVICE}")
            
            # Enhanced policy kwargs with transformer conditioning
            policy_kwargs = self.default_policy_kwargs or {
                'activation_fn': nn.ReLU,
                'lstm_hidden_size': 512,
                'net_arch': dict(pi=[96, 96], vf=[96, 96]),  # Larger for strategy
                'shared_lstm': True,
                'enable_critic_lstm': False,
                'share_features_extractor': True,
            }
            
            # Add transformer-conditioned features extractor with device support
            policy_kwargs['features_extractor_class'] = TransformerConditionedExtractor
            policy_kwargs['features_extractor_kwargs'] = {
                'features_dim': 256,
                'latent_dim': self.latent_dim,
                'device': TORCH_DEVICE
            }
            debug_log(
                "agent_init",
                (
                    "Creating new RecurrentPPO "
                    f"(n_steps={self.default_n_steps or 30*90*20}, "
                    f"batch_size={self.default_batch_size or 32}, "
                    f"ent_coef={self.default_ent_coef or 0.10})"
                ),
            )
            
            self.model = RecurrentPPO(
                "MlpLstmPolicy",
                self.env,
                verbose=0,
                n_steps=self.default_n_steps or 30*90*20,
                batch_size=self.default_batch_size or 128,
                ent_coef=self.default_ent_coef or 0.10,
                policy_kwargs=policy_kwargs,
                device=TORCH_DEVICE,  # Use MPS/CUDA/CPU device
            )
            debug_log("agent_init", f"RecurrentPPO model initialized on device: {TORCH_DEVICE}")
            del self.env
        else:
            # Load existing model with device mapping for MPS/CUDA/CPU
            self.model = RecurrentPPO.load(self.file_path, device=TORCH_DEVICE)
            print(f"✓ Loaded RecurrentPPO model from {self.file_path} to {TORCH_DEVICE}")
            debug_log("agent_init", f"Loaded existing RecurrentPPO from {self.file_path} on {TORCH_DEVICE}")
            
            # Try to load transformer encoder
            encoder_path = self.file_path.replace('.zip', '_transformer_encoder.pth')
            if os.path.exists(encoder_path):
                self.strategy_encoder = TransformerStrategyEncoder(
                    opponent_obs_dim=self.opponent_obs_dim or 32,
                    latent_dim=self.latent_dim,
                    num_heads=self.num_heads,
                    num_layers=self.num_layers,
                    device=TORCH_DEVICE
                )
                # Load weights with proper device mapping for MPS/CUDA/CPU
                state_dict = torch.load(encoder_path, map_location=TORCH_DEVICE)
                self.strategy_encoder.load_state_dict(state_dict)
                print(f"✓ Loaded transformer strategy encoder from {encoder_path} to {TORCH_DEVICE}")
                debug_log("agent_init", f"Loaded transformer encoder weights from {encoder_path} on {TORCH_DEVICE}")
            else:
                # Create new encoder with device support
                self.strategy_encoder = TransformerStrategyEncoder(
                    opponent_obs_dim=self.opponent_obs_dim or 32,
                    latent_dim=self.latent_dim,
                    num_heads=self.num_heads,
                    num_layers=self.num_layers,
                    device=TORCH_DEVICE
                )
                print(f"✓ Created new transformer strategy encoder on {TORCH_DEVICE}")
                debug_log("agent_init", f"Transformer encoder weights not found; initialized new encoder on {TORCH_DEVICE}")
    
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
        # Convert history to tensor [1, seq_len, obs_dim] on the correct device
        history_array = np.array(self.opponent_history)
        history_tensor = torch.tensor(
            history_array,
            dtype=torch.float32,
            device=TORCH_DEVICE  # Ensure tensor is on MPS/CUDA/CPU
        ).unsqueeze(0)  # Add batch dimension
        
        # Extract latent via transformer (already on correct device)
        with torch.no_grad():
            latent_tensor = self.strategy_encoder(history_tensor)
        self.current_strategy_latent = latent_tensor
        if DEBUG_FLAGS.get("strategy_latent", False):
            latent_norm = torch.norm(latent_tensor, p=2).item()
            debug_log(
                "strategy_latent",
                f"Updated latent (history_len={len(self.opponent_history)}, norm={latent_norm:.4f})",
            )
    
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
            dtype=torch.float32,
            device=TORCH_DEVICE  # Ensure tensor is on MPS/CUDA/CPU
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
    
    def learn(self, env, total_timesteps, log_interval: int = 2, verbose=0, callback=None):
        """
        Standard learning interface with callback support.
        
        Args:
            env: Training environment
            total_timesteps: Total training timesteps
            log_interval: Logging interval
            verbose: Verbosity level
            callback: Optional callback for monitoring (Stable-Baselines3 callback)
        """
        self.model.set_env(env)
        self.model.verbose = verbose
        self.model.learn(
            total_timesteps=total_timesteps, 
            log_interval=log_interval,
            callback=callback
        )


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
                device=TORCH_DEVICE,  # Use MPS/CUDA/CPU device
            )
            del self.env
        else:
            self.model = self.sb3_class.load(self.file_path, device=TORCH_DEVICE)

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
                device=TORCH_DEVICE,  # Use MPS/CUDA/CPU device
            )
            del self.env
        else:
            self.model = self.sb3_class.load(self.file_path, device=TORCH_DEVICE)

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
    debug_log("config", f"Preparing agent type='{agent_type}' load_path='{load_path}'")

    if agent_type == "custom":
        extractor_cls = _resolve_callable(agent_cfg.get("extractor_class"), MLPExtractor)
        sb3_cls = _resolve_callable(agent_cfg.get("sb3_class"), PPO)
        debug_log(
            "agent_init",
            f"Instantiating CustomAgent with extractor='{extractor_cls.__name__}' "
            f"and algo='{sb3_cls.__name__}'",
        )
        return CustomAgent(sb3_class=sb3_cls, file_path=load_path, extractor=extractor_cls)
    
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
        debug_log(
            "agent_init",
            (
                "Instantiated TransformerStrategyAgent "
                f"(latent_dim={agent.latent_dim}, num_heads={agent.num_heads}, "
                f"num_layers={agent.num_layers}, sequence_length={agent.sequence_length})"
            ),
        )
        return agent

    if agent_type == "sb3":
        sb3_cls = _resolve_callable(agent_cfg.get("sb3_class"), PPO)
        debug_log(
            "agent_init",
            f"Instantiating SB3Agent using algorithm='{sb3_cls.__name__}'",
        )
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
    """
    🔧 REWARD SYSTEM - FIXED FOR LEARNING
    
    Key changes:
    1. Increased ALL reward weights by 20-50x (rewards were too small to learn from)
    2. Damage rewards are now PRIMARY signal (weight=50.0 instead of 1.0)
    3. Sparse event rewards remain strong to provide clear goals
    4. Penalty rewards increased to provide clear negative feedback
    
    Previous issue: rewards like -0.001 were invisible to the network
    Now: rewards in range of -5.0 to +100.0 provide strong learning signals
    """
    reward_functions = {
        #'target_height_reward': RewTerm(func=base_height_l2, weight=0.0, params={'target_height': -4, 'obj_name': 'player'}),
        'danger_zone_reward': RewTerm(func=danger_zone_reward, weight=15.0),  # Was 0.5 → Now 15.0 (30x increase)
        'damage_interaction_reward': RewTerm(func=damage_interaction_reward, weight=50.0),  # Was 1.0 → Now 50.0 (PRIMARY REWARD!)
        #'head_to_middle_reward': RewTerm(func=head_to_middle_reward, weight=0.01),
        #'head_to_opponent': RewTerm(func=head_to_opponent, weight=0.05),
        'penalize_attack_reward': RewTerm(func=in_state_reward, weight=-1.0, params={'desired_state': AttackState}),  # Was -0.04 → Now -1.0 (25x)
        'holding_more_than_3_keys': RewTerm(func=holding_more_than_3_keys, weight=-0.5),  # Was -0.01 → Now -0.5 (50x)
        #'taunt_reward': RewTerm(func=in_state_reward, weight=0.2, params={'desired_state': TauntState}),
    }
    signal_subscriptions = {
        'on_win_reward': ('win_signal', RewTerm(func=on_win_reward, weight=100)),  # Was 50 → Now 100 (major goal!)
        'on_knockout_reward': ('knockout_signal', RewTerm(func=on_knockout_reward, weight=20)),  # Was 8 → Now 20
        'on_combo_reward': ('hit_during_stun', RewTerm(func=on_combo_reward, weight=10)),  # Was 5 → Now 10
        'on_equip_reward': ('weapon_equip_signal', RewTerm(func=on_equip_reward, weight=15)),  # Was 10 → Now 15
        'on_drop_reward': ('weapon_drop_signal', RewTerm(func=on_drop_reward, weight=20))  # Was 15 → Now 20
    }
    return RewardManager(reward_functions, signal_subscriptions)


# --------------------------------------------------------------------------------
# ----------------------------- 6. Optimized Training Monitoring System -----------------------------
# --------------------------------------------------------------------------------
# Lightweight, hierarchical logging with minimal overhead during training.
# Tracks critical metrics at different frequencies to balance insight vs performance.

from stable_baselines3.common.callbacks import BaseCallback
from collections import deque
import time
import csv

class TransformerHealthMonitor:
    """
    Monitors transformer encoder health during training.
    Tracks latent vector statistics and attention patterns.
    """
    
    def __init__(self):
        # Circular buffer for recent latent norms (lightweight memory)
        self.latent_norms = deque(maxlen=100)
        self.attention_entropies = deque(maxlen=100)
        
    def update(self, agent: 'TransformerStrategyAgent'):
        """
        Extract health metrics from transformer agent.
        
        Args:
            agent: TransformerStrategyAgent instance
        """
        if not isinstance(agent, TransformerStrategyAgent):
            return
        
        # Get latent vector norm (is encoder producing meaningful outputs?)
        if agent.current_strategy_latent is not None:
            norm = torch.norm(agent.current_strategy_latent, p=2).item()
            self.latent_norms.append(norm)
        
        # Get attention entropy (is transformer focusing or diffuse?)
        if len(agent.opponent_history) >= 10:
            try:
                history_array = np.array(agent.opponent_history)
                history_tensor = torch.tensor(
                    history_array, 
                    dtype=torch.float32, 
                    device=TORCH_DEVICE
                ).unsqueeze(0)
                
                with torch.no_grad():
                    _, attention_info = agent.strategy_encoder(
                        history_tensor, 
                        return_attention=True
                    )
                    
                    # Calculate entropy of pooling attention
                    attn_weights = attention_info['pooling_attention'].squeeze().cpu().numpy()
                    # Entropy: -sum(p * log(p))
                    attn_weights = attn_weights + 1e-10  # Avoid log(0)
                    entropy = -np.sum(attn_weights * np.log(attn_weights))
                    self.attention_entropies.append(entropy)
            except:
                pass  # Silent fail - don't disrupt training
    
    def get_stats(self) -> Dict[str, float]:
        """Get current health statistics."""
        stats = {}
        
        if len(self.latent_norms) > 0:
            stats['latent_norm_mean'] = np.mean(self.latent_norms)
            stats['latent_norm_std'] = np.std(self.latent_norms)
        
        if len(self.attention_entropies) > 0:
            stats['attention_entropy_mean'] = np.mean(self.attention_entropies)
        
        return stats


class RewardBreakdownTracker:
    """
    Tracks reward term contributions with minimal overhead.
    Accumulates data in memory, writes to CSV periodically.
    """
    
    def __init__(self, reward_manager: RewardManager, log_dir: str):
        self.reward_manager = reward_manager
        self.log_dir = log_dir
        self.csv_path = os.path.join(log_dir, "reward_breakdown.csv")
        
        # In-memory accumulation (write every N steps)
        self.accumulated_data = []
        self.term_activation_counts = {}
        self.total_steps = 0
        
        # Initialize CSV
        self._init_csv()
    
    def _init_csv(self):
        """Initialize CSV with headers."""
        os.makedirs(self.log_dir, exist_ok=True)
        
        headers = ["step"]
        if self.reward_manager.reward_functions:
            headers.extend(self.reward_manager.reward_functions.keys())
        headers.append("total_reward")
        
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    def compute_breakdown(self, env) -> Dict[str, float]:
        """
        Compute reward breakdown for current step.
        Returns dict of term contributions.
        """
        breakdown = {}
        total = 0.0
        
        if self.reward_manager.reward_functions:
            for name, term_cfg in self.reward_manager.reward_functions.items():
                if term_cfg.weight == 0.0:
                    value = 0.0
                else:
                    value = _to_float(term_cfg.func(env, **term_cfg.params) * term_cfg.weight)
                    total += value
                    
                    # Track activation
                    if abs(value) > 1e-6:
                        self.term_activation_counts[name] = self.term_activation_counts.get(name, 0) + 1
                
                breakdown[name] = value
        
        breakdown['total_reward'] = total
        return breakdown
    
    def record_step(self, step: int, breakdown: Dict[str, float]):
        """Record breakdown for a step (in memory)."""
        self.total_steps += 1
        self.accumulated_data.append((step, breakdown))
    
    def flush_to_csv(self):
        """Write accumulated data to CSV and clear buffer."""
        if not self.accumulated_data:
            return
        
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            for step, breakdown in self.accumulated_data:
                row = [step]
                if self.reward_manager.reward_functions:
                    row.extend([f"{breakdown.get(name, 0.0):.6f}" 
                               for name in self.reward_manager.reward_functions.keys()])
                row.append(f"{breakdown.get('total_reward', 0.0):.6f}")
                writer.writerow(row)
        
        self.accumulated_data.clear()
    
    def get_active_terms(self) -> List[str]:
        """Get list of reward terms that have activated."""
        return [name for name, count in self.term_activation_counts.items() if count > 0]


class PerformanceBenchmark:
    """
    Runs comprehensive performance evaluation at checkpoint saves.
    Tests against multiple opponent types and measures strategy diversity.
    """
    
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.csv_path = os.path.join(log_dir, "checkpoint_benchmarks.csv")
        
        # Track latent vectors for diversity analysis
        self.recent_latent_vectors = deque(maxlen=50)
        
        self._init_csv()
    
    def _init_csv(self):
        """Initialize benchmark CSV."""
        os.makedirs(self.log_dir, exist_ok=True)
        headers = [
            "checkpoint_step", "vs_based_winrate", "vs_constant_winrate",
            "avg_damage_ratio", "strategy_diversity_score", "eval_time_sec"
        ]
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    def run_benchmark(self, agent: Agent, checkpoint_step: int, num_matches: int = 5) -> Dict[str, float]:
        """
        Run full performance benchmark at checkpoint save.
        
        Args:
            agent: Agent to benchmark
            checkpoint_step: Current training step
            num_matches: Number of matches per opponent type
            
        Returns:
            Dictionary of benchmark results
        """
        start_time = time.time()
        
        print("\n" + "="*70)
        print(f"🎯 CHECKPOINT BENCHMARK (Step {checkpoint_step})")
        print("="*70)
        
        # Test vs BasedAgent
        print(f"Testing vs BasedAgent ({num_matches} matches)...")
        based_results = self._run_matches(agent, partial(BasedAgent), num_matches)
        based_winrate = np.mean([r['won'] for r in based_results]) * 100
        
        # Test vs ConstantAgent
        print(f"Testing vs ConstantAgent ({num_matches} matches)...")
        constant_results = self._run_matches(agent, partial(ConstantAgent), num_matches)
        constant_winrate = np.mean([r['won'] for r in constant_results]) * 100
        
        # Calculate average damage ratio
        all_results = based_results + constant_results
        avg_damage_ratio = np.mean([
            r['damage_dealt'] / max(r['damage_taken'], 1.0) 
            for r in all_results
        ])
        
        # Calculate strategy diversity score
        diversity_score = self._calculate_strategy_diversity(agent)
        
        eval_time = time.time() - start_time
        
        # Print summary
        print("-" * 70)
        print(f"  vs BasedAgent:       {based_winrate:.1f}% wins")
        print(f"  vs ConstantAgent:    {constant_winrate:.1f}% wins")
        print(f"  Avg Damage Ratio:    {avg_damage_ratio:.2f}")
        print(f"  Strategy Diversity:  {diversity_score:.3f}")
        print(f"  Benchmark Time:      {eval_time:.1f}s")
        print("="*70 + "\n")
        
        # Save to CSV
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                checkpoint_step,
                f"{based_winrate:.2f}",
                f"{constant_winrate:.2f}",
                f"{avg_damage_ratio:.2f}",
                f"{diversity_score:.4f}",
                f"{eval_time:.1f}"
            ])
        
        return {
            'based_winrate': based_winrate,
            'constant_winrate': constant_winrate,
            'avg_damage_ratio': avg_damage_ratio,
            'diversity_score': diversity_score
        }
    
    def _run_matches(self, agent: Agent, opponent_factory, num_matches: int) -> List[Dict]:
        """Run matches and return results."""
        results = []
        for _ in range(num_matches):
            match_stats = env_run_match(
                agent,
                opponent_factory,
                max_timesteps=30*60,  # 60 seconds
                resolution=CameraResolution.LOW,
                train_mode=True
            )
            
            results.append({
                'won': match_stats.player1_result == Result.WIN,
                # 🔧 FIXED: Changed total_damage → damage_taken (correct attribute)
                'damage_dealt': match_stats.player2.damage_taken,  # Damage we dealt
                'damage_taken': match_stats.player1.damage_taken   # Damage we took
            })
        
        return results
    
    def _calculate_strategy_diversity(self, agent: Agent) -> float:
        """
        Calculate strategy diversity score from recent latent vectors.
        Higher score = more diverse strategies employed.
        
        Returns standard deviation of latent vector norms.
        """
        if not isinstance(agent, TransformerStrategyAgent):
            return 0.0
        
        if len(self.recent_latent_vectors) < 10:
            return 0.0
        
        # Calculate std dev of norms (diversity in strategy strength)
        norms = [np.linalg.norm(v) for v in self.recent_latent_vectors]
        diversity = np.std(norms)
        
        return diversity
    
    def record_latent_vector(self, agent: Agent):
        """Record latent vector for diversity tracking."""
        if not isinstance(agent, TransformerStrategyAgent):
                return
            
        if agent.current_strategy_latent is not None:
            latent_numpy = agent.current_strategy_latent.cpu().detach().numpy().flatten()
            self.recent_latent_vectors.append(latent_numpy)


class TrainingMonitorCallback(BaseCallback):
    """
    Stable-Baselines3 callback for hierarchical training monitoring.
    
    Logging Hierarchy:
    - Every 100 steps: Light checks (alerts only)
    - Every 500-1000 steps: Reward breakdown, transformer health, PPO metrics
    - Every 5000 steps: Quick evaluation, behavior summary, sanity checks
    - Every checkpoint: Full benchmark
    """
    
    def __init__(
        self,
        agent: Agent,
        reward_manager: RewardManager,
        save_handler: SaveHandler,
        log_dir: str,
        light_log_freq: int = 500,
        eval_freq: int = 5000,
        eval_matches: int = 3,
        verbose: int = 1
    ):
        super().__init__(verbose)
        
        self.agent = agent
        self.reward_manager = reward_manager
        self.save_handler = save_handler
        self.log_dir = log_dir
        self.light_log_freq = light_log_freq
        self.eval_freq = eval_freq
        self.eval_matches = eval_matches
        
        # Initialize tracking components
        self.transformer_monitor = TransformerHealthMonitor()
        self.reward_tracker = RewardBreakdownTracker(reward_manager, log_dir)
        self.benchmark = PerformanceBenchmark(log_dir)
        
        # Frame-level alert buffers
        self.recent_rewards = deque(maxlen=100)
        self.recent_losses = deque(maxlen=100)
        
        # Episode tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        
        # Behavior tracking (lightweight)
        self.total_damage_dealt = 0.0
        self.total_damage_taken = 0.0
        self.danger_zone_steps = 0
        self.total_steps = 0
        
        # Sanity check state
        self.reward_stuck_warning_issued = False
        self.loss_explosion_warning_issued = False
        
        # Timing
        self.last_light_log = 0
        self.last_eval = 0
        self.last_checkpoint_step = 0
        
        print(f"✓ Training monitor initialized (log_dir: {log_dir})")
    
    def _init_callback(self) -> None:
        """Called once at start of training."""
        # Create summary CSV for episode-level metrics
        self.episode_csv_path = os.path.join(self.log_dir, "episode_summary.csv")
        with open(self.episode_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'episode', 'reward', 'length', 'damage_ratio'])
    
    def _on_step(self) -> bool:
        """
        Called at every training step.
        Performs lightweight checks and periodic logging.
        """
        self.current_episode_length += 1
        self.total_steps += 1
        
        # Extract current reward (from training env)
        if len(self.model.ep_info_buffer) > 0:
            recent_ep = self.model.ep_info_buffer[-1]
            step_reward = recent_ep.get('r', 0.0)
            self.current_episode_reward = step_reward
        
        # Frame-level alerts (console only, no file I/O)
        self._check_frame_alerts()
        
        # Light logging every N steps
        if self.num_timesteps - self.last_light_log >= self.light_log_freq:
            self._light_logging()
            self.last_light_log = self.num_timesteps
        
        # Quick evaluation every N steps
        if self.num_timesteps - self.last_eval >= self.eval_freq:
            self._quick_evaluation()
            self.last_eval = self.num_timesteps
        
        # Check if checkpoint was just saved
        if self.save_handler is not None:
            current_checkpoint_step = self.save_handler.num_timesteps
            if current_checkpoint_step > self.last_checkpoint_step + self.save_handler.save_freq - 100:
                self._checkpoint_benchmark()
                self.last_checkpoint_step = current_checkpoint_step
        
        return True  # Continue training
    
    def _check_frame_alerts(self):
        """
        Frame-level alerts (console only, no disk writes).
        Checks for critical issues that need immediate attention.
        """
        # Check for gradient explosions
        if hasattr(self.model, 'logger') and self.model.logger:
            try:
                # Try to get loss from logger
                log_data = self.model.logger.name_to_value
                if 'train/policy_gradient_loss' in log_data:
                    loss = log_data['train/policy_gradient_loss']
                    self.recent_losses.append(loss)
                    
                    if loss > 100 and not self.loss_explosion_warning_issued:
                        print(f"\n⚠️  ALERT: Gradient explosion detected (loss={loss:.2f}) at step {self.num_timesteps}\n")
                        self.loss_explosion_warning_issued = True
                
                # Check for NaN
                if 'train/loss' in log_data:
                    total_loss = log_data['train/loss']
                    if np.isnan(total_loss):
                        print(f"\n🚨 CRITICAL: NaN detected in loss at step {self.num_timesteps}!\n")
            except:
                pass  # Silent fail
        
        # Check for reward spikes
        if len(self.recent_rewards) > 10:
            recent_mean = np.mean(list(self.recent_rewards)[-10:])
            if self.current_episode_reward > 1000 * abs(recent_mean) + 1e-6:
                print(f"\n⚠️  ALERT: Reward spike detected ({self.current_episode_reward:.1f}) at step {self.num_timesteps}\n")
    
    def _light_logging(self):
        """
        Light logging every 500-1000 steps.
        - Reward breakdown
        - Transformer health
        - PPO core metrics
        """
        print(f"\n--- Step {self.num_timesteps} ---")
        
        # 1. Reward breakdown
        if hasattr(self.training_env, 'envs') and len(self.training_env.envs) > 0:
            try:
                env = self.training_env.envs[0]
                # Get to the base environment
                while hasattr(env, 'env'):
                    env = env.env
                if hasattr(env, 'raw_env'):
                    breakdown = self.reward_tracker.compute_breakdown(env.raw_env)
                    active_terms = [k for k, v in breakdown.items() if k != 'total_reward' and abs(v) > 1e-6]
                    print(f"  Reward Breakdown: {', '.join([f'{k}={v:.3f}' for k, v in breakdown.items() if k in active_terms[:3]])}")
                    print(f"  Active Terms: {', '.join(active_terms)}")
                    
                    # Record for later flush
                    self.reward_tracker.record_step(self.num_timesteps, breakdown)
            except:
                pass
        
        # 2. Transformer health
        self.transformer_monitor.update(self.agent)
        health_stats = self.transformer_monitor.get_stats()
        if health_stats:
            print(f"  Transformer: Latent Norm={health_stats.get('latent_norm_mean', 0):.3f} " +
                  f"(±{health_stats.get('latent_norm_std', 0):.3f}), " +
                  f"Attention Entropy={health_stats.get('attention_entropy_mean', 0):.3f}")
        
        # 3. PPO core metrics
        if hasattr(self.model, 'logger') and self.model.logger:
            try:
                log_data = self.model.logger.name_to_value
                policy_loss = log_data.get('train/policy_gradient_loss', 0)
                value_loss = log_data.get('train/value_loss', 0)
                explained_var = log_data.get('train/explained_variance', 0)
                
                print(f"  PPO: Policy Loss={policy_loss:.4f}, Value Loss={value_loss:.4f}, " +
                      f"Explained Var={explained_var:.3f}")
            except:
                pass
        
        # Flush reward data to CSV every 10 light logs
        if self.num_timesteps % (self.light_log_freq * 10) == 0:
            self.reward_tracker.flush_to_csv()
    
    def _quick_evaluation(self):
        """
        Quick evaluation every 5000 steps.
        - Win rate spot check
        - Behavior metrics summary
        - Sanity checks
        """
        print(f"\n{'='*70}")
        print(f"🔍 QUICK EVALUATION (Step {self.num_timesteps})")
        print(f"{'='*70}")
        
        # 1. Win rate spot check (3 quick matches)
        wins = 0
        total_damage_dealt = 0
        total_damage_taken = 0
        
        for i in range(self.eval_matches):
            try:
                match_stats = env_run_match(
                    self.agent,
                    partial(BasedAgent),
                    max_timesteps=30*60,
                    resolution=CameraResolution.LOW,
                    train_mode=True
                )
                if match_stats.player1_result == Result.WIN:
                    wins += 1
                # 🔧 FIXED: Changed total_damage → damage_done (correct PlayerStats attribute)
                total_damage_dealt += match_stats.player2.damage_taken  # Damage we dealt = opponent's damage_taken
                total_damage_taken += match_stats.player1.damage_taken  # Damage we took
            except:
                pass
        
        win_rate = (wins / self.eval_matches) * 100
        avg_damage_ratio = total_damage_dealt / max(total_damage_taken, 1.0)
        
        print(f"  Win Rate: {win_rate:.1f}% ({wins}/{self.eval_matches} matches)")
        print(f"  Damage Ratio: {avg_damage_ratio:.2f}")
        
        # 2. Behavior metrics summary (from episode buffer)
        if len(self.model.ep_info_buffer) >= 10:
            recent_rewards = [ep['r'] for ep in list(self.model.ep_info_buffer)[-10:]]
            recent_lengths = [ep['l'] for ep in list(self.model.ep_info_buffer)[-10:]]
            
            print(f"  Avg Episode Reward (last 10): {np.mean(recent_rewards):.2f}")
            print(f"  Avg Episode Length (last 10): {np.mean(recent_lengths):.1f}")
        
        # 3. Sanity checks
        self._run_sanity_checks()
        
        print(f"{'='*70}\n")
        
        # Record latent vector for diversity tracking
        self.benchmark.record_latent_vector(self.agent)
    
    def _run_sanity_checks(self):
        """Run sanity checks on training progress."""
        issues = []
        
        # Check if reward is stuck
        if len(self.model.ep_info_buffer) >= 50:
            recent_rewards = [ep['r'] for ep in list(self.model.ep_info_buffer)[-50:]]
            unique_values = len(set([round(r, 1) for r in recent_rewards]))
            if unique_values <= 3 and not self.reward_stuck_warning_issued:
                issues.append("Reward appears STUCK (very little variation)")
                self.reward_stuck_warning_issued = True
        
        # Check if agent is improving
        if len(self.model.ep_info_buffer) >= 100:
            early_rewards = [ep['r'] for ep in list(self.model.ep_info_buffer)[20:40]]
            recent_rewards = [ep['r'] for ep in list(self.model.ep_info_buffer)[-20:]]
            
            if np.mean(recent_rewards) <= np.mean(early_rewards) * 1.05:
                issues.append("NO IMPROVEMENT detected (agent not learning)")
        
        # Check for loss explosions
        if len(self.recent_losses) >= 10:
            recent_losses = list(self.recent_losses)[-10:]
            if any(l > 1000 for l in recent_losses):
                issues.append("Loss values very high (check learning rate)")
        
        # Print issues
        if issues:
            print(f"  ⚠️  Sanity Check Issues:")
            for issue in issues:
                print(f"      - {issue}")
        else:
            print(f"  ✓ Sanity checks passed")
    
    def _checkpoint_benchmark(self):
        """
        Full performance benchmark when checkpoint is saved.
        Most comprehensive evaluation.
        """
        try:
            self.benchmark.run_benchmark(
                self.agent,
                self.num_timesteps,
                num_matches=5
            )
        except Exception as e:
            print(f"⚠️  Checkpoint benchmark failed: {e}")
    
    def _on_rollout_end(self) -> None:
        """Called at end of each rollout (batch of episodes)."""
        # Flush reward breakdown to CSV
        self.reward_tracker.flush_to_csv()


# --------------------------------------------------------------------------------
# ----------------------------- 7. Self-Play Infrastructure -----------------------------
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
    """
    Configure snapshot saving and opponent sampling for self-play.

    If opponent_mix contains 'self_play' with None as handler, it will be
    automatically replaced with the created selfplay_handler.
    """

    # Create self-play handler for loading past snapshots
    selfplay_handler = selfplay_handler_cls(partial(type(agent)))

    # Create save handler for checkpointing during training
    save_handler = SaveHandler(
        agent=agent,
        save_freq=save_freq,
        max_saved=max_saved,
        save_path=save_path,
        run_name=run_name,
        mode=mode,
    )

    # Setup opponent mix with self-play handler injection
    if opponent_mix is None:
        # Default opponent mix
        opponent_mix = {
            'self_play': (8, selfplay_handler),
            'constant_agent': (0.5, partial(ConstantAgent)),
            'based_agent': (1.5, partial(BasedAgent)),
        }
    else:
        # If user provided opponent_mix with 'self_play': (weight, None),
        # inject the actual handler
        opponent_mix_copy = opponent_mix.copy()
        if 'self_play' in opponent_mix_copy:
            weight, handler = opponent_mix_copy['self_play']
            if handler is None:
                # Inject the selfplay_handler
                opponent_mix_copy['self_play'] = (weight, selfplay_handler)
                debug_log(
                    "config",
                    f"Injected selfplay_handler into opponent_mix (weight={weight})"
                )
        opponent_mix = opponent_mix_copy

    opponent_cfg = OpponentsCfg(opponents=opponent_mix)
    debug_log(
        "config",
        (
            f"Self-play setup run='{run_name}' save_freq={save_freq} "
            f"max_saved={max_saved} opponents={list(opponent_cfg.opponents.keys())}"
        ),
    )
    opponent_cfg = attach_opponent_debug(opponent_cfg)
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
    monitor_callback: Optional[TrainingMonitorCallback] = None,
):
    """
    Launch training with the provided environment, reward, and self-play setup.
    
    Args:
        agent: Agent to train
        reward_manager: Reward function manager
        save_handler: Checkpoint save handler
        opponent_cfg: Opponent configuration
        resolution: Camera resolution for rendering
        train_timesteps: Total training timesteps
        train_logging: Logging mode
        monitor_callback: Optional training monitor callback for enhanced logging
    """

    debug_log(
        "training_loop",
        (
            f"Starting training timesteps={train_timesteps}, "
            f"resolution={resolution.name}, logging={train_logging.name}"
        ),
    )

    # Create environment
    from environment.agent import SelfPlayWarehouseBrawl
    from stable_baselines3.common.monitor import Monitor
    
    env = SelfPlayWarehouseBrawl(
        reward_manager=reward_manager,
        opponent_cfg=opponent_cfg,
        save_handler=save_handler,
        resolution=resolution
    )
    reward_manager.subscribe_signals(env.raw_env)
    
    if train_logging != TrainLogging.NONE:
        # Create log dir
        log_dir = f"{save_handler._experiment_path()}/" if save_handler is not None else "/tmp/gym/"
        os.makedirs(log_dir, exist_ok=True)
        
        # Logs will be saved in log_dir/monitor.csv
        env = Monitor(env, log_dir)
    
    base_env = env.unwrapped if hasattr(env, 'unwrapped') else env
    
    try:
        agent.get_env_info(base_env)
        base_env.on_training_start()
        
        # Train with callback if provided
        if monitor_callback is not None:
            agent.learn(env, total_timesteps=train_timesteps, verbose=1, callback=monitor_callback)
        else:
            agent.learn(env, total_timesteps=train_timesteps, verbose=1)
        
        base_env.on_training_end()
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        pass
    
    env.close()
    
    if save_handler is not None:
        save_handler.save_agent()
    
    if train_logging == TrainLogging.PLOT:
        from environment.agent import plot_results
        plot_results(log_dir)
    
    debug_log("training_loop", "Training loop completed")


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
    
    # Display device information at start
    print("=" * 70)
    print(f"🚀 UTMIST AI² Training - Device: {TORCH_DEVICE}")
    
    # Check if monitoring is enabled
    training_cfg = TRAIN_CONFIG.get("training", {})
    enable_monitoring = training_cfg.get("enable_debug", True)  # Default to True (always monitor)
    if enable_monitoring:
        print(f"📝 MODE: MONITORED TRAINING (Hierarchical logging active)")
    else:
        print(f"📝 MODE: MINIMAL LOGGING")
    print("=" * 70)

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
    reward_terms = getattr(reward_manager, "reward_functions", {}) or {}
    debug_log("config", f"Reward terms active: {list(reward_terms.keys())}")
    if DEBUG_FLAGS.get("reward_terms", False):
        reward_manager = attach_reward_debug(reward_manager, steps=8)

    self_play_cfg = TRAIN_CONFIG.get("self_play", {})
    self_play_run_name = self_play_cfg.get("run_name", "experiment_9")
    self_play_save_freq = self_play_cfg.get("save_freq", 100_000)
    self_play_max_saved = self_play_cfg.get("max_saved", 40)
    self_play_mode = self_play_cfg.get("mode", SaveHandlerMode.FORCE)
    self_play_opponent_mix = self_play_cfg.get("opponent_mix")
    self_play_handler_cls = _resolve_callable(self_play_cfg.get("handler"), SelfPlayRandom)

    _self_play_handler, save_handler, opponent_cfg = build_self_play_components(
        learning_agent,
        save_path=CHECKPOINT_BASE_PATH,  # Auto-adjusts for Colab or local
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
    debug_log(
        "config",
        f"Training config: timesteps={train_timesteps}, resolution={train_resolution.name}, logging={train_logging.name}",
    )
    
    # Create monitoring callback if enabled
    monitor_callback = None
    if enable_monitoring:
        log_dir = f"{save_handler._experiment_path()}/" if save_handler is not None else f"{CHECKPOINT_BASE_PATH}/tmp/"
        light_log_freq = training_cfg.get("light_log_freq", 500)  # Every 500 steps
        eval_freq = training_cfg.get("eval_freq", 5000)  # Every 5000 steps
        eval_matches = training_cfg.get("eval_episodes", 3)  # 3 matches per eval
        
        print("\n" + "🔬 " + "="*68)
        print("OPTIMIZED MONITORING SYSTEM ACTIVE")
        print("="*70)
        print(f"  • Light logging every {light_log_freq} steps (reward breakdown, transformer health, PPO metrics)")
        print(f"  • Quick evaluation every {eval_freq} steps ({eval_matches} matches)")
        print(f"  • Full benchmarks at each checkpoint save (~{self_play_save_freq} steps)")
        print(f"  • Frame-level alerts: Console only (gradient explosions, NaN, reward spikes)")
        print(f"  • Log directory: {log_dir}")
        print("="*70 + "\n")
        
        monitor_callback = TrainingMonitorCallback(
            agent=learning_agent,
            reward_manager=reward_manager,
            save_handler=save_handler,
            log_dir=log_dir,
            light_log_freq=light_log_freq,
            eval_freq=eval_freq,
            eval_matches=eval_matches,
            verbose=1
    )

    run_training_loop(
        agent=learning_agent,
        reward_manager=reward_manager,
        save_handler=save_handler,
        opponent_cfg=opponent_cfg,
        resolution=train_resolution,
        train_timesteps=train_timesteps,
        train_logging=train_logging,
        monitor_callback=monitor_callback,
    )
    
    # Print final summary if monitoring was enabled
    if enable_monitoring and monitor_callback is not None:
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE - MONITORING SUMMARY")
        print("="*70)
        print(f"  • Total steps tracked: {monitor_callback.total_steps}")
        print(f"  • Reward breakdown CSV: {monitor_callback.reward_tracker.csv_path}")
        print(f"  • Episode summary CSV: {monitor_callback.episode_csv_path}")
        print(f"  • Checkpoint benchmarks CSV: {monitor_callback.benchmark.csv_path}")
        
        # Check for any issues
        if monitor_callback.reward_stuck_warning_issued or monitor_callback.loss_explosion_warning_issued:
            print(f"\n  ⚠️  Warnings issued during training:")
            if monitor_callback.reward_stuck_warning_issued:
                print(f"      - Reward appeared stuck at some point")
            if monitor_callback.loss_explosion_warning_issued:
                print(f"      - Gradient explosion detected")
        else:
            print(f"\n  ✓ No critical issues detected during training")
        
        print("="*70 + "\n")


if __name__ == '__main__':
    main()


"""
================================================================================
T4 GPU OPTIMIZATION & SCALING LAW TRAINING SUMMARY
================================================================================

This training script has been optimized for NVIDIA T4 GPU (16GB VRAM) with 
proper scaling laws for efficient hyperparameter tuning.

1. AUTOMATIC DEVICE DETECTION
   - Automatically detects CUDA (NVIDIA), MPS (Apple Silicon), or CPU
   - Priority: CUDA > MPS > CPU
   - Device info displayed at training start (VRAM, CUDA version)

2. T4 GPU OPTIMIZATIONS
   - Transformer encoder runs entirely on CUDA
   - RecurrentPPO (LSTM policy) accelerated on GPU
   - cuDNN autotuner enabled for optimal performance
   - Batch size: 128 (safe for 16GB VRAM, can be increased to 256 if needed)
   - All tensor operations use GPU (no CPU bottlenecks)

3. TRANSFORMER + LSTM ARCHITECTURE
   - TransformerStrategyEncoder: 6 layers, 8 heads, 256-dim latent space
   - Tracks opponent behavior sequences (90 frames = 3 seconds)
   - Discovers patterns via self-attention (like AlphaGo)
   - LSTM policy (512 hidden) conditions on transformer latent
   - Total params: ~2.5M (transformer) + ~1.5M (policy) = ~4M params

4. SELF-ADVERSARIAL TRAINING LOOP
   - Self-play: Trains vs snapshots of past versions (80%)
   - Scripted opponents: BasedAgent (15%), ConstantAgent (5%)
   - Curriculum learning via opponent mix
   - Checkpoints saved every 100k steps (10M) or 5k steps (50k test)

5. SCALING LAW CONFIGURATION
   - Test Config: 50k timesteps (1:200 ratio, ~15 min on T4)
   - Full Config: 10M timesteps (200x scaling, ~10-12 hours on T4)
   - IDENTICAL hyperparameters (learning rate, batch size, architecture)
   - If test config works well, 10M will behave identically (just longer)

6. MEMORY MANAGEMENT (T4 16GB VRAM)
   - Transformer encoder: ~500MB VRAM
   - LSTM policy: ~300MB VRAM
   - PPO rollout buffer (54k steps): ~2-3GB VRAM
   - Gradient computation: ~1-2GB VRAM
   - Total usage: ~4-6GB VRAM (safe margin for 16GB)

7. REWARD SHAPING
   - Damage interaction: +1.0 weight (damage dealt - damage taken)
   - Danger zone penalty: -0.5 weight (discourages being knocked off)
   - Attack spam penalty: -0.04 weight (encourages strategic attacks)
   - Signal rewards: Win (+50), Knockout (+8), Combo (+5), Weapon pickup (+10)

ESTIMATED TRAINING TIMES (T4 GPU):
- 50k test run: ~15 minutes (validate configuration)
- 10M full training: ~10-12 hours (complete training)
- Per 1M timesteps: ~60-75 minutes

EXPECTED PERFORMANCE IMPROVEMENTS (vs CPU):
- T4 GPU: ~8-10x speedup for transformer operations
- T4 GPU: ~5-7x speedup for LSTM operations
- Overall speedup: ~6-8x vs CPU-only training

MEMORY OPTIMIZATION TIPS:
- If OOM error: Reduce batch_size from 128 to 64
- If OOM error: Reduce n_steps from 54,000 to 27,000
- If OOM error: Reduce transformer num_layers from 6 to 4
- Monitor VRAM: nvidia-smi in terminal during training

SCALING LAW WORKFLOW:
1. Run 50k test config first (TRAIN_CONFIG_TEST) - 15 minutes
2. Check reward curves, win rate, behavior metrics
3. Tweak hyperparameters if needed (reward weights, learning rate, etc.)
4. Re-run 50k test to validate changes
5. Once satisfied, switch to TRAIN_CONFIG_10M for full training
6. 10M training will exhibit same behavior, just scaled up

TO SWITCH CONFIGURATIONS:
  Line 373: Change TRAIN_CONFIG = TRAIN_CONFIG_TEST
         to TRAIN_CONFIG = TRAIN_CONFIG_10M

TROUBLESHOOTING:
- "CUDA out of memory": Reduce batch_size or n_steps (see above)
- Training slow: Check nvidia-smi for GPU utilization (should be ~90%)
- Reward stuck: Check reward_breakdown.csv in test mode
- Agent not improving: Check behavior_metrics.csv for damage ratio trends

MONITORING TRAINING:
- Watch checkpoints/test_50k_t4/monitor.csv for episode rewards
- Check reward_breakdown.csv for per-term reward contributions
- Check behavior_metrics.csv for damage dealt/taken ratios
- Check evaluation_results.csv for win rates over time

================================================================================
"""
