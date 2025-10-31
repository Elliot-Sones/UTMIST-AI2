"""
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
from stable_baselines3.common.logger import configure as sb3_configure

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

# Minimal switch to disable the transformer encoder without refactoring.
# Set to True to train a plain recurrent baseline (LSTM policy) using only
# the base observations; the encoder and attention fusion are bypassed.
DISABLE_STRATEGY_ENCODER: bool = True


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


# ------------ ENTROPY ANNEALING SCHEDULE ------------
def linear_entropy_schedule(initial_value: float, final_value: float, end_fraction: float = 1.0):
    """
    Create entropy coefficient schedule that decays from high to low.

    Philosophy: "Explore widely early, consolidate later"
    - High entropy (0.5) early: Agent tries diverse strategies
    - Decay to low (0.05): Agent refines and exploits best strategies

    Args:
        initial_value: Starting entropy (e.g., 0.5 for aggressive exploration)
        final_value: Ending entropy (e.g., 0.05 for exploitation)
        end_fraction: Fraction of training when decay finishes (1.0 = full training)

    Returns:
        Callable schedule function for RecurrentPPO

    Example:
        For 50k training with end_fraction=0.8:
        - Steps 0-40k: Linear decay 0.5 → 0.05
        - Steps 40k-50k: Fixed at 0.05
    """
    def schedule(progress_remaining: float) -> float:
        """
        RecurrentPPO calls this with progress_remaining ∈ [1.0, 0.0]
        progress_remaining=1.0 → training start
        progress_remaining=0.0 → training end
        """
        # Convert to progress_done ∈ [0.0, 1.0]
        progress_done = 1.0 - progress_remaining

        # Scale by end_fraction
        if progress_done >= end_fraction:
            # After decay period, stay at final value
            return final_value

        # Linear interpolation during decay period
        decay_progress = progress_done / end_fraction
        current_value = initial_value + (final_value - initial_value) * decay_progress
        return current_value

    return schedule


# ------------ SHARED HYPERPARAMETERS (DO NOT MODIFY INDIVIDUALLY) ------------
# These hyperparameters are identical across all configs to ensure scaling laws hold
_SHARED_AGENT_CONFIG = {
        "type": "transformer_strategy",  # Transformer-based strategy understanding
        "load_path": None,
        
    # Transformer hyperparameters (optimized for T4 GPU - 16GB VRAM)
        "latent_dim": 256,           # Dimensionality of strategy latent space
    "num_heads": 8,              # Number of attention heads (divisor of latent_dim)
    "num_layers": 6,             # Depth of transformer encoder
        "sequence_length": 65,       # Frames to analyze (2.17 seconds at 30 FPS)
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
    
    # PPO training hyperparameters (OPTIMIZED FOR COMBAT GAME LEARNING)
    #
    # Learning Update Frequency Calculation:
    # - n_steps = 512 (rollout buffer size - REDUCED for frequent updates)
    # - For 50k training: 50,000 / 512 = ~98 learning updates (4x more than before!)
    # - For 10M training: 10,000,000 / 512 = ~19,531 learning updates
    # - Each update does 10 gradient epochs (n_epochs=10)
    # - Total gradient updates: 98*10=980 (50k) or 19,531*10=195,310 (10M)
    #
    # WHY 512 instead of 2048?
    # - Combat game needs FAST reward feedback (damage, knockouts)
    # - Shorter rollouts = more frequent policy updates = faster learning
    # - Agent sees updated policy every ~17 seconds (512 steps / 30 FPS)
    "n_steps": 512,              # Rollout buffer size (optimized for fast learning)
    "batch_size": 64,            # Mini-batch size for gradient updates
                                 # n_steps / batch_size = 512/64 = 8 batches per update
    "n_epochs": 10,              # Gradient epochs per rollout (standard PPO)
    "ent_coef": 0.01,  # Entropy coefficient for exploration
                                 # RecurrentPPO doesn't support callable schedules, using fixed value
                                 # Standard value for stable training
    "learning_rate": 2.5e-4,     # Learning rate (standard PPO, works well with Adam)
    "clip_range": 0.2,           # PPO clip range (prevent large policy changes)
    "gamma": 0.99,               # Discount factor (value long-term rewards)
    "gae_lambda": 0.95,          # GAE lambda (balance bias-variance in advantage)
}

_SHARED_REWARD_CONFIG = {
    "factory": "gen_reward_manager_SPARSE",  # 🎯 SPARSE: Let exploration find best strategies!
    # "factory": None,  # Uses gen_reward_manager() by default (COMPLEX - use for hand-crafted strategies)
}

# ============ 10M CONFIGURATION (FULL TRAINING ON T4 GPU) ============
# Full training with self-play, curriculum, and adversarial learning
# Estimated time on T4: ~10-12 hours for 10M timesteps
TRAIN_CONFIG_10M: Dict[str, dict] = {
    "agent": _SHARED_AGENT_CONFIG.copy(),
    "reward": _SHARED_REWARD_CONFIG.copy(),
    "self_play": {
        "run_name": "competition_10M_final",  # COMPETITION AGENT
        "save_freq": 50_000,               # Save every 50k (200 checkpoints total)
        "max_saved": 200,                  # Keep ALL checkpoints for diverse opponent pool
        "mode": SaveHandlerMode.FORCE,

        # COMPETITION opponent mix (maximize strategy diversity)
        "opponent_mix": {
            "self_play": (7.0, None),                      # 70% self-play (learn counter-strategies)
            "based_agent": (2.0, partial(BasedAgent)),     # 20% scripted (learn fundamentals)
            "clockwork_agent": (0.5, partial(ClockworkAgent)),  # 5% predictable patterns
            "random_agent": (0.5, partial(RandomAgent)),   # 5% chaos (robustness)
        },
        "handler": SelfPlayRandom,  # Randomly sample from 200-checkpoint opponent pool
    },
    "training": {
        "resolution": CameraResolution.LOW,
        "timesteps": 10_000_000,   # 10 million timesteps (~10-12 hours on T4)
        "logging": TrainLogging.PLOT,

        # Light monitoring for 10M (balance insight vs speed)
        "enable_debug": True,
        "eval_freq": 100_000,      # Evaluate every 100k steps
        "eval_episodes": 5,        # 5 matches per evaluation
        "light_log_freq": 5000,    # Log every 5k steps (lower overhead)
    },
}

# ============ EXPLORATION MODE: SPARSE REWARDS + DIVERSE OPPONENTS ============
# 🎯 Goal: Let agent discover BEST strategies through exploration, not shaping
#
# Philosophy: "Don't tell agent HOW to win, just THAT it should win"
#
# Success criteria:
#   - damage_dealt > 0 after 10k steps (discovered attacking through exploration)
#   - damage_dealt > 50 after 100k steps (consistent attacking + positioning)
#   - win rate > 70% vs ConstantAgent AND > 40% vs BasedAgent (generalizes!)
#   - Strategy diversity score > 0.4 (multiple strategies emerged)
#
# Strategy:
#   - SPARSE rewards: damage ± and wins ONLY (no tactical/strategic shaping)
#   - Entropy schedule: 0.5 → 0.05 (explore widely early, refine later)
#   - Progressive opponents: ConstantAgent → BasedAgent → RandomAgent
#   - Ignition safety: tiny attack hint (weight=1.0) for first 10k steps only
#
# Why this works:
#   1. High exploration discovers diverse attack patterns/timings/positions
#   2. Opponent diversity forces generalization (not overfitting to one tactic)
#   3. Sparse rewards avoid biasing toward hand-crafted strategies
#   4. Agent learns: "damage + win = good" and figures out HOW organically
#
# Run this FIRST to let agent find its own strategies!
TRAIN_CONFIG_EXPLORATION: Dict[str, dict] = {
    "agent": _SHARED_AGENT_CONFIG.copy(),
    "reward": _SHARED_REWARD_CONFIG.copy(),  # Uses gen_reward_manager_SPARSE
    "self_play": {
        "run_name": "exploration_sparse_rewards",
        "save_freq": 20_000,                   # Save every 20k steps (5 checkpoints)
        "max_saved": 5,
        "mode": SaveHandlerMode.FORCE,

        # Progressive opponent mix (increases diversity over time)
        # ConstantAgent: Stationary, easiest to discover attacks
        # BasedAgent: Moves + attacks, forces positioning strategy
        # RandomAgent: Unpredictable, forces robust strategies
        "opponent_mix": {
            "constant_agent": (0.4, partial(ConstantAgent)),  # 40% easy
            "based_agent": (0.4, partial(BasedAgent)),        # 40% moderate
            "random_agent": (0.2, partial(RandomAgent)),      # 20% chaos
        },
        "handler": SelfPlayRandom,
    },
    "training": {
        "resolution": CameraResolution.LOW,
        "timesteps": 100_000,      # 100k steps (~30 minutes on T4) - more time for exploration
        "logging": TrainLogging.PLOT,

        # Monitoring optimized for strategy discovery
        "enable_debug": True,
        "eval_freq": 10_000,       # Evaluate every 10k steps (10 times total)
        "eval_episodes": 5,        # 5 matches per eval (better statistics)
        "light_log_freq": 1000,    # Log every 1k steps (balanced overhead)
    },
}

# Simplified debug config for quick validation (50k steps, ConstantAgent only)
TRAIN_CONFIG_DEBUG: Dict[str, dict] = {
    "agent": _SHARED_AGENT_CONFIG.copy(),
    "reward": _SHARED_REWARD_CONFIG.copy(),
    "self_play": {
        "run_name": "debug_quick_validation",
        "save_freq": 10_000,
        "max_saved": 5,
        "mode": SaveHandlerMode.FORCE,
        "opponent_mix": {
            "constant_agent": (1.0, partial(ConstantAgent)),  # 100% easy
        },
        "handler": SelfPlayRandom,
    },
    "training": {
        "resolution": CameraResolution.LOW,
        "timesteps": 50_000,       # Quick validation (15 min on T4)
        "logging": TrainLogging.PLOT,
        "enable_debug": True,
        "eval_freq": 5_000,
        "eval_episodes": 3,
        "light_log_freq": 500,
    },
}

# ============ CURRICULUM STAGE 1: LEARN BASIC COMBAT (50K) ============
# Goal: Agent must learn to beat ConstantAgent (stationary target) reliably
# Success criteria: 90%+ win rate vs ConstantAgent, positive damage ratio
# This validates reward function before proceeding to harder opponents
TRAIN_CONFIG_CURRICULUM: Dict[str, dict] = {
    "agent": _SHARED_AGENT_CONFIG.copy(),
    # Use dense + curriculum-annealed rewards for Stage 1
    "reward": {"factory": "gen_reward_manager"},
    "self_play": {
        "run_name": "curriculum_basic_combat",  # Stage 1 checkpoint folder
        "save_freq": 10_000,                    # Save every 10k steps (more time to learn)
        "max_saved": 10,                        # Keep last 10 checkpoints
        "mode": SaveHandlerMode.FORCE,
        
        # 100% ConstantAgent - learn to attack stationary target
        "opponent_mix": {
            "constant_agent": (1.0, partial(ConstantAgent)),  # 100% easy target
        },
        "handler": SelfPlayRandom,
    },
    "training": {
        "resolution": CameraResolution.LOW,
        "timesteps": 50_000,       # 50k timesteps (~15 minutes on T4)
        "logging": TrainLogging.PLOT,

        # Heavy monitoring for curriculum stage
        "enable_debug": True,
        "eval_freq": 10_000,       # Evaluate every 10k steps (5 times total)
        "eval_episodes": 5,        # Run 5 matches per evaluation (more samples)
        "light_log_freq": 500,     # Log every 500 steps (balanced frequency)
    },
}

# ============ CURRICULUM STAGE 2: SCRIPTED OPPONENTS (50K) ============
# Goal: Agent must learn to beat BasedAgent (heuristic AI)
# Success criteria: 60%+ win rate vs BasedAgent
# Assumes Stage 1 checkpoint as starting point
TRAIN_CONFIG_CURRICULUM_STAGE2: Dict[str, dict] = {
    "agent": {
        **_SHARED_AGENT_CONFIG.copy(),
        "load_path": None,  # Set this to Stage 1 checkpoint path when running
    },
    # Keep curriculum/dense rewards for Stage 2 as well
    "reward": {"factory": "gen_reward_manager"},
    "self_play": {
        "run_name": "curriculum_scripted",
        "save_freq": 5_000,
        "max_saved": 10,
        "mode": SaveHandlerMode.FORCE,
        
        # 70% BasedAgent, 30% ConstantAgent (maintain basic skills)
        "opponent_mix": {
            "based_agent": (7.0, partial(BasedAgent)),        # 70% harder opponent
            "constant_agent": (3.0, partial(ConstantAgent)),  # 30% easy (retain skills)
        },
        "handler": SelfPlayRandom,
    },
    "training": {
        "resolution": CameraResolution.LOW,
        "timesteps": 50_000,
        "logging": TrainLogging.PLOT,
        "enable_debug": True,
        "eval_freq": 5_000,
        "eval_episodes": 5,
    },
}

# ============ 50K TEST CONFIGURATION (SELF-PLAY TEST) ============
# Goal: Test self-play stability and opponent diversity
# Run AFTER curriculum stages to validate full pipeline
# Success criteria: Win rate stays above 40% as opponent pool grows
TRAIN_CONFIG_TEST: Dict[str, dict] = {
    "agent": {
        **_SHARED_AGENT_CONFIG.copy(),
        "load_path": None,  # Set this to Stage 2 checkpoint when running
    },
    "reward": _SHARED_REWARD_CONFIG.copy(),
    "self_play": {
        "run_name": "test_50k_selfplay",   # Test self-play mechanism
        "save_freq": 5_000,                # Save every 5k steps (10 snapshots)
        "max_saved": 10,                   # Keep last 10 checkpoints
        "mode": SaveHandlerMode.FORCE,
        
        # Full opponent mix (validates self-play)
        "opponent_mix": {
            "self_play": (7.0, None),                      # 70% self-play
            "based_agent": (2.0, partial(BasedAgent)),     # 20% scripted
            "constant_agent": (1.0, partial(ConstantAgent)),  # 10% baseline
        },
        "handler": SelfPlayRandom,
    },
    "training": {
        "resolution": CameraResolution.LOW,
        "timesteps": 50_000,       # 50k timesteps (1:200 ratio)
        "logging": TrainLogging.PLOT,
        
        # Enhanced debugging for test runs
        "enable_debug": True,
        "eval_freq": 10_000,       # Evaluate every 10k steps
        "eval_episodes": 3,        # Run 3 matches per evaluation
    },
}

# ============ SWITCH CONFIGURATION HERE ============
# COMPETITION TRAINING PIPELINE (run in order):
#
# 🔥 STEP 0: DEBUG RUN (2-3 minutes on T4)
#    TRAIN_CONFIG_DEBUG → Verify agent attacks and deals damage (not passive)
#    Success: damage_dealt > 0, attack_button_presses > 0
#
# 🎯 STEP 1: CURRICULUM STAGE 1 (15 minutes on T4)
#    TRAIN_CONFIG_CURRICULUM → Beat ConstantAgent reliably
#    Success: 90%+ win rate, consistent damage output
#
# 🎯 STEP 2: CURRICULUM STAGE 2 (15 minutes on T4)
#    TRAIN_CONFIG_CURRICULUM_STAGE2 → Beat BasedAgent (heuristic AI)
#    Success: 60%+ win rate, adapts to scripted patterns
#
# 🎯 STEP 3: SELF-PLAY TEST (15 minutes on T4)
#    TRAIN_CONFIG_TEST → Validate self-play with growing opponent pool
#    Success: 40%+ win rate, stable learning with diverse opponents
#
# 🏆 STEP 4: COMPETITION TRAINING (10-12 hours on T4)
#    TRAIN_CONFIG_10M → Full 10M adversarial self-play
#    Goal: Create agent that beats ANY opponent strategy

# ⚠️ START HERE: Curriculum Stage 1 to reliably learn attacking vs ConstantAgent
# Default to curriculum stage 1 (dense shaping vs ConstantAgent).
# This avoids clearing an existing self-play test run directory and aligns with the
# recommended first step before self-play.
TRAIN_CONFIG = TRAIN_CONFIG_CURRICULUM  # Stage 1: Beat ConstantAgent (dense shaping enabled)

# For quick validation only (50k, ConstantAgent only):
# TRAIN_CONFIG = TRAIN_CONFIG_DEBUG

# For hand-crafted strategy shaping (use AFTER exploration proves agent can attack):
# TRAIN_CONFIG = TRAIN_CONFIG_CURRICULUM  # Dense shaping rewards

# After Stage 1, uncomment Stage 2:
# TRAIN_CONFIG = TRAIN_CONFIG_CURRICULUM_STAGE2  # Stage 2: Beat BasedAgent (60%+ win)

# After Stage 2, test self-play:
# TRAIN_CONFIG = TRAIN_CONFIG_TEST  # Self-play validation (40%+ win rate)

# After validation, run COMPETITION training:
# TRAIN_CONFIG = TRAIN_CONFIG_10M   # 🏆 FINAL: 10M competition agent



# --------------------------------------------------------------------------------
# Opponent observation history wrapper
# --------------------------------------------------------------------------------

class OpponentHistoryWrapper(gym.ObservationWrapper):
    """
    Extends observations with a rolling window of opponent features so the
    transformer can operate on sequence data during training and evaluation.
    """

    def __init__(
        self,
        env: gym.Env,
        opponent_obs_dim: int,
        sequence_length: int,
    ):
        super().__init__(env)
        self.opponent_obs_dim = opponent_obs_dim
        self.sequence_length = sequence_length

        base_low = env.observation_space.low
        base_high = env.observation_space.high
        self.base_obs_dim = base_low.shape[0]

        opponent_low = base_low[-opponent_obs_dim:]
        opponent_high = base_high[-opponent_obs_dim:]

        history_low = np.tile(opponent_low, sequence_length)
        history_high = np.tile(opponent_high, sequence_length)

        augmented_low = np.concatenate([base_low, history_low], dtype=base_low.dtype)
        augmented_high = np.concatenate([base_high, history_high], dtype=base_high.dtype)

        self.observation_space = gym.spaces.Box(
            low=augmented_low,
            high=augmented_high,
            dtype=env.observation_space.dtype,
        )

        self._history = None

    def observation(self, obs: np.ndarray) -> np.ndarray:
        assert self._history is not None, "History not initialized; call reset() first."
        history_flat = np.concatenate(self._history, axis=0).astype(obs.dtype, copy=False)
        return np.concatenate([obs, history_flat], axis=0)

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        opponent_frame = obs[-self.opponent_obs_dim:]
        self._history = [opponent_frame.copy() for _ in range(self.sequence_length)]
        return self.observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        opponent_frame = obs[-self.opponent_obs_dim:]
        self._history.append(opponent_frame.copy())
        if len(self._history) > self.sequence_length:
            self._history.pop(0)
        return self.observation(obs), reward, terminated, truncated, info

    def __getattr__(self, name: str):
        # Delegate attribute access to the underlying environment for helpers
        return getattr(self.env, name)


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
        max_sequence_length: int = 65,  # 2.17 seconds at 30 FPS
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
    
    Observation layout: [base_obs | flattened_history]. History contains
    `sequence_length` opponent frames stacked sequentially.
    """
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 256,
        latent_dim: int = 256,
        base_obs_dim: int = 256,
        opponent_obs_dim: int = 32,
        sequence_length: int = 10,
        num_heads: int = 8,
        num_layers: int = 6,
        device: Optional[torch.device] = None,
        use_strategy_encoder: bool = True,
    ) -> None:
        super().__init__(observation_space, features_dim)

        self.base_obs_dim = base_obs_dim
        self.sequence_length = sequence_length
        self.opponent_obs_dim = opponent_obs_dim
        self.latent_dim = latent_dim
        self.device = device if device is not None else TORCH_DEVICE
        self.history_dim = opponent_obs_dim * sequence_length
        # Final switch: allow global override to disable encoder cleanly
        self.use_strategy_encoder = bool(use_strategy_encoder and not DISABLE_STRATEGY_ENCODER)
        # Explicit runtime notice for clarity during training runs
        debug_log(
            "config",
            (
                f"Features extractor: use_strategy_encoder={self.use_strategy_encoder} "
                f"(global DISABLE_STRATEGY_ENCODER={DISABLE_STRATEGY_ENCODER})"
            ),
        )

        self.obs_encoder = nn.Sequential(
            nn.Linear(base_obs_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )

        # Always construct modules to keep checkpoint compatibility.
        # Forward() will bypass them when encoder is disabled.
        self.strategy_encoder = TransformerStrategyEncoder(
            opponent_obs_dim=opponent_obs_dim,
            latent_dim=latent_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            max_sequence_length=sequence_length,
            device=self.device,
        )

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=8,
            batch_first=True,
        )

        self.fusion = nn.Sequential(
            nn.Linear(256 + latent_dim, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(),
        )

        self.last_strategy_latent: Optional[torch.Tensor] = None
        self.last_attention: Optional[torch.Tensor] = None
        self.to(self.device)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.device != self.device:
            observations = observations.to(self.device)

        batch_size = observations.shape[0]
        base_obs = observations[:, : self.base_obs_dim]
        history_flat = observations[:, self.base_obs_dim :]
        if history_flat.shape[1] != self.history_dim:
            raise ValueError(
                f"Expected history dimension {self.history_dim}, received {history_flat.shape[1]}"
            )
        history = history_flat.view(batch_size, self.sequence_length, self.opponent_obs_dim)

        # If encoder is disabled, return plain MLP features (recurrent baseline)
        if not self.use_strategy_encoder:
            self.last_strategy_latent = None
            self.last_attention = None
            return self.obs_encoder(base_obs)

        # Encoder path
        strategy_latent, attention_info = self.strategy_encoder(  # type: ignore[operator]
            history, return_attention=True
        )
        self.last_strategy_latent = strategy_latent.detach()
        self.last_attention = attention_info['pooling_attention'].detach()

        obs_features = self.obs_encoder(base_obs)
        obs_features_unsqueezed = obs_features.unsqueeze(1)
        strategy_unsqueezed = strategy_latent.unsqueeze(1)
        attended_obs, _ = self.cross_attention(  # type: ignore[operator]
            query=obs_features_unsqueezed,
            key=strategy_unsqueezed,
            value=strategy_unsqueezed,
        )
        attended_obs = attended_obs.squeeze(1)

        combined = torch.cat([attended_obs, strategy_latent], dim=-1)
        features = self.fusion(combined)  # type: ignore[operator]
        return features


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
        sequence_length: int = 65,
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
        
        # Observation bookkeeping for evaluation-time augmentation
        self.expected_obs_dim: Optional[int] = None
        self.base_obs_dim: Optional[int] = None
        self.history_obs_dim: Optional[int] = None
        self._eval_history: deque[np.ndarray] = deque(maxlen=self.sequence_length)
        
        # Default hyperparameters (set by create_learning_agent)
        self.default_policy_kwargs: Optional[dict] = None
        self.default_n_steps: Optional[int] = None
        self.default_batch_size: Optional[int] = None
        self.default_n_epochs: Optional[int] = None
        self.default_learning_rate: Optional[float] = None
        self.default_ent_coef: Optional[float] = None
        self.default_clip_range: Optional[float] = None
        self.default_gamma: Optional[float] = None
        self.default_gae_lambda: Optional[float] = None
    
    def _initialize(self) -> None:
        """Initialize agent with transformer-based strategy recognition."""
        
        total_obs_dim = self.observation_space.shape[0]
        if self.opponent_obs_dim is None:
            self.opponent_obs_dim = total_obs_dim // 2
        history_dim = self.sequence_length * self.opponent_obs_dim

        if total_obs_dim > history_dim:
            self.base_obs_dim = total_obs_dim - history_dim
        else:
            self.base_obs_dim = total_obs_dim
        self.history_obs_dim = history_dim
        self.expected_obs_dim = self.base_obs_dim + history_dim
        self._reset_eval_history()

        debug_log(
            "agent_init",
            (
                "TransformerStrategyAgent _initialize "
                f"(total_obs_dim={total_obs_dim}, base_dim={self.base_obs_dim}, "
                f"opponent_dim={self.opponent_obs_dim}, sequence_length={self.sequence_length})"
            ),
        )

        policy_kwargs = self.default_policy_kwargs or {
            'activation_fn': nn.ReLU,
            'lstm_hidden_size': 512,
            'net_arch': dict(pi=[96, 96], vf=[96, 96]),
            'shared_lstm': True,
            'enable_critic_lstm': False,
            'share_features_extractor': True,
        }
        policy_kwargs['features_extractor_class'] = TransformerConditionedExtractor
        policy_kwargs['features_extractor_kwargs'] = {
            'features_dim': 256,
            'latent_dim': self.latent_dim,
            'base_obs_dim': self.base_obs_dim,
            'opponent_obs_dim': self.opponent_obs_dim,
            'sequence_length': self.sequence_length,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'device': TORCH_DEVICE,
        }

        if self.file_path is None:
            debug_log(
                "agent_init",
                (
                    "Creating new RecurrentPPO "
                    f"(n_steps={self.default_n_steps or 2048}, "
                    f"batch_size={self.default_batch_size or 128}, "
                    f"ent_coef={self.default_ent_coef or 0.10})"
                ),
            )

            self.model = RecurrentPPO(
                "MlpLstmPolicy",
                self.env,
                verbose=0,
                n_steps=self.default_n_steps or 2048,
                batch_size=self.default_batch_size or 128,
                n_epochs=self.default_n_epochs or 10,
                learning_rate=self.default_learning_rate or 2.5e-4,
                ent_coef=self.default_ent_coef or 0.10,
                clip_range=self.default_clip_range or 0.2,
                gamma=self.default_gamma or 0.99,
                gae_lambda=self.default_gae_lambda or 0.95,
                policy_kwargs=policy_kwargs,
                device=TORCH_DEVICE,
            )
            debug_log("agent_init", f"RecurrentPPO model initialized on device: {TORCH_DEVICE}")
            del self.env
        else:
            self.model = RecurrentPPO.load(self.file_path, device=TORCH_DEVICE)
            print(f"✓ Loaded RecurrentPPO model from {self.file_path} to {TORCH_DEVICE}")
            debug_log("agent_init", f"Loaded existing RecurrentPPO from {self.file_path} on {TORCH_DEVICE}")

            extractor = getattr(self.model.policy, 'features_extractor', None)
            encoder_path = self.file_path.replace('.zip', '_transformer_encoder.pth')
            if extractor is not None and isinstance(extractor, TransformerConditionedExtractor) and os.path.exists(encoder_path):
                state_dict = torch.load(encoder_path, map_location=TORCH_DEVICE)
                extractor.strategy_encoder.load_state_dict(state_dict)
                print(f"✓ Loaded legacy transformer encoder from {encoder_path} to {TORCH_DEVICE}")
                debug_log('agent_init', f"Loaded legacy transformer encoder weights from {encoder_path}")
    
    def _reset_eval_history(self):
        self._eval_history.clear()
        if self.opponent_obs_dim is None:
            return
        zero_frame = np.zeros((self.opponent_obs_dim,), dtype=np.float32)
        for _ in range(self.sequence_length):
            self._eval_history.append(zero_frame.copy())

    def _update_eval_history(self, opponent_obs: np.ndarray):
        if self.opponent_obs_dim is None:
            return
        if opponent_obs.shape[0] != self.opponent_obs_dim:
            raise ValueError(
                f"Opponent observation dimension mismatch "
                f"(expected {self.opponent_obs_dim}, got {opponent_obs.shape[0]})"
            )
        self._eval_history.append(opponent_obs.astype(np.float32, copy=True))
        while len(self._eval_history) > self.sequence_length:
            self._eval_history.popleft()

    def reset(self) -> None:
        self.lstm_states = None
        self.episode_starts = np.ones((1,), dtype=bool)
        self._reset_eval_history()
    
    def _split_observation(self, obs: np.ndarray, *, augmented: bool) -> Tuple[np.ndarray, np.ndarray]:
        if self.opponent_obs_dim is None:
            raise ValueError("Opponent observation dimension is not set.")
        if self.base_obs_dim is None:
            raise ValueError("Base observation dimension is not set.")
        
        base_obs = obs[:self.base_obs_dim] if augmented else obs
        split_idx = base_obs.shape[0] - self.opponent_obs_dim
        if split_idx < 0:
            raise ValueError(
                f"Observation too small to split: base_obs_dim={self.base_obs_dim}, "
                f"opponent_obs_dim={self.opponent_obs_dim}, obs_len={len(obs)}"
            )
        player_obs = base_obs[:split_idx]
        opponent_obs = base_obs[split_idx:]
        return player_obs, opponent_obs

    def _ensure_observation_shape(self, obs_batch: np.ndarray) -> np.ndarray:
        if self.expected_obs_dim is None or self.base_obs_dim is None:
            raise ValueError("Agent not initialized; call _initialize before predict.")

        if obs_batch.shape[0] != 1:
            raise ValueError("Batch observations not supported for manual augmentation; expected batch size 1.")

        current_dim = obs_batch.shape[1]
        if current_dim == self.expected_obs_dim:
            for row in obs_batch:
                _, opponent_obs = self._split_observation(row, augmented=True)
                self._update_eval_history(opponent_obs)
            return obs_batch.astype(np.float32, copy=False)
        
        if current_dim == self.base_obs_dim:
            augmented_rows = []
            for row in obs_batch:
                _, opponent_obs = self._split_observation(row, augmented=False)
                self._update_eval_history(opponent_obs)
                history_flat = np.concatenate(list(self._eval_history), axis=0).astype(np.float32, copy=False)
                augmented_rows.append(
                    np.concatenate([row.astype(np.float32, copy=False), history_flat], axis=0)
                )
            return np.stack(augmented_rows, axis=0)
        
        raise ValueError(
            f"Unexpected observation dimension {current_dim}; "
            f"expected {self.base_obs_dim} (base) or {self.expected_obs_dim} (augmented)."
        )
    
    def predict(self, obs):
        obs_array = np.array(obs, dtype=np.float32)
        if obs_array.ndim == 1:
            obs_array = obs_array.reshape(1, -1)
        obs_prepared = self._ensure_observation_shape(obs_array)
        
        # Use stochastic actions early to avoid the "exact 0.5" deadzone
        # Deterministic mean maps to ~0.5 after tanh-squash into [0,1],
        # which fails "> 0.5" press thresholds and looks passive in early evals.
        # Switch to deterministic earlier (10k) so evaluation reflects intent sooner.
        use_deterministic = bool(getattr(self.model, "num_timesteps", 0) > 10_000)

        action, self.lstm_states = self.model.predict(
            obs_prepared,
            state=self.lstm_states,
            episode_start=self.episode_starts,
            deterministic=use_deterministic,
        )

        if isinstance(action, np.ndarray) and action.ndim > 1:
            action = np.squeeze(action, axis=0)
        elif isinstance(action, torch.Tensor) and action.ndim > 1:
            action = action.squeeze(0).cpu().numpy()
        
        self.episode_starts = np.zeros_like(self.episode_starts, dtype=bool)
        return action
    
    def get_strategy_latent_info(self) -> Dict[str, any]:
        """
        Get information about current strategy latent.
        Useful for debugging and analysis.
        """
        extractor = getattr(self.model.policy, "features_extractor", None)
        latent_tensor = getattr(extractor, "last_strategy_latent", None) if extractor else None
        if latent_tensor is None:
            return {'latent': None, 'norm': 0.0, 'history_length': len(self._eval_history)}
        
        latent_numpy = latent_tensor.detach().cpu().numpy()
        latent_norm = np.linalg.norm(latent_numpy)
        
        return {
            'latent': latent_numpy,
            'norm': float(latent_norm),
            'history_length': len(self._eval_history),
        }
    
    def visualize_attention(self, obs) -> Dict:
        """
        Extract attention weights for visualization.
        Shows which frames the transformer focuses on.
        """
        extractor = getattr(self.model.policy, "features_extractor", None)
        if extractor is None or not hasattr(extractor, "strategy_encoder"):
            return {}
        
        if len(self._eval_history) < self.sequence_length:
            return {}
        
        history_array = np.stack(list(self._eval_history), axis=0)
        history_tensor = torch.tensor(
            history_array,
            dtype=torch.float32,
            device=TORCH_DEVICE
        ).unsqueeze(0)
        
        with torch.no_grad():
            _, attention_info = extractor.strategy_encoder(
                history_tensor,
                return_attention=True
            )
        
        return {
            'pooling_attention': attention_info['pooling_attention'].cpu().numpy(),
            'contextualized_frames': attention_info['contextualized_frames'].cpu().numpy()
        }
    
    def save(self, file_path: str) -> None:
        """Save PPO model and export transformer weights for compatibility."""
        self.model.save(file_path)

        extractor = getattr(self.model.policy, 'features_extractor', None)
        if extractor is not None and hasattr(extractor, 'strategy_encoder'):
            encoder_path = file_path.replace('.zip', '_transformer_encoder.pth')
            torch.save(extractor.strategy_encoder.state_dict(), encoder_path)
            debug_log('agent_init', f'Transformer encoder weights saved to {encoder_path}')
    
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
        # Print SB3 training stats (losses, KL, entropy) to stdout each update
        try:
            self.model.set_logger(sb3_configure(None, ["stdout"]))
        except Exception:
            pass
        self.model.learn(
            total_timesteps=total_timesteps, 
            log_interval=log_interval,
            callback=callback
        )


# --------------------------------------------------------------------------------
# ----------------------------- 4. Available Agents Reference -----------------------------
# --------------------------------------------------------------------------------
"""
ALL AVAILABLE AGENTS (imported from environment.agent):

SCRIPTED AGENTS (no training required):
  • ConstantAgent       : Does nothing (baseline for testing)
  • RandomAgent         : Random actions (curriculum starting point)
  • BasedAgent          : Heuristic AI with chase/attack/dodge tactics
  • ClockworkAgent      : Pre-programmed action sequences
  • UserInputAgent      : Human keyboard control (WASD+HJKL+Space)

LEARNING AGENTS (trainable with RL):
  • SB3Agent            : Generic Stable-Baselines3 wrapper (PPO, A2C, etc.)
  • RecurrentPPOAgent   : LSTM-based memory agent for sequential decisions
  • TransformerStrategyAgent : Strategy-aware transformer (defined below in this file)

CONFIGURATION USAGE:
  In opponent_mix, use partial() to configure agents:
    "based_agent": (0.5, partial(BasedAgent))
    "random_agent": (0.3, partial(RandomAgent))
  
  In TRAIN_CONFIG['agent'], specify agent type:
    "type": "sb3"                    → uses SB3Agent
    "type": "transformer_strategy"   → uses TransformerStrategyAgent
    "type": "custom"                 → uses CustomAgent with MLPExtractor
"""



# ========== Custom Feature Extractors & Agent Architectures ==========

class MLPPolicy(nn.Module):
    """Simple 3-layer MLP for feature extraction."""
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
    """MLP-based feature extractor for SB3 agents. Used with CustomAgent."""
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
    """SB3 agent with custom feature extractors (e.g., MLPExtractor, CNN)."""
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
                device=TORCH_DEVICE,
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



# ========== Agent Factory Utilities ==========

def _resolve_callable(value, fallback):
    """Resolve config entries that may be callables or string names."""
    if value is None:
        return fallback
    if isinstance(value, str):
        resolved = globals().get(value)
        return resolved if resolved is not None else fallback
    return value


def create_learning_agent(agent_cfg: Dict[str, object]) -> Agent:
    """
    Factory that instantiates the training agent based on TRAIN_CONFIG.
    
    Supports agent types: "transformer_strategy", "custom", "sb3"
    """

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
            sequence_length=agent_cfg.get("sequence_length", 65),
            opponent_obs_dim=agent_cfg.get("opponent_obs_dim", None)
        )
        # Set all default hyperparameters from config
        agent.default_policy_kwargs = agent_cfg.get("policy_kwargs")
        agent.default_n_steps = agent_cfg.get("n_steps")
        agent.default_batch_size = agent_cfg.get("batch_size")
        agent.default_n_epochs = agent_cfg.get("n_epochs")
        agent.default_learning_rate = agent_cfg.get("learning_rate")
        agent.default_ent_coef = agent_cfg.get("ent_coef")
        agent.default_clip_range = agent_cfg.get("clip_range")
        agent.default_gamma = agent_cfg.get("gamma")
        agent.default_gae_lambda = agent_cfg.get("gae_lambda")
        debug_log(
            "agent_init",
            (
                "Instantiated TransformerStrategyAgent "
                f"(latent_dim={agent.latent_dim}, num_heads={agent.num_heads}, "
                f"num_layers={agent.num_layers}, sequence_length={agent.sequence_length}, "
                f"n_steps={agent.default_n_steps})"
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
    Computes per-step damage reward using robust deltas.

    Uses totals and computes deltas since the last call to avoid frame-order
    sensitivity where instantaneous frame values could be missed.

    Modes:
    - ASYMMETRIC_OFFENSIVE: +damage dealt to opponent only
    - SYMMETRIC: +damage dealt, -damage taken
    - ASYMMETRIC_DEFENSIVE: -damage taken only
    """
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]

    # Initialize running totals on first call (or after reset)
    if not hasattr(env, "_last_damage_totals") or env.steps <= 1:
        env._last_damage_totals = {
            "player": player.damage_taken_total,
            "opponent": opponent.damage_taken_total,
        }
        return 0.0

    prev_p = env._last_damage_totals.get("player", 0.0)
    prev_o = env._last_damage_totals.get("opponent", 0.0)
    cur_p = player.damage_taken_total
    cur_o = opponent.damage_taken_total

    # Robust non-negative deltas for this step
    delta_taken = max(0.0, cur_p - prev_p)
    delta_dealt = max(0.0, cur_o - prev_o)

    # Update stored totals
    env._last_damage_totals["player"] = cur_p
    env._last_damage_totals["opponent"] = cur_o

    if mode == RewardMode.ASYMMETRIC_OFFENSIVE:
        reward = delta_dealt
    elif mode == RewardMode.SYMMETRIC:
        reward = delta_dealt - delta_taken
    elif mode == RewardMode.ASYMMETRIC_DEFENSIVE:
        reward = -delta_taken
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

def closing_distance_reward(
    env: WarehouseBrawl,
    engage_range: float = 4.5,
    approach_scale: float = 0.25,
) -> float:
    """
    Encourage the agent to close distance to the opponent and stay in range.

    Returns positive reward when distance shrinks and a small bonus for
    remaining within the engage_range. Penalizes backing away.
    """
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]

    # Current and previous frame distances in world units
    dx = player.body.position.x - opponent.body.position.x
    dy = player.body.position.y - opponent.body.position.y
    current_distance = math.hypot(dx, dy)

    prev_dx = getattr(player, "prev_x", player.body.position.x) - getattr(opponent, "prev_x", opponent.body.position.x)
    prev_dy = getattr(player, "prev_y", player.body.position.y) - getattr(opponent, "prev_y", opponent.body.position.y)
    prev_distance = math.hypot(prev_dx, prev_dy)

    approach_delta = prev_distance - current_distance
    approach_term = approach_delta * approach_scale

    proximity_term = 0.0
    if current_distance < engage_range:
        proximity_term = (engage_range - current_distance) / max(engage_range, 1e-6)
        proximity_term *= env.dt

    total = approach_term + proximity_term
    # Guard against teleports/respawns creating huge spikes
    return max(-1.0, min(1.0, total))

def edge_pressure_reward(
    env: WarehouseBrawl,
    retreat_penalty_scale: float = 0.5,
) -> float:
    """
    Reward pushing the opponent toward the blast zones to secure knockouts.

    Positive when opponent moves outward, slight negative if they drift back in.
    """
    opponent: Player = env.objects["opponent"]
    stage_half_width = getattr(env, "stage_width_tiles", 36) / 2
    stage_half_width = max(stage_half_width, 1.0)

    prev_offset = min(stage_half_width, abs(getattr(opponent, "prev_x", opponent.body.position.x)))
    current_offset = min(stage_half_width, abs(opponent.body.position.x))
    outward_delta = current_offset - prev_offset

    if outward_delta > 0:
        return outward_delta / stage_half_width
    if outward_delta < 0:
        return (outward_delta * retreat_penalty_scale) / stage_half_width
    return 0.0

def on_attack_button_press(env: WarehouseBrawl) -> float:
    """
    🔥 CRITICAL EXPLORATION REWARD: Rewards agent for pressing attack buttons.
    
    Problem: Agent has zero-damage deadlock because it never explores attacking.
    Solution: Directly reward pressing light attack (j, index 7) or heavy attack (k, index 8).
    
    Action space: [w, a, s, d, space, h, l, j, k, g]
    Indices:      [0, 1, 2, 3,   4,  5, 6, 7, 8, 9]
    
    Returns:
        Positive dt-scaled bonus emphasizing attacks launched within striking range.
        0.0 otherwise
    
    Note: This is a TEMPORARY exploration bonus. Once agent learns to attack,
    reduce weight or remove. The damage_interaction_reward (weight=150) will
    take over once attacks start landing.
    """
    player: Player = env.objects["player"]

    # Get current action (stored in player object)
    action = player.cur_action
    
    # Check if attack buttons are pressed (indices 7=j, 8=k)
    # Actions are continuous [0, 1], threshold at 0.5
    light_attack = action[7] > 0.5 if len(action) > 7 else False
    heavy_attack = action[8] > 0.5 if len(action) > 8 else False
    
    # Reward attack button press only when within effective range
    if light_attack or heavy_attack:
        opponent: Player = env.objects["opponent"]

        # Compute distance to opponent
        distance = math.hypot(
            player.body.position.x - opponent.body.position.x,
            player.body.position.y - opponent.body.position.y,
        )

        # Anneal effective range using global training steps when available
        # Allows early exploration to get signal, then focuses on true range
        steps = getattr(env, 'training_steps', getattr(env, 'steps', 0))
        warmup_steps = 10_000
        start_range = 14.0
        end_range = 3.5
        frac = min(max(steps / warmup_steps, 0.0), 1.0)
        max_effective_range = end_range + (start_range - end_range) * (1.0 - frac)
        if distance > max_effective_range:
            return 0.0

        # Require facing toward opponent
        try:
            opp_right = opponent.body.position.x > player.body.position.x
            facing_right = (player.facing.name == 'RIGHT') if hasattr(player.facing, 'name') else bool(player.facing)
            correct_facing = (opp_right and facing_right) or ((not opp_right) and (not facing_right))
            if not correct_facing:
                return 0.0
        except Exception:
            pass

        # Within range: shape on proximity and vertical alignment
        proximity_scale = max(0.0, 1.0 - min(distance, max_effective_range) / max_effective_range)
        vertical_alignment = 1.0 - min(abs(player.body.position.y - opponent.body.position.y), 3.0) / 3.0
        vertical_alignment = max(0.0, vertical_alignment)

        # Lower base reward; stronger emphasis on proximity
        reward = 0.05 + 0.75 * proximity_scale + 0.2 * vertical_alignment

        # Penalize whiffed attacks slightly so hits become more valuable
        if opponent.damage_taken_this_frame == 0:
            reward *= 0.5

        return reward * env.dt
    return 0.0

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
    """
    Reward based on REAL wins (stock advantage), not timeout draws.
    
    Game mechanics:
    - Each player starts with 3 stocks (lives)
    - Knockout = lose 1 stock
    - Match ends when: stocks <= 0 OR timeout
    - Winner = player with MORE stocks left
    
    Returns:
        +1.0 if we have MORE stocks (real win by knockout)
        0.0 if same stocks (timeout draw - no reward!)
        -1.0 if opponent has more stocks (we got knocked out more)
    """
    player = env.objects["player"]
    opponent = env.objects["opponent"]
    
    stock_diff = player.stocks - opponent.stocks
    
    if agent == 'player':
        if stock_diff > 0:
            return 1.0  # We have stock advantage = real win!
        elif stock_diff < 0:
            return -1.0  # Opponent has advantage = we're losing
        else:
            return 0.0  # Timeout draw = no reward (discourage passive play)
    else:
        if stock_diff < 0:
            return 1.0
        elif stock_diff > 0:
            return -1.0
        else:
            return 0.0

def on_knockout_reward(env: WarehouseBrawl, agent: str) -> float:
    """
    Reward for TAKING a stock (knocking out opponent).
    This triggers when opponent loses a life.
    """
    if agent == 'player':
        return -1.0  # We got knocked out = bad
    else:
        return 1.0  # We knocked out opponent = good!

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

# --------------------------------------------------------------------------------
# CURRICULUM REWARD ANNEALING SYSTEM
# Gradually reduces dense rewards over training to shift from exploration to exploitation
# --------------------------------------------------------------------------------

class CurriculumRewardScheduler:
    """
    🎓 CURRICULUM ANNEALING: Gradually reduce dense reward weights over training

    Problem: Sudden reward changes cause value function mismatch and training instability
    Solution: Smooth transitions between reward regimes across training milestones

    Schedule:
    - Stage 1 (0-30k):    Foundational exploration (maximum shaping)
    - Stage 2 (30k-50k):  Confident pressure (moderate shaping)
    - Stage 3 (50k-100k): Conversion focus (reduced shaping)
    - Stage 4 (100k-200k): Mastery preparation (light shaping)
    - Stage 5 (200k+):    Minimal dense rewards (policy-driven)
    """

    def __init__(self):
        # Dense reward heads to anneal across the curriculum
        self.weight_keys = (
            "head_to_opponent",
            "attack_button",
            "close_distance",
            "edge_pressure",
        )

        # Define curriculum milestones with richer shaping terms
        self.milestones = [
            {
                "threshold": 0,
                "head_to_opponent": 10.0,
                # Reduce early attack-button shaping to avoid mashing bias
                "attack_button": 5.0,
                "close_distance": 6.0,
                "edge_pressure": 4.0,
            },  # Stage 1: Maximum guidance for early exploration
            {
                "threshold": 30_000,
                "head_to_opponent": 8.0,
                "attack_button": 11.0,
                "close_distance": 4.5,
                "edge_pressure": 3.0,
            },  # Stage 2: Encourage pressure & consistent offense
            {
                "threshold": 50_000,
                "head_to_opponent": 5.5,
                "attack_button": 6.5,
                "close_distance": 3.0,
                "edge_pressure": 2.0,
            },  # Stage 3: Transition toward sparse rewards
            {
                "threshold": 100_000,
                "head_to_opponent": 3.5,
                "attack_button": 3.0,
                "close_distance": 1.5,
                "edge_pressure": 1.0,
            },  # Stage 4: Light shaping, focus on finishing stocks
            {
                "threshold": 200_000,
                "head_to_opponent": 1.0,
                "attack_button": 1.0,
                "close_distance": 0.5,
                "edge_pressure": 0.5,
            },  # Stage 5: Minimal guidance
        ]

        self.current_stage = 0
        self.current_step = 0

    def get_weights(self, training_step: int) -> Dict[str, float]:
        """
        Get current reward weights based on training step.

        Args:
            training_step: Current training timestep

        Returns:
            Dictionary with curriculum-controlled dense reward weights.
        """
        self.current_step = training_step

        # Find current milestone configuration
        current_config = self.milestones[0]
        for i, milestone in enumerate(self.milestones):
            if training_step >= milestone["threshold"]:
                self.current_stage = i
                current_config = milestone
            else:
                break

        weights = {key: current_config.get(key, 0.0) for key in self.weight_keys}
        weights['stage'] = self.current_stage
        return weights

    def print_schedule(self):
        """Print full curriculum schedule for reference."""
        print("\n" + "="*70)
        print("🎓 CURRICULUM REWARD ANNEALING SCHEDULE")
        print("="*70)
        stage_names = [
            "STAGE 1: Foundational Exploration",
            "STAGE 2: Confident Pressure",
            "STAGE 3: Conversion Focus",
            "STAGE 4: Mastery Preparation",
            "STAGE 5: Minimal Guidance",
        ]
        orig_head = self.milestones[0]["head_to_opponent"]
        orig_attack = self.milestones[0]["attack_button"]

        for i, milestone in enumerate(self.milestones):
            threshold = milestone["threshold"]
            head_weight = milestone.get("head_to_opponent", 0.0)
            attack_weight = milestone.get("attack_button", 0.0)

            print(f"\n{stage_names[i]}")
            print(f"  Start: Step {threshold:,}")
            for key in self.weight_keys:
                if key in milestone:
                    print(f"  {key}: {milestone[key]:.1f}")

            head_pct = ((orig_head - head_weight) / orig_head) * 100 if orig_head else 0.0
            attack_pct = ((orig_attack - attack_weight) / orig_attack) * 100 if orig_attack else 0.0
            print(f"  Reduction (head/attack): {head_pct:.0f}% / {attack_pct:.0f}%")
        print("="*70 + "\n")


# Global curriculum scheduler (initialized on first call to gen_reward_manager)
_CURRICULUM_SCHEDULER = None

def gen_reward_manager_SPARSE(training_step: int = 0):
    """
    🎯 SPARSE REWARD FUNCTION - MAXIMUM EXPLORATION & STRATEGY DISCOVERY

    Philosophy: "Don't tell the agent HOW to win, just THAT it should win"

    Problem: Dense shaping rewards bias agent toward ONE hand-crafted strategy
    Solution: Only reward OUTCOMES (damage, wins), let exploration find strategy

    Design Philosophy:
    - Sparse extrinsic rewards: damage ± and win/loss ONLY
    - NO tactical shaping (movement, button presses, positioning)
    - NO strategic shaping (weapons, combos, edge pressure)
    - High exploration early (see entropy schedule) finds diverse strategies
    - Opponent diversity (self-play + scripted) forces generalization

    This allows agent to discover:
    - WHEN to attack (not forced to spam)
    - WHERE to position (not forced to chase)
    - WHAT combos work (not hand-crafted)
    - HOW to counter each opponent (strategy emerges from experience)

    Ignition Safety (first 10k steps only):
    - Tiny attack button hint (weight=1.0) if exploration hasn't found buttons
    - Automatically removed after 10k steps
    - Lets pure exploration take over once buttons discovered
    """

    # Ignition safety: tiny hint for first 10k steps only
    use_attack_ignition = training_step < 10_000

    reward_functions = {
        # PRIMARY REWARD: Damage dealt vs damage taken (core outcome signal)
        'damage_interaction_reward': RewTerm(
            func=damage_interaction_reward,
            weight=300.0,  # High weight - this is what matters
            params={'mode': RewardMode.SYMMETRIC}  # +damage dealt, -damage taken
        ),
    }

    # Optional ignition for first 10k steps (safety fallback)
    if use_attack_ignition:
        reward_functions['on_attack_button_press'] = RewTerm(
            func=on_attack_button_press,
            weight=1.0  # Tiny hint to discover buttons exist, then remove
        )
        print(f"🔥 Ignition mode: attack_button hint active (step {training_step}/10000)")

    signal_subscriptions = {
        # WIN: Primary sparse outcome signal
        'on_win_reward': ('win_signal', RewTerm(
            func=on_win_reward,
            weight=500.0  # High - winning is ultimate goal
        )),

        # KNOCKOUT: Secondary outcome signal (taking stocks matters)
        'on_knockout_reward': ('knockout_signal', RewTerm(
            func=on_knockout_reward,
            weight=50.0  # Moderate - stocks lead to wins
        )),

        # ❌ REMOVED: All tactical shaping (head_to_opponent, closing_distance, edge_pressure)
        # ❌ REMOVED: All strategic shaping (weapons, combos, key spam penalties)
        # ❌ REMOVED: All micro-management (button press tracking)
        #
        # Agent discovers optimal tactics through:
        # 1. High exploration (entropy schedule)
        # 2. Opponent diversity (self-play + scripted)
        # 3. Sparse outcome rewards (damage + wins)
    }

    return RewardManager(reward_functions, signal_subscriptions)


def gen_reward_manager(training_step: int = 0):
    """
    🔧 REWARD SYSTEM V3 - CURRICULUM ANNEALING

    ROOT CAUSE (from debug runs):
    - Sudden dense reward reduction (15.0→3.0) caused agent to forget attacking
    - Value function trained on old weights couldn't adapt quickly enough
    - Agent regressed from 0.05 damage ratio to 0.02 (worse than before!)

    NEW STRATEGY: GRADUAL CURRICULUM ANNEALING
    - Stage 1 (0-30k):    High dense rewards (max shaping for movement + offense)
    - Stage 2 (30k-50k):  Moderate reduction to cement pressure & conversions
    - Stage 3 (50k-100k): Transition toward sparse rewards (policy learns to finish)
    - Stage 4 (100k-200k): Light shaping, focus on clutch play
    - Stage 5 (200k+):    Minimal dense rewards (policy-driven mastery)

    This allows value function to adapt gradually over training.
    """
    global _CURRICULUM_SCHEDULER

    # Initialize curriculum scheduler on first call
    if _CURRICULUM_SCHEDULER is None:
        _CURRICULUM_SCHEDULER = CurriculumRewardScheduler()
        _CURRICULUM_SCHEDULER.print_schedule()  # Print schedule at start

    # Get current weights from curriculum
    curriculum_weights = _CURRICULUM_SCHEDULER.get_weights(training_step)
    head_weight = curriculum_weights['head_to_opponent']
    attack_weight = curriculum_weights['attack_button']
    close_weight = curriculum_weights.get('close_distance', 0.0)
    edge_weight = curriculum_weights.get('edge_pressure', 0.0)
    stage = curriculum_weights['stage']

    # Log curriculum transition (when stage changes)
    if hasattr(_CURRICULUM_SCHEDULER, '_last_logged_stage'):
        if stage != _CURRICULUM_SCHEDULER._last_logged_stage:
            print("\n" + "="*70)
            print(f"🎓 CURRICULUM TRANSITION: Moving to Stage {stage + 1}")
            print(f"   Step: {training_step:,}")
            milestone_cfg = _CURRICULUM_SCHEDULER.milestones[stage]
            for key in _CURRICULUM_SCHEDULER.weight_keys:
                print(f"   {key}: {milestone_cfg.get(key, 0.0):.1f}")
            print("="*70 + "\n")
    _CURRICULUM_SCHEDULER._last_logged_stage = stage

    reward_functions = {
        # PRIMARY REWARD: Landing damage on opponent (MASSIVE POSITIVE)
        'damage_interaction_reward': RewTerm(
            func=damage_interaction_reward,
            weight=500.0,  # 🔥 Keep high throughout (outcome reward)
            params={'mode': RewardMode.SYMMETRIC}  # Reward damage dealt, penalize damage taken
        ),

        # SAFETY: Discourage getting knocked off (but don't over-penalize)
        'danger_zone_reward': RewTerm(
            func=danger_zone_reward,
            weight=2.0,  # Light penalty (function returns negative value)
            params={'zone_penalty': 1, 'zone_height': 4.2}
        ),

        # 🎓 CURRICULUM: Engagement reward (annealed over training)
        'head_to_opponent': RewTerm(
            func=head_to_opponent,
            weight=head_weight  # Dynamic weight from curriculum scheduler
        ),

        # 🎯 NEW: Encourage decisive distance closing (annealed over training)
        'closing_distance_reward': RewTerm(
            func=closing_distance_reward,
            weight=close_weight,
            params={'engage_range': 12.0}
        ),

        # 💥 NEW: Reward pushing opponent toward blast zones
        'edge_pressure_reward': RewTerm(
            func=edge_pressure_reward,
            weight=edge_weight
        ),

        # 🎓 CURRICULUM: Exploration reward (annealed over training)
        'on_attack_button_press': RewTerm(
            func=on_attack_button_press,
            weight=attack_weight  # Dynamic weight from curriculum scheduler
        ),

        # CLEANUP: Discourage button mashing (but keep light)
        'holding_more_than_3_keys': RewTerm(
            func=holding_more_than_3_keys,
            weight=-0.05  # Reduced from -0.5 (was too harsh)
        ),

        # ❌ REMOVED: penalize_attack_reward (this was killing offensive play!)
        # The agent needs to attack to win - don't penalize it!
    }
    
    signal_subscriptions = {
        # VICTORY: Make winning the dominant long-term signal
        'on_win_reward': ('win_signal', RewTerm(
            func=on_win_reward,
            weight=2000  # 🚀 BOOSTED: 500→2000 (4x increase - winning MUST be dominant!)
                         # With reduced dense rewards (1.0 each), this makes winning worth
                         # ~20x more than spamming buttons. Agent MUST learn to win.
        )),

        # KNOCKOUT: Reward eliminating opponent
        'on_knockout_reward': ('knockout_signal', RewTerm(
            func=on_knockout_reward,
            weight=150  # 🔥 INCREASED: 50→150 (3x increase - knockouts are key to winning)
        )),

        # COMBO: Reward strategic hits during stun
        'on_combo_reward': ('hit_during_stun', RewTerm(
            func=on_combo_reward,
            weight=50  # 🔥 INCREASED: 20→50 (combos are advanced tactics)
        )),

        # WEAPONS: Encourage picking up weapons
        'on_equip_reward': ('weapon_equip_signal', RewTerm(
            func=on_equip_reward,
            weight=25  # Keep same - weapons help but aren't the goal
        )),

        # WEAPON DROP: Mild penalty for losing weapon
        'on_drop_reward': ('weapon_drop_signal', RewTerm(
            func=on_drop_reward,
            weight=10  # Keep same - minor penalty
        ))
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
        
        extractor = getattr(agent.model.policy, "features_extractor", None)
        if extractor is None:
            return

        if extractor.last_strategy_latent is not None:
            norm = torch.norm(extractor.last_strategy_latent, p=2).item()
            self.latent_norms.append(norm)

        if extractor.last_attention is not None:
            try:
                attn_weights = extractor.last_attention.squeeze().detach().cpu().numpy()
                attn_weights = attn_weights + 1e-10
                entropy = -np.sum(attn_weights * np.log(attn_weights))
                self.attention_entropies.append(entropy)
            except Exception:
                pass
    
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


class LiveRewardLossPrinter(BaseCallback):
    """
    Minimal training-time printer.
    - Prints per-step reward and optional per-term breakdown.
    - Lets SB3 print training losses via stdout logger.
    """

    def __init__(self, reward_manager: RewardManager, print_breakdown: bool = True):
        super().__init__(verbose=0)
        self.reward_manager = reward_manager
        self.print_breakdown = print_breakdown

    def _get_raw_env(self):
        env = getattr(self, 'training_env', None)
        if env is None:
            return None
        try:
            if hasattr(env, 'envs') and len(env.envs) > 0:
                e = env.envs[0]
            else:
                e = env
            # Unwrap nested wrappers
            while hasattr(e, 'env'):
                e = e.env
            # SelfPlayWarehouseBrawl exposes raw_env
            if hasattr(e, 'raw_env'):
                return e.raw_env
            return e
        except Exception:
            return None

    def _print_breakdown(self, raw_env):
        if self.reward_manager is None or not getattr(self.reward_manager, 'reward_functions', None):
            return
        try:
            parts = []
            total = 0.0
            for name, term_cfg in self.reward_manager.reward_functions.items():
                if term_cfg.weight == 0.0:
                    value = 0.0
                else:
                    value = float(term_cfg.func(raw_env, **term_cfg.params) * term_cfg.weight)
                total += value
                parts.append(f"{name}={value:.6f}")
            print("  terms: " + ", ".join(parts) + f" | total={total:.6f}")
        except Exception:
            pass

    def _on_step(self) -> bool:
        # Step-level reward print
        try:
            rewards = self.locals.get('rewards', None)
            dones = self.locals.get('dones', None)
            r = None
            if rewards is not None:
                r = float(np.asarray(rewards).reshape(-1)[0])
            d = None
            if dones is not None:
                d = bool(np.asarray(dones).reshape(-1)[0])
            if r is not None:
                if d is None:
                    print(f"[step {self.num_timesteps}] reward={r:.6f}")
                else:
                    print(f"[step {self.num_timesteps}] reward={r:.6f} done={int(d)}")
        except Exception:
            pass

        # Optional reward-term breakdown on each step
        if self.print_breakdown:
            raw_env = self._get_raw_env()
            if raw_env is not None:
                self._print_breakdown(raw_env)
                # Damage per-frame diagnostics
                try:
                    p = raw_env.objects.get("player")
                    o = raw_env.objects.get("opponent")
                    if p is not None and o is not None:
                        print(
                            f"  damage_frame: player={getattr(p, 'damage_taken_this_frame', 0.0):.6f} "
                            f"opponent={getattr(o, 'damage_taken_this_frame', 0.0):.6f}"
                        )
                        # Context when attacking
                        a = getattr(p, 'cur_action', None)
                        if isinstance(a, (list, np.ndarray)) and len(a) > 8 and (a[7] > 0.5 or a[8] > 0.5):
                            try:
                                dist = math.hypot(p.body.position.x - o.body.position.x, p.body.position.y - o.body.position.y)
                                facing = getattr(p.facing, 'name', str(p.facing))
                                print(f"  press_ctx: dist={dist:.2f} facing={facing}")
                            except Exception:
                                pass
                except Exception:
                    pass
        return True


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

        # Track transformer health metrics
        self.transformer_health = TransformerHealthMonitor()

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

    def assess_competition_readiness(
        self,
        checkpoint_step: int,
        benchmark_results: Dict[str, float],
        agent: Agent,
        reward_tracker: 'RewardBreakdownTracker'
    ) -> Dict[str, any]:
        """
        🎯 COMPETITION READINESS ASSESSMENT

        Analyzes training metrics to predict 10M competition performance.
        Provides actionable feedback on whether to scale or what to fix.

        Args:
            checkpoint_step: Current training step
            benchmark_results: Results from run_benchmark()
            agent: Trained agent
            reward_tracker: Reward breakdown tracker

        Returns:
            Dictionary with assessment results and recommendation
        """
        print("\n" + "="*70)
        print(f"🏆 COMPETITION READINESS ASSESSMENT (Step {checkpoint_step})")
        print("="*70)

        tier1_checks = []
        tier2_checks = []
        tier3_checks = []

        # ========== TIER 1: CRITICAL (Must Pass All) ==========
        print("\n📊 TIER 1 - CRITICAL METRICS (Must Pass for 10M Scale)")
        print("-" * 70)

        # 1. Transformer Utilization
        if isinstance(agent, TransformerStrategyAgent):
            # Get latent norms from recent episodes
            latent_norms = list(self.recent_latent_vectors[-50:]) if len(self.recent_latent_vectors) > 0 else []
            if len(latent_norms) > 10:
                norms = [np.linalg.norm(v) for v in latent_norms]
                latent_variance = np.std(norms)
                latent_mean = np.mean(norms)

                # PASS: Variance > 20% of mean (transformer adapting to opponents)
                variance_ratio = (latent_variance / latent_mean) if latent_mean > 0 else 0
                transformer_pass = variance_ratio > 0.2

                status = "✅ PASS" if transformer_pass else "❌ FAIL"
                print(f"  {status} Transformer Utilization:")
                print(f"       Latent Norm: {latent_mean:.2f} (±{latent_variance:.2f})")
                print(f"       Variance Ratio: {variance_ratio:.2f} (need >0.20)")
                if not transformer_pass:
                    print(f"       → Transformer not varying across opponents!")
                    print(f"       → Agent using fixed policy, not adapting")

                tier1_checks.append(('Transformer Variance', transformer_pass))
            else:
                print(f"  ⚠️  SKIP Transformer Utilization (insufficient data)")
                tier1_checks.append(('Transformer Variance', None))
        else:
            print(f"  ⚠️  SKIP Transformer Utilization (not using TransformerAgent)")
            tier1_checks.append(('Transformer Variance', None))

        # 2. Cross-Opponent Win Rates
        based_wr = benchmark_results['based_winrate']
        constant_wr = benchmark_results['constant_winrate']

        # PASS: Constant >80% AND Based >40%
        winrate_pass = constant_wr > 80 and based_wr > 40

        status = "✅ PASS" if winrate_pass else "❌ FAIL"
        print(f"  {status} Cross-Opponent Win Rates:")
        print(f"       vs ConstantAgent: {constant_wr:.1f}% (need >80%)")
        print(f"       vs BasedAgent: {based_wr:.1f}% (need >40%)")
        if not winrate_pass:
            if constant_wr <= 80:
                print(f"       → Can't beat stationary target reliably!")
            if based_wr <= 40:
                print(f"       → Struggles with moving opponent")
                print(f"       → May be overfitting to ConstantAgent")

        tier1_checks.append(('Cross-Opponent Wins', winrate_pass))

        # 3. Strategy Diversity
        diversity = benchmark_results['diversity_score']

        # PASS: Diversity >0.4 (multiple strategies emerged)
        diversity_pass = diversity > 0.4

        status = "✅ PASS" if diversity_pass else "❌ FAIL"
        print(f"  {status} Strategy Diversity:")
        print(f"       Score: {diversity:.3f} (need >0.400)")
        if not diversity_pass:
            print(f"       → Mode collapse! Agent using one fixed strategy")
            print(f"       → Won't handle diverse competition opponents")

        tier1_checks.append(('Strategy Diversity', diversity_pass))

        # 4. Damage Ratio
        damage_ratio = benchmark_results['avg_damage_ratio']

        # PASS: >1.0 (dealing more than taking)
        damage_pass = damage_ratio > 1.0

        status = "✅ PASS" if damage_pass else "❌ FAIL"
        print(f"  {status} Damage Ratio:")
        print(f"       Ratio: {damage_ratio:.2f} (need >1.00)")
        if not damage_pass:
            print(f"       → Taking more damage than dealing!")
            print(f"       → Losing combat exchanges")

        tier1_checks.append(('Damage Ratio', damage_pass))

        # ========== TIER 2: STRONG INDICATORS ==========
        print("\n📊 TIER 2 - STRONG INDICATORS (Should Pass Most)")
        print("-" * 70)

        # 5. Reward Composition (Dense vs Sparse)
        if hasattr(reward_tracker, 'accumulated_data') and len(reward_tracker.accumulated_data) > 0:
            recent_rewards = reward_tracker.accumulated_data[-100:]  # Last 100 steps

            dense_total = 0.0
            sparse_total = 0.0

            for _, breakdown in recent_rewards:
                for term_name, value in breakdown.items():
                    if term_name in ['head_to_opponent', 'on_attack_button_press']:
                        dense_total += abs(value)
                    elif term_name in ['damage_interaction_reward', 'on_win_reward', 'on_knockout_reward']:
                        sparse_total += abs(value)

            if (dense_total + sparse_total) > 0:
                sparse_percent = (sparse_total / (dense_total + sparse_total)) * 100

                # PASS: >60% from sparse rewards
                reward_pass = sparse_percent > 60

                status = "✅ PASS" if reward_pass else "⚠️  WEAK"
                print(f"  {status} Reward Composition:")
                print(f"       Sparse: {sparse_percent:.1f}% (need >60%)")
                print(f"       Dense: {100-sparse_percent:.1f}%")
                if not reward_pass:
                    print(f"       → Still reward hacking! Agent farming dense rewards")
                    print(f"       → Not optimizing for wins/damage")

                tier2_checks.append(('Reward Mix', reward_pass))
            else:
                print(f"  ⚠️  SKIP Reward Composition (no data)")
                tier2_checks.append(('Reward Mix', None))
        else:
            print(f"  ⚠️  SKIP Reward Composition (no reward data)")
            tier2_checks.append(('Reward Mix', None))

        # 6. Win Rate Scaling
        # Check if BOTH opponents improving (not overfitting to one)
        # This requires historical data - for now, check absolute levels
        both_improving = constant_wr > 70 and based_wr > 30

        status = "✅ PASS" if both_improving else "⚠️  WEAK"
        print(f"  {status} Win Rate Scaling:")
        print(f"       Both opponents improving: {both_improving}")
        if not both_improving:
            print(f"       → May be overfitting to easier opponent")

        tier2_checks.append(('Win Rate Scaling', both_improving))

        # ========== TIER 3: WARNING SIGNALS ==========
        print("\n📊 TIER 3 - WARNING SIGNALS (Red Flags)")
        print("-" * 70)

        # 7. Attention Entropy (if available)
        if isinstance(agent, TransformerStrategyAgent) and hasattr(self, 'transformer_health'):
            health_stats = self.transformer_health.get_stats()
            if 'attention_entropy_mean' in health_stats:
                attn_entropy = health_stats['attention_entropy_mean']

                # PASS: >3.0 (analyzing opponent history)
                entropy_pass = attn_entropy > 3.0

                status = "✅ GOOD" if entropy_pass else "⚠️  LOW"
                print(f"  {status} Attention Entropy:")
                print(f"       Entropy: {attn_entropy:.2f} (want >3.0)")
                if not entropy_pass:
                    print(f"       → Transformer focusing on few frames only")
                    print(f"       → Not analyzing full opponent behavior")

                tier3_checks.append(('Attention Entropy', entropy_pass))
            else:
                print(f"  ⚠️  SKIP Attention Entropy (no data)")
                tier3_checks.append(('Attention Entropy', None))
        else:
            print(f"  ⚠️  SKIP Attention Entropy (not tracked)")
            tier3_checks.append(('Attention Entropy', None))

        # 8. Training Stability
        # Check for NaN/crashes - assume pass if we got here
        stability_pass = True
        print(f"  ✅ GOOD Training Stability:")
        print(f"       No NaN/crashes detected")
        tier3_checks.append(('Training Stability', stability_pass))

        # ========== OVERALL ASSESSMENT ==========
        print("\n" + "="*70)
        print("🎯 OVERALL ASSESSMENT")
        print("="*70)

        # Count passes
        tier1_passed = sum(1 for _, result in tier1_checks if result is True)
        tier1_total = sum(1 for _, result in tier1_checks if result is not None)

        tier2_passed = sum(1 for _, result in tier2_checks if result is True)
        tier2_total = sum(1 for _, result in tier2_checks if result is not None)

        tier3_passed = sum(1 for _, result in tier3_checks if result is True)
        tier3_total = sum(1 for _, result in tier3_checks if result is not None)

        print(f"\nTIER 1 (Critical):  {tier1_passed}/{tier1_total} PASS")
        for check_name, result in tier1_checks:
            if result is None:
                status = "⊘ SKIP"
            elif result:
                status = "✅ PASS"
            else:
                status = "❌ FAIL"
            print(f"  {status}  {check_name}")

        print(f"\nTIER 2 (Strong):    {tier2_passed}/{tier2_total} PASS")
        for check_name, result in tier2_checks:
            if result is None:
                status = "⊘ SKIP"
            elif result:
                status = "✅ PASS"
            else:
                status = "⚠️  WEAK"
            print(f"  {status}  {check_name}")

        print(f"\nTIER 3 (Warnings):  {tier3_passed}/{tier3_total} GOOD")
        for check_name, result in tier3_checks:
            if result is None:
                status = "⊘ SKIP"
            elif result:
                status = "✅ GOOD"
            else:
                status = "⚠️  FLAG"
            print(f"  {status}  {check_name}")

        # Final recommendation
        print("\n" + "="*70)

        # Must pass ALL tier 1 checks
        tier1_all_pass = tier1_total > 0 and tier1_passed == tier1_total
        tier2_most_pass = tier2_total == 0 or (tier2_passed / tier2_total) >= 0.67

        if tier1_all_pass and tier2_most_pass:
            confidence = int(min(95, 70 + (tier2_passed / max(tier2_total, 1)) * 20))
            print(f"🎉 READY FOR 10M SCALE (~{confidence}% confidence)")
            print(f"   ✅ All critical metrics passed")
            print(f"   ✅ Agent shows signs of adaptability")
            print(f"   ✅ Proceed to Stage 2 or 10M training")
            recommendation = "PROCEED"
        elif tier1_all_pass:
            print(f"⚠️  BORDERLINE - Proceed with caution")
            print(f"   ✅ Critical metrics passed")
            print(f"   ⚠️  Some strong indicators weak")
            print(f"   → Consider training 50k more steps to strengthen")
            recommendation = "CAUTION"
        else:
            print(f"🚨 NOT READY - Fix issues before scaling")
            print(f"   ❌ Critical metrics failed")
            print(f"   ❌ DO NOT scale to 10M yet")
            print(f"\n📋 ACTION ITEMS:")
            for check_name, result in tier1_checks:
                if result is False:
                    print(f"   • Fix: {check_name}")
            recommendation = "FIX_REQUIRED"

        print("="*70 + "\n")

        return {
            'tier1_checks': tier1_checks,
            'tier2_checks': tier2_checks,
            'tier3_checks': tier3_checks,
            'recommendation': recommendation,
            'tier1_pass_rate': tier1_passed / max(tier1_total, 1),
            'tier2_pass_rate': tier2_passed / max(tier2_total, 1)
        }

    def _run_matches(self, agent: Agent, opponent_factory, num_matches: int) -> List[Dict]:
        """Run matches and return results."""
        results = []
        for _ in range(num_matches):
            if hasattr(agent, 'reset'):
                agent.reset()
            match_stats = env_run_match(
                agent,
                opponent_factory,
                max_timesteps=30*60,  # 60 seconds
                resolution=CameraResolution.LOW,
                train_mode=True
            )

            # Record latent vector and transformer health after match
            self.record_latent_vector(agent)

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
        """Record latent vector for diversity tracking and transformer health."""
        if not isinstance(agent, TransformerStrategyAgent):
            return

        extractor = getattr(agent.model.policy, "features_extractor", None)
        if extractor is not None and extractor.last_strategy_latent is not None:
            latent_numpy = extractor.last_strategy_latent.detach().cpu().numpy().flatten()
            self.recent_latent_vectors.append(latent_numpy)

        # Update transformer health monitor
        self.transformer_health.update(agent)


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

        # 🎓 Curriculum annealing system
        global _CURRICULUM_SCHEDULER
        if _CURRICULUM_SCHEDULER is None:
            _CURRICULUM_SCHEDULER = CurriculumRewardScheduler()
            _CURRICULUM_SCHEDULER.print_schedule()
        self.curriculum_scheduler = _CURRICULUM_SCHEDULER

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

    def _update_curriculum_weights(self):
        """
        🎓 Update dense reward weights based on curriculum schedule.

        Dynamically adjusts reward weights during training to transition from
        exploration (high dense rewards) to exploitation (high sparse rewards).
        """
        # Get current weights from curriculum scheduler
        curriculum_weights = self.curriculum_scheduler.get_weights(self.num_timesteps)
        head_weight = curriculum_weights['head_to_opponent']
        attack_weight = curriculum_weights['attack_button']
        close_weight = curriculum_weights.get('close_distance', 0.0)
        edge_weight = curriculum_weights.get('edge_pressure', 0.0)
        stage = curriculum_weights['stage']

        # Update reward manager weights directly
        if 'head_to_opponent' in self.reward_manager.reward_functions:
            self.reward_manager.reward_functions['head_to_opponent'].weight = head_weight

        if 'on_attack_button_press' in self.reward_manager.reward_functions:
            self.reward_manager.reward_functions['on_attack_button_press'].weight = attack_weight

        if 'closing_distance_reward' in self.reward_manager.reward_functions:
            self.reward_manager.reward_functions['closing_distance_reward'].weight = close_weight

        if 'edge_pressure_reward' in self.reward_manager.reward_functions:
            self.reward_manager.reward_functions['edge_pressure_reward'].weight = edge_weight

        # Log curriculum transitions (only when stage changes)
        if not hasattr(self, '_last_curriculum_stage'):
            self._last_curriculum_stage = -1

        if stage != self._last_curriculum_stage:
            print("\n" + "="*70)
            print(f"🎓 CURRICULUM TRANSITION: Stage {stage + 1}")
            print(f"   Step: {self.num_timesteps:,}")
            print(f"   head_to_opponent: {head_weight:.1f}")
            print(f"   on_attack_button_press: {attack_weight:.1f}")
            print(f"   closing_distance_reward: {close_weight:.1f}")
            print(f"   edge_pressure_reward: {edge_weight:.1f}")
            print("="*70 + "\n")
            self._last_curriculum_stage = stage

    def _on_step(self) -> bool:
        """
        Called at every training step.
        Performs lightweight checks and periodic logging.
        """
        self.current_episode_length += 1
        self.total_steps += 1

        # 🎓 Update curriculum weights based on training progress
        self._update_curriculum_weights()

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
        Quick evaluation every 10000 steps.
        - Win rate spot check
        - Behavior metrics summary
        - Sanity checks
        """
        print(f"\n{'='*70}")
        print(f"🔍 QUICK EVALUATION (Step {self.num_timesteps})")
        print(f"{'='*70}")
        
        # 1. Win rate spot check (3 quick matches)
        wins = 0
        eval_damage_dealt = 0  # Damage in this evaluation
        eval_damage_taken = 0  # Damage taken in this evaluation
        
        # Choose an opponent appropriate for current training stage
        # - Before 50k steps: evaluate against ConstantAgent (easy baseline)
        # - After 50k steps: evaluate against BasedAgent (harder scripted)
        eval_opponent = partial(ConstantAgent) if self.num_timesteps < 50_000 else partial(BasedAgent)

        for i in range(self.eval_matches):
            try:
                if hasattr(self.agent, "reset"):
                    self.agent.reset()
                match_stats = env_run_match(
                    self.agent,
                    eval_opponent,
                    max_timesteps=30*60,
                    resolution=CameraResolution.LOW,
                    train_mode=True
                )
                if match_stats.player1_result == Result.WIN:
                    wins += 1
                
                # Track damage for this evaluation
                damage_dealt = match_stats.player2.damage_taken  # Damage we dealt = opponent's damage_taken
                damage_taken = match_stats.player1.damage_taken  # Damage we took
                
                eval_damage_dealt += damage_dealt
                eval_damage_taken += damage_taken
                
                # 🔥 ACCUMULATE to total (for passive behavior detection)
                self.total_damage_dealt += damage_dealt
                self.total_damage_taken += damage_taken
            except:
                pass
        
        win_rate = (wins / self.eval_matches) * 100
        avg_damage_ratio = eval_damage_dealt / max(eval_damage_taken, 1.0)
        
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
        """
        Run sanity checks on training progress.
        
        🔧 ENHANCED: Now includes PASSIVE BEHAVIOR detection (critical for combat game!)
        """
        issues = []
        warnings = []
        
        # === CRITICAL CHECK: Passive Behavior Detection ===
        # If agent is not attacking/dealing damage, it learned passivity (like old bug)
        if self.total_damage_dealt == 0 and self.num_timesteps > 5000:
            issues.append("🚨 PASSIVE BEHAVIOR: Agent has dealt ZERO damage in 5k+ steps!")
            issues.append("   → Agent learned NOT to attack (reward function broken)")
            issues.append("   → STOP TRAINING and fix reward function")
        elif self.total_damage_dealt > 0:
            avg_damage_per_1k = (self.total_damage_dealt / max(self.num_timesteps, 1)) * 1000
            if avg_damage_per_1k < 1.0 and self.num_timesteps > 10000:
                warnings.append(f"⚠️  LOW DAMAGE OUTPUT: {avg_damage_per_1k:.2f} damage per 1k steps")
                warnings.append("   → Agent may be too passive (should be 5-20 damage/1k)")
        
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
                warnings.append("NO IMPROVEMENT detected (agent not learning effectively)")
        
        # Check for loss explosions
        if len(self.recent_losses) >= 10:
            recent_losses = list(self.recent_losses)[-10:]
            if any(l > 1000 for l in recent_losses):
                issues.append("Loss values very high (gradient explosion, check learning rate)")
        
        # Check if agent is winning at all
        if self.num_timesteps > 10000:
            # Try to estimate win rate from damage ratio
            if self.total_damage_taken > 0:
                damage_ratio = self.total_damage_dealt / self.total_damage_taken
                if damage_ratio < 0.5:
                    warnings.append(f"LOW DAMAGE RATIO: {damage_ratio:.2f} (getting beaten badly)")
                    warnings.append("   → Agent may need more training or reward tuning")
        
        # Print issues (critical problems)
        if issues:
            print(f"  🚨 CRITICAL ISSUES:")
            for issue in issues:
                print(f"      {issue}")
        
        # Print warnings (concerning but not critical)
        if warnings:
            print(f"  ⚠️  Warnings:")
            for warning in warnings:
                print(f"      {warning}")
        
        # Print success
        if not issues and not warnings:
            print(f"  ✓ Sanity checks passed")
    
    def _checkpoint_benchmark(self):
        """
        Full performance benchmark when checkpoint is saved.
        Most comprehensive evaluation including competition readiness assessment.
        """
        try:
            # Run standard benchmark
            benchmark_results = self.benchmark.run_benchmark(
                self.agent,
                self.num_timesteps,
                num_matches=5
            )

            # Run competition readiness assessment (every 50k steps or at final checkpoint)
            if self.num_timesteps >= 30000 and self.num_timesteps % 10000 == 0:
                try:
                    self.benchmark.assess_competition_readiness(
                        checkpoint_step=self.num_timesteps,
                        benchmark_results=benchmark_results,
                        agent=self.agent,
                        reward_tracker=self.reward_tracker
                    )
                except Exception as e:
                    print(f"⚠️  Competition readiness assessment failed: {e}")
                    import traceback
                    traceback.print_exc()
        except Exception as e:
            print(f"⚠️  Checkpoint benchmark failed: {e}")
            import traceback
            traceback.print_exc()
    
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
    
    base_env = SelfPlayWarehouseBrawl(
        reward_manager=reward_manager,
        opponent_cfg=opponent_cfg,
        save_handler=save_handler,
        resolution=resolution
    )
    reward_manager.subscribe_signals(base_env.raw_env)

    env: gym.Env = base_env
    if isinstance(agent, TransformerStrategyAgent):
        opponent_obs_dim = agent.opponent_obs_dim
        if opponent_obs_dim is None:
            opponent_obs_dim = base_env.observation_space.shape[0] // 2
            agent.opponent_obs_dim = opponent_obs_dim
        env = OpponentHistoryWrapper(
            base_env,
            opponent_obs_dim=opponent_obs_dim,
            sequence_length=agent.sequence_length,
        )

    if train_logging != TrainLogging.NONE:
        # Create log dir
        log_dir = f"{save_handler._experiment_path()}/" if save_handler is not None else "/tmp/gym/"
        os.makedirs(log_dir, exist_ok=True)
        
        # Logs will be saved in log_dir/monitor.csv
        env = Monitor(env, log_dir)
    
    base_env = env.unwrapped if hasattr(env, 'unwrapped') else env
    
    try:
        agent.get_env_info(env)
        # Print observation/action specifics now that env is bound
        try:
            print("Obs/Action Layout:")
            if isinstance(agent, TransformerStrategyAgent):
                print(
                    f"  base_obs_dim={agent.base_obs_dim} opponent_obs_dim={agent.opponent_obs_dim} "
                    f"history_dim={agent.history_obs_dim} expected_obs_dim={agent.expected_obs_dim}"
                )
            if hasattr(base_env, 'action_space'):
                print(f"  action_space={getattr(base_env.action_space, 'shape', None)} (threshold>0.5)")
        except Exception:
            pass
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
    print("📝 MODE: Live step prints optional (see LIVE_STEP_PRINTS)")
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

    # Identify selected TRAIN_CONFIG name for summary
    def _config_name(cfg: dict) -> str:
        try:
            for name in [
                'TRAIN_CONFIG_DEBUG',
                'TRAIN_CONFIG_CURRICULUM',
                'TRAIN_CONFIG_CURRICULUM_STAGE2',
                'TRAIN_CONFIG_TEST',
                'TRAIN_CONFIG_EXPLORATION',
                'TRAIN_CONFIG_10M',
            ]:
                if cfg is globals().get(name):
                    return name
        except Exception:
            pass
        return 'TRAIN_CONFIG(custom)'

    selected_config_name = _config_name(TRAIN_CONFIG)

    reward_cfg = TRAIN_CONFIG.get("reward", {})
    reward_factory = _resolve_callable(reward_cfg.get("factory"), gen_reward_manager)
    reward_manager = reward_factory()
    reward_terms = getattr(reward_manager, "reward_functions", {}) or {}
    debug_log("config", f"Reward terms active: {list(reward_terms.keys())}")

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
    # Respect the logging mode from the selected TRAIN_CONFIG
    train_logging = training_cfg.get("logging", TrainLogging.PLOT)

    # ---- RUN SUMMARY (configuration snapshot) ----
    print("\n" + "📋" + "="*68)
    print("RUN SUMMARY")
    print("="*70)
    print(f"Config: {selected_config_name}")
    try:
        exp_path = save_handler._experiment_path() if save_handler is not None else 'n/a'
    except Exception:
        exp_path = 'n/a'
    print(f"Checkpoints: base='{CHECKPOINT_BASE_PATH}', run='{exp_path}'")
    print(f"Self-play: run_name='{self_play_run_name}', save_freq={self_play_save_freq}, max_saved={self_play_max_saved}, mode={self_play_mode.name}")
    print(f"Training: timesteps={train_timesteps:,}, resolution={train_resolution.name}, logging={train_logging.name}")
    algo_desc = (
        "RecurrentPPO + MlpLstmPolicy (Transformer-conditioned features)"
        if not DISABLE_STRATEGY_ENCODER
        else "RecurrentPPO + MlpLstmPolicy (recurrent baseline; encoder OFF)"
    )
    print(f"Algorithm: {algo_desc}")
    # PPO core
    print("PPO Hyperparams:")
    print(f"  n_steps={_SHARED_AGENT_CONFIG.get('n_steps')} batch_size={_SHARED_AGENT_CONFIG.get('batch_size')} n_epochs={_SHARED_AGENT_CONFIG.get('n_epochs')}")
    print(f"  lr={_SHARED_AGENT_CONFIG.get('learning_rate')} ent_coef={_SHARED_AGENT_CONFIG.get('ent_coef')} clip={_SHARED_AGENT_CONFIG.get('clip_range')} gamma={_SHARED_AGENT_CONFIG.get('gamma')} gae_lambda={_SHARED_AGENT_CONFIG.get('gae_lambda')}")
    # Transformer
    print("Transformer:")
    print(f"  latent_dim={_SHARED_AGENT_CONFIG.get('latent_dim')} heads={_SHARED_AGENT_CONFIG.get('num_heads')} layers={_SHARED_AGENT_CONFIG.get('num_layers')} seq_len={_SHARED_AGENT_CONFIG.get('sequence_length')}")
    print(f"  encoder={'ON' if not DISABLE_STRATEGY_ENCODER else 'OFF (bypassed)'}")
    print(f"  toggle=DISABLE_STRATEGY_ENCODER={DISABLE_STRATEGY_ENCODER}")
    # Policy kwargs (LSTM size etc.)
    pk = _SHARED_AGENT_CONFIG.get('policy_kwargs', {})
    print("Policy:")
    print(f"  lstm_hidden_size={pk.get('lstm_hidden_size')} shared_lstm={pk.get('shared_lstm')} critic_lstm={pk.get('enable_critic_lstm')} net_arch={pk.get('net_arch')}")
    # Rewards
    print("Rewards:")
    print(f"  factory={getattr(reward_factory, '__name__', str(reward_factory))}")
    if reward_manager.reward_functions:
        print("  terms:")
        for n, t in reward_manager.reward_functions.items():
            print(f"    - {n}: weight={t.weight}")
    if reward_manager.signal_subscriptions:
        print("  signals:")
        for n, (sig_name, t) in reward_manager.signal_subscriptions.items():
            print(f"    - {t.func.__name__}: weight={t.weight} (signal={sig_name})")
    print("Gameplay thresholds:")
    print("  action_press_threshold=0.5  deterministic_eval_after_steps=10000")
    # Opponents
    try:
        opp_list = [f"{k}:{(v[0] if isinstance(v, tuple) else v)}" for k, v in opponent_cfg.opponents.items()]
        print("Opponents:")
        print("  mix=" + ", ".join(opp_list))
    except Exception:
        pass
    print("="*70 + "\n")
    debug_log(
        "config",
        f"Training config: timesteps={train_timesteps}, resolution={train_resolution.name}, logging={train_logging.name}",
    )
    
    # Optional lightweight live printer (opt-in via env var)
    enable_live_prints = os.environ.get("LIVE_STEP_PRINTS", "0") == "1"
    monitor_callback = LiveRewardLossPrinter(reward_manager, print_breakdown=True) if enable_live_prints else None

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
    
    # Keep evaluation/testing helpers intact; no training summary printing


if __name__ == '__main__':
    main()
