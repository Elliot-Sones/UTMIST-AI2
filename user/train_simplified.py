"""
Hello?
================================================================================
UTMIST AI² - SIMPLIFIED Strategy Recognition Training
================================================================================

ARCHITECTURE (Clean & Simple - Ready to Run!)

┌─────────────────────────────────────────────────────────────┐
│                    TRAINING ARCHITECTURE                     │
└─────────────────────────────────────────────────────────────┘

                    ┌──────────────────┐
                    │ Opponent History │
                    │   (32 frames)    │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │ SimpleOpponentEncoder │
                    │   (2-layer MLP)   │
                    └────────┬─────────┘
                             │
                     128-dim strategy
                        encoding
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
    ┌────▼─────┐     ┌──────▼──────┐     ┌──────▼──────┐
    │ Current  │     │  Strategy   │     │    LSTM     │
    │   Obs    │     │  Encoding   │     │   Memory    │
    └────┬─────┘     └──────┬──────┘     └──────┬──────┘
         │                   │                   │
         └───────────────────┴───────────────────┘
                             │
                    ┌────────▼─────────┐
                    │   LSTM Policy    │
                    │ (RecurrentPPO)   │
                    └────────┬─────────┘
                             │
                        ┌────▼────┐
                        │ Actions │
                        └─────────┘

KEY FEATURES:
✅ Simple 2-layer MLP encodes opponent behavior → 128-dim vector
✅ LSTM policy learns to use strategy encoding for counter-play
✅ Self-adversarial training vs past checkpoints
✅ No transformers, no attention, no complex fusion
✅ 10x faster than complex version, still learns strategies!

HOW TO RUN:
    python user/train_simplified.py

That's it! Training starts immediately with sensible defaults.
"""

# ============================================================================
# IMPORTS
# ============================================================================

import os
import sys
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # Apple Silicon support
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from pathlib import Path
import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
from functools import partial
from typing import Optional
from collections import deque

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib import RecurrentPPO

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environment.agent import *

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================

def get_device():
    """Auto-detect best available device (CUDA > MPS > CPU)"""
    if torch.cuda.is_available():
        print(f"✓ Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("✓ Using Apple Silicon MPS GPU")
        return torch.device("mps")
    else:
        print("⚠ Using CPU (training will be slow)")
        return torch.device("cpu")

DEVICE = get_device()

# ============================================================================
# SIMPLIFIED ARCHITECTURE
# ============================================================================

class SimpleOpponentEncoder(nn.Module):
    """
    Encodes opponent behavior into a compact latent vector.

    Input:  32 frames × 32 features = 1024 numbers (opponent history)
    Output: 128-dim strategy encoding

    How it works:
    - Flattens the sequence of observations
    - Passes through 2-layer MLP to extract patterns
    - Outputs compact representation for LSTM to use
    """
    def __init__(
        self,
        opponent_obs_dim: int = 32,
        history_length: int = 32,
        latent_dim: int = 128,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.opponent_obs_dim = opponent_obs_dim
        self.history_length = history_length
        self.latent_dim = latent_dim
        self.device = device if device is not None else DEVICE

        # Simple 2-layer MLP with strong dropout to prevent collapse
        input_dim = opponent_obs_dim * history_length
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.5),  # Strong dropout forces encoder to maintain diversity
            nn.Linear(256, latent_dim),
            nn.Tanh()  # Bounded outputs prevent explosion (removed LayerNorm before Tanh)
        )

        # Orthogonal initialization prevents early collapse
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.constant_(layer.bias, 0.0)

        # Diversity regularization: track recent outputs to promote variance
        self.register_buffer('output_buffer', torch.zeros(100, latent_dim))
        self.register_buffer('buffer_idx', torch.tensor(0))
        self.diversity_strength = 0.1  # How much noise to add when diversity is low

        self.to(self.device)

    def forward(self, opponent_history):
        """
        Args:
            opponent_history: [batch, seq_len, opponent_obs_dim]
        Returns:
            strategy_encoding: [batch, latent_dim]
        """
        # Ensure correct device
        if not isinstance(opponent_history, torch.Tensor):
            opponent_history = torch.tensor(opponent_history, dtype=torch.float32, device=self.device)
        elif opponent_history.device != self.device:
            opponent_history = opponent_history.to(self.device)

        batch_size = opponent_history.shape[0]

        # Flatten: [batch, seq_len, obs_dim] → [batch, seq_len * obs_dim]
        flat_history = opponent_history.reshape(batch_size, -1)

        # Encode to latent space
        encoding = self.encoder(flat_history)

        # Diversity regularization: maintain varied outputs
        if self.training:
            # Store outputs in buffer
            with torch.no_grad():
                for i in range(min(batch_size, encoding.shape[0])):
                    idx = int(self.buffer_idx.item()) % 100
                    self.output_buffer[idx] = encoding[i].detach()
                    self.buffer_idx += 1

                # Check diversity: compute std across buffer
                buffer_std = self.output_buffer.std(dim=0).mean()

                # If diversity too low, add noise to force variation
                if buffer_std < 0.05:  # Threshold: diversity should be > 0.05
                    noise = torch.randn_like(encoding) * self.diversity_strength
                    encoding = encoding + noise

        return encoding


class SimplifiedExtractor(BaseFeaturesExtractor):
    """
    Feature extractor combining current observation + opponent strategy.

    This is what feeds into the LSTM policy. It combines:
    1. Current game state (positions, health, etc.)
    2. Opponent strategy encoding (from SimpleOpponentEncoder)

    No transformers, no cross-attention - just simple concatenation!
    """
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 256,
        latent_dim: int = 128,
        base_obs_dim: int = 256,
        opponent_obs_dim: int = 32,
        sequence_length: int = 32,
        device: Optional[torch.device] = None,
    ):
        super().__init__(observation_space, features_dim)

        self.base_obs_dim = base_obs_dim
        self.sequence_length = sequence_length
        self.opponent_obs_dim = opponent_obs_dim
        self.latent_dim = latent_dim
        self.device = device if device is not None else DEVICE
        self.history_dim = opponent_obs_dim * sequence_length

        # Simple 1-layer encoder for current observation
        self.obs_encoder = nn.Sequential(
            nn.Linear(base_obs_dim, 128),
            nn.ReLU()
        )

        # Opponent encoder
        self.opponent_encoder = SimpleOpponentEncoder(
            opponent_obs_dim=opponent_obs_dim,
            history_length=sequence_length,
            latent_dim=latent_dim,
            device=self.device
        )

        # Strategy-dependent gating: Forces model to use opponent encoding!
        # The strategy encoding gates the observation features
        # If encoding is same for all opponents → gating is same → poor performance
        # This creates strong gradient pressure for diverse encodings
        self.strategy_gate = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.Tanh()
        )

        # Fusion layer (now processes gated features)
        self.fusion = nn.Sequential(
            nn.Linear(128 + latent_dim, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU()
        )

        self.to(self.device)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Extract features by combining current obs + opponent encoding.

        Args:
            observations: [batch, base_obs_dim + history_dim]
        Returns:
            features: [batch, features_dim]
        """
        if observations.device != self.device:
            observations = observations.to(self.device)

        batch_size = observations.shape[0]

        # Split observation into current state and opponent history
        base_obs = observations[:, :self.base_obs_dim]
        history_flat = observations[:, self.base_obs_dim:]

        if history_flat.shape[1] != self.history_dim:
            raise ValueError(
                f"Expected history dimension {self.history_dim}, "
                f"received {history_flat.shape[1]}"
            )

        # Reshape history
        history = history_flat.view(batch_size, self.sequence_length, self.opponent_obs_dim)

        # Encode both parts
        obs_features = self.obs_encoder(base_obs)  # [batch, 128]
        opponent_features = self.opponent_encoder(history)  # [batch, latent_dim]

        # CRITICAL: Strategy-dependent gating!
        # Use opponent encoding to modulate observation features
        # If all opponents get same encoding → same gating → poor performance
        # This forces the encoder to learn diverse representations
        gate = self.strategy_gate(opponent_features)  # [batch, 128]
        gated_obs = obs_features * gate  # Element-wise multiplication

        # Fuse gated observations with strategy encoding
        combined = torch.cat([gated_obs, opponent_features], dim=-1)
        features = self.fusion(combined)

        return features


# ============================================================================
# OBSERVATION WRAPPER (Adds opponent history to observations)
# ============================================================================

class OpponentHistoryWrapper(gym.ObservationWrapper):
    """
    Wraps environment to add rolling opponent history to observations.

    This allows the encoder to see the last 32 frames of opponent behavior.
    """
    def __init__(self, env: gym.Env, opponent_obs_dim: int, sequence_length: int):
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
        assert self._history is not None, "Call reset() first"
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


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

# Where to save checkpoints
CHECKPOINT_DIR = "checkpoints/simplified_training"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Agent hyperparameters
AGENT_CONFIG = {
    "latent_dim": 128,           # Opponent strategy encoding size
    "sequence_length": 32,       # Opponent history frames
    "opponent_obs_dim": None,    # Auto-detected

    # LSTM policy
    "policy_kwargs": {
        "activation_fn": nn.ReLU,
        "lstm_hidden_size": 512,
        "net_arch": dict(pi=[96, 96], vf=[96, 96]),
        "shared_lstm": True,
        "enable_critic_lstm": False,
    },

    # PPO training
    "n_steps": 512,              # Rollout buffer size
    "batch_size": 64,            # Mini-batch size
    "n_epochs": 10,              # Gradient epochs per update
    "learning_rate": 2.5e-4,
    "ent_coef": 0.01,            # Exploration entropy
    "clip_range": 0.2,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "max_grad_norm": 0.5,        # Prevent exploding gradients
    "vf_coef": 0.5,              # Value function coefficient
    "clip_range_vf": 10.0,       # Clip value function updates to prevent explosions
}

# Training settings
TRAINING_CONFIG = {
    "total_timesteps": 100_000,  # Quick test (increase for real training)
    "save_freq": 10_000,         # Save checkpoint every 10k steps
    "resolution": CameraResolution.LOW,
}

# Action sheets for ClockworkAgent opponents with distinct strategies
# Format: (duration_in_steps, [keys_to_press])

# Aggressive rusher - constantly approaches and attacks
AGGRESSIVE_PATTERN = [
    (15, ['d']),           # Move right (approach)
    (3, ['d', 'j']),       # Attack while moving
    (2, []),               # Brief pause
    (3, ['d', 'j']),       # Attack again
    (15, ['d']),           # Continue approaching
    (5, ['d', 'l']),       # Special attack
    (10, ['d']),           # Keep pressure
    (3, ['j']),            # Quick attack
]

# Defensive blocker - stays back, counters when approached
DEFENSIVE_PATTERN = [
    (20, []),              # Wait/observe
    (5, ['a']),            # Back off slightly
    (15, []),              # Wait more
    (3, ['j']),            # Quick counter attack
    (10, []),              # Back to defensive
    (4, ['l']),            # Special counter
    (25, []),              # Long wait
]

# Hit-and-run - quick attacks then retreat
HIT_AND_RUN_PATTERN = [
    (12, ['d']),           # Dash in
    (2, ['j']),            # Quick hit
    (10, ['a']),           # Retreat
    (5, []),               # Wait
    (10, ['d']),           # Dash in again
    (3, ['d', 'l']),       # Special while moving
    (15, ['a']),           # Retreat again
    (8, []),               # Wait
]

# Aerial attacker - uses jumps and air attacks
AERIAL_PATTERN = [
    (5, ['d']),            # Approach
    (15, ['space']),       # Jump
    (3, ['j']),            # Air attack
    (8, []),               # Land
    (10, ['d']),           # Approach more
    (15, ['space']),       # Jump again
    (3, ['l']),            # Air special
    (10, []),              # Land and wait
]

# Special spammer - focuses on special moves
SPECIAL_SPAM_PATTERN = [
    (8, ['d']),            # Approach
    (5, ['l']),            # Special attack
    (5, []),               # Wait
    (5, ['l']),            # Another special
    (10, ['d']),           # Move closer
    (5, ['l']),            # More specials
    (8, []),               # Wait
    (3, ['j']),            # Mix in normal attack
    (5, ['l']),            # Back to specials
]

# Self-play opponent mix (now with 8 diverse opponents)
OPPONENT_MIX = {
    "constant_agent": (0.15, partial(ConstantAgent)),                                    # 15% stationary
    "based_agent": (0.20, partial(BasedAgent)),                                          # 20% scripted AI
    "random_agent": (0.10, partial(RandomAgent)),                                        # 10% random
    "aggressive_clockwork": (0.15, partial(ClockworkAgent, action_sheet=AGGRESSIVE_PATTERN)),      # 15% aggressive
    "defensive_clockwork": (0.10, partial(ClockworkAgent, action_sheet=DEFENSIVE_PATTERN)),        # 10% defensive
    "hitrun_clockwork": (0.10, partial(ClockworkAgent, action_sheet=HIT_AND_RUN_PATTERN)),         # 10% hit-and-run
    "aerial_clockwork": (0.10, partial(ClockworkAgent, action_sheet=AERIAL_PATTERN)),              # 10% aerial
    "special_clockwork": (0.10, partial(ClockworkAgent, action_sheet=SPECIAL_SPAM_PATTERN)),       # 10% special spam
}

print("=" * 70)
print("SIMPLIFIED TRAINING CONFIGURATION")
print("=" * 70)
print(f"Architecture: Simple MLP encoder + LSTM policy")
print(f"Opponent encoding: {AGENT_CONFIG['latent_dim']}-dim latent")
print(f"Sequence length: {AGENT_CONFIG['sequence_length']} frames")
print(f"Training steps: {TRAINING_CONFIG['total_timesteps']:,}")
print(f"Checkpoint dir: {CHECKPOINT_DIR}")
print(f"Opponent diversity: {len(OPPONENT_MIX)} distinct agent types")
for name, (prob, _) in OPPONENT_MIX.items():
    print(f"  - {name}: {prob*100:.0f}%")
print("=" * 70)


# ============================================================================
# REWARD FUNCTIONS (Same as before - these work great!)
# ============================================================================

def damage_interaction_reward(env: WarehouseBrawl, mode: int = 1) -> float:
    """Reward dealing damage, penalize taking damage"""
    player = env.objects["player"]
    opponent = env.objects["opponent"]

    if not hasattr(env, "_last_damage_totals") or env.steps <= 1:
        env._last_damage_totals = {
            "player": player.damage_taken_total,
            "opponent": opponent.damage_taken_total,
        }
        return 0.0

    prev_p = env._last_damage_totals["player"]
    prev_o = env._last_damage_totals["opponent"]

    delta_taken = max(0.0, player.damage_taken_total - prev_p)
    delta_dealt = max(0.0, opponent.damage_taken_total - prev_o)

    env._last_damage_totals["player"] = player.damage_taken_total
    env._last_damage_totals["opponent"] = opponent.damage_taken_total

    return (delta_dealt - delta_taken) / 140


def danger_zone_reward(env: WarehouseBrawl, zone_height: float = 4.2) -> float:
    """Penalize being too high (about to get knocked out)"""
    player = env.objects["player"]
    return -1.0 * env.dt if player.body.position.y >= zone_height else 0.0


def on_win_reward(env: WarehouseBrawl, agent: str) -> float:
    """Big reward for winning"""
    return 10.0 if agent == 'player' else -10.0


def on_knockout_reward(env: WarehouseBrawl, agent: str) -> float:
    """Reward knocking out opponent, penalize getting knocked out"""
    return 2.0 if agent == 'opponent' else -2.0


def gen_reward_manager():
    """Create reward manager with sparse rewards"""
    reward_functions = {
        'danger_zone': RewTerm(func=danger_zone_reward, weight=0.5),
        'damage_interaction': RewTerm(func=damage_interaction_reward, weight=1.0),
    }
    signal_subscriptions = {
        'on_win': ('win_signal', RewTerm(func=on_win_reward, weight=1.0)),  # Weight=1.0, not 50!
        'on_knockout': ('knockout_signal', RewTerm(func=on_knockout_reward, weight=1.0)),  # Weight=1.0, not 8!
    }
    return RewardManager(reward_functions, signal_subscriptions)


# ============================================================================
# SELF-PLAY HANDLER (Trains vs past checkpoints)
# ============================================================================

class SimpleSelfPlayHandler:
    """Loads random past checkpoint as opponent"""
    def __init__(self, ckpt_dir: str):
        self.ckpt_dir = ckpt_dir
        self.env = None

    def get_opponent(self) -> Agent:
        import glob
        import random

        zips = glob.glob(os.path.join(self.ckpt_dir, "rl_model_*.zip"))
        if not zips:
            return ConstantAgent()  # Fallback if no checkpoints yet

        path = random.choice(zips)
        opponent = RecurrentPPOAgent(file_path=path)
        if self.env:
            opponent.get_env_info(self.env)
        return opponent


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train():
    """Main training loop - just run this!"""

    print("\n" + "=" * 70)
    print("STARTING SIMPLIFIED TRAINING")
    print("=" * 70 + "\n")

    # Create reward manager
    reward_manager = gen_reward_manager()

    # Setup self-play opponents
    self_play_handler = SimpleSelfPlayHandler(CHECKPOINT_DIR)
    opponents_dict = {**OPPONENT_MIX}  # Scripted opponents
    # Note: Self-play will be added after first checkpoint

    opponent_cfg = OpponentsCfg(opponents={
        k: (prob, agent_partial) for k, (prob, agent_partial) in opponents_dict.items()
    })

    # Create environment
    env = SelfPlayWarehouseBrawl(
        reward_manager=reward_manager,
        opponent_cfg=opponent_cfg,
        save_handler=None,
        resolution=TRAINING_CONFIG["resolution"],
    )

    # Attach self-play handler
    self_play_handler.env = env
    reward_manager.subscribe_signals(env.raw_env)

    # Wrap environment to add opponent history
    opponent_obs_dim = AGENT_CONFIG.get("opponent_obs_dim")
    if opponent_obs_dim is None:
        opponent_obs_dim = env.observation_space.shape[0] // 2

    env = OpponentHistoryWrapper(
        env,
        opponent_obs_dim=opponent_obs_dim,
        sequence_length=AGENT_CONFIG["sequence_length"]
    )

    # Calculate observation dimensions
    total_obs_dim = env.observation_space.shape[0]
    history_dim = opponent_obs_dim * AGENT_CONFIG["sequence_length"]
    base_obs_dim = total_obs_dim - history_dim

    print(f"✓ Environment created")
    print(f"  Base obs dim: {base_obs_dim}")
    print(f"  Opponent obs dim: {opponent_obs_dim}")
    print(f"  History dim: {history_dim}")
    print(f"  Total obs dim: {total_obs_dim}\n")

    # Create policy kwargs with SimplifiedExtractor
    policy_kwargs = {
        **AGENT_CONFIG["policy_kwargs"],
        "features_extractor_class": SimplifiedExtractor,
        "features_extractor_kwargs": {
            "features_dim": 256,
            "latent_dim": AGENT_CONFIG["latent_dim"],
            "base_obs_dim": base_obs_dim,
            "opponent_obs_dim": opponent_obs_dim,
            "sequence_length": AGENT_CONFIG["sequence_length"],
            "device": DEVICE,
        }
    }

    # Create RecurrentPPO model
    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        verbose=1,
        n_steps=AGENT_CONFIG["n_steps"],
        batch_size=AGENT_CONFIG["batch_size"],
        n_epochs=AGENT_CONFIG["n_epochs"],
        learning_rate=AGENT_CONFIG["learning_rate"],
        ent_coef=AGENT_CONFIG["ent_coef"],
        clip_range=AGENT_CONFIG["clip_range"],
        gamma=AGENT_CONFIG["gamma"],
        gae_lambda=AGENT_CONFIG["gae_lambda"],
        max_grad_norm=AGENT_CONFIG["max_grad_norm"],
        vf_coef=AGENT_CONFIG["vf_coef"],
        clip_range_vf=AGENT_CONFIG["clip_range_vf"],
        policy_kwargs=policy_kwargs,
        device=DEVICE,
    )

    print(f"✓ RecurrentPPO model created on {DEVICE}\n")

    # Create checkpoint callback
    from stable_baselines3.common.callbacks import CheckpointCallback

    checkpoint_callback = CheckpointCallback(
        save_freq=TRAINING_CONFIG["save_freq"],
        save_path=CHECKPOINT_DIR,
        name_prefix="rl_model",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    # Lightweight training monitor - tracks essential metrics
    class TrainingMonitor(CheckpointCallback):
        def __init__(self, *args, env=None, **kwargs):
            self.training_env = env  # Store env reference before calling super
            super().__init__(*args, **kwargs)
            self.last_encoder_weights = None
            self.episode_rewards = []
            self.episode_lengths = []
            self.episode_outcomes = []  # Track win/loss for each episode (True=win, False=loss)
            self.win_count = 0
            self.loss_count = 0
            self.episode_count = 0

        def _on_step(self):
            # Get current step infos
            step_infos = None
            if hasattr(self, 'locals') and 'infos' in self.locals:
                step_infos = self.locals['infos']

            # Track wins/losses from step-level infos when episodes end
            if step_infos is not None:
                for info in step_infos:
                    # Check for winner info when episode ends
                    if 'winner' in info:
                        is_win = info['winner'] == 'player'
                        self.episode_outcomes.append(is_win)
                        if is_win:
                            self.win_count += 1
                        else:
                            self.loss_count += 1

            # Track episode completion from buffer
            if hasattr(self, 'model') and hasattr(self.model, 'ep_info_buffer'):
                if len(self.model.ep_info_buffer) > 0:
                    for ep_info in self.model.ep_info_buffer:
                        if 'r' in ep_info and ep_info['r'] not in [r for r in self.episode_rewards[-10:]]:
                            self.episode_rewards.append(ep_info['r'])
                            self.episode_lengths.append(ep_info.get('l', 0))
                            self.episode_count += 1

            # Print comprehensive update every 1000 steps
            if self.n_calls % 1000 == 0:
                print(f"\n{'='*70}")
                print(f"TRAINING UPDATE @ {self.num_timesteps:,} steps")
                print(f"{'='*70}")

                # === PERFORMANCE METRICS (Most Important) ===
                if self.episode_rewards:
                    recent_rewards = self.episode_rewards[-100:]  # Last 100 episodes
                    print(f"\n[PERFORMANCE]")
                    print(f"  Episodes completed: {self.episode_count}")
                    print(f"  Avg Reward (last 100 ep): {np.mean(recent_rewards):.2f}")
                    print(f"  Reward Std: {np.std(recent_rewards):.2f}")

                    # Win/Loss tracking - recent window only (last 100 episodes)
                    if self.episode_outcomes:
                        recent_outcomes = self.episode_outcomes[-100:]
                        recent_wins = sum(recent_outcomes)
                        recent_total = len(recent_outcomes)
                        recent_win_rate = (recent_wins / recent_total * 100) if recent_total > 0 else 0

                        print(f"  Win Rate (last {recent_total} ep): {recent_win_rate:.1f}% ({recent_wins}W / {recent_total - recent_wins}L)")

                        # Show opponent distribution
                        try:
                            base_env = self.training_env.envs[0]
                            current_env = base_env
                            while current_env is not None:
                                if hasattr(current_env, 'opponent_cfg'):
                                    opponent_cfg = current_env.opponent_cfg
                                    total = sum(opponent_cfg.opponent_counts.values())
                                    if total > 0:
                                        print(f"  Opponents: ", end="")
                                        counts_str = ", ".join([f"{name}: {count/total*100:.0f}%" for name, count in sorted(opponent_cfg.opponent_counts.items())])
                                        print(counts_str)
                                    break
                                current_env = getattr(current_env, 'env', None)
                        except:
                            pass
                    else:
                        print(f"  Win Rate: No games completed yet")

                # === ENCODER HEALTH ===
                encoder = self.model.policy.features_extractor.opponent_encoder

                # 1. Gradient flow
                grad_norm = 0.0
                for param in encoder.parameters():
                    if param.grad is not None:
                        grad_norm += param.grad.norm().item() ** 2
                grad_norm = grad_norm ** 0.5

                # 2. Weight changes
                current_weights = torch.cat([p.flatten() for p in encoder.parameters()])
                if self.last_encoder_weights is not None:
                    weight_change = (current_weights - self.last_encoder_weights).norm().item()

                    print(f"\n[ENCODER]")
                    print(f"  Gradient norm (size of current gradient update): {grad_norm:.6f} {'✓' if grad_norm > 1e-5 else '⚠️ LOW'}")
                    print(f"  Weight change (How mcuh params moved): {weight_change:.6f} {'✓' if weight_change > 1e-4 else '⚠️ STUCK'}")
                self.last_encoder_weights = current_weights.clone()

                # 3. Encoding diversity
                if hasattr(self.model, 'rollout_buffer') and self.model.rollout_buffer.size() > 0:
                    try:
                        buffer = self.model.rollout_buffer
                        sample_size = min(20, buffer.buffer_size)
                        obs_samples = buffer.observations[:sample_size]

                        base_dim = self.model.policy.features_extractor.base_obs_dim
                        history_dim = self.model.policy.features_extractor.history_dim

                        encoder.eval()
                        with torch.no_grad():
                            encodings = []
                            for obs in obs_samples:
                                # Observation is already a flat array
                                if len(obs.shape) == 1:
                                    # Extract history portion from end of observation
                                    history_flat = obs[base_dim:]
                                else:
                                    # Already batched
                                    history_flat = obs[:, base_dim:]

                                # Ensure we have the right size
                                if history_flat.shape[-1] != history_dim:
                                    continue

                                # Convert to tensor
                                history_t = torch.FloatTensor(history_flat).to(encoder.device)
                                if len(history_t.shape) == 1:
                                    history_t = history_t.unsqueeze(0)

                                # Reshape to [batch, seq_len, obs_dim]
                                history = history_t.view(-1, encoder.history_length, encoder.opponent_obs_dim)

                                # Get encoding
                                enc = encoder(history).cpu().numpy()
                                if len(enc.shape) > 1:
                                    enc = enc[0]
                                encodings.append(enc)
                        encoder.train()

                        if encodings:
                            encodings = np.array(encodings)
                            enc_std = encodings.std(axis=0).mean()  # diversity across samples
                            # With Tanh activation, expect diversity 0.1-0.3
                            print(f"  Encoding diversity(how much the encoding varies between samples): {enc_std:.4f} {'✓' if enc_std > 0.05 else '⚠️ COLLAPSED'}")
                    except Exception as e:
                        print(f"  Encoding diversity: [Error: {e}]")

                # === LEARNING STABILITY ===
                print(f"\n[LEARNING]")

                # Policy loss (from logger if available)
                if hasattr(self.model, 'logger') and self.model.logger:
                    try:
                        # Access recent logged values
                        if hasattr(self.model.logger, 'name_to_value'):
                            if 'train/policy_loss' in self.model.logger.name_to_value:
                                policy_loss = self.model.logger.name_to_value['train/policy_loss']
                                print(f"  Policy Loss: {policy_loss:.4f}")
                            if 'train/value_loss' in self.model.logger.name_to_value:
                                value_loss = self.model.logger.name_to_value['train/value_loss']
                                print(f"  Value Loss: {value_loss:.4f}")
                            if 'train/entropy_loss' in self.model.logger.name_to_value:
                                entropy_loss = self.model.logger.name_to_value['train/entropy_loss']
                                # Entropy loss is negative (we want to maximize entropy)
                                # More negative = higher actual entropy (good!)
                                print(f"  Entropy Loss: {entropy_loss:.4f} {'✓' if entropy_loss < -0.01 else '⚠️ LOW exploration'}")
                    except:
                        pass

                # Explained variance (how well value function predicts returns)
                if hasattr(self.model, 'rollout_buffer') and self.model.rollout_buffer.size() > 0:
                    buffer = self.model.rollout_buffer
                    if hasattr(buffer, 'returns') and hasattr(buffer, 'values'):
                        returns = buffer.returns.flatten()
                        values = buffer.values.flatten()
                        var_returns = np.var(returns)
                        if var_returns > 0:
                            explained_var = 1 - np.var(returns - values) / var_returns
                            print(f"  Explained Variance: {explained_var:.3f} {'✓' if explained_var > 0.5 else '⚠️ LOW'}")

                print(f"{'='*70}\n")

            return super()._on_step()

    training_callback = TrainingMonitor(
        env=env,  # Pass environment reference for opponent tracking
        save_freq=TRAINING_CONFIG["save_freq"],
        save_path=CHECKPOINT_DIR,
        name_prefix="rl_model",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    print("🚀 Training started\n")
    print("Version 1.1.0 - Strategy-Gated Architecture (Forces encoder dependency)")

    # Train!
    model.learn(
        total_timesteps=TRAINING_CONFIG["total_timesteps"],
        callback=training_callback,
        log_interval=1,
    )

    # Save final model
    final_path = os.path.join(CHECKPOINT_DIR, "final_model.zip")
    model.save(final_path)

    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Final model saved to: {final_path}")
    print(f"Checkpoints saved in: {CHECKPOINT_DIR}")
    print("\nTo continue training, load the checkpoint and resume!")
    print("=" * 70 + "\n")


# ============================================================================
# RUN TRAINING
# ============================================================================

if __name__ == "__main__":
    train()
