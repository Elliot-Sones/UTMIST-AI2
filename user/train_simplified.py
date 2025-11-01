"""
UTMIST AI² - RecurrentPPO (LSTM-only) Training
================================================

Architecture: Pure RecurrentPPO with MlpLstmPolicy (no custom encoder).
- No opponent history wrapper
- No feature extractor customization
- LSTM handles temporal credit assignment over raw observations

Run:
    python user/train_simplified.py
"""

# ============================================================================
# IMPORTS
# ============================================================================

import os
import sys
import random
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # Apple Silicon support
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from pathlib import Path
from typing import Any, Callable, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from functools import partial

from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environment.agent import *

# ============================================================================
# ADVANCED LSTM POLICY WITH ENHANCED ARCHITECTURE
# ============================================================================

class EnhancedRecurrentActorCriticPolicy(RecurrentActorCriticPolicy):
    """
    Enhanced RecurrentPPO policy with advanced LSTM architecture:
    - Layer normalization for stable training
    - Dropout for regularization
    - Optimized hidden dimensions
    - Better gradient flow
    """

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        lstm_hidden_size=384,
        n_lstm_layers=3,
        net_arch=None,
        activation_fn=nn.ReLU,
        shared_lstm=False,
        enable_critic_lstm=True,
        share_features_extractor=True,
        dropout=0.1,
        **kwargs
    ):
        # Store our custom dropout parameter
        self.dropout_rate = dropout

        # Override the architecture parameters with our enhanced settings
        enhanced_net_arch = {
            "pi": [lstm_hidden_size, lstm_hidden_size],
            "vf": [lstm_hidden_size, lstm_hidden_size],
        }

        # Call parent with enhanced parameters
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            lstm_hidden_size=lstm_hidden_size,
            n_lstm_layers=n_lstm_layers,
            net_arch=enhanced_net_arch,
            activation_fn=activation_fn,
            shared_lstm=shared_lstm,
            enable_critic_lstm=enable_critic_lstm,
            share_features_extractor=share_features_extractor,
            **kwargs
        )

    def _build_mlp_extractor(self) -> None:
        """Override to add layer normalization and dropout"""
        super()._build_mlp_extractor()

        # Add layer normalization to MLP layers for stable training
        self.mlp_extractor.policy_net = nn.Sequential(
            nn.Linear(self.features_dim, self.net_arch["pi"][0]),
            nn.LayerNorm(self.net_arch["pi"][0]),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.net_arch["pi"][0], self.net_arch["pi"][1]),
            nn.LayerNorm(self.net_arch["pi"][1]),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
        )

        self.mlp_extractor.value_net = nn.Sequential(
            nn.Linear(self.features_dim, self.net_arch["vf"][0]),
            nn.LayerNorm(self.net_arch["vf"][0]),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.net_arch["vf"][0], self.net_arch["vf"][1]),
            nn.LayerNorm(self.net_arch["vf"][1]),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
        )

    def _build_lstm(self) -> None:
        """Override to add layer normalization to LSTM"""
        super()._build_lstm()

        # Add layer normalization after LSTM for better gradient flow
        if hasattr(self, 'lstm_actor'):
            self.lstm_actor_norm = nn.LayerNorm(self.lstm_hidden_size)
        if hasattr(self, 'lstm_critic'):
            self.lstm_critic_norm = nn.LayerNorm(self.lstm_hidden_size)

    def forward(self, obs, lstm_states, episode_starts, deterministic=False):
        """Override forward to apply layer norm after LSTM"""
        features = self.extract_features(obs, lstm_states[:, :self.lstm_hidden_size], episode_starts)
        latent_pi, latent_vf = self.mlp_extractor(features)
        latent_pi = self._apply_lstm_norm(latent_pi, "actor")
        latent_vf = self._apply_lstm_norm(latent_vf, "critic")

        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))

        return actions, values, log_prob, lstm_states

    def _apply_lstm_norm(self, latent, net_type):
        """Apply layer normalization after LSTM processing"""
        if net_type == "actor" and hasattr(self, 'lstm_actor_norm'):
            return self.lstm_actor_norm(latent)
        elif net_type == "critic" and hasattr(self, 'lstm_critic_norm'):
            return self.lstm_critic_norm(latent)
        return latent


# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

GLOBAL_SEED = int(os.environ.get("UTMIST_RL_SEED", "42"))


def seed_everything(seed: int) -> None:
    """Seed every library we depend on for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def linear_schedule(start: float, end: float) -> Callable[[float], float]:
    """Linear schedule compatible with SB3 progress_remaining callback."""
    def schedule(progress_remaining: float) -> float:
        return end + (start - end) * progress_remaining
    return schedule


def cosine_schedule(start: float, end: float, warmup_fraction: float = 0.1) -> Callable[[float], float]:
    """Cosine annealing schedule with warmup for better convergence."""
    import math

    def schedule(progress_remaining: float) -> float:
        progress = 1.0 - progress_remaining

        if progress < warmup_fraction:
            # Linear warmup
            return start + (end - start) * (progress / warmup_fraction)
        else:
            # Cosine annealing
            cosine_progress = (progress - warmup_fraction) / (1.0 - warmup_fraction)
            return end + 0.5 * (start - end) * (1 + math.cos(math.pi * cosine_progress))

    return schedule


def adaptive_entropy_schedule(initial_ent_coef: float, final_ent_coef: float,
                             exploration_phase: float = 0.3) -> Callable[[float], float]:
    """Adaptive entropy schedule: high exploration early, then anneal."""
    def schedule(progress_remaining: float) -> float:
        progress = 1.0 - progress_remaining

        if progress < exploration_phase:
            # High entropy for exploration
            return initial_ent_coef
        else:
            # Linear anneal to final value
            anneal_progress = (progress - exploration_phase) / (1.0 - exploration_phase)
            return initial_ent_coef + (final_ent_coef - initial_ent_coef) * anneal_progress

    return schedule


seed_everything(GLOBAL_SEED)

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

# (No encoder or observation wrapper required for LSTM-only setup)


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

# Where to save checkpoints
CHECKPOINT_DIR = "checkpoints/simplified_training"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
TENSORBOARD_DIR = Path(CHECKPOINT_DIR) / "tb_logs"
TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)

# Enhanced Agent hyperparameters - OPTIMIZED FOR STATE-OF-THE-ART PERFORMANCE
AGENT_CONFIG = {
    # Enhanced LSTM policy with advanced architecture
    "policy_class": EnhancedRecurrentActorCriticPolicy,  # Use our enhanced policy
    "lstm_hidden_size": 384,     # Increased capacity for better temporal modeling
    "n_lstm_layers": 3,          # Deeper LSTM for complex pattern recognition
    "dropout": 0.1,              # Regularization to prevent overfitting

    # Optimized PPO training parameters
    "n_steps": 4096,             # Longer rollouts for better temporal credit assignment
    "batch_size": 512,           # Larger batches for more stable gradient estimates
    "n_epochs": 6,               # Slightly more epochs for better sample utilization
    "learning_rate": 2.5e-4,     # Slightly lower for stability with larger architecture
    "min_learning_rate": 1e-5,   # More aggressive annealing
    "ent_coef": 0.005,           # Higher entropy bonus for better exploration early on
    "ent_coef_final": 0.001,     # Anneal entropy coefficient
    "clip_range": 0.25,          # Slightly higher initial clip for more aggressive updates
    "clip_range_final": 0.02,    # More conservative final clip range
    "gamma": 0.995,              # Slightly higher discount for longer-term planning
    "gae_lambda": 0.98,          # Higher GAE lambda for better bias-variance tradeoff
    "max_grad_norm": 0.75,       # Slightly higher grad clipping for larger model
    "vf_coef": 0.6,              # Slightly higher value function weight
    "clip_range_vf": 5.0,        # Tighter value function clipping
    "use_sde": True,             # State-dependent exploration
    "sde_sample_freq": 8,        # Less frequent SDE updates for stability
    "target_kl": 0.02,           # More conservative KL target
}

# Enhanced Training settings - AGGRESSIVE TRAINING FOR STATE-OF-THE-ART RESULTS
TRAINING_CONFIG = {
    "total_timesteps": 5_000_000,  # Extended training for better convergence
    "save_freq": 25_000,           # More frequent checkpoints for better self-play
    "eval_freq": 50_000,           # Evaluate every 50k steps
    "eval_games": 5,               # More games per evaluation for stable metrics
    "resolution": CameraResolution.LOW,
}

CURRICULUM_CONFIG = {
    "target_win_rate": 0.75,   # Aim for 75% win rate before down-weighting
    "min_probability": 0.05,   # Never let an opponent vanish from the curriculum
    "adaptation_strength": 0.6,
    "self_play_weight": 0.2,
}

EVAL_TO_TRAIN_KEY = {
    "Constant": "constant_agent",
    "Based": "based_agent",
    "Random": "random_agent",
    "Aggressive": "aggressive_clockwork",
    "Defensive": "defensive_clockwork",
    "Hit&Run": "hitrun_clockwork",
    "Aerial": "aerial_clockwork",
    "SpecialSpam": "special_clockwork",
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
print(f"Architecture: LSTM-only RecurrentPPO (MlpLstmPolicy)")
print(f"Training steps: {TRAINING_CONFIG['total_timesteps']:,}")
print(f"Global seed: {GLOBAL_SEED}")
print(f"Checkpoint dir: {CHECKPOINT_DIR}")
print(f"Opponent diversity: {len(OPPONENT_MIX)} distinct agent types")
for name, (prob, _) in OPPONENT_MIX.items():
    print(f"  - {name}: {prob*100:.0f}%")
print("=" * 70)


# ============================================================================
# REWARD FUNCTIONS (Same as before - these work great!)
# ============================================================================


def _get_diag_stats(env: WarehouseBrawl) -> dict:
    if not hasattr(env, '_diag_stats'):
        env._diag_stats = {
            'damage_dealt': 0.0,
            'damage_taken': 0.0,
            'zone_time': 0.0,
            'sparsity_penalty': 0.0,
            'reward': 0.0,
        }
    return env._diag_stats

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

    stats = _get_diag_stats(env)
    stats['damage_dealt'] += delta_dealt
    stats['damage_taken'] += delta_taken

    return (delta_dealt - delta_taken) / 140


def danger_zone_reward(env: WarehouseBrawl, zone_height: float = 4.2) -> float:
    """Penalize being too high (about to get knocked out)"""
    player = env.objects["player"]
    if player.body.position.y >= zone_height:
        stats = _get_diag_stats(env)
        stats['zone_time'] += env.dt
        return -1.0 * env.dt
    return 0.0


def on_win_reward(env: WarehouseBrawl, agent: str) -> float:
    """Strong reward for winning - dominates other rewards without exploding value function
    Winning needs to be very valuable to force opponent-specific strategies."""
    return 30.0 if agent == 'player' else -30.0  # Reduced from 100 to prevent value explosion


def on_knockout_reward(env: WarehouseBrawl, agent: str) -> float:
    """Reward knocking out opponent, penalize getting knocked out"""
    return 5.0 if agent == 'opponent' else -5.0  # Reduced from 10


def action_sparsity_reward(env: WarehouseBrawl, max_active: int = 2, penalty_per_key: float = 0.005) -> float:
    """Small per-step penalty when too many keys are pressed simultaneously."""
    action = getattr(env, "cur_action", {}).get(0)
    if action is None:
        return 0.0

    pressed = float((action > 0.5).sum())
    excess = max(0.0, pressed - max_active)
    penalty = -penalty_per_key * excess

    if excess > 0.0:
        stats = _get_diag_stats(env)
        stats['sparsity_penalty'] += penalty

    return penalty


def position_control_reward(env: WarehouseBrawl) -> float:
    """Reward for maintaining advantageous positions and punishing disadvantageous ones."""
    player = env.objects["player"]
    opponent = env.objects["opponent"]

    # Horizontal position advantage (prefer center control)
    player_x = abs(player.body.position.x)
    opponent_x = abs(opponent.body.position.x)

    # Vertical position advantage (higher is generally better, but not too high)
    player_y = player.body.position.y
    opponent_y = opponent.body.position.y

    # Distance-based advantage
    distance = abs(player.body.position.x - opponent.body.position.x)

    # Reward being closer horizontally (better positioning)
    distance_reward = max(0.0, (10.0 - distance) / 100.0)  # Small reward for proximity

    # Reward for height advantage (but penalize being too high)
    height_advantage = (player_y - opponent_y) / 10.0
    height_penalty = -0.01 if player_y > 5.0 else 0.0  # Small penalty for being too high

    return distance_reward + height_advantage + height_penalty


def combo_incentive_reward(env: WarehouseBrawl) -> float:
    """Reward building and maintaining combo pressure."""
    player = env.objects["player"]
    opponent = env.objects["opponent"]

    # Reward for consecutive hits (combo building)
    if hasattr(opponent, 'hit_stun_frames') and opponent.hit_stun_frames > 0:
        combo_reward = min(0.1, opponent.hit_stun_frames / 100.0)  # Small reward for combos
        return combo_reward

    return 0.0


def timing_reward(env: WarehouseBrawl) -> float:
    """Reward for good timing (attacks during opponent recovery, movement during stun)."""
    player = env.objects["player"]
    opponent = env.objects["opponent"]

    # Reward attacking when opponent is in hit stun
    if hasattr(opponent, 'hit_stun_frames') and opponent.hit_stun_frames > 0:
        action = getattr(env, "cur_action", {}).get(0)
        if action is not None:
            # Check if attacking (j or k pressed)
            attacking = action[7] > 0.5 or action[8] > 0.5  # j and k indices
            if attacking:
                return 0.05  # Reward for timing attacks well

    return 0.0


def momentum_reward(env: WarehouseBrawl) -> float:
    """Reward maintaining and using momentum effectively."""
    player = env.objects["player"]

    # Reward horizontal momentum (speed in x direction)
    momentum_x = abs(player.body.velocity.x)
    momentum_reward = min(0.02, momentum_x / 200.0)  # Small reward for maintaining speed

    # Penalize being stuck/not moving
    if momentum_x < 0.1:
        momentum_reward -= 0.005

    return momentum_reward


def gen_reward_manager():
    """Create enhanced reward manager with sophisticated learning signals

    Enhanced reward structure for state-of-the-art RecurrentPPO:
    - Win-focused rewards for opponent-specific strategies
    - Position and momentum control for better game understanding
    - Combo and timing incentives for aggressive play
    - Balanced weights to prevent reward hacking
    """
    reward_functions = {
        'danger_zone': RewTerm(func=danger_zone_reward, weight=0.15),        # Reduced weight
        'damage_interaction': RewTerm(func=damage_interaction_reward, weight=0.6),  # Core damage reward
        'position_control': RewTerm(func=position_control_reward, weight=0.1),     # New: position advantage
        'combo_incentive': RewTerm(func=combo_incentive_reward, weight=0.1),       # New: combo building
        'timing': RewTerm(func=timing_reward, weight=0.15),                       # New: attack timing
        'momentum': RewTerm(func=momentum_reward, weight=0.05),                   # New: movement efficiency
        'action_sparsity': RewTerm(func=action_sparsity_reward, weight=0.8),      # Reduced weight
    }
    signal_subscriptions = {
        'on_win': ('win_signal', RewTerm(func=on_win_reward, weight=1.0)),        # 30 points - primary objective
        'on_knockout': ('knockout_signal', RewTerm(func=on_knockout_reward, weight=1.0)),  # 5 points
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
    opponents_dict["self_play"] = (CURRICULUM_CONFIG["self_play_weight"], self_play_handler)

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
    env.action_space.seed(GLOBAL_SEED)
    env.reset(seed=GLOBAL_SEED)

    # Attach self-play handler
    self_play_handler.env = env
    reward_manager.subscribe_signals(env.raw_env)
    opponent_cfg.base_probabilities = {
        name: (value if isinstance(value, float) else value[0])
        for name, value in opponent_cfg.opponents.items()
    }

    print(f"✓ Environment created")
    print(f"  Observation dim: {env.observation_space.shape[0]}")
    print(f"  LSTM handles temporal patterns over raw obs\n")

    # Use enhanced policy class (fallback to default MlpLstmPolicy if not specified)
    policy_class = AGENT_CONFIG.get("policy_class")
    if policy_class is None:
        # Import default policy for fallback
        from sb3_contrib.common.recurrent.policies import MlpLstmPolicy
        policy_class = MlpLstmPolicy

    # Build advanced schedules for better convergence
    lr_schedule = cosine_schedule(
        AGENT_CONFIG["learning_rate"],
        AGENT_CONFIG["min_learning_rate"],
        warmup_fraction=0.05  # 5% warmup period
    )
    clip_schedule = linear_schedule(
        AGENT_CONFIG["clip_range"],
        AGENT_CONFIG["clip_range_final"],
    )
    ent_schedule = adaptive_entropy_schedule(
        AGENT_CONFIG["ent_coef"],
        AGENT_CONFIG.get("ent_coef_final", AGENT_CONFIG["ent_coef"] * 0.2),
        exploration_phase=0.3  # 30% exploration phase
    )

    # Create Enhanced RecurrentPPO model with advanced architecture
    # For our custom policy class, we need to pass LSTM parameters via policy_kwargs
    # SB3 will extract these and pass them as direct arguments to the policy constructor
    if policy_class == EnhancedRecurrentActorCriticPolicy:
        policy_kwargs = {
            "lstm_hidden_size": AGENT_CONFIG["lstm_hidden_size"],
            "n_lstm_layers": AGENT_CONFIG["n_lstm_layers"],
            "dropout": AGENT_CONFIG["dropout"],
            "net_arch": {
                "pi": [AGENT_CONFIG["lstm_hidden_size"], AGENT_CONFIG["lstm_hidden_size"]],
                "vf": [AGENT_CONFIG["lstm_hidden_size"], AGENT_CONFIG["lstm_hidden_size"]],
            },
            "activation_fn": nn.ReLU,
            "shared_lstm": False,
            "enable_critic_lstm": True,
            "share_features_extractor": True,
        }
    else:
        # For default policy, use standard policy_kwargs
        policy_kwargs = {}

    model = RecurrentPPO(
        policy_class,  # Use our enhanced policy class
        env,
        verbose=1,
        n_steps=AGENT_CONFIG["n_steps"],
        batch_size=AGENT_CONFIG["batch_size"],
        n_epochs=AGENT_CONFIG["n_epochs"],
        learning_rate=lr_schedule,
        ent_coef=ent_schedule,  # Use adaptive entropy schedule
        clip_range=clip_schedule,
        gamma=AGENT_CONFIG["gamma"],
        gae_lambda=AGENT_CONFIG["gae_lambda"],
        max_grad_norm=AGENT_CONFIG["max_grad_norm"],
        vf_coef=AGENT_CONFIG["vf_coef"],
        clip_range_vf=AGENT_CONFIG["clip_range_vf"],
        target_kl=AGENT_CONFIG["target_kl"],
        use_sde=AGENT_CONFIG["use_sde"],
        sde_sample_freq=AGENT_CONFIG["sde_sample_freq"],
        policy_kwargs=policy_kwargs,  # Pass LSTM parameters here
        device=DEVICE,
        tensorboard_log=str(TENSORBOARD_DIR),
        seed=GLOBAL_SEED,
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
        def __init__(self, *args, env=None, eval_freq=10000, eval_games=10, **kwargs):
            self.env_ref = env  # Store env reference before calling super
            super().__init__(*args, **kwargs)
            self.episode_rewards = []
            self.episode_lengths = []
            self.episode_count = 0
            self.eval_freq = eval_freq  # Evaluate every N steps
            self.eval_games = eval_games  # Games per opponent during eval
            self.last_eval_step = 0
            self._seen_ep_ids = deque()
            self._seen_ep_id_set = set()
            self.last_episode_summary = None
            self.last_opponent_probs = None

        def evaluate_against_all_opponents(self):
            """Run evaluation games against all opponent types and return win rates"""
            print(f"\n{'='*70}")
            print(f"RUNNING EVALUATION @ {self.num_timesteps:,} steps")
            print(f"{'='*70}")

            # Create agent wrapper from current model
            temp_model_path = os.path.join(CHECKPOINT_DIR, f"temp_eval_{self.num_timesteps}.zip")
            self.model.save(temp_model_path)
            eval_agent = RecurrentPPOAgent(file_path=temp_model_path)

            # Define all opponent types (same as training)
            opponent_types = {
                "Constant": partial(ConstantAgent),
                "Based": partial(BasedAgent),
                "Random": partial(RandomAgent),
                "Aggressive": partial(ClockworkAgent, action_sheet=AGGRESSIVE_PATTERN),
                "Defensive": partial(ClockworkAgent, action_sheet=DEFENSIVE_PATTERN),
                "Hit&Run": partial(ClockworkAgent, action_sheet=HIT_AND_RUN_PATTERN),
                "Aerial": partial(ClockworkAgent, action_sheet=AERIAL_PATTERN),
                "SpecialSpam": partial(ClockworkAgent, action_sheet=SPECIAL_SPAM_PATTERN),
            }

            results = {}
            total_wins = 0
            total_games = 0

            for opp_name, opp_factory in opponent_types.items():
                wins = 0
                for _ in range(self.eval_games):
                    match_stats = run_match(
                        agent_1=eval_agent,
                        agent_2=opp_factory,
                        max_timesteps=30*60,
                        video_path=None,
                        resolution=TRAINING_CONFIG["resolution"],
                        train_mode=False
                    )
                    if match_stats.player1_result == Result.WIN:
                        wins += 1

                win_rate = (wins / self.eval_games * 100)
                results[opp_name] = {'wins': wins, 'total': self.eval_games, 'win_rate': win_rate}
                total_wins += wins
                total_games += self.eval_games

            # Clean up temp file
            if os.path.exists(temp_model_path):
                os.remove(temp_model_path)

            # Print results table
            print(f"\n{'Opponent':<15} {'Win Rate':<12} {'Record':<10}")
            print("-" * 40)
            for opp_name, stats in results.items():
                wr = stats['win_rate']
                record = f"{stats['wins']}/{stats['total']}"
                if wr >= 70:
                    wr_str = f"{wr:>5.1f}% ✓"
                elif wr >= 50:
                    wr_str = f"{wr:>5.1f}%"
                else:
                    wr_str = f"{wr:>5.1f}% ✗"
                print(f"{opp_name:<15} {wr_str:<12} {record:<10}")

            overall_wr = (total_wins / total_games * 100) if total_games > 0 else 0
            print("-" * 40)
            print(f"{'OVERALL':<15} {overall_wr:>5.1f}%     {total_wins}/{total_games}")
            print(f"{'='*70}\n")

            if hasattr(self.model, "logger") and self.model.logger:
                for opp_name, stats in results.items():
                    self.model.logger.record(
                        f"eval/win_rate/{opp_name}",
                        stats['win_rate'] / 100.0,
                        exclude=("stdout",),
                    )
                self.model.logger.record(
                    "eval/overall_win_rate",
                    overall_wr / 100.0,
                    exclude=("stdout",),
                )

            self._update_curriculum(results)

            if hasattr(self.model, "logger") and self.model.logger:
                self.model.logger.dump(self.num_timesteps)

            return results, overall_wr

        def _update_curriculum(self, eval_results: dict) -> None:
            """Dynamically adjust opponent sampling based on evaluation results."""
            if self.env_ref is None:
                return

            cfg = getattr(self.env_ref, "opponent_cfg", None)
            if cfg is None or not hasattr(cfg, "opponents"):
                return

            base_probs = getattr(cfg, "base_probabilities", None)
            if base_probs is None:
                base_probs = {
                    name: value[0] if isinstance(value, tuple) else value
                    for name, value in cfg.opponents.items()
                }
                cfg.base_probabilities = base_probs

            prev_probs = {
                name: value[0] if isinstance(value, tuple) else value
                for name, value in cfg.opponents.items()
            }

            new_opponents: Dict[str, Tuple[float, Any]] = {}
            for name, value in cfg.opponents.items():
                if isinstance(value, tuple):
                    current_prob, agent_factory = value
                else:
                    current_prob, agent_factory = value, None
                base_prob = base_probs.get(name, current_prob)

                adjustment = 0.0
                for eval_name, stats in eval_results.items():
                    mapped_name = EVAL_TO_TRAIN_KEY.get(eval_name)
                    if mapped_name == name:
                        win_rate = stats['win_rate'] / 100.0
                        deficit = max(0.0, CURRICULUM_CONFIG["target_win_rate"] - win_rate)
                        adjustment = deficit * CURRICULUM_CONFIG["adaptation_strength"]
                        break

                if name == "self_play":
                    adjusted_prob = max(base_prob, CURRICULUM_CONFIG["min_probability"])
                else:
                    adjusted_prob = max(CURRICULUM_CONFIG["min_probability"], base_prob + adjustment)

                new_opponents[name] = (adjusted_prob, agent_factory)

            cfg.opponents = new_opponents
            cfg.validate_probabilities()
            cfg.base_probabilities = {
                name: value[0] if isinstance(value, tuple) else value
                for name, value in cfg.opponents.items()
            }

            updated = any(
                abs(cfg.opponents[name][0] - prev_probs.get(name, 0.0)) > 1e-5
                for name in cfg.opponents
            )

            if updated:
                print("[CURRICULUM] Opponent probabilities (normalized):")
                for name, (prob, _) in cfg.opponents.items():
                    print(f"  - {name}: {prob:.3f}")
                self.last_opponent_probs = {name: prob for name, (prob, _) in cfg.opponents.items()}

            if hasattr(self.model, "logger") and self.model.logger:
                for name, (prob, _) in cfg.opponents.items():
                    self.model.logger.record(
                        f"curriculum/opponent_prob/{name}",
                        prob,
                        exclude=("stdout",),
                    )

        def _on_step(self):
            # Track episode completion from buffer
            if hasattr(self, 'model') and hasattr(self.model, 'ep_info_buffer'):
                buffer = list(self.model.ep_info_buffer)
                for ep_info in buffer:
                    ep_id = id(ep_info)
                    if ep_id in self._seen_ep_id_set:
                        continue
                    self._seen_ep_ids.append(ep_id)
                    self._seen_ep_id_set.add(ep_id)
                    if len(self._seen_ep_ids) > 200:
                        old_id = self._seen_ep_ids.popleft()
                        self._seen_ep_id_set.discard(old_id)

                    reward = ep_info.get('r')
                    length = ep_info.get('l', 0)
                    if reward is not None:
                        self.episode_rewards.append(reward)
                        self.episode_lengths.append(length)
                        self.episode_count += 1

                    if self.env_ref is not None and hasattr(self.env_ref, 'raw_env'):
                        summary = getattr(self.env_ref.raw_env, '_diag_last_episode', None)
                        if summary is not None:
                            self.last_episode_summary = summary

            # Run evaluation every eval_freq steps
            if self.num_timesteps >= self.last_eval_step + self.eval_freq and self.num_timesteps > 0:
                self.last_eval_step = self.num_timesteps
                self.evaluate_against_all_opponents()

            # Print comprehensive update every 1000 steps
            if self.n_calls % 1000 == 0:
                print(f"\n{'='*70}")
                print(f"TRAINING UPDATE @ {self.num_timesteps:,} steps")
                print(f"{'='*70}")

                # === PERFORMANCE METRICS ===
                if self.episode_rewards:
                    recent_rewards = self.episode_rewards[-100:]  # Last 100 episodes
                    print(f"\n[PERFORMANCE]")
                    print(f"  Episodes completed: {self.episode_count}")
                    print(f"  Avg Reward (last 100 ep): {np.mean(recent_rewards):.2f}")
                    print(f"  Reward Std: {np.std(recent_rewards):.2f}")

                if self.last_episode_summary:
                    stats = self.last_episode_summary
                    print(f"\n[EPISODE DIAGNOSTICS]")
                    print(f"  Damage dealt: {stats.get('damage_dealt', 0):.1f}")
                    print(f"  Damage taken: {stats.get('damage_taken', 0):.1f}")
                    print(f"  Time in danger zone (s): {stats.get('zone_time', 0):.1f}")
                    print(f"  Sparsity penalty: {stats.get('sparsity_penalty', 0):.2f}")
                    print(f"  Episode reward: {stats.get('reward', 0):.2f}")

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
        eval_freq=TRAINING_CONFIG["eval_freq"],  # Use config-based evaluation frequency
        eval_games=TRAINING_CONFIG["eval_games"],  # Use config-based games per evaluation
        save_freq=TRAINING_CONFIG["save_freq"],
        save_path=CHECKPOINT_DIR,
        name_prefix="rl_model",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    print("🚀 Training started\n")
    print("Version 4.0 - ENHANCED STATE-OF-THE-ART RecurrentPPO-Grok")
    print("="*70)
    print("  - Advanced LSTM: 3-layer, 384-hidden with LayerNorm + Dropout")
    print("  - Enhanced rewards: Position control, combos, timing, momentum")
    print("  - Adaptive schedules: Cosine LR, entropy annealing")
    print("  - Aggressive curriculum: Extended training with frequent eval")
    print("  - Optimized PPO: Better credit assignment and stability")
    print("="*70 + "\n")

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
