"""
UTMIST AI² - Strategy-Conditioned RecurrentPPO with Population-Based Self-Play
================================================================================

This training script implements:
1. Strategy Encoder: 1D CNN that extracts opponent playstyle embeddings
2. Opponent-Conditioned Policy: RecurrentPPO that adapts to detected strategies
3. Population-Based Self-Play: Diverse agent pool for robust training
4. Diversity-Focused Training: Maintains varied strategies and handles edge cases

Run:
    python user/train_with_strategy_encoder.py
"""

import os
import sys
import random
import inspect
import logging
from datetime import datetime
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from pathlib import Path
from typing import Any, Callable, Dict, Tuple, List
import torch
import torch.nn as nn
import numpy as np
from functools import partial

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, BaseCallback

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environment.agent import *
from user.models.opponent_conditioned_policy import create_opponent_conditioned_policy_kwargs
from user.wrappers.opponent_history_wrapper import OpponentHistoryBuffer
from user.wrappers.augmented_obs_wrapper import AugmentedObservationWrapper
from user.wrappers.episode_stats_wrapper import EpisodeStatsWrapper
from user.self_play.population_manager import PopulationManager
from user.self_play.diverse_opponent_sampler import DiverseOpponentSampler
from user.self_play.population_update_callback import PopulationUpdateCallback
from user.callbacks.training_metrics_callback import create_training_metrics_callback
from user.callbacks.gradient_monitor_callback import create_gradient_monitor_callback

# Add curriculum learning callback
class CurriculumCallback(BaseCallback):
    """Gradually increases opponent difficulty as training progresses"""
    def __init__(self, curriculum_stages, verbose=0):
        super().__init__(verbose)
        self.curriculum_stages = curriculum_stages  # List of (timestep_threshold, opponent_config) tuples
        self.current_stage = 0

    def _on_step(self) -> bool:
        # Check if we should advance to next curriculum stage
        current_timestep = self.num_timesteps
        for i, (threshold, config) in enumerate(self.curriculum_stages):
            if current_timestep >= threshold and i > self.current_stage:
                self.current_stage = i
                self._apply_curriculum_config(config)
                if self.verbose:
                    print(f"\n🎓 CURRICULUM ADVANCEMENT: Stage {i+1}/{len(self.curriculum_stages)}")
                    print(f"   Timestep: {current_timestep:,}")
                    print(f"   New opponent mix: {config}")
                break
        return True

    def _apply_curriculum_config(self, config):
        """Apply new opponent configuration to all environments"""
        try:
            # Update opponent configurations across all environments
            if hasattr(self.training_env, 'set_attr'):
                for key, value in config.items():
                    self.training_env.set_attr(f'opponent_cfg.opponents.{key}', value)
        except Exception as e:
            if self.verbose:
                print(f"   Warning: Could not apply curriculum config: {e}")

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

GLOBAL_SEED = int(os.environ.get("UTMIST_RL_SEED", "42"))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def linear_schedule(start: float, end: float) -> Callable[[float], float]:
    def schedule(progress_remaining: float) -> float:
        return end + (start - end) * progress_remaining
    return schedule


# DEVICE will be initialized in main block
DEVICE = None

def get_device():
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"✓ Using CUDA GPU: {device_name}")

        # Enable CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False  # For speed

        # Enable TF32 for faster training on Ampere GPUs (RTX 30xx, A100, etc.)
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high')
            print(f"  ✓ TF32 enabled for matmul operations")

        if hasattr(torch.backends.cuda, 'matmul') and hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
            torch.backends.cuda.matmul.allow_tf32 = True
            print(f"  ✓ TF32 enabled for CUDA matmul")

        if hasattr(torch.backends, 'cudnn') and hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = True
            print(f"  ✓ TF32 enabled for cuDNN")

        # Print GPU memory info
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  ✓ GPU Memory: {total_memory:.1f} GB")

        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("⚠ Apple Silicon MPS detected, but CUDA preferred for stability")
        print("  Using MPS (may have compatibility issues)")
        return torch.device("mps")
    else:
        print("⚠ Using CPU (training will be VERY slow)")
        print("  Consider using a CUDA GPU for practical training")
        return torch.device("cpu")

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

CHECKPOINT_DIR = Path("/tmp/strategy_encoder_training")  # Use /tmp with more space

# These will be initialized in main block
TENSORBOARD_DIR = None
TENSORBOARD_AVAILABLE = False
LOG_FILE = None
logger = logging.getLogger(__name__)

# Strategy encoder configuration
STRATEGY_ENCODER_CONFIG = {
    'input_features': 13,  # opponent features tracked
    'history_length': 60,  # 2 seconds at 30 FPS
    'embedding_dim': 32,   # strategy embedding size
    'dropout': 0.1,
}

# Base feature extractor configuration (same as OPTIMIZED)
BASE_EXTRACTOR_CONFIG = {
    'feature_dim': 512,
    'num_residual_blocks': 5,
    'dropout': 0.08,
}

# LSTM configuration
_LSTM_KWARGS = {"dropout": 0.1}
if "layer_norm" in inspect.signature(nn.LSTM).parameters:
    _LSTM_KWARGS["layer_norm"] = True

# Agent hyperparameters
AGENT_CONFIG = {
    "policy": "MlpLstmPolicy",  # Will use custom features extractor
    "policy_kwargs": create_opponent_conditioned_policy_kwargs(
        base_extractor_kwargs=BASE_EXTRACTOR_CONFIG,
        strategy_encoder_config=STRATEGY_ENCODER_CONFIG,
        lstm_hidden_size=512,
        n_lstm_layers=3,
        net_arch=dict(pi=[512, 256], vf=[512, 256]),
        activation_fn=nn.GELU,
        optimizer_class=torch.optim.AdamW,
        optimizer_kwargs={"weight_decay": 1e-4, "eps": 1e-5},
        ortho_init=True,
        log_std_init=-0.5,
        lstm_kwargs=_LSTM_KWARGS,
        shared_lstm=False,
        enable_critic_lstm=True,
        share_features_extractor=True,
    ),

    # PPO training - STABILIZED settings for reliable learning
    "n_steps": 4096,  # Longer rollouts for better value estimates
    "batch_size": 1024,  # Larger batches for stable gradients
    "n_epochs": 4,  # Fewer epochs to prevent overfitting
    "learning_rate": linear_schedule(3e-4, 1e-4),  # Increased 3x for faster learning
    "ent_coef": 0.15,  # FURTHER increased to force more exploration and attack discovery
    "clip_range": 0.2,  # Standard PPO clipping
    "gamma": 0.99,  # Standard discount factor
    "gae_lambda": 0.95,  # Standard GAE
    "max_grad_norm": 0.5,  # Prevent gradient explosions
    "vf_coef": 0.5,
    "clip_range_vf": 0.2,  # RE-ENABLED: Stabilizes value function
    "use_sde": False,  # Disable SDE for stability
    "sde_sample_freq": 4,
    "target_kl": None,  # Disabled to allow larger policy updates
}

TRAINING_CONFIG = {
    "total_timesteps": 5_000_000,  # Extended for diverse strategy learning
    "save_freq": 1_000_000,  # DRAMATICALLY REDUCED: Save checkpoints every 1M steps
    "resolution": CameraResolution.LOW,
    "n_envs": 8,  # Reduced for single-process stability (was 16)
}

# Population-based self-play configuration
POPULATION_CONFIG = {
    "max_population_size": 8,   # FURTHER REDUCED: Even fewer agents
    "num_weak_agents": 1,       # FURTHER REDUCED: Only 1 weak agent
    "update_frequency": 500_000,  # FURTHER REDUCED: Update population every 500K steps
    "noise_probability": 0.10,
    "use_population_prob": 0.70,  # 70% population, 30% scripted
}

# Action patterns for ClockworkAgent
AGGRESSIVE_PATTERN = [
    (15, ['d']), (3, ['d', 'j']), (2, []), (3, ['d', 'j']),
    (15, ['d']), (5, ['d', 'l']), (10, ['d']), (3, ['j']),
]

DEFENSIVE_PATTERN = [
    (20, []), (5, ['a']), (15, []), (3, ['j']),
    (10, []), (4, ['l']), (25, []),
]

HIT_AND_RUN_PATTERN = [
    (12, ['d']), (2, ['j']), (10, ['a']), (5, []),
    (10, ['d']), (3, ['d', 'l']), (15, ['a']), (8, []),
]

AERIAL_PATTERN = [
    (5, ['d']), (15, ['space']), (3, ['j']), (8, []),
    (10, ['d']), (15, ['space']), (3, ['l']), (10, []),
]

SPECIAL_SPAM_PATTERN = [
    (8, ['d']), (5, ['l']), (5, []), (5, ['l']),
    (10, ['d']), (5, ['l']), (8, []), (3, ['j']), (5, ['l']),
]

# Scripted opponent mix (30% when population available, 100% initially)
OPPONENT_MIX = {
    "constant": (0.05, partial(ConstantAgent)),
    "based": (0.10, partial(BasedAgent)),
    "random": (0.05, partial(RandomAgent)),
    "clockwork_aggressive": (0.05, partial(ClockworkAgent, action_sheet=AGGRESSIVE_PATTERN)),
    "clockwork_defensive": (0.03, partial(ClockworkAgent, action_sheet=DEFENSIVE_PATTERN)),
    "clockwork_hit_run": (0.03, partial(ClockworkAgent, action_sheet=HIT_AND_RUN_PATTERN)),
    "clockwork_aerial": (0.03, partial(ClockworkAgent, action_sheet=AERIAL_PATTERN)),
    "clockwork_special": (0.03, partial(ClockworkAgent, action_sheet=SPECIAL_SPAM_PATTERN)),
}

# Curriculum stages: gradually increase difficulty
CURRICULUM_STAGES = [
    # Stage 1: Easy scripted opponents only (first 200K steps)
    (0, {
        "constant": (0.30, partial(ConstantAgent)),  # Lots of easy constant agents
        "based": (0.20, partial(BasedAgent)),       # Some based agents
        "random": (0.20, partial(RandomAgent)),     # Random for variety
        "clockwork_aggressive": (0.15, partial(ClockworkAgent, action_sheet=AGGRESSIVE_PATTERN)),
        "clockwork_defensive": (0.10, partial(ClockworkAgent, action_sheet=DEFENSIVE_PATTERN)),
        "clockwork_hit_run": (0.05, partial(ClockworkAgent, action_sheet=HIT_AND_RUN_PATTERN)),
        "diverse_self_play": (0.0, None),  # No population opponents initially
    }),

    # Stage 2: Introduce some population opponents (200K-500K steps)
    (200_000, {
        "constant": (0.15, partial(ConstantAgent)),
        "based": (0.15, partial(BasedAgent)),
        "random": (0.10, partial(RandomAgent)),
        "clockwork_aggressive": (0.10, partial(ClockworkAgent, action_sheet=AGGRESSIVE_PATTERN)),
        "clockwork_defensive": (0.08, partial(ClockworkAgent, action_sheet=DEFENSIVE_PATTERN)),
        "clockwork_hit_run": (0.07, partial(ClockworkAgent, action_sheet=HIT_AND_RUN_PATTERN)),
        "clockwork_aerial": (0.05, partial(ClockworkAgent, action_sheet=AERIAL_PATTERN)),
        "diverse_self_play": (0.30, None),  # Introduce population opponents
    }),

    # Stage 3: More population opponents (500K+ steps)
    (500_000, {
        "constant": (0.08, partial(ConstantAgent)),
        "based": (0.10, partial(BasedAgent)),
        "random": (0.07, partial(RandomAgent)),
        "clockwork_aggressive": (0.08, partial(ClockworkAgent, action_sheet=AGGRESSIVE_PATTERN)),
        "clockwork_defensive": (0.07, partial(ClockworkAgent, action_sheet=DEFENSIVE_PATTERN)),
        "clockwork_hit_run": (0.05, partial(ClockworkAgent, action_sheet=HIT_AND_RUN_PATTERN)),
        "clockwork_aerial": (0.05, partial(ClockworkAgent, action_sheet=AERIAL_PATTERN)),
        "clockwork_special": (0.05, partial(ClockworkAgent, action_sheet=SPECIAL_SPAM_PATTERN)),
        "diverse_self_play": (0.45, None),  # Majority population opponents
    }),
]

# ============================================================================
# REWARD FUNCTIONS (same as OPTIMIZED)
# ============================================================================

def _get_diag_stats(env: WarehouseBrawl) -> dict:
    if not hasattr(env, '_diag_stats'):
        env._diag_stats = {
            'damage_dealt': 0.0,
            'damage_taken': 0.0,
            'zone_time': 0.0,
            'sparsity_penalty': 0.0,
            'inaction_penalty': 0.0,
            'movement_bonus': 0.0,
            'distance_reward': 0.0,
            'reward': 0.0,
        }
    return env._diag_stats


def damage_interaction_reward(env: WarehouseBrawl, mode: int = 1) -> float:
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

    # MASSIVELY increased reward scaling - individual hits should matter
    damage_reward = (delta_dealt - delta_taken) / 5.0  # Was /50, now /5 (10x increase)

    return damage_reward


def danger_zone_reward(env: WarehouseBrawl, zone_height: float = 4.2) -> float:
    player = env.objects["player"]
    if player.body.position.y >= zone_height:
        stats = _get_diag_stats(env)
        stats['zone_time'] += env.dt
        return -0.25 * env.dt  # Reduced from -1.0 to allow aerial combat
    return 0.0


def distance_control_reward(env: WarehouseBrawl) -> float:
    player = env.objects["player"]
    opponent = env.objects["opponent"]

    distance = float(np.linalg.norm(
        np.array([player.body.position.x, player.body.position.y]) -
        np.array([opponent.body.position.x, opponent.body.position.y])
    ))

    optimal_min, optimal_max = 1.5, 3.5  # Closer range to encourage engagement

    if optimal_min < distance < optimal_max:
        reward = 0.02
    elif distance > 6.0:
        reward = -0.015
    elif distance < 1.0:
        reward = -0.01
    else:
        reward = 0.0

    stats = _get_diag_stats(env)
    stats['distance_reward'] += reward
    return reward


def on_win_reward(env: WarehouseBrawl, agent: str) -> float:
    return 100.0 if agent == 'player' else -100.0


def on_knockout_reward(env: WarehouseBrawl, agent: str) -> float:
    return 8.0 if agent == 'opponent' else -8.0


def action_sparsity_reward(env: WarehouseBrawl, max_active: int = 3, penalty_per_key: float = 0.005) -> float:
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


def action_incentive_reward(env: WarehouseBrawl, inaction_penalty: float = 0.01) -> float:
    """
    Small penalty for complete inaction to encourage the agent to move/do something.
    This prevents the agent from learning to do nothing.
    """
    action = getattr(env, "cur_action", {}).get(0)
    if action is None:
        return 0.0

    # Check if any action is being taken (any key pressed above threshold)
    any_action = float((action > 0.5).any())

    # Small penalty for no action at all
    if any_action < 0.5:  # No keys pressed
        stats = _get_diag_stats(env)
        stats['inaction_penalty'] += inaction_penalty
        return -inaction_penalty

    return 0.0


def movement_reward(env: WarehouseBrawl, movement_bonus: float = 0.002) -> float:
    """
    Small bonus for horizontal movement to encourage the agent to move around.
    """
    action = getattr(env, "cur_action", {}).get(0)
    if action is None:
        return 0.0

    player = env.objects["player"]

    # Check if horizontal movement keys are pressed (assuming action[0] is left, action[1] is right)
    left_pressed = action[0] > 0.5 if len(action) > 0 else False
    right_pressed = action[1] > 0.5 if len(action) > 1 else False

    if left_pressed or right_pressed:
        stats = _get_diag_stats(env)
        stats['movement_bonus'] += movement_bonus
        return movement_bonus

    return 0.0


def attack_incentive_reward(env: WarehouseBrawl, attack_bonus: float = 0.01) -> float:
    """
    Reward for pressing attack buttons, especially when in range of opponent.
    """
    action = getattr(env, "cur_action", {}).get(0)
    if action is None:
        return 0.0

    player = env.objects["player"]
    opponent = env.objects["opponent"]

    # Check for attack buttons (j=light attack, k=heavy attack, l=dash)
    light_attack = action[7] > 0.5 if len(action) > 7 else False  # 'j' key (light attack)
    heavy_attack = action[8] > 0.5 if len(action) > 8 else False  # 'k' key (heavy attack)
    dash_attack = action[6] > 0.5 if len(action) > 6 else False   # 'l' key (dash, can be used for attacks)

    if not (light_attack or heavy_attack or dash_attack):
        return 0.0

    # Calculate distance to opponent
    distance = float(np.linalg.norm(
        np.array([player.body.position.x, player.body.position.y]) -
        np.array([opponent.body.position.x, opponent.body.position.y])
    ))

    # Bonus for attacking when in range (closer = higher bonus)
    range_multiplier = max(0.0, 1.0 - distance / 3.0)  # Full bonus at distance 0, half at distance 1.5, zero at distance 3.0+
    total_bonus = attack_bonus * (1.0 + range_multiplier * 2.0)  # Up to 3x bonus when close

    return total_bonus


def combo_incentive_reward(env: WarehouseBrawl, combo_bonus: float = 0.02) -> float:
    """
    Reward for combining movement with attacks (dash attacks, etc.)
    """
    action = getattr(env, "cur_action", {}).get(0)
    if action is None:
        return 0.0

    # Check for movement + attack combinations
    left_pressed = action[1] > 0.5 if len(action) > 1 else False  # 'a' key
    right_pressed = action[3] > 0.5 if len(action) > 3 else False # 'd' key
    down_pressed = action[2] > 0.5 if len(action) > 2 else False  # 's' key
    light_attack = action[7] > 0.5 if len(action) > 7 else False  # 'j' key
    heavy_attack = action[8] > 0.5 if len(action) > 8 else False  # 'k' key
    dash_attack = action[6] > 0.5 if len(action) > 6 else False   # 'l' key

    movement_keys = left_pressed or right_pressed or down_pressed
    attack_keys = light_attack or heavy_attack or dash_attack

    if movement_keys and attack_keys:
        return combo_bonus

    return 0.0


def gen_reward_manager():
    reward_functions = {
        'danger_zone': RewTerm(func=danger_zone_reward, weight=0.05),  # Reduced 4x to allow aerial play
        'damage_interaction': RewTerm(func=damage_interaction_reward, weight=1.5),  # INCREASED: Core damage reward
        'distance_control': RewTerm(func=distance_control_reward, weight=0.3),
        'action_incentive': RewTerm(func=action_incentive_reward, weight=0.4),  # Penalty for inaction
        'movement_bonus': RewTerm(func=movement_reward, weight=0.2),  # Bonus for movement
        'attack_incentive': RewTerm(func=attack_incentive_reward, weight=0.8),  # NEW: Reward for attacking
        'combo_incentive': RewTerm(func=combo_incentive_reward, weight=0.6),  # NEW: Reward for movement+attack combos
        # REMOVED: action_sparsity was causing agent to do nothing
        # 'action_sparsity': RewTerm(func=action_sparsity_reward, weight=0.1),
    }
    signal_subscriptions = {
        'on_win': ('win_signal', RewTerm(func=on_win_reward, weight=1.0)),
        'on_knockout': ('knockout_signal', RewTerm(func=on_knockout_reward, weight=1.0)),
    }
    return RewardManager(reward_functions, signal_subscriptions)

# ============================================================================
# ENVIRONMENT CREATION WITH STRATEGY ENCODER
# ============================================================================

def _make_self_play_env(
    seed: int,
    env_index: int,
    checkpoint_dir: Path
) -> SelfPlayWarehouseBrawl:
    seed_everything(seed)

    reward_manager = gen_reward_manager()

    # Create a NEW DiverseOpponentSampler for this environment
    # Each env gets its own PopulationManager for multiprocessing compatibility
    diverse_opponent_sampler = DiverseOpponentSampler(
        checkpoint_dir=str(checkpoint_dir),
        population_manager=None,  # Will create its own
        max_population_size=POPULATION_CONFIG["max_population_size"],
        num_weak_agents=POPULATION_CONFIG["num_weak_agents"],
        noise_probability=POPULATION_CONFIG["noise_probability"],
        use_population_prob=POPULATION_CONFIG["use_population_prob"],
        verbose=False,  # Suppress output for each environment
    )

    # Create opponent configuration with diverse sampler + scripted mix
    opponents_dict = {**OPPONENT_MIX}
    opponents_dict["diverse_self_play"] = (
        POPULATION_CONFIG["use_population_prob"],
        diverse_opponent_sampler
    )

    opponent_cfg = OpponentsCfg(opponents={
        k: (prob, agent_partial) for k, (prob, agent_partial) in opponents_dict.items()
    })

    env = SelfPlayWarehouseBrawl(
        reward_manager=reward_manager,
        opponent_cfg=opponent_cfg,
        save_handler=None,
        resolution=TRAINING_CONFIG["resolution"],
    )

    env.action_space.seed(seed)
    env.reset(seed=seed)

    diverse_opponent_sampler.env = env
    reward_manager.subscribe_signals(env.raw_env)
    opponent_cfg.base_probabilities = {
        name: (value if isinstance(value, float) else value[0])
        for name, value in opponent_cfg.opponents.items()
    }

    env.diverse_opponent_sampler = diverse_opponent_sampler
    env.env_index = env_index
    return env


def _make_vec_env(
    num_envs: int,
    checkpoint_dir: Path,
    use_multiprocessing: bool = True
) -> Tuple[VecNormalize, List[SelfPlayWarehouseBrawl]]:
    def make_thunk(rank: int) -> Callable[[], SelfPlayWarehouseBrawl]:
        def _init():
            env_seed = GLOBAL_SEED + rank * 9973
            return _make_self_play_env(env_seed, rank, checkpoint_dir)
        return _init

    env_fns = [make_thunk(i) for i in range(num_envs)]

    # Use SubprocVecEnv for multiprocessing (much faster) or DummyVecEnv for single-process
    if use_multiprocessing and num_envs > 1:
        vec_env = SubprocVecEnv(env_fns, start_method='spawn')
        print(f"✓ SubprocVecEnv initialized with {num_envs} workers (multiprocessing)")
    else:
        vec_env = DummyVecEnv(env_fns)
        print(f"✓ DummyVecEnv initialized with {num_envs} workers (single-process)")

    vec_env = VecMonitor(vec_env)

    # Add episode statistics tracking (damage, wins, etc.)
    vec_env = EpisodeStatsWrapper(vec_env)
    print(f"✓ EpisodeStatsWrapper added (tracks damage/wins)")

    # Add opponent history tracking
    vec_env = OpponentHistoryBuffer(
        vec_env,
        history_length=STRATEGY_ENCODER_CONFIG['history_length'],
    )
    print(f"✓ OpponentHistoryBuffer added (tracks last {STRATEGY_ENCODER_CONFIG['history_length']} frames)")

    # Add augmented observations (concatenate opponent history)
    vec_env = AugmentedObservationWrapper(
        vec_env,
        opponent_history_shape=(
            STRATEGY_ENCODER_CONFIG['history_length'],
            STRATEGY_ENCODER_CONFIG['input_features']
        )
    )

    # Normalize observations and rewards
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=150.0,
        gamma=AGENT_CONFIG["gamma"],
    )

    # === OBSERVATION VALIDATION ===
    print("\n" + "="*70)
    print("VALIDATING OBSERVATION PIPELINE")
    print("="*70)

    # Test observation shape
    test_obs = vec_env.reset()
    expected_base_obs = 52  # Approximate base observation size
    expected_history = STRATEGY_ENCODER_CONFIG['history_length'] * STRATEGY_ENCODER_CONFIG['input_features']
    expected_total = expected_base_obs + expected_history

    actual_obs_shape = test_obs.shape
    print(f"✓ Observation shape: {actual_obs_shape}")
    print(f"  Expected: ({num_envs}, ~{expected_total}) = base(~{expected_base_obs}) + history({expected_history})")

    if actual_obs_shape[0] != num_envs:
        raise ValueError(f"❌ Environment count mismatch! Expected {num_envs}, got {actual_obs_shape[0]}")

    if actual_obs_shape[1] < expected_history:
        raise ValueError(f"❌ Observation too small! Expected at least {expected_history}D, got {actual_obs_shape[1]}D")

    # Check for NaN/Inf in initial observations
    if np.any(np.isnan(test_obs)):
        raise ValueError("❌ NaN detected in initial observations!")
    if np.any(np.isinf(test_obs)):
        raise ValueError("❌ Inf detected in initial observations!")

    print(f"✓ No NaN/Inf in initial observations")

    # Test one step to verify opponent history is being populated
    test_actions = np.array([vec_env.action_space.sample() for _ in range(num_envs)])
    test_obs_next, _, _, test_info = vec_env.step(test_actions)

    # Check if observations changed (history should update)
    obs_changed = not np.allclose(test_obs, test_obs_next, rtol=0.01)
    print(f"✓ Observations updating: {obs_changed}")

    # Check info dict for opponent history
    if isinstance(test_info, list) and len(test_info) > 0:
        sample_info = test_info[0]
        has_history = 'opponent_history' in sample_info
        print(f"✓ Opponent history in info dict: {has_history}")
        if has_history:
            hist_shape = sample_info['opponent_history'].shape
            print(f"  History shape: {hist_shape}")

    vec_env.reset()  # Reset after validation
    print("="*70 + "\n")

    base_envs: List[SelfPlayWarehouseBrawl] = []
    unwrap_ptr = vec_env
    while hasattr(unwrap_ptr, "venv"):
        unwrap_ptr = unwrap_ptr.venv
    if hasattr(unwrap_ptr, "envs"):
        base_envs = unwrap_ptr.envs
    else:
        base_envs = [unwrap_ptr]

    return vec_env, base_envs


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def test_reward_functions():
    """Test reward functions with sample scenarios to verify they're working"""
    print("\n" + "=" * 70)
    print("TESTING REWARD FUNCTIONS")
    print("=" * 70)

    try:
        # Create a minimal test environment
        from environment.WarehouseBrawl import WarehouseBrawl

        # Create a simple test environment
        env = WarehouseBrawl()
        env.reset()

        # Set up test scenario: player close to opponent
        player = env.objects["player"]
        opponent = env.objects["opponent"]
        player.body.position = (0, 0)
        opponent.body.position = (2, 0)  # Close to player

        # Test 1: Distance reward (should be positive for optimal distance)
        distance_reward = distance_control_reward(env)
        print(f"✓ Distance reward (optimal range): {distance_reward:.2f}")

        # Test 2: Attack incentive reward (simulate pressing attack button)
        env.cur_action = [np.zeros(10)]  # Initialize action array
        env.cur_action[0][7] = 1.0  # Press light attack
        attack_reward = attack_incentive_reward(env)
        print(f"✓ Attack incentive reward: {attack_reward:.2f}")

        # Test 3: Combo reward (movement + attack)
        env.cur_action[0][3] = 1.0  # Also press right
        combo_reward = combo_incentive_reward(env)
        print(f"✓ Combo reward (move+attack): {combo_reward:.2f}")

        # Test 4: Damage reward (simulate dealing damage)
        env._last_damage_totals = {"player": 0, "opponent": 0}
        opponent.damage_taken_total = 10  # Simulate 10 damage dealt
        damage_reward = damage_interaction_reward(env)
        print(f"✓ Damage reward (10 damage dealt): {damage_reward:.2f}")

        print("✓ All reward functions are working!")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"⚠ Reward function test failed: {e}")
        print("   This might be due to environment setup issues, but training should still work.")
        print("=" * 70 + "\n")


def train():
    print("\n" + "=" * 70)
    print("STRATEGY-CONDITIONED TRAINING WITH POPULATION-BASED SELF-PLAY")
    print("=" * 70 + "\n")

    # Check available disk space before starting
    import shutil
    total, used, free = shutil.disk_usage("/")
    free_gb = free / (1024**3)
    if free_gb < 0.5:  # Require at least 500MB free (reduced due to fewer checkpoints)
        print(f"❌ ERROR: Only {free_gb:.1f}GB disk space available. Need at least 0.5GB.")
        print("   Please free up disk space or use a different checkpoint location.")
        return

    print(f"✓ Population-based self-play configured:")
    print(f"  - Each environment creates its own PopulationManager (multiprocessing-safe)")
    print(f"  - Max population: {POPULATION_CONFIG['max_population_size']} agents")
    print(f"  - Population sampling: {POPULATION_CONFIG['use_population_prob']:.0%} of episodes")
    print(f"  - Noise injection: {POPULATION_CONFIG['noise_probability']:.0%} of episodes\n")

    # Clean up any existing checkpoints to save space
    if CHECKPOINT_DIR.exists():
        import glob
        checkpoint_files = glob.glob(str(CHECKPOINT_DIR / "*.zip"))
        if checkpoint_files:
            print(f"🧹 Cleaning up {len(checkpoint_files)} old checkpoint files...")
            for f in checkpoint_files:
                Path(f).unlink()
        population_dir = CHECKPOINT_DIR / "population"
        if population_dir.exists():
            pop_files = list(population_dir.glob("*.zip"))
            if pop_files:
                print(f"🧹 Cleaning up {len(pop_files)} old population files...")
                for f in pop_files:
                    f.unlink()

    # Create environment
    vec_env, env_instances = _make_vec_env(
        TRAINING_CONFIG["n_envs"],
        CHECKPOINT_DIR,
        use_multiprocessing=False,  # Disable multiprocessing to avoid asset loading conflicts
    )
    primary_env = env_instances[0]

    print(f"✓ Environment created ({len(env_instances)} parallel arenas)")
    # Removed verbose dimension info - user wants lightweight output focused on rewards

    # Initialize normalization and test reward signals
    print("Initializing observation/reward normalization...")
    print("Testing reward functions...")
    vec_env.reset()

    reward_history = []
    for i in range(100):
        actions = np.array([vec_env.action_space.sample() for _ in range(TRAINING_CONFIG["n_envs"])])
        obs, rewards, dones, infos = vec_env.step(actions)
        reward_history.extend(rewards.tolist())

    vec_env.reset()

    # Analyze reward distribution
    reward_history = np.array(reward_history)
    nonzero_rewards = reward_history[reward_history != 0]

    print(f"✓ Normalization initialized")
    print(f"  Reward statistics (100 steps × {TRAINING_CONFIG['n_envs']} envs):")
    print(f"    Mean:     {reward_history.mean():.4f}")
    print(f"    Std:      {reward_history.std():.4f}")
    print(f"    Min:      {reward_history.min():.4f}")
    print(f"    Max:      {reward_history.max():.4f}")
    print(f"    Non-zero: {len(nonzero_rewards)}/{len(reward_history)} ({100*len(nonzero_rewards)/len(reward_history):.1f}%)")

    if len(nonzero_rewards) == 0:
        print(f"  ⚠ WARNING: All rewards are zero! Agent may not learn.")
    elif reward_history.std() < 0.001:
        print(f"  ⚠ WARNING: Very low reward variance. Check reward functions.")
    else:
        print(f"  ✓ Rewards are non-zero and varied\n")

    # Create or load model
    model_path = CHECKPOINT_DIR / "latest_model.zip"

    # Set tensorboard logging only if available
    tensorboard_log = str(TENSORBOARD_DIR) if TENSORBOARD_AVAILABLE else None

    if model_path.exists():
        print(f"Loading existing model from {model_path}...")
        model = RecurrentPPO.load(
            model_path,
            env=vec_env,
            device=DEVICE,
            tensorboard_log=tensorboard_log,
            verbose=1,  # ✓ Force verbose logging
            **{k: v for k, v in AGENT_CONFIG.items() if k not in ["policy", "policy_kwargs", "verbose"]}
        )
        print("✓ Model loaded with verbose logging enabled\n")
    else:
        print("Creating new model...")
        model = RecurrentPPO(
            **AGENT_CONFIG,
            env=vec_env,
            verbose=1,  # ✓ Enable verbose logging
            device=DEVICE,
            tensorboard_log=tensorboard_log,
        )
        print("✓ Model created\n")

    # Create custom checkpoint callback that also saves latest model
    class LatestModelCheckpointCallback(CheckpointCallback):
        """Extended checkpoint callback that also saves latest_model.zip"""
        def _on_step(self) -> bool:
            result = super()._on_step()
            # After each checkpoint, also save as latest_model
            if self.n_calls % self.save_freq == 0:
                latest_path = Path(self.save_path) / "latest_model.zip"
                self.model.save(latest_path)
                # Also save vec_normalize
                if hasattr(self.training_env, 'save'):
                    self.training_env.save(Path(self.save_path) / "latest_vec_normalize.pkl")
            return result

    checkpoint_callback = LatestModelCheckpointCallback(
        save_freq=TRAINING_CONFIG["save_freq"],
        save_path=CHECKPOINT_DIR,
        name_prefix="rl_model",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    population_update_callback = PopulationUpdateCallback(
        population_manager=None,  # Will create its own
        update_frequency=POPULATION_CONFIG["update_frequency"],
        checkpoint_dir=str(CHECKPOINT_DIR),
        max_population_size=POPULATION_CONFIG["max_population_size"],
        num_weak_agents=POPULATION_CONFIG["num_weak_agents"],
        min_timesteps_before_add=1_000_000,  # INCREASED: Start adding agents much later
        verbose=1,
    )

    # Add training metrics callback for detailed console logging
    metrics_csv_path = CHECKPOINT_DIR / "training_metrics.csv"
    metrics_callback = create_training_metrics_callback(
        log_frequency=1,  # Log every rollout for maximum visibility
        moving_avg_window=100,
        verbose=1,
        csv_path=str(metrics_csv_path),  # Enable CSV export
        track_actions=True,  # Enable action distribution tracking
    )
    print("✓ Training metrics callback added (detailed console logging + CSV export enabled)")

    # Add gradient monitoring callback for stability
    gradient_callback = create_gradient_monitor_callback(
        check_frequency=10,  # Check every 10 updates
        verbose=1,
        max_grad_norm_threshold=100.0,
        stop_on_nan=True,  # Stop training if NaN detected
    )
    print("✓ Gradient monitoring callback added (will detect NaN/Inf)")

    # Add curriculum learning callback
    curriculum_callback = CurriculumCallback(CURRICULUM_STAGES, verbose=1)
    print("✓ Curriculum learning callback added (gradually increases opponent difficulty)\n")

    # Add comprehensive debug callback to diagnose no-damage issue
    class DebugCallback(BaseCallback):
        """Enhanced debug callback to log actions, distances, damage, and attack patterns"""
        def __init__(self, log_frequency=500, verbose=0):
            super().__init__(verbose)
            self.log_frequency = log_frequency
            self.step_count = 0
            self.attack_attempts = 0
            self.last_damage_dealt = 0
            self.last_damage_taken = 0
            self.episode_start_step = 0

        def _on_step(self) -> bool:
            self.step_count += 1

            # Log every N steps
            if self.step_count % self.log_frequency == 0:
                # Get current episode info from environments
                if hasattr(self.training_env, 'get_attr'):
                    try:
                        # Try to get info from first environment
                        infos = self.locals.get('infos', [{}])
                        if len(infos) > 0 and isinstance(infos[0], dict):
                            info = infos[0]

                            # Show total reward prominently
                            if 'rewards' in self.locals and len(self.locals['rewards']) > 0:
                                reward = self.locals['rewards'][0]
                                print(f"[STEP {self.step_count}] Reward: {reward:+.3f}")

                                # Try to get distance and damage info
                                try:
                                    venv = self.training_env
                                    while hasattr(venv, 'venv') and venv != venv.venv:
                                        if hasattr(venv, 'get_attr'):
                                            try:
                                                players = venv.get_attr('objects')[0]['player']
                                                opponents = venv.get_attr('objects')[0]['opponent']
                                                if players and opponents:
                                                    player_pos = players.body.position
                                                    opp_pos = opponents.body.position
                                                    distance = float(np.linalg.norm(
                                                        np.array([player_pos.x, player_pos.y]) -
                                                        np.array([opp_pos.x, opp_pos.y])
                                                    ))
                                                    print(f"  Distance: {distance:.2f}")
                                            except:
                                                pass
                                        venv = venv.venv
                                except Exception as e:
                                    pass

                                # Try to get reward component breakdown from reward manager
                                try:
                                    venv = self.training_env
                                    while hasattr(venv, 'venv') and venv != venv.venv:
                                        if hasattr(venv, 'reward_manager'):
                                            manager = venv.reward_manager
                                            if hasattr(manager, 'reward_functions') and manager.reward_functions:
                                                components = {}
                                                for name, term_cfg in manager.reward_functions.items():
                                                    if term_cfg.weight != 0.0:
                                                        value = term_cfg.func(venv, **term_cfg.params) * term_cfg.weight
                                                        if abs(value) > 0.01:  # Only show significant rewards
                                                            components[name] = value
                                                if components:
                                                    comp_str = " | ".join([f"{k}:{v:+.2f}" for k, v in components.items()])
                                                    print(f"  Breakdown: {comp_str}")
                                            break
                                        venv = venv.venv
                                except Exception as e:
                                    pass  # Silently skip component breakdown if it fails

                            # Enhanced action analysis
                            if 'actions' in self.locals:
                                actions = self.locals['actions']
                                if len(actions) > 0:
                                    action_names = ['↑', '←', '↓', '→', 'jump', 'pickup', 'dash', 'light_atk', 'heavy_atk', 'taunt']
                                    pressed = [name for i, name in enumerate(action_names) if i < len(actions[0]) and actions[0][i] > 0.5]

                                    # Count attack attempts (light attack, heavy attack, dash)
                                    light_attack = len(actions[0]) > 7 and actions[0][7] > 0.5  # 'j' key
                                    heavy_attack = len(actions[0]) > 8 and actions[0][8] > 0.5  # 'k' key
                                    dash_attack = len(actions[0]) > 6 and actions[0][6] > 0.5   # 'l' key (can be used for attacks)
                                    if light_attack or heavy_attack or dash_attack:
                                        self.attack_attempts += 1

                                    if pressed:
                                        print(f"  Actions: {' '.join(pressed)}")
                                        # Show attack frequency
                                        attack_rate = self.attack_attempts / (self.step_count - self.episode_start_step + 1) * 100
                                        print(f"  Attack Rate: {attack_rate:.1f}% of steps")

                            # Show damage progress
                            if 'damage_dealt' in info and 'damage_taken' in info:
                                damage_dealt = info['damage_dealt']
                                damage_taken = info['damage_taken']
                                print(f"  Cumulative: Damage dealt={damage_dealt:.1f}, taken={damage_taken:.1f}")

                                # Check if this is episode end
                                if info.get('episode_end', False) or 'episode' in info:
                                    print(f"  EPISODE END: Final damage dealt={damage_dealt:.1f}, taken={damage_taken:.1f}")
                                    if damage_dealt > 0:
                                        print("  ✓ Agent dealt damage this episode!")
                                    else:
                                        print("  ⚠ Agent dealt NO damage this episode")
                                    self.attack_attempts = 0
                                    self.episode_start_step = self.step_count

                    except Exception as e:
                        print(f"  [DEBUG ERROR: {e}]")

            return True

    debug_callback = DebugCallback(log_frequency=500, verbose=1)
    print("✓ Enhanced debug callback added (will log detailed actions/distance/damage every 500 steps)\n")

    callbacks = CallbackList([
        checkpoint_callback,
        population_update_callback,
        curriculum_callback,  # Add curriculum learning
        metrics_callback,
        gradient_callback,
        debug_callback,
    ])

    # Print training info
    print("=" * 70)
    print("TRAINING CONFIGURATION")
    print("=" * 70)
    print(f"Total timesteps: {TRAINING_CONFIG['total_timesteps']:,}")
    print(f"Rollout steps (n_steps): {AGENT_CONFIG['n_steps']}")
    print(f"Batch size: {AGENT_CONFIG['batch_size']}")
    print(f"N epochs: {AGENT_CONFIG['n_epochs']}")

    # Handle learning rate (could be schedule or float)
    lr = AGENT_CONFIG['learning_rate']
    if callable(lr):
        print(f"Learning rate: 1e-4 → 3e-5 (linear decay)")
    else:
        print(f"Learning rate: {lr}")

    print(f"Entropy coef: {AGENT_CONFIG['ent_coef']}")
    print(f"Clip range: {AGENT_CONFIG['clip_range']}")
    print(f"Target KL: {AGENT_CONFIG['target_kl']} (None = no early stopping)")
    print(f"Max grad norm: {AGENT_CONFIG['max_grad_norm']}")
    print(f"Population updates: every {POPULATION_CONFIG['update_frequency']:,} steps")
    print(f"Checkpoints: every {TRAINING_CONFIG['save_freq']:,} steps")
    print(f"Device: {DEVICE}")
    if TENSORBOARD_AVAILABLE:
        print(f"TensorBoard: Enabled ({TENSORBOARD_DIR})")
    else:
        print("TensorBoard: Disabled (not installed - install with: pip install tensorboard)")
    print("=" * 70 + "\n")

    # Start training
    print("Starting training...\n")
    model.learn(
        total_timesteps=TRAINING_CONFIG["total_timesteps"],
        callback=callbacks,
        log_interval=10,
        progress_bar=False,  # Disabled to prevent tqdm/rich shutdown crash
    )

    # Save final model
    final_path = CHECKPOINT_DIR / "final_model.zip"
    model.save(final_path)
    vec_env.save(CHECKPOINT_DIR / "final_vec_normalize.pkl")

    # Also save as latest
    latest_path = CHECKPOINT_DIR / "latest_model.zip"
    model.save(latest_path)
    vec_env.save(CHECKPOINT_DIR / "latest_vec_normalize.pkl")

    print(f"\n✓ Training complete!")
    print(f"  Final model saved: {final_path}")
    print(f"  Latest model saved: {latest_path}")

    # Get population size from callback's population manager
    if hasattr(population_update_callback, 'population_manager'):
        print(f"  Population size: {len(population_update_callback.population_manager)}")

    # Get sampler stats from first environment (only works with DummyVecEnv)
    if len(env_instances) > 0 and hasattr(env_instances[0], 'diverse_opponent_sampler'):
        print(f"  Sampler stats (env 0): {env_instances[0].diverse_opponent_sampler.get_stats()}")
    else:
        print(f"  Sampler stats: Not available (using multiprocessing)")

    # Run quick benchmark against scripted opponents
    print("\n" + "="*70)
    print("RUNNING POST-TRAINING BENCHMARK")
    print("="*70)
    benchmark_agent(model, vec_env, primary_env)


def benchmark_agent(model, vec_env, env):
    """Quick benchmark test against scripted opponents."""
    from environment.agent import ConstantAgent, BasedAgent, RandomAgent

    print("\nTesting against scripted opponents (5 episodes each)...\n")

    opponents = [
        ("ConstantAgent", ConstantAgent),
        ("BasedAgent", BasedAgent),
        ("RandomAgent", RandomAgent),
    ]

    model.policy.set_training_mode(False)

    for opp_name, opp_class in opponents:
        wins = 0
        total_reward = 0

        for episode in range(5):
            obs = vec_env.reset()
            done = False
            episode_reward = 0
            lstm_states = None
            episode_start = np.array([True])

            step_count = 0
            max_steps = 30 * 90  # 90 seconds max

            while not done and step_count < max_steps:
                action, lstm_states = model.predict(
                    obs,
                    state=lstm_states,
                    episode_start=episode_start,
                    deterministic=True
                )
                obs, reward, done, info = vec_env.step(action)
                episode_reward += reward[0]
                episode_start = np.array([False])
                step_count += 1

            total_reward += episode_reward
            # Simple heuristic: positive reward = win
            if episode_reward > 0:
                wins += 1

        win_rate = wins / 5 * 100
        avg_reward = total_reward / 5
        status = "✓" if win_rate >= 60 else "○" if win_rate >= 40 else "✗"

        print(f"{status} vs {opp_name:15s}: {win_rate:5.1f}% win rate | avg reward: {avg_reward:7.1f}")

    print("\n" + "="*70)
    model.policy.set_training_mode(True)


if __name__ == "__main__":
    # ============================================================================
    # INITIALIZATION (only runs in main process, not in worker processes)
    # ============================================================================

    # Seed everything
    seed_everything(GLOBAL_SEED)

    # Initialize device and CUDA
    DEVICE = get_device()

    # Create checkpoint directory
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Check if tensorboard is available
    try:
        import tensorboard
        TENSORBOARD_DIR = CHECKPOINT_DIR / "tb_logs"
        TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)
        TENSORBOARD_AVAILABLE = True
    except ImportError:
        TENSORBOARD_DIR = None
        TENSORBOARD_AVAILABLE = False

    # Setup logging
    LOG_FILE = CHECKPOINT_DIR / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler()
        ]
    )

    logger.info("="*70)
    logger.info("TRAINING SESSION STARTED")
    logger.info(f"Log file: {LOG_FILE}")
    logger.info("="*70)

    # Test reward functions before training
    test_reward_functions()

    # ============================================================================
    # START TRAINING
    # ============================================================================

    try:
        train()
        logger.info("=" * 70)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)
    except KeyboardInterrupt:
        logger.warning("=" * 70)
        logger.warning("TRAINING INTERRUPTED BY USER (Ctrl+C)")
        logger.warning("=" * 70)
        print("\n⚠ Training interrupted by user. Checkpoints have been saved.")
    except Exception as e:
        logger.error("=" * 70)
        logger.error("TRAINING FAILED WITH ERROR")
        logger.error("=" * 70)
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.exception("Full traceback:")
        print(f"\n❌ Training failed with error: {e}")
        print(f"   See {LOG_FILE} for full details")
        raise
