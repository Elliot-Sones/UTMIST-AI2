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
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

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


seed_everything(GLOBAL_SEED)

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================

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

DEVICE = get_device()

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

CHECKPOINT_DIR = Path("checkpoints/strategy_encoder_training")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Check if tensorboard is available (optional dependency)
try:
    import tensorboard
    TENSORBOARD_DIR = CHECKPOINT_DIR / "tb_logs"
    TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_DIR = None
    TENSORBOARD_AVAILABLE = False

# Setup comprehensive logging
LOG_FILE = CHECKPOINT_DIR / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()  # Also print to console
    ]
)
logger = logging.getLogger(__name__)

logger.info("="*70)
logger.info("TRAINING SESSION STARTED")
logger.info(f"Log file: {LOG_FILE}")
logger.info("="*70)

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
    "learning_rate": linear_schedule(1e-4, 3e-5),  # Conservative learning rate (10x lower)
    "ent_coef": 0.01,  # Moderate entropy for exploration
    "clip_range": 0.2,  # Standard PPO clipping
    "gamma": 0.99,  # Standard discount factor
    "gae_lambda": 0.95,  # Standard GAE
    "max_grad_norm": 0.5,  # Prevent gradient explosions
    "vf_coef": 0.5,
    "clip_range_vf": 0.2,  # RE-ENABLED: Stabilizes value function
    "use_sde": False,  # Disable SDE for stability
    "sde_sample_freq": 4,
    "target_kl": 0.03,  # RE-ENABLED: Prevents policy divergence
}

TRAINING_CONFIG = {
    "total_timesteps": 5_000_000,  # Extended for diverse strategy learning
    "save_freq": 50_000,
    "resolution": CameraResolution.LOW,
    "n_envs": 16,  # Increased for efficiency (will use multiprocessing)
}

# Population-based self-play configuration
POPULATION_CONFIG = {
    "max_population_size": 15,
    "num_weak_agents": 3,
    "update_frequency": 100_000,
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

    return (delta_dealt - delta_taken) / 140


def danger_zone_reward(env: WarehouseBrawl, zone_height: float = 4.2) -> float:
    player = env.objects["player"]
    if player.body.position.y >= zone_height:
        stats = _get_diag_stats(env)
        stats['zone_time'] += env.dt
        return -1.0 * env.dt
    return 0.0


def distance_control_reward(env: WarehouseBrawl) -> float:
    player = env.objects["player"]
    opponent = env.objects["opponent"]

    distance = float(np.linalg.norm(
        np.array([player.body.position.x, player.body.position.y]) -
        np.array([opponent.body.position.x, opponent.body.position.y])
    ))

    optimal_min, optimal_max = 2.0, 4.5

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


def gen_reward_manager():
    reward_functions = {
        'danger_zone': RewTerm(func=danger_zone_reward, weight=0.2),
        'damage_interaction': RewTerm(func=damage_interaction_reward, weight=0.75),
        'distance_control': RewTerm(func=distance_control_reward, weight=0.5),
        'action_sparsity': RewTerm(func=action_sparsity_reward, weight=1.0),
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

def train():
    print("\n" + "=" * 70)
    print("STRATEGY-CONDITIONED TRAINING WITH POPULATION-BASED SELF-PLAY")
    print("=" * 70 + "\n")

    print(f"✓ Population-based self-play configured:")
    print(f"  - Each environment creates its own PopulationManager (multiprocessing-safe)")
    print(f"  - Max population: {POPULATION_CONFIG['max_population_size']} agents")
    print(f"  - Population sampling: {POPULATION_CONFIG['use_population_prob']:.0%} of episodes")
    print(f"  - Noise injection: {POPULATION_CONFIG['noise_probability']:.0%} of episodes\n")

    # Create environment
    vec_env, env_instances = _make_vec_env(
        TRAINING_CONFIG["n_envs"],
        CHECKPOINT_DIR,
        use_multiprocessing=True,  # Enable multiprocessing for speed
    )
    primary_env = env_instances[0]

    print(f"✓ Environment created ({len(env_instances)} parallel arenas)")
    print(f"  Observation dim: {vec_env.observation_space.shape[0]}")
    print(f"    - Agent obs: ~52D")
    print(f"    - Opponent history: {STRATEGY_ENCODER_CONFIG['history_length']} × {STRATEGY_ENCODER_CONFIG['input_features']} = {STRATEGY_ENCODER_CONFIG['history_length'] * STRATEGY_ENCODER_CONFIG['input_features']}D")
    print(f"  Strategy encoder outputs: {STRATEGY_ENCODER_CONFIG['embedding_dim']}D embedding")
    print(f"  Combined features → LSTM: {BASE_EXTRACTOR_CONFIG['feature_dim'] + STRATEGY_ENCODER_CONFIG['embedding_dim']}D\n")

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
        min_timesteps_before_add=100_000,  # Start adding agents after 100K steps
        verbose=1,
    )

    # Add training metrics callback for detailed console logging
    metrics_callback = create_training_metrics_callback(
        log_frequency=1,  # Log every rollout for maximum visibility
        moving_avg_window=100,
        verbose=1,
    )
    print("✓ Training metrics callback added (detailed console logging enabled)")

    # Add gradient monitoring callback for stability
    gradient_callback = create_gradient_monitor_callback(
        check_frequency=10,  # Check every 10 updates
        verbose=1,
        max_grad_norm_threshold=100.0,
        stop_on_nan=True,  # Stop training if NaN detected
    )
    print("✓ Gradient monitoring callback added (will detect NaN/Inf)\n")

    callbacks = CallbackList([
        checkpoint_callback,
        population_update_callback,
        metrics_callback,
        gradient_callback,
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
        progress_bar=True,
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
    print(f"  Population size: {len(population_manager)}")

    # Get sampler stats from first environment
    if hasattr(env_instances[0], 'diverse_opponent_sampler'):
        print(f"  Sampler stats (env 0): {env_instances[0].diverse_opponent_sampler.get_stats()}")

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
