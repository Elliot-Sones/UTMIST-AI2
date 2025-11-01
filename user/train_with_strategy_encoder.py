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
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from pathlib import Path
from typing import Any, Callable, Dict, Tuple, List
import torch
import torch.nn as nn
import numpy as np
from functools import partial

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environment.agent import *
from user.models.opponent_conditioned_policy import create_opponent_conditioned_policy_kwargs
from user.wrappers.opponent_history_wrapper import OpponentHistoryBuffer
from user.wrappers.augmented_obs_wrapper import AugmentedObservationWrapper
from user.self_play.population_manager import PopulationManager
from user.self_play.diverse_opponent_sampler import DiverseOpponentSampler
from user.self_play.population_update_callback import PopulationUpdateCallback

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
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def linear_schedule(start: float, end: float) -> Callable[[float], float]:
    def schedule(progress_remaining: float) -> float:
        return end + (start - end) * progress_remaining
    return schedule


seed_everything(GLOBAL_SEED)

if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")
if torch.cuda.is_available() and hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
    torch.backends.cuda.matmul.allow_tf32 = True

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================

def get_device():
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
# TRAINING CONFIGURATION
# ============================================================================

CHECKPOINT_DIR = Path("checkpoints/strategy_encoder_training")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
TENSORBOARD_DIR = CHECKPOINT_DIR / "tb_logs"
TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)

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

    # PPO training
    "n_steps": 4096,
    "batch_size": 1024,
    "n_epochs": 6,
    "learning_rate": linear_schedule(3e-4, 3e-5),  # Decay from 3e-4 to 3e-5
    "ent_coef": 0.02,  # High exploration (constant for stability)
    "clip_range": linear_schedule(0.2, 0.1),  # Decay from 0.2 to 0.1
    "gamma": 0.995,
    "gae_lambda": 0.98,
    "max_grad_norm": 1.5,
    "vf_coef": 0.5,
    "clip_range_vf": 0.2,
    "use_sde": True,
    "sde_sample_freq": 4,
    "target_kl": 0.035,
}

TRAINING_CONFIG = {
    "total_timesteps": 5_000_000,  # Extended for diverse strategy learning
    "save_freq": 50_000,
    "resolution": CameraResolution.LOW,
    "n_envs": 4,
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
    population_manager: PopulationManager
) -> SelfPlayWarehouseBrawl:
    seed_everything(seed)

    reward_manager = gen_reward_manager()

    # Create a NEW DiverseOpponentSampler for this environment
    # This avoids pickle issues with shared state
    diverse_opponent_sampler = DiverseOpponentSampler(
        checkpoint_dir=CHECKPOINT_DIR,
        population_manager=population_manager,
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
    population_manager: PopulationManager
) -> Tuple[VecNormalize, List[SelfPlayWarehouseBrawl]]:
    def make_thunk(rank: int) -> Callable[[], SelfPlayWarehouseBrawl]:
        def _init():
            env_seed = GLOBAL_SEED + rank * 9973
            return _make_self_play_env(env_seed, rank, population_manager)
        return _init

    env_fns = [make_thunk(i) for i in range(num_envs)]

    # Use DummyVecEnv to avoid multiprocessing pickle issues with CUDA models
    vec_env = DummyVecEnv(env_fns)
    print(f"✓ DummyVecEnv initialized with {num_envs} workers")

    vec_env = VecMonitor(vec_env)

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

    # Create population manager (shared across all environments)
    population_manager = PopulationManager(
        checkpoint_dir=CHECKPOINT_DIR,
        max_population_size=POPULATION_CONFIG["max_population_size"],
        num_weak_agents=POPULATION_CONFIG["num_weak_agents"],
    )

    print(f"✓ Population-based self-play configured:")
    print(f"  - Each environment creates its own opponent sampler")
    print(f"  - Shared population: {POPULATION_CONFIG['max_population_size']} max agents")
    print(f"  - Population sampling: {POPULATION_CONFIG['use_population_prob']:.0%} of episodes")
    print(f"  - Noise injection: {POPULATION_CONFIG['noise_probability']:.0%} of episodes\n")

    # Create environment
    vec_env, env_instances = _make_vec_env(
        TRAINING_CONFIG["n_envs"],
        population_manager
    )
    primary_env = env_instances[0]

    print(f"✓ Environment created ({len(env_instances)} parallel arenas)")
    print(f"  Observation dim: {vec_env.observation_space.shape[0]}")
    print(f"    - Agent obs: ~52D")
    print(f"    - Opponent history: {STRATEGY_ENCODER_CONFIG['history_length']} × {STRATEGY_ENCODER_CONFIG['input_features']} = {STRATEGY_ENCODER_CONFIG['history_length'] * STRATEGY_ENCODER_CONFIG['input_features']}D")
    print(f"  Strategy encoder outputs: {STRATEGY_ENCODER_CONFIG['embedding_dim']}D embedding")
    print(f"  Combined features → LSTM: {BASE_EXTRACTOR_CONFIG['feature_dim'] + STRATEGY_ENCODER_CONFIG['embedding_dim']}D\n")

    # Initialize normalization
    print("Initializing observation/reward normalization...")
    vec_env.reset()
    for _ in range(100):
        actions = np.array([vec_env.action_space.sample() for _ in range(TRAINING_CONFIG["n_envs"])])
        vec_env.step(actions)
    vec_env.reset()
    print("✓ Normalization initialized\n")

    # Create or load model
    model_path = CHECKPOINT_DIR / "latest_model.zip"
    if model_path.exists():
        print(f"Loading existing model from {model_path}...")
        model = RecurrentPPO.load(
            model_path,
            env=vec_env,
            device=DEVICE,
            **{k: v for k, v in AGENT_CONFIG.items() if k not in ["policy", "policy_kwargs"]}
        )
        print("✓ Model loaded\n")
    else:
        print("Creating new model...")
        model = RecurrentPPO(
            **AGENT_CONFIG,
            env=vec_env,
            verbose=1,
            device=DEVICE,
            tensorboard_log=str(TENSORBOARD_DIR),
        )
        print("✓ Model created\n")

    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=TRAINING_CONFIG["save_freq"],
        save_path=CHECKPOINT_DIR,
        name_prefix="rl_model",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    population_update_callback = PopulationUpdateCallback(
        population_manager=population_manager,
        update_frequency=POPULATION_CONFIG["update_frequency"],
        checkpoint_dir=CHECKPOINT_DIR,
        verbose=1,
    )

    callbacks = CallbackList([checkpoint_callback, population_update_callback])

    # Print training info
    print("=" * 70)
    print("TRAINING CONFIGURATION")
    print("=" * 70)
    print(f"Total timesteps: {TRAINING_CONFIG['total_timesteps']:,}")
    print(f"Learning rate: 3e-4 → 3e-5 (linear decay)")
    print(f"Batch size: {AGENT_CONFIG['batch_size']}")
    print(f"Entropy coef: 0.02 → 0.005 (linear decay)")
    print(f"Clip range: 0.2 → 0.1 (linear decay)")
    print(f"Population updates: every {POPULATION_CONFIG['update_frequency']:,} steps")
    print(f"Checkpoints: every {TRAINING_CONFIG['save_freq']:,} steps")
    print(f"Device: {DEVICE}")
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

    print(f"\n✓ Training complete!")
    print(f"  Final model saved: {final_path}")
    print(f"  Population size: {len(population_manager)}")

    # Get sampler stats from first environment
    if hasattr(env_instances[0], 'diverse_opponent_sampler'):
        print(f"  Sampler stats (env 0): {env_instances[0].diverse_opponent_sampler.get_stats()}")


if __name__ == "__main__":
    train()
