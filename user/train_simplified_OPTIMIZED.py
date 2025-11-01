"""
UTMIST AI² - RecurrentPPO (LSTM-only) Training - OPTIMIZED VERSION
==================================================================

Key Optimizations:
1. Increased win reward dominance (30 → 100)
2. Higher exploration (ent_coef: 0.005 → 0.02)
3. Less restrictive policy updates (target_kl, clip_range, grad_norm)
4. Deeper feature extractor (3 → 5 residual blocks)
5. Added positional/strategic rewards
6. Increased self-play weight (20% → 40%)
7. Enabled orthogonal initialization
8. Larger batch size for stability
9. More evaluation games for better curriculum signals
10. Relaxed action sparsity constraint

Run:
    python user/train_simplified_OPTIMIZED.py
"""

# ============================================================================
# IMPORTS
# ============================================================================

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
from collections import deque
from functools import partial
import gymnasium as gym

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environment.agent import *

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


class ResidualMLPBlock(nn.Module):
    """Enhanced residual MLP block with LayerNorm and GELU."""

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


class WarehouseFeatureExtractor(BaseFeaturesExtractor):
    """Enhanced feature extractor with deeper architecture."""

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        feature_dim: int = 512,
        num_residual_blocks: int = 5,  # INCREASED from 3
        dropout: float = 0.08,  # INCREASED from 0.05
    ):
        super().__init__(observation_space, features_dim=feature_dim)
        input_dim = int(np.prod(observation_space.shape))

        self.preprocess = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, feature_dim),
            nn.GELU(),
        )
        self.residual_stack = nn.Sequential(
            *[ResidualMLPBlock(feature_dim, expansion=3, dropout=dropout) for _ in range(num_residual_blocks)]
        )
        self.output_norm = nn.LayerNorm(feature_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs.float()
        x = self.preprocess(x)
        x = self.residual_stack(x)
        return self.output_norm(x)


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
# TRAINING CONFIGURATION - OPTIMIZED
# ============================================================================

CHECKPOINT_DIR = "checkpoints/simplified_training_OPTIMIZED"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
TENSORBOARD_DIR = Path(CHECKPOINT_DIR) / "tb_logs"
TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)

_LSTM_KWARGS = {
    "dropout": 0.1,
}
if "layer_norm" in inspect.signature(nn.LSTM).parameters:
    _LSTM_KWARGS["layer_norm"] = True
else:
    print("⚠ PyTorch LSTM does not support layer_norm; continuing without it.")

# Agent hyperparameters - OPTIMIZED
AGENT_CONFIG = {
    "policy_kwargs": {
        "activation_fn": nn.GELU,
        "lstm_hidden_size": 512,
        "n_lstm_layers": 3,
        "net_arch": dict(pi=[512, 256], vf=[512, 256]),
        "shared_lstm": False,
        "enable_critic_lstm": True,
        "share_features_extractor": True,
        "features_extractor_class": WarehouseFeatureExtractor,
        "features_extractor_kwargs": {
            "feature_dim": 512,
            "num_residual_blocks": 5,  # INCREASED from 3
            "dropout": 0.08,  # INCREASED from 0.05
        },
        "optimizer_class": torch.optim.AdamW,
        "optimizer_kwargs": {
            "weight_decay": 1e-4,
            "eps": 1e-5,
        },
        "ortho_init": True,  # CHANGED from False - proven to help!
        "log_std_init": -0.5,
        "lstm_kwargs": _LSTM_KWARGS,
    },

    # PPO training - OPTIMIZED
    "n_steps": 4096,
    "batch_size": 1024,  # INCREASED from 512 for more stable updates
    "n_epochs": 6,
    "learning_rate": 3e-4,
    "min_learning_rate": 3e-5,  # INCREASED from 1e-5 to keep learning
    "ent_coef": 0.02,  # INCREASED from 0.005 for more exploration!
    "clip_range": 0.2,
    "clip_range_final": 0.1,  # INCREASED from 0.05
    "gamma": 0.995,
    "gae_lambda": 0.98,
    "max_grad_norm": 1.5,  # INCREASED from 0.5
    "vf_coef": 0.5,
    "clip_range_vf": 0.2,  # REDUCED from 0.3 for stability
    "use_sde": True,
    "sde_sample_freq": 4,
    "target_kl": 0.035,  # INCREASED from 0.015 for more flexible updates
}

TRAINING_CONFIG = {
    "total_timesteps": 3_000_000,
    "save_freq": 50_000,
    "resolution": CameraResolution.LOW,
    "n_envs": 4,
}

CURRICULUM_CONFIG = {
    "target_win_rate": 0.75,
    "min_probability": 0.05,
    "adaptation_strength": 0.6,
    "self_play_weight": 0.0,  # START at 0% - increase as model improves!
    "self_play_max_weight": 0.50,  # Max self-play weight once model is good
    "self_play_warmup_steps": 200_000,  # Start increasing self-play after this
    "self_play_enable_winrate": 0.50,  # Enable self-play once we hit 50% overall win rate
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

# Action patterns
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

# Opponent mix - REBALANCED for more self-play
OPPONENT_MIX = {
    "constant_agent": (0.10, partial(ConstantAgent)),  # Reduced from 0.15
    "based_agent": (0.15, partial(BasedAgent)),  # Reduced from 0.20
    "random_agent": (0.08, partial(RandomAgent)),  # Reduced from 0.10
    "aggressive_clockwork": (0.12, partial(ClockworkAgent, action_sheet=AGGRESSIVE_PATTERN)),
    "defensive_clockwork": (0.08, partial(ClockworkAgent, action_sheet=DEFENSIVE_PATTERN)),
    "hitrun_clockwork": (0.08, partial(ClockworkAgent, action_sheet=HIT_AND_RUN_PATTERN)),
    "aerial_clockwork": (0.08, partial(ClockworkAgent, action_sheet=AERIAL_PATTERN)),
    "special_clockwork": (0.08, partial(ClockworkAgent, action_sheet=SPECIAL_SPAM_PATTERN)),
}

print("=" * 70)
print("OPTIMIZED TRAINING CONFIGURATION")
print("=" * 70)
print(f"Architecture: LSTM-only RecurrentPPO (MlpLstmPolicy)")
print(f"Training steps: {TRAINING_CONFIG['total_timesteps']:,}")
print(f"Global seed: {GLOBAL_SEED}")
print(f"Checkpoint dir: {CHECKPOINT_DIR}")
print(f"\nKEY OPTIMIZATIONS:")
print(f"  • Entropy coef: 0.005 → 0.02 (4x exploration)")
print(f"  • Win reward: 30 → 100 (dominates damage trading)")
print(f"  • Batch size: 512 → 1024 (2x stability)")
print(f"  • Feature blocks: 3 → 5 (deeper feature extraction)")
print(f"  • Self-play: Curriculum-based (0% → 50%)")
print(f"    - Disabled until 200k steps + 50% win rate")
print(f"  • Target KL: 0.015 → 0.035 (more flexible updates)")
print(f"  • Ortho init: False → True (better gradient flow)")
print(f"  • Grad norm: 0.5 → 1.5 (larger policy updates)")
print("=" * 70)

# ============================================================================
# REWARD FUNCTIONS - OPTIMIZED
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


def distance_control_reward(env: WarehouseBrawl) -> float:
    """NEW: Reward maintaining optimal fighting distance

    Encourages strategic positioning rather than mindless rushing.
    """
    player = env.objects["player"]
    opponent = env.objects["opponent"]

    distance = float(np.linalg.norm(
        np.array([player.body.position.x, player.body.position.y]) -
        np.array([opponent.body.position.x, opponent.body.position.y])
    ))

    # Optimal fighting range: 2-4 units
    optimal_min, optimal_max = 2.0, 4.5

    if optimal_min < distance < optimal_max:
        reward = 0.02  # Small reward for good spacing
    elif distance > 6.0:
        reward = -0.015  # Penalty for being too far
    elif distance < 1.0:
        reward = -0.01  # Small penalty for being too close (vulnerable)
    else:
        reward = 0.0

    stats = _get_diag_stats(env)
    stats['distance_reward'] += reward
    return reward


def on_win_reward(env: WarehouseBrawl, agent: str) -> float:
    """OPTIMIZED: Much stronger win reward to dominate other signals

    Win reward MUST be the dominant signal to force opponent-specific strategies.
    With 4096 steps * 0.75 damage weight, accumulated damage can be ~200+
    Win reward of 100 ensures winning is 3-5x more important than damage optimization.
    """
    return 100.0 if agent == 'player' else -100.0  # INCREASED from 30


def on_knockout_reward(env: WarehouseBrawl, agent: str) -> float:
    """Reward knocking out opponent, penalize getting knocked out"""
    return 8.0 if agent == 'opponent' else -8.0  # INCREASED from 5


def action_sparsity_reward(env: WarehouseBrawl, max_active: int = 3, penalty_per_key: float = 0.005) -> float:
    """RELAXED: Allow up to 3 keys (some optimal combos need 3)"""
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
    """OPTIMIZED reward manager with WIN-DOMINANT rewards and strategic positioning"""
    reward_functions = {
        'danger_zone': RewTerm(func=danger_zone_reward, weight=0.2),
        'damage_interaction': RewTerm(func=damage_interaction_reward, weight=0.75),
        'distance_control': RewTerm(func=distance_control_reward, weight=0.5),  # NEW!
        'action_sparsity': RewTerm(func=action_sparsity_reward, weight=1.0),
    }
    signal_subscriptions = {
        'on_win': ('win_signal', RewTerm(func=on_win_reward, weight=1.0)),  # 100 points!
        'on_knockout': ('knockout_signal', RewTerm(func=on_knockout_reward, weight=1.0)),  # 8 points
    }
    return RewardManager(reward_functions, signal_subscriptions)


# ============================================================================
# SELF-PLAY HANDLER
# ============================================================================

class SimpleSelfPlayHandler:
    def __init__(self, ckpt_dir: str):
        self.ckpt_dir = ckpt_dir
        self.env = None

    def get_opponent(self) -> Agent:
        import glob
        import random

        zips = glob.glob(os.path.join(self.ckpt_dir, "rl_model_*.zip"))
        if not zips:
            # Fallback if no checkpoints yet - MUST call get_env_info!
            opponent = ConstantAgent()
            if self.env:
                opponent.get_env_info(self.env)
            return opponent

        path = random.choice(zips)
        opponent = RecurrentPPOAgent(file_path=path)
        if self.env:
            opponent.get_env_info(self.env)
        return opponent


def _make_self_play_env(seed: int, env_index: int) -> SelfPlayWarehouseBrawl:
    seed_everything(seed)

    reward_manager = gen_reward_manager()

    self_play_handler = SimpleSelfPlayHandler(CHECKPOINT_DIR)
    opponents_dict = {**OPPONENT_MIX}
    opponents_dict["self_play"] = (CURRICULUM_CONFIG["self_play_weight"], self_play_handler)

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

    self_play_handler.env = env
    reward_manager.subscribe_signals(env.raw_env)
    opponent_cfg.base_probabilities = {
        name: (value if isinstance(value, float) else value[0])
        for name, value in opponent_cfg.opponents.items()
    }

    env.self_play_handler = self_play_handler
    env.env_index = env_index
    return env


def _make_vec_env(num_envs: int) -> Tuple[VecNormalize, List[SelfPlayWarehouseBrawl]]:
    def make_thunk(rank: int) -> Callable[[], SelfPlayWarehouseBrawl]:
        def _init():
            env_seed = GLOBAL_SEED + rank * 9973
            return _make_self_play_env(env_seed, rank)
        return _init

    env_fns = [make_thunk(i) for i in range(num_envs)]

    try:
        vec_env = SubprocVecEnv(env_fns, start_method="spawn")
        print(f"✓ SubprocVecEnv initialized with {num_envs} workers")
    except Exception as exc:
        print(f"⚠ SubprocVecEnv creation failed ({exc}). Falling back to DummyVecEnv.")
        vec_env = DummyVecEnv(env_fns)

    vec_env = VecMonitor(vec_env)
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=5.0,
        clip_reward=10.0,
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
    print("STARTING OPTIMIZED TRAINING")
    print("=" * 70 + "\n")

    vec_env, env_instances = _make_vec_env(TRAINING_CONFIG["n_envs"])
    primary_env = env_instances[0]

    print(f"✓ Environment created ({len(env_instances)} parallel arenas)")
    print(f"  Observation dim: {primary_env.observation_space.shape[0]}")
    print(f"  LSTM handles temporal patterns over raw obs\n")

    policy_kwargs = {
        **AGENT_CONFIG["policy_kwargs"],
    }

    lr_schedule = linear_schedule(
        AGENT_CONFIG["learning_rate"],
        AGENT_CONFIG["min_learning_rate"],
    )
    clip_schedule = linear_schedule(
        AGENT_CONFIG["clip_range"],
        AGENT_CONFIG["clip_range_final"],
    )

    model = RecurrentPPO(
        "MlpLstmPolicy",
        vec_env,
        verbose=1,
        n_steps=AGENT_CONFIG["n_steps"],
        batch_size=AGENT_CONFIG["batch_size"],
        n_epochs=AGENT_CONFIG["n_epochs"],
        learning_rate=lr_schedule,
        ent_coef=AGENT_CONFIG["ent_coef"],
        clip_range=clip_schedule,
        gamma=AGENT_CONFIG["gamma"],
        gae_lambda=AGENT_CONFIG["gae_lambda"],
        max_grad_norm=AGENT_CONFIG["max_grad_norm"],
        vf_coef=AGENT_CONFIG["vf_coef"],
        clip_range_vf=AGENT_CONFIG["clip_range_vf"],
        target_kl=AGENT_CONFIG["target_kl"],
        use_sde=AGENT_CONFIG["use_sde"],
        sde_sample_freq=AGENT_CONFIG["sde_sample_freq"],
        policy_kwargs=policy_kwargs,
        device=DEVICE,
        tensorboard_log=str(TENSORBOARD_DIR),
        seed=GLOBAL_SEED,
    )

    print(f"✓ RecurrentPPO model created on {DEVICE}\n")

    from stable_baselines3.common.callbacks import CheckpointCallback

    class TrainingMonitor(CheckpointCallback):
        def __init__(self, *args, env=None, env_instances=None, eval_freq=20_000, eval_games=10, **kwargs):
            self.env_group = list(env_instances) if env_instances else []
            if env is not None and (not self.env_group or env is not self.env_group[0]):
                self.env_group.insert(0, env)
            self.primary_env = self.env_group[0] if self.env_group else env
            self.env_ref = self.primary_env
            super().__init__(*args, **kwargs)
            self.episode_rewards = []
            self.episode_lengths = []
            self.episode_count = 0
            self.eval_freq = eval_freq
            self.eval_games = eval_games  # INCREASED from 5 to 10
            self.last_eval_step = 0
            self._seen_ep_ids = deque()
            self._seen_ep_id_set = set()
            self.last_episode_summary = None
            self.last_opponent_probs = None

        def evaluate_against_all_opponents(self):
            print(f"\n{'='*70}")
            print(f"RUNNING EVALUATION @ {self.num_timesteps:,} steps")
            print(f"{'='*70}")

            temp_model_path = os.path.join(CHECKPOINT_DIR, f"temp_eval_{self.num_timesteps}.zip")
            self.model.save(temp_model_path)
            eval_agent = RecurrentPPOAgent(file_path=temp_model_path)

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

            if os.path.exists(temp_model_path):
                os.remove(temp_model_path)

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
            if self.primary_env is None:
                return

            cfg = getattr(self.primary_env, "opponent_cfg", None)
            if cfg is None or not hasattr(cfg, "opponents"):
                return

            # === DYNAMIC SELF-PLAY SCHEDULING ===
            # Calculate overall win rate
            total_wins = sum(stats['wins'] for stats in eval_results.values())
            total_games = sum(stats['total'] for stats in eval_results.values())
            overall_win_rate = (total_wins / total_games) if total_games > 0 else 0.0

            # Compute target self-play weight based on progress
            warmup_steps = CURRICULUM_CONFIG["self_play_warmup_steps"]
            enable_winrate = CURRICULUM_CONFIG["self_play_enable_winrate"]
            max_sp_weight = CURRICULUM_CONFIG["self_play_max_weight"]

            if self.num_timesteps < warmup_steps or overall_win_rate < enable_winrate:
                # Before warmup or below threshold: keep self-play at 0
                target_self_play_weight = 0.0
            else:
                # After warmup and above threshold: gradually increase self-play
                progress = (self.num_timesteps - warmup_steps) / (TRAINING_CONFIG["total_timesteps"] - warmup_steps)
                progress = min(1.0, max(0.0, progress))
                # Also scale by how much we exceed the threshold
                performance_factor = min(1.0, (overall_win_rate - enable_winrate) / 0.25)  # Full weight at 75%+
                target_self_play_weight = max_sp_weight * progress * performance_factor

            # Update self-play probability in base_probabilities
            base_probs = getattr(cfg, "base_probabilities", None)
            if base_probs is None:
                base_probs = {
                    name: value[0] if isinstance(value, tuple) else value
                    for name, value in cfg.opponents.items()
                }
                cfg.base_probabilities = base_probs

            if "self_play" in base_probs:
                old_sp_weight = base_probs["self_play"]
                base_probs["self_play"] = target_self_play_weight
                if abs(target_self_play_weight - old_sp_weight) > 0.01:
                    print(f"[CURRICULUM] Self-play weight: {old_sp_weight:.3f} → {target_self_play_weight:.3f} "
                          f"(WR: {overall_win_rate:.1%}, Step: {self.num_timesteps:,})")

            prev_probs = {
                name: value[0] if isinstance(value, tuple) else value
                for name, value in cfg.opponents.items()
            }

            new_opponents: Dict[str, Tuple[float, Any]] = {}
            updated_probabilities: Dict[str, float] = {}
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
                updated_probabilities[name] = adjusted_prob

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

            if len(self.env_group) > 1:
                for env_instance in self.env_group[1:]:
                    other_cfg = getattr(env_instance, "opponent_cfg", None)
                    if other_cfg is None or not hasattr(other_cfg, "opponents"):
                        continue
                    aligned: Dict[str, Tuple[float, Any]] = {}
                    for name, value in other_cfg.opponents.items():
                        if isinstance(value, tuple):
                            _, agent_factory = value
                        else:
                            agent_factory = None

                        prob_value = updated_probabilities.get(
                            name,
                            value[0] if isinstance(value, tuple) else value
                        )
                        if name == "self_play" and hasattr(env_instance, "self_play_handler"):
                            agent_factory = env_instance.self_play_handler
                        aligned[name] = (prob_value, agent_factory)
                    other_cfg.opponents = aligned
                    other_cfg.validate_probabilities()
                    other_cfg.base_probabilities = {
                        name: val[0] if isinstance(val, tuple) else val
                        for name, val in other_cfg.opponents.items()
                    }

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

            if self.num_timesteps >= self.last_eval_step + self.eval_freq and self.num_timesteps > 0:
                self.last_eval_step = self.num_timesteps
                self.evaluate_against_all_opponents()

            if self.n_calls % 1000 == 0:
                print(f"\n{'='*70}")
                print(f"TRAINING UPDATE @ {self.num_timesteps:,} steps")
                print(f"{'='*70}")

                if self.episode_rewards:
                    recent_rewards = self.episode_rewards[-100:]
                    print(f"\n[PERFORMANCE]")
                    print(f"  Episodes completed: {self.episode_count}")
                    print(f"  Avg Reward (last 100 ep): {safe_mean(recent_rewards):.2f}")
                    print(f"  Reward Std: {np.std(recent_rewards):.2f}")

                if self.last_episode_summary:
                    stats = self.last_episode_summary
                    print(f"\n[EPISODE DIAGNOSTICS]")
                    print(f"  Damage dealt: {stats.get('damage_dealt', 0):.1f}")
                    print(f"  Damage taken: {stats.get('damage_taken', 0):.1f}")
                    print(f"  Time in danger zone (s): {stats.get('zone_time', 0):.1f}")
                    print(f"  Distance control reward: {stats.get('distance_reward', 0):.2f}")
                    print(f"  Sparsity penalty: {stats.get('sparsity_penalty', 0):.2f}")
                    print(f"  Episode reward: {stats.get('reward', 0):.2f}")

                print(f"\n[LEARNING]")

                if hasattr(self.model, "ent_coef"):
                    progress_remaining = max(
                        0.0, 1.0 - (self.num_timesteps / TRAINING_CONFIG["total_timesteps"])
                    )
                    # GENTLER entropy decay (from 0.02 to 0.005 instead of to 0.001)
                    target_ent = float(np.clip(
                        AGENT_CONFIG["ent_coef"] * (0.25 + 0.75 * progress_remaining),
                        5e-3,  # Min 0.005 (not 0.0005)
                        AGENT_CONFIG["ent_coef"],
                    ))
                    if abs(float(getattr(self.model, "ent_coef", target_ent)) - target_ent) > 1e-6:
                        self.model.ent_coef = target_ent
                    print(f"  Entropy Coef: {float(self.model.ent_coef):.5f}")

                if hasattr(self.model, 'logger') and self.model.logger:
                    try:
                        if hasattr(self.model.logger, 'name_to_value'):
                            if 'train/policy_loss' in self.model.logger.name_to_value:
                                policy_loss = self.model.logger.name_to_value['train/policy_loss']
                                print(f"  Policy Loss: {policy_loss:.4f}")
                            if 'train/value_loss' in self.model.logger.name_to_value:
                                value_loss = self.model.logger.name_to_value['train/value_loss']
                                print(f"  Value Loss: {value_loss:.4f}")
                            if 'train/entropy_loss' in self.model.logger.name_to_value:
                                entropy_loss = self.model.logger.name_to_value['train/entropy_loss']
                                print(f"  Entropy Loss: {entropy_loss:.4f} {'✓' if entropy_loss < -0.01 else '⚠️ LOW exploration'}")
                    except:
                        pass

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
        env=primary_env,
        env_instances=env_instances,
        eval_freq=25_000,
        eval_games=5,  # INCREASED from 5
        save_freq=TRAINING_CONFIG["save_freq"],
        save_path=CHECKPOINT_DIR,
        name_prefix="rl_model",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    print("🚀 Training started\n")
    print("Version 3.1 - OPTIMIZED LSTM RecurrentPPO with Curriculum Self-Play")
    print("="*70)
    print("  - 4x more exploration (ent_coef: 0.02)")
    print("  - 3.3x stronger win signal (win reward: 100)")
    print("  - Deeper feature extraction (5 residual blocks)")
    print("  - Curriculum self-play (0% → 50% as model improves)")
    print("    • Disabled until 200k steps AND 50% win rate")
    print("    • Gradually increases to 50% by end of training")
    print("  - Strategic positioning rewards")
    print("  - More flexible policy updates (target_kl: 0.035)")
    print("="*70 + "\n")

    model.learn(
        total_timesteps=TRAINING_CONFIG["total_timesteps"],
        callback=training_callback,
        log_interval=1,
    )

    final_path = os.path.join(CHECKPOINT_DIR, "final_model.zip")
    model.save(final_path)
    vecnorm_path = os.path.join(CHECKPOINT_DIR, "vecnormalize_final.pkl")
    vec_env.save(vecnorm_path)

    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Final model saved to: {final_path}")
    print(f"Checkpoints saved in: {CHECKPOINT_DIR}")
    print(f"VecNormalize stats saved to: {vecnorm_path}")
    print("\nTo continue training, load the checkpoint and resume!")
    print("=" * 70 + "\n")

    vec_env.close()


if __name__ == "__main__":
    train()
