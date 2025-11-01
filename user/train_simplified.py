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
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # Apple Silicon support
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from functools import partial

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

# (No encoder or observation wrapper required for LSTM-only setup)


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

# Where to save checkpoints
CHECKPOINT_DIR = "checkpoints/simplified_training"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Agent hyperparameters
AGENT_CONFIG = {
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
print(f"Architecture: LSTM-only RecurrentPPO (MlpLstmPolicy)")
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
    """Strong reward for winning - dominates other rewards without exploding value function
    Winning needs to be very valuable to force opponent-specific strategies."""
    return 30.0 if agent == 'player' else -30.0  # Reduced from 100 to prevent value explosion


def on_knockout_reward(env: WarehouseBrawl, agent: str) -> float:
    """Reward knocking out opponent, penalize getting knocked out"""
    return 5.0 if agent == 'opponent' else -5.0  # Reduced from 10


def gen_reward_manager():
    """Create reward manager with WIN-FOCUSED rewards

    Philosophy: Winning must be very valuable to force the model to
    learn opponent-specific strategies, but not so large it breaks value function.
    """
    reward_functions = {
        'danger_zone': RewTerm(func=danger_zone_reward, weight=0.2),
        'damage_interaction': RewTerm(func=damage_interaction_reward, weight=0.5),
    }
    signal_subscriptions = {
        'on_win': ('win_signal', RewTerm(func=on_win_reward, weight=1.0)),  # 30 points!
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

    print(f"✓ Environment created")
    print(f"  Observation dim: {env.observation_space.shape[0]}")
    print(f"  LSTM handles temporal patterns over raw obs\n")

    # Create policy kwargs (no custom feature extractor)
    policy_kwargs = {
        **AGENT_CONFIG["policy_kwargs"],
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
        def __init__(self, *args, env=None, eval_freq=10000, eval_games=10, **kwargs):
            self.env_ref = env  # Store env reference before calling super
            super().__init__(*args, **kwargs)
            self.episode_rewards = []
            self.episode_lengths = []
            self.episode_count = 0
            self.eval_freq = eval_freq  # Evaluate every N steps
            self.eval_games = eval_games  # Games per opponent during eval
            self.last_eval_step = 0

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
                        max_timesteps=30*90,
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

            return results, overall_wr

        def _on_step(self):
            # Track episode completion from buffer
            if hasattr(self, 'model') and hasattr(self.model, 'ep_info_buffer'):
                if len(self.model.ep_info_buffer) > 0:
                    for ep_info in self.model.ep_info_buffer:
                        if 'r' in ep_info and ep_info['r'] not in [r for r in self.episode_rewards[-10:]]:
                            self.episode_rewards.append(ep_info['r'])
                            self.episode_lengths.append(ep_info.get('l', 0))
                            self.episode_count += 1

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
        eval_freq=10_000,  # Evaluate every 10k steps
        eval_games=10,  # 10 games per opponent during evaluation
        save_freq=TRAINING_CONFIG["save_freq"],
        save_path=CHECKPOINT_DIR,
        name_prefix="rl_model",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    print("🚀 Training started\n")
    print("Version 3.0 - LSTM-only RecurrentPPO")
    print("="*70)
    print("  - No encoder or history wrapper")
    print("  - LSTM learns temporal patterns directly from raw obs")
    print("  - Stable reward shaping (win-focused)")
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
