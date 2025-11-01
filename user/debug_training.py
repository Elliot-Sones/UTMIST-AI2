"""
COMPREHENSIVE TRAINING DEBUGGER
================================

This script runs diagnostic tests to identify why your model isn't learning.

Run this BEFORE training to catch issues early:
    python user/debug_training.py
"""

import os
import sys
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from functools import partial
import gymnasium as gym

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environment.agent import *
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

print("="*80)
print("TRAINING DEBUGGER - Finding Why Your Model Isn't Learning")
print("="*80)

# Import your training config
import train_simplified_OPTIMIZED as training_module

# ============================================================================
# TEST 1: OBSERVATION SPACE SANITY
# ============================================================================

print("\n[TEST 1] OBSERVATION SPACE DIAGNOSTICS")
print("-"*80)

# Create a single environment
reward_manager = training_module.gen_reward_manager()
opponent_cfg = OpponentsCfg(opponents={
    'constant_agent': (1.0, partial(ConstantAgent)),
})

env = SelfPlayWarehouseBrawl(
    reward_manager=reward_manager,
    opponent_cfg=opponent_cfg,
    save_handler=None,
    resolution=CameraResolution.LOW,
)

obs, info = env.reset()

print(f"✓ Observation shape: {obs.shape}")
print(f"✓ Observation dtype: {obs.dtype}")
print(f"✓ Observation range: [{obs.min():.3f}, {obs.max():.3f}]")
print(f"✓ Observation mean: {obs.mean():.3f}")
print(f"✓ Observation std: {obs.std():.3f}")

# Check for NaN/Inf
if np.any(np.isnan(obs)):
    print("❌ CRITICAL: Observations contain NaN!")
elif np.any(np.isinf(obs)):
    print("❌ CRITICAL: Observations contain Inf!")
else:
    print("✓ No NaN/Inf in observations")

# Check if observations change
obs_samples = []
for _ in range(10):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    obs_samples.append(obs.copy())
    if done or truncated:
        obs, info = env.reset()

obs_samples = np.array(obs_samples)
obs_variance = obs_samples.var(axis=0)

if obs_variance.max() < 1e-6:
    print("❌ WARNING: Observations barely change! (max variance: {:.2e})".format(obs_variance.max()))
    print("   → Model can't learn if observations don't vary")
else:
    print(f"✓ Observations vary appropriately (max variance: {obs_variance.max():.3f})")

# Check for constant features
constant_features = (obs_variance < 1e-6).sum()
if constant_features > 0:
    print(f"⚠️  WARNING: {constant_features}/{len(obs_variance)} observation features are constant")
    print("   → These features are useless for learning")

# ============================================================================
# TEST 2: ACTION SPACE & CONTROL
# ============================================================================

print("\n[TEST 2] ACTION SPACE & CONTROL DIAGNOSTICS")
print("-"*80)

obs, info = env.reset()

print(f"✓ Action space: {env.action_space}")
print(f"✓ Action shape: {env.action_space.shape}")

# Test if actions actually do something
obs_before, _ = env.reset()
action_zero = np.zeros(env.action_space.shape)
obs_after_zero, _, _, _, _ = env.step(action_zero)

obs_before2, _ = env.reset()
action_random = env.action_space.sample()
obs_after_random, _, _, _, _ = env.step(action_random)

diff_zero = np.abs(obs_after_zero - obs_before).max()
diff_random = np.abs(obs_after_random - obs_before2).max()

print(f"✓ Max obs change (zero action): {diff_zero:.6f}")
print(f"✓ Max obs change (random action): {diff_random:.6f}")

if diff_random < 1e-4:
    print("❌ CRITICAL: Actions don't seem to affect observations!")
    print("   → Model can't learn control if actions do nothing")
elif diff_random < diff_zero * 1.5:
    print("⚠️  WARNING: Random actions barely change obs more than zero action")
    print("   → Actions might not have enough effect")
else:
    print("✓ Actions appropriately affect observations")

# ============================================================================
# TEST 3: REWARD SIGNAL QUALITY
# ============================================================================

print("\n[TEST 3] REWARD SIGNAL DIAGNOSTICS")
print("-"*80)

rewards_collected = []
episode_lengths = []

for episode in range(20):
    obs, info = env.reset()
    episode_reward = 0
    steps = 0
    done = False

    while not done and steps < 500:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        steps += 1
        rewards_collected.append(reward)

        if done or truncated:
            break

    episode_lengths.append(steps)

rewards_array = np.array(rewards_collected)
print(f"✓ Collected {len(rewards_collected)} rewards from {len(episode_lengths)} episodes")
print(f"✓ Reward range: [{rewards_array.min():.3f}, {rewards_array.max():.3f}]")
print(f"✓ Reward mean: {rewards_array.mean():.3f}")
print(f"✓ Reward std: {rewards_array.std():.3f}")
print(f"✓ Non-zero rewards: {(rewards_array != 0).sum()}/{len(rewards_array)} ({(rewards_array != 0).mean()*100:.1f}%)")

# Check reward issues
if np.all(rewards_array == 0):
    print("❌ CRITICAL: All rewards are zero!")
    print("   → Model can't learn without reward signal")
elif (rewards_array != 0).mean() < 0.05:
    print("⚠️  WARNING: >95% of rewards are zero (very sparse)")
    print("   → Model will struggle to learn from such sparse signals")
elif rewards_array.std() < 0.01:
    print("⚠️  WARNING: Rewards have very low variance")
    print("   → Model might not distinguish good from bad actions")
else:
    print("✓ Reward signal appears adequate")

if np.any(np.isnan(rewards_array)):
    print("❌ CRITICAL: Rewards contain NaN!")
elif np.any(np.isinf(rewards_array)):
    print("❌ CRITICAL: Rewards contain Inf!")
else:
    print("✓ No NaN/Inf in rewards")

# ============================================================================
# TEST 4: ENVIRONMENT DETERMINISM
# ============================================================================

print("\n[TEST 4] ENVIRONMENT DETERMINISM")
print("-"*80)

# Run same action sequence twice
env.reset(seed=42)
traj1_obs = []
traj1_rewards = []
for _ in range(50):
    action = np.ones(env.action_space.shape) * 0.5
    obs, reward, done, truncated, info = env.step(action)
    traj1_obs.append(obs.copy())
    traj1_rewards.append(reward)
    if done or truncated:
        break

env.reset(seed=42)
traj2_obs = []
traj2_rewards = []
for _ in range(50):
    action = np.ones(env.action_space.shape) * 0.5
    obs, reward, done, truncated, info = env.step(action)
    traj2_obs.append(obs.copy())
    traj2_rewards.append(reward)
    if done or truncated:
        break

if len(traj1_obs) == len(traj2_obs):
    obs_diff = np.abs(np.array(traj1_obs) - np.array(traj2_obs)).max()
    reward_diff = np.abs(np.array(traj1_rewards) - np.array(traj2_rewards)).max()

    if obs_diff < 1e-5 and reward_diff < 1e-5:
        print("✓ Environment is deterministic (given same seed)")
    else:
        print(f"⚠️  WARNING: Environment not fully deterministic")
        print(f"   Max obs diff: {obs_diff:.6f}, Max reward diff: {reward_diff:.6f}")
        print("   → Training might be noisier than expected")
else:
    print("⚠️  WARNING: Episode lengths differ with same seed!")

# ============================================================================
# TEST 5: VALUE FUNCTION LEARNING
# ============================================================================

print("\n[TEST 5] VALUE FUNCTION GRADIENT FLOW")
print("-"*80)

# Create minimal model
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class TestFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, feature_dim: int = 256):
        super().__init__(observation_space, features_dim=feature_dim)
        input_dim = int(np.prod(observation_space.shape))
        self.net = nn.Sequential(
            nn.Linear(input_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs.float())

vec_env = DummyVecEnv([lambda: env])

try:
    test_model = RecurrentPPO(
        "MlpLstmPolicy",
        vec_env,
        verbose=0,
        n_steps=128,
        batch_size=64,
        learning_rate=3e-4,
        policy_kwargs={
            "lstm_hidden_size": 128,
            "n_lstm_layers": 1,
            "net_arch": dict(pi=[128], vf=[128]),
            "features_extractor_class": TestFeatureExtractor,
            "features_extractor_kwargs": {"feature_dim": 128},
        },
        device="cpu",
    )

    print("✓ Test model created successfully")

    # Collect some rollouts
    test_model.learn(total_timesteps=512, log_interval=None)

    # Check if parameters changed
    param_norms = []
    for name, param in test_model.policy.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            param_norms.append((name, grad_norm))

    if len(param_norms) == 0:
        print("❌ WARNING: No gradients computed!")
    else:
        max_grad_norm = max(grad_norm for _, grad_norm in param_norms)
        print(f"✓ Gradients computed (max norm: {max_grad_norm:.6f})")

        if max_grad_norm < 1e-6:
            print("❌ CRITICAL: Gradients are vanishingly small!")
            print("   → Model parameters won't update")
        elif max_grad_norm > 100:
            print("⚠️  WARNING: Very large gradients detected!")
            print("   → May cause instability")
        else:
            print("✓ Gradient magnitudes look reasonable")

    # Check value function predictions
    obs_sample = vec_env.reset()
    with torch.no_grad():
        obs_tensor = torch.FloatTensor(obs_sample).to("cpu")
        lstm_states = test_model.policy.lstm_actor.lstm.get_initial_hidden_states(1)
        features, _ = test_model.policy.extract_features(obs_tensor, lstm_states)
        values = test_model.policy.value_net(features)

    print(f"✓ Value function output: {values.item():.3f}")

    if abs(values.item()) < 1e-4:
        print("⚠️  WARNING: Value function outputs near zero")
    elif abs(values.item()) > 1000:
        print("⚠️  WARNING: Value function outputs very large values")
    else:
        print("✓ Value function outputs reasonable values")

except Exception as e:
    print(f"❌ ERROR creating/training test model: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# TEST 6: OPPONENT DIFFICULTY
# ============================================================================

print("\n[TEST 6] OPPONENT DIFFICULTY ANALYSIS")
print("-"*80)

opponent_types = {
    "ConstantAgent": partial(ConstantAgent),
    "RandomAgent": partial(RandomAgent),
    "BasedAgent": partial(BasedAgent),
}

for opp_name, opp_factory in opponent_types.items():
    opponent_cfg = OpponentsCfg(opponents={
        opp_name.lower(): (1.0, opp_factory),
    })

    test_env = SelfPlayWarehouseBrawl(
        reward_manager=training_module.gen_reward_manager(),
        opponent_cfg=opponent_cfg,
        save_handler=None,
        resolution=CameraResolution.LOW,
    )

    # Run random agent
    wins = 0
    total_reward = 0
    games = 10

    for _ in range(games):
        obs, info = test_env.reset()
        done = False
        ep_reward = 0

        while not done:
            action = test_env.action_space.sample()
            obs, reward, done, truncated, info = test_env.step(action)
            ep_reward += reward
            if done or truncated:
                break

        total_reward += ep_reward
        if ep_reward > 0:  # Rough heuristic for win
            wins += 1

    avg_reward = total_reward / games
    win_rate = wins / games * 100

    print(f"  {opp_name:15} - Random policy win rate: {win_rate:5.1f}%, Avg reward: {avg_reward:6.2f}")

    if opp_name == "ConstantAgent" and win_rate < 80:
        print(f"    ⚠️  WARNING: Random policy should beat ConstantAgent >80%")
    elif opp_name != "ConstantAgent" and win_rate > 60:
        print(f"    ⚠️  WARNING: Opponent might be too easy")

# ============================================================================
# TEST 7: NETWORK ARCHITECTURE CHECKS
# ============================================================================

print("\n[TEST 7] NETWORK ARCHITECTURE DIAGNOSTICS")
print("-"*80)

from train_simplified_OPTIMIZED import WarehouseFeatureExtractor

# Test feature extractor
obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=env.observation_space.shape)
feature_extractor = WarehouseFeatureExtractor(obs_space)

# Forward pass test
test_obs = torch.randn(4, *env.observation_space.shape)
try:
    features = feature_extractor(test_obs)
    print(f"✓ Feature extractor output shape: {features.shape}")
    print(f"✓ Feature extractor output range: [{features.min().item():.3f}, {features.max().item():.3f}]")

    if torch.any(torch.isnan(features)):
        print("❌ CRITICAL: Feature extractor outputs NaN!")
    elif torch.any(torch.isinf(features)):
        print("❌ CRITICAL: Feature extractor outputs Inf!")
    elif features.std().item() < 0.01:
        print("⚠️  WARNING: Feature extractor outputs have very low variance")
    else:
        print("✓ Feature extractor outputs look healthy")

    # Check gradient flow through feature extractor
    loss = features.sum()
    loss.backward()

    has_grad = False
    for param in feature_extractor.parameters():
        if param.grad is not None and param.grad.abs().max() > 1e-6:
            has_grad = True
            break

    if has_grad:
        print("✓ Gradients flow through feature extractor")
    else:
        print("❌ WARNING: No gradients through feature extractor!")

except Exception as e:
    print(f"❌ ERROR in feature extractor: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# TEST 8: REWARD FUNCTION BREAKDOWN
# ============================================================================

print("\n[TEST 8] REWARD FUNCTION COMPONENT ANALYSIS")
print("-"*80)

# Test each reward component
test_env = SelfPlayWarehouseBrawl(
    reward_manager=training_module.gen_reward_manager(),
    opponent_cfg=OpponentsCfg(opponents={'random': (1.0, partial(RandomAgent))}),
    save_handler=None,
    resolution=CameraResolution.LOW,
)

component_rewards = {
    'damage_interaction': [],
    'danger_zone': [],
    'distance_control': [],
    'action_sparsity': [],
}

obs, info = test_env.reset()
for _ in range(500):
    action = test_env.action_space.sample()
    obs, total_reward, done, truncated, info = test_env.step(action)

    # Access raw env to get component rewards
    if hasattr(test_env, 'raw_env') and hasattr(test_env.raw_env, '_diag_stats'):
        stats = test_env.raw_env._diag_stats
        # These are cumulative, so we'd need to track deltas
        # For now, just show that they exist

    if done or truncated:
        if hasattr(test_env.raw_env, '_diag_stats'):
            stats = test_env.raw_env._diag_stats
            print(f"\n  Episode stats:")
            print(f"    Damage dealt: {stats.get('damage_dealt', 0):.1f}")
            print(f"    Damage taken: {stats.get('damage_taken', 0):.1f}")
            print(f"    Danger zone time: {stats.get('zone_time', 0):.1f}s")
            print(f"    Distance reward: {stats.get('distance_reward', 0):.2f}")
            print(f"    Sparsity penalty: {stats.get('sparsity_penalty', 0):.2f}")
            print(f"    Total reward: {stats.get('reward', 0):.2f}")
        break

# ============================================================================
# SUMMARY & RECOMMENDATIONS
# ============================================================================

print("\n" + "="*80)
print("DIAGNOSIS SUMMARY")
print("="*80)

print("\n✓ Tests completed! Review warnings above.")
print("\nCOMMON ISSUES & FIXES:")
print("  1. If observations don't change → Check environment step function")
print("  2. If rewards are all zero → Check reward manager is connected")
print("  3. If gradients are tiny → Increase learning rate or check normalization")
print("  4. If random policy beats all opponents → Opponents too weak")
print("  5. If features have low variance → Observation normalization might be wrong")
print("\nNext steps:")
print("  • Review all ⚠️  warnings and ❌ errors above")
print("  • If environment issues found, fix those FIRST")
print("  • If learning issues found, adjust hyperparameters")
print("  • Run: python user/debug_training.py > debug_output.txt to save full output")
print("="*80)
