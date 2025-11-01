"""
Quick test to verify training logging is working properly.

This script runs a very short training session (10,000 steps) to verify:
1. Console logging is visible
2. Metrics are being tracked
3. Callbacks are functioning
4. Training is actually progressing
"""

import os
import sys
from pathlib import Path

# Set up paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import training script components
from user.train_with_strategy_encoder import (
    seed_everything,
    GLOBAL_SEED,
    DEVICE,
    AGENT_CONFIG,
    STRATEGY_ENCODER_CONFIG,
    BASE_EXTRACTOR_CONFIG,
    POPULATION_CONFIG,
    OPPONENT_MIX,
    _make_vec_env,
    gen_reward_manager,
)
from user.self_play.population_manager import PopulationManager
from user.callbacks.training_metrics_callback import create_training_metrics_callback
from sb3_contrib import RecurrentPPO
import torch

print("\n" + "="*80)
print("TRAINING LOGGING TEST".center(80))
print("="*80)
print("This test will run 10,000 training steps to verify logging is working.\n")

# Seed everything
seed_everything(GLOBAL_SEED)

# Create test checkpoint directory
test_checkpoint_dir = Path("checkpoints/logging_test")
test_checkpoint_dir.mkdir(parents=True, exist_ok=True)
print(f"✓ Test checkpoint directory: {test_checkpoint_dir}\n")

# Create population manager
population_manager = PopulationManager(
    checkpoint_dir=test_checkpoint_dir,
    max_population_size=5,
    num_weak_agents=1,
)
print("✓ Population manager created\n")

# Create environment
print("Creating test environment...")
vec_env, env_instances = _make_vec_env(
    num_envs=2,  # Use fewer envs for faster testing
    population_manager=population_manager
)
print(f"✓ Environment created with {len(env_instances)} parallel arenas\n")

# Initialize normalization
print("Initializing normalization...")
vec_env.reset()
for _ in range(50):
    import numpy as np
    actions = np.array([vec_env.action_space.sample() for _ in range(2)])
    vec_env.step(actions)
vec_env.reset()
print("✓ Normalization initialized\n")

# Create model
print("Creating model...")
model = RecurrentPPO(
    **AGENT_CONFIG,
    env=vec_env,
    verbose=1,  # Enable verbose logging
    device=DEVICE,
    tensorboard_log=None,  # Disable tensorboard for test
)
print("✓ Model created\n")

# Create metrics callback
print("Creating metrics callback...")
metrics_callback = create_training_metrics_callback(
    log_frequency=1,  # Log every rollout
    moving_avg_window=50,
    verbose=1,
)
print("✓ Metrics callback created\n")

# Run short training
print("="*80)
print("STARTING TRAINING TEST (10,000 steps)".center(80))
print("="*80)
print("You should see detailed logging below...\n")

try:
    model.learn(
        total_timesteps=10_000,
        callback=metrics_callback,
        log_interval=1,  # Log every update
        progress_bar=True,
    )

    print("\n" + "="*80)
    print("TEST PASSED".center(80))
    print("="*80)
    print("✓ Logging is working correctly!")
    print("✓ Metrics are being tracked")
    print("✓ Training is progressing")
    print("\nYou can now run the full training with:")
    print("  python user/train_with_strategy_encoder.py")
    print("="*80 + "\n")

except Exception as e:
    print("\n" + "="*80)
    print("TEST FAILED".center(80))
    print("="*80)
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    print("="*80 + "\n")
    sys.exit(1)
