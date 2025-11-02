#!/usr/bin/env python3
"""
Pre-Flight Check Script for Overnight Training

This script verifies that all systems are ready for a long overnight training run.
It checks:
- GPU/device availability and performance
- Disk space for checkpoints
- Logging infrastructure
- Reward function sanity
- Training loop functionality
- Estimated training time

Run this BEFORE starting your overnight training!

Usage:
    python user/pre_flight_check.py
"""

import os
import sys
import shutil
import time
import torch
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print("\n" + "="*80)
print("PRE-FLIGHT CHECK FOR OVERNIGHT TRAINING".center(80))
print("="*80 + "\n")

# Track all checks
checks_passed = []
checks_failed = []


def check_passed(name: str, message: str = ""):
    """Mark a check as passed."""
    checks_passed.append(name)
    status = "✓"
    print(f"{status} {name}")
    if message:
        print(f"  {message}")


def check_failed(name: str, message: str, fix: str = ""):
    """Mark a check as failed."""
    checks_failed.append(name)
    status = "✗"
    print(f"{status} {name}")
    print(f"  ERROR: {message}")
    if fix:
        print(f"  FIX: {fix}")


def check_warning(name: str, message: str):
    """Show a warning."""
    status = "⚠"
    print(f"{status} {name}")
    print(f"  WARNING: {message}")


# ============================================================================
# 1. CHECK DEVICE AVAILABILITY
# ============================================================================

print("1. Checking Device Availability...")
print("-" * 80)

try:
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        check_passed(
            "CUDA GPU Available",
            f"{device_name} with {total_memory:.1f} GB memory"
        )

        # Test GPU speed with a simple operation
        start = time.time()
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.matmul(x, x)
        torch.cuda.synchronize()
        elapsed = time.time() - start

        if elapsed < 0.1:
            check_passed("GPU Performance", f"Matrix multiply: {elapsed*1000:.1f}ms (GOOD)")
        else:
            check_warning("GPU Performance", f"Matrix multiply: {elapsed*1000:.1f}ms (slower than expected)")

    elif torch.backends.mps.is_available():
        check_passed("Apple Silicon MPS Available", "Using MPS (slower than CUDA)")
        check_warning("MPS Device", "Training will be slower than CUDA. Expect 12-18 hours for 5M steps.")
    else:
        check_failed(
            "No GPU Available",
            "Only CPU available - training will be VERY slow (24-48 hours)",
            "Consider using a machine with CUDA GPU for practical training"
        )
except Exception as e:
    check_failed("Device Check Failed", str(e))

print()

# ============================================================================
# 2. CHECK DISK SPACE
# ============================================================================

print("2. Checking Disk Space...")
print("-" * 80)

try:
    # Check /tmp for strategy_encoder_training
    total, used, free = shutil.disk_usage("/")
    free_gb = free / (1024**3)

    # Need ~2GB for 5M steps (5 checkpoints + population + logs)
    required_gb = 2.0

    if free_gb >= required_gb:
        check_passed(
            "Disk Space",
            f"{free_gb:.1f} GB available (need {required_gb:.1f} GB)"
        )
    elif free_gb >= 1.0:
        check_warning(
            "Disk Space",
            f"Only {free_gb:.1f} GB available (recommended {required_gb:.1f} GB). May run out during training."
        )
    else:
        check_failed(
            "Insufficient Disk Space",
            f"Only {free_gb:.1f} GB available (need {required_gb:.1f} GB)",
            "Free up disk space or change CHECKPOINT_DIR in training script"
        )

    # Check if checkpoint dir exists and is writable
    checkpoint_dir = Path("/tmp/strategy_encoder_training")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    test_file = checkpoint_dir / ".write_test"
    try:
        test_file.write_text("test")
        test_file.unlink()
        check_passed("Checkpoint Directory Writable", str(checkpoint_dir))
    except Exception as e:
        check_failed("Checkpoint Directory Not Writable", str(e))

except Exception as e:
    check_failed("Disk Space Check Failed", str(e))

print()

# ============================================================================
# 3. CHECK PYTHON DEPENDENCIES
# ============================================================================

print("3. Checking Python Dependencies...")
print("-" * 80)

required_packages = [
    ('torch', 'PyTorch'),
    ('numpy', 'NumPy'),
    ('stable_baselines3', 'Stable Baselines3'),
    ('sb3_contrib', 'SB3 Contrib'),
]

optional_packages = [
    ('tensorboard', 'TensorBoard (for live monitoring)'),
]

all_deps_ok = True
for package, name in required_packages:
    try:
        __import__(package)
        check_passed(f"{name} installed")
    except ImportError:
        check_failed(f"{name} missing", f"Install with: pip install {package}")
        all_deps_ok = False

for package, name in optional_packages:
    try:
        __import__(package)
        check_passed(f"{name} installed")
    except ImportError:
        check_warning(f"{name} not installed", f"Install with: pip install {package}")

print()

# ============================================================================
# 4. TEST REWARD FUNCTIONS
# ============================================================================

print("4. Testing Reward Functions...")
print("-" * 80)

try:
    from environment.WarehouseBrawl import WarehouseBrawl
    from environment.agent import ConstantAgent

    # Create test environment
    env = WarehouseBrawl()
    env.reset()

    # Run a few steps and collect rewards
    rewards = []
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        rewards.append(reward)
        if done:
            env.reset()

    rewards = np.array(rewards)
    nonzero = rewards[rewards != 0]

    if len(nonzero) > 0:
        check_passed(
            "Reward Functions Working",
            f"Mean: {rewards.mean():.3f}, Std: {rewards.std():.3f}, Non-zero: {len(nonzero)}/{len(rewards)}"
        )
    else:
        check_warning(
            "All Rewards Zero",
            "Reward functions may not be configured correctly"
        )

except Exception as e:
    check_failed("Reward Function Test Failed", str(e))

print()

# ============================================================================
# 5. TEST TRAINING LOOP (SHORT RUN)
# ============================================================================

print("5. Testing Training Loop (10k steps, ~2 minutes)...")
print("-" * 80)

try:
    print("  Starting 10k step test run...")
    print("  This will verify logging, checkpointing, and basic training works.")
    print()

    start_time = time.time()

    # Import and run a short training test
    from user.train_with_strategy_encoder import (
        _make_vec_env,
        CHECKPOINT_DIR,
        TRAINING_CONFIG,
        AGENT_CONFIG,
        DEVICE,
        get_device,
    )
    from sb3_contrib import RecurrentPPO

    # Ensure device is initialized
    if DEVICE is None:
        DEVICE = get_device()

    # Create test checkpoint dir
    test_dir = Path("/tmp/preflight_test")
    test_dir.mkdir(parents=True, exist_ok=True)

    # Create small environment (2 envs for speed)
    vec_env, _ = _make_vec_env(2, test_dir, use_multiprocessing=False)

    # Create minimal model
    test_config = {k: v for k, v in AGENT_CONFIG.items()}
    test_config['n_steps'] = 512  # Smaller rollouts
    test_config['batch_size'] = 128

    model = RecurrentPPO(
        **test_config,
        env=vec_env,
        verbose=0,
        device=DEVICE,
    )

    # Train for 10k steps
    model.learn(total_timesteps=10_000, progress_bar=False, log_interval=None)

    elapsed = time.time() - start_time
    fps = 10_000 / elapsed

    # Clean up
    del model
    del vec_env
    import gc
    gc.collect()

    check_passed(
        "Training Loop Working",
        f"10k steps completed in {elapsed:.0f}s ({fps:.0f} FPS)"
    )

    # Estimate full training time
    total_steps = 5_000_000
    estimated_time_sec = total_steps / fps
    estimated_hours = estimated_time_sec / 3600

    print()
    print(f"  📊 ESTIMATED TRAINING TIME FOR 5M STEPS:")
    print(f"     {estimated_hours:.1f} hours ({estimated_hours/24:.1f} days)")
    print()

    if estimated_hours < 12:
        print(f"  ✓ Fast training speed! Should complete overnight.")
    elif estimated_hours < 24:
        print(f"  ○ Moderate speed. Will complete in ~{estimated_hours:.0f} hours.")
    else:
        print(f"  ⚠ Slow training speed. Consider using a faster GPU.")

except KeyboardInterrupt:
    print("\n  Test interrupted by user (Ctrl+C)")
    check_failed("Training Loop Test", "Interrupted by user")
except Exception as e:
    check_failed("Training Loop Test Failed", str(e))

print()

# ============================================================================
# 6. CHECK LOGGING INFRASTRUCTURE
# ============================================================================

print("6. Checking Logging Infrastructure...")
print("-" * 80)

try:
    # Check if CSV export path is writable
    csv_path = Path("/tmp/strategy_encoder_training/training_metrics.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Test write
    with open(csv_path, 'w') as f:
        f.write("test\n")

    if csv_path.exists():
        csv_path.unlink()
        check_passed("CSV Export Path Writable", str(csv_path))

    # Check tensorboard
    try:
        import tensorboard
        tb_dir = Path("/tmp/strategy_encoder_training/tb_logs")
        tb_dir.mkdir(parents=True, exist_ok=True)
        check_passed("TensorBoard Available", f"Logs will be saved to {tb_dir}")
        print(f"     To monitor: tensorboard --logdir {tb_dir}")
    except ImportError:
        check_warning("TensorBoard Not Installed", "Install with: pip install tensorboard")

except Exception as e:
    check_failed("Logging Infrastructure Check Failed", str(e))

print()

# ============================================================================
# SUMMARY
# ============================================================================

print("="*80)
print("PRE-FLIGHT CHECK SUMMARY".center(80))
print("="*80)

total_checks = len(checks_passed) + len(checks_failed)
print(f"\nTotal Checks: {total_checks}")
print(f"✓ Passed: {len(checks_passed)}")
print(f"✗ Failed: {len(checks_failed)}")

if len(checks_failed) == 0:
    print("\n" + "="*80)
    print("🚀 ALL SYSTEMS GO! 🚀".center(80))
    print("You are ready to start overnight training!".center(80))
    print("="*80)
    print("\nTo start training:")
    print("  1. Start a tmux/screen session: tmux new -s training")
    print("  2. Run: python user/train_with_strategy_encoder.py")
    print("  3. Detach with: Ctrl+B, then D (tmux) or Ctrl+A, then D (screen)")
    print("  4. Monitor progress: tail -f /tmp/strategy_encoder_training/training_*.log")
    print()
else:
    print("\n" + "="*80)
    print("⚠ FIX ISSUES BEFORE TRAINING ⚠".center(80))
    print("="*80)
    print("\nFailed checks:")
    for check in checks_failed:
        print(f"  ✗ {check}")
    print("\nPlease fix the above issues before starting training.")
    print()

print("="*80 + "\n")

# Exit with error code if any checks failed
sys.exit(len(checks_failed))
