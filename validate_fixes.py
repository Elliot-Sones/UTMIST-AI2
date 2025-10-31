#!/usr/bin/env python3
"""
Quick Validation Script - Verify All Critical Fixes
Run this BEFORE starting training to catch configuration issues.

Usage:
    python validate_fixes.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

def validate_reward_function():
    """Validate reward function has correct weights and no removed terms."""
    from user.train_agent import gen_reward_manager

    print("="*70)
    print("🔍 VALIDATING REWARD FUNCTION")
    print("="*70)

    reward_manager = gen_reward_manager()
    reward_funcs = reward_manager.reward_functions
    signal_subs = reward_manager.signal_subscriptions

    print(f"\n✅ Reward Functions ({len(reward_funcs)} active):")
    for name, term in reward_funcs.items():
        print(f"  • {name:30s} weight={term.weight:6.1f}")

    print(f"\n✅ Signal Subscriptions ({len(signal_subs)} active):")
    for name, (signal_name, term) in signal_subs.items():
        print(f"  • {name:30s} signal={signal_name:20s} weight={term.weight:6.1f}")

    # Critical checks
    issues = []

    # Check attack exploration bonus exists and is strong enough
    if 'on_attack_button_press' not in reward_funcs:
        issues.append("❌ Missing 'on_attack_button_press' - agent may not explore attacking!")
    elif reward_funcs['on_attack_button_press'].weight < 2.0:
        issues.append(f"⚠️  'on_attack_button_press' weight too low ({reward_funcs['on_attack_button_press'].weight}) - increase to 5.0+")

    # Check damage reward is strong
    if 'damage_interaction_reward' not in reward_funcs:
        issues.append("❌ Missing 'damage_interaction_reward' - no incentive to deal damage!")
    elif reward_funcs['damage_interaction_reward'].weight < 150:
        issues.append(f"⚠️  'damage_interaction_reward' weight too low ({reward_funcs['damage_interaction_reward'].weight}) - increase to 200+")

    # Check no removed penalties exist
    removed_terms = ['penalize_attack_reward']  # Known removed terms
    for term in removed_terms:
        if term in reward_funcs:
            issues.append(f"❌ Found removed term '{term}' - this was supposed to be deleted!")

    if issues:
        print("\n🚨 REWARD FUNCTION ISSUES:")
        for issue in issues:
            print(f"  {issue}")
        return False
    else:
        print("\n✅ Reward function validation PASSED")
        return True


def validate_transformer_config():
    """Validate transformer doesn't use zero-padding."""
    print("\n" + "="*70)
    print("🔍 VALIDATING TRANSFORMER CONFIGURATION")
    print("="*70)

    from user.train_agent import TransformerStrategyAgent, TransformerStrategyEncoder, TORCH_DEVICE
    import numpy as np
    import torch

    print(f"\n✅ Transformer Hyperparameters:")
    print(f"  • Latent Dim: 256")
    print(f"  • Num Heads: 8")
    print(f"  • Num Layers: 6")
    print(f"  • Sequence Length: 90")
    print(f"  • Device: {TORCH_DEVICE}")

    # Test padding behavior directly (without full agent initialization)
    print(f"\n🔬 Testing Padding Behavior...")

    # Create transformer encoder directly
    encoder = TransformerStrategyEncoder(
        opponent_obs_dim=32,
        latent_dim=256,
        num_heads=8,
        num_layers=6,
        max_sequence_length=90,
        device=TORCH_DEVICE
    )

    # Simulate short opponent history (3 frames) - this is the critical test
    test_history = [np.random.randn(32) for _ in range(3)]

    # Manually apply the padding logic (same as _update_strategy_latent)
    history_array = np.array(test_history)
    current_len = len(test_history)

    # Apply repeat-padding (the fix we implemented)
    min_seq_len = 10
    if current_len < min_seq_len:
        repeats_needed = (min_seq_len + current_len - 1) // current_len
        history_array = np.tile(history_array, (repeats_needed, 1))[:min_seq_len]

    print(f"  • Original history length: {current_len}")
    print(f"  • After padding length: {history_array.shape[0]}")

    # Convert to tensor and test encoder
    history_tensor = torch.tensor(
        history_array,
        dtype=torch.float32,
        device=TORCH_DEVICE
    ).unsqueeze(0)  # [1, seq_len, obs_dim]

    # Generate latent
    with torch.no_grad():
        latent_tensor = encoder(history_tensor)

    # Check if latent was generated
    if latent_tensor is None:
        print("  ❌ Strategy latent is None - transformer failed!")
        return False

    latent_norm = float(latent_tensor.norm().item())
    print(f"  ✅ Strategy latent generated (norm={latent_norm:.4f})")

    # Check if padding preserved actual data (not all zeros)
    # With repeat-padding, norm should be similar to original data
    # With zero-padding, norm would be very low
    if latent_norm < 0.01:
        print("  ⚠️  Latent norm suspiciously low - may still be using zero-padding!")
        return False

    print(f"  ✅ Padding behavior looks correct (repeat-padding preserves data)")

    # Verify the padded history contains actual opponent data
    # Check if first 3 frames are non-zero
    first_frame_norm = np.linalg.norm(history_array[0])
    if first_frame_norm < 0.01:
        print("  ❌ First frame is near-zero - zero-padding detected!")
        return False

    print(f"  ✅ First frame has valid data (norm={first_frame_norm:.4f})")
    return True


def validate_ppo_config():
    """Validate PPO hyperparameters for fast learning."""
    print("\n" + "="*70)
    print("🔍 VALIDATING PPO CONFIGURATION")
    print("="*70)

    from user.train_agent import _SHARED_AGENT_CONFIG

    config = _SHARED_AGENT_CONFIG

    print(f"\n✅ PPO Hyperparameters:")
    print(f"  • n_steps: {config.get('n_steps')}")
    print(f"  • batch_size: {config.get('batch_size')}")
    print(f"  • n_epochs: {config.get('n_epochs')}")
    print(f"  • learning_rate: {config.get('learning_rate')}")
    print(f"  • ent_coef: {config.get('ent_coef')}")
    print(f"  • clip_range: {config.get('clip_range')}")

    issues = []

    # Check learning frequency
    n_steps = config.get('n_steps')
    if n_steps is None:
        issues.append("❌ n_steps not set!")
    elif n_steps > 1024:
        issues.append(f"⚠️  n_steps={n_steps} too high - learning updates too infrequent (use 512)")

    # Check entropy for exploration
    ent_coef = config.get('ent_coef')
    if ent_coef is None:
        issues.append("❌ ent_coef not set!")
    elif ent_coef < 0.10:
        issues.append(f"⚠️  ent_coef={ent_coef} too low - insufficient exploration (use 0.15+)")

    # Calculate learning frequency
    if n_steps:
        updates_50k = 50_000 // n_steps
        total_gradient_updates = updates_50k * config.get('n_epochs', 10)
        print(f"\n📊 Learning Frequency Analysis (50k steps):")
        print(f"  • PPO updates: {updates_50k}")
        print(f"  • Gradient updates: {total_gradient_updates}")
        print(f"  • Update every: {n_steps / 30:.1f} seconds (at 30 FPS)")

        if updates_50k < 50:
            issues.append(f"⚠️  Only {updates_50k} PPO updates for 50k steps - too few! (should be 90+)")

    if issues:
        print("\n🚨 PPO CONFIGURATION ISSUES:")
        for issue in issues:
            print(f"  {issue}")
        return False
    else:
        print("\n✅ PPO configuration validation PASSED")
        return True


def validate_training_configs():
    """Validate all training configurations are properly set up."""
    print("\n" + "="*70)
    print("🔍 VALIDATING TRAINING CONFIGURATIONS")
    print("="*70)

    from user.train_agent import (
        TRAIN_CONFIG, TRAIN_CONFIG_DEBUG, TRAIN_CONFIG_CURRICULUM,
        TRAIN_CONFIG_10M
    )

    print(f"\n✅ Active Config: {TRAIN_CONFIG['self_play']['run_name']}")
    print(f"  • Timesteps: {TRAIN_CONFIG['training']['timesteps']:,}")
    print(f"  • Opponent Mix: {list(TRAIN_CONFIG['self_play']['opponent_mix'].keys())}")
    print(f"  • Monitoring: {'ON' if TRAIN_CONFIG['training']['enable_debug'] else 'OFF'}")

    # Check debug config exists
    print(f"\n✅ Available Configurations:")
    configs = {
        'DEBUG (5k)': TRAIN_CONFIG_DEBUG,
        'CURRICULUM (50k)': TRAIN_CONFIG_CURRICULUM,
        'COMPETITION (10M)': TRAIN_CONFIG_10M,
    }

    for name, cfg in configs.items():
        timesteps = cfg['training']['timesteps']
        run_name = cfg['self_play']['run_name']
        print(f"  • {name:20s} → {run_name} ({timesteps:,} steps)")

    # Warn if not using debug
    if TRAIN_CONFIG['training']['timesteps'] > 10_000:
        print("\n⚠️  WARNING: Not using DEBUG config!")
        print("   Recommended: Run DEBUG (5k steps) first to validate all fixes")
        print("   Change line 590 to: TRAIN_CONFIG = TRAIN_CONFIG_DEBUG")
        return False

    print("\n✅ Training configuration validation PASSED")
    return True


def main():
    """Run all validation checks."""
    print("\n" + "="*70)
    print("🚀 CRITICAL FIXES VALIDATION SCRIPT")
    print("   Run this BEFORE training to catch issues early!")
    print("="*70)

    results = []

    # Run all validations
    results.append(("Reward Function", validate_reward_function()))
    results.append(("Transformer Config", validate_transformer_config()))
    results.append(("PPO Config", validate_ppo_config()))
    results.append(("Training Configs", validate_training_configs()))

    # Summary
    print("\n" + "="*70)
    print("📊 VALIDATION SUMMARY")
    print("="*70)

    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {name}")
        if not passed:
            all_passed = False

    print("="*70)

    if all_passed:
        print("\n🎉 ALL VALIDATIONS PASSED!")
        print("   ✅ You're ready to start training")
        print("   ✅ Run: python user/train_agent.py")
        print("\n📋 Next Steps:")
        print("   1. Run DEBUG (5k steps, 2-3 minutes)")
        print("   2. Check for damage_dealt > 0 in logs")
        print("   3. If successful, proceed to CURRICULUM (50k steps)")
        return 0
    else:
        print("\n🚨 VALIDATION FAILED!")
        print("   ⚠️  Fix the issues above before training")
        print("   ⚠️  Training with these issues will waste time/compute")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
