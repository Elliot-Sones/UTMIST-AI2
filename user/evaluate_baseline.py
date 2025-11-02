#!/usr/bin/env python3
"""
Baseline Evaluation Script

Evaluates a trained agent against all opponent types to establish performance baseline.
Useful for:
- Testing current model before overnight training
- Comparing performance before/after training
- Understanding which opponents the agent struggles against

Usage:
    python user/evaluate_baseline.py --model checkpoints/simplified_training/latest_model.zip
    python user/evaluate_baseline.py --model /tmp/strategy_encoder_training/latest_model.zip
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from functools import partial

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environment.agent import (
    ConstantAgent, BasedAgent, RandomAgent, ClockworkAgent
)
from environment.WarehouseBrawl import SelfPlayWarehouseBrawl, CameraResolution
from sb3_contrib import RecurrentPPO


# Opponent action patterns
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


def evaluate_against_opponent(model, opponent_class, opponent_name, num_episodes=10):
    """
    Evaluate the model against a specific opponent.

    Args:
        model: Trained RecurrentPPO model
        opponent_class: Opponent agent class or callable
        opponent_name: Name of the opponent for display
        num_episodes: Number of episodes to run

    Returns:
        dict: Evaluation results (wins, losses, avg_reward, etc.)
    """
    from environment.WarehouseBrawl import WarehouseBrawl

    wins = 0
    losses = 0
    draws = 0
    total_reward = 0
    total_damage_dealt = 0
    total_damage_taken = 0
    episode_lengths = []

    for episode in range(num_episodes):
        # Create environment
        env = WarehouseBrawl(resolution=CameraResolution.LOW)
        env.reset()

        # Set opponent
        if callable(opponent_class):
            env.objects["opponent"] = opponent_class()
        else:
            env.objects["opponent"] = opponent_class

        # Reset for this episode
        obs = env.reset()
        done = False
        episode_reward = 0
        lstm_states = None
        episode_start = np.array([True])
        step_count = 0
        max_steps = 30 * 90  # 90 seconds at 30 FPS

        while not done and step_count < max_steps:
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_start,
                deterministic=True
            )
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            episode_start = np.array([False])
            step_count += 1

        # Record results
        total_reward += episode_reward
        episode_lengths.append(step_count)

        # Determine winner
        player = env.objects["player"]
        opponent = env.objects["opponent"]

        if player.stocks > opponent.stocks:
            wins += 1
        elif opponent.stocks > player.stocks:
            losses += 1
        else:
            # Same stocks, check damage
            if player.damage_taken_total < opponent.damage_taken_total:
                wins += 1
            elif player.damage_taken_total > opponent.damage_taken_total:
                losses += 1
            else:
                draws += 1

        total_damage_dealt += opponent.damage_taken_total
        total_damage_taken += player.damage_taken_total

        env.close()

    # Compute statistics
    win_rate = wins / num_episodes
    avg_reward = total_reward / num_episodes
    avg_damage_dealt = total_damage_dealt / num_episodes
    avg_damage_taken = total_damage_taken / num_episodes
    avg_episode_length = np.mean(episode_lengths)

    return {
        'opponent': opponent_name,
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'win_rate': win_rate,
        'avg_reward': avg_reward,
        'avg_damage_dealt': avg_damage_dealt,
        'avg_damage_taken': avg_damage_taken,
        'avg_damage_diff': avg_damage_dealt - avg_damage_taken,
        'avg_episode_length': avg_episode_length,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate agent baseline performance')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=10, help='Episodes per opponent (default: 10)')
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Error: Model not found at {model_path}")
        sys.exit(1)

    print("\n" + "="*80)
    print("BASELINE EVALUATION".center(80))
    print("="*80)
    print(f"Model: {model_path}")
    print(f"Episodes per opponent: {args.episodes}")
    print()

    # Load model
    print("Loading model...")
    try:
        model = RecurrentPPO.load(model_path)
        model.policy.set_training_mode(False)
        print("✓ Model loaded successfully\n")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)

    # Define opponents to test against
    opponents = [
        (ConstantAgent, "ConstantAgent"),
        (BasedAgent, "BasedAgent"),
        (RandomAgent, "RandomAgent"),
        (partial(ClockworkAgent, action_sheet=AGGRESSIVE_PATTERN), "Clockwork-Aggressive"),
        (partial(ClockworkAgent, action_sheet=DEFENSIVE_PATTERN), "Clockwork-Defensive"),
        (partial(ClockworkAgent, action_sheet=HIT_AND_RUN_PATTERN), "Clockwork-HitAndRun"),
        (partial(ClockworkAgent, action_sheet=AERIAL_PATTERN), "Clockwork-Aerial"),
        (partial(ClockworkAgent, action_sheet=SPECIAL_SPAM_PATTERN), "Clockwork-Special"),
    ]

    # Evaluate against each opponent
    results = []
    print("="*80)
    print("EVALUATION RESULTS".center(80))
    print("="*80)
    print(f"{'Opponent':<25} {'W-L-D':<12} {'Win%':<8} {'Avg Dmg':<12} {'Avg Reward':<12}")
    print("-"*80)

    for opp_class, opp_name in opponents:
        print(f"Testing vs {opp_name}...", end=' ')
        sys.stdout.flush()

        result = evaluate_against_opponent(model, opp_class, opp_name, args.episodes)
        results.append(result)

        # Print result
        wld = f"{result['wins']}-{result['losses']}-{result['draws']}"
        win_pct = result['win_rate'] * 100
        dmg_diff = result['avg_damage_diff']
        avg_rew = result['avg_reward']

        status = "✓" if win_pct >= 60 else "○" if win_pct >= 40 else "✗"
        print(f"\r{status} {opp_name:<23} {wld:<12} {win_pct:>6.1f}% "
              f"{dmg_diff:>+10.1f} {avg_rew:>12.1f}")

    # Overall summary
    print("-"*80)
    overall_wins = sum(r['wins'] for r in results)
    overall_total = sum(r['wins'] + r['losses'] + r['draws'] for r in results)
    overall_win_rate = overall_wins / overall_total if overall_total > 0 else 0

    avg_reward = np.mean([r['avg_reward'] for r in results])
    avg_damage_diff = np.mean([r['avg_damage_diff'] for r in results])

    print(f"\n{'OVERALL PERFORMANCE':<25} {overall_wins}/{overall_total} "
          f"{overall_win_rate*100:>6.1f}% {avg_damage_diff:>+10.1f} {avg_reward:>12.1f}")
    print("="*80)

    # Performance breakdown
    print("\nPERFORMANCE BREAKDOWN:")
    print("-"*80)

    strong_opponents = [r for r in results if r['win_rate'] >= 0.6]
    medium_opponents = [r for r in results if 0.4 <= r['win_rate'] < 0.6]
    weak_opponents = [r for r in results if r['win_rate'] < 0.4]

    print(f"Strong Against ({len(strong_opponents)}): ", end='')
    print(", ".join([r['opponent'] for r in strong_opponents]) if strong_opponents else "None")

    print(f"Competitive Against ({len(medium_opponents)}): ", end='')
    print(", ".join([r['opponent'] for r in medium_opponents]) if medium_opponents else "None")

    print(f"Weak Against ({len(weak_opponents)}): ", end='')
    print(", ".join([r['opponent'] for r in weak_opponents]) if weak_opponents else "None")

    print()

    # Assessment
    print("ASSESSMENT:")
    print("-"*80)

    if overall_win_rate >= 0.7:
        print("🏆 EXCELLENT: Agent is highly competitive! Ready for competition.")
    elif overall_win_rate >= 0.5:
        print("✓ GOOD: Agent is competitive. Could benefit from more training.")
    elif overall_win_rate >= 0.3:
        print("○ FAIR: Agent is learning but needs more training.")
    elif overall_win_rate >= 0.1:
        print("⚠ POOR: Agent is struggling. Check reward functions and training config.")
    else:
        print("✗ VERY POOR: Agent is barely learning. Major issues likely.")

    print("="*80 + "\n")


if __name__ == "__main__":
    main()
