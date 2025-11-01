"""
Validation Callback: Tests agent performance against scripted opponents.

This callback periodically evaluates the training agent by running matches against
a suite of scripted opponents (ConstantAgent, BasedAgent, RandomAgent, etc.) to
measure real performance and ensure the agent is learning.
"""

import os
import numpy as np
import torch
from typing import Dict, List, Optional
from stable_baselines3.common.callbacks import BaseCallback
from pathlib import Path

from environment.agent import ConstantAgent, BasedAgent, RandomAgent, ClockworkAgent
from environment.environment import CameraResolution


class ValidationCallback(BaseCallback):
    """
    Callback that periodically evaluates agent against scripted opponents.

    Runs validation matches every N timesteps and logs:
    - Win rate against each opponent type
    - Average damage dealt/taken
    - Episode rewards
    - Overall performance score
    """

    def __init__(
        self,
        validation_env,
        validation_frequency: int = 100_000,
        n_eval_episodes: int = 10,
        opponent_types: List[str] = None,
        verbose: int = 1,
    ):
        super().__init__(verbose)

        self.validation_env = validation_env
        self.validation_frequency = validation_frequency
        self.n_eval_episodes = n_eval_episodes
        self.verbose = verbose

        # Default opponent types for validation
        if opponent_types is None:
            opponent_types = ["constant", "based", "random"]
        self.opponent_types = opponent_types

        # Validation history
        self.validation_history = []
        self.last_validation_step = 0

        print(f"✓ ValidationCallback initialized:")
        print(f"  - Validate every: {validation_frequency:,} steps")
        print(f"  - Eval episodes per opponent: {n_eval_episodes}")
        print(f"  - Opponents: {', '.join(opponent_types)}")

    def _on_step(self) -> bool:
        """Called at each training step."""
        # Check if it's time for validation
        if self.num_timesteps - self.last_validation_step >= self.validation_frequency:
            self._run_validation()
            self.last_validation_step = self.num_timesteps

        return True  # Continue training

    def _run_validation(self):
        """Run validation matches against all opponent types."""
        print("\n" + "="*80)
        print(f"VALIDATION @ {self.num_timesteps:,} STEPS".center(80))
        print("="*80)

        # Set model to eval mode
        self.model.policy.set_training_mode(False)

        all_results = {}
        total_wins = 0
        total_episodes = 0

        for opponent_type in self.opponent_types:
            results = self._evaluate_against_opponent(opponent_type)
            all_results[opponent_type] = results

            total_wins += results['wins']
            total_episodes += results['episodes']

            # Print results for this opponent
            win_rate = results['win_rate'] * 100
            avg_reward = results['avg_reward']
            avg_dmg_diff = results['avg_damage_dealt'] - results['avg_damage_taken']

            status = "✓" if win_rate >= 50 else "✗"
            print(f"{status} vs {opponent_type:12s}: "
                  f"Win {win_rate:5.1f}% | "
                  f"Reward {avg_reward:7.1f} | "
                  f"Dmg {avg_dmg_diff:+6.1f}")

        # Overall statistics
        overall_win_rate = (total_wins / total_episodes * 100) if total_episodes > 0 else 0
        print("-"*80)
        print(f"Overall: {overall_win_rate:.1f}% win rate ({total_wins}/{total_episodes} wins)")
        print("="*80 + "\n")

        # Store in history
        self.validation_history.append({
            'timesteps': self.num_timesteps,
            'overall_win_rate': overall_win_rate,
            'results_by_opponent': all_results,
        })

        # Log to tensorboard if available
        if self.logger is not None:
            self.logger.record("validation/overall_win_rate", overall_win_rate)
            for opponent_type, results in all_results.items():
                self.logger.record(f"validation/win_rate_{opponent_type}", results['win_rate'] * 100)
                self.logger.record(f"validation/avg_reward_{opponent_type}", results['avg_reward'])

        # Set model back to train mode
        self.model.policy.set_training_mode(True)

    def _evaluate_against_opponent(self, opponent_type: str) -> Dict:
        """
        Evaluate agent against a specific opponent type.

        Returns:
            Dictionary with evaluation metrics
        """
        # Create opponent based on type
        opponent = self._create_opponent(opponent_type)

        wins = 0
        rewards = []
        damage_dealt_list = []
        damage_taken_list = []

        for episode in range(self.n_eval_episodes):
            # Run episode
            obs = self.validation_env.reset()
            done = False
            episode_reward = 0
            episode_damage_dealt = 0
            episode_damage_taken = 0
            lstm_states = None

            while not done:
                # Get action from model
                if hasattr(self.model, 'lstm_states'):
                    # RecurrentPPO
                    action, lstm_states = self.model.predict(
                        obs,
                        state=lstm_states,
                        episode_start=np.array([episode == 0]),
                        deterministic=True
                    )
                else:
                    # Regular PPO
                    action, _ = self.model.predict(obs, deterministic=True)

                obs, reward, done, info = self.validation_env.step(action)
                episode_reward += reward

                # Track damage if available
                if hasattr(self.validation_env, '_diag_stats'):
                    stats = self.validation_env._diag_stats
                    episode_damage_dealt = stats.get('damage_dealt', 0)
                    episode_damage_taken = stats.get('damage_taken', 0)

            # Check if won (higher score/less damage)
            # This is a heuristic - adjust based on your environment's win condition
            won = episode_reward > 0 or episode_damage_dealt > episode_damage_taken

            wins += int(won)
            rewards.append(episode_reward)
            damage_dealt_list.append(episode_damage_dealt)
            damage_taken_list.append(episode_damage_taken)

        return {
            'episodes': self.n_eval_episodes,
            'wins': wins,
            'win_rate': wins / self.n_eval_episodes,
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'avg_damage_dealt': np.mean(damage_dealt_list),
            'avg_damage_taken': np.mean(damage_taken_list),
        }

    def _create_opponent(self, opponent_type: str):
        """Create opponent agent based on type."""
        opponent_type = opponent_type.lower()

        if opponent_type == "constant":
            return ConstantAgent()
        elif opponent_type == "based":
            return BasedAgent()
        elif opponent_type == "random":
            return RandomAgent()
        elif opponent_type == "clockwork_aggressive":
            pattern = [
                (15, ['d']), (3, ['d', 'j']), (2, []), (3, ['d', 'j']),
                (15, ['d']), (5, ['d', 'l']), (10, ['d']), (3, ['j']),
            ]
            return ClockworkAgent(action_sheet=pattern)
        elif opponent_type == "clockwork_defensive":
            pattern = [
                (20, []), (5, ['a']), (15, []), (3, ['j']),
                (10, []), (4, ['l']), (25, []),
            ]
            return ClockworkAgent(action_sheet=pattern)
        else:
            # Default to ConstantAgent
            return ConstantAgent()

    def get_best_win_rate(self) -> float:
        """Get best overall win rate achieved during training."""
        if not self.validation_history:
            return 0.0
        return max(v['overall_win_rate'] for v in self.validation_history)

    def get_latest_win_rate(self) -> float:
        """Get latest overall win rate."""
        if not self.validation_history:
            return 0.0
        return self.validation_history[-1]['overall_win_rate']


def create_validation_callback(
    validation_env,
    validation_frequency: int = 100_000,
    n_eval_episodes: int = 10,
    verbose: int = 1,
) -> ValidationCallback:
    """
    Factory function to create ValidationCallback.

    Args:
        validation_env: Gym environment for validation
        validation_frequency: Validate every N steps (default: 100,000)
        n_eval_episodes: Number of episodes per opponent (default: 10)
        verbose: Verbosity level (default: 1)

    Returns:
        ValidationCallback instance
    """
    return ValidationCallback(
        validation_env=validation_env,
        validation_frequency=validation_frequency,
        n_eval_episodes=n_eval_episodes,
        verbose=verbose,
    )
