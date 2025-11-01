"""
Training Metrics Callback: Comprehensive logging for training visibility.

This callback provides detailed console logging of:
- Episode rewards and statistics
- Win/loss rates
- Damage dealt vs taken
- KL divergence and entropy (policy health)
- Learning rate and clip range schedules
- Opponent distribution
"""

import numpy as np
import time
from typing import Dict, Any, List
from collections import deque
from stable_baselines3.common.callbacks import BaseCallback


class TrainingMetricsCallback(BaseCallback):
    """
    Callback that logs detailed training metrics to console.

    Provides real-time visibility into:
    - Episode performance (rewards, wins, damage)
    - Training stability (KL, entropy, loss)
    - Learning progress (moving averages)
    - Opponent statistics
    """

    def __init__(
        self,
        log_frequency: int = 10,  # Log every N rollouts
        moving_avg_window: int = 100,  # Window for moving averages
        verbose: int = 1,
    ):
        super().__init__(verbose)

        self.log_frequency = log_frequency
        self.moving_avg_window = moving_avg_window

        # Episode statistics
        self.episode_rewards = deque(maxlen=moving_avg_window)
        self.episode_lengths = deque(maxlen=moving_avg_window)
        self.episode_wins = deque(maxlen=moving_avg_window)

        # Damage statistics
        self.damage_dealt = deque(maxlen=moving_avg_window)
        self.damage_taken = deque(maxlen=moving_avg_window)

        # Training metrics
        self.last_log_step = 0
        self.start_time = None
        self.last_log_time = None

        # Rollout counter
        self.num_rollouts = 0

    def _on_training_start(self) -> None:
        """Called at the beginning of training."""
        self.start_time = time.time()
        self.last_log_time = self.start_time

        print("\n" + "="*80)
        print("TRAINING STARTED".center(80))
        print("="*80)
        print(f"{'Timestep':<12} {'Rollout':<8} {'Reward':<12} {'Win%':<8} "
              f"{'Dmg+/-':<15} {'FPS':<10} {'Time':<10}")
        print("-"*80)

    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout."""
        self.num_rollouts += 1

        # Collect episode info from rollout buffer if available
        if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
            # Extract episode statistics
            for ep_info in self.model.ep_info_buffer:
                if 'r' in ep_info:
                    self.episode_rewards.append(ep_info['r'])
                if 'l' in ep_info:
                    self.episode_lengths.append(ep_info['l'])

        # Check if we should log
        if self.num_rollouts % self.log_frequency == 0:
            self._log_metrics()

    def _on_step(self) -> bool:
        """Called at each environment step."""
        # Collect info from environments
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                # Check for episode end
                if 'episode' in info:
                    episode_info = info['episode']
                    self.episode_rewards.append(episode_info['r'])
                    self.episode_lengths.append(episode_info['l'])

                # Track wins (if available in info)
                if 'is_success' in info or 'win' in info:
                    win = info.get('is_success', info.get('win', 0))
                    self.episode_wins.append(float(win))

                # Track damage statistics (if available)
                if 'damage_dealt' in info:
                    self.damage_dealt.append(info['damage_dealt'])
                if 'damage_taken' in info:
                    self.damage_taken.append(info['damage_taken'])

        return True  # Continue training

    def _log_metrics(self):
        """Log comprehensive training metrics."""
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        time_since_last_log = current_time - self.last_log_time

        timesteps = self.num_timesteps

        # Compute FPS
        steps_since_last = timesteps - self.last_log_step
        fps = steps_since_last / time_since_last_log if time_since_last_log > 0 else 0

        # Compute statistics
        avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0.0
        avg_length = np.mean(self.episode_lengths) if self.episode_lengths else 0.0
        win_rate = np.mean(self.episode_wins) if self.episode_wins else 0.0

        avg_dmg_dealt = np.mean(self.damage_dealt) if self.damage_dealt else 0.0
        avg_dmg_taken = np.mean(self.damage_taken) if self.damage_taken else 0.0
        dmg_diff = avg_dmg_dealt - avg_dmg_taken

        # Format time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        # Print compact metrics line
        print(f"{timesteps:<12,d} {self.num_rollouts:<8d} "
              f"{avg_reward:<12.2f} {win_rate*100:<8.1f} "
              f"{dmg_diff:>+7.1f}({avg_dmg_dealt:.0f}/{avg_dmg_taken:.0f}) "
              f"{fps:<10.0f} {time_str:<10}")

        # Every 5 logs, print detailed statistics
        if self.num_rollouts % (self.log_frequency * 5) == 0:
            self._log_detailed_stats()

        self.last_log_step = timesteps
        self.last_log_time = current_time

    def _log_detailed_stats(self):
        """Log detailed training statistics."""
        print("\n" + "-"*80)
        print(f"DETAILED STATS @ {self.num_timesteps:,} steps".center(80))
        print("-"*80)

        # Episode statistics
        if self.episode_rewards:
            print(f"Episodes: {len(self.episode_rewards)} completed")
            print(f"  Reward:  avg={np.mean(self.episode_rewards):.2f}  "
                  f"std={np.std(self.episode_rewards):.2f}  "
                  f"min={np.min(self.episode_rewards):.2f}  "
                  f"max={np.max(self.episode_rewards):.2f}")

        if self.episode_lengths:
            print(f"  Length:  avg={np.mean(self.episode_lengths):.0f}  "
                  f"std={np.std(self.episode_lengths):.0f}")

        if self.episode_wins:
            win_rate = np.mean(self.episode_wins)
            print(f"  Win Rate: {win_rate*100:.1f}% ({np.sum(self.episode_wins):.0f}/{len(self.episode_wins)})")

        # Damage statistics
        if self.damage_dealt and self.damage_taken:
            print(f"\nDamage Stats:")
            print(f"  Dealt: avg={np.mean(self.damage_dealt):.1f}  "
                  f"max={np.max(self.damage_dealt):.1f}")
            print(f"  Taken: avg={np.mean(self.damage_taken):.1f}  "
                  f"max={np.max(self.damage_taken):.1f}")
            print(f"  Net:   avg={np.mean(self.damage_dealt) - np.mean(self.damage_taken):+.1f}")

        # Training metrics from logger (if available)
        if hasattr(self.logger, 'name_to_value'):
            metrics = self.logger.name_to_value
            if 'train/policy_loss' in metrics:
                print(f"\nTraining Metrics:")
                print(f"  Policy Loss:     {metrics.get('train/policy_loss', 0):.4f}")
                print(f"  Value Loss:      {metrics.get('train/value_loss', 0):.4f}")
                print(f"  Entropy:         {metrics.get('train/entropy_loss', 0):.4f}")
                print(f"  KL Divergence:   {metrics.get('train/approx_kl', 0):.6f}")
                print(f"  Clip Fraction:   {metrics.get('train/clip_fraction', 0):.3f}")

                # Learning rate info
                if 'train/learning_rate' in metrics:
                    print(f"  Learning Rate:   {metrics.get('train/learning_rate', 0):.2e}")
                if 'train/clip_range' in metrics:
                    print(f"  Clip Range:      {metrics.get('train/clip_range', 0):.3f}")

        print("-"*80 + "\n")

    def _on_training_end(self) -> None:
        """Called at the end of training."""
        total_time = time.time() - self.start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)

        print("\n" + "="*80)
        print("TRAINING COMPLETED".center(80))
        print("="*80)
        print(f"Total timesteps: {self.num_timesteps:,}")
        print(f"Total rollouts:  {self.num_rollouts:,}")
        print(f"Total time:      {hours}h {minutes}m")
        print(f"Avg FPS:         {self.num_timesteps / total_time:.0f}")

        if self.episode_rewards:
            print(f"\nFinal Performance (last {len(self.episode_rewards)} episodes):")
            print(f"  Avg Reward:  {np.mean(self.episode_rewards):.2f}")
            if self.episode_wins:
                print(f"  Win Rate:    {np.mean(self.episode_wins)*100:.1f}%")
            if self.damage_dealt and self.damage_taken:
                print(f"  Avg Damage:  {np.mean(self.damage_dealt) - np.mean(self.damage_taken):+.1f}")

        print("="*80 + "\n")


def create_training_metrics_callback(
    log_frequency: int = 10,
    moving_avg_window: int = 100,
    verbose: int = 1,
) -> TrainingMetricsCallback:
    """
    Factory function to create TrainingMetricsCallback.

    Args:
        log_frequency: Log metrics every N rollouts (default: 10)
        moving_avg_window: Window size for moving averages (default: 100)
        verbose: Verbosity level (default: 1)

    Returns:
        TrainingMetricsCallback instance
    """
    return TrainingMetricsCallback(
        log_frequency=log_frequency,
        moving_avg_window=moving_avg_window,
        verbose=verbose,
    )
