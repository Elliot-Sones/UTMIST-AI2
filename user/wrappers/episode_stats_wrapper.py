"""
Episode Statistics Wrapper: Tracks and reports episode-level stats.

This wrapper extracts diagnostic information from the environment and
adds it to the info dict for logging by callbacks.
"""

import numpy as np
from typing import Any, Dict
from stable_baselines3.common.vec_env import VecEnvWrapper


class EpisodeStatsWrapper(VecEnvWrapper):
    """
    Wrapper that tracks episode statistics and adds them to info dict.

    Tracks:
    - Damage dealt/taken per episode
    - Win/loss outcomes
    - Episode rewards
    """

    def __init__(self, venv):
        super().__init__(venv)
        self.num_envs = venv.num_envs

        # Per-environment episode statistics
        self.episode_damage_dealt = np.zeros(self.num_envs)
        self.episode_damage_taken = np.zeros(self.num_envs)
        self.episode_wins = np.zeros(self.num_envs)
        self.episode_started = np.zeros(self.num_envs, dtype=bool)

    def reset(self):
        """Reset all environments."""
        obs = self.venv.reset()

        # Reset all episode stats
        self.episode_damage_dealt[:] = 0
        self.episode_damage_taken[:] = 0
        self.episode_wins[:] = 0
        self.episode_started[:] = True

        return obs

    def step_wait(self):
        """Step all environments and collect statistics."""
        obs, rewards, dones, infos = self.venv.step_wait()

        # Process each environment
        for i in range(self.num_envs):
            # Try to extract diagnostic stats from base environment
            # Navigate through wrapper layers to find base env
            base_env = self._get_base_env(i)

            if base_env is not None and hasattr(base_env, '_diag_stats'):
                stats = base_env._diag_stats

                # Accumulate damage for this episode
                if 'damage_dealt' in stats:
                    self.episode_damage_dealt[i] = stats['damage_dealt']
                if 'damage_taken' in stats:
                    self.episode_damage_taken[i] = stats['damage_taken']

            # Check for episode termination
            if dones[i]:
                if self.episode_started[i]:
                    # Check if player won (reward > 50 is a heuristic for win)
                    # This is approximate since we don't have direct win signal yet
                    win = 1.0 if rewards[i] > 50.0 else 0.0
                    self.episode_wins[i] = win

                    # Add episode statistics to info dict
                    if not isinstance(infos[i], dict):
                        infos[i] = {}

                    infos[i]['win'] = win
                    infos[i]['damage_dealt'] = float(self.episode_damage_dealt[i])
                    infos[i]['damage_taken'] = float(self.episode_damage_taken[i])

                    # Reset episode stats for this environment
                    self.episode_damage_dealt[i] = 0
                    self.episode_damage_taken[i] = 0
                    self.episode_started[i] = False
            else:
                # Mark episode as started if it wasn't already
                if not self.episode_started[i]:
                    self.episode_started[i] = True

        return obs, rewards, dones, infos

    def _get_base_env(self, env_idx: int):
        """
        Navigate through wrapper layers to find the base environment.

        Args:
            env_idx: Index of the environment

        Returns:
            Base environment or None if not found
        """
        # Try to access unwrapped environments
        unwrapped = self.venv

        while hasattr(unwrapped, 'venv'):
            unwrapped = unwrapped.venv

        # Check if we have access to individual environments
        if hasattr(unwrapped, 'envs'):
            if env_idx < len(unwrapped.envs):
                return unwrapped.envs[env_idx]

        return None
