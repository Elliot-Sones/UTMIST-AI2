"""
Opponent History Buffer Wrapper: Tracks opponent behavior over time.

This wrapper maintains a sliding window of opponent observations to enable
strategy encoding. It extracts key opponent features (position, velocity,
move type, etc.) and stacks them into a history buffer that can be fed
into the strategy encoder.
"""

import numpy as np
from collections import deque
from typing import Any, Dict, Tuple, Optional
import gymnasium as gym
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper


class OpponentHistoryBuffer(VecEnvWrapper):
    """
    VecEnv wrapper that tracks opponent observation history for strategy encoding.

    Maintains a rolling buffer of the last N timesteps of opponent observations,
    extracting key features that characterize opponent behavior patterns.

    Args:
        venv: Vectorized environment to wrap
        history_length: Number of timesteps to track (default: 60 = 2 seconds at 30 FPS)
        opponent_features: List of feature names to extract from observations
    """

    def __init__(
        self,
        venv: VecEnv,
        history_length: int = 60,
        opponent_features: Optional[list] = None
    ):
        super().__init__(venv)

        self.history_length = history_length
        self.num_envs = venv.num_envs

        # Default opponent features to track (13 features total)
        if opponent_features is None:
            self.opponent_features = [
                'opponent_pos',         # 2D: x, y position
                'opponent_vel',         # 2D: x, y velocity
                'opponent_facing',      # 1D: direction
                'opponent_grounded',    # 1D: on ground?
                'opponent_aerial',      # 1D: in air?
                'opponent_jumps_left',  # 1D: jumps remaining
                'opponent_damage',      # 1D: damage taken
                'opponent_stocks',      # 1D: lives left
                'opponent_move_type',   # 1D: current move
                'opponent_state',       # 1D: current state
                'opponent_stun_frames', # 1D: stun duration
            ]
        else:
            self.opponent_features = opponent_features

        # Get observation helper from base environment
        # We need to access the underlying environment to get obs_helper
        self.obs_helper = self._get_obs_helper()

        # Calculate total feature dimension
        self.opponent_feature_dim = self._calculate_feature_dim()

        # Initialize history buffers as pre-allocated numpy arrays (much faster!)
        # Shape: (num_envs, history_length, opponent_feature_dim)
        self.history_buffers = np.zeros(
            (self.num_envs, history_length, self.opponent_feature_dim),
            dtype=np.float32
        )

        # Track current position in circular buffer for each env
        self.buffer_positions = np.zeros(self.num_envs, dtype=np.int32)
        self.buffer_filled = np.zeros(self.num_envs, dtype=bool)  # Has buffer been filled once?

    def _get_obs_helper(self):
        """Extract obs_helper from the base environment."""
        # Navigate through wrappers to find obs_helper
        env = self.venv
        while hasattr(env, 'venv'):
            env = env.venv

        # Check if we have envs (SubprocVecEnv/DummyVecEnv)
        if hasattr(env, 'envs') and len(env.envs) > 0:
            base_env = env.envs[0]
            if hasattr(base_env, 'obs_helper'):
                return base_env.obs_helper
            # If it's wrapped (e.g., SelfPlayWarehouseBrawl), check raw_env
            if hasattr(base_env, 'raw_env') and hasattr(base_env.raw_env, 'obs_helper'):
                return base_env.raw_env.obs_helper

        # Fallback: create a minimal obs_helper if we can't find one
        # This shouldn't happen in normal operation
        print("⚠ Warning: Could not find obs_helper, using fallback")
        return None

    def _calculate_feature_dim(self):
        """Calculate total dimensionality of opponent features."""
        if self.obs_helper is None:
            # Fallback: assume 13 features
            return 13

        total_dim = 0
        for feature_name in self.opponent_features:
            if hasattr(self.obs_helper, 'sections') and feature_name.lower() in self.obs_helper.sections:
                start, end = self.obs_helper.sections[feature_name.lower()]
                total_dim += (end - start)
            else:
                # Fallback: guess dimension based on feature name
                if 'pos' in feature_name or 'vel' in feature_name:
                    total_dim += 2
                else:
                    total_dim += 1

        return total_dim

    def _extract_opponent_features(self, obs: np.ndarray, env_idx: int) -> np.ndarray:
        """
        Extract opponent features from full observation.

        Args:
            obs: Full observation array
            env_idx: Environment index (for multi-env)

        Returns:
            Extracted opponent features as 1D array
        """
        if self.obs_helper is None:
            # Fallback: return first 13 elements (not ideal, but prevents crash)
            return obs[:self.opponent_feature_dim].astype(np.float32)

        features = []

        for feature_name in self.opponent_features:
            feature_name_lower = feature_name.lower()

            if hasattr(self.obs_helper, 'sections') and feature_name_lower in self.obs_helper.sections:
                feature_section = self.obs_helper.get_section(obs, feature_name_lower)
                features.append(feature_section)
            else:
                # If feature not found, append zeros
                if 'pos' in feature_name or 'vel' in feature_name:
                    features.append(np.zeros(2, dtype=np.float32))
                else:
                    features.append(np.zeros(1, dtype=np.float32))

        return np.concatenate(features).astype(np.float32)

    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        """
        Step environment and update opponent history.

        Returns:
            Tuple of (observations, rewards, dones, infos) where infos now includes
            'opponent_history' for each environment
        """
        obs, rewards, dones, infos = self.venv.step_wait()

        # Update history for each environment (optimized circular buffer)
        for env_idx in range(self.num_envs):
            # Extract opponent features from current observation
            opponent_features = self._extract_opponent_features(obs[env_idx], env_idx)

            # Write to circular buffer at current position
            pos = self.buffer_positions[env_idx]
            self.history_buffers[env_idx, pos, :] = opponent_features

            # Update position (circular)
            self.buffer_positions[env_idx] = (pos + 1) % self.history_length

            # Mark as filled after first complete pass
            if pos == self.history_length - 1:
                self.buffer_filled[env_idx] = True

            # Get history in correct temporal order (no array copy needed - just view!)
            # If buffer is filled, reorder to get oldest->newest
            # Otherwise, just use the filled portion
            if self.buffer_filled[env_idx]:
                # Reorder: from current position to end, then from start to current position
                current_pos = self.buffer_positions[env_idx]
                opponent_history = np.concatenate([
                    self.history_buffers[env_idx, current_pos:, :],
                    self.history_buffers[env_idx, :current_pos, :]
                ], axis=0)
            else:
                # Buffer not yet filled - just use what we have (zero-padded)
                opponent_history = self.history_buffers[env_idx]

            # Add to info dict
            if infos[env_idx] is None:
                infos[env_idx] = {}
            infos[env_idx]['opponent_history'] = opponent_history

        return obs, rewards, dones, infos

    def reset(self) -> np.ndarray:
        """
        Reset environments and clear history buffers.

        Returns:
            Initial observations
        """
        obs = self.venv.reset()

        # Reset all history buffers (zero them out, much faster!)
        self.history_buffers[:] = 0
        self.buffer_positions[:] = 0
        self.buffer_filled[:] = False

        # Initialize first frame with current opponent features
        for env_idx in range(self.num_envs):
            opponent_features = self._extract_opponent_features(obs[env_idx], env_idx)
            self.history_buffers[env_idx, 0, :] = opponent_features
            self.buffer_positions[env_idx] = 1

        return obs

    def get_opponent_history(self, env_idx: int = 0) -> np.ndarray:
        """
        Get current opponent history for a specific environment.

        Args:
            env_idx: Environment index

        Returns:
            Opponent history array of shape (history_length, opponent_feature_dim)
        """
        return np.array(
            list(self.history_buffers[env_idx]),
            dtype=np.float32
        )


def create_opponent_history_wrapper(
    venv: VecEnv,
    history_length: int = 60,
    opponent_features: Optional[list] = None
) -> OpponentHistoryBuffer:
    """
    Factory function to create OpponentHistoryBuffer wrapper.

    Args:
        venv: Vectorized environment
        history_length: Number of timesteps to track
        opponent_features: Optional list of features to extract

    Returns:
        Wrapped environment with opponent history tracking
    """
    return OpponentHistoryBuffer(
        venv=venv,
        history_length=history_length,
        opponent_features=opponent_features
    )


if __name__ == "__main__":
    print("OpponentHistoryBuffer wrapper created successfully!")
    print("\nThis wrapper should be used after creating a VecEnv.")
    print("Example usage:")
    print("  vec_env = DummyVecEnv([make_env_fn])")
    print("  vec_env = VecNormalize(vec_env, ...)")
    print("  vec_env = OpponentHistoryBuffer(vec_env, history_length=60)")
    print("\nThe wrapper adds 'opponent_history' to the info dict at each step.")
    print("Shape: (history_length, opponent_feature_dim)")
    print("Default features: position, velocity, facing, grounded, aerial, jumps, damage, stocks, move_type, state, stun")
