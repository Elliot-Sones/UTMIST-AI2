"""
Augmented Observation Wrapper: Adds opponent history to observations.

This wrapper takes opponent history from the info dict (populated by
OpponentHistoryBuffer) and appends it to observations. This allows the
strategy encoder to process opponent behavior as part of the policy's input.
"""

import numpy as np
from typing import Any, Dict, Tuple
import gymnasium as gym
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper


class AugmentedObservationWrapper(VecEnvWrapper):
    """
    VecEnv wrapper that augments observations with opponent history.

    Takes 'opponent_history' from info dict and flattens it into the observation.
    This is necessary because the policy expects opponent history as part of
    the observation tensor, not in the info dict.

    Args:
        venv: Vectorized environment (should already have OpponentHistoryBuffer)
        opponent_history_shape: Shape of opponent history (history_length, feature_dim)
    """

    def __init__(
        self,
        venv: VecEnv,
        opponent_history_shape: Tuple[int, int] = (60, 13),
    ):
        super().__init__(venv)

        self.history_length, self.feature_dim = opponent_history_shape
        self.history_flat_dim = self.history_length * self.feature_dim

        # Get original observation space
        original_obs_shape = self.venv.observation_space.shape
        original_obs_dim = original_obs_shape[0]

        # Create new observation space with augmented observations
        new_obs_dim = original_obs_dim + self.history_flat_dim

        self.observation_space = gym.spaces.Box(
            low=-float('inf'),
            high=float('inf'),
            shape=(new_obs_dim,),
            dtype=self.venv.observation_space.dtype
        )

        print(f"✓ AugmentedObservationWrapper:")
        print(f"  - Original obs dim: {original_obs_dim}")
        print(f"  - Opponent history: {self.history_length} × {self.feature_dim} = {self.history_flat_dim}")
        print(f"  - New obs dim: {new_obs_dim}")

        # Cache for last opponent history (used when stepping)
        self.last_opponent_histories = [
            np.zeros((self.history_length, self.feature_dim), dtype=np.float32)
            for _ in range(self.num_envs)
        ]

    def _augment_observations(
        self,
        obs: np.ndarray,
        infos: list
    ) -> np.ndarray:
        """
        Augment observations with opponent history from infos.

        Args:
            obs: Original observations (num_envs, obs_dim)
            infos: List of info dicts, should contain 'opponent_history'

        Returns:
            Augmented observations (num_envs, obs_dim + history_flat_dim)
        """
        augmented_obs = np.zeros(
            (self.num_envs, obs.shape[1] + self.history_flat_dim),
            dtype=obs.dtype
        )

        for env_idx in range(self.num_envs):
            # Get opponent history from info (or use cached version)
            if infos[env_idx] is not None and 'opponent_history' in infos[env_idx]:
                opponent_history = infos[env_idx]['opponent_history']
                self.last_opponent_histories[env_idx] = opponent_history
            else:
                # Use last known history if not in info
                opponent_history = self.last_opponent_histories[env_idx]

            # Flatten opponent history
            opponent_history_flat = opponent_history.flatten()

            # Concatenate: [original_obs, flattened_opponent_history]
            augmented_obs[env_idx] = np.concatenate([
                obs[env_idx],
                opponent_history_flat
            ])

        return augmented_obs

    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        """
        Step environment and augment observations with opponent history.

        Returns:
            Tuple of (augmented_observations, rewards, dones, infos)
        """
        obs, rewards, dones, infos = self.venv.step_wait()

        # Augment observations with opponent history
        augmented_obs = self._augment_observations(obs, infos)

        # IMPORTANT: Also augment terminal_observation in infos if present
        # VecNormalize expects terminal observations to have the same shape
        for env_idx in range(self.num_envs):
            if infos[env_idx] is not None and 'terminal_observation' in infos[env_idx]:
                terminal_obs = infos[env_idx]['terminal_observation']

                # Get opponent history for this terminal observation
                if 'opponent_history' in infos[env_idx]:
                    opponent_history = infos[env_idx]['opponent_history']
                else:
                    # Use last known history
                    opponent_history = self.last_opponent_histories[env_idx]

                # Flatten and concatenate
                opponent_history_flat = opponent_history.flatten()
                augmented_terminal_obs = np.concatenate([
                    terminal_obs,
                    opponent_history_flat
                ])

                infos[env_idx]['terminal_observation'] = augmented_terminal_obs

        return augmented_obs, rewards, dones, infos

    def reset(self) -> np.ndarray:
        """
        Reset environments and augment initial observations.

        Returns:
            Augmented initial observations
        """
        obs = self.venv.reset()

        # Clear cached histories
        for env_idx in range(self.num_envs):
            self.last_opponent_histories[env_idx] = np.zeros(
                (self.history_length, self.feature_dim),
                dtype=np.float32
            )

        # For reset, we need to get opponent history somehow
        # Since reset() doesn't return infos, we'll use zeros initially
        # The first step will update with real data
        augmented_obs = np.zeros(
            (self.num_envs, obs.shape[1] + self.history_flat_dim),
            dtype=obs.dtype
        )

        for env_idx in range(self.num_envs):
            opponent_history_flat = self.last_opponent_histories[env_idx].flatten()
            augmented_obs[env_idx] = np.concatenate([
                obs[env_idx],
                opponent_history_flat
            ])

        return augmented_obs


def create_augmented_observation_wrapper(
    venv: VecEnv,
    opponent_history_shape: Tuple[int, int] = (60, 13)
) -> AugmentedObservationWrapper:
    """
    Factory function to create AugmentedObservationWrapper.

    Args:
        venv: Vectorized environment (should have OpponentHistoryBuffer)
        opponent_history_shape: Shape of opponent history (history_length, feature_dim)

    Returns:
        Wrapped environment with augmented observations
    """
    return AugmentedObservationWrapper(
        venv=venv,
        opponent_history_shape=opponent_history_shape
    )


if __name__ == "__main__":
    print("AugmentedObservationWrapper created successfully!")
    print("\nThis wrapper should be used AFTER OpponentHistoryBuffer.")
    print("Example usage:")
    print("  vec_env = DummyVecEnv([make_env_fn])")
    print("  vec_env = VecNormalize(vec_env, ...)")
    print("  vec_env = OpponentHistoryBuffer(vec_env, history_length=60)")
    print("  vec_env = AugmentedObservationWrapper(vec_env, opponent_history_shape=(60, 13))")
    print("\nThe wrapper concatenates opponent_history to observations.")
    print("New observation = [original_obs, flattened_opponent_history]")
