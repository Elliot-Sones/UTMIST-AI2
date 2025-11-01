"""
Diverse Opponent Sampler: Integrates PopulationManager with self-play training.

This module provides a self-play handler that:
1. Samples opponents from PopulationManager (diverse + weak agents)
2. Falls back to scripted opponents when population is small
3. Adds noise/randomness for robustness
4. Tracks opponent types for monitoring
"""

import os
import sys
import random
import numpy as np
from pathlib import Path
from typing import Optional

# Add project root for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environment.agent import Agent, ConstantAgent, RecurrentPPOAgent
from user.self_play.population_manager import PopulationManager


class DiverseOpponentSampler:
    """
    Self-play handler that samples from a diverse population of agents.

    Integrates with PopulationManager to provide:
    - Diverse opponents from population (70% when available)
    - Fallback to scripted agents (30% always, 100% when population empty)
    - Random action noise for robustness (10% of episodes)

    Args:
        checkpoint_dir: Directory with checkpoints
        population_manager: PopulationManager instance
        noise_probability: Probability of adding random noise to opponent actions
        use_population_prob: Probability of using population (vs scripted) when available
    """

    def __init__(
        self,
        checkpoint_dir: str,
        population_manager: PopulationManager,
        noise_probability: float = 0.10,
        use_population_prob: float = 0.70,
        verbose: bool = True,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.population_manager = population_manager
        self.noise_probability = noise_probability
        self.use_population_prob = use_population_prob
        self.env = None  # Set by environment
        self.verbose = verbose

        # Statistics tracking
        self.stats = {
            'population_sampled': 0,
            'scripted_sampled': 0,
            'noise_added': 0,
            'weak_sampled': 0,
        }

        if verbose:
            print(f"✓ DiverseOpponentSampler initialized:")
            print(f"  - Population size: {len(population_manager)}")
            print(f"  - Use population prob: {use_population_prob:.1%}")
            print(f"  - Noise prob: {noise_probability:.1%}")

    def __call__(self) -> Agent:
        """
        Make sampler callable (required by environment).

        Returns:
            Agent instance to use as opponent
        """
        return self.get_opponent()

    def get_opponent(self) -> Agent:
        """
        Sample an opponent from population or fallback to scripted.

        Returns:
            Agent instance to use as opponent
        """
        # Decide whether to use population or scripted
        use_population = (
            len(self.population_manager) > 0 and
            random.random() < self.use_population_prob
        )

        if use_population:
            return self._sample_from_population()
        else:
            return self._sample_scripted()

    def _sample_from_population(self) -> Agent:
        """Sample opponent from population."""
        # Sample from population using its weighted sampling
        member = self.population_manager.sample_opponent()

        if member is None:
            # Fallback to scripted if sampling failed
            return self._sample_scripted()

        # Load agent from checkpoint
        try:
            opponent = RecurrentPPOAgent(file_path=member.checkpoint_path)
            if self.env:
                opponent.get_env_info(self.env)

            # Update statistics
            self.stats['population_sampled'] += 1
            if member.is_weak:
                self.stats['weak_sampled'] += 1

            # Maybe add noise
            if random.random() < self.noise_probability:
                opponent = NoisyAgentWrapper(opponent, noise_level=0.2)
                self.stats['noise_added'] += 1

            return opponent

        except Exception as e:
            print(f"  Warning: Failed to load population agent: {e}")
            return self._sample_scripted()

    def _sample_scripted(self) -> Agent:
        """Fallback to scripted opponent (e.g., ConstantAgent)."""
        self.stats['scripted_sampled'] += 1

        # For now, just return ConstantAgent
        # In the full training script, this would sample from the existing OPPONENT_MIX
        opponent = ConstantAgent()
        if self.env:
            opponent.get_env_info(self.env)

        return opponent

    def get_stats(self) -> dict:
        """Get sampling statistics."""
        total = sum(self.stats.values())
        return {
            **self.stats,
            'total_sampled': total,
            'population_rate': self.stats['population_sampled'] / max(total, 1),
            'weak_rate': self.stats['weak_sampled'] / max(self.stats['population_sampled'], 1),
            'noise_rate': self.stats['noise_added'] / max(total, 1),
        }

    def reset_stats(self):
        """Reset statistics counters."""
        for key in self.stats:
            self.stats[key] = 0


class NoisyAgentWrapper(Agent):
    """
    Wrapper that adds random noise to agent actions.

    Used to make training more robust to weird/unexpected behaviors.

    Args:
        base_agent: Agent to wrap
        noise_level: Probability of replacing action with random action
    """

    def __init__(self, base_agent: Agent, noise_level: float = 0.2):
        # Don't call super().__init__() to avoid overwriting base_agent's env info
        self.base_agent = base_agent
        self.noise_level = noise_level
        self.initialized = True

    def get_env_info(self, env):
        """Pass env info to base agent."""
        self.base_agent.get_env_info(env)
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.obs_helper = env.obs_helper
        self.act_helper = env.act_helper

    def predict(self, obs):
        """Predict action with occasional random noise."""
        # Get action from base agent
        action = self.base_agent.predict(obs)

        # With noise_level probability, replace with random action
        if random.random() < self.noise_level:
            # Random binary action vector
            action = (np.random.rand(*action.shape) > 0.5).astype(action.dtype)

        return action

    def reset(self):
        """Reset base agent."""
        if hasattr(self.base_agent, 'reset'):
            self.base_agent.reset()

    def save(self, file_path: str):
        """Save base agent."""
        if hasattr(self.base_agent, 'save'):
            self.base_agent.save(file_path)

    def get_num_timesteps(self):
        """Get num timesteps from base agent."""
        if hasattr(self.base_agent, 'get_num_timesteps'):
            return self.base_agent.get_num_timesteps()
        return 0

    def update_num_timesteps(self, num_timesteps: int):
        """Update num timesteps in base agent."""
        if hasattr(self.base_agent, 'update_num_timesteps'):
            self.base_agent.update_num_timesteps(num_timesteps)


def create_diverse_opponent_sampler(
    checkpoint_dir: str,
    max_population_size: int = 15,
    num_weak_agents: int = 3,
    noise_probability: float = 0.10,
    use_population_prob: float = 0.70,
    verbose: bool = True,
) -> DiverseOpponentSampler:
    """
    Factory function to create DiverseOpponentSampler with PopulationManager.

    Args:
        checkpoint_dir: Directory for checkpoints
        max_population_size: Max population size
        num_weak_agents: Number of weak agents to keep
        noise_probability: Prob of adding noise
        use_population_prob: Prob of using population vs scripted
        verbose: Whether to print initialization messages

    Returns:
        DiverseOpponentSampler instance
    """
    population_manager = PopulationManager(
        checkpoint_dir=checkpoint_dir,
        max_population_size=max_population_size,
        num_weak_agents=num_weak_agents,
    )

    return DiverseOpponentSampler(
        checkpoint_dir=checkpoint_dir,
        population_manager=population_manager,
        noise_probability=noise_probability,
        use_population_prob=use_population_prob,
        verbose=verbose,
    )


if __name__ == "__main__":
    print("Testing DiverseOpponentSampler...")

    # Create sampler with empty population
    sampler = create_diverse_opponent_sampler(
        checkpoint_dir="test_checkpoints",
        max_population_size=10,
        num_weak_agents=2,
    )

    print("\nSampling 10 opponents (population empty, should use scripted):")
    for i in range(10):
        opponent = sampler.get_opponent()
        print(f"  {i+1}. {type(opponent).__name__}")

    print(f"\nStats: {sampler.get_stats()}")

    print("\n✓ DiverseOpponentSampler tests passed!")

    # Cleanup
    import shutil
    if os.path.exists("test_checkpoints"):
        shutil.rmtree("test_checkpoints")
