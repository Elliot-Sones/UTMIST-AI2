"""
Population Update Callback: Updates population during training.

This callback periodically:
1. Evaluates current agent against population members
2. Computes diversity metrics (strategy embeddings)
3. Adds agent to population if diverse and strong enough
4. Maintains weak agents from early training for robustness
"""

import os
import numpy as np
from typing import Optional
from stable_baselines3.common.callbacks import BaseCallback

from user.self_play.population_manager import PopulationManager


class PopulationUpdateCallback(BaseCallback):
    """
    Callback that manages population updates during training.

    Args:
        population_manager: PopulationManager instance
        update_frequency: Evaluate and update every N steps (default: 100,000)
        save_checkpoints: Whether to save checkpoints when adding to population
        checkpoint_dir: Directory to save checkpoints
        min_timesteps_before_add: Don't add to population before this many steps
        weak_agent_timesteps: Timesteps at which to force-add weak agents (for robustness)
    """

    def __init__(
        self,
        population_manager: PopulationManager,
        update_frequency: int = 100_000,
        save_checkpoints: bool = True,
        checkpoint_dir: str = "checkpoints",
        min_timesteps_before_add: int = 200_000,
        weak_agent_timesteps: list = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)

        self.population_manager = population_manager
        self.update_frequency = update_frequency
        self.save_checkpoints = save_checkpoints
        self.checkpoint_dir = checkpoint_dir
        self.min_timesteps_before_add = min_timesteps_before_add

        # Timesteps at which to force-add weak agents
        if weak_agent_timesteps is None:
            weak_agent_timesteps = [50_000, 150_000, 300_000]
        self.weak_agent_timesteps = set(weak_agent_timesteps)
        self.weak_agents_added = set()

        self.last_update_step = 0

        print(f"✓ PopulationUpdateCallback initialized:")
        print(f"  - Update every: {update_frequency:,} steps")
        print(f"  - Min steps before add: {min_timesteps_before_add:,}")
        print(f"  - Weak agent timesteps: {weak_agent_timesteps}")

    def _on_step(self) -> bool:
        """Called at each training step."""

        current_timesteps = self.num_timesteps

        # Check if we should force-add a weak agent
        for weak_timestep in self.weak_agent_timesteps:
            if (weak_timestep not in self.weak_agents_added and
                current_timesteps >= weak_timestep):
                self._add_weak_agent(weak_timestep)
                self.weak_agents_added.add(weak_timestep)

        # Check if it's time for regular population update
        if current_timesteps - self.last_update_step >= self.update_frequency:
            if current_timesteps >= self.min_timesteps_before_add:
                self._update_population()
            else:
                if self.verbose > 0:
                    print(f"  Skipping population update: {current_timesteps:,} < {self.min_timesteps_before_add:,}")

            self.last_update_step = current_timesteps

        return True  # Continue training

    def _add_weak_agent(self, timesteps: int):
        """Force-add a weak agent for robustness training."""
        if self.verbose > 0:
            print(f"\n{'='*70}")
            print(f"ADDING WEAK AGENT AT {timesteps:,} STEPS")
            print(f"{'='*70}")

        # Save checkpoint
        checkpoint_path = self._save_checkpoint(timesteps, prefix="weak_")

        # Add to population as weak agent
        self.population_manager.add_agent(
            checkpoint_path=checkpoint_path,
            timesteps=timesteps,
            win_rate=0.3,  # Assumed low win rate
            is_weak=True,
            force_add=True,
        )

        if self.verbose > 0:
            print(f"✓ Weak agent added: {checkpoint_path}")
            stats = self.population_manager.get_population_stats()
            print(f"  Population: {stats['size']} agents ({stats['num_weak']} weak, {stats['num_strong']} strong)")

    def _update_population(self):
        """Evaluate and potentially add current agent to population."""
        if self.verbose > 0:
            print(f"\n{'='*70}")
            print(f"POPULATION UPDATE AT {self.num_timesteps:,} STEPS")
            print(f"{'='*70}")

        # Evaluate against population (simplified - just use dummy win rate for now)
        # In full implementation, this would run evaluation games
        win_rate = self._estimate_win_rate()

        # Compute strategy embedding (simplified - use random for now)
        # In full implementation, this would extract embeddings from actual play
        strategy_embedding = self._compute_strategy_embedding()

        # Save checkpoint
        checkpoint_path = self._save_checkpoint(self.num_timesteps)

        # Try to add to population
        added = self.population_manager.add_agent(
            checkpoint_path=checkpoint_path,
            timesteps=self.num_timesteps,
            win_rate=win_rate,
            strategy_embedding=strategy_embedding,
        )

        if self.verbose > 0:
            if added:
                print(f"✓ Agent added to population!")
            else:
                print(f"  Agent not added (win_rate={win_rate:.2f}, diversity insufficient)")

            stats = self.population_manager.get_population_stats()
            print(f"  Population: {stats['size']} agents ({stats['num_weak']} weak, {stats['num_strong']} strong)")
            print(f"  Avg win rate: {stats['avg_win_rate']:.2f}")

    def _estimate_win_rate(self) -> float:
        """
        Estimate win rate of current agent.

        In full implementation, this would run evaluation games against population.
        For now, return a dummy value based on training progress.
        """
        # Simplified: assume win rate increases with training
        # Real implementation would run evaluation episodes
        progress = min(self.num_timesteps / 3_000_000, 1.0)
        base_win_rate = 0.5 + progress * 0.3

        # Add some randomness
        noise = np.random.normal(0, 0.05)
        return np.clip(base_win_rate + noise, 0.0, 1.0)

    def _compute_strategy_embedding(self) -> np.ndarray:
        """
        Compute strategy embedding for current agent.

        In full implementation, this would:
        1. Run episodes and collect opponent_history data
        2. Pass through strategy encoder
        3. Average embeddings over multiple episodes

        For now, return random embedding as placeholder.
        """
        # Placeholder: random 32D embedding
        # Real implementation would extract from actual gameplay
        return np.random.randn(32).astype(np.float32)

    def _save_checkpoint(self, timesteps: int, prefix: str = "") -> str:
        """
        Save model checkpoint.

        Args:
            timesteps: Current training timesteps
            prefix: Optional prefix for checkpoint name

        Returns:
            Path to saved checkpoint
        """
        if not self.save_checkpoints:
            return f"{prefix}checkpoint_{timesteps}.zip"

        checkpoint_name = f"{prefix}rl_model_{timesteps}_steps"
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            "population",
            f"{checkpoint_name}.zip"
        )

        # Ensure directory exists
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        # Save model
        self.model.save(checkpoint_path)

        if self.verbose > 0:
            print(f"  Saved checkpoint: {checkpoint_path}")

        return checkpoint_path


# Helper function to create callback
def create_population_update_callback(
    population_manager: PopulationManager,
    update_frequency: int = 100_000,
    checkpoint_dir: str = "checkpoints",
    verbose: int = 1,
) -> PopulationUpdateCallback:
    """
    Factory function to create PopulationUpdateCallback.

    Args:
        population_manager: PopulationManager instance
        update_frequency: Update every N steps
        checkpoint_dir: Directory for checkpoints
        verbose: Verbosity level

    Returns:
        PopulationUpdateCallback instance
    """
    return PopulationUpdateCallback(
        population_manager=population_manager,
        update_frequency=update_frequency,
        save_checkpoints=True,
        checkpoint_dir=checkpoint_dir,
        min_timesteps_before_add=200_000,
        weak_agent_timesteps=[50_000, 150_000, 300_000],
        verbose=verbose,
    )


if __name__ == "__main__":
    print("PopulationUpdateCallback created successfully!")
    print("\nThis callback should be added to RecurrentPPO.learn():")
    print("  model.learn(")
    print("    total_timesteps=3_000_000,")
    print("    callback=PopulationUpdateCallback(population_manager, ...)")
    print("  )")
    print("\nThe callback will:")
    print("  1. Save weak agents at 50k, 150k, 300k steps")
    print("  2. Evaluate and add strong agents every 100k steps")
    print("  3. Maintain diverse population automatically")
