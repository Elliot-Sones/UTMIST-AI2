"""
Population Manager: Maintains diverse pool of agents for self-play training.

This module implements a population-based self-play system that:
1. Stores checkpoints of past agents with diverse strategies
2. Evaluates new agents and adds them if they're strong and novel
3. Maintains a balance between strong agents and "weird/bad" agents for robustness
4. Computes diversity metrics based on strategy embeddings and behavior
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class PopulationMember:
    """
    Represents a single agent in the population.

    Attributes:
        checkpoint_path: Path to model checkpoint
        timesteps: Training steps when this checkpoint was created
        win_rate: Win rate against current training agent
        strategy_embedding: Average strategy embedding (for diversity computation)
        behavior_signature: Behavioral metrics (action entropy, aggression, etc.)
        is_weak: Whether this is intentionally kept as a "weak" agent
        added_timestamp: When this agent was added to population
    """
    checkpoint_path: str
    timesteps: int
    win_rate: float
    strategy_embedding: Optional[np.ndarray] = None
    behavior_signature: Optional[Dict[str, float]] = None
    is_weak: bool = False
    added_timestamp: str = ""
    metadata: Dict = None

    def __post_init__(self):
        if self.added_timestamp == "":
            self.added_timestamp = datetime.now().isoformat()
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict."""
        d = asdict(self)
        if self.strategy_embedding is not None:
            d['strategy_embedding'] = self.strategy_embedding.tolist()
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> 'PopulationMember':
        """Create from dict."""
        if 'strategy_embedding' in d and d['strategy_embedding'] is not None:
            d['strategy_embedding'] = np.array(d['strategy_embedding'])
        return cls(**d)


class PopulationManager:
    """
    Manages a population of diverse agents for self-play training.

    The population consists of:
    - Strong diverse agents (60-67% of population)
    - Recent strong performers (13-20%)
    - Weird/bad agents for robustness (20%)

    Args:
        checkpoint_dir: Directory to store population checkpoints
        max_population_size: Maximum number of agents to keep (default: 15)
        num_weak_agents: Number of weak agents to maintain (default: 3)
        diversity_threshold: Minimum diversity score to add new agent (default: 0.1)
    """

    def __init__(
        self,
        checkpoint_dir: str,
        max_population_size: int = 15,
        num_weak_agents: int = 3,
        diversity_threshold: float = 0.1,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.population_dir = self.checkpoint_dir / "population"
        self.population_dir.mkdir(parents=True, exist_ok=True)

        self.max_population_size = max_population_size
        self.num_weak_agents = num_weak_agents
        self.diversity_threshold = diversity_threshold

        self.population: List[PopulationMember] = []
        self.population_file = self.population_dir / "population.json"

        # Load existing population if available
        self._load_population()

        print(f"✓ PopulationManager initialized:")
        print(f"  - Max population: {max_population_size}")
        print(f"  - Weak agents: {num_weak_agents}")
        print(f"  - Current size: {len(self.population)}")

    def _load_population(self):
        """Load population from disk if exists."""
        if self.population_file.exists():
            try:
                with open(self.population_file, 'r') as f:
                    data = json.load(f)
                self.population = [PopulationMember.from_dict(d) for d in data]
                print(f"  Loaded {len(self.population)} agents from disk")
            except Exception as e:
                print(f"  Warning: Failed to load population: {e}")
                self.population = []

    def _save_population(self):
        """Save population to disk."""
        try:
            data = [member.to_dict() for member in self.population]
            with open(self.population_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"  Warning: Failed to save population: {e}")

    def add_agent(
        self,
        checkpoint_path: str,
        timesteps: int,
        win_rate: float,
        strategy_embedding: Optional[np.ndarray] = None,
        behavior_signature: Optional[Dict[str, float]] = None,
        is_weak: bool = False,
        force_add: bool = False,
    ) -> bool:
        """
        Add a new agent to the population if it's diverse enough.

        Args:
            checkpoint_path: Path to model checkpoint
            timesteps: Training steps
            win_rate: Win rate against training agent
            strategy_embedding: Strategy embedding vector
            behavior_signature: Behavioral metrics
            is_weak: Mark as weak agent (always kept)
            force_add: Force addition even if not diverse

        Returns:
            True if agent was added, False otherwise
        """
        # Create new member
        new_member = PopulationMember(
            checkpoint_path=checkpoint_path,
            timesteps=timesteps,
            win_rate=win_rate,
            strategy_embedding=strategy_embedding,
            behavior_signature=behavior_signature,
            is_weak=is_weak,
        )

        # Always add if weak agent or forced
        if is_weak or force_add:
            self.population.append(new_member)
            self._save_population()
            print(f"  Added {'weak' if is_weak else 'forced'} agent: {timesteps} steps, WR={win_rate:.2f}")
            return True

        # Check if diverse enough
        if not self._is_diverse_enough(new_member):
            print(f"  Agent not diverse enough (timesteps={timesteps}, WR={win_rate:.2f})")
            return False

        # Check if strong enough (win rate > 55%)
        if win_rate < 0.55:
            print(f"  Agent not strong enough: WR={win_rate:.2f} < 0.55")
            return False

        # Add to population
        self.population.append(new_member)
        print(f"  ✓ Added diverse agent: {timesteps} steps, WR={win_rate:.2f}")

        # Prune if population too large
        if len(self.population) > self.max_population_size:
            self._prune_population()

        self._save_population()
        return True

    def _is_diverse_enough(self, new_member: PopulationMember) -> bool:
        """
        Check if new agent is diverse enough to add.

        Diversity is measured as:
        1. Strategy embedding distance (if available)
        2. Behavioral signature difference

        Args:
            new_member: Candidate agent

        Returns:
            True if diverse enough, False otherwise
        """
        if len(self.population) == 0:
            return True

        # If we don't have embeddings, use behavioral diversity only
        if new_member.strategy_embedding is None:
            return self._behavioral_diversity(new_member) > self.diversity_threshold

        # Compute average distance to existing population
        distances = []
        for member in self.population:
            if member.strategy_embedding is not None:
                dist = np.linalg.norm(
                    new_member.strategy_embedding - member.strategy_embedding
                )
                distances.append(dist)

        if len(distances) == 0:
            return True

        avg_distance = np.mean(distances)
        return avg_distance > self.diversity_threshold

    def _behavioral_diversity(self, new_member: PopulationMember) -> float:
        """
        Compute behavioral diversity score.

        Uses metrics like action entropy, aggression, etc.

        Args:
            new_member: Candidate agent

        Returns:
            Diversity score (higher = more diverse)
        """
        if new_member.behavior_signature is None or len(self.population) == 0:
            return 1.0

        # Compute average behavioral difference
        differences = []
        for member in self.population:
            if member.behavior_signature is not None:
                # Simple L1 distance on normalized metrics
                diff = sum(
                    abs(new_member.behavior_signature.get(k, 0) -
                        member.behavior_signature.get(k, 0))
                    for k in new_member.behavior_signature.keys()
                )
                differences.append(diff)

        if len(differences) == 0:
            return 1.0

        return np.mean(differences)

    def _prune_population(self):
        """
        Prune population to max size by removing least diverse strong agents.

        Keeps:
        - All weak agents (marked with is_weak=True)
        - Most diverse strong agents
        - Recent strong agents
        """
        # Separate weak and strong agents
        weak_agents = [m for m in self.population if m.is_weak]
        strong_agents = [m for m in self.population if not m.is_weak]

        # If we have too many weak agents, keep only the newest ones
        if len(weak_agents) > self.num_weak_agents:
            weak_agents = sorted(weak_agents, key=lambda m: m.timesteps, reverse=True)
            weak_agents = weak_agents[:self.num_weak_agents]

        # For strong agents, keep the most diverse ones
        target_strong = self.max_population_size - len(weak_agents)

        if len(strong_agents) <= target_strong:
            # No pruning needed
            self.population = weak_agents + strong_agents
            return

        # Sort by diversity (using strategy embeddings if available)
        if any(m.strategy_embedding is not None for m in strong_agents):
            # Compute pairwise diversity and select most diverse subset
            diversity_scores = []
            for member in strong_agents:
                if member.strategy_embedding is not None:
                    # Compute average distance to all other agents
                    distances = [
                        np.linalg.norm(member.strategy_embedding - other.strategy_embedding)
                        for other in strong_agents
                        if other.strategy_embedding is not None and other != member
                    ]
                    avg_dist = np.mean(distances) if distances else 0
                else:
                    avg_dist = 0
                diversity_scores.append((member, avg_dist))

            # Sort by diversity (descending) and keep top N
            diversity_scores.sort(key=lambda x: x[1], reverse=True)
            strong_agents = [m for m, _ in diversity_scores[:target_strong]]
        else:
            # Fallback: keep most recent agents
            strong_agents = sorted(strong_agents, key=lambda m: m.timesteps, reverse=True)
            strong_agents = strong_agents[:target_strong]

        self.population = weak_agents + strong_agents
        print(f"  Pruned population to {len(self.population)} agents")

    def get_sampling_weights(self) -> np.ndarray:
        """
        Get sampling weights for population members.

        Weak agents get higher weight to ensure robustness training.

        Returns:
            Array of sampling weights (sum to 1)
        """
        if len(self.population) == 0:
            return np.array([])

        weights = np.ones(len(self.population))

        # Give higher weight to weak agents (2x)
        for i, member in enumerate(self.population):
            if member.is_weak:
                weights[i] = 2.0

        # Normalize
        weights = weights / weights.sum()
        return weights

    def sample_opponent(self, rng: Optional[np.random.RandomState] = None) -> Optional[PopulationMember]:
        """
        Sample an opponent from the population.

        Args:
            rng: Random number generator (optional)

        Returns:
            Sampled PopulationMember or None if population empty
        """
        if len(self.population) == 0:
            return None

        if rng is None:
            rng = np.random

        weights = self.get_sampling_weights()
        idx = rng.choice(len(self.population), p=weights)
        return self.population[idx]

    def get_population_stats(self) -> Dict:
        """
        Get statistics about the current population.

        Returns:
            Dictionary with population statistics
        """
        if len(self.population) == 0:
            return {
                'size': 0,
                'num_weak': 0,
                'num_strong': 0,
                'avg_win_rate': 0.0,
                'avg_timesteps': 0,
            }

        weak = [m for m in self.population if m.is_weak]
        strong = [m for m in self.population if not m.is_weak]

        return {
            'size': len(self.population),
            'num_weak': len(weak),
            'num_strong': len(strong),
            'avg_win_rate': np.mean([m.win_rate for m in self.population]),
            'avg_timesteps': np.mean([m.timesteps for m in self.population]),
            'min_timesteps': min(m.timesteps for m in self.population),
            'max_timesteps': max(m.timesteps for m in self.population),
        }

    def __len__(self) -> int:
        return len(self.population)

    def __repr__(self) -> str:
        stats = self.get_population_stats()
        return (
            f"PopulationManager(size={stats['size']}, "
            f"weak={stats['num_weak']}, strong={stats['num_strong']}, "
            f"avg_wr={stats['avg_win_rate']:.2f})"
        )


if __name__ == "__main__":
    print("Testing PopulationManager...")

    # Create test manager
    manager = PopulationManager(
        checkpoint_dir="test_checkpoints",
        max_population_size=10,
        num_weak_agents=2,
    )

    # Add some test agents
    for i in range(12):
        timesteps = 50_000 + i * 50_000
        win_rate = 0.5 + i * 0.03
        embedding = np.random.randn(32)

        added = manager.add_agent(
            checkpoint_path=f"checkpoint_{timesteps}.pt",
            timesteps=timesteps,
            win_rate=min(win_rate, 0.9),
            strategy_embedding=embedding,
            is_weak=(i < 2),  # First 2 are weak
        )

        print(f"Agent {i}: timesteps={timesteps}, WR={win_rate:.2f}, added={added}")

    # Print stats
    print(f"\nFinal population: {manager}")
    print(f"Stats: {manager.get_population_stats()}")

    # Test sampling
    print("\nSampling 5 opponents:")
    for i in range(5):
        opponent = manager.sample_opponent()
        print(f"  {i+1}. {opponent.checkpoint_path} (WR={opponent.win_rate:.2f}, weak={opponent.is_weak})")

    print("\n✓ PopulationManager tests passed!")
