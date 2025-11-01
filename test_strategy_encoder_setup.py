"""
Quick test to verify all strategy encoder components can be imported and initialized.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print("Testing Strategy Encoder System Setup")
print("=" * 70)

# Test 1: Import all modules
print("\n1. Testing imports...")
try:
    from user.models.strategy_encoder import StrategyEncoder
    from user.models.opponent_conditioned_policy import OpponentConditionedFeatureExtractor
    from user.wrappers.opponent_history_wrapper import OpponentHistoryBuffer
    from user.wrappers.augmented_obs_wrapper import AugmentedObservationWrapper
    from user.self_play.population_manager import PopulationManager
    from user.self_play.diverse_opponent_sampler import DiverseOpponentSampler
    from user.self_play.population_update_callback import PopulationUpdateCallback
    print("✓ All imports successful!")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Initialize StrategyEncoder
print("\n2. Testing StrategyEncoder initialization...")
try:
    encoder = StrategyEncoder(
        input_features=13,
        history_length=60,
        embedding_dim=32
    )
    print(f"✓ StrategyEncoder created: {sum(p.numel() for p in encoder.parameters()):,} parameters")
except Exception as e:
    print(f"✗ StrategyEncoder initialization failed: {e}")
    sys.exit(1)

# Test 3: Initialize PopulationManager
print("\n3. Testing PopulationManager initialization...")
try:
    import tempfile
    import shutil
    temp_dir = tempfile.mkdtemp()

    population_manager = PopulationManager(
        checkpoint_dir=temp_dir,
        max_population_size=15,
        num_weak_agents=3,
    )
    print(f"✓ PopulationManager created: {len(population_manager)} agents")

    # Cleanup
    shutil.rmtree(temp_dir)
except Exception as e:
    print(f"✗ PopulationManager initialization failed: {e}")
    sys.exit(1)

# Test 4: Initialize DiverseOpponentSampler
print("\n4. Testing DiverseOpponentSampler initialization...")
try:
    temp_dir = tempfile.mkdtemp()

    population_manager = PopulationManager(
        checkpoint_dir=temp_dir,
        max_population_size=10,
    )

    sampler = DiverseOpponentSampler(
        checkpoint_dir=temp_dir,
        population_manager=population_manager,
    )
    print(f"✓ DiverseOpponentSampler created")

    # Test sampling
    opponent = sampler.get_opponent()
    print(f"  Sampled opponent: {type(opponent).__name__}")

    # Cleanup
    shutil.rmtree(temp_dir)
except Exception as e:
    print(f"✗ DiverseOpponentSampler initialization failed: {e}")
    sys.exit(1)

# Test 5: Check PyTorch GPU availability
print("\n5. Checking GPU availability...")
try:
    import torch
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif torch.backends.mps.is_available():
        print("✓ Apple MPS available")
    else:
        print("⚠ No GPU available - training will be slow on CPU")
except Exception as e:
    print(f"✗ GPU check failed: {e}")

print("\n" + "=" * 70)
print("✓ All tests passed! System is ready for training.")
print("\nTo start training:")
print("  python user/train_with_strategy_encoder.py")
print("\nRecommended GPU: NVIDIA 4090 or equivalent (24GB VRAM)")
print("Expected training time: 8-10 hours for 5M steps")
print("=" * 70)
