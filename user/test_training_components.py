"""
🧪 COMPREHENSIVE TRAINING COMPONENT TESTS
================================================================================

This file contains tests to validate EVERY component of the training pipeline
BEFORE running expensive 10M training. Run these tests after curriculum Stage 1
to ensure 95% confidence that full training will succeed.

Test Categories:
1. Reward Function Tests - Verify rewards trigger correctly
2. Strategy Encoder Tests - Verify transformer produces meaningful latents
3. Policy Integration Tests - Verify LSTM conditions on strategy properly
4. Opponent Selection Tests - Verify self-play sampling works
5. End-to-End Tests - Verify agent can beat ConstantAgent consistently

Usage:
    python user/test_training_components.py --checkpoint path/to/checkpoint.zip
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from functools import partial

# Setup project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import training modules
from user.train_agent import (
    TransformerStrategyAgent, 
    TransformerStrategyEncoder,
    gen_reward_manager,
    TORCH_DEVICE,
    RewardMode,
    damage_interaction_reward,
    danger_zone_reward,
    head_to_opponent,
)

from environment.agent import (
    ConstantAgent,
    BasedAgent,
    Agent,
    CameraResolution,
    run_match as env_run_match,
    Result,
    RewardManager,
)

from environment.environment import WarehouseBrawl


# ============================================================================
# TEST 1: REWARD FUNCTION VALIDATION
# ============================================================================

class RewardFunctionTester:
    """Tests that reward functions trigger correctly and have proper magnitudes."""
    
    def __init__(self):
        self.results = {}
    
    def test_damage_reward_triggers(self) -> bool:
        """
        Test: Damage reward should fire when agent hits opponent.
        
        Setup: Create simple match where agent attacks ConstantAgent
        Expected: damage_interaction_reward > 0 when hit lands
        """
        print("\n" + "="*70)
        print("TEST 1.1: Damage Reward Triggering")
        print("="*70)
        
        # Create test environment
        reward_manager = gen_reward_manager()
        
        # Run short match with BasedAgent (will attack) vs ConstantAgent
        print("  Running test match: BasedAgent vs ConstantAgent...")
        match_stats = env_run_match(
            partial(BasedAgent),
            partial(ConstantAgent),
            max_timesteps=30*10,  # 10 seconds
            resolution=CameraResolution.LOW,
            reward_manager=reward_manager,
            train_mode=True
        )
        
        # Check that damage was dealt
        damage_dealt = match_stats.player2.damage_taken  # How much ConstantAgent took
        damage_taken = match_stats.player1.damage_taken  # How much BasedAgent took
        
        print(f"\n  Results:")
        print(f"    Damage dealt to ConstantAgent: {damage_dealt:.1f}")
        print(f"    Damage taken by BasedAgent: {damage_taken:.1f}")
        print(f"    Total reward accumulated: {reward_manager.total_reward:.2f}")
        
        # Validation
        success = damage_dealt > 0 and reward_manager.total_reward > 0
        
        if success:
            print(f"\n  ✅ PASS: Damage reward is triggering correctly")
        else:
            print(f"\n  ❌ FAIL: Damage reward did not trigger!")
            print(f"     - Expected: damage > 0 and reward > 0")
            print(f"     - Got: damage={damage_dealt}, reward={reward_manager.total_reward}")
        
        self.results['damage_reward_triggers'] = success
        return success
    
    def test_reward_magnitudes(self) -> bool:
        """
        Test: Reward magnitudes should be in expected ranges.
        
        Expected ranges (per frame):
        - damage_interaction_reward: +150 per hit (large positive)
        - danger_zone_reward: -2.0 when high (moderate negative)
        - approach_reward: +0.1 when approaching (small positive)
        """
        print("\n" + "="*70)
        print("TEST 1.2: Reward Magnitude Validation")
        print("="*70)
        
        reward_manager = gen_reward_manager()
        
        # Check configured weights
        print("\n  Configured reward weights:")
        for name, term in reward_manager.reward_functions.items():
            print(f"    {name}: {term.weight}")
        
        print("\n  Signal reward weights:")
        for name, (signal, term) in reward_manager.signal_subscriptions.items():
            print(f"    {name}: {term.weight}")
        
        # Validate weights are in expected ranges
        damage_weight = reward_manager.reward_functions['damage_interaction_reward'].weight
        danger_weight = reward_manager.reward_functions['danger_zone_reward'].weight
        approach_weight = reward_manager.reward_functions['head_to_opponent'].weight
        
        checks = {
            'damage_weight_large': (damage_weight >= 100, f"damage weight {damage_weight} >= 100"),
            'danger_weight_moderate': (-5 <= danger_weight <= -1, f"-5 <= danger {danger_weight} <= -1"),
            'approach_weight_small': (0 < approach_weight <= 1, f"0 < approach {approach_weight} <= 1"),
        }
        
        print("\n  Validation checks:")
        all_pass = True
        for check_name, (passed, msg) in checks.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"    {status}: {msg}")
            if not passed:
                all_pass = False
        
        self.results['reward_magnitudes'] = all_pass
        return all_pass
    
    def test_no_attack_penalty(self) -> bool:
        """
        Test: Agent should NOT be penalized for attacking.
        
        OLD BUG: penalize_attack_reward = -1.0 fired every attack frame
        FIX: Removed this penalty entirely
        """
        print("\n" + "="*70)
        print("TEST 1.3: No Attack Penalty")
        print("="*70)
        
        reward_manager = gen_reward_manager()
        
        # Check that penalize_attack_reward is NOT in reward functions
        has_attack_penalty = 'penalize_attack_reward' in reward_manager.reward_functions
        
        if not has_attack_penalty:
            print("  ✅ PASS: Attack penalty has been removed (good!)")
        else:
            print("  ❌ FAIL: Attack penalty still exists!")
            print("     This will cause agent to learn passivity")
        
        self.results['no_attack_penalty'] = not has_attack_penalty
        return not has_attack_penalty
    
    def run_all_tests(self) -> bool:
        """Run all reward function tests."""
        print("\n" + "🧪 " + "="*68)
        print("REWARD FUNCTION TEST SUITE")
        print("="*70)
        
        tests = [
            self.test_damage_reward_triggers,
            self.test_reward_magnitudes,
            self.test_no_attack_penalty,
        ]
        
        all_passed = all(test() for test in tests)
        
        print("\n" + "="*70)
        print(f"REWARD TESTS: {'✅ ALL PASSED' if all_passed else '❌ SOME FAILED'}")
        print("="*70)
        
        return all_passed


# ============================================================================
# TEST 2: STRATEGY ENCODER VALIDATION
# ============================================================================

class StrategyEncoderTester:
    """Tests that transformer strategy encoder produces meaningful latents."""
    
    def __init__(self):
        self.results = {}
    
    def test_encoder_forward_pass(self) -> bool:
        """
        Test: Transformer encoder should produce valid latent vectors.
        
        Expected:
        - Input: [batch, seq_len, obs_dim]
        - Output: [batch, latent_dim]
        - No NaN or Inf values
        """
        print("\n" + "="*70)
        print("TEST 2.1: Encoder Forward Pass")
        print("="*70)
        
        # Create encoder
        encoder = TransformerStrategyEncoder(
            opponent_obs_dim=32,
            latent_dim=256,
            num_heads=8,
            num_layers=6,
            max_sequence_length=90,
            device=TORCH_DEVICE
        )
        
        # Create dummy input
        batch_size = 4
        seq_len = 90
        obs_dim = 32
        
        dummy_input = torch.randn(batch_size, seq_len, obs_dim, device=TORCH_DEVICE)
        
        print(f"  Input shape: {dummy_input.shape}")
        
        # Forward pass
        with torch.no_grad():
            latent = encoder(dummy_input)
        
        print(f"  Output shape: {latent.shape}")
        print(f"  Expected: [{batch_size}, 256]")
        
        # Validate output
        checks = {
            'correct_shape': latent.shape == (batch_size, 256),
            'no_nan': not torch.isnan(latent).any(),
            'no_inf': not torch.isinf(latent).any(),
            'reasonable_magnitude': (latent.abs().mean() > 0.01) and (latent.abs().mean() < 100),
        }
        
        print("\n  Validation checks:")
        all_pass = True
        for check_name, passed in checks.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"    {status}: {check_name}")
            if not passed:
                all_pass = False
        
        if all_pass:
            print(f"\n  Latent statistics:")
            print(f"    Mean: {latent.mean().item():.4f}")
            print(f"    Std: {latent.std().item():.4f}")
            print(f"    Min: {latent.min().item():.4f}")
            print(f"    Max: {latent.max().item():.4f}")
        
        self.results['encoder_forward_pass'] = all_pass
        return all_pass
    
    def test_encoder_diversity(self) -> bool:
        """
        Test: Encoder should produce DIFFERENT latents for DIFFERENT inputs.
        
        Expected: Latents for random sequences should be distinguishable
        """
        print("\n" + "="*70)
        print("TEST 2.2: Encoder Latent Diversity")
        print("="*70)
        
        encoder = TransformerStrategyEncoder(
            opponent_obs_dim=32,
            latent_dim=256,
            num_heads=8,
            num_layers=6,
            device=TORCH_DEVICE
        )
        
        # Generate 10 different input sequences
        num_samples = 10
        latents = []
        
        for i in range(num_samples):
            dummy_input = torch.randn(1, 90, 32, device=TORCH_DEVICE)
            with torch.no_grad():
                latent = encoder(dummy_input)
            latents.append(latent.cpu().numpy())
        
        latents = np.array(latents).squeeze()  # [num_samples, 256]
        
        # Calculate pairwise distances
        from scipy.spatial.distance import pdist
        distances = pdist(latents, metric='euclidean')
        mean_distance = np.mean(distances)
        min_distance = np.min(distances)
        
        print(f"\n  Pairwise latent distances:")
        print(f"    Mean: {mean_distance:.2f}")
        print(f"    Min: {min_distance:.2f}")
        print(f"    Max: {np.max(distances):.2f}")
        
        # Validation: latents should be sufficiently different
        # (not all the same, not degenerate to same point)
        # NOTE: Untrained encoder will have lower diversity - this is normal!
        # We're just checking it's not degenerate (all zeros or all same)
        success = min_distance > 0.5 and mean_distance > 1.0
        
        if success:
            print(f"\n  ✅ PASS: Encoder produces diverse latents")
            if mean_distance < 3.0:
                print(f"     ℹ️  Note: Low diversity is normal for untrained encoder")
        else:
            print(f"\n  ❌ FAIL: Encoder latents are too similar!")
            print(f"     - May indicate mode collapse or poor initialization")
        
        self.results['encoder_diversity'] = success
        return success
    
    def test_encoder_attention_weights(self) -> bool:
        """
        Test: Attention mechanism should produce valid probability distributions.
        
        Expected: Attention weights sum to 1, all non-negative
        """
        print("\n" + "="*70)
        print("TEST 2.3: Attention Weight Validation")
        print("="*70)
        
        encoder = TransformerStrategyEncoder(
            opponent_obs_dim=32,
            latent_dim=256,
            num_heads=8,
            num_layers=6,
            device=TORCH_DEVICE
        )
        
        dummy_input = torch.randn(1, 90, 32, device=TORCH_DEVICE)
        
        with torch.no_grad():
            latent, attention_info = encoder(dummy_input, return_attention=True)
        
        pooling_attn = attention_info['pooling_attention'].cpu().numpy().squeeze()
        
        print(f"  Attention weights shape: {pooling_attn.shape}")
        print(f"  Sum of weights: {pooling_attn.sum():.4f} (should be ~1.0)")
        print(f"  Min weight: {pooling_attn.min():.4f} (should be >= 0)")
        print(f"  Max weight: {pooling_attn.max():.4f}")
        print(f"  Mean weight: {pooling_attn.mean():.4f}")
        
        # Validation
        checks = {
            'sum_to_one': abs(pooling_attn.sum() - 1.0) < 0.01,
            'all_non_negative': (pooling_attn >= 0).all(),
            'not_uniform': pooling_attn.std() > 0.001,  # Should focus on some frames more
        }
        
        print("\n  Validation checks:")
        all_pass = True
        for check_name, passed in checks.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"    {status}: {check_name}")
            if not passed:
                all_pass = False
        
        self.results['attention_weights'] = all_pass
        return all_pass
    
    def run_all_tests(self) -> bool:
        """Run all strategy encoder tests."""
        print("\n" + "🧪 " + "="*68)
        print("STRATEGY ENCODER TEST SUITE")
        print("="*70)
        
        tests = [
            self.test_encoder_forward_pass,
            self.test_encoder_diversity,
            self.test_encoder_attention_weights,
        ]
        
        all_passed = all(test() for test in tests)
        
        print("\n" + "="*70)
        print(f"ENCODER TESTS: {'✅ ALL PASSED' if all_passed else '❌ SOME FAILED'}")
        print("="*70)
        
        return all_passed


# ============================================================================
# TEST 3: POLICY INTEGRATION TESTS
# ============================================================================

class PolicyIntegrationTester:
    """Tests that policy properly integrates with strategy encoder."""
    
    def __init__(self, checkpoint_path: str = None):
        self.checkpoint_path = checkpoint_path
        self.results = {}
    
    def test_agent_initialization(self) -> bool:
        """
        Test: TransformerStrategyAgent should initialize without errors.
        """
        print("\n" + "="*70)
        print("TEST 3.1: Agent Initialization")
        print("="*70)
        
        try:
            # Create agent from scratch
            agent = TransformerStrategyAgent(
                file_path=self.checkpoint_path,
                latent_dim=256,
                num_heads=8,
                num_layers=6,
                sequence_length=90,
            )
            
            # Get env info (needed for initialization)
            from environment.environment import WarehouseBrawl
            from shimmy.openai_gym_compatibility import _convert_space
            import gymnasium
            
            test_env = WarehouseBrawl(resolution=CameraResolution.LOW)
            
            # Wrap for Gymnasium compatibility (SB3 expects Gymnasium)
            class GymCompatWrapper:
                def __init__(self, env):
                    self._env = env
                    # Convert spaces to Gymnasium format
                    self.observation_space = _convert_space(env.observation_space)
                    self.action_space = _convert_space(env.action_space)
                def __getattr__(self, name):
                    return getattr(self._env, name)
            
            test_env_wrapped = GymCompatWrapper(test_env)
            agent.get_env_info(test_env_wrapped)
            
            print("  ✅ PASS: Agent initialized successfully")
            print(f"     - Observation space: {agent.observation_space.shape}")
            print(f"     - Action space: {agent.action_space.shape}")
            print(f"     - Strategy encoder: {agent.strategy_encoder is not None}")
            print(f"     - Policy model: {agent.model is not None}")
            
            test_env.close()
            self.results['agent_init'] = True
            return True
            
        except Exception as e:
            print(f"  ❌ FAIL: Agent initialization failed!")
            print(f"     Error: {e}")
            self.results['agent_init'] = False
            return False
    
    def test_strategy_latent_updates(self) -> bool:
        """
        Test: Strategy latent should update as opponent history grows.
        """
        print("\n" + "="*70)
        print("TEST 3.2: Strategy Latent Updates")
        print("="*70)
        
        try:
            agent = TransformerStrategyAgent(
                file_path=self.checkpoint_path,
                latent_dim=256,
                num_heads=8,
                num_layers=6,
                sequence_length=90,
            )
            
            from environment.environment import WarehouseBrawl
            from shimmy.openai_gym_compatibility import _convert_space
            
            test_env = WarehouseBrawl(resolution=CameraResolution.LOW)
            
            # Wrap for Gymnasium compatibility
            class GymCompatWrapper:
                def __init__(self, env):
                    self._env = env
                    self.observation_space = _convert_space(env.observation_space)
                    self.action_space = _convert_space(env.action_space)
                def __getattr__(self, name):
                    return getattr(self._env, name)
            
            test_env_wrapped = GymCompatWrapper(test_env)
            agent.get_env_info(test_env_wrapped)
            
            # Reset agent
            agent.reset()
            
            # Simulate opponent observations
            obs_dim = agent.opponent_obs_dim
            print(f"  Opponent observation dimension: {obs_dim}")
            
            # Feed observations one by one
            latent_norms = []
            for i in range(20):
                # Create dummy observation
                full_obs = np.random.randn(agent.observation_space.shape[0])
                
                # Predict action (this updates opponent history internally)
                action = agent.predict(full_obs)
                
                # Check latent
                if agent.current_strategy_latent is not None:
                    norm = torch.norm(agent.current_strategy_latent).item()
                    latent_norms.append(norm)
                    if i == 10:
                        print(f"  Step {i}: Latent norm = {norm:.4f}")
                
            print(f"\n  Latent updates:")
            print(f"    First latent at step: {len(latent_norms) - len(latent_norms) + 1 if latent_norms else 'Never'}")
            print(f"    Total updates: {len(latent_norms)}")
            
            # Validation: latent should start updating after ~10 frames
            success = len(latent_norms) >= 10
            
            if success:
                print(f"  ✅ PASS: Strategy latent updates correctly")
            else:
                print(f"  ❌ FAIL: Strategy latent not updating!")
            
            test_env.close()
            self.results['latent_updates'] = success
            return success
            
        except Exception as e:
            print(f"  ❌ FAIL: Strategy latent test failed!")
            print(f"     Error: {e}")
            self.results['latent_updates'] = False
            return False
    
    def run_all_tests(self) -> bool:
        """Run all policy integration tests."""
        print("\n" + "🧪 " + "="*68)
        print("POLICY INTEGRATION TEST SUITE")
        print("="*70)
        
        tests = [
            self.test_agent_initialization,
            self.test_strategy_latent_updates,
        ]
        
        all_passed = all(test() for test in tests)
        
        print("\n" + "="*70)
        print(f"POLICY TESTS: {'✅ ALL PASSED' if all_passed else '❌ SOME FAILED'}")
        print("="*70)
        
        return all_passed


# ============================================================================
# TEST 4: END-TO-END VALIDATION
# ============================================================================

class EndToEndTester:
    """Tests that trained agent can beat ConstantAgent consistently."""
    
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        self.results = {}
    
    def test_beats_constant_agent(self, num_matches: int = 10) -> bool:
        """
        Test: Agent should beat ConstantAgent with 90%+ win rate.
        
        This is the CRITICAL success criterion for curriculum Stage 1.
        """
        print("\n" + "="*70)
        print(f"TEST 4.1: Beat ConstantAgent ({num_matches} matches)")
        print("="*70)
        
        if not self.checkpoint_path or not os.path.exists(self.checkpoint_path):
            print(f"  ⚠️  SKIP: No checkpoint provided or file not found")
            print(f"     Path: {self.checkpoint_path}")
            self.results['beats_constant_agent'] = None
            return False
        
        try:
            # Load agent
            agent = TransformerStrategyAgent(
                file_path=self.checkpoint_path,
                latent_dim=256,
                num_heads=8,
                num_layers=6,
                sequence_length=90,
            )
            
            print(f"  Loaded checkpoint: {self.checkpoint_path}")
            
            # Run matches
            wins = 0
            total_damage_dealt = 0
            total_damage_taken = 0
            
            for i in range(num_matches):
                match_stats = env_run_match(
                    agent,
                    partial(ConstantAgent),
                    max_timesteps=30*60,  # 60 seconds
                    resolution=CameraResolution.LOW,
                    train_mode=True
                )
                
                won = match_stats.player1_result == Result.WIN
                if won:
                    wins += 1
                
                damage_dealt = match_stats.player2.damage_taken
                damage_taken = match_stats.player1.damage_taken
                
                total_damage_dealt += damage_dealt
                total_damage_taken += damage_taken
                
                print(f"  Match {i+1}/{num_matches}: {'WIN' if won else 'LOSS'} " +
                      f"(damage: {damage_dealt:.0f} dealt / {damage_taken:.0f} taken)")
            
            # Calculate metrics
            win_rate = (wins / num_matches) * 100
            avg_damage_dealt = total_damage_dealt / num_matches
            avg_damage_taken = total_damage_taken / num_matches
            damage_ratio = avg_damage_dealt / max(avg_damage_taken, 1.0)
            
            print(f"\n  Results:")
            print(f"    Win Rate: {win_rate:.1f}% ({wins}/{num_matches} matches)")
            print(f"    Avg Damage Dealt: {avg_damage_dealt:.1f}")
            print(f"    Avg Damage Taken: {avg_damage_taken:.1f}")
            print(f"    Damage Ratio: {damage_ratio:.2f}")
            
            # Success criteria: 90%+ win rate OR 80%+ win rate with high damage ratio
            success = (win_rate >= 90.0) or (win_rate >= 80.0 and damage_ratio >= 3.0)
            
            if success:
                print(f"\n  ✅ PASS: Agent reliably beats ConstantAgent!")
                print(f"     Ready for curriculum Stage 2 (BasedAgent training)")
            else:
                print(f"\n  ❌ FAIL: Agent cannot beat ConstantAgent consistently")
                print(f"     Required: 90%+ win rate (got {win_rate:.1f}%)")
                print(f"     OR 80%+ win rate with 3.0+ damage ratio (got {damage_ratio:.2f})")
            
            self.results['beats_constant_agent'] = success
            return success
            
        except Exception as e:
            print(f"  ❌ FAIL: Test crashed!")
            print(f"     Error: {e}")
            import traceback
            traceback.print_exc()
            self.results['beats_constant_agent'] = False
            return False
    
    def test_positive_rewards(self, num_matches: int = 3) -> bool:
        """
        Test: Agent should accumulate POSITIVE rewards when playing.
        
        This validates that the reward function is properly shaped.
        """
        print("\n" + "="*70)
        print(f"TEST 4.2: Positive Reward Accumulation")
        print("="*70)
        
        if not self.checkpoint_path or not os.path.exists(self.checkpoint_path):
            print(f"  ⚠️  SKIP: No checkpoint provided")
            self.results['positive_rewards'] = None
            return False
        
        try:
            agent = TransformerStrategyAgent(file_path=self.checkpoint_path)
            reward_manager = gen_reward_manager()
            
            print(f"  Running {num_matches} matches with reward tracking...")
            
            total_rewards = []
            
            for i in range(num_matches):
                reward_manager.reset()
                
                match_stats = env_run_match(
                    agent,
                    partial(ConstantAgent),
                    max_timesteps=30*30,  # 30 seconds
                    resolution=CameraResolution.LOW,
                    reward_manager=reward_manager,
                    train_mode=True
                )
                
                total_reward = reward_manager.total_reward
                total_rewards.append(total_reward)
                
                print(f"  Match {i+1}: Total reward = {total_reward:.2f}")
            
            avg_reward = np.mean(total_rewards)
            min_reward = np.min(total_rewards)
            
            print(f"\n  Results:")
            print(f"    Average reward: {avg_reward:.2f}")
            print(f"    Min reward: {min_reward:.2f}")
            
            # Success: average reward should be positive
            success = avg_reward > 0 and min_reward > -50
            
            if success:
                print(f"\n  ✅ PASS: Agent accumulates positive rewards")
            else:
                print(f"\n  ❌ FAIL: Agent has negative rewards!")
                print(f"     This indicates reward function is still broken")
            
            self.results['positive_rewards'] = success
            return success
            
        except Exception as e:
            print(f"  ❌ FAIL: Test crashed!")
            print(f"     Error: {e}")
            self.results['positive_rewards'] = False
            return False
    
    def run_all_tests(self) -> bool:
        """Run all end-to-end tests."""
        print("\n" + "🧪 " + "="*68)
        print("END-TO-END VALIDATION SUITE")
        print("="*70)
        
        tests = [
            lambda: self.test_beats_constant_agent(num_matches=10),
            lambda: self.test_positive_rewards(num_matches=3),
        ]
        
        all_passed = all(test() for test in tests)
        
        print("\n" + "="*70)
        print(f"E2E TESTS: {'✅ ALL PASSED' if all_passed else '❌ SOME FAILED'}")
        print("="*70)
        
        return all_passed


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_all_tests(checkpoint_path: str = None):
    """
    Run complete test suite to validate training pipeline.
    
    Args:
        checkpoint_path: Path to trained agent checkpoint (optional for early tests)
    """
    print("\n" + "🚀 " + "="*68)
    print("COMPREHENSIVE TRAINING VALIDATION SUITE")
    print("="*70)
    print(f"Device: {TORCH_DEVICE}")
    print(f"Checkpoint: {checkpoint_path or 'None (testing from scratch)'}")
    print("="*70)
    
    results = {}
    
    # Test Suite 1: Reward Functions (no checkpoint needed)
    reward_tester = RewardFunctionTester()
    results['reward_functions'] = reward_tester.run_all_tests()
    
    # Test Suite 2: Strategy Encoder (no checkpoint needed)
    encoder_tester = StrategyEncoderTester()
    results['strategy_encoder'] = encoder_tester.run_all_tests()
    
    # Test Suite 3: Policy Integration (can use checkpoint if available)
    policy_tester = PolicyIntegrationTester(checkpoint_path)
    results['policy_integration'] = policy_tester.run_all_tests()
    
    # Test Suite 4: End-to-End (requires checkpoint)
    if checkpoint_path:
        e2e_tester = EndToEndTester(checkpoint_path)
        results['end_to_end'] = e2e_tester.run_all_tests()
    else:
        print("\n" + "⚠️  " + "="*68)
        print("SKIPPING END-TO-END TESTS (no checkpoint provided)")
        print("="*70)
        print("  Run these tests after curriculum Stage 1 training:")
        print("  python user/test_training_components.py --checkpoint path/to/checkpoint.zip")
        print("="*70)
        results['end_to_end'] = None
    
    # Final summary
    print("\n" + "🎯 " + "="*68)
    print("FINAL TEST SUMMARY")
    print("="*70)
    
    for suite_name, passed in results.items():
        if passed is None:
            status = "⚠️  SKIPPED"
        elif passed:
            status = "✅ PASSED"
        else:
            status = "❌ FAILED"
        print(f"  {status}: {suite_name}")
    
    all_tested = [v for v in results.values() if v is not None]
    all_passed = all(all_tested) if all_tested else False
    
    print("="*70)
    if all_passed:
        print("✅ ALL TESTS PASSED - READY FOR FULL TRAINING!")
        print("\nNext steps:")
        print("  1. Training looks good! Proceed to curriculum Stage 2")
        print("  2. Update TRAIN_CONFIG = TRAIN_CONFIG_CURRICULUM_STAGE2")
        print("  3. Set load_path to Stage 1 checkpoint")
        print("  4. Run training again")
    else:
        print("❌ SOME TESTS FAILED - FIX BEFORE PROCEEDING!")
        print("\nDebugging steps:")
        print("  1. Review failed test outputs above")
        print("  2. Check reward_breakdown.csv in training logs")
        print("  3. Verify agent is attacking (not passive)")
        print("  4. Re-run tests after fixes")
    print("="*70 + "\n")
    
    return all_passed


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test training components")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained agent checkpoint (for end-to-end tests)"
    )
    
    args = parser.parse_args()
    
    success = run_all_tests(checkpoint_path=args.checkpoint)
    
    sys.exit(0 if success else 1)

