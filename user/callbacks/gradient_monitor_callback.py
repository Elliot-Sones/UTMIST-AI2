"""
Gradient Monitoring Callback: Detect training instabilities early.

This callback monitors gradients during training to catch:
- NaN/Inf gradients (indicates divergence)
- Exploding gradients (very large norms)
- Vanishing gradients (very small norms)
- Strategy encoder gradient flow
"""

import torch
import numpy as np
from typing import Dict, Optional
from stable_baselines3.common.callbacks import BaseCallback


class GradientMonitorCallback(BaseCallback):
    """
    Callback that monitors gradients for training stability.

    Tracks:
    - Gradient norms for policy, value, and strategy encoder
    - NaN/Inf detection with automatic stopping
    - Gradient statistics over time
    """

    def __init__(
        self,
        check_frequency: int = 10,  # Check every N updates
        verbose: int = 1,
        max_grad_norm_threshold: float = 100.0,  # Warning threshold
        stop_on_nan: bool = True,  # Stop training if NaN detected
    ):
        super().__init__(verbose)

        self.check_frequency = check_frequency
        self.max_grad_norm_threshold = max_grad_norm_threshold
        self.stop_on_nan = stop_on_nan

        # Statistics
        self.num_updates = 0
        self.num_nan_detections = 0
        self.num_inf_detections = 0
        self.num_exploding_grad_warnings = 0

        # Track gradient norms
        self.policy_grad_norms = []
        self.value_grad_norms = []
        self.strategy_encoder_grad_norms = []

    def _on_training_start(self) -> None:
        """Called at the beginning of training."""
        if self.verbose > 0:
            print(f"✓ Gradient monitoring active (checking every {self.check_frequency} updates)")
            print(f"  Stop on NaN: {self.stop_on_nan}")
            print(f"  Exploding gradient threshold: {self.max_grad_norm_threshold}")

    def _on_step(self) -> bool:
        """Called after each environment step."""
        # We'll check gradients after training updates, not after each step
        return True

    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout (after training update)."""
        self.num_updates += 1

        if self.num_updates % self.check_frequency == 0:
            has_nan_inf = self._check_gradients()

            if has_nan_inf and self.stop_on_nan:
                print("\n" + "="*80)
                print("❌ TRAINING STOPPED: NaN/Inf detected in gradients!")
                print("="*80)
                print("This indicates training divergence. Possible causes:")
                print("  1. Learning rate too high")
                print("  2. Unstable value function (try enabling clip_range_vf)")
                print("  3. Observation/reward not properly normalized")
                print("  4. BatchNorm instability (consider LayerNorm)")
                print("="*80 + "\n")
                return False  # Stop training

        return True

    def _check_gradients(self) -> bool:
        """
        Check gradients for NaN/Inf and excessive norms.

        Returns:
            True if NaN/Inf detected, False otherwise
        """
        has_nan_inf = False

        if not hasattr(self.model, 'policy'):
            return False

        policy = self.model.policy

        # Collect gradient statistics
        policy_grads = []
        value_grads = []
        strategy_encoder_grads = []

        for name, param in policy.named_parameters():
            if param.grad is not None:
                grad = param.grad

                # Check for NaN/Inf
                if torch.isnan(grad).any():
                    self.num_nan_detections += 1
                    has_nan_inf = True
                    if self.verbose > 0:
                        print(f"❌ NaN detected in gradient: {name}")

                if torch.isinf(grad).any():
                    self.num_inf_detections += 1
                    has_nan_inf = True
                    if self.verbose > 0:
                        print(f"❌ Inf detected in gradient: {name}")

                # Compute gradient norm
                grad_norm = grad.norm().item()

                # Categorize by component
                if 'strategy_encoder' in name.lower():
                    strategy_encoder_grads.append(grad_norm)
                elif 'value' in name.lower() or 'vf' in name.lower():
                    value_grads.append(grad_norm)
                else:
                    policy_grads.append(grad_norm)

                # Check for exploding gradients
                if grad_norm > self.max_grad_norm_threshold:
                    self.num_exploding_grad_warnings += 1
                    if self.verbose > 0 and self.num_exploding_grad_warnings <= 5:
                        print(f"⚠ Large gradient in {name}: {grad_norm:.2f}")

        # Store average norms
        if policy_grads:
            self.policy_grad_norms.append(np.mean(policy_grads))
        if value_grads:
            self.value_grad_norms.append(np.mean(value_grads))
        if strategy_encoder_grads:
            self.strategy_encoder_grad_norms.append(np.mean(strategy_encoder_grads))

        # Log summary
        if self.verbose > 0 and not has_nan_inf:
            print(f"\nGradient Stats @ update {self.num_updates}:")
            if policy_grads:
                print(f"  Policy:          avg={np.mean(policy_grads):.4f}  "
                      f"max={np.max(policy_grads):.4f}  "
                      f"min={np.min(policy_grads):.6f}")
            if value_grads:
                print(f"  Value:           avg={np.mean(value_grads):.4f}  "
                      f"max={np.max(value_grads):.4f}  "
                      f"min={np.min(value_grads):.6f}")
            if strategy_encoder_grads:
                print(f"  Strategy Encoder: avg={np.mean(strategy_encoder_grads):.4f}  "
                      f"max={np.max(strategy_encoder_grads):.4f}  "
                      f"min={np.min(strategy_encoder_grads):.6f}")

            # Check for vanishing gradients
            all_grads = policy_grads + value_grads + strategy_encoder_grads
            if all_grads and np.max(all_grads) < 1e-6:
                print("  ⚠ Warning: Very small gradients detected (possible vanishing gradient)")

        return has_nan_inf

    def _on_training_end(self) -> None:
        """Called at the end of training."""
        if self.verbose > 0:
            print("\n" + "="*80)
            print("GRADIENT MONITORING SUMMARY")
            print("="*80)
            print(f"Total updates monitored: {self.num_updates}")
            print(f"NaN detections: {self.num_nan_detections}")
            print(f"Inf detections: {self.num_inf_detections}")
            print(f"Exploding gradient warnings: {self.num_exploding_grad_warnings}")

            if self.policy_grad_norms:
                print(f"\nAverage gradient norms:")
                print(f"  Policy:          {np.mean(self.policy_grad_norms):.4f}")
            if self.value_grad_norms:
                print(f"  Value:           {np.mean(self.value_grad_norms):.4f}")
            if self.strategy_encoder_grad_norms:
                print(f"  Strategy Encoder: {np.mean(self.strategy_encoder_grad_norms):.4f}")

            print("="*80 + "\n")


def create_gradient_monitor_callback(
    check_frequency: int = 10,
    verbose: int = 1,
    max_grad_norm_threshold: float = 100.0,
    stop_on_nan: bool = True,
) -> GradientMonitorCallback:
    """
    Factory function to create GradientMonitorCallback.

    Args:
        check_frequency: Check gradients every N updates (default: 10)
        verbose: Verbosity level (default: 1)
        max_grad_norm_threshold: Warning threshold for large gradients (default: 100.0)
        stop_on_nan: Stop training if NaN detected (default: True)

    Returns:
        GradientMonitorCallback instance
    """
    return GradientMonitorCallback(
        check_frequency=check_frequency,
        verbose=verbose,
        max_grad_norm_threshold=max_grad_norm_threshold,
        stop_on_nan=stop_on_nan,
    )
