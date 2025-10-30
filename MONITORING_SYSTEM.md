# Optimized Training Monitoring System

## Overview

This document describes the hierarchical, lightweight monitoring system implemented for tracking training performance with minimal overhead.

## Architecture

The monitoring system is built around a **Stable-Baselines3 callback** (`TrainingMonitorCallback`) that integrates seamlessly with the PPO training loop. It uses a hierarchical logging strategy to balance insight with performance.

## Logging Hierarchy

### 1. **Frame-Level Alerts (Every Step - Console Only)**

**Frequency:** Continuous  
**Output:** Console only (no disk I/O)  
**Purpose:** Catch critical issues immediately

**Tracked Metrics:**
- **Gradient Explosions:** Loss > 100
- **NaN Values:** NaN in loss
- **Reward Spikes:** Reward > 1000x normal

**Example Output:**
```
⚠️  ALERT: Gradient explosion detected (loss=152.34) at step 12500
🚨 CRITICAL: NaN detected in loss at step 15000!
```

### 2. **Light Logging (Every 500-1000 Steps)**

**Frequency:** Every 500 steps (configurable via `light_log_freq`)  
**Output:** Console + CSV (batched writes)  
**Purpose:** Track reward composition, transformer health, PPO metrics

**Tracked Metrics:**

#### Reward Breakdown
- Each reward term contribution
- Which terms are currently active
- Total reward

**CSV Output:** `reward_breakdown.csv`
```csv
step,danger_zone_reward,damage_interaction_reward,penalize_attack_reward,...,total_reward
500,-0.025000,0.142857,-0.000800,...,0.117057
```

#### Transformer Health Check
- **Latent Vector Norm:** Measures if encoder is producing meaningful outputs
  - Mean and standard deviation tracked
  - Low norm → encoder may be dead
  - High variance → diverse strategy recognition
- **Attention Entropy:** Measures if transformer is focusing or diffuse
  - High entropy → attention is spread out (may not be learning patterns)
  - Low entropy → attention is focused (learning specific patterns)

#### PPO Core Metrics (from SB3 Logger)
- **Policy Loss:** Gradient magnitude for policy network
- **Value Loss:** Prediction error for value function
- **Explained Variance:** Most important metric - how well value function predicts returns
  - Close to 1.0 → Learning well
  - Near 0 or negative → Not learning

**Example Console Output:**
```
--- Step 1000 ---
  Reward Breakdown: danger_zone_reward=-0.025, damage_interaction_reward=0.143, penalize_attack_reward=-0.001
  Active Terms: danger_zone_reward, damage_interaction_reward, penalize_attack_reward, holding_more_than_3_keys
  Transformer: Latent Norm=12.543 (±2.134), Attention Entropy=2.843
  PPO: Policy Loss=0.0234, Value Loss=1.2456, Explained Var=0.847
```

### 3. **Quick Evaluation (Every 5000 Steps)**

**Frequency:** Every 5000 steps (configurable via `eval_freq`)  
**Output:** Console + detailed analysis  
**Purpose:** Validate learning progress with actual matches

**Tracked Metrics:**

#### Win Rate Spot Check
- Runs 3 quick matches vs BasedAgent (configurable)
- Measures current performance

#### Behavior Metrics Summary
- Average episode reward (last 10 episodes)
- Average episode length
- Damage dealt/taken ratio
- Attack frequency

#### Sanity Checks
- **Is reward stuck?** Checks if reward has little variation
- **Is agent improving?** Compares recent vs early episode rewards
- **Are losses exploding?** Checks for very high loss values

**Example Console Output:**
```
======================================================================
🔍 QUICK EVALUATION (Step 5000)
======================================================================
  Win Rate: 66.7% (2/3 matches)
  Damage Ratio: 1.84
  Avg Episode Reward (last 10): 45.23
  Avg Episode Length (last 10): 1847.2
  ✓ Sanity checks passed
======================================================================
```

### 4. **Checkpoint Benchmarks (Every 50k-100k Steps)**

**Frequency:** At each checkpoint save (e.g., every 50k or 100k steps)  
**Output:** Console + CSV  
**Purpose:** Comprehensive performance evaluation

**Tracked Metrics:**

#### Performance Benchmarks
- **vs BasedAgent:** Win rate over 5 matches
- **vs ConstantAgent:** Win rate over 5 matches
- **Average Damage Ratio:** Across all matches

#### Strategy Diversity Score
- Calculated from recent latent vectors
- Measures variation in strategy representations
- Higher score = more diverse strategies employed

**CSV Output:** `checkpoint_benchmarks.csv`
```csv
checkpoint_step,vs_based_winrate,vs_constant_winrate,avg_damage_ratio,strategy_diversity_score,eval_time_sec
50000,60.00,80.00,1.85,3.4521,45.2
100000,75.00,90.00,2.34,4.1234,47.8
```

**Example Console Output:**
```
======================================================================
🎯 CHECKPOINT BENCHMARK (Step 50000)
======================================================================
Testing vs BasedAgent (5 matches)...
Testing vs ConstantAgent (5 matches)...
----------------------------------------------------------------------
  vs BasedAgent:       60.0% wins
  vs ConstantAgent:    80.0% wins
  Avg Damage Ratio:    1.85
  Strategy Diversity:  3.452
  Benchmark Time:      45.2s
======================================================================
```

## Key Components

### 1. TransformerHealthMonitor

**Purpose:** Tracks transformer encoder health  
**Memory:** Circular buffers (100 most recent values)  
**Metrics:**
- Latent vector norms
- Attention entropy

**Methods:**
- `update(agent)`: Extract metrics from agent
- `get_stats()`: Get current statistics

### 2. RewardBreakdownTracker

**Purpose:** Tracks reward term contributions  
**Strategy:** Accumulate in memory, batch write to CSV  
**CSV:** `reward_breakdown.csv`

**Methods:**
- `compute_breakdown(env)`: Calculate reward breakdown
- `record_step(step, breakdown)`: Record in memory
- `flush_to_csv()`: Write accumulated data (every 10 light logs)
- `get_active_terms()`: Get list of activated reward terms

### 3. PerformanceBenchmark

**Purpose:** Comprehensive evaluation at checkpoints  
**Opponents:** BasedAgent, ConstantAgent  
**CSV:** `checkpoint_benchmarks.csv`

**Methods:**
- `run_benchmark(agent, checkpoint_step)`: Full benchmark
- `_run_matches(agent, opponent_factory, num_matches)`: Run evaluation matches
- `_calculate_strategy_diversity(agent)`: Calculate diversity score
- `record_latent_vector(agent)`: Track latent for diversity

### 4. TrainingMonitorCallback

**Purpose:** Main SB3 callback coordinating all monitoring  
**Integration:** Hooks into training loop via SB3 callback mechanism

**Methods:**
- `_on_step()`: Called every training step
- `_check_frame_alerts()`: Frame-level alert checks
- `_light_logging()`: Light logging every N steps
- `_quick_evaluation()`: Quick eval every N steps
- `_checkpoint_benchmark()`: Benchmark at checkpoints
- `_run_sanity_checks()`: Sanity check logic

## Configuration

### In TRAIN_CONFIG

```python
"training": {
    "resolution": CameraResolution.LOW,
    "timesteps": 50_000,
    "logging": TrainLogging.PLOT,
    
    # Monitoring settings
    "enable_debug": True,        # Enable monitoring (default: True)
    "light_log_freq": 500,       # Light logging frequency
    "eval_freq": 5000,           # Quick evaluation frequency
    "eval_episodes": 3,          # Matches per quick evaluation
}
```

### Default Values

- **Light Logging:** Every 500 steps
- **Quick Evaluation:** Every 5000 steps
- **Eval Matches:** 3 per evaluation
- **Checkpoint Benchmarks:** 5 matches per opponent type

## Output Files

All files are saved to the experiment directory (e.g., `checkpoints/test_50k_t4/`):

1. **`monitor.csv`** - SB3's default episode tracking (already exists)
   - Episode rewards
   - Episode lengths
   - Timesteps

2. **`reward_breakdown.csv`** - Reward term contributions
   - Per-step breakdown of each reward term
   - Batched writes (every 5000 steps)

3. **`episode_summary.csv`** - Episode-level summary
   - Step, episode number, reward, length, damage ratio

4. **`checkpoint_benchmarks.csv`** - Checkpoint performance
   - Win rates vs different opponents
   - Damage ratios
   - Strategy diversity scores

## Performance Impact

The monitoring system is designed for **minimal overhead**:

### Frame-Level (Every Step)
- Only in-memory checks
- No disk I/O
- ~0.1ms overhead per step

### Light Logging (Every 500 Steps)
- Minimal computation (reward breakdown already needed)
- Batched CSV writes (every 5000 steps)
- ~5ms overhead per log

### Quick Evaluation (Every 5000 Steps)
- Runs 3 matches (~60 seconds total)
- Pauses training temporarily
- ~1% of total training time for 50k run

### Checkpoint Benchmarks (Every 50k-100k Steps)
- Runs 10 matches (~120 seconds total)
- Only happens at checkpoints (infrequent)
- Negligible impact on long runs

### Total Overhead
- **50k training run:** ~3-5% overhead
- **10M training run:** ~1-2% overhead

## Usage Example

### Enable Monitoring (Default)

```python
# In train_agent.py main()
TRAIN_CONFIG = TRAIN_CONFIG_TEST  # or TRAIN_CONFIG_10M

# Monitoring enabled by default
# To disable: TRAIN_CONFIG["training"]["enable_debug"] = False
```

### Custom Frequencies

```python
TRAIN_CONFIG["training"]["light_log_freq"] = 1000  # Every 1000 steps
TRAIN_CONFIG["training"]["eval_freq"] = 10000      # Every 10k steps
TRAIN_CONFIG["training"]["eval_episodes"] = 5      # 5 matches per eval
```

### Disable Monitoring

```python
TRAIN_CONFIG["training"]["enable_debug"] = False
```

## Interpreting Results

### Healthy Training Signs

✅ **Reward Breakdown:**
- Multiple reward terms active
- Terms contributing in expected proportions

✅ **Transformer Health:**
- Latent norm: 5-20 (not too high, not too low)
- Attention entropy: 2-4 (focused but not too narrow)

✅ **PPO Metrics:**
- Explained variance: 0.7-0.95 (high is good!)
- Policy loss: Decreasing over time
- Value loss: Stable or decreasing

✅ **Sanity Checks:**
- No stuck rewards
- Agent improving over time
- No loss explosions

### Warning Signs

⚠️ **Reward Issues:**
- Single term dominating (>90% contribution)
- Terms never activating (might be broken)
- Reward stuck at same value

⚠️ **Transformer Issues:**
- Latent norm < 1 (encoder may be dead)
- Attention entropy > 5 (not learning patterns)

⚠️ **PPO Issues:**
- Explained variance < 0.3 (not learning)
- Loss > 100 (gradient explosion)
- NaN values (training broken)

⚠️ **Performance Issues:**
- Win rate not improving
- Always losing vs scripted opponents
- No strategy diversity (all latent vectors similar)

## Advanced: Accessing Metrics Programmatically

If you want to access metrics during training:

```python
# After training
if monitor_callback is not None:
    # Access transformer health
    health_stats = monitor_callback.transformer_monitor.get_stats()
    print(f"Latent norm: {health_stats['latent_norm_mean']}")
    
    # Access reward breakdown
    active_terms = monitor_callback.reward_tracker.get_active_terms()
    print(f"Active reward terms: {active_terms}")
    
    # Access benchmark results (read from CSV)
    import pandas as pd
    benchmarks = pd.read_csv(monitor_callback.benchmark.csv_path)
    print(benchmarks)
```

## Comparison to Old System

### Old System (Removed)
- ❌ Separate classes (`RewardDebugger`, `BehaviorMetrics`, `QuickEvaluator`, `SanityChecker`)
- ❌ Not integrated with training loop (manual calls needed)
- ❌ Heavy disk I/O (wrote every step)
- ❌ Redundant tracking across multiple classes
- ❌ No hierarchical strategy

### New System
- ✅ Single unified callback (`TrainingMonitorCallback`)
- ✅ Integrated with SB3 training loop (automatic)
- ✅ Lightweight (batched writes, in-memory buffers)
- ✅ Clean separation of concerns (focused components)
- ✅ Hierarchical logging (different frequencies for different metrics)

## Troubleshooting

### Issue: "Callback not working"
**Solution:** Ensure `enable_debug=True` in training config and agent's `learn()` method accepts `callback` parameter.

### Issue: "CSV files empty"
**Solution:** CSV files are flushed periodically. Wait for light logging flush (every 5000 steps) or training completion.

### Issue: "Evaluation taking too long"
**Solution:** Reduce `eval_episodes` or increase `eval_freq` to evaluate less often.

### Issue: "Too much console spam"
**Solution:** This is expected! The system provides real-time feedback. Redirect to file: `python train_agent.py > training.log 2>&1`

### Issue: "Monitoring slowing down training"
**Solution:** Increase logging frequencies:
```python
TRAIN_CONFIG["training"]["light_log_freq"] = 2000
TRAIN_CONFIG["training"]["eval_freq"] = 20000
```

## Future Enhancements

Possible future additions:
- [ ] TensorBoard integration for richer visualization
- [ ] Async evaluation (don't block training)
- [ ] Custom alert thresholds (configurable warning levels)
- [ ] Automatic hyperparameter suggestions based on metrics
- [ ] Strategy clustering visualization (t-SNE of latent vectors)

