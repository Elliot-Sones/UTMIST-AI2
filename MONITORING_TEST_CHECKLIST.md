# Monitoring System Testing Checklist

## Pre-Testing Setup

- [ ] Ensure `user/train_agent.py` has no linter errors
- [ ] Verify all dependencies installed (`stable-baselines3`, `sb3-contrib`, etc.)
- [ ] Check TRAIN_CONFIG is set to TRAIN_CONFIG_TEST (50k run for testing)

## Test 1: Basic Functionality

### Setup
```bash
cd /Users/elliot18/.cursor/worktrees/UTMIST-AI2/ZxdMY
python user/train_agent.py
```

### Expected Output at Start
- [ ] Device detection message (CUDA/MPS/CPU)
- [ ] "MONITORED TRAINING" mode message
- [ ] Monitoring system initialization message showing:
  - [ ] Light logging frequency (500 steps)
  - [ ] Evaluation frequency (5000 steps)
  - [ ] Checkpoint save frequency
  - [ ] Log directory path
- [ ] "✓ Training monitor initialized" message

### Expected During Training

#### Every 500 Steps (Light Logging)
- [ ] "--- Step N ---" header
- [ ] Reward breakdown line showing term values
- [ ] Active terms list
- [ ] Transformer health metrics (Latent Norm, Attention Entropy)
- [ ] PPO metrics (Policy Loss, Value Loss, Explained Var)

Example:
```
--- Step 500 ---
  Reward Breakdown: danger_zone_reward=-0.025, damage_interaction_reward=0.143
  Active Terms: danger_zone_reward, damage_interaction_reward, penalize_attack_reward
  Transformer: Latent Norm=12.543 (±2.134), Attention Entropy=2.843
  PPO: Policy Loss=0.0234, Value Loss=1.2456, Explained Var=0.847
```

#### Every 5000 Steps (Quick Evaluation)
- [ ] "🔍 QUICK EVALUATION" header
- [ ] Win rate percentage and match count
- [ ] Damage ratio
- [ ] Average episode reward (last 10)
- [ ] Average episode length (last 10)
- [ ] Sanity check status (✓ or ⚠️)
- [ ] Closing separator line

Example:
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

#### At Checkpoint Saves
- [ ] "🎯 CHECKPOINT BENCHMARK" header
- [ ] "Testing vs BasedAgent" message
- [ ] "Testing vs ConstantAgent" message
- [ ] Win rates for both opponents
- [ ] Average damage ratio
- [ ] Strategy diversity score
- [ ] Benchmark time
- [ ] Closing separator line

### Expected at End
- [ ] "✅ TRAINING COMPLETE - MONITORING SUMMARY" header
- [ ] Total steps tracked
- [ ] CSV file paths listed:
  - [ ] reward_breakdown.csv
  - [ ] episode_summary.csv
  - [ ] checkpoint_benchmarks.csv
- [ ] Warning summary (if any issues detected)
- [ ] "✓ No critical issues detected" (if healthy)

### Verify Output Files
```bash
ls checkpoints/test_50k_t4/
```

Expected files:
- [ ] `monitor.csv` (SB3 default)
- [ ] `reward_breakdown.csv` (new)
- [ ] `episode_summary.csv` (new - if episodes completed)
- [ ] `checkpoint_benchmarks.csv` (new - if checkpoint saved)
- [ ] `rl_model_*_steps.zip` (checkpoint files)
- [ ] `rl_model_*_steps_transformer_encoder.pth` (transformer weights)

### Verify CSV Contents

#### monitor.csv
```bash
head -5 checkpoints/test_50k_t4/monitor.csv
```
- [ ] Has headers: `r,l,t`
- [ ] Has data rows with episode rewards, lengths, times

#### reward_breakdown.csv
```bash
head -5 checkpoints/test_50k_t4/reward_breakdown.csv
```
- [ ] Has headers with reward term names
- [ ] Has data rows with step numbers and term values
- [ ] Values are reasonable (not all zeros, not all NaN)

#### checkpoint_benchmarks.csv (if checkpoint reached)
```bash
cat checkpoints/test_50k_t4/checkpoint_benchmarks.csv
```
- [ ] Has headers: checkpoint_step, vs_based_winrate, vs_constant_winrate, etc.
- [ ] Has data rows (if checkpoint was saved)
- [ ] Win rates are percentages (0-100)
- [ ] Diversity score is a positive float

## Test 2: Frame-Level Alerts

### Trigger Gradient Explosion (Optional Advanced Test)

**Warning:** This will likely break training, only for testing alert system

Modify TRAIN_CONFIG temporarily:
```python
"learning_rate": 1.0,  # Very high (normally 2.5e-4)
```

Run training:
```bash
python user/train_agent.py
```

Expected:
- [ ] "⚠️  ALERT: Gradient explosion detected" within first 1000 steps
- [ ] Loss value shown in alert message
- [ ] Training may continue or crash (expected)

**Reset learning rate after test!**

### Trigger Reward Spike (Hard to Test Manually)

This typically happens naturally if reward function has bugs. Skip unless you want to artificially inject spikes.

## Test 3: Configuration Changes

### Test Custom Frequencies

Modify `TRAIN_CONFIG["training"]`:
```python
"light_log_freq": 1000,  # Changed from 500
"eval_freq": 10000,       # Changed from 5000
"eval_episodes": 1,       # Changed from 3
```

Run training:
```bash
python user/train_agent.py
```

Expected:
- [ ] Light logging appears every 1000 steps (not 500)
- [ ] Quick evaluation appears every 10000 steps (not 5000)
- [ ] Evaluation runs only 1 match (not 3)

### Test Disabled Monitoring

Modify `TRAIN_CONFIG["training"]`:
```python
"enable_debug": False,
```

Run training:
```bash
python user/train_agent.py
```

Expected:
- [ ] "MINIMAL LOGGING" in startup message (not "MONITORED TRAINING")
- [ ] No monitoring initialization message
- [ ] No "--- Step N ---" light logging during training
- [ ] No "🔍 QUICK EVALUATION" during training
- [ ] Only SB3's default logging (if verbose=1)
- [ ] No monitoring summary at end

**Reset `enable_debug=True` after test!**

## Test 4: Error Handling

### Test Missing Transformer (for non-transformer agents)

This is automatically handled. If using a non-transformer agent:
- [ ] Transformer health monitoring gracefully skips
- [ ] No errors or crashes
- [ ] Other monitoring continues normally

### Test CSV Write Failure (Advanced)

Make checkpoint directory read-only:
```bash
chmod 444 checkpoints/test_50k_t4/
```

Run training (will create new run):
```bash
python user/train_agent.py
```

Expected:
- [ ] Training continues despite CSV write failures
- [ ] Warning messages printed (optional)
- [ ] No crashes

**Reset permissions after test:**
```bash
chmod 755 checkpoints/test_50k_t4/
```

## Test 5: Memory and Performance

### Monitor Memory Usage

In a separate terminal:
```bash
# macOS
top -pid $(pgrep -f train_agent.py)

# Linux
top -p $(pgrep -f train_agent.py)
```

Expected:
- [ ] Memory usage stable (not growing unbounded)
- [ ] Memory increase < 100MB total due to monitoring
- [ ] No memory leaks visible over time

### Time First 1000 Steps

```bash
time python user/train_agent.py
# Let it run to 1000 steps, then Ctrl+C
```

Calculate overhead:
- [ ] Compare with monitoring disabled
- [ ] Overhead should be < 10% for first 1000 steps
- [ ] Most overhead is from initialization, not per-step cost

## Test 6: Integration with Transformer Agent

### Verify Transformer Health Tracking

Run training and monitor console output at step 500+:

Expected:
- [ ] Latent Norm appears and is non-zero
- [ ] Latent Norm is reasonable (5-50 range typically)
- [ ] Attention Entropy appears and is non-zero
- [ ] Values change over time (not stuck)

### Verify Strategy Diversity Tracking

Check checkpoint benchmark CSV after first checkpoint:

```bash
cat checkpoints/test_50k_t4/checkpoint_benchmarks.csv
```

Expected:
- [ ] strategy_diversity_score column present
- [ ] Value is non-zero
- [ ] Value is reasonable (0.1-10 range typically)

## Test 7: Long Run Validation (Optional)

### Run Full Test Configuration (50k steps)

```bash
python user/train_agent.py 2>&1 | tee training_full.log
```

Expected:
- [ ] Completes without crashes
- [ ] All monitoring milestones hit (500, 1000, ..., 50000)
- [ ] All CSV files populated
- [ ] Final summary shows total steps = ~50000
- [ ] No memory leaks (check `top` periodically)
- [ ] Training time reasonable (~15-20 minutes on good hardware)

Verify log file:
```bash
grep "Step" training_full.log | wc -l
# Should be ~100 (light logging every 500 steps)

grep "QUICK EVALUATION" training_full.log | wc -l
# Should be ~10 (evaluation every 5000 steps)

grep "CHECKPOINT BENCHMARK" training_full.log | wc -l
# Should be 0-1 (depends on checkpoint timing)
```

## Test 8: Callback Integration

### Verify Callback is Passed Correctly

Add debug print to verify (temporary):

In `user/train_agent.py`, line ~2318, add:
```python
print(f"DEBUG: Callback type = {type(monitor_callback)}")
if monitor_callback is not None:
    agent.learn(env, total_timesteps=train_timesteps, verbose=1, callback=monitor_callback)
```

Run training:
```bash
python user/train_agent.py | grep DEBUG
```

Expected:
- [ ] "DEBUG: Callback type = <class 'user.train_agent.TrainingMonitorCallback'>"

Remove debug print after verification.

## Test 9: CSV Data Integrity

### Check reward_breakdown.csv Structure

```python
import pandas as pd
import numpy as np

df = pd.read_csv('checkpoints/test_50k_t4/reward_breakdown.csv')

# Verify structure
print("Columns:", df.columns.tolist())
print("Shape:", df.shape)
print("Steps:", df['step'].min(), "to", df['step'].max())
print("NaN count:", df.isna().sum().sum())
print("\nSample rows:")
print(df.head())
```

Expected:
- [ ] Columns include all reward terms + 'step' + 'total_reward'
- [ ] No NaN values (or very few)
- [ ] Step values increase monotonically
- [ ] Reward values are reasonable (not all zero, not astronomically high)

### Check checkpoint_benchmarks.csv Structure

```python
import pandas as pd

df = pd.read_csv('checkpoints/test_50k_t4/checkpoint_benchmarks.csv')

print("Columns:", df.columns.tolist())
print("Shape:", df.shape)
print("\nData:")
print(df)
```

Expected:
- [ ] Columns: checkpoint_step, vs_based_winrate, vs_constant_winrate, avg_damage_ratio, strategy_diversity_score, eval_time_sec
- [ ] Win rates between 0 and 100
- [ ] Damage ratios > 0
- [ ] Diversity scores > 0
- [ ] Eval times reasonable (30-300 seconds)

## Test 10: Sanity Checks

### Verify Sanity Check Detection

Run a deliberately bad configuration:

```python
# In gen_reward_manager(), set all weights to 0
reward_functions = {
    'danger_zone_reward': RewTerm(func=danger_zone_reward, weight=0.0),
    'damage_interaction_reward': RewTerm(func=damage_interaction_reward, weight=0.0),
    # ...
}
```

Run training:
```bash
python user/train_agent.py
```

Expected:
- [ ] At first quick evaluation (5000 steps):
  - [ ] "⚠️  Sanity Check Issues:" message
  - [ ] "Reward appears STUCK" or "NO IMPROVEMENT detected"
  - [ ] Agent likely has 0% win rate

**Reset reward weights after test!**

## Test Completion Checklist

### Core Functionality
- [ ] Basic training works with monitoring enabled
- [ ] Light logging appears every N steps
- [ ] Quick evaluations run and display results
- [ ] Checkpoint benchmarks run (if checkpoint reached)
- [ ] All CSV files created and populated
- [ ] Final summary displays correctly

### Configuration
- [ ] Custom frequencies work
- [ ] Disabled monitoring works (falls back to minimal)
- [ ] All TRAIN_CONFIG options respected

### Performance
- [ ] Memory usage reasonable (< 100MB overhead)
- [ ] Time overhead acceptable (< 10% for 50k run)
- [ ] No memory leaks over long run
- [ ] CSV batching reduces I/O overhead

### Error Handling
- [ ] Graceful handling of missing components
- [ ] No crashes from monitoring failures
- [ ] Alerts display correctly when issues detected
- [ ] Sanity checks detect common problems

### Integration
- [ ] Callback passes through correctly
- [ ] Transformer health tracking works (when applicable)
- [ ] PPO metrics extracted from SB3 logger
- [ ] Reward breakdown computation accurate

### Data Quality
- [ ] CSV files well-formed
- [ ] No NaN values (or handled gracefully)
- [ ] Data values reasonable and interpretable
- [ ] Time-series data properly ordered

## Known Issues / Expected Behavior

### "Transformer health shows 0 initially"
**Expected:** Transformer needs 10+ frames to produce latent vectors. Health metrics appear after ~10 frames.

### "Checkpoint benchmark doesn't appear in 50k run"
**Expected:** If checkpoint frequency > 50k (e.g., 100k), no checkpoint benchmark will run. This is normal.

### "Lots of console output"
**Expected:** Real-time feedback is verbose. Use `tee` to save to file or increase logging frequencies.

### "Quick evaluation pauses training"
**Expected:** Evaluations block training temporarily. This is by design for accurate measurements.

### "CSV files update slowly"
**Expected:** Batched writes mean files update every ~5000 steps, not every step. Wait or check after training completes.

## Sign-Off

After completing all applicable tests:

- [ ] All critical tests passed
- [ ] No crashes or errors during normal operation
- [ ] Performance overhead acceptable
- [ ] Data quality verified
- [ ] Ready for production use

**Tested by:** _____________  
**Date:** _____________  
**Notes:** _____________________________________________

