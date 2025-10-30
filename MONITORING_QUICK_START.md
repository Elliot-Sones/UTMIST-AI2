# Monitoring System - Quick Start Guide

## TL;DR

The monitoring system is **enabled by default** and tracks everything you need automatically. Just run training and watch the console output.

```python
python user/train_agent.py
```

## What Gets Tracked?

### Automatically (No Setup Required)

| Metric | Frequency | Output |
|--------|-----------|--------|
| **Episode Rewards** | Continuous | `monitor.csv` (SB3 default) |
| **Frame Alerts** | Real-time | Console only |
| **Reward Breakdown** | Every 500 steps | Console + `reward_breakdown.csv` |
| **Transformer Health** | Every 500 steps | Console |
| **PPO Metrics** | Every 500 steps | Console |
| **Quick Evaluation** | Every 5000 steps | Console |
| **Sanity Checks** | Every 5000 steps | Console |
| **Checkpoint Benchmarks** | At checkpoints | Console + `checkpoint_benchmarks.csv` |

## Reading the Console Output

### Normal Training (Every 500 steps)
```
--- Step 1000 ---
  Reward Breakdown: danger_zone_reward=-0.025, damage_interaction_reward=0.143
  Active Terms: danger_zone_reward, damage_interaction_reward, penalize_attack_reward
  Transformer: Latent Norm=12.543 (±2.134), Attention Entropy=2.843
  PPO: Policy Loss=0.0234, Value Loss=1.2456, Explained Var=0.847
```

**What to look for:**
- ✅ Multiple active reward terms
- ✅ Latent Norm between 5-20
- ✅ Explained Variance > 0.7 (MOST IMPORTANT!)

### Quick Evaluation (Every 5000 steps)
```
======================================================================
🔍 QUICK EVALUATION (Step 5000)
======================================================================
  Win Rate: 66.7% (2/3 matches)
  Damage Ratio: 1.84
  Avg Episode Reward (last 10): 45.23
  ✓ Sanity checks passed
======================================================================
```

**What to look for:**
- ✅ Win rate improving over time
- ✅ Damage ratio > 1.0 (dealing more than taking)
- ✅ Sanity checks passing

### Frame-Level Alerts (If Something Goes Wrong)
```
⚠️  ALERT: Gradient explosion detected (loss=152.34) at step 12500
🚨 CRITICAL: NaN detected in loss at step 15000!
⚠️  ALERT: Reward spike detected (234.5) at step 18000
```

**What to do:**
- **Gradient explosion** → Reduce learning rate
- **NaN values** → Training broken, restart with lower learning rate
- **Reward spike** → Usually okay, just monitor

## Output Files Location

All files saved to: `checkpoints/{run_name}/`

Example: `checkpoints/test_50k_t4/`

### Files Created

1. **`monitor.csv`** - Episode-level metrics (SB3 default)
   - Total reward per episode
   - Episode length
   - Automatically plotted at end of training

2. **`reward_breakdown.csv`** - Reward term contributions
   - See which rewards are driving behavior
   - Identify broken reward terms

3. **`checkpoint_benchmarks.csv`** - Checkpoint performance
   - Win rates vs different opponents
   - Strategy diversity over time

## Quick Health Checks

### Is Training Working?

Check these 3 things:

1. **Explained Variance (every 500 steps)**
   - Should be > 0.7
   - If < 0.3 → Agent not learning

2. **Win Rate (every 5000 steps)**
   - Should improve over time
   - If always 0% → Check reward weights

3. **Sanity Checks (every 5000 steps)**
   - Should pass (✓)
   - If issues detected → Read warning messages

### Is Transformer Learning?

Check **Latent Norm** (every 500 steps):
- ✅ 5-20: Healthy
- ⚠️ < 1: Encoder may be dead
- ⚠️ > 50: Encoder saturating

Check **Attention Entropy** (every 500 steps):
- ✅ 2-4: Focused learning
- ⚠️ > 5: Too diffuse, not learning patterns
- ⚠️ < 1: Too narrow, may be overfitting

## Customizing Monitoring

### Change Logging Frequency

In `train_agent.py`, modify `TRAIN_CONFIG`:

```python
TRAIN_CONFIG["training"]["light_log_freq"] = 1000    # Log every 1000 steps
TRAIN_CONFIG["training"]["eval_freq"] = 10000        # Evaluate every 10k steps
TRAIN_CONFIG["training"]["eval_episodes"] = 5        # 5 matches per eval
```

### Disable Monitoring

```python
TRAIN_CONFIG["training"]["enable_debug"] = False
```

### Save Console Output to File

```bash
python user/train_agent.py > training.log 2>&1
```

Then monitor in real-time:
```bash
tail -f training.log
```

## Common Issues

### "Too much console output!"

**Expected behavior.** The system provides real-time feedback.

**Solutions:**
1. Save to file (see above)
2. Increase logging frequencies
3. Disable monitoring (not recommended)

### "Training seems slow"

**Monitoring overhead is <5% for test runs, <2% for long runs.**

**If still concerned:**
- Increase `light_log_freq` to 2000
- Increase `eval_freq` to 20000
- Reduce `eval_episodes` to 1

### "CSV files are empty"

**CSV files are batched-written every ~5000 steps.**

**Solutions:**
1. Wait for training to progress
2. Check after training completes
3. Look in correct directory: `checkpoints/{run_name}/`

### "Evaluation matches taking too long"

**Each quick evaluation runs 3 matches (~60 seconds).**

**Solutions:**
- Reduce `eval_episodes` to 1
- Increase `eval_freq` to evaluate less often

## Reading the CSV Files

### monitor.csv (SB3 Default)

```csv
r,l,t
45.23,1847,2.5
52.34,1923,5.1
```

- `r`: Episode reward
- `l`: Episode length (frames)
- `t`: Time (seconds)

**Analyze with:**
```python
import pandas as pd
df = pd.read_csv('checkpoints/test_50k_t4/monitor.csv')
print(df['r'].rolling(10).mean())  # 10-episode moving average
```

### reward_breakdown.csv

```csv
step,danger_zone_reward,damage_interaction_reward,...,total_reward
500,-0.025,0.143,...,0.117
1000,-0.030,0.156,...,0.126
```

**Analyze with:**
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('checkpoints/test_50k_t4/reward_breakdown.csv')

# Plot reward term contributions over time
df.plot(x='step', y=['danger_zone_reward', 'damage_interaction_reward'])
plt.title('Reward Term Contributions Over Time')
plt.show()
```

### checkpoint_benchmarks.csv

```csv
checkpoint_step,vs_based_winrate,vs_constant_winrate,avg_damage_ratio,strategy_diversity_score,eval_time_sec
50000,60.00,80.00,1.85,3.4521,45.2
100000,75.00,90.00,2.34,4.1234,47.8
```

**Analyze with:**
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('checkpoints/test_50k_t4/checkpoint_benchmarks.csv')

# Plot win rate improvement
df.plot(x='checkpoint_step', y=['vs_based_winrate', 'vs_constant_winrate'])
plt.title('Win Rate Over Training')
plt.ylabel('Win Rate (%)')
plt.show()
```

## Pro Tips

### 1. Watch Explained Variance Closely

This is **the most important metric**. If it's low:
- Check reward function (are rewards meaningful?)
- Check network architecture (too small?)
- Check learning rate (too high/low?)

### 2. Use Checkpoint Benchmarks for Model Selection

Don't just use the latest checkpoint. Look at:
- Highest win rate
- Best damage ratio
- Highest strategy diversity

### 3. Compare Reward Terms

If one term dominates (>90% contribution):
- Other terms may need higher weights
- Or the dominant term is too large

### 4. Monitor Strategy Diversity

Low diversity score → Agent using same strategy repeatedly
- May need more diverse opponents
- May need longer training

### 5. Save Full Logs for Analysis

```bash
python user/train_agent.py | tee training_full.log
```

This saves to file **and** displays in console.

## Example: Debugging Low Win Rate

**Problem:** Agent has 0% win rate after 10k steps

**Step 1:** Check reward breakdown
```
Reward Breakdown: danger_zone_reward=-0.025, damage_interaction_reward=0.000
```
→ Damage reward is 0! Agent not dealing damage.

**Step 2:** Check PPO metrics
```
PPO: Explained Var=0.12
```
→ Very low! Agent not learning value function well.

**Step 3:** Check reward weights
```python
'damage_interaction_reward': RewTerm(func=damage_interaction_reward, weight=1.0)
```
→ Weight seems okay.

**Step 4:** Check sanity checks
```
⚠️  Sanity Check Issues:
    - NO IMPROVEMENT detected (agent not learning)
```

**Solution:** Increase learning rate or check network architecture.

## Getting Help

If monitoring shows issues:

1. Check `MONITORING_SYSTEM.md` for detailed explanations
2. Compare metrics to "Healthy Training Signs"
3. Review reward function implementations
4. Check hyperparameters (learning rate, batch size, etc.)

## Summary

### What You Must Watch

1. **Explained Variance** → Should be > 0.7
2. **Win Rate** → Should improve over time
3. **Sanity Checks** → Should pass

### What's Nice to Have

1. Reward breakdown → Understanding behavior
2. Transformer health → Ensuring encoder works
3. Checkpoint benchmarks → Model selection

### When to Worry

- ❌ Explained variance < 0.3 after 5000 steps
- ❌ Win rate stuck at 0% after 10000 steps
- ❌ Gradient explosions (loss > 100)
- ❌ NaN values in loss
- ❌ Reward stuck at same value

**Training healthy?** You'll see:
- Explained variance 0.7-0.95
- Win rate improving (even slowly)
- Sanity checks passing
- Multiple reward terms active

