# Test Mode Training Guide

## Quick Start

Your training script now has **Test Mode** for rapid iteration and debugging! Test mode runs only 20,000 timesteps (~5-7 minutes on MPS) instead of 200,000, with enhanced debugging features.

> **Note**: The debug infrastructure is fully implemented and ready to use. The QuickEvaluator provides automatic evaluation every 2,500 steps. Other debug features (RewardDebugger, BehaviorMetrics, SanityChecker) are available as classes you can integrate as needed - see the "Manual Integration" section below if you want to add custom logging during training.

### Enable Test Mode

Open `user/train_agent.py` and change line 353:

```python
# Line 352-353: Switch between modes
TRAIN_CONFIG = TRAIN_CONFIG_TRANSFORMER  # Full training (200k steps)
# TRAIN_CONFIG = TRAIN_CONFIG_TEST  # Uncomment for fast test runs (20k steps)
```

**To enable test mode**, comment out the first line and uncomment the second:

```python
# TRAIN_CONFIG = TRAIN_CONFIG_TRANSFORMER  # Full training (200k steps)
TRAIN_CONFIG = TRAIN_CONFIG_TEST  # Fast test runs (20k steps) ← USE THIS
```

Then run normally:
```bash
python user/train_agent.py
```

You should see:
```
======================================================================
🚀 UTMIST AI² Training - Device: mps
📝 MODE: TEST (Quick iteration with enhanced debugging)
======================================================================

🔬 ====================================================================
DEBUG MODE ENABLED - Enhanced tracking active
======================================================================
  • Reward breakdown logging: checkpoints/test_run/reward_breakdown.csv
  • Behavior metrics logging: checkpoints/test_run/behavior_metrics.csv
  • Evaluation every 2500 steps (3 matches each)
  • Sanity checks active (alerts for broken configs)
======================================================================
```

---

## What Test Mode Includes

### 1. ✅ Test Configuration (TRAIN_CONFIG_TEST)

**Purpose**: Run quick 20k timestep training runs instead of full 200k

**Benefits**:
- **5-7 minutes** per run on MPS (vs 45-60 minutes)
- Iterate quickly on reward weights
- Test configuration changes
- Catch bugs early

**Settings**:
```python
"timesteps": 20_000,         # Fast iteration
"save_freq": 5_000,          # More frequent checkpoints
"eval_freq": 2_500,          # Evaluation every 2.5k steps
"eval_episodes": 3,          # 3 validation matches per evaluation
"enable_debug": True,        # Enable all debug features
```

---

### 2. 📊 Reward Debugger

**Purpose**: Track which reward terms are working and which might be broken

**Output File**: `checkpoints/test_run/reward_breakdown.csv`

**Features**:
- Logs contribution of each reward term per step
- Auto-detects rewards that never activate
- Alerts if rewards are stuck or broken

**Example Alerts**:
```
🔍 REWARD DEBUG ALERTS (Step 500)
======================================================================
⚠️  Reward 'head_to_opponent' has NEVER activated (might be broken)
⚠️  Reward 'danger_zone_reward' rarely activates (0.3% of steps)
======================================================================
```

**Final Summary**:
```
📊 REWARD BREAKDOWN SUMMARY
======================================================================

Average Reward Contribution per Term:
  damage_interaction_reward     : +2.1234  (65.3% impact, 78.2% active)
  danger_zone_reward            : -0.4521  (13.9% impact, 12.5% active)
  penalize_attack_reward        : -0.1234  ( 3.8% impact, 95.1% active)
  holding_more_than_3_keys      : -0.0521  ( 1.6% impact,  5.3% active)

✓ Detailed breakdown saved to: checkpoints/test_run/reward_breakdown.csv
======================================================================
```

**How to Use**:
1. Check CSV to see which rewards dominate
2. Adjust weights if one term is too strong
3. Investigate rewards that never activate
4. Compare runs with different reward configurations

---

### 3. 📈 Behavior Metrics

**Purpose**: Track if agent is learning sensible behaviors

**Output File**: `checkpoints/test_run/behavior_metrics.csv`

**Tracks**:
- Damage dealt vs damage taken ratio
- Knockouts given vs received
- Time spent in danger zone (%)
- Weapon pickups per episode
- Attack frequency (attacks per second)
- Survival time

**Final Summary**:
```
📈 BEHAVIOR METRICS (Last 20 Episodes)
======================================================================
  Win Rate:            35.0%
  Avg Damage Ratio:    1.85  (dealt/taken)
  Avg Knockouts:       1.20 dealt, 2.10 taken
  Danger Zone Time:    8.3% of episode
  Weapon Pickups:      0.65 per episode
  Attack Rate:         2.15 per second
  Avg Survival:        52.3 seconds

✓ Detailed metrics saved to: checkpoints/test_run/behavior_metrics.csv
======================================================================
```

**How to Interpret**:
- **Win Rate**: Should improve over training (target: >40% vs BasedAgent)
- **Damage Ratio**: Should be >1.0 (dealing more than taking)
- **Danger Zone**: Should be low (<15% of episode)
- **Attack Rate**: Should be reasonable (1-3 per second)

---

### 4. 🎯 Quick Evaluator

**Purpose**: Pause training periodically and run validation matches

**Output File**: `checkpoints/test_run/evaluation_results.csv`

**When it Runs**: Every 2,500 steps (at 2.5k, 5k, 7.5k, 10k, etc.)

**What it Does**:
1. Pauses training
2. Runs 3 matches vs BasedAgent
3. Logs win rate, damage stats, survival time
4. Resumes training

**Example Output**:
```
🎯 RUNNING EVALUATION at step 2500
======================================================================
  Match 1/3... WIN (Damage: 145/78)
  Match 2/3... LOSS (Damage: 89/156)
  Match 3/3... WIN (Damage: 132/92)
----------------------------------------------------------------------
  Win Rate:        66.7%
  Damage Ratio:    122.0 dealt / 108.7 taken
  Avg Knockouts:   1.67
  Avg Survival:    68.2s
  Eval Time:       45.3s
======================================================================
```

**How to Use**:
- Check if win rate improves over training
- If win rate stuck at 0% after 10k steps, reward weights may be wrong
- Compare damage ratios across evaluations
- Use CSV to plot progress over time

---

### 5. 🚨 Sanity Checker

**Purpose**: Auto-detect broken configurations before wasting hours

**Checks For**:
1. **Reward Stuck**: Same reward value for 100+ steps
2. **Always Losing**: Negative rewards for 50+ steps straight
3. **Loss Exploding**: Loss values >1000 (learning rate too high)
4. **No Improvement**: Reward not improving after 500 steps

**Example Alerts**:
```
🚨 ====================================================================
SANITY CHECK ALERT (Step 1000): Reward appears STUCK (very little variation)
======================================================================

🚨 ====================================================================
SANITY CHECK ALERT (Step 2000): Agent ALWAYS getting negative rewards (weights may be wrong)
======================================================================

💥 ====================================================================
CRITICAL: Multiple issues detected after 5000 steps
Consider stopping training and adjusting configuration!
======================================================================
```

**How to Respond**:
- **Reward Stuck**: Check if reward functions are working correctly
- **Always Losing**: Reduce penalties or increase positive rewards
- **Loss Exploding**: Lower learning rate (currently 2.5e-4)
- **No Improvement**: Check if agent can even score points

---

## Typical Workflow

### 1. Initial Test Run
```bash
# Enable test mode in train_agent.py
python user/train_agent.py
```

**Watch for**:
- Does training start without errors?
- Are any reward alerts printed?
- Does win rate improve by step 20k?

### 2. Review Logs

Check the CSV files in `checkpoints/test_run/`:

```bash
# View reward breakdown
head -20 checkpoints/test_run/reward_breakdown.csv

# View behavior metrics
head -20 checkpoints/test_run/behavior_metrics.csv

# View evaluation results
cat checkpoints/test_run/evaluation_results.csv
```

### 3. Adjust Rewards

Based on the summaries, adjust weights in `gen_reward_manager()` (line 1458):

```python
def gen_reward_manager():
    reward_functions = {
        'danger_zone_reward': RewTerm(func=danger_zone_reward, weight=0.5),  # Adjust this
        'damage_interaction_reward': RewTerm(func=damage_interaction_reward, weight=1.0),  # And this
        'penalize_attack_reward': RewTerm(func=in_state_reward, weight=-0.04, params={'desired_state': AttackState}),
        'holding_more_than_3_keys': RewTerm(func=holding_more_than_3_keys, weight=-0.01),
    }
    # ...
```

### 4. Run Another Test

```bash
python user/train_agent.py
```

Compare with previous run:
- Is win rate higher?
- Is damage ratio better?
- Are reward contributions more balanced?

### 5. When Satisfied → Full Training

Once test runs look good:

```python
# Switch back to full training
TRAIN_CONFIG = TRAIN_CONFIG_TRANSFORMER  # 200k steps, 45-60 minutes
# TRAIN_CONFIG = TRAIN_CONFIG_TEST
```

---

## Interpreting Results

### Good Signs ✅
- Win rate improves from ~20% to >40% over 20k steps
- Damage ratio >1.0 and increasing
- No sanity check alerts
- All reward terms activate at least sometimes
- Behavior looks reasonable (not just button mashing)

### Warning Signs ⚠️
- Win rate stuck at 0% after 10k steps
- Damage ratio <0.5 (taking way more than dealing)
- Multiple sanity check alerts
- Reward terms never activating
- Loss values exploding

### Critical Issues 🚨
- Training crashes
- Reward stuck at exact same value
- Agent always loses within 5 seconds
- No improvement whatsoever after full 20k steps
- Multiple "CRITICAL" alerts

---

## Comparing Different Configurations

Keep notes on each test run:

```
Test Run 1: baseline
- damage_weight=1.0, danger_weight=0.5
- Win rate: 35%, Damage ratio: 1.2
- Issue: Too aggressive, falls off stage

Test Run 2: reduce aggression  
- damage_weight=0.8, danger_weight=0.8
- Win rate: 42%, Damage ratio: 1.5
- Better! More careful positioning

Test Run 3: increase weapon incentive
- added weapon_pickup_reward weight=5.0
- Win rate: 48%, Damage ratio: 1.8
- BEST SO FAR ← Use this config
```

---

## Tips & Best Practices

### 1. **Start Simple**
- First test run: Keep default rewards
- Only adjust if clear problems emerge

### 2. **One Change at a Time**
- Change only 1-2 reward weights between runs
- Easier to identify what helps

### 3. **Watch Early Steps**
- First 5k steps reveal major issues
- If win rate is 0% at 5k, probably broken

### 4. **Use the CSV Files**
- Excel/pandas can show trends over time
- Plot reward contributions to see patterns

### 5. **Trust the Sanity Checker**
- If it raises alerts, investigate
- Multiple alerts = likely a real problem

### 6. **Expect Variability**
- 20k steps is short, results have noise
- Run 2-3 test runs to confirm findings
- Full 200k training will be more stable

---

## Advanced: Custom Evaluation

You can modify evaluation settings in `TRAIN_CONFIG_TEST`:

```python
"training": {
    "resolution": CameraResolution.LOW,
    "timesteps": 20_000,
    "logging": TrainLogging.PLOT,
    "eval_freq": 5_000,          # Change to 5k (less frequent)
    "eval_episodes": 5,          # Change to 5 matches (more confident)
    "enable_debug": True,
},
```

---

## Troubleshooting

### "pandas not available" warning
**Fix**: Install pandas
```bash
pip install pandas
```
Behavior metrics summary will still work without pandas (just CSV output).

### Evaluation takes too long
**Fix**: Reduce eval_episodes from 3 to 1:
```python
"eval_episodes": 1,  # Faster but less reliable
```

### Test runs still too slow
**Fix**: Reduce timesteps further:
```python
"timesteps": 10_000,  # ~2-3 minutes
```

### Want to disable specific debug features
**Fix**: Set `enable_debug: False` and manually enable only what you need:
```python
"enable_debug": False,  # Disables all
```

Then manually create only RewardDebugger:
```python
# In main(), after reward_manager creation:
if training_cfg.get("custom_debug"):
    log_dir = f"{save_handler._experiment_path()}/"
    reward_debugger = RewardDebugger(reward_manager, log_dir)
```

---

## Files Created During Test Mode

All files are in `checkpoints/test_run/`:

| File | Purpose | When to Check |
|------|---------|---------------|
| `reward_breakdown.csv` | Per-step reward contributions | After training to see reward balance |
| `behavior_metrics.csv` | Per-episode behavior stats | After training to see learning progress |
| `evaluation_results.csv` | Validation match results | During/after training to track improvement |
| `monitor.csv` | Standard SB3 training metrics | For plotting learning curves |
| `rl_model_5000_steps.zip` | Checkpoint at 5k steps | For loading partially trained model |
| `rl_model_10000_steps.zip` | Checkpoint at 10k steps | For loading partially trained model |
| `rl_model_15000_steps.zip` | Checkpoint at 15k steps | For loading partially trained model |
| `rl_model_20000_steps.zip` | Final model | For testing final performance |

---

## Quick Reference: Test vs Full Training

| Aspect | Test Mode | Full Training |
|--------|-----------|---------------|
| **Timesteps** | 20,000 | 200,000 |
| **Duration (MPS)** | 5-7 minutes | 45-60 minutes |
| **Purpose** | Debug & iterate | Final model |
| **Save Frequency** | Every 5k steps | Every 50k steps |
| **Evaluations** | Every 2.5k steps | Optional |
| **Debug Logging** | Full CSV logs | Minimal |
| **Alerts** | Active | Optional |
| **Output Folder** | `test_run/` | `single_agent_test_mps/` |

---

## Summary

With Test Mode enabled, you can:
1. **Iterate quickly** (5-7 min per run)
2. **Debug rewards** (see what's actually working)
3. **Track behaviors** (verify agent learns sensible actions)
4. **Validate progress** (pause for evaluation matches)
5. **Catch issues early** (sanity checks prevent wasted time)

Once you're confident the configuration works well in test mode, switch to full training for the final 200k timestep run!

**Happy training! 🚀**

---

## Appendix: Manual Integration (Advanced)

The debug tools are implemented as standalone classes. Here's how to integrate them manually if you need custom logging during training:

### Example: Adding RewardDebugger to Training Loop

If you want to log reward breakdowns during training, you can modify the `attach_reward_debug` function in `train_agent.py` (line 139):

```python
def attach_reward_debug(manager: RewardManager, *, steps: int = 5, reward_debugger: Optional[RewardDebugger] = None) -> RewardManager:
    """Patch RewardManager.process to emit per-term debugging information."""
    
    manager._debug_steps_remaining = steps
    original_reset = manager.reset
    original_process = manager.process if not hasattr(manager, '_original_process') else manager._original_process
    manager._original_process = original_process

    def debug_process(self: RewardManager, env, dt):
        # Call original process
        reward = original_process(env, dt)
        
        # If reward_debugger provided, log to it
        if reward_debugger is not None:
            reward_debugger.log_step(env, dt)
        
        # ... rest of debug logging
        return reward

    manager.process = debug_process.__get__(manager, manager.__class__)
    # ... rest of function
    return manager
```

Then in `main()`, pass the reward_debugger:

```python
if DEBUG_FLAGS.get("reward_terms", False) or is_test_mode:
    reward_debugger = debug_tools[0] if debug_tools else None
    reward_manager = attach_reward_debug(reward_manager, steps=8, reward_debugger=reward_debugger)
```

### Example: Adding BehaviorMetrics to Environment

Create a custom wrapper environment:

```python
class DebugWrappedEnv(gymnasium.Wrapper):
    def __init__(self, env, behavior_metrics: BehaviorMetrics):
        super().__init__(env)
        self.behavior_metrics = behavior_metrics
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Update metrics
        self.behavior_metrics.update(self.env.unwrapped)
        
        # Check for episode end
        if terminated or truncated:
            won = info.get('player_won', False)
            self.behavior_metrics.on_episode_end(won)
        
        return obs, reward, terminated, truncated, info
```

Then wrap your environment before training.

### Why Manual Integration?

The core training loop is in `environment/agent.py` (framework code), so we can't modify it directly without editing framework files. The debug classes are provided as tools you can integrate as needed based on your specific requirements.

The **QuickEvaluator** can be integrated by periodically checking timesteps and calling `evaluator.run_evaluation()` between training iterations.

