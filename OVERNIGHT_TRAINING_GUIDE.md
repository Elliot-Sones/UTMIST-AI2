# Overnight Training Guide

This guide will help you successfully run and monitor your RL agent training overnight for 5-10 million steps.

## Pre-Launch Checklist

### 1. Run Pre-Flight Check
```bash
python user/pre_flight_check.py
```

**What it checks:**
- ✓ GPU availability and performance
- ✓ Disk space (need 2GB+ free)
- ✓ Python dependencies installed
- ✓ Reward functions working
- ✓ Training loop functional
- ✓ Estimates training time

**DO NOT START** overnight training until all checks pass!

### 2. (Optional) Evaluate Current Baseline
If you have an existing model at 4000 steps:
```bash
python user/evaluate_baseline.py --model checkpoints/simplified_training/latest_model.zip
```

This gives you a baseline to compare against after overnight training.

### 3. Choose Training Script

**Option A: Strategy Encoder (RECOMMENDED for competition)**
- More advanced: opponent-adaptive agent
- Better for competition goal ("beat you and counter your strategy")
- File: `user/train_with_strategy_encoder.py`
- Checkpoint dir: `/tmp/strategy_encoder_training/`

**Option B: Simplified Training**
- Simpler architecture
- Already has 4000 steps trained
- File: `user/train_simplified.py`
- Checkpoint dir: `checkpoints/simplified_training/`

**Recommendation:** Use `train_with_strategy_encoder.py` for best competition performance.

---

## Launching Training

### 1. Start Persistent Session

Training will take 8-18 hours. You MUST use tmux or screen to prevent disconnection from stopping training.

**Using tmux (recommended):**
```bash
# Start new session
tmux new -s rl_training

# Inside tmux, start training (see step 2)

# Detach: Press Ctrl+B, then D
# Reattach later: tmux attach -t rl_training
```

**Using screen:**
```bash
# Start new session
screen -S rl_training

# Inside screen, start training (see step 2)

# Detach: Press Ctrl+A, then D
# Reattach later: screen -r rl_training
```

### 2. Start Training

```bash
# Inside your tmux/screen session:
python user/train_with_strategy_encoder.py 2>&1 | tee training_output.log
```

The `| tee` part saves output to a file while still showing it on screen.

### 3. Verify Training Started

Watch the console for ~30 seconds. You should see:

```
TRAINING STARTED
Timestep     Rollout  Reward       Win%     Dmg+/-          FPS        Time
--------------------------------------------------------------------------------
4,096        1        -12.34       0.0      -5.2(10.3/15.5) 512        00:00:08
```

**Check for:**
- ✓ FPS is 300-600 (GPU) or 50-100 (CPU)
- ✓ Reward values are changing (not stuck at 0)
- ✓ No NaN/Inf warnings
- ✓ No error messages

**If everything looks good, detach!**

---

## Monitoring Training

### Remote Monitoring (Check Progress Anywhere)

**1. Check if training is still running:**
```bash
# See if process exists
ps aux | grep train_with_strategy_encoder

# Or check tmux sessions
tmux ls
```

**2. View recent console output:**
```bash
# Last 50 lines
tail -n 50 training_output.log

# Follow live (Ctrl+C to exit)
tail -f training_output.log
```

**3. Check training progress:**
```bash
# Look for latest checkpoint
ls -lht /tmp/strategy_encoder_training/*.zip | head -5

# Count lines in CSV (each line = 1 rollout logged)
wc -l /tmp/strategy_encoder_training/training_metrics.csv
```

**4. Reattach to session:**
```bash
tmux attach -t rl_training
```

### TensorBoard (Live Monitoring Dashboard)

If you have TensorBoard installed, you can view live training graphs:

**1. Start TensorBoard (in a separate terminal):**
```bash
tensorboard --logdir /tmp/strategy_encoder_training/tb_logs
```

**2. Open in browser:**
```
http://localhost:6006
```

**3. Key metrics to watch:**
- `rollout/ep_rew_mean` - Should increase over time
- `train/entropy_loss` - Should slowly decrease (agent becoming confident)
- `train/approx_kl` - Should stay < 0.05 (stable training)
- `train/clip_fraction` - Should be 0.1-0.3 (healthy updates)

### CSV Metrics (For Plotting Later)

Training automatically exports metrics to:
```
/tmp/strategy_encoder_training/training_metrics.csv
```

**Columns:**
- timestep, rollout, avg_reward, win_rate
- damage_dealt, damage_taken, damage_diff
- avg_episode_length, fps, elapsed_time_sec
- policy_loss, value_loss, entropy, kl_divergence, clip_fraction

**Plot with Python:**
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/tmp/strategy_encoder_training/training_metrics.csv')

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(df['timestep'], df['avg_reward'])
plt.title('Average Reward')
plt.xlabel('Timesteps')

plt.subplot(2, 2, 2)
plt.plot(df['timestep'], df['win_rate'] * 100)
plt.title('Win Rate (%)')
plt.xlabel('Timesteps')

plt.subplot(2, 2, 3)
plt.plot(df['timestep'], df['damage_diff'])
plt.title('Damage Differential')
plt.xlabel('Timesteps')

plt.subplot(2, 2, 4)
plt.plot(df['timestep'], df['entropy'])
plt.title('Entropy (Exploration)')
plt.xlabel('Timesteps')

plt.tight_layout()
plt.savefig('training_progress.png')
print("Saved plot to training_progress.png")
```

---

## What Good Training Looks Like

### Console Output Timeline (5M steps)

**0-100k steps (0-1 hour): Learning Basics**
```
Timestep     Rollout  Reward       Win%     Dmg+/-          FPS        Time
50,000       12       -8.2         5.0      -2.1(8.5/10.6)  520        00:01:35
```
- Reward: Negative but improving from initial -15
- Win%: 0-10% (learning to not immediately die)
- Dmg+/-: Slightly negative (getting hit more than hitting)

**100k-500k steps (1-4 hours): Getting Competitive**
```
Timestep     Rollout  Reward       Win%     Dmg+/-          FPS        Time
250,000      61       +5.3         25.0     +3.5(12.1/8.6)  515        00:08:05
```
- Reward: Crossing into positive territory
- Win%: 20-40% (winning against weak opponents)
- Dmg+/-: Positive (dealing more than taking)

**500k-2M steps (4-12 hours): Solid Performance**
```
Timestep     Rollout  Reward       Win%     Dmg+/-          FPS        Time
1,000,000    244      +18.7        55.0     +8.2(18.5/10.3) 512        00:32:15
```
- Reward: Strongly positive
- Win%: 50-70% (competitive against most opponents)
- Dmg+/-: Large positive gap

**2M-5M steps (12-24 hours): Mastery**
```
Timestep     Rollout  Reward       Win%     Dmg+/-          FPS        Time
5,000,000    1220     +35.2        75.0     +15.1(25.3/10.2) 510       16:12:45
```
- Reward: Very high
- Win%: 70-85% (beating most opponents consistently)
- Dmg+/-: Dominating damage trades

### Success Milestones

The training will automatically print celebration messages when hitting milestones:

```
================================================================================
                          🎉 MILESTONE ACHIEVED! 🎉
                  Win rate reached 25% at 180,000 steps!
================================================================================
```

**Milestones:**
- 25% win rate
- 50% win rate
- 75% win rate

### Warning Signs

**❌ Training is NOT working if you see:**

1. **Reward stuck near 0 or decreasing:**
   ```
   500,000      122      -0.5         0.0      -10.2(2.1/12.3)
   ```
   → Agent not learning. Check reward functions.

2. **Win rate stays at 0% after 200k steps:**
   ```
   200,000      48       -15.2        0.0      -20.5(1.2/21.7)
   ```
   → Agent can't beat even weak opponents. Major issue.

3. **NaN/Inf warnings:**
   ```
   ❌ NaN detected in gradients! Stopping training.
   ```
   → Training unstable. May need to adjust learning rate.

4. **Very low entropy (< 0.01):**
   ```
   Entropy:         0.003
   ```
   → Policy collapsed. Agent doing same thing every time.

5. **KL divergence > 0.1:**
   ```
   KL Divergence:   0.156
   ```
   → Training too unstable. Policy changing too fast.

---

## Checkpoints

### Automatic Saves

Training saves checkpoints every **1,000,000 steps**:

```
/tmp/strategy_encoder_training/
  ├── rl_model_1000000_steps.zip
  ├── rl_model_2000000_steps.zip
  ├── rl_model_3000000_steps.zip
  ├── rl_model_4000000_steps.zip
  ├── rl_model_5000000_steps.zip
  ├── latest_model.zip           ← Always points to most recent
  ├── final_model.zip             ← Saved when training completes
  ├── training_metrics.csv
  └── tb_logs/
```

### Manual Checkpoint (If Needed)

If you need to stop training early:

1. Reattach to session: `tmux attach -t rl_training`
2. Press `Ctrl+C` (training will save before exiting)
3. Model saved to `latest_model.zip`

---

## After Training Completes

### 1. Verify Completion

```bash
# Check final log output
tail -n 100 training_output.log
```

Should see:
```
================================================================================
                            TRAINING COMPLETED
================================================================================
Total timesteps: 5,000,000
Total rollouts:  1,220
Total time:      16h 12m
Avg FPS:         510

Final Performance (last 100 episodes):
  Avg Reward:  +35.2
  Win Rate:    75.0%
  Avg Damage:  +15.1
================================================================================
```

### 2. Plot Training Progress

```bash
python -c "
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/tmp/strategy_encoder_training/training_metrics.csv')

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(df['timestep'], df['avg_reward'])
axes[0, 0].set_title('Average Reward Over Time')
axes[0, 0].set_xlabel('Timesteps')
axes[0, 0].set_ylabel('Reward')
axes[0, 0].grid(True)

axes[0, 1].plot(df['timestep'], df['win_rate'] * 100)
axes[0, 1].set_title('Win Rate Over Time')
axes[0, 1].set_xlabel('Timesteps')
axes[0, 1].set_ylabel('Win Rate (%)')
axes[0, 1].grid(True)

axes[1, 0].plot(df['timestep'], df['damage_diff'])
axes[1, 0].set_title('Damage Differential')
axes[1, 0].set_xlabel('Timesteps')
axes[1, 0].set_ylabel('Damage Dealt - Taken')
axes[1, 0].grid(True)

axes[1, 1].plot(df['timestep'], df['entropy'])
axes[1, 1].set_title('Entropy (Exploration)')
axes[1, 1].set_xlabel('Timesteps')
axes[1, 1].set_ylabel('Entropy')
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('final_training_progress.png', dpi=150)
print('Saved to final_training_progress.png')
"
```

### 3. Evaluate Final Performance

```bash
python user/evaluate_baseline.py --model /tmp/strategy_encoder_training/final_model.zip --episodes 20
```

This tests your agent against all 8 opponent types (20 episodes each).

**Target Performance:**
- Overall win rate: **> 60%** (competitive)
- Strong against (> 60% win): At least 5-6 opponents
- Weak against (< 40% win): At most 1-2 opponents

### 4. Test Against Yourself

```bash
# Edit pvp_match.py to use your model
python user/pvp_match.py
```

Play against your agent to verify it can counter your strategies!

---

## Troubleshooting

### Training Crashes

**1. Check error in log:**
```bash
tail -n 100 training_output.log
```

**2. Common issues:**

**Out of Memory (OOM):**
```
CUDA out of memory
```
→ Reduce `n_envs` in training script from 8 to 4

**Disk Full:**
```
No space left on device
```
→ Free up space or change `CHECKPOINT_DIR`

**NaN/Inf gradients:**
```
NaN detected in gradients
```
→ Reduce learning rate or check reward functions

### Training Too Slow

**If FPS < 100:**
- Using CPU instead of GPU? Check device with pre-flight script
- Too many parallel environments? Reduce `n_envs`
- Other programs using GPU? Close them

**Expected FPS:**
- CUDA GPU: 300-600 FPS (5M steps in 8-12 hours)
- MPS (Apple): 100-200 FPS (5M steps in 12-18 hours)
- CPU: 20-50 FPS (5M steps in 24-48 hours - NOT recommended)

### Resume from Checkpoint

If training crashes, you can resume:

```python
# In train_with_strategy_encoder.py, it automatically resumes if latest_model.zip exists
# Just re-run the script:
python user/train_with_strategy_encoder.py
```

---

## Quick Reference Commands

```bash
# Pre-flight check
python user/pre_flight_check.py

# Start training in tmux
tmux new -s rl_training
python user/train_with_strategy_encoder.py 2>&1 | tee training_output.log
# Press Ctrl+B, then D to detach

# Monitor progress
tail -f training_output.log                    # Live output
tmux attach -t rl_training                     # Reattach
ls -lht /tmp/strategy_encoder_training/        # Check files
tensorboard --logdir /tmp/strategy_encoder_training/tb_logs  # Dashboard

# After training
python user/evaluate_baseline.py --model /tmp/strategy_encoder_training/final_model.zip
```

---

## Expected Timeline (5M steps on GPU)

| Time | Steps | Win Rate | Status |
|------|-------|----------|--------|
| 0:00 | 0 | 0% | Training starts |
| 0:30 | 100k | 5-10% | Learning basics |
| 2:00 | 500k | 20-40% | Getting competitive |
| 4:00 | 1M | 40-60% | First checkpoint |
| 8:00 | 2M | 60-70% | Second checkpoint |
| 12:00 | 3M | 65-75% | Third checkpoint |
| 16:00 | 4M | 70-80% | Fourth checkpoint |
| 20:00 | 5M | 75-85% | **Training complete!** |

---

## Good Luck!

You're all set for overnight training! The agent will learn and improve while you sleep.

**When you wake up:**
1. Check the console output or log file
2. Verify win rate improved (should be 60%+)
3. Evaluate against all opponents
4. Test against yourself in PvP mode
5. Submit to competition!

**Need help?** Check the TRAINING_MONITORING.md file for detailed explanations of metrics.
