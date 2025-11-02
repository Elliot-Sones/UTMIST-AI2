# Quick Start: Overnight Training

**Goal:** Train your RL agent overnight for 5-10M steps to be competitive for the UTMIST AI² competition.

## TL;DR - 3 Commands to Start

```bash
# 1. Run pre-flight check (~2 minutes)
python user/pre_flight_check.py

# 2. (Optional) Test current model baseline
python user/evaluate_baseline.py --model checkpoints/simplified_training/latest_model.zip

# 3. Launch training (automated setup)
./launch_training.sh
```

That's it! Training will run overnight in a persistent tmux session.

---

## What Was Enhanced

Your training infrastructure now has:

### ✅ Enhanced Monitoring
- **CSV Export**: All metrics automatically saved to CSV for easy plotting
- **Action Distribution Tracking**: See which moves your agent is using
- **Success Milestones**: Automatic celebration messages at 25%, 50%, 75% win rates
- **Detailed Console Logs**: Every rollout shows reward, win rate, damage stats

### ✅ Pre-Flight Validation
- **Pre-Flight Check Script**: Validates GPU, disk space, dependencies, reward functions
- **Training Loop Test**: Runs 10k steps to estimate training time
- **Baseline Evaluation**: Compare before/after training performance

### ✅ Easy Launch & Monitoring
- **Automated Launch Script**: One command to set up tmux and start training
- **Monitoring Guide**: Complete instructions for checking progress remotely
- **Plotting Tools**: Ready-to-use Python snippets for visualizing progress

---

## File Guide

| File | Purpose |
|------|---------|
| **[OVERNIGHT_TRAINING_GUIDE.md](OVERNIGHT_TRAINING_GUIDE.md)** | **📖 START HERE** - Complete guide with monitoring, troubleshooting, expected timelines |
| `launch_training.sh` | Automated launch script (runs checks + starts tmux session) |
| `user/pre_flight_check.py` | Validates system ready for overnight run |
| `user/evaluate_baseline.py` | Test agent against all 8 opponent types |
| `user/train_with_strategy_encoder.py` | Main training script (now with CSV export!) |
| `user/callbacks/training_metrics_callback.py` | Enhanced with CSV, actions, milestones |

---

## Expected Results After 5M Steps

**If training goes well:**
- ✅ Overall win rate: **60-80%**
- ✅ Positive average reward: **+20 to +40**
- ✅ Damage differential: **+10 to +20**
- ✅ Can beat at least 6 out of 8 opponent types
- ✅ Recognizes and adapts to opponent strategies

**Training time:**
- CUDA GPU: **8-12 hours**
- Apple MPS: **12-18 hours**
- CPU: **24-48 hours** (not recommended)

---

## Quick Monitoring Cheat Sheet

### Before You Sleep
```bash
# 1. Launch training
./launch_training.sh

# 2. Watch for 30 seconds - verify:
#    - FPS > 300 (GPU) or > 50 (CPU)
#    - Rewards are changing
#    - No errors

# 3. Detach: Press Ctrl+B, then D

# 4. Go to sleep! 😴
```

### When You Wake Up
```bash
# 1. Check if still running
tmux ls                                  # Should see "rl_training"

# 2. View progress
tail -n 50 training_output*.log          # Last 50 lines

# 3. Attach to see live output
tmux attach -t rl_training

# 4. Evaluate final performance
python user/evaluate_baseline.py --model /tmp/strategy_encoder_training/final_model.zip
```

### Remote Monitoring (From Phone/Tablet via SSH)
```bash
# Quick status check
tail -n 20 training_output*.log          # See recent progress
ls -lht /tmp/strategy_encoder_training/  # Check latest checkpoint
```

---

## Understanding the Console Output

### Compact Line (Every Rollout)
```
Timestep     Rollout  Reward       Win%     Dmg+/-          FPS        Time
250,000      61       +5.3         25.0     +3.5(12.1/8.6)  515        00:08:05
```

**What to watch:**
- **Timestep**: Progress toward 5M
- **Reward**: Should increase over time (-15 → 0 → +20 → +40)
- **Win%**: Should increase (0% → 25% → 50% → 75%)
- **Dmg+/-**: Net damage (dealt/taken) - should become positive
- **FPS**: Training speed - higher is better

### Detailed Stats (Every 50 Rollouts)
```
DETAILED STATS @ 250,000 steps
Episodes: 100 completed
  Reward:  avg=+5.3  std=12.5  min=-45.2  max=+85.3
  Win Rate: 25.0% (25/100)

Action Distribution (last 40,960 actions):
  →          : 45.2% (18,514 times)
  light_atk  : 28.3% (11,592 times)
  jump       : 15.1% (6,185 times)
  Attack Rate: 35.4% (combined light/heavy/dash)
```

**Good signs:**
- ✅ Reward increasing
- ✅ Win rate growing
- ✅ Attack rate > 20% (agent is fighting, not running away)
- ✅ Action distribution varied (exploring different moves)

---

## Success Metrics Timeline

| Steps | Time | Expected Win% | Expected Reward | Status |
|-------|------|---------------|-----------------|--------|
| 0 | 0:00 | 0% | -15 | Just started |
| 100k | 1:00 | 5-10% | -5 | Learning basics |
| 500k | 4:00 | 25-40% | +5 | Getting competitive |
| 1M | 8:00 | 40-60% | +15 | Solid performance |
| 2M | 12:00 | 60-70% | +25 | Strong agent |
| 5M | 20:00 | **70-85%** | **+35** | **🏆 Competition ready!** |

---

## Troubleshooting Quick Fixes

**Training stuck at 0% win rate after 200k steps?**
```bash
# Check if agent is attacking
tail -n 100 training_output*.log | grep "Attack Rate"
# Should be > 20%
```

**Training too slow (< 100 FPS)?**
```bash
# Check device
python -c "import torch; print(torch.cuda.is_available())"
# Should be True (CUDA) for fast training
```

**Out of disk space?**
```bash
# Check space
df -h /tmp
# Need at least 2GB free
```

**Want to stop training early?**
```bash
tmux attach -t rl_training
# Press Ctrl+C (will save checkpoint before exiting)
```

---

## After Training: What to Do

### 1. Evaluate Performance
```bash
python user/evaluate_baseline.py --model /tmp/strategy_encoder_training/final_model.zip --episodes 20
```

**Good result:**
- Overall win rate > 60%
- Strong against 5+ opponent types
- Average damage differential > +10

### 2. Plot Training Progress
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/tmp/strategy_encoder_training/training_metrics.csv')

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(df['timestep'], df['avg_reward'])
plt.title('Reward')

plt.subplot(1, 3, 2)
plt.plot(df['timestep'], df['win_rate'] * 100)
plt.title('Win Rate (%)')

plt.subplot(1, 3, 3)
plt.plot(df['timestep'], df['damage_diff'])
plt.title('Damage Differential')

plt.tight_layout()
plt.savefig('training_results.png')
```

### 3. Test Against Yourself
```bash
# Edit user/pvp_match.py to use your model
python user/pvp_match.py
```

Play against your trained agent and see if it can counter your strategies!

### 4. Submit to Competition
Follow the competition submission guidelines with your final model!

---

## Need More Help?

📖 **Read the full guide:** [OVERNIGHT_TRAINING_GUIDE.md](OVERNIGHT_TRAINING_GUIDE.md)

Contains:
- Detailed monitoring instructions
- What good vs. bad training looks like
- Comprehensive troubleshooting
- TensorBoard setup
- Advanced plotting examples

---

## Final Checklist

Before you sleep:

- [ ] Ran `python user/pre_flight_check.py` - all checks passed
- [ ] Started training with `./launch_training.sh`
- [ ] Watched console for 30 seconds - FPS > 300, rewards changing
- [ ] Detached from tmux (Ctrl+B, then D)
- [ ] Training session still running: `tmux ls` shows "rl_training"

When you wake up:

- [ ] Training completed (or still running if < 20 hours)
- [ ] Final model exists: `ls -l /tmp/strategy_encoder_training/final_model.zip`
- [ ] Win rate > 60% in evaluation
- [ ] Agent can beat you in PvP!

**🚀 You're ready! Good luck with your overnight training!**
