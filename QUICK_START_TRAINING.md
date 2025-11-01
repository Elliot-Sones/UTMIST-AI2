# Quick Start: Training Your Agent

## TL;DR - Start Training Now

```bash
# Test that logging works (10k steps, ~2 minutes)
python user/test_training_logging.py

# If test passes, start full training (5M steps, ~12 hours on MPS)
python user/train_with_strategy_encoder.py
```

## What Changed

✅ **Now you get real-time feedback!**

### Before:
- No console output during training
- No way to know if learning is happening
- Had to wait hours to check if it worked

### After:
- **Live metrics every rollout**: reward, win rate, damage, FPS
- **Detailed stats every 50 rollouts**: complete breakdown of performance
- **Population updates**: see when agents are added to training pool
- **Post-training benchmark**: automatic testing against scripted opponents
- **Auto-save latest model**: always have a recent checkpoint

## What You'll See While Training

```
================================================================================
                         TRAINING STARTED
================================================================================
Timestep     Rollout  Reward       Win%     Dmg+/-          FPS        Time
--------------------------------------------------------------------------------
4,096        1        -15.23       0.0      -8.2(8.3/16.5)  487        00:00:08
8,192        2        -12.45       5.0      -5.1(12.4/17.5) 512        00:00:16
12,288       3        -9.87        10.0     -2.3(15.2/17.5) 524        00:00:24
16,384       4        -7.23        15.0     +0.5(18.3/17.8) 518        00:00:32
...
```

Every 50 rollouts, you'll see detailed breakdowns of:
- Episode statistics (reward mean/std/min/max)
- Win rates and episode lengths
- Damage dealt vs taken
- Training metrics (policy loss, KL divergence, entropy, learning rate)

## Quick Health Check

**Is my agent learning?**

Look at these 3 metrics:

1. **Reward** - Should increase over time (from -20 toward +50)
2. **Win%** - Should increase from 0% toward 50%+
3. **Dmg+/-** - Should trend from negative toward positive

If all three are improving, your agent is learning! 🎉

**How long until it's good?**

| Steps | Win Rate | Status |
|-------|----------|--------|
| 100k  | 0-10%    | Learning basics |
| 500k  | 20-40%   | Getting competitive |
| 1M    | 40-60%   | Beating weak opponents |
| 2M+   | 60-80%   | Strong performance |

## Test Your Agent Anytime

While training or after:

```bash
# Edit pvp_match.py
AGENT_1_MODEL_PATH = "checkpoints/strategy_encoder_training/latest_model.zip"
AGENT_2_TYPE = "human"  # or "rules" or "rl"

# Play against it
python user/pvp_match.py
```

## Troubleshooting

**"No logging appears"**
- Run test first: `python user/test_training_logging.py`
- Check verbose=1 is set in the model

**"Agent not learning after 500k steps"**
- Check that reward is changing (even if slowly)
- Win rate should be >0% after 200k steps
- Try increasing learning rate if completely stuck

**"Training is slow (<100 FPS)"**
- Normal on CPU (~50-100 FPS)
- Should be 300-600 FPS on MPS/GPU
- Reduce n_envs or batch_size if needed

## Full Documentation

For detailed explanations of all metrics, troubleshooting, and advanced monitoring:

📖 See [TRAINING_MONITORING.md](TRAINING_MONITORING.md)

## Summary

You now have:
- ✅ Real-time console logging
- ✅ Detailed performance metrics
- ✅ Training health indicators
- ✅ Auto-save latest model
- ✅ Post-training benchmark
- ✅ Complete documentation

**Just run:** `python user/train_with_strategy_encoder.py`

And watch your agent learn! 🚀
