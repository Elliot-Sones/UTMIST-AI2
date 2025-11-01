# Training Monitoring Guide

This guide explains how to monitor your agent's training progress and understand the metrics.

## Quick Start

### Test Logging (Recommended First Step)
Before running full training, verify logging is working:

```bash
python user/test_training_logging.py
```

This runs a quick 10,000-step test to ensure all logging is functional.

### Start Full Training

```bash
python user/train_with_strategy_encoder.py
```

## What You'll See

### 1. Training Metrics (Every Rollout)

You'll see a continuous stream of metrics like this:

```
Timestep     Rollout  Reward       Win%     Dmg+/-          FPS        Time
--------------------------------------------------------------------------------
4,096        1        -12.34       0.0      -5.2(10.3/15.5) 512        00:00:08
8,192        2        -8.45        10.0     -2.1(15.2/17.3) 524        00:00:16
12,288       3        -5.23        20.0     +1.5(18.4/16.9) 518        00:00:24
```

**Columns:**
- **Timestep**: Total environment steps completed
- **Rollout**: Number of rollouts completed (one rollout = n_steps * n_envs)
- **Reward**: Average episode reward (moving average over last 100 episodes)
- **Win%**: Win rate percentage
- **Dmg+/-**: Net damage (dealt - taken), with breakdown (dealt/taken)
- **FPS**: Training speed (frames per second)
- **Time**: Elapsed time (HH:MM:SS)

### 2. Detailed Stats (Every 50 Rollouts)

Periodically you'll see detailed breakdowns:

```
--------------------------------------------------------------------------------
                      DETAILED STATS @ 200,000 steps
--------------------------------------------------------------------------------
Episodes: 100 completed
  Reward:  avg=-3.45  std=12.34  min=-45.23  max=78.90
  Length:  avg=1234  std=456
  Win Rate: 35.0% (35/100)

Damage Stats:
  Dealt: avg=45.3  max=123.4
  Taken: avg=38.7  max=98.2
  Net:   avg=+6.6

Training Metrics:
  Policy Loss:     0.0234
  Value Loss:      12.3456
  Entropy:         0.0123
  KL Divergence:   0.012345
  Clip Fraction:   0.123
  Learning Rate:   7.50e-05
  Clip Range:      0.135
--------------------------------------------------------------------------------
```

### 3. Population Updates (Every 100k Steps)

```
======================================================================
POPULATION UPDATE AT 200,000 STEPS
======================================================================
✓ Agent added to population!
  Population: 5 agents (1 weak, 4 strong)
  Avg win rate: 0.65
```

### 4. Checkpoints (Every 50k Steps)

```
Saving checkpoint at step 50000
✓ Saved: checkpoints/strategy_encoder_training/rl_model_50000_steps.zip
```

## Understanding the Metrics

### Signs Your Agent is Learning

✅ **Good Signs:**
1. **Reward increasing**: Average reward should trend upward over time
2. **Win rate improving**: Should increase from ~0% toward 50%+
3. **Net damage positive**: Damage dealt > damage taken
4. **KL divergence stable**: Should stay below 0.05 (target_kl)
5. **Entropy decreasing slowly**: Agent becoming more confident (but not too fast)

❌ **Warning Signs:**
1. **Reward stuck or decreasing**: Model not learning or unstable
2. **Win rate at 0%**: Agent not competitive at all
3. **Large negative damage**: Getting destroyed consistently
4. **KL divergence > 0.1**: Training unstable, learning too fast
5. **Entropy near 0**: Policy collapsed, not exploring

### Key Metrics Explained

**Reward**:
- Combines damage dealt/taken, positioning, and win/loss bonuses
- Should gradually increase from negative toward positive
- Target: > 0 means winning more than losing

**Win Rate**:
- Percentage of episodes won against opponents
- Starts low (~0-20%), should reach 50%+ against scripted opponents
- Target: 60-80% indicates strong performance

**Damage Net**:
- Dealt - Taken (higher is better)
- Positive means you're dealing more damage than taking
- Target: +20 or more indicates dominance

**KL Divergence**:
- Measures how much policy changed in an update
- Too high = unstable training, too low = slow learning
- Target: 0.01-0.03 (controlled by target_kl=0.05)

**Entropy**:
- Measures exploration (higher = more random)
- Should decrease slowly as agent becomes more confident
- Too low too fast = policy collapse

**Clip Fraction**:
- Percentage of policy updates that were clipped
- 0.1-0.3 is healthy (means updates are being controlled)
- Too high (>0.5) = very unstable, too low (<0.05) = not learning much

## Monitoring Training Progress

### Option 1: Console Output (Recommended)

Just watch the terminal! The metrics callback provides everything you need.

### Option 2: TensorBoard

Training also logs to TensorBoard:

```bash
tensorboard --logdir checkpoints/strategy_encoder_training/tb_logs
```

Then open: http://localhost:6006

TensorBoard shows:
- Reward curves over time
- Loss curves (policy, value)
- Learning rate schedule
- Population updates

### Option 3: Check Saved Models

Test your latest model anytime:

```bash
# Edit pvp_match.py to use your model
AGENT_1_MODEL_PATH = "checkpoints/strategy_encoder_training/latest_model.zip"

# Run a match
python user/pvp_match.py
```

## Troubleshooting

### No Logging Output

If you see no logging:
1. Check that verbose=1 is set in RecurrentPPO
2. Verify metrics callback is added to callback list
3. Run the test script: `python user/test_training_logging.py`

### Training Too Slow

If FPS is very low (<100):
1. Reduce n_envs (currently 4)
2. Reduce n_steps (currently 4096)
3. Use a GPU (CUDA > MPS > CPU)
4. Reduce batch_size (currently 2048)

### Agent Not Learning

If reward stays negative after 500k steps:
1. Check KL divergence (should be 0.01-0.05)
2. Check entropy (should decrease slowly)
3. Verify win rate is improving (even slowly)
4. Try increasing learning_rate
5. Check that rewards are being calculated correctly

### Training Unstable

If reward oscillates wildly:
1. Check KL divergence (should be <0.1)
2. Reduce learning_rate
3. Reduce n_epochs (currently 4)
4. Increase batch_size for more stable gradients

## Expected Training Timeline

**Typical milestones (5M timesteps total):**

| Timesteps | Expected Performance |
|-----------|---------------------|
| 0-100k    | Learning basics, ~0-10% win rate |
| 100k-500k | Becoming competitive, 20-40% win rate |
| 500k-1M   | Beating weak opponents, 40-60% win rate |
| 1M-2M     | Solid performance, 60-70% win rate |
| 2M-5M     | Mastery & consistency, 70-85% win rate |

**Training time estimates:**
- **With GPU (CUDA)**: ~8-12 hours for 5M steps
- **With MPS (Apple Silicon)**: ~12-18 hours
- **With CPU**: ~24-48 hours (not recommended)

## Post-Training Benchmark

After training completes, you'll see a benchmark test:

```
======================================================================
RUNNING POST-TRAINING BENCHMARK
======================================================================

Testing against scripted opponents (5 episodes each)...

✓ vs ConstantAgent  : 100.0% win rate | avg reward:   125.3
✓ vs BasedAgent     :  80.0% win rate | avg reward:    78.5
○ vs RandomAgent    :  60.0% win rate | avg reward:    34.2

======================================================================
```

**Symbols:**
- ✓ (≥60% win rate): Strong performance
- ○ (40-59% win rate): Competitive
- ✗ (<40% win rate): Needs more training

## Advanced: Population-Based Training

The training uses population-based self-play:

1. **Weak agents** are saved at 50k, 150k, 300k steps
2. **Strong agents** are added every 100k steps based on diversity
3. **Opponent sampling** pulls from both population and scripted agents
4. This ensures robustness against diverse strategies

Monitor population growth in the logs:
```
Population: 8 agents (3 weak, 5 strong)
```

## Questions?

If something seems wrong:
1. Check this guide for expected behavior
2. Run the test: `python user/test_training_logging.py`
3. Check TensorBoard for detailed curves
4. Test latest model: `python user/pvp_match.py`
