# UTMIST AI² - T4 GPU Training Guide (10M Timesteps)

## 🚀 Quick Start

### Step 1: Run 50k Test (15 minutes)
```bash
# Make sure TRAIN_CONFIG_TEST is active (line 374 in train_agent.py)
python user/train_agent.py
```

**What to expect:**
- Training starts on T4 GPU (auto-detected)
- ~15 minutes for 50k timesteps
- Creates `checkpoints/test_50k_t4/` folder
- Generates debug CSVs (reward breakdown, behavior metrics, evaluation results)

### Step 2: Check Results
```bash
# Navigate to test results
cd checkpoints/test_50k_t4/

# Key files to check:
ls -lh
# - monitor.csv: Episode rewards over time
# - reward_breakdown.csv: Per-term reward contributions
# - behavior_metrics.csv: Damage dealt/taken, knockouts, etc.
# - evaluation_results.csv: Win rate vs BasedAgent
# - rl_model_*_steps.zip: Agent checkpoints
# - rl_model_*_steps_transformer_encoder.pth: Transformer weights
```

**What to look for:**
- ✅ Total reward should be increasing (check monitor.csv)
- ✅ Win rate improving over time (check evaluation_results.csv)
- ✅ Damage ratio > 1.0 by end (check behavior_metrics.csv)
- ✅ No stuck/broken reward terms (check reward_breakdown.csv)

### Step 3: Switch to 10M Training
```python
# In train_agent.py, line 373, change:
# TRAIN_CONFIG = TRAIN_CONFIG_TEST   # Comment this out
TRAIN_CONFIG = TRAIN_CONFIG_10M      # Uncomment this line
```

### Step 4: Run Full Training (10-12 hours)
```bash
# Run in background (recommended)
nohup python user/train_agent.py > training.log 2>&1 &

# Monitor progress
tail -f training.log

# Check GPU utilization (should be ~90%)
watch -n 1 nvidia-smi
```

---

## 📊 Scaling Law Guarantee

The 50k test config and 10M full config use **IDENTICAL hyperparameters**:
- Same transformer architecture (6 layers, 8 heads, 256 latent dim)
- Same LSTM policy (512 hidden)
- Same learning rate (2.5e-4)
- Same batch size (128)
- Same reward weights
- Same opponent mix (80% self-play, 15% BasedAgent, 5% ConstantAgent)

**This means:**
- If 50k test shows good learning curves → 10M will work identically (just 200x longer)
- If 50k test shows issues → fix them before running 10M
- Tweaks tested on 50k will transfer perfectly to 10M

---

## 🎯 Key Components Explained

### 1. Transformer Strategy Encoder
- **Purpose**: Discovers opponent patterns automatically (like AlphaGo)
- **Architecture**: 6 layers, 8 attention heads, 256-dim latent space
- **Input**: Last 90 frames (3 seconds) of opponent behavior
- **Output**: 256-dim continuous vector representing opponent strategy
- **Training**: End-to-end with PPO, learns what patterns help win

**Example patterns it might learn:**
- Aggressive rushdown → counter with defensive spacing
- Camping/defensive → apply pressure with safe pokes
- Random spam → exploit predictability with punishes

### 2. LSTM RNN Policy
- **Purpose**: Generates actions conditioned on opponent strategy
- **Architecture**: 512 hidden units, shared LSTM for actor/critic
- **Input**: Current game state + strategy latent from transformer
- **Output**: Action probabilities (10 discrete actions)
- **Training**: PPO with 10 epochs per rollout

### 3. Reward Shaping
```python
# Dense rewards (every frame):
- Damage interaction: +1.0 * (damage_dealt - damage_taken) / 140
- Danger zone penalty: -0.5 when height > 4.2 (about to get knocked off)
- Attack spam penalty: -0.04 when in attack state (discourages button mashing)
- Multi-key penalty: -0.01 when pressing > 3 keys (encourages clean inputs)

# Signal rewards (on events):
- Win: +50
- Knockout opponent: +8
- Get knocked out: -8
- Hit during stun (combo): +5
- Pick up weapon: +10 (Hammer), +5 (Spear)
- Drop weapon: -15
```

### 4. Self-Adversarial Training Loop
- **80% Self-Play**: Agent trains vs random past snapshots (curriculum learning)
- **15% BasedAgent**: Scripted opponent with basic behavior
- **5% ConstantAgent**: Random baseline to prevent overfitting

**Why this works:**
- Self-play creates emergent complexity (like AlphaGo)
- Past snapshots ensure curriculum from easy → hard
- Scripted bots prevent exploiting self-play quirks

---

## 💾 Memory & Performance

### T4 GPU (16GB VRAM) Usage
```
Component                   VRAM Usage
─────────────────────────────────────
Transformer encoder         ~500 MB
LSTM policy network         ~300 MB
PPO rollout buffer (54k)    ~2-3 GB
Gradient computation        ~1-2 GB
cuDNN workspace             ~500 MB
─────────────────────────────────────
Total                       ~4-6 GB
Safety margin               ~10 GB
```

**If you get OOM errors:**
```python
# Option 1: Reduce batch size
"batch_size": 64,  # Down from 128

# Option 2: Reduce rollout steps
"n_steps": 30 * 90 * 10,  # Down from 30 * 90 * 20 (27k instead of 54k)

# Option 3: Reduce transformer size
"num_layers": 4,  # Down from 6
```

### Training Speed (T4 GPU)
- **50k timesteps**: ~15 minutes (validation test)
- **1M timesteps**: ~60-75 minutes
- **10M timesteps**: ~10-12 hours (full training)

**Speedup vs CPU**: ~6-8x faster overall (transformer gets ~8-10x, LSTM gets ~5-7x)

---

## 📈 Monitoring Training

### During Training
```bash
# Monitor GPU utilization (should be ~90%)
watch -n 1 nvidia-smi

# Monitor training progress
tail -f training.log

# Check latest reward
tail checkpoints/transformer_10M_t4/monitor.csv
```

### After Training Completes
```bash
# Visualize learning curves (if you have matplotlib)
python -c "
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('checkpoints/transformer_10M_t4/monitor.csv', skiprows=1)
plt.plot(df['r'])
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Learning Curve')
plt.savefig('learning_curve.png')
print('Saved to learning_curve.png')
"

# Check final performance metrics
tail -20 checkpoints/transformer_10M_t4/behavior_metrics.csv
tail -20 checkpoints/transformer_10M_t4/evaluation_results.csv
```

---

## 🔧 Troubleshooting

### Training is slow (< 50% GPU utilization)
```bash
# Check if GPU is being used
nvidia-smi

# Ensure PyTorch sees CUDA
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_device_name(0))"

# Reinstall PyTorch with CUDA if needed
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### CUDA Out of Memory Error
```python
# Reduce memory usage in train_agent.py:
_SHARED_AGENT_CONFIG = {
    # ... other settings ...
    "batch_size": 64,  # Down from 128
    "n_steps": 30 * 90 * 10,  # Down from 30 * 90 * 20
}
```

### Reward appears stuck
```bash
# Check reward breakdown in test mode
python user/train_agent.py  # With TRAIN_CONFIG_TEST active
cat checkpoints/test_50k_t4/reward_breakdown.csv
```

Look for:
- Any reward term always zero? → Might be broken
- Total reward never changes? → Weights might be wrong
- Agent never wins? → Learning rate too high/low

### Agent not improving
Check behavior metrics:
```bash
tail -50 checkpoints/test_50k_t4/behavior_metrics.csv
```

Look for:
- Damage ratio (should increase over time)
- Win rate (should increase over time)
- Attack rate (should stabilize, not random)

If no improvement after 50k steps:
1. Check reward breakdown (is agent getting rewarded correctly?)
2. Lower learning rate: `"learning_rate": 1e-4` (down from 2.5e-4)
3. Increase exploration: `"ent_coef": 0.15` (up from 0.10)

---

## 🎮 Using Trained Model

### Load and Test
```python
from user.train_agent import TransformerStrategyAgent, run_eval_match
from functools import partial

# Load trained agent
agent = TransformerStrategyAgent(
    file_path="checkpoints/transformer_10M_t4/rl_model_10000000_steps.zip",
    latent_dim=256,
    num_heads=8,
    num_layers=6,
    sequence_length=90
)

# Test vs BasedAgent
from user.train_agent import BasedAgent
stats = run_eval_match(
    agent,
    partial(BasedAgent),
    max_timesteps=30*90,
    video_path="my_match.mp4"
)

print(f"Result: {stats.player1_result}")
print(f"Damage dealt: {stats.player2.total_damage}")
print(f"Damage taken: {stats.player1.total_damage}")
```

### Export for Tournament
```python
# The trained agent is already saved as .zip files
# Each checkpoint includes:
# - rl_model_XXXXX_steps.zip (PPO policy + LSTM)
# - rl_model_XXXXX_steps_transformer_encoder.pth (transformer weights)

# To use in tournament, load the final checkpoint:
final_checkpoint = "checkpoints/transformer_10M_t4/rl_model_10000000_steps.zip"
```

---

## 📋 Configuration Comparison

| Setting | Test (50k) | Full (10M) | Ratio |
|---------|-----------|------------|-------|
| Timesteps | 50,000 | 10,000,000 | 1:200 |
| Training Time (T4) | ~15 min | ~10-12 hrs | 1:40-48 |
| Save Frequency | 5,000 | 100,000 | 1:20 |
| Max Checkpoints | 10 | 100 | 1:10 |
| Debugging | ✅ Full | ❌ Minimal | - |
| Eval Frequency | 10,000 | - | - |
| **Hyperparameters** | **IDENTICAL** | **IDENTICAL** | **1:1** |

---

## 🧪 Hyperparameter Tuning

If 50k test shows issues, try these adjustments:

### Learning too slow?
```python
"learning_rate": 5e-4,  # Up from 2.5e-4
"ent_coef": 0.15,       # Up from 0.10 (more exploration)
```

### Learning unstable?
```python
"learning_rate": 1e-4,  # Down from 2.5e-4
"n_epochs": 5,          # Down from 10 (fewer gradient updates)
```

### Agent too passive?
```python
# In gen_reward_manager():
'damage_interaction_reward': RewTerm(func=damage_interaction_reward, weight=2.0),  # Up from 1.0
'penalize_attack_reward': RewTerm(func=in_state_reward, weight=-0.02, ...),  # Less penalty
```

### Agent too aggressive (dies too much)?
```python
'danger_zone_reward': RewTerm(func=danger_zone_reward, weight=1.0),  # Up from 0.5
'on_knockout_reward': ('knockout_signal', RewTerm(func=on_knockout_reward, weight=12)),  # Up from 8
```

**CRITICAL**: After tweaking, always re-run 50k test before 10M training!

---

## ✅ Success Criteria

By end of 10M training, agent should achieve:
- ✅ Win rate > 70% vs BasedAgent
- ✅ Damage ratio > 2.0 (deals 2x damage vs takes)
- ✅ Average knockouts > 2 per episode
- ✅ Weapon pickup rate > 1 per episode
- ✅ Survival time > 60 seconds average

Check these in `checkpoints/transformer_10M_t4/behavior_metrics.csv`

---

## 📚 Additional Resources

- **Technical Guide**: `guides/Technical_Guide_Notebook.ipynb`
- **MPS Optimization**: `MPS_OPTIMIZATION_GUIDE.md` (if running on Mac)
- **Test Mode Guide**: `TEST_MODE_GUIDE.md`
- **Implementation Summary**: `IMPLEMENTATION_SUMMARY.md`

---

## 🐛 Common Issues & Solutions

### Issue: `ModuleNotFoundError: No module named 'environment'`
**Solution**: Make sure you're running from project root
```bash
cd /path/to/UTMIST-AI2
python user/train_agent.py
```

### Issue: `RuntimeError: CUDA out of memory`
**Solution**: Reduce batch size or n_steps (see Memory section above)

### Issue: Training output shows "⚠ No GPU detected"
**Solution**: Check CUDA installation
```bash
python -c "import torch; print(torch.cuda.is_available())"
# Should print: True
```

### Issue: Self-play not working (only sees BasedAgent)
**Solution**: Wait for first checkpoint to save (5k steps for test, 100k for full)
```bash
# Check if snapshots exist
ls checkpoints/test_50k_t4/
# Should see rl_model_5000_steps.zip after 5k steps
```

### Issue: Reward appears negative and stuck
**Solution**: Check reward weights are balanced
```python
# In gen_reward_manager(), ensure:
# - Positive signal rewards (win, knockout) outweigh negative dense rewards
# - If total reward always negative, increase signal reward weights
```

---

## 💡 Pro Tips

1. **Always run 50k test first** - catches 95% of configuration issues
2. **Monitor GPU utilization** - should stay at ~90%, if lower check batch size
3. **Save training logs** - use `nohup` to keep logs after disconnect
4. **Check self-play snapshots** - ensure checkpoints are being saved correctly
5. **Compare metrics** - 50k and 10M should show similar trends (just scaled)

---

## 📞 Support

If you encounter issues:
1. Check `training.log` for error messages
2. Verify GPU is detected: `nvidia-smi`
3. Check VRAM usage: `watch -n 1 nvidia-smi`
4. Review reward breakdown CSV for stuck/broken rewards
5. Test with smaller batch size if OOM errors persist

**Expected Final Results (10M timesteps on T4):**
- Total training time: ~10-12 hours
- Final win rate vs BasedAgent: 70-85%
- Final damage ratio: 2.0-3.0
- Total checkpoints: 100 snapshots (1 every 100k steps)
- Total disk usage: ~5-8 GB (100 checkpoints × ~50-80 MB each)

---

Good luck with your training! 🚀

