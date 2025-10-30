# Quick Reference - T4 GPU Training

## 🎯 One-Page Cheat Sheet

### Running Training

```bash
# 1. Test run (15 minutes) - DO THIS FIRST
python user/train_agent.py

# 2. Check results
cd checkpoints/test_50k_t4/
cat monitor.csv | tail -20          # Episode rewards
cat evaluation_results.csv | tail  # Win rate trends

# 3. Switch to 10M training
# Edit train_agent.py line 373:
# TRAIN_CONFIG = TRAIN_CONFIG_10M

# 4. Full training (10-12 hours)
nohup python user/train_agent.py > training.log 2>&1 &
tail -f training.log
```

---

## 📊 Key Files to Monitor

| File | What it shows | When to check |
|------|---------------|---------------|
| `training.log` | Console output | During training (real-time) |
| `monitor.csv` | Episode rewards | After training (plot trends) |
| `reward_breakdown.csv` | Per-term rewards | After test (debug rewards) |
| `behavior_metrics.csv` | Win rate, damage | After test (check improvement) |
| `evaluation_results.csv` | Win rate vs BasedAgent | After test (validate learning) |

---

## 🔧 Configuration Switch

```python
# Line 373 in train_agent.py

# For 50k test (15 min):
TRAIN_CONFIG = TRAIN_CONFIG_TEST

# For 10M full training (10-12 hrs):
TRAIN_CONFIG = TRAIN_CONFIG_10M
```

---

## 📈 Expected Results

### 50k Test (~15 min)
- ✅ Reward increases from ~0 to ~20-30
- ✅ Win rate improves from 0% to 30-50%
- ✅ Damage ratio improves from 0.5 to 1.2

### 10M Full (~10-12 hrs)
- ✅ Reward increases from ~0 to ~60-80
- ✅ Win rate improves from 0% to 70-85%
- ✅ Damage ratio improves from 0.5 to 2.0-3.0

---

## 🚨 Troubleshooting

| Problem | Solution |
|---------|----------|
| "CUDA out of memory" | Reduce `batch_size: 64` (line 300) |
| "No GPU detected" | Check `torch.cuda.is_available()` |
| GPU usage < 50% | Check batch size, increase if possible |
| Reward stuck/negative | Check reward_breakdown.csv, adjust weights |
| Self-play not working | Wait for checkpoint (5k for test, 100k for full) |

---

## 💾 Memory Usage (T4 16GB)

```
Transformer encoder:    ~500 MB
LSTM policy:            ~300 MB
Rollout buffer:         ~2-3 GB
Gradients:              ~1-2 GB
────────────────────────────────
Total:                  ~4-6 GB (safe!)
```

---

## ⚙️ Key Hyperparameters

```python
Transformer:
- Layers: 6
- Heads: 8
- Latent dim: 256
- Sequence length: 90 frames (3 sec)

LSTM Policy:
- Hidden size: 512
- Net arch: [96, 96] (actor), [96, 96] (critic)

PPO:
- Learning rate: 2.5e-4
- Batch size: 128
- Rollout steps: 54,000
- Epochs: 10

Opponent Mix:
- Self-play: 80%
- BasedAgent: 15%
- ConstantAgent: 5%
```

---

## 📁 Checkpoint Structure

```
checkpoints/
├── test_50k_t4/               # Test run (50k)
│   ├── monitor.csv
│   ├── reward_breakdown.csv
│   ├── behavior_metrics.csv
│   ├── evaluation_results.csv
│   ├── rl_model_5000_steps.zip
│   ├── rl_model_5000_steps_transformer_encoder.pth
│   ├── ... (every 5k)
│   └── rl_model_50000_steps.zip
│
└── transformer_10M_t4/        # Full run (10M)
    ├── monitor.csv
    ├── rl_model_100000_steps.zip
    ├── rl_model_100000_steps_transformer_encoder.pth
    ├── ... (every 100k)
    └── rl_model_10000000_steps.zip
```

---

## 🎮 Using Trained Agent

```python
from user.train_agent import TransformerStrategyAgent

# Load final model
agent = TransformerStrategyAgent(
    file_path="checkpoints/transformer_10M_t4/rl_model_10000000_steps.zip",
    latent_dim=256,
    num_heads=8,
    num_layers=6,
    sequence_length=90
)

# Test it
from user.train_agent import run_eval_match, BasedAgent
from functools import partial

stats = run_eval_match(
    agent, 
    partial(BasedAgent),
    video_path="my_match.mp4"
)

print(f"Result: {stats.player1_result}")
```

---

## ✅ Pre-Flight Checklist

Before 10M training:
- [ ] 50k test completed successfully
- [ ] GPU detected ("Using NVIDIA CUDA GPU" in log)
- [ ] Reward increasing (check monitor.csv)
- [ ] Win rate improving (check evaluation_results.csv)
- [ ] No OOM errors in training.log
- [ ] Self-play checkpoints saving (ls checkpoints/test_50k_t4/)
- [ ] GPU utilization ~90% (nvidia-smi)
- [ ] Changed line 373 to TRAIN_CONFIG_10M

---

## 📊 Scaling Law Guarantee

```
Test:  50k timesteps × 1 = 50k    (~15 min)
Full:  50k timesteps × 200 = 10M  (~10-12 hrs)

Ratio: 1:200
Hyperparameters: IDENTICAL
Behavior: IDENTICAL (just scaled)
```

**This means**: If 50k test works well → 10M will work identically!

---

## 🔍 Monitoring Commands

```bash
# GPU utilization (run in separate terminal)
watch -n 1 nvidia-smi

# Training progress
tail -f training.log

# Latest episode reward
tail -1 checkpoints/transformer_10M_t4/monitor.csv

# Check if training still running
ps aux | grep train_agent.py
```

---

## 🎯 Success Metrics (10M Training)

| Metric | Target |
|--------|--------|
| Win rate vs BasedAgent | > 70% |
| Damage ratio (dealt/taken) | > 2.0 |
| Avg knockouts per episode | > 2 |
| Avg survival time | > 60 sec |
| Weapon pickups per episode | > 1 |

---

## 📞 Quick Links

- **Full Guide**: `T4_TRAINING_GUIDE.md`
- **Change Summary**: `CHANGES_SUMMARY.md`
- **Training Script**: `user/train_agent.py`

---

**TIP**: Bookmark this page and keep it open during training! 📌

