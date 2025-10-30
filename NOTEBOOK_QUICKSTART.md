# 📓 Notebook Training - Quick Start

## What I've Created for You

I've set up everything you need to train your AI in Jupyter notebooks:

### 1. **Quick Start Notebook** ⚡
**File**: `guides/Quick_Start_Training.ipynb`

A ready-to-run notebook with:
- ✅ All imports and setup
- ✅ Easy-to-modify configuration
- ✅ Complete training pipeline
- ✅ Evaluation and visualization
- ✅ Heavily commented for learning

**Perfect for**: First-time users, quick experiments, learning

### 2. **Comprehensive Guide** 📚
**File**: `NOTEBOOK_TRAINING_GUIDE.md`

A detailed guide covering:
- Why use notebooks vs scripts
- Complete setup instructions
- Cell-by-cell breakdown
- Pro tips and best practices
- Debugging and troubleshooting
- Advanced visualization techniques

**Perfect for**: Reference, advanced usage, solving problems

---

## 🚀 How to Get Started (3 Steps)

### Step 1: Launch Jupyter

```bash
# From your project root
cd /Users/elliot18/Desktop/Home/Projects/UTMIST-AI2

# Start Jupyter Lab (recommended)
jupyter lab

# OR Jupyter Notebook
jupyter notebook
```

### Step 2: Open the Quick Start Notebook

In Jupyter, navigate to:
```
guides/Quick_Start_Training.ipynb
```

### Step 3: Run All Cells!

1. Click "Run" → "Run All Cells" (or use Shift+Enter on each cell)
2. Wait ~5-10 minutes for training to complete
3. See your results!

That's it! 🎉

---

## 📊 What You'll See

The notebook will:
1. ✓ Detect your device (MPS/CUDA/CPU)
2. ✓ Create a transformer strategy agent
3. ✓ Train for 20,000 steps (~5-10 min)
4. ✓ Run evaluation matches
5. ✓ Show training curves
6. ✓ Save checkpoints

---

## 💡 Key Differences: Notebook vs Script

### Your Current Script (`train_agent.py`)
```python
# Modify config at top of file
TRAIN_CONFIG = {
    "total_timesteps": 200_000,
    # ... other settings
}

# Run entire script
python user/train_agent.py
```

**Pros**: Great for long runs, production
**Cons**: Must restart to change config

### New Notebook (`Quick_Start_Training.ipynb`)
```python
# Cell 3: Easy config modification
CONFIG = {
    "total_timesteps": 20_000,  # Just change this!
    # ... other settings
}
# Run just this cell, then continue
```

**Pros**: Interactive, visual, easy to experiment
**Cons**: Not ideal for very long runs

---

## 🎯 Common Use Cases

### Experiment with Hyperparameters
```python
# In notebook Cell 3 (CONFIG)
CONFIG["learning_rate"] = 5e-4  # Try higher learning rate
CONFIG["ent_coef"] = 0.15       # More exploration

# Re-run training cell to test!
```

### Quick Test Run
```python
CONFIG["total_timesteps"] = 10_000  # Fast 5-min test
CONFIG["run_name"] = "test_1"
```

### Load and Continue Training
```python
CONFIG["load_checkpoint"] = "checkpoints/notebook_quick_test/rl_model_20000_steps.zip"
CONFIG["total_timesteps"] = 50_000  # Train 30k more steps
```

### Compare Experiments
```python
# Run multiple times with different configs
# Then plot together:
runs = ['experiment_1', 'experiment_2', 'experiment_3']
for run in runs:
    df = pd.read_csv(f"checkpoints/{run}/monitor.csv", skiprows=1)
    plt.plot(df['r'].rolling(20).mean(), label=run)
plt.legend()
plt.show()
```

---

## 🔧 Customization Guide

### Modify Training Duration
```python
# Cell 3
CONFIG["total_timesteps"] = 50_000  # ~15-20 min
# or
CONFIG["total_timesteps"] = 100_000  # ~30-40 min
```

### Change Transformer Architecture
```python
# Cell 3
CONFIG["latent_dim"] = 128       # Smaller (faster)
CONFIG["num_heads"] = 4          # Fewer heads
CONFIG["num_layers"] = 4         # Shallower
```

### Adjust PPO Settings
```python
# Cell 3
CONFIG["learning_rate"] = 5e-4   # Higher LR (faster learning, less stable)
CONFIG["ent_coef"] = 0.05        # Less exploration
CONFIG["batch_size"] = 64        # Smaller (less memory)
```

### Enable Test/Debug Mode
```python
# Cell 3
CONFIG["total_timesteps"] = 10_000
CONFIG["save_freq"] = 2_000
CONFIG["eval_freq"] = 2_000
# Faster feedback for debugging!
```

---

## ⚠️ Troubleshooting

### "Module not found" Error
**Fix**: Make sure you run Cell 1 (Setup) first!

### "MPS out of memory"
**Fix**: Reduce batch size in Cell 3:
```python
CONFIG["batch_size"] = 64  # Instead of 128
```

### Notebook becomes slow/unresponsive
**Fix**: 
1. Kernel → Interrupt
2. Avoid printing in loops
3. Close other applications

### Can't see plots
**Fix**: Make sure you have this at the top:
```python
%matplotlib inline
```

### Changes not reflected
**Fix**: Use auto-reload:
```python
# Add at top of notebook
%load_ext autoreload
%autoreload 2
```

---

## 📈 Next Steps

### 1. Run Your First Training (5 minutes)
- Open `Quick_Start_Training.ipynb`
- Run all cells
- See results!

### 2. Experiment (10 minutes)
- Modify CONFIG in Cell 3
- Try different learning rates, batch sizes
- Compare results

### 3. Read the Guide (20 minutes)
- Open `NOTEBOOK_TRAINING_GUIDE.md`
- Learn advanced techniques
- Explore visualization options

### 4. Scale Up (optional)
- Increase `total_timesteps` to 100k-200k
- Add self-play opponents
- Use overnight for long runs

---

## 💬 Questions?

### "Should I switch from my script to notebooks?"
**Answer**: Use both!
- Notebooks for: Experimentation, debugging, short runs
- Scripts for: Production, overnight training, automation

### "Can I use the same config as my script?"
**Answer**: Yes! Just copy `TRAIN_CONFIG` from `train_agent.py` to Cell 3.

### "Will this work on my Apple Silicon Mac?"
**Answer**: Yes! The notebook automatically detects and uses MPS GPU acceleration.

### "Can I resume training if I close the notebook?"
**Answer**: Yes! Load from checkpoint:
```python
CONFIG["load_checkpoint"] = "checkpoints/your_run/rl_model_10000_steps.zip"
```

### "How long does training take?"
**Answer**:
- 10k steps: ~3-5 minutes (testing)
- 20k steps: ~5-10 minutes (quick experiment)
- 50k steps: ~15-25 minutes (good results)
- 200k steps: ~1-1.5 hours (production)

---

## 📁 File Structure

After using notebooks, you'll have:

```
UTMIST-AI2/
├── guides/
│   ├── Quick_Start_Training.ipynb    ← Your main training notebook
│   └── UTMIST-RL-SB3-Demo.ipynb      ← Original demo (CartPole)
├── user/
│   └── train_agent.py                ← Your original script (still works!)
├── checkpoints/
│   └── notebook_quick_test/          ← Notebook training outputs
│       ├── monitor.csv               ← Training logs
│       ├── rl_model_5000_steps.zip   ← Checkpoints
│       └── ...
├── NOTEBOOK_TRAINING_GUIDE.md        ← Comprehensive reference
└── NOTEBOOK_QUICKSTART.md            ← This file!
```

---

## 🎉 Summary

✅ **Created**: Ready-to-use training notebook  
✅ **Created**: Comprehensive training guide  
✅ **Works with**: Your existing `train_agent.py` code  
✅ **Supports**: MPS (Apple Silicon), CUDA, CPU  
✅ **Includes**: Visualization, evaluation, debugging  

**You're all set!** Open Jupyter and start experimenting! 🚀

---

## Quick Reference Card

| Task | Command/Cell |
|------|--------------|
| Start Jupyter | `jupyter lab` |
| Open notebook | `guides/Quick_Start_Training.ipynb` |
| Run cell | `Shift + Enter` |
| Interrupt training | ⏹ Stop button |
| Change config | Edit Cell 3, re-run |
| Plot results | Cell 9 |
| Evaluate agent | Cell 8 |
| Save model | Cell 10 |
| Resume training | Set `load_checkpoint` in Cell 3 |

---

**Happy training!** 🎮🤖

