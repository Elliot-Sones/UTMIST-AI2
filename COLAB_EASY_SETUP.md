# 🚀 Super Easy Google Colab Setup (1 Minute!)

## ✨ NEW: Automatic Checkpoint Saving!

Your `train_agent.py` now **automatically detects Colab** and saves checkpoints to Google Drive!

**No manual setup needed** - just run the code! 🎉

---

## 📋 Complete Colab Notebook (Copy & Paste)

```python
# ============================================================================
# Cell 1: Clone Repository
# ============================================================================
!git clone https://github.com/YOUR_USERNAME/UTMIST-AI2.git
%cd UTMIST-AI2

# ============================================================================
# Cell 2: Install Dependencies
# ============================================================================
!pip install -q stable-baselines3[extra] sb3-contrib gymnasium pygame
!pip install -q torch --index-url https://download.pytorch.org/whl/cu118

# ============================================================================
# Cell 3: Verify GPU
# ============================================================================
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================================================
# Cell 4: Run Training (Checkpoints Auto-Save to Drive!)
# ============================================================================
!python user/train_agent.py

# That's it! The script will:
# 1. Auto-detect you're in Colab
# 2. Ask you to mount Google Drive (click the link, authorize)
# 3. Auto-save all checkpoints to Drive
# 4. Keep checkpoints safe even if Colab disconnects!
```

---

## 🎯 What Happens Automatically

When you run `python user/train_agent.py` in Colab:

```
🔍 Google Colab detected!
======================================================================
📁 Setting up Google Drive for persistent checkpoint storage...
  ⏳ Mounting Google Drive (you may need to authorize)...
  
  [You click the link and authorize Drive access]
  
  ✓ Google Drive mounted successfully!
  ✓ Checkpoints will auto-save to: /content/drive/MyDrive/UTMIST-AI2-Checkpoints
  ✓ Your checkpoints are safe even if Colab disconnects!
======================================================================
```

**That's it!** No manual configuration needed! ✨

---

## 📁 Where Are My Checkpoints?

### In Google Drive:
```
MyDrive/
└── UTMIST-AI2-Checkpoints/
    ├── test_50k_t4/              # 50k test run
    │   ├── monitor.csv
    │   ├── rl_model_5000_steps.zip
    │   ├── rl_model_10000_steps.zip
    │   └── ...
    │
    └── transformer_10M_t4/       # 10M full run
        ├── monitor.csv
        ├── rl_model_100000_steps.zip
        └── ...
```

### Access Anytime:
1. **In Colab**: `/content/drive/MyDrive/UTMIST-AI2-Checkpoints`
2. **In Drive web**: Go to drive.google.com → "UTMIST-AI2-Checkpoints" folder
3. **On your computer**: Download from Drive or sync with Google Drive app

---

## 🔄 Resume Training After Disconnection

If Colab disconnects, just re-run the notebook:

```python
# The script will:
# 1. Auto-detect Colab again
# 2. Mount Drive (your checkpoints are still there!)
# 3. Find the latest checkpoint automatically
# 4. Resume training from where it left off

# You can also manually resume:
!python user/train_agent.py --load-checkpoint /content/drive/MyDrive/UTMIST-AI2-Checkpoints/transformer_10M_t4/rl_model_5000000_steps.zip
```

Or edit the config:
```python
# Add this before running
import sys
sys.path.insert(0, '/content/UTMIST-AI2')
from user.train_agent import TRAIN_CONFIG_10M

# Load from checkpoint
TRAIN_CONFIG_10M["agent"]["load_path"] = "/content/drive/MyDrive/UTMIST-AI2-Checkpoints/transformer_10M_t4/rl_model_5000000_steps.zip"

# Reduce timesteps by what's already done
TRAIN_CONFIG_10M["training"]["timesteps"] = 10_000_000 - 5_000_000  # Continue from 5M

# Then run training
!python user/train_agent.py
```

---

## 📊 Monitor Training (Optional)

Add this cell to watch progress in real-time:

```python
# Run this in a separate cell while training runs
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time

checkpoint_dir = "/content/drive/MyDrive/UTMIST-AI2-Checkpoints/test_50k_t4"

while True:
    clear_output(wait=True)
    
    try:
        # Read monitor file
        df = pd.read_csv(f"{checkpoint_dir}/monitor.csv", skiprows=1)
        
        # Plot rewards
        plt.figure(figsize=(10, 4))
        plt.plot(df.iloc[:, 0])
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title(f'Learning Curve ({len(df)} episodes)')
        plt.grid(True)
        plt.show()
        
        # Print stats
        print(f"Episodes: {len(df)}")
        print(f"Latest reward: {df.iloc[-1, 0]:.2f}")
        print(f"Avg (last 10): {df.iloc[-10:, 0].mean():.2f}")
        
    except FileNotFoundError:
        print("⏳ Waiting for training to start...")
    
    time.sleep(30)  # Update every 30 seconds
```

---

## 🎮 Complete Colab Workflow

### 1️⃣ First Run (50k Test - 15 minutes)
```python
# Just run these 4 cells:

# Cell 1
!git clone https://github.com/YOUR_USERNAME/UTMIST-AI2.git
%cd UTMIST-AI2

# Cell 2
!pip install -q stable-baselines3[extra] sb3-contrib gymnasium pygame torch

# Cell 3
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Cell 4
!python user/train_agent.py
# When prompted, click link and authorize Drive
# Wait 15 minutes
# Done! ✅
```

### 2️⃣ Check Results
```python
# Check your results
!ls /content/drive/MyDrive/UTMIST-AI2-Checkpoints/test_50k_t4/
!tail /content/drive/MyDrive/UTMIST-AI2-Checkpoints/test_50k_t4/monitor.csv
```

### 3️⃣ Run 10M Training (If You Have Colab Pro)
```python
# Edit train_agent.py line 373 first:
# Change TRAIN_CONFIG = TRAIN_CONFIG_TEST
# To     TRAIN_CONFIG = TRAIN_CONFIG_10M

# Then run:
!python user/train_agent.py
# Takes ~10-12 hours on Colab Pro (24hr limit)
# Checkpoints auto-save every 100k steps to Drive!
```

---

## ⚡ Advantages of Auto-Setup

### ✅ Before (Manual):
```python
# You had to do:
from google.colab import drive
drive.mount('/content/drive')

import sys
sys.path.insert(0, '/content/drive/MyDrive/UTMIST-AI2')

# Manually edit checkpoint paths in code
# Remember to save to Drive each time
# Risk losing checkpoints if you forget
```

### ✨ After (Automatic):
```python
# You just do:
!python user/train_agent.py

# Everything else is automatic! 🎉
```

---

## 🔒 Your Checkpoints Are Safe!

With auto-save to Drive:
- ✅ **Colab disconnects?** → Checkpoints still in Drive
- ✅ **Browser crash?** → Checkpoints still in Drive  
- ✅ **Forgot to save?** → Already saved automatically
- ✅ **Want to continue later?** → Just re-run, resumes automatically
- ✅ **Access from anywhere?** → drive.google.com

---

## 💡 Pro Tips

1. **First time**: When script asks to mount Drive, click the link and authorize
2. **Check space**: Make sure you have 5-10 GB free on Drive (100 checkpoints ≈ 5-8 GB)
3. **Monitor progress**: Use the monitoring cell (above) in a separate cell
4. **Colab Pro recommended**: Free tier times out at 12 hours (10M training takes 10-12 hours)
5. **Resume anytime**: Just re-run the script, it finds latest checkpoint automatically

---

## 🆘 Troubleshooting

### "Drive mount failed"
```python
# Manually mount:
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# Then run training
!python user/train_agent.py
```

### "Checkpoints not saving to Drive"
```python
# Check if Drive is mounted:
!ls /content/drive/MyDrive

# Should show your Drive files
# If empty, mount failed - see above
```

### "Can't find checkpoint folder"
```python
# It's created automatically when training starts:
!ls /content/drive/MyDrive/UTMIST-AI2-Checkpoints

# If training hasn't started yet, folder won't exist
# Wait for first checkpoint at 5k steps (test) or 100k (full)
```

---

## 📱 Quick Reference

| Action | Command |
|--------|---------|
| Clone repo | `!git clone YOUR_REPO_URL && %cd UTMIST-AI2` |
| Install deps | `!pip install -q stable-baselines3[extra] sb3-contrib gymnasium pygame torch` |
| Check GPU | `import torch; print(torch.cuda.get_device_name(0))` |
| Run training | `!python user/train_agent.py` |
| Check checkpoints | `!ls /content/drive/MyDrive/UTMIST-AI2-Checkpoints` |
| View results | `!tail /content/drive/.../monitor.csv` |

---

## 🎉 Summary

**Before**: 10 steps of manual Drive setup, path configuration, risk of data loss

**Now**: 
1. Run `!python user/train_agent.py`
2. Click "Authorize Drive" when prompted
3. Done! Everything else is automatic! ✨

**Your checkpoints are automatically saved to Google Drive and survive disconnections!** 🚀

---

Need more details? Check:
- `COLAB_TRAINING_GUIDE.md` - Full Colab guide with advanced features
- `T4_TRAINING_GUIDE.md` - General training guide
- `QUICK_REFERENCE.md` - Command cheat sheet

