# Google Colab Training Guide - 10M Timesteps with Persistence

## 🚨 Critical: Colab Limitations

**⚠️ Important Colab Facts:**
- **Free tier timeout**: 12 hours max runtime (then disconnects)
- **Idle timeout**: 90 minutes if no activity
- **Storage**: Files are deleted when session ends
- **Solution**: Save to Google Drive continuously

**For 10M training (~10-12 hours):**
- ✅ Will work on Colab Pro (24hr limit)
- ⚠️ Will NOT finish on free tier (12hr limit)
- 💡 Solution: Use checkpointing + resume training

---

## 🔧 Setup: Mount Google Drive

### Step 1: Add to Top of Notebook

```python
# Cell 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Verify mount worked
import os
print("Drive mounted:", os.path.exists('/content/drive/MyDrive'))
```

### Step 2: Clone Repository to Colab

```python
# Cell 2: Setup project
import os

# Change to your Drive directory
%cd /content/drive/MyDrive

# Clone repo (if not already there)
if not os.path.exists('UTMIST-AI2'):
    !git clone https://github.com/YOUR_USERNAME/UTMIST-AI2.git

%cd UTMIST-AI2

# Verify we're in the right place
!pwd
!ls -la
```

### Step 3: Install Dependencies

```python
# Cell 3: Install requirements
!pip install -q stable-baselines3[extra]
!pip install -q sb3-contrib
!pip install -q gymnasium
!pip install -q pygame
!pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify GPU
import torch
print("CUDA Available:", torch.cuda.is_available())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
```

---

## 💾 Modify Training to Save to Drive

### Option A: Edit Config Directly in Notebook

```python
# Cell 4: Configure training with Drive paths
import sys
sys.path.insert(0, '/content/drive/MyDrive/UTMIST-AI2')

from user.train_agent import *

# Override save path to Google Drive
DRIVE_CHECKPOINT_PATH = "/content/drive/MyDrive/UTMIST-AI2/checkpoints"

# Modify configs to use Drive path
TRAIN_CONFIG_TEST["self_play"]["run_name"] = "test_50k_colab"
TRAIN_CONFIG_10M["self_play"]["run_name"] = "transformer_10M_colab"

# Use TEST config for initial run
TRAIN_CONFIG = TRAIN_CONFIG_TEST  # or TRAIN_CONFIG_10M for full run

print(f"✓ Checkpoints will save to: {DRIVE_CHECKPOINT_PATH}")
print(f"✓ Run name: {TRAIN_CONFIG['self_play']['run_name']}")
```

### Option B: Edit train_agent.py Directly

```python
# In train_agent.py, modify the build_self_play_components call in main():

# Around line 2272, change:
_self_play_handler, save_handler, opponent_cfg = build_self_play_components(
    learning_agent,
    save_path="/content/drive/MyDrive/UTMIST-AI2/checkpoints",  # ADD THIS
    run_name=self_play_run_name,
    save_freq=self_play_save_freq,
    max_saved=self_play_max_saved,
    mode=self_play_mode,
    opponent_mix=self_play_opponent_mix,
    selfplay_handler_cls=self_play_handler_cls,
)
```

---

## 🚀 Running Training in Colab

### For 50k Test (~15 minutes - Safe for Colab Free)

```python
# Cell 5: Run 50k test
# Make sure TRAIN_CONFIG_TEST is active (line 374 in train_agent.py)

# Option 1: Run directly
%cd /content/drive/MyDrive/UTMIST-AI2
!python user/train_agent.py

# Option 2: Run from notebook
from user.train_agent import main
main()
```

### For 10M Training (~10-12 hours - Needs Colab Pro or Checkpointing)

```python
# Cell 6: Run 10M training
# IMPORTANT: Switch to TRAIN_CONFIG_10M in train_agent.py line 373

# Option 1: Direct execution
%cd /content/drive/MyDrive/UTMIST-AI2
!python user/train_agent.py

# Option 2: From notebook with monitoring
import time
from user.train_agent import main

start_time = time.time()
try:
    main()
except KeyboardInterrupt:
    print(f"\n⚠️  Training interrupted after {(time.time() - start_time)/3600:.2f} hours")
    print("✓ Checkpoints saved to Google Drive")
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("✓ Last checkpoint saved to Google Drive")
```

---

## 🔄 Resume Training After Disconnection

If Colab disconnects, you can resume from last checkpoint:

### Step 1: Find Latest Checkpoint

```python
# Cell: Find latest checkpoint
import os
import glob

checkpoint_dir = "/content/drive/MyDrive/UTMIST-AI2/checkpoints/transformer_10M_colab"

# Find all checkpoints
checkpoints = glob.glob(f"{checkpoint_dir}/rl_model_*_steps.zip")
checkpoints.sort(key=lambda x: int(x.split('_')[-2]))  # Sort by step count

if checkpoints:
    latest = checkpoints[-1]
    step_count = int(latest.split('_')[-2])
    print(f"✓ Latest checkpoint: {latest}")
    print(f"✓ Steps completed: {step_count:,}")
    print(f"✓ Steps remaining: {10_000_000 - step_count:,}")
    print(f"✓ Est. time remaining: {(10_000_000 - step_count) / 1_000_000 * 1:.1f} hours")
else:
    print("⚠️  No checkpoints found. Starting from scratch.")
```

### Step 2: Resume Training

```python
# Cell: Resume training from checkpoint
from user.train_agent import *

# Modify config to load from checkpoint
if checkpoints:
    latest_checkpoint = checkpoints[-1]
    steps_completed = int(latest_checkpoint.split('_')[-2])
    steps_remaining = 10_000_000 - steps_completed
    
    # Update config
    TRAIN_CONFIG_10M["agent"]["load_path"] = latest_checkpoint
    TRAIN_CONFIG_10M["training"]["timesteps"] = steps_remaining
    
    print(f"✓ Resuming from: {latest_checkpoint}")
    print(f"✓ Training for {steps_remaining:,} more steps")
    
    # Use the modified config
    TRAIN_CONFIG = TRAIN_CONFIG_10M
else:
    print("⚠️  No checkpoint found, starting fresh training")
    TRAIN_CONFIG = TRAIN_CONFIG_10M

# Run training
main()
```

---

## 📊 Monitor Training Progress

### Real-time Monitoring in Colab

```python
# Cell: Monitor while training runs
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time

checkpoint_dir = "/content/drive/MyDrive/UTMIST-AI2/checkpoints/test_50k_colab"

while True:
    clear_output(wait=True)
    
    # Check if monitor.csv exists
    monitor_path = f"{checkpoint_dir}/monitor.csv"
    if os.path.exists(monitor_path):
        try:
            df = pd.read_csv(monitor_path, skiprows=1, names=['r', 'l', 't'])
            
            # Plot rewards
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(df['r'])
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.title(f'Learning Curve ({len(df)} episodes)')
            plt.grid(True)
            
            plt.subplot(1, 2, 2)
            # Moving average
            window = min(20, len(df))
            if len(df) >= window:
                df['r_ma'] = df['r'].rolling(window=window).mean()
                plt.plot(df['r_ma'], label=f'{window}-episode MA')
                plt.xlabel('Episode')
                plt.ylabel('Reward (Moving Avg)')
                plt.title('Smoothed Learning Curve')
                plt.legend()
                plt.grid(True)
            
            plt.tight_layout()
            plt.show()
            
            # Print stats
            print(f"📊 Training Statistics:")
            print(f"  Episodes: {len(df)}")
            print(f"  Latest reward: {df['r'].iloc[-1]:.2f}")
            print(f"  Avg reward (last 10): {df['r'].tail(10).mean():.2f}")
            print(f"  Best reward: {df['r'].max():.2f}")
            print(f"  Total time: {df['t'].iloc[-1]/3600:.2f} hours")
            
        except Exception as e:
            print(f"⚠️  Error reading monitor: {e}")
    else:
        print("⏳ Waiting for training to start...")
    
    time.sleep(30)  # Update every 30 seconds
```

---

## 🛡️ Prevent Colab Disconnection

### Keep Session Alive (JavaScript Trick)

```python
# Cell: Keep Colab alive
from IPython.display import HTML, display

def keep_colab_alive():
    display(HTML('''
        <script>
        function ClickConnect(){
            console.log("Keeping Colab alive..."); 
            document.querySelector("colab-connect-button").click()
        }
        setInterval(ClickConnect, 60000)  // Click every 60 seconds
        </script>
    '''))

keep_colab_alive()
print("✓ Keep-alive script activated")
```

### Enable Desktop Notifications

```python
# Cell: Setup notifications
from IPython.display import Javascript

def notify_training_complete():
    display(Javascript('''
        new Notification("Training Complete!", {
            body: "Your UTMIST AI² training has finished!",
            icon: "https://colab.research.google.com/img/colab_favicon_256px.png"
        });
    '''))

# Request notification permission
display(Javascript('Notification.requestPermission()'))
```

---

## 📥 Download Checkpoints (Backup)

If you want local backups in addition to Drive:

```python
# Cell: Download checkpoints
from google.colab import files
import shutil

checkpoint_dir = "/content/drive/MyDrive/UTMIST-AI2/checkpoints/transformer_10M_colab"

# Create zip of all checkpoints
!zip -r checkpoints_backup.zip {checkpoint_dir}

# Download
files.download('checkpoints_backup.zip')

print("✓ Checkpoints downloaded to your computer")
```

---

## 🎯 Recommended Colab Strategy

### Strategy 1: Multiple Short Sessions (Free Tier)

```python
# Run in chunks that fit within 12hr limit
# Each session resumes from last checkpoint

# Session 1: 0 → 2M steps (~2 hours)
TRAIN_CONFIG_10M["training"]["timesteps"] = 2_000_000
TRAIN_CONFIG_10M["agent"]["load_path"] = None  # Start fresh

# Session 2: 2M → 4M steps (~2 hours)  
TRAIN_CONFIG_10M["training"]["timesteps"] = 2_000_000
TRAIN_CONFIG_10M["agent"]["load_path"] = "path/to/rl_model_2000000_steps.zip"

# ... Continue until 10M
```

### Strategy 2: One Long Session (Colab Pro)

```python
# Colab Pro has 24hr limit - enough for 10M training!
TRAIN_CONFIG = TRAIN_CONFIG_10M  # Full 10M run
main()
```

### Strategy 3: Test on Colab, Full Training Elsewhere

```python
# Use Colab for 50k test (15 min - perfect!)
TRAIN_CONFIG = TRAIN_CONFIG_TEST
main()

# Then download code and run 10M on:
# - Local machine with GPU
# - Cloud VM (AWS/GCP/Azure)
# - University compute cluster
```

---

## 📋 Complete Colab Notebook Template

```python
# ============================================================================
# UTMIST AI² - Google Colab Training Notebook
# ============================================================================

# Cell 1: Mount Drive and Setup
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive
if not os.path.exists('UTMIST-AI2'):
    !git clone YOUR_REPO_URL
%cd UTMIST-AI2

# Cell 2: Install Dependencies
!pip install -q stable-baselines3[extra] sb3-contrib gymnasium pygame
!pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Cell 3: Verify GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# Cell 4: Keep Alive (Optional)
from IPython.display import HTML, display
display(HTML('''
    <script>
    function ClickConnect(){
        console.log("Keeping alive..."); 
        document.querySelector("colab-connect-button").click()
    }
    setInterval(ClickConnect, 60000)
    </script>
'''))

# Cell 5: Run Training
# For 50k test:
!python user/train_agent.py

# Cell 6: Monitor Progress (Run in parallel)
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time

while True:
    clear_output(wait=True)
    df = pd.read_csv('checkpoints/test_50k_colab/monitor.csv', skiprows=1)
    plt.plot(df.iloc[:, 0])
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Progress')
    plt.show()
    print(f"Episodes: {len(df)}, Latest reward: {df.iloc[-1, 0]:.2f}")
    time.sleep(30)

# Cell 7: Check Results
%cd checkpoints/test_50k_colab
!ls -lh
!tail monitor.csv
!tail evaluation_results.csv
```

---

## ⚠️ Important Colab-Specific Notes

### 1. **Checkpoint Frequency**
```python
# For Colab, save MORE frequently than normal
TRAIN_CONFIG_10M["self_play"]["save_freq"] = 50_000  # Every 50k instead of 100k
# Reason: If Colab disconnects, you lose less progress
```

### 2. **Monitor Drive Quota**
```python
# Check your Drive space
!df -h /content/drive

# Each checkpoint is ~50-80 MB
# 100 checkpoints = ~5-8 GB total
# Make sure you have enough space!
```

### 3. **Runtime Type**
```
Runtime → Change runtime type → Hardware accelerator → GPU
- Free: T4 (16GB) or K80 (12GB)
- Pro: T4, P100, or V100
- Pro+: A100 (40GB) - fastest!
```

### 4. **Session Management**
- Colab disconnects if:
  - 12 hours elapsed (free) / 24 hours (pro)
  - 90 minutes idle (no interaction)
  - Browser tab closed without keep-alive
- **Solution**: Keep tab open, use keep-alive script, save frequently

---

## 🎓 Pro Tips for Colab Training

1. **Start with 50k test** - Always test on Colab before committing to 10M
2. **Use Colab Pro for 10M** - Free tier can't finish in 12 hours
3. **Save to Drive, not local** - `/content` is wiped on disconnect
4. **Checkpoint every 50k** - More frequent = less lost progress
5. **Monitor Drive space** - Check before starting long runs
6. **Keep tab open** - Use keep-alive script
7. **Download final model** - Backup to local computer when done

---

## 📊 Colab vs Local Comparison

| Feature | Colab Free | Colab Pro | Local T4 | Local A6000 |
|---------|------------|-----------|----------|-------------|
| GPU | T4/K80 | T4/P100/V100 | T4 | A6000 |
| VRAM | 12-16GB | 16-40GB | 16GB | 48GB |
| Runtime Limit | 12 hrs | 24 hrs | ∞ | ∞ |
| 50k Test | ✅ 15 min | ✅ 15 min | ✅ 15 min | ✅ 10 min |
| 10M Training | ❌ Can't finish | ✅ ~10 hrs | ✅ ~10 hrs | ✅ ~6 hrs |
| Cost | Free | $10/mo | Electricity | Electricity |
| Storage | Drive 15GB | Drive 100GB+ | Local disk | Local disk |

---

## ✅ Colab Checklist

Before starting training:
- [ ] Drive mounted (`/content/drive/MyDrive`)
- [ ] Repository cloned to Drive
- [ ] Dependencies installed
- [ ] GPU detected (T4/P100/V100)
- [ ] Checkpoint path points to Drive
- [ ] Keep-alive script running (optional)
- [ ] Drive has 5-10 GB free space
- [ ] Using Colab Pro for 10M training (or will resume)

---

## 🆘 Troubleshooting Colab Issues

### "Drive not mounted"
```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

### "CUDA out of memory"
```python
# Reduce batch size in train_agent.py:
_SHARED_AGENT_CONFIG["batch_size"] = 64  # Down from 128
```

### "Session disconnected"
```python
# Resume training from latest checkpoint (see "Resume Training" section above)
```

### "Files disappearing"
```python
# Make sure you're saving to Drive, not /content
# Check path starts with: /content/drive/MyDrive
```

---

Need more help? Check:
- `T4_TRAINING_GUIDE.md` for general training info
- `QUICK_REFERENCE.md` for commands and tips
- Colab Help → View runtime logs for error details

Happy training in the cloud! ☁️🚀

