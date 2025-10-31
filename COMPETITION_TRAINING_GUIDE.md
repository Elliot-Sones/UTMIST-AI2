# 🏆 COMPETITION TRAINING GUIDE - UTMIST AI² Agent

**Goal:** Train an unbeatable agent that adapts to ANY opponent strategy for competition

**Hardware:** Google Colab T4 GPU (16GB VRAM)

**Total Time:** ~12 hours (including all curriculum stages)

---

## 🚨 **CRITICAL BUGS FIXED** (2024-10-31)

### **Bug #1: Zero-Padding Transformer (Lines 1097-1105)**
**Problem:** Agent received meaningless ZERO-PADDED opponent observations for first 5-10 frames
- Transformer learned to ignore early game behavior
- Strategy latent was invalid for 16% of episode

**Fix:** Changed to REPEAT-PADDING
- Repeats available opponent frames instead of zeros
- Preserves actual opponent information from frame 1
- Transformer gets meaningful data immediately

---

### **Bug #2: Learning Frequency Too Low (Lines 384-388)**
**Problem:** Only 24 PPO updates for entire 50k curriculum run
- n_steps=2048 meant policy updated every 2048 steps
- Too infrequent for combat game (needs fast reward feedback)

**Fix:** Reduced to n_steps=512
- 98 PPO updates for 50k run (4x more frequent!)
- Policy updates every ~17 seconds (512 steps / 30 FPS)
- Faster learning from damage/knockout rewards

---

### **Bug #3: Passive Behavior - Insufficient Exploration (Lines 1695-1702)**
**Problem:** Agent never discovered attacking was rewarding
- Entropy 0.10 wasn't enough for discrete action space
- `head_to_opponent` (5.0) fired every frame
- `damage_interaction_reward` (150) only fired on HIT (rare initially)
- Agent learned: "approaching is rewarding, attacking is neutral"

**Fix:** Boosted exploration signals
- Increased entropy to 0.15 (more random exploration)
- Increased `on_attack_button_press` to 5.0 (strong exploration bonus)
- Increased `damage_interaction_reward` to 200.0 (make damage supremely valuable)
- Increased `head_to_opponent` to 10.0 (must close distance to attack)

---

## 📋 **TRAINING PIPELINE** (5 Stages)

### **🔥 STAGE 0: DEBUG RUN (2-3 minutes)**
**Config:** `TRAIN_CONFIG_DEBUG` (currently ACTIVE in code)
- **Timesteps:** 5,000
- **Opponent:** 100% ConstantAgent (stationary target)
- **Goal:** Verify agent ATTACKS and DEALS DAMAGE

**Success Criteria:**
- ✅ `damage_dealt > 0` (agent must land at least 1 hit)
- ✅ `attack_button_presses > 0` (agent pressing j or k)
- ✅ No NaN/explosion errors
- ✅ Reward increasing over time

**How to Run:**
```python
# Line 590 already set to:
TRAIN_CONFIG = TRAIN_CONFIG_DEBUG
```

**What to Check:**
```bash
# After 5k steps, check:
tail -20 checkpoints/debug_5k_attack_test/reward_breakdown.csv
# Look for: damage_interaction_reward > 0, on_attack_button_press > 0

tail -10 checkpoints/debug_5k_attack_test/episode_summary.csv
# Look for: damage_ratio column > 0 (agent dealt damage)
```

**If Debug Fails:**
- Agent dealt ZERO damage → Increase `on_attack_button_press` weight to 10.0
- NaN errors → Reduce learning rate to 1e-4
- Reward stuck at 0 → Check monitoring logs for "PASSIVE BEHAVIOR" alert

---

### **🎯 STAGE 1: CURRICULUM BASIC COMBAT (15 minutes)**
**Config:** `TRAIN_CONFIG_CURRICULUM`
- **Timesteps:** 50,000
- **Opponent:** 100% ConstantAgent
- **Goal:** Beat stationary target reliably (validate reward function)

**Success Criteria:**
- ✅ 90%+ win rate vs ConstantAgent
- ✅ Damage ratio > 3.0 (dealing 3x more damage than taking)
- ✅ Consistent attacking behavior (not passive)

**How to Run:**
```python
# After debug success, change line 590:
TRAIN_CONFIG = TRAIN_CONFIG_CURRICULUM
```

**Monitoring:**
```bash
# Check win rate at 10k, 20k, 30k, 40k, 50k steps:
grep "Win Rate" checkpoints/curriculum_basic_combat/*.log

# Check damage ratio trend:
tail -50 checkpoints/curriculum_basic_combat/episode_summary.csv
```

---

### **🎯 STAGE 2: CURRICULUM SCRIPTED OPPONENTS (15 minutes)**
**Config:** `TRAIN_CONFIG_CURRICULUM_STAGE2`
- **Timesteps:** 50,000
- **Opponent:** 70% BasedAgent, 30% ConstantAgent
- **Goal:** Beat heuristic AI that chases/attacks/dodges
- **Load From:** Stage 1 best checkpoint

**Success Criteria:**
- ✅ 60%+ win rate vs BasedAgent
- ✅ Agent adapts to moving opponent
- ✅ No regression on ConstantAgent (maintain 90%+ win rate)

**How to Run:**
```python
# 1. Find best Stage 1 checkpoint:
ls -lt checkpoints/curriculum_basic_combat/*.zip | head -1
# Example output: rl_model_50000_steps.zip

# 2. Update config line 498:
"load_path": "checkpoints/curriculum_basic_combat/rl_model_50000_steps.zip"

# 3. Change line 590:
TRAIN_CONFIG = TRAIN_CONFIG_CURRICULUM_STAGE2
```

---

### **🎯 STAGE 3: SELF-PLAY VALIDATION (15 minutes)**
**Config:** `TRAIN_CONFIG_TEST`
- **Timesteps:** 50,000
- **Opponent:** 70% self-play, 20% BasedAgent, 10% ConstantAgent
- **Goal:** Validate self-play stability (agent vs its past selves)
- **Load From:** Stage 2 best checkpoint

**Success Criteria:**
- ✅ 40%+ win rate (challenging due to diverse opponents)
- ✅ Strategy diversity score > 0.5
- ✅ Win rate stays stable (not collapsing)

**How to Run:**
```python
# 1. Update load_path with Stage 2 checkpoint (line 533)
# 2. Change line 590:
TRAIN_CONFIG = TRAIN_CONFIG_TEST
```

---

### **🏆 STAGE 4: COMPETITION TRAINING (10-12 hours)**
**Config:** `TRAIN_CONFIG_10M`
- **Timesteps:** 10,000,000
- **Opponent:** 70% self-play, 20% BasedAgent, 5% Clockwork, 5% Random
- **Goal:** Create competition-grade agent
- **Load From:** Stage 3 best checkpoint

**Success Criteria:**
- ✅ Beats BasedAgent 80%+ win rate
- ✅ Beats ConstantAgent 95%+ win rate
- ✅ Diverse strategy repertoire (200 self-play checkpoints)
- ✅ Adapts to unseen opponent patterns

**How to Run:**
```python
# 1. Update load_path with Stage 3 checkpoint (line 497)
# 2. Change line 590:
TRAIN_CONFIG = TRAIN_CONFIG_10M

# 3. Upload to Google Colab and run for 10-12 hours
```

**Competition Agent Location:**
```
checkpoints/competition_10M_final/rl_model_10000000_steps.zip
checkpoints/competition_10M_final/rl_model_10000000_steps_transformer_encoder.pth
```

---

## 📊 **MONITORING CRITICAL METRICS**

### **Every Training Run:**
Watch for these alerts in console output:

#### ✅ **Good Signs:**
```
✓ Sanity checks passed
Win Rate: 60.0% (3/5 matches)
Damage Ratio: 2.45
Transformer: Latent Norm=12.456 (±2.134)
PPO: Policy Loss=0.0234, Value Loss=1.234, Explained Var=0.856
```

#### 🚨 **Critical Issues:**
```
🚨 PASSIVE BEHAVIOR: Agent has dealt ZERO damage in 5k+ steps!
⚠️ ALERT: Gradient explosion detected (loss=154.23)
🚨 CRITICAL: NaN detected in loss
⚠️ LOW DAMAGE OUTPUT: 0.23 damage per 1k steps
NO IMPROVEMENT detected (agent not learning effectively)
```

### **CSV Monitoring:**
```bash
# Real-time reward tracking:
watch -n 5 'tail -5 checkpoints/*/reward_breakdown.csv'

# Episode performance:
watch -n 10 'tail -10 checkpoints/*/episode_summary.csv'

# Checkpoint benchmarks:
cat checkpoints/*/checkpoint_benchmarks.csv
```

---

## 🎯 **SUCCESS METRICS BY STAGE**

| Stage | Timesteps | Time | Win Rate Goal | Damage Ratio | Key Behavior |
|-------|-----------|------|---------------|--------------|--------------|
| 0: Debug | 5k | 2-3 min | N/A | > 0 | Attacks at all |
| 1: Basic | 50k | 15 min | 90% vs Constant | > 3.0 | Reliable attacking |
| 2: Scripted | 50k | 15 min | 60% vs Based | > 1.5 | Adapts to movement |
| 3: Self-play | 50k | 15 min | 40% vs diverse | > 1.0 | Stable vs past selves |
| 4: Competition | 10M | 10-12 hrs | 80% vs Based | > 2.0 | Beats any strategy |

---

## 🔧 **TROUBLESHOOTING**

### **Problem: Agent Still Passive (Zero Damage)**
**Symptoms:** `damage_dealt = 0` after 5k steps

**Solutions:**
1. Increase exploration bonus (line 1698):
   ```python
   weight=10.0  # Was 5.0
   ```

2. Check action distribution in logs - are attack buttons (indices 7, 8) being pressed?

3. Reduce penalty for button mashing (line 1706):
   ```python
   weight=-0.01  # Was -0.05 (too harsh)
   ```

---

### **Problem: NaN / Gradient Explosion**
**Symptoms:** Loss becomes NaN or > 1000

**Solutions:**
1. Reduce learning rate (line 389):
   ```python
   "learning_rate": 1e-4,  # Was 2.5e-4
   ```

2. Add gradient clipping (line 963):
   ```python
   max_grad_norm=0.5,  # Add this parameter to RecurrentPPO
   ```

---

### **Problem: Agent Not Improving**
**Symptoms:** Win rate stays flat after 20k+ steps

**Solutions:**
1. Check if reward signal is stuck (all values similar)
   - Increase weight on `damage_interaction_reward`

2. Check if opponent is too hard
   - Start with easier opponent (ConstantAgent)

3. Increase learning frequency (line 384):
   ```python
   "n_steps": 256,  # Was 512 (even more frequent updates)
   ```

---

### **Problem: Self-Play Collapse**
**Symptoms:** Win rate drops to 0% during self-play

**Solutions:**
1. Reduce self-play ratio (line 412):
   ```python
   "self_play": (5.0, None),  # Was 7.0 (too much self-play)
   "based_agent": (3.0, partial(BasedAgent)),  # Increase scripted
   ```

2. Keep more diverse checkpoints (line 408):
   ```python
   "max_saved": 300,  # Was 200 (more opponent diversity)
   ```

---

## 📈 **EXPECTED LEARNING CURVES**

### **Stage 1 (Basic Combat):**
- **Steps 0-10k:** Random exploration, damage_dealt slowly increases
- **Steps 10k-30k:** Consistent attacking, win rate climbs to 60-80%
- **Steps 30k-50k:** Refined strategy, win rate stabilizes at 90%+

### **Stage 4 (Competition 10M):**
- **Steps 0-1M:** Learning fundamentals, 40-60% win rate vs BasedAgent
- **Steps 1M-5M:** Strategy refinement, 60-75% win rate
- **Steps 5M-10M:** Mastery, 75-85% win rate, diverse counter-strategies

---

## 🏁 **FINAL COMPETITION AGENT**

After 10M training, your agent will have:

✅ **200 checkpoints** spanning 10M timesteps (every 50k)
✅ **Strategy diversity** from self-play vs 200 past versions
✅ **Robustness** from training vs scripted + random opponents
✅ **Transformer encoder** that recognizes opponent patterns
✅ **Adaptive policy** that counter-strategies in real-time

**Competition Deployment:**
```python
# Load best checkpoint:
from train_agent import TransformerStrategyAgent

agent = TransformerStrategyAgent(
    file_path="checkpoints/competition_10M_final/rl_model_10000000_steps.zip",
    latent_dim=256,
    num_heads=8,
    num_layers=6,
    sequence_length=90
)

# Agent is ready for competition matches!
```

---

## 💾 **BACKUP STRATEGY**

**CRITICAL:** Checkpoints are saved to Google Drive (if using Colab)
- Path: `/content/drive/MyDrive/UTMIST-AI2-Checkpoints/`
- Checkpoints persist even if Colab disconnects
- Download best checkpoints locally after each stage

---

## 📞 **NEXT STEPS**

1. **Run DEBUG now** (2-3 minutes) - verify all fixes work
2. **Check for "damage > 0"** in logs
3. **If successful** → proceed to Stage 1 (50k curriculum)
4. **After each stage** → download checkpoints + CSV logs
5. **Stage 4** → Upload to Colab, run 10M overnight

**Good luck in the competition! 🏆**
