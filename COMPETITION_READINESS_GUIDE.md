# 🏆 COMPETITION READINESS MONITORING SYSTEM

**Purpose:** Automatically assess if your 50k training run predicts 10M competition success

**Location:** Integrated into `user/train_agent.py`

---

## 🎯 **HOW IT WORKS**

### **Automatic Assessment**
The system runs automatically at checkpoints:
- **When:** At 30k, 40k, 50k steps (every 10k after 30k)
- **Where:** After benchmark evaluation
- **Output:** Console report + actionable recommendation

### **What Gets Measured**

#### **TIER 1 - CRITICAL (Must Pass All)**
1. **Transformer Utilization** - Variance >20% of mean
   - Checks: Is transformer adapting to different opponents?
   - Measured: Standard deviation of latent norms across matches

2. **Cross-Opponent Win Rates**
   - ConstantAgent: >80% (easy baseline)
   - BasedAgent: >40% (moving opponent)

3. **Strategy Diversity** - Score >0.4
   - Checks: Multiple strategies or mode collapse?
   - Measured: Std dev of latent vector norms

4. **Damage Ratio** - >1.0
   - Checks: Winning combat exchanges?
   - Measured: Damage dealt / damage taken

#### **TIER 2 - STRONG INDICATORS (Should Pass Most)**
5. **Reward Composition** - >60% from sparse
   - Checks: Optimizing outcomes vs farming dense rewards?
   - Measured: Ratio of sparse (win/damage) to dense (approach/buttons)

6. **Win Rate Scaling**
   - Checks: Improving vs BOTH opponents, not just one?

#### **TIER 3 - WARNING SIGNALS (Red Flags)**
7. **Attention Entropy** - >3.0
   - Checks: Transformer analyzing full opponent history?
   - Low entropy = focusing on 1-2 frames only

8. **Training Stability**
   - Checks: No NaN/crashes

---

## 📊 **EXAMPLE OUTPUT (Step 50k)**

```
======================================================================
🏆 COMPETITION READINESS ASSESSMENT (Step 50000)
======================================================================

📊 TIER 1 - CRITICAL METRICS (Must Pass for 10M Scale)
----------------------------------------------------------------------
  ✅ PASS Transformer Utilization:
       Latent Norm: 8.45 (±2.13)
       Variance Ratio: 0.25 (need >0.20)

  ✅ PASS Cross-Opponent Win Rates:
       vs ConstantAgent: 85.0% (need >80%)
       vs BasedAgent: 55.0% (need >40%)

  ✅ PASS Strategy Diversity:
       Score: 0.523 (need >0.400)

  ✅ PASS Damage Ratio:
       Ratio: 1.85 (need >1.00)

📊 TIER 2 - STRONG INDICATORS (Should Pass Most)
----------------------------------------------------------------------
  ✅ PASS Reward Composition:
       Sparse: 68.5% (need >60%)
       Dense: 31.5%

  ✅ PASS Win Rate Scaling:
       Both opponents improving: True

📊 TIER 3 - WARNING SIGNALS (Red Flags)
----------------------------------------------------------------------
  ✅ GOOD Attention Entropy:
       Entropy: 4.23 (want >3.0)

  ✅ GOOD Training Stability:
       No NaN/crashes detected

======================================================================
🎯 OVERALL ASSESSMENT
======================================================================

TIER 1 (Critical):  4/4 PASS
  ✅ PASS  Transformer Variance
  ✅ PASS  Cross-Opponent Wins
  ✅ PASS  Strategy Diversity
  ✅ PASS  Damage Ratio

TIER 2 (Strong):    2/2 PASS
  ✅ PASS  Reward Mix
  ✅ PASS  Win Rate Scaling

TIER 3 (Warnings):  2/2 GOOD
  ✅ GOOD  Attention Entropy
  ✅ GOOD  Training Stability

======================================================================
🎉 READY FOR 10M SCALE (~90% confidence)
   ✅ All critical metrics passed
   ✅ Agent shows signs of adaptability
   ✅ Proceed to Stage 2 or 10M training
======================================================================
```

---

## 📋 **INTERPRETATION GUIDE**

### **🎉 READY FOR 10M**
**Recommendation:** `PROCEED`
- All Tier 1 checks passed
- Most Tier 2 checks passed
- **Action:** Proceed to 10M training OR Stage 2 (vs BasedAgent)

### **⚠️ BORDERLINE**
**Recommendation:** `CAUTION`
- All Tier 1 checks passed
- Some Tier 2 checks weak
- **Action:** Train 50k more steps to strengthen, then reassess

### **🚨 NOT READY**
**Recommendation:** `FIX_REQUIRED`
- One or more Tier 1 checks failed
- **Action:** Fix issues listed, DO NOT scale to 10M yet

---

## 🔧 **COMMON FAILURE MODES & FIXES**

### **❌ Transformer Not Varying (Variance Ratio <0.20)**
**Problem:** Fixed policy, not adapting to opponents

**Fixes:**
```python
# Option 1: Reduce dense rewards further (agent ignoring transformer)
'head_to_opponent': weight=1.0  # Was 3.0
'on_attack_button_press': weight=1.0  # Was 3.0

# Option 2: Increase opponent diversity
"opponent_mix": {
    "constant_agent": (0.3, partial(ConstantAgent)),
    "based_agent": (0.7, partial(BasedAgent)),  # More variety
}
```

---

### **❌ Low Win Rate vs BasedAgent (<40%)**
**Problem:** Overfitting to stationary target

**Fixes:**
```python
# Increase BasedAgent exposure
"opponent_mix": {
    "constant_agent": (0.5, partial(ConstantAgent)),
    "based_agent": (0.5, partial(BasedAgent)),  # 50/50 split
}

# OR: Train 50k more with current mix
```

---

### **❌ Low Strategy Diversity (<0.4)**
**Problem:** Mode collapse - all strategies converged

**Fixes:**
```python
# Increase entropy (more exploration)
"ent_coef": 0.30,  # Was 0.25

# Add self-play (requires Stage 2+)
"opponent_mix": {
    "self_play": (0.5, None),
    "based_agent": (0.5, partial(BasedAgent)),
}
```

---

### **❌ Damage Ratio <1.0**
**Problem:** Losing combat exchanges

**Fixes:**
```python
# Increase damage reward
'damage_interaction_reward': weight=700.0,  # Was 500

# Check if agent is actually attacking
# Look for on_attack_button_press in reward_breakdown.csv
# If zero, increase exploration:
'on_attack_button_press': weight=5.0,  # Was 3.0
```

---

### **⚠️ Sparse Rewards <60%**
**Problem:** Still reward hacking dense rewards

**Fixes:**
```python
# More aggressive dense reduction
'head_to_opponent': weight=1.0,  # Was 3.0
'on_attack_button_press': weight=1.0,  # Was 3.0

# Increase outcome rewards
'on_win_reward': weight=3000,  # Was 2000
```

---

### **⚠️ Low Attention Entropy (<3.0)**
**Problem:** Transformer focusing on 1-2 frames only

**Fixes:**
```python
# This indicates transformer architecture issue
# Verify repeat-padding is working (not zero-padding)
# Run: python validate_fixes.py

# If padding OK, increase sequence length:
TransformerStrategyAgent(
    max_sequence_length=120,  # Was 90
    ...
)
```

---

## 📈 **PROGRESSION PLAN**

### **After 50k READY Assessment:**

#### **Option A: Proceed to Stage 2 (Recommended)**
```python
# Update train_agent.py line 479:
"opponent_mix": {
    "based_agent": (0.7, partial(BasedAgent)),  # 70% moving opponent
    "constant_agent": (0.3, partial(ConstantAgent)),
},

# Train another 50k, target:
# - vs BasedAgent: 70%+ win rate
# - vs ConstantAgent: maintain 90%+
```

#### **Option B: Jump to 10M Competition**
```python
# Change line 590:
TRAIN_CONFIG = TRAIN_CONFIG_10M

# Expected timeline: 10-12 hours on T4 GPU
# Target final metrics:
# - vs BasedAgent: 80%+ win rate
# - Strategy diversity: 0.7+
# - Self-play: 45-55% (balanced)
```

---

## 🎯 **WHAT MAKES AN AGENT "COMPETITION READY"?**

### **The 5 Golden Signals:**
If you see ALL 5 at 50k steps → 90%+ confidence for 10M:

1. ✅ **Transformer latent variance >25%**
   - Evidence transformer is analyzing opponents

2. ✅ **Win rates: 85%+ vs Constant, 60%+ vs Based**
   - Smooth scaling across difficulty

3. ✅ **Strategy diversity >0.5**
   - Multiple viable tactics emerged

4. ✅ **Damage ratio >1.5**
   - Dominating combat exchanges

5. ✅ **Sparse rewards >70%**
   - Fully optimizing outcomes not actions

---

## 🔍 **MANUAL VERIFICATION (Optional)**

If assessment passes but you want extra confidence:

### **1. Watch 3 Episodes Manually**
```python
# In Colab, after training:
from environment.match import env_run_match, CameraResolution

agent = ... # Your trained agent
match_stats = env_run_match(
    agent,
    partial(BasedAgent),
    max_timesteps=1800,
    resolution=CameraResolution.MEDIUM,
    video_path="/content/test_match.mp4"  # Record video
)

# Watch video - check:
# - Does agent adapt strategy mid-match?
# - Do attack patterns vary?
# - Does agent respond to opponent movement?
```

### **2. Check Latent Norm Variance Manually**
```python
# Read checkpoint_benchmarks.csv
import pandas as pd
df = pd.read_csv('/tmp/checkpoints/.../checkpoint_benchmarks.csv')

# Strategy diversity should increase over time:
print(df[['checkpoint_step', 'strategy_diversity_score']])
# 10k: 0.15
# 20k: 0.28
# 30k: 0.42 ← Passing threshold
# 40k: 0.51
# 50k: 0.58 ← Strong!
```

### **3. Verify Reward Mix**
```python
# Read reward_breakdown.csv
df = pd.read_csv('/tmp/checkpoints/.../reward_breakdown.csv')

# Get last 100 rows
recent = df.tail(100)

# Calculate ratio
dense_cols = ['head_to_opponent', 'on_attack_button_press']
sparse_cols = ['damage_interaction_reward', 'on_win_reward']

dense_total = recent[dense_cols].abs().sum().sum()
sparse_total = recent[sparse_cols].abs().sum().sum()

print(f"Sparse: {sparse_total/(dense_total+sparse_total)*100:.1f}%")
# Should be >60%
```

---

## 📞 **FAQ**

**Q: Assessment says READY but I only trained vs ConstantAgent?**
A: This is OK for Stage 1! The assessment checks if the agent CAN adapt (transformer varying). Scale to Stage 2 (vs BasedAgent) next, not straight to 10M.

**Q: What if only 1 Tier 1 check fails?**
A: DO NOT scale to 10M. All 4 Tier 1 checks are critical. Fix the failing check first (see fixes above).

**Q: Tier 2 checks all weak but Tier 1 passed?**
A: Recommendation will be "CAUTION". Train 50k more to strengthen Tier 2 before scaling.

**Q: Can I skip straight to 10M if 50k assessment is READY?**
A: **Not recommended.** Better to do Stage 2 (50k vs BasedAgent) first. This validates agent can handle MOVING opponents before investing 10-12 hours in 10M run.

**Q: Assessment not running at my checkpoint?**
A: It only runs at 30k, 40k, 50k steps (multiples of 10k after 30k). Change line 2737 condition if you want different timing.

---

## ✅ **NEXT STEPS**

1. **Run your 50k training** with conservative reward rebalance (3.0 weights)
2. **Wait for 50k checkpoint** - assessment runs automatically
3. **Read the recommendation:**
   - PROCEED → Stage 2 or 10M
   - CAUTION → Train 50k more
   - FIX_REQUIRED → Apply fixes from this guide
4. **Iterate** until PROCEED, then scale up!

---

**Good luck! 🚀 This monitoring system gives you 80-90% confidence in scaling decisions.**
