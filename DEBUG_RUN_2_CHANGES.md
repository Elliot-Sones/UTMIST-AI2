# 🔥 DEBUG RUN 2 - BOOSTED EXPLORATION

**Date:** 2024-10-31
**Goal:** Land at least 1 hit on opponent in 5k steps
**Changes:** 3x stronger attack exploration signal

---

## ✅ **CHANGES MADE**

### **1. Entropy Coefficient (Line 388)**
```python
# BEFORE:
"ent_coef": 0.15,  # Entropy coefficient (INCREASED for attack exploration)

# AFTER:
"ent_coef": 0.25,  # 🔥 BOOSTED: 0.15→0.25 for aggressive exploration (debug run 2)
```

**Impact:** +67% more random exploration
**Effect:** Agent will try more random action combinations, increasing chance of discovering successful attack timing

---

### **2. Attack Button Press Reward (Line 1755)**
```python
# BEFORE:
weight=5.0  # 🚀 INCREASED: Strong exploration signal to discover attacking

# AFTER:
weight=15.0  # 🚀🚀 BOOSTED: 5.0→15.0 (3x stronger!) for debug run 2
```

**Impact:** 3x stronger reward for pressing attack buttons (j, k)
**Effect:** Agent gets MUCH bigger reward for trying to attack, even if it doesn't land

---

## 📊 **EXPECTED RESULTS (5K Steps)**

### **Success Criteria:**
- ✅ `damage_interaction_reward` appears in logs (AT LEAST ONCE)
- ✅ `damage_dealt > 0` in episode_summary.csv (AT LEAST ONE EPISODE)
- ✅ Win rate vs ConstantAgent > 0% (if damage lands)

### **What to Watch For:**

**Good Signs (Agent Learning to Attack):**
```
--- Step XXXX ---
  Reward Breakdown: damage_interaction_reward=1.428, on_attack_button_press=0.500
  👆 THIS IS SUCCESS! damage_interaction_reward means HIT LANDED!
```

**Still Exploring (Not Landed Yet):**
```
--- Step XXXX ---
  Reward Breakdown: on_attack_button_press=0.500, head_to_opponent=0.684
  👆 Agent pressing buttons but not landing (keep waiting)
```

**Failure (Still Zero Damage After 5K):**
```
Damage Ratio: 0.00
vs ConstantAgent: 0.0% wins
👆 If this happens again, we need Option C (even MORE exploration or longer training)
```

---

## 🎯 **MONITORING CHECKLIST**

During training, watch for these milestones:

**Step 0-1000:**
- [ ] `on_attack_button_press` firing frequently (agent trying to attack)
- [ ] Entropy should be high (~-14.7 or lower in logs)

**Step 1000-3000:**
- [ ] Agent picking up weapons (Hammer, Spear in logs)
- [ ] `head_to_opponent` positive (agent approaching)
- [ ] **HOPE FOR:** First `damage_interaction_reward` appearance!

**Step 3000-5000:**
- [ ] Consistent attack button presses
- [ ] **CRITICAL:** At least 1 successful hit before 5000 steps

**Final Benchmark (Step 5000):**
- [ ] Check: `Avg Damage Ratio > 0.00`
- [ ] Check: Episode summary CSV has non-zero damage in ANY row
- [ ] Check: No NaN/explosion errors

---

## 🚀 **HOW TO RUN (Google Colab T4)**

1. **Upload the updated file to Colab:**
   - Replace the existing `user/train_agent.py` with this updated version
   - Or re-upload the entire project folder

2. **Run training:**
   ```python
   !python user/train_agent.py
   ```

3. **Expected runtime:**
   - 5k steps = ~3-4 minutes on T4 GPU
   - Watch console for `damage_interaction_reward`

4. **After completion, check damage:**
   ```bash
   # Look for any non-zero damage:
   !tail -20 /tmp/checkpoints/debug_5k_attack_test/episode_summary.csv
   !grep "damage_interaction_reward" /tmp/checkpoints/debug_5k_attack_test/*.log
   ```

---

## 📈 **WHAT HAPPENS AFTER DEBUG RUN 2?**

### **If Damage > 0 (SUCCESS!):**
✅ Proceed to **Curriculum Stage 1** (50k steps)
```python
# Change line 590:
TRAIN_CONFIG = TRAIN_CONFIG_CURRICULUM
```
- Expected: 90%+ win rate vs ConstantAgent by 50k
- Time: ~20 minutes on T4

### **If Still Zero Damage:**
⚠️ Need **Option C: Extreme Exploration Boost**
```python
# Line 1755: Increase to 25.0 (5x original)
weight=25.0

# Line 388: Max out entropy
"ent_coef": 0.30

# AND/OR: Switch to longer debug (10k steps instead of 5k)
"timesteps": 10_000
```

---

## 🔧 **ROLLBACK (If Needed)**

If you need to revert these changes:

```python
# Line 388: Revert entropy
"ent_coef": 0.15,

# Line 1755: Revert attack weight
weight=5.0
```

---

## 📞 **TROUBLESHOOTING**

**Q: Training crashes with OOM error?**
A: Reduce batch size (line 385): `"batch_size": 32,`

**Q: Agent attacking TOO much (spamming)?**
A: Good! This means exploration is working. The `holding_more_than_3_keys` penalty (-0.05) will balance it.

**Q: Damage_interaction_reward appears but ratio still 0.00?**
A: Check logs carefully - even 1 point of damage counts as success! The ratio might be 0.01 (rounds to 0.00 in display).

**Q: Training takes longer than 4 minutes?**
A: Normal - benchmarking takes extra time. Total runtime could be 5-6 minutes including evaluations.

---

## ✅ **FILES MODIFIED**

- [user/train_agent.py](user/train_agent.py) - Lines 388, 1755

---

**Good luck! 🎯 With 3x stronger exploration signal, the agent should discover attacking within 5k steps!**
