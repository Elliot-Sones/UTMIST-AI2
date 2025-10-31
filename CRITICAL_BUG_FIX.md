# 🐛 CRITICAL BUG FIX - Danger Zone Reward Inversion

**Date:** 2025-10-31  
**Status:** ✅ FIXED  
**Severity:** CRITICAL (caused passive behavior)

---

## 🚨 **THE BUG**

### **Symptoms:**
- Agent dealt ZERO damage in 10k steps
- Win rate vs ConstantAgent: 0%
- Win rate vs BasedAgent: 60% (by timeout, not combat)
- Agent jumped around avoiding ground combat
- Reward logs showed: `danger_zone_reward=0.067` (POSITIVE!)

### **Root Cause:**

The `danger_zone_reward` had **inverted sign logic**:

```python
# Function definition (returns NEGATIVE when high):
def danger_zone_reward(env, zone_penalty=1, zone_height=4.2):
    reward = -zone_penalty if player.body.position.y >= zone_height else 0.0
    return reward * env.dt  # Returns: -1.0 * dt when player is high

# BROKEN configuration:
'danger_zone_reward': RewTerm(
    func=danger_zone_reward, 
    weight=-2.0,  # ❌ NEGATIVE WEIGHT
    params={'zone_penalty': 1, 'zone_height': 4.2}
),

# Math when player jumps high:
#   Function returns: -1.0 * dt = -0.033
#   Multiply by weight: -0.033 * (-2.0) = +0.067
#   Result: REWARDS jumping high! ❌
```

### **Agent Learned:**
- "Jump high = +0.067 reward"
- "Stay on ground = 0 reward"
- "Attack opponent = unknown (never tried)"
- **Strategy:** Jump around, avoid ground, win by timeout

---

## ✅ **THE FIX**

Changed weight from **negative to positive**:

```python
# FIXED configuration:
'danger_zone_reward': RewTerm(
    func=danger_zone_reward, 
    weight=2.0,  # ✅ POSITIVE WEIGHT (function already returns negative)
    params={'zone_penalty': 1, 'zone_height': 4.2}
),

# Math when player jumps high (FIXED):
#   Function returns: -1.0 * dt = -0.033
#   Multiply by weight: -0.033 * (+2.0) = -0.067
#   Result: PENALIZES jumping high! ✅
```

---

## 📊 **EXPECTED BEHAVIOR AFTER FIX**

### **Before Fix:**
```
Step 5,000:
  Reward breakdown: danger_zone_reward=0.067  (positive)
  Damage dealt: 0.0
  Agent behavior: Jumps around, avoids combat
  Win strategy: Timeout
```

### **After Fix:**
```
Step 5,000:
  Reward breakdown: danger_zone_reward=-0.067  (negative when high)
                     damage_interaction_reward=12.5  (appears!)
  Damage dealt: 50-200
  Agent behavior: Stays on ground, attacks opponent
  Win strategy: Combat damage
```

---

## 🎯 **SUCCESS CRITERIA (After Fix)**

Training should now show:

| Metric | Target | Previous (Broken) | After Fix |
|--------|--------|-------------------|-----------|
| Damage dealt (5k steps) | 50+ | 0.0 ❌ | 50-200 ✅ |
| Win rate vs ConstantAgent | 60%+ | 0% ❌ | 60%+ ✅ |
| danger_zone_reward sign | Negative | Positive ❌ | Negative ✅ |
| damage_interaction_reward | Appears | Never ❌ | Appears ✅ |

---

## 📝 **FILES CHANGED**

1. **user/train_agent.py** (Line 1614)
   - Changed: `weight=-2.0`
   - To: `weight=2.0`

---

## 🔍 **HOW WE CAUGHT IT**

The **passive behavior detection system** worked perfectly:

```python
# In TrainingMonitorCallback._run_sanity_checks():
if self.total_damage_dealt == 0 and self.num_timesteps > 5000:
    🚨 PASSIVE BEHAVIOR: Agent has dealt ZERO damage in 5k+ steps!
    → STOP TRAINING and fix reward function
```

This saved ~40k steps of wasted training!

---

## 💡 **LESSONS LEARNED**

### **Design Flaw:**
When reward functions return **negative values internally**, using **negative weights** creates **double negatives** that invert the reward.

### **Best Practice:**
Either:
1. Functions return positive values → use negative weights for penalties
2. Functions return negative values → use positive weights for penalties

**Be consistent!**

### **Testing Improvement:**
Add unit test that validates reward signs:
```python
def test_reward_signs():
    # When player is high (bad behavior)
    env = create_test_env(player_height=5.0)  # Above danger zone
    reward = compute_danger_zone_reward(env)
    assert reward < 0, "Should penalize being high"
```

---

## 🚀 **NEXT STEPS**

1. ✅ Fix applied to `user/train_agent.py`
2. ⏭️ Upload fixed file to Colab
3. ⏭️ Restart training: `python user/train_agent.py`
4. ⏭️ Verify at step 5k: damage > 0 and `danger_zone_reward` is negative

---

## 📞 **VERIFICATION CHECKLIST**

After restarting training, verify:

- [ ] Step 250-500: `danger_zone_reward` shows **negative** values (e.g., -0.067)
- [ ] Step 1000-2000: `damage_interaction_reward` appears in logs
- [ ] Step 5000: Damage dealt > 50
- [ ] Step 5000: No "PASSIVE BEHAVIOR" alert
- [ ] Step 10000: Damage dealt > 100
- [ ] Step 50000: Win rate vs ConstantAgent > 80%

---

**Status:** ✅ Fix implemented, ready for testing  
**Confidence:** 95% (reward function logic is now correct)  
**Expected training time:** 15 minutes to validate fix

