# 🎯 MINIMAL REWARD FUNCTION - Quick Start Guide

## Problem Identified
Agent was **NOT attacking** during training - dealing 0 damage because:
- Complex reward function with 7+ dense rewards
- Agent learned to optimize movement shaping (head_to_opponent, closing_distance)
- Never discovered that pressing attack buttons gives the biggest reward (damage)
- Classic "reward hacking" - getting rewarded for easy behaviors without core task

---

## Solution Implemented

### 1. **Minimal Reward Function** (`gen_reward_manager_MINIMAL`)
**Location**: [user/train_agent.py:1935-1975](user/train_agent.py:1935-1975)

**What it does:**
- **ONLY** rewards damage dealt/taken (no movement, no button press shaping)
- Forces agent to randomly explore until it discovers attacking
- Once it attacks → gets reward → reinforces attacking behavior

**Rewards (simplified to 3 terms):**
```python
damage_interaction_reward: 100.0 weight  # +damage dealt, -damage taken
on_win_reward: 50.0 weight               # Small bonus for winning
on_knockout_reward: 20.0 weight          # Small bonus for eliminating stock
```

**What was REMOVED:**
- ❌ `head_to_opponent` (10.0) - moving toward opponent
- ❌ `closing_distance_reward` (6.0) - getting closer
- ❌ `edge_pressure_reward` (4.0) - pushing to edge
- ❌ `on_attack_button_press` (15.0) - pressing attack buttons
- ❌ `danger_zone_reward` (2.0) - staying on stage
- ❌ `holding_more_than_3_keys` (-0.05) - button mashing penalty
- ❌ All curriculum annealing complexity

---

### 2. **Boosted Exploration** (Entropy Coefficient)
**Location**: [user/train_agent.py:388](user/train_agent.py:388)

**Changed**: `ent_coef: 0.25 → 0.5` (100% increase!)

**Why**: With minimal reward, agent must randomly explore action space until it accidentally presses attack button and gets reward. High entropy = more random exploration.

---

### 3. **Updated DEBUG Config** (50k steps)
**Location**: [user/train_agent.py:449-475](user/train_agent.py:449-475)

**Key changes:**
- Uses `gen_reward_manager_MINIMAL` (set in `_SHARED_REWARD_CONFIG`)
- 100% ConstantAgent opponent (easiest to hit - stationary target)
- 50k timesteps (~15 minutes on T4 GPU)
- Evaluates every 5k steps to track damage progression

**Success criteria:**
- ✅ `damage_dealt > 0` after 10k steps (discovered attacking)
- ✅ `damage_dealt > 50` after 50k steps (attacks semi-consistently)
- ✅ `win_rate > 80%` vs ConstantAgent (baseline competence)

---

## How to Use

### Step 1: Run Minimal Reward Training
```bash
python user/train_agent.py
```

**Current config**: `TRAIN_CONFIG = TRAIN_CONFIG_DEBUG` (line 602)

**What to watch:**
1. Monitor `reward_breakdown.csv` in `checkpoints/debug_minimal_reward/`
2. Check `damage_dealt` in evaluation logs every 5k steps
3. Look for: "damage_interaction_reward" value increasing over time

**Expected behavior:**
- First 5-10k steps: Random exploration, ~0 damage
- Around 10-20k steps: Agent discovers attacking, damage spikes
- By 50k steps: Agent attacks consistently, damage > 50

---

### Step 2: Validate Success

**After 50k steps**, check final checkpoint benchmark:

```
🎯 CHECKPOINT BENCHMARK (Step 50000)
  vs ConstantAgent:    XX% wins    # Should be >80%
  Avg Damage Ratio:    X.XX        # Should be >1.0 (dealing more than taking)
```

**If successful** (damage > 50, win rate > 80%):
- Agent learned to attack! ✅
- Proceed to Step 3 (add back complexity)

**If failed** (damage < 10, win rate < 50%):
- Agent still not attacking ❌
- Try: Increase training to 100k steps OR increase ent_coef to 0.7

---

### Step 3: Switch to Complex Reward (Curriculum)

**Edit** [user/train_agent.py:602-605](user/train_agent.py:602-605):

```python
# Comment out minimal:
# TRAIN_CONFIG = TRAIN_CONFIG_DEBUG

# Activate curriculum with complex rewards:
TRAIN_CONFIG = TRAIN_CONFIG_CURRICULUM
```

**Also update reward factory** [user/train_agent.py:396-398](user/train_agent.py:396-398):

```python
_SHARED_REWARD_CONFIG = {
    # "factory": gen_reward_manager_MINIMAL,  # Comment out
    "factory": None,  # Uses gen_reward_manager() by default (complex rewards)
}
```

**Then run:**
```bash
python user/train_agent.py
```

This adds back:
- Movement shaping (head_to_opponent, closing_distance)
- Attack button press rewards
- Curriculum annealing (gradually reduce dense rewards)
- More complex opponent mix (BasedAgent, RandomAgent)

---

## Key Files Modified

1. **[user/train_agent.py:1935-1975](user/train_agent.py:1935-1975)** - New `gen_reward_manager_MINIMAL()` function
2. **[user/train_agent.py:396](user/train_agent.py:396)** - Config uses minimal reward by default
3. **[user/train_agent.py:388](user/train_agent.py:388)** - Boosted `ent_coef` to 0.5
4. **[user/train_agent.py:449-475](user/train_agent.py:449-475)** - Updated DEBUG config (50k steps, clear success criteria)
5. **[user/train_agent.py:602](user/train_agent.py:602)** - Active config switched to DEBUG

---

## Comparison: Before vs After

### Before (Complex Reward - 7 terms)
```python
damage_interaction_reward: 500.0    # Main reward
head_to_opponent: 10.0              # Movement shaping
closing_distance: 6.0               # Getting closer
edge_pressure: 4.0                  # Pushing opponent
on_attack_button_press: 15.0        # Button press reward
danger_zone: 2.0                    # Stay on stage
holding_more_than_3_keys: -0.05     # Penalty

+ 5 signal rewards (win, knockout, combo, equip, drop)
```

**Problem**: Agent optimized movement (10+6+4 = 20/frame) without attacking (0 damage = 0 reward)

---

### After (Minimal Reward - 1 term)
```python
damage_interaction_reward: 100.0    # ONLY reward

+ 2 signal rewards (win, knockout)
```

**Solution**: Agent can ONLY get reward by dealing damage → forces attacking discovery

---

## FAQ

**Q: Why not just increase `on_attack_button_press` weight?**
A: Already tried (weight=15.0). Agent learned to spam attack in place without hitting opponent. Movement shaping (head_to_opponent=10.0) let it get reward without attacking.

**Q: Won't minimal reward make agent dumb?**
A: Short-term yes - agent will be passive until it discovers attacking. But once it learns attacking → gets damage → reinforcement loop. Then we add back complexity.

**Q: Why 50k steps instead of 5k?**
A: Random exploration to discover attack buttons can take 10-20k steps. 50k gives buffer for agent to not just discover, but consistently attack.

**Q: What if agent still doesn't attack after 50k?**
A: Two options:
1. Increase to 100k steps (more exploration time)
2. Increase `ent_coef` to 0.7 or 0.8 (even more random)
3. Add small `on_attack_button_press` reward (1.0 weight) as hint

**Q: When do I know it's safe to add back complex rewards?**
A: When `damage_dealt > 50` after 50k steps AND `win_rate > 80%` vs ConstantAgent. This proves agent learned attacking is core behavior.

---

## Expected Training Timeline

```
Step 0-5k:    Random exploration, 0 damage
              [Agent walks randomly, sometimes near opponent]

Step 5-15k:   First attack discovery!
              [Agent accidentally presses j/k near opponent]
              [Gets +damage reward for first time]
              [Starts pressing attack more often]

Step 15-30k:  Consistent attacking emerges
              [Agent learns: near opponent + attack = reward]
              [Damage increases from 1-5 to 20-40]

Step 30-50k:  Refinement & reliability
              [Agent reliably attacks when near opponent]
              [Win rate climbs to 80-90% vs ConstantAgent]
              [Damage > 50, ready for complexity]

Step 50k+:    Switch to CURRICULUM with complex rewards
              [Add movement shaping, strategy, self-play]
```

---

## Monitoring Commands

**Watch training progress:**
```bash
# View reward breakdown CSV
cat checkpoints/debug_minimal_reward/reward_breakdown.csv | tail -20

# View evaluation results
cat checkpoints/debug_minimal_reward/checkpoint_benchmarks.csv

# View episode summaries
cat checkpoints/debug_minimal_reward/episode_summary.csv | tail -20
```

**Key metrics to watch:**
- `damage_interaction_reward` value (should increase from 0 → positive)
- `vs_constant_winrate` (should reach 80%+)
- `avg_damage_ratio` (should reach 1.0+)

---

## Next Steps (After Minimal Reward Success)

1. ✅ **Validate attacking** (50k minimal reward) ← YOU ARE HERE
2. 🎯 **Add curriculum** (50k with complex rewards on ConstantAgent)
3. 🎯 **Scale to BasedAgent** (50k with BasedAgent opponent)
4. 🎯 **Add self-play** (50k with 70% self-play mix)
5. 🏆 **Competition training** (10M with full opponent diversity)

Each step adds ONE new complexity. If agent regresses (stops attacking), go back one step.

---

## Troubleshooting

**Issue**: Agent still does 0 damage after 50k steps
**Fix**: Increase `ent_coef` to 0.7 (line 388) and re-run

**Issue**: Agent learns to attack but only spam in place
**Fix**: This is OK! Spamming attack is progress. Next curriculum stage will add movement shaping.

**Issue**: Training crashes with CUDA OOM
**Fix**: Reduce `batch_size` from 64 to 32 (line 385)

**Issue**: Agent walks off stage and dies
**Fix**: This is OK at this stage. Once it learns attacking, we'll add `danger_zone_reward` back.

---

## Success Metrics Summary

| Metric | Initial | Target @ 50k | Status |
|--------|---------|--------------|--------|
| Damage Dealt | 0 | >50 | Check logs |
| Win Rate vs Constant | ~0% | >80% | Check benchmarks |
| Damage Ratio | 0.0 | >1.0 | Check benchmarks |
| Agent Attacks | Never | Consistently | Watch episodes |

**When all targets met → Switch to CURRICULUM training! 🎉**
