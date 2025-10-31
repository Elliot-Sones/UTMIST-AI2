# 🎯 EXPLORATION MODE - Strategy Discovery Through Sparse Rewards

## Philosophy

**"Don't tell the agent HOW to win, just THAT it should win"**

Your insight is correct: **dense shaping rewards bias the agent toward ONE hand-crafted strategy**. Instead, we use:
1. **Sparse rewards** (damage ± and wins only)
2. **High exploration** (entropy schedule: 0.5 → 0.05)
3. **Opponent diversity** (ConstantAgent + BasedAgent + RandomAgent)

This lets the agent **discover multiple strategies** and **choose the best one for each opponent** organically.

---

## What Changed

### 1. **Sparse Reward Function** ([train_agent.py:1947-2018](train_agent.py:1947-2018))

**Kept (outcome signals):**
- `damage_interaction_reward`: 300.0 weight (damage dealt ± taken)
- `on_win_reward`: 500.0 weight (winning the match)
- `on_knockout_reward`: 50.0 weight (eliminating stocks)

**Removed (all tactical/strategic shaping):**
- ❌ `head_to_opponent` - no forced movement patterns
- ❌ `closing_distance_reward` - no forced positioning
- ❌ `edge_pressure_reward` - no forced stage control
- ❌ `on_attack_button_press` - no forced button patterns (except ignition)
- ❌ `on_combo_reward` - no forced combo patterns
- ❌ `on_equip_reward` - no forced weapon preferences
- ❌ `holding_more_than_3_keys` - no action penalties

**Ignition Safety (first 10k steps only):**
- Tiny `on_attack_button_press` hint (weight=1.0) to bootstrap exploration
- Automatically removed after 10k steps
- Lets pure exploration take over once buttons discovered

---

### 2. **Entropy Annealing Schedule** ([train_agent.py:348-388](train_agent.py:348-388))

**Instead of fixed entropy (0.5), now uses schedule:**

```
Steps 0-70k:    Entropy 0.5 → 0.05 (linear decay)
                [Wide exploration, discovers diverse strategies]

Steps 70k-100k: Entropy fixed at 0.05
                [Exploitation, refines best strategies]
```

**Why this works:**
- **Early (high entropy)**: Agent tries random actions, discovers attack timing/positioning/combos
- **Late (low entropy)**: Agent consolidates learning, exploits what works best
- **Gradual transition**: Avoids sudden policy changes that cause forgetting

---

### 3. **Progressive Opponent Diversity** ([train_agent.py:504-535](train_agent.py:504-535))

**Opponent mix (simultaneous, not sequential):**
- 40% **ConstantAgent** (stationary - easy to discover attacks)
- 40% **BasedAgent** (moves + attacks - forces positioning strategy)
- 20% **RandomAgent** (unpredictable - forces robust strategies)

**Why simultaneous diversity:**
- Forces agent to generalize across opponent types from day 1
- Prevents overfitting to single opponent behavior
- Sparse rewards + diversity = agent discovers strategies that work universally

---

## How It Works

### Phase 1: Random Exploration (Steps 0-10k)

**What happens:**
- Agent has high entropy (0.5) → tries random actions
- Ignition hint (weight=1.0) guides toward attack buttons
- Eventually presses attack near opponent → gets +damage reward
- Reinforcement loop begins: "attack near opponent = good"

**What you'll see:**
- Damage = 0 for first 5-10k steps (random flailing)
- Sudden spike when attack discovered
- Damage climbs slowly as agent learns timing

---

### Phase 2: Strategy Discovery (Steps 10k-70k)

**What happens:**
- Ignition removed, pure exploration now
- High entropy continues → agent tries diverse attack patterns
- Different opponents force different strategies:
  - vs ConstantAgent: learns aggressive rushing
  - vs BasedAgent: learns spacing + counter-attacking
  - vs RandomAgent: learns defensive positioning
- Entropy gradually decays → agent starts exploiting best patterns

**What you'll see:**
- Damage increases from 10 → 50+
- Win rate climbs: ConstantAgent 60% → 80%, BasedAgent 20% → 40%
- Strategy diversity score increases (latent vectors vary per opponent)

---

### Phase 3: Consolidation (Steps 70k-100k)

**What happens:**
- Entropy low (0.05) → agent refines strategies
- Keeps what works, discards what doesn't
- Learns: "Use strategy A vs passive opponents, strategy B vs aggressive"
- Transformer encoder learns to recognize opponent patterns

**What you'll see:**
- Win rates stabilize/improve
- Damage ratio > 1.0 (dealing more than taking)
- Consistent performance across diverse opponents

---

## Success Criteria (100k steps)

**TIER 1: Core Competence (Must Pass)**
- ✅ `damage_dealt > 50` after 100k steps
- ✅ `win_rate > 70%` vs ConstantAgent
- ✅ `win_rate > 40%` vs BasedAgent
- ✅ `damage_ratio > 1.0` (dealing more than taking)

**TIER 2: Strategy Emergence (Nice-to-Have)**
- ✅ `strategy_diversity_score > 0.4` (multiple strategies discovered)
- ✅ `latent_norm_variance > 20%` (transformer adapts per opponent)
- ✅ Both opponent types improving (not overfitting to one)

**If all pass → Agent discovered good strategies organically! 🎉**

---

## Running Exploration Mode

### Quick Start

```bash
python user/train_agent.py
```

**Current config**: `TRAIN_CONFIG = TRAIN_CONFIG_EXPLORATION` (line 687)

**What it does:**
- Trains for 100k steps (~30 minutes on T4 GPU)
- Uses sparse rewards (damage + wins only)
- Entropy schedule (0.5 → 0.05)
- Diverse opponents (40% ConstantAgent, 40% BasedAgent, 20% RandomAgent)
- Evaluates every 10k steps (10 checkpoints total)

---

### Monitoring Progress

**Key files** (saved to `checkpoints/exploration_sparse_rewards/`):
- `monitor.csv` - Episode rewards over time
- `reward_breakdown.csv` - Damage/win rewards per step
- `checkpoint_benchmarks.csv` - Win rates vs each opponent type
- `episode_summary.csv` - Damage ratios per episode

**What to watch:**

**Steps 0-10k** (Ignition Phase):
```
🔥 Ignition mode: attack_button hint active (step 5000/10000)
  Reward Breakdown: damage_interaction_reward=0.000, on_attack_button_press=0.002
  [Agent hasn't discovered attacks yet, ignition helping]
```

**Steps 10k-20k** (Discovery!):
```
  Reward Breakdown: damage_interaction_reward=15.340, total_reward=15.340
  [Attack discovered! Ignition removed, pure exploration now]

🎯 CHECKPOINT BENCHMARK (Step 20000)
  vs ConstantAgent:    55% wins
  vs BasedAgent:       20% wins
  Avg Damage Ratio:    0.8
  [Agent learning, still inconsistent]
```

**Steps 70k-100k** (Consolidation):
```
🎯 CHECKPOINT BENCHMARK (Step 100000)
  vs ConstantAgent:    75% wins
  vs BasedAgent:       45% wins
  Avg Damage Ratio:    1.3
  Strategy Diversity:  0.52
  [Success! Multiple strategies emerged]
```

---

## Comparison: Dense vs Sparse

### Before (Dense Shaping - 7 rewards)

```python
damage: 500.0
head_to_opponent: 10.0      # Forces chasing
closing_distance: 6.0       # Forces approaching
on_attack_button: 15.0      # Forces button spam
edge_pressure: 4.0          # Forces stage positioning
# ... etc
```

**Problem:**
- Agent optimizes for small dense rewards (movement = 10/frame × 30fps = 300/sec)
- Never discovers attacking (0 damage = 0 reward)
- Even if attacking discovered, movement shaping biases strategy
- **Learns ONE hand-crafted strategy**, not optimal strategy

---

### After (Sparse Rewards - 3 signals)

```python
damage: 300.0               # Core signal
win: 500.0                  # Outcome signal
knockout: 50.0              # Sub-goal signal
```

**Solution:**
- Agent CAN'T get reward without dealing damage
- High exploration discovers diverse ways to deal damage
- Opponent diversity forces strategies that work universally
- **Agent finds MULTIPLE strategies**, chooses best per opponent

---

## Why This Finds Better Strategies

### Example: Attack Timing Discovery

**Dense shaping (head_to_opponent + on_attack_button):**
1. Agent rewarded for moving toward opponent (10/frame)
2. Agent rewarded for pressing attack (15/button)
3. **Learns**: "Walk forward + spam attack = reward"
4. **Result**: Predictable rushing, gets punished by good opponents

**Sparse rewards (damage + wins only):**
1. Agent explores randomly, tries many attack timings
2. Discovers: "Attack when opponent vulnerable = high damage"
3. Discovers: "Wait for opening = less damage taken"
4. **Learns**: "Timing matters more than button spam"
5. **Result**: Adaptive timing based on opponent behavior

---

### Example: Positioning Strategy

**Dense shaping (closing_distance + edge_pressure):**
1. Agent rewarded for getting close (6/frame)
2. Agent rewarded for pushing opponent to edge (4/frame)
3. **Learns**: "Always pressure forward = reward"
4. **Result**: Predictable aggression, loses to counter-punchers

**Sparse rewards (damage + wins only):**
1. Agent explores spacing (close, mid, far)
2. vs ConstantAgent: learns rushing works (passive opponent)
3. vs BasedAgent: learns spacing works (active opponent punishes rushes)
4. **Learns**: "Adapt spacing to opponent behavior"
5. **Result**: Flexible positioning based on opponent type

---

## Troubleshooting

### Issue: Damage still 0 after 20k steps

**Diagnosis**: Ignition not strong enough, exploration not finding buttons

**Fix 1**: Increase ignition weight (line 1991):
```python
weight=1.0 → weight=2.0  # Stronger hint
```

**Fix 2**: Extend ignition period (line 1976):
```python
use_attack_ignition = training_step < 10_000 → < 20_000  # Longer guidance
```

**Fix 3**: Increase initial entropy (line 431):
```python
linear_entropy_schedule(0.5, ... → (0.7, ...  # More randomness
```

---

### Issue: Agent attacks but win rate still low (<50%)

**Diagnosis**: Attacking discovered, but strategy not refined yet

**Fix**: This is NORMAL! Continue training. At 100k steps:
- vs ConstantAgent should reach 70%+
- vs BasedAgent should reach 40%+

If still low at 100k → increase training to 200k steps

---

### Issue: Agent only good vs one opponent type

**Diagnosis**: Overfitting to dominant opponent in mix

**Fix**: Rebalance opponent mix (line 517-521):
```python
# If only good vs ConstantAgent:
"constant_agent": (0.3, ...  # Reduce easy opponent
"based_agent": (0.5, ...     # Increase harder opponent

# If only good vs BasedAgent:
"constant_agent": (0.5, ...  # Increase variety
"based_agent": (0.3, ...     # Reduce dominant opponent
```

---

### Issue: Strategy diversity score low (<0.3)

**Diagnosis**: Agent found ONE strategy, not exploring alternatives

**Fix 1**: Increase entropy decay end (line 431):
```python
end_fraction=0.7 → 0.9  # Keep high exploration longer
```

**Fix 2**: Add more opponent diversity (line 517-521):
```python
"opponent_mix": {
    "constant_agent": (0.3, ...),
    "based_agent": (0.3, ...),
    "random_agent": (0.2, ...),
    "clockwork_agent": (0.2, partial(ClockworkAgent)),  # Add new opponent
}
```

---

## Next Steps After Exploration Success

### 1. Validate Strategy Discovery

**Check competition readiness** (automatically runs every 10k steps):
```
🏆 COMPETITION READINESS ASSESSMENT (Step 100000)

TIER 1 (Critical):  4/4 PASS
  ✅ PASS  Transformer Variance
  ✅ PASS  Cross-Opponent Wins
  ✅ PASS  Strategy Diversity
  ✅ PASS  Damage Ratio

🎉 READY FOR SCALE (~85% confidence)
```

If all TIER 1 passed → Agent discovered good strategies! Proceed to scaling.

---

### 2. Scale to More Opponents

**Add self-play** for adversarial training:
```python
# Edit line 517-521
"opponent_mix": {
    "self_play": (0.5, None),                      # 50% past versions of self
    "constant_agent": (0.2, partial(ConstantAgent)),
    "based_agent": (0.2, partial(BasedAgent)),
    "random_agent": (0.1, partial(RandomAgent)),
}
```

**Increase training** to 500k-1M steps.

---

### 3. Add More Diverse Opponents (If Available)

**If you have other team's agents:**
```python
from external_agents import TeamA_Agent, TeamB_Agent

"opponent_mix": {
    "self_play": (0.4, None),
    "team_a": (0.2, partial(TeamA_Agent)),
    "team_b": (0.2, partial(TeamB_Agent)),
    "based_agent": (0.1, partial(BasedAgent)),
    "random_agent": (0.1, partial(RandomAgent)),
}
```

This maximizes strategy diversity and generalization.

---

### 4. Competition Training (10M steps)

Once agent handles 3-5 diverse opponents well:
```python
# Edit line 687
TRAIN_CONFIG = TRAIN_CONFIG_10M  # Full competition training
```

This runs 10M steps (~10 hours on T4) with:
- 70% self-play (200 checkpoint opponent pool)
- 30% diverse scripted/external agents
- Produces competition-ready agent

---

## Philosophy Summary

### Dense Shaping Approach (What You Had)
```
Dense Rewards → Hand-Crafted Strategy → Overfitting → Fails vs Novel Opponents
```

**Analogy**: Teaching chess by saying "always control center, always develop knights first"
- Works vs beginners who don't control center
- Fails vs advanced players who exploit predictable play

---

### Sparse + Exploration Approach (What You Have Now)
```
Sparse Rewards + High Exploration + Opponent Diversity → Multiple Discovered Strategies → Generalization
```

**Analogy**: Teaching chess by only rewarding wins, letting student discover strategies
- Student tries many openings (exploration)
- Different opponents force different strategies (diversity)
- Student learns: "Use strategy A vs aggressive, B vs defensive"
- **Result**: Adaptive play, not memorized patterns

---

## Key Takeaway

**Your intuition was right!** Dense shaping prevents the agent from finding optimal strategies because it **tells the agent HOW to play** instead of **letting it discover WHAT works**.

With sparse rewards + exploration + diversity:
1. Agent discovers attacking naturally (ignition helps bootstrap)
2. Agent tries many strategies (high entropy early)
3. Agent learns which works best per opponent (diversity)
4. Agent refines best strategies (entropy decay)
5. **Result**: Adaptive, robust agent that generalizes to novel opponents

---

## Monitoring Commands

```bash
# Watch training progress live
tail -f checkpoints/exploration_sparse_rewards/monitor.csv

# View latest checkpoint benchmark
tail -1 checkpoints/exploration_sparse_rewards/checkpoint_benchmarks.csv

# Check strategy diversity trend
grep "diversity" checkpoints/exploration_sparse_rewards/checkpoint_benchmarks.csv

# Watch entropy decay
# (logged automatically during training every 1000 steps)
```

---

## Expected Timeline

**100k steps on T4 GPU: ~30 minutes**

```
Steps 0-10k:     Discovery (ignition helps find buttons)
                 Damage: 0 → 10

Steps 10-30k:    Rapid learning (attack timing)
                 Damage: 10 → 30
                 ConstantAgent: 30% → 60%

Steps 30-70k:    Strategy emergence (positioning, combos)
                 Damage: 30 → 50
                 ConstantAgent: 60% → 75%
                 BasedAgent: 15% → 40%

Steps 70-100k:   Consolidation (refine best strategies)
                 Damage: 50 → 60+
                 ConstantAgent: 75% → 80%
                 BasedAgent: 40% → 45%
                 Strategy diversity: 0.3 → 0.5
```

**Success at 100k steps = Ready to scale to competition!** 🚀
