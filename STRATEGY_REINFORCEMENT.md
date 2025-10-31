# 🧠 Strategy Reinforcement - How the Agent Remembers What Works

## Your Question

**"When the model finds a good path, how do we clearly reinforce it so when the agent sees the same strategy, it knows exactly what to do?"**

---

## The Reinforcement Loop

### 1. **Discovery** (High Entropy Phase)

**Agent explores randomly:**
```
Match 1 vs BasedAgent:
  - Tries rushing → gets hit → damage_taken = 50 → reward = -50
  - Tries spacing + counter → lands hits → damage_dealt = 80 → reward = +80

Match 2 vs BasedAgent:
  - Tries rushing again → same result → reward = -50
  - Tries spacing again → same result → reward = +80
```

**What PPO learns:**
- Rushing vs BasedAgent → **negative value** (bad outcome)
- Spacing vs BasedAgent → **positive value** (good outcome)

**Transformer encoder learns:**
- BasedAgent movement pattern → latent vector L_based
- Policy learns: when latent ≈ L_based → use spacing strategy

---

### 2. **Reinforcement** (Repeated Trials)

**Agent faces BasedAgent many times:**
```
Match 3, 4, 5, 6 vs BasedAgent:
  - Each time: transformer outputs latent ≈ L_based
  - Each time: policy uses spacing strategy
  - Each time: gets reward = +80

Value function updates:
  V(state, latent=L_based) = 80  (high value for this strategy)
```

**What gets reinforced:**
- Transformer: "These movement patterns → L_based"
- Policy: "When latent=L_based → spacing actions get reward"
- Value function: "Being in this state with latent=L_based is valuable"

**After many trials:**
- Transformer reliably recognizes BasedAgent → L_based
- Policy reliably uses spacing when latent=L_based
- **Strategy is CEMENTED in memory**

---

### 3. **Recognition** (Future Matches)

**New match vs BasedAgent:**
```
Step 1: Transformer sees opponent observations (first 90 frames)
        → Recognizes movement pattern
        → Outputs latent ≈ L_based

Step 2: Policy receives latent=L_based
        → Looks up learned strategy
        → Executes spacing actions (high Q-value)

Step 3: Gets reward = +80
        → Further reinforces: "This was correct strategy"
```

**Agent "knows exactly what to do" because:**
1. Transformer recognizes opponent type
2. Policy retrieves correct strategy from memory
3. Value function confirms this strategy is valuable

---

## Why Sparse Rewards Enable This

### Problem with Dense Shaping

**With dense rewards (head_to_opponent, closing_distance):**
```
Match vs BasedAgent:
  - Agent rushes (gets head_to_opponent reward = +10/frame)
  - Gets hit (damage_taken = -50)
  - Total reward: (+10 × 90 frames) - 50 = +850 still positive!

Agent learns: "Rushing is good" (dense reward overpowers damage)
```

**Result:** Dense shaping **obscures the true signal** (spacing was better strategy)

---

### Solution with Sparse Rewards

**With sparse rewards (damage only):**
```
Match vs BasedAgent:
  - Agent rushes → damage_taken = 50 → reward = -50 (clear negative)
  - Agent spaces → damage_dealt = 80 → reward = +80 (clear positive)

Difference: 130 reward points!
```

**Result:** Sparse rewards **amplify the signal** → agent learns spacing is MUCH better

---

## How We Ensure Strong Reinforcement

### 1. **Extended Exploration Period** ([train_agent.py:431-435](train_agent.py:431-435))

```python
ent_coef: 0.5 → 0.05 over 80% of training (was 70%)
```

**Why:**
- Agent needs multiple trials to cement strategy
- Example: BasedAgent appears 40% of matches
  - 100k steps = ~2000 episodes
  - 40% = 800 episodes vs BasedAgent
  - With 80% exploration (80k steps), agent gets ~640 trials to reinforce
  - With 70% exploration (70k steps), only ~560 trials

**More trials = stronger reinforcement**

---

### 2. **Clear Sparse Rewards** ([train_agent.py:1978-1984](train_agent.py:1978-1984))

```python
damage_interaction_reward: 300.0 weight  # Strong signal
on_win_reward: 500.0 weight              # Ultimate outcome
```

**Why weight=300.0 for damage:**
- Typical match: 50-100 damage dealt/taken
- Reward magnitude: ±15-30 per episode
- PPO advantage calculation: A = R - V(s)
  - Good strategy: A = +30 - 0 = +30 (strong positive advantage)
  - Bad strategy: A = -30 - 0 = -30 (strong negative advantage)

**Large advantage = strong learning signal = fast reinforcement**

---

### 3. **Opponent Diversity** ([train_agent.py:517-521](train_agent.py:517-521))

```python
"constant_agent": 40%   # Teaches rushing strategy
"based_agent": 40%      # Teaches spacing strategy
"random_agent": 20%     # Teaches defensive strategy
```

**Why diversity matters:**
- Agent learns: "Different opponents need different strategies"
- Transformer learns: "These patterns → use strategy A, those patterns → use strategy B"
- **Forces strategic memory, not one fixed policy**

---

## Validation: How to Know It's Working

### 1. **Strategy Diversity Score** (Monitored automatically)

**What it measures:**
- Standard deviation of transformer latent vectors across episodes
- High variance = transformer outputs different latents per opponent

**Target:** > 0.4 after 100k steps

**What it means:**
- latent_norm = 10 vs ConstantAgent (aggressive rushing)
- latent_norm = 15 vs BasedAgent (spacing + counters)
- latent_norm = 8 vs RandomAgent (defensive play)

**If low (<0.3):** Agent using ONE strategy for all opponents (not learning opponent-specific strategies)

---

### 2. **Latent Vector Clustering** (Check manually)

**How to visualize** (after training):

```python
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Extract latent vectors from 100 episodes
latents = []  # Shape: [100, 256]
opponent_labels = []  # ["constant", "based", "based", "constant", ...]

# PCA to 2D
pca = PCA(n_components=2)
latents_2d = pca.fit_transform(latents)

# Plot
plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=opponent_labels)
plt.title("Transformer Latent Space (should cluster by opponent)")
plt.show()
```

**What you want to see:**
- Clear clusters: ConstantAgent episodes in one region, BasedAgent in another
- **This proves transformer recognizes opponents**

**If no clusters:** Transformer not learning opponent recognition → strategies not being reinforced correctly

---

### 3. **Win Rate Progression** (Logged every 10k steps)

**Expected progression:**
```
Step 20k:
  vs ConstantAgent: 50% → 65%
  vs BasedAgent: 15% → 30%
  [Agent discovering strategies]

Step 50k:
  vs ConstantAgent: 65% → 75%
  vs BasedAgent: 30% → 40%
  [Strategies getting reinforced]

Step 80k:
  vs ConstantAgent: 75% → 78%
  vs BasedAgent: 40% → 43%
  [Strategies cemented, refinement only]

Step 100k:
  vs ConstantAgent: 78% → 80%
  vs BasedAgent: 43% → 45%
  [Strategies fully reinforced]
```

**If win rates plateau early** (e.g., ConstantAgent 60% by step 30k, doesn't improve):
- Agent found A strategy, but not optimal strategy
- Entropy decay too fast → agent stopped exploring
- **Fix:** Increase initial entropy to 0.6 or extend end_fraction to 0.9

---

## The Complete Reinforcement Cycle

```
┌─────────────────────────────────────────────────────────────┐
│ EPISODE START vs BasedAgent                                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. RECOGNITION                                              │
│    - Transformer sees opponent observations (90 frames)     │
│    - Encodes movement pattern → latent L_based = [0.2, ..] │
│    - "I've seen this pattern before!"                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. STRATEGY RETRIEVAL                                       │
│    - Policy receives latent=L_based                         │
│    - Looks up learned strategy from memory                  │
│    - π(a | s, L_based) = [0.1, 0.6, 0.1, ...]             │
│    - "When I see L_based, I should use spacing actions"     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. EXECUTION                                                │
│    - Agent uses spacing strategy (action=1: "move back")    │
│    - Waits for opponent to approach                         │
│    - Counter-attacks when opponent commits                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. OUTCOME                                                  │
│    - Damage dealt = 85                                      │
│    - Damage taken = 30                                      │
│    - Reward = (85 - 30) × 300 = +16,500                    │
│    - Win = True → +500                                      │
│    - Total reward = +17,000                                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. REINFORCEMENT (PPO Update)                               │
│    - Value function: V(s, L_based) += 17,000               │
│    - Policy: Increase prob of spacing actions when L_based  │
│    - Transformer: Strengthen L_based for BasedAgent pattern│
│    - "This strategy works! Remember it!"                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ NEXT EPISODE vs BasedAgent (10 episodes later)              │
│    - Transformer outputs L_based again (recognition)        │
│    - Policy uses spacing strategy again (retrieval)         │
│    - Gets high reward again (confirmation)                  │
│    - Reinforcement strengthens further                      │
│    - After 50+ trials: strategy CEMENTED                    │
└─────────────────────────────────────────────────────────────┘
```

---

## Why This Works Better Than Dense Shaping

### Dense Shaping Problem

**Agent gets reward for following rules, not for winning:**
```
Episode vs BasedAgent:
  - Rule: "Move toward opponent" → +10/frame × 90 = +900
  - Rule: "Press attack button" → +15/press × 20 = +300
  - Outcome: Lost match, damage_taken = 100 → -100
  - Total: +900 + 300 - 100 = +1100 (POSITIVE!)

Agent learns: "Following rules = good"
Problem: Rules don't lead to wins vs this opponent!
```

**Reinforcement is WEAK because:**
- Dense rewards dominate sparse rewards
- Agent doesn't learn "this strategy loses"
- No signal to try different strategy

---

### Sparse Rewards Solution

**Agent only gets reward for outcomes:**
```
Episode vs BasedAgent (rushing strategy):
  - No dense rewards
  - Outcome: damage_taken = 100, damage_dealt = 20
  - Total: (20 - 100) × 300 = -24,000 (LARGE NEGATIVE!)

Episode vs BasedAgent (spacing strategy):
  - No dense rewards
  - Outcome: damage_taken = 30, damage_dealt = 85
  - Total: (85 - 30) × 300 = +16,500 (LARGE POSITIVE!)

Difference: 40,500 reward points between strategies!
```

**Reinforcement is STRONG because:**
- Clear signal: spacing > rushing (40k reward difference)
- PPO rapidly updates: π(spacing | L_based) ↑↑↑
- After 10 trials: agent reliably uses spacing vs BasedAgent

---

## Potential Issues & Fixes

### Issue 1: Agent discovers strategy but doesn't reinforce it

**Symptoms:**
- Agent sometimes uses spacing vs BasedAgent (50% of matches)
- Sometimes uses rushing vs BasedAgent (50% of matches)
- Win rate stuck at ~50%

**Diagnosis:** Exploration too high, not consolidating learning

**Fix:** Decrease entropy faster
```python
# Line 431
linear_entropy_schedule(0.5, 0.05, end_fraction=0.8)
→ linear_entropy_schedule(0.5, 0.03, end_fraction=0.7)  # Lower final, faster decay
```

---

### Issue 2: Agent reinforces SUB-OPTIMAL strategy

**Symptoms:**
- Agent uses rushing vs ConstantAgent (works, 80% win)
- Agent uses rushing vs BasedAgent (fails, 20% win)
- Agent never tries spacing vs BasedAgent

**Diagnosis:** Exploration too low, agent converged early

**Fix:** Increase exploration period
```python
# Line 431
linear_entropy_schedule(0.5, 0.05, end_fraction=0.8)
→ linear_entropy_schedule(0.6, 0.05, end_fraction=0.9)  # Higher initial, longer exploration
```

---

### Issue 3: Transformer not recognizing opponents

**Symptoms:**
- Agent uses same strategy vs all opponents
- Latent vectors don't cluster by opponent type
- Strategy diversity score < 0.3

**Diagnosis:** Transformer not learning opponent-specific patterns

**Fix 1:** Increase opponent observation dimension
```python
# Line 400 (auto-detected, but can override)
"opponent_obs_dim": None → "opponent_obs_dim": 64  # Give transformer more info
```

**Fix 2:** Increase transformer capacity
```python
# Line 399-401
"num_layers": 6 → "num_layers": 8  # Deeper transformer
"num_heads": 8 → "num_heads": 12   # More attention heads
```

**Fix 3:** Increase sequence length
```python
# Line 402
"sequence_length": 90 → "sequence_length": 120  # More frames to analyze (4 seconds)
```

---

## Summary

**Your question:** "How do we reinforce good strategies so agent remembers them?"

**Answer:** The system already does this through:

1. **Sparse rewards** (outcome-based, clear signal)
   - Good strategy = +16k reward
   - Bad strategy = -24k reward
   - 40k difference → strong learning signal

2. **Repeated trials** (extended exploration)
   - 80% of training in exploration mode
   - 800+ episodes per opponent type
   - Each success reinforces strategy

3. **Transformer memory** (opponent recognition)
   - Encodes opponent pattern → latent L_opp
   - Policy learns π(strategy | L_opp)
   - When sees L_opp again → uses learned strategy

4. **PPO reinforcement** (value + policy updates)
   - V(s, L_opp) learns "this latent is valuable"
   - π(a | s, L_opp) learns "use these actions with this latent"
   - After 50+ trials: strategy cemented

**The reinforcement loop is automatic and strong with sparse rewards + exploration + diversity!**

**To verify it's working:**
- Check strategy diversity score > 0.4 (transformer learning opponent-specific latents)
- Check win rate improvement over time (strategies getting reinforced)
- Visualize latent clustering (manual PCA check)

**If reinforcement is weak:**
- Extend exploration period (end_fraction 0.8 → 0.9)
- Increase reward magnitude (weight 300 → 500)
- Add more opponent diversity
