# train_agent.py Deep Dive: Complete System Architecture & Debugging Guide

**Version:** 123
**Purpose:** Understand exactly what every component does, how data flows, where problems occur, and how your model will behave.

---

## Table of Contents

1. [System Overview & Data Flow](#1-system-overview--data-flow)
2. [Configuration System](#2-configuration-system)
3. [Agent Architectures](#3-agent-architectures)
4. [Transformer Strategy Recognition](#4-transformer-strategy-recognition)
5. [Reward System](#5-reward-system)
6. [Self-Play & Curriculum Learning](#6-self-play--curriculum-learning)
7. [Training Loop Mechanics](#7-training-loop-mechanics)
8. [Monitoring & Callbacks](#8-monitoring--callbacks)
9. [Common Problems & Solutions](#9-common-problems--solutions)
10. [Model Behavior Predictions](#10-model-behavior-predictions)

---

## 1. System Overview & Data Flow

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       TRAINING LOOP                              │
│                                                                  │
│  ┌────────────────┐      ┌─────────────────┐                   │
│  │  Environment   │ ───> │  Agent Observe  │                   │
│  │  (Combat Game) │      │                 │                   │
│  └────────────────┘      └─────────────────┘                   │
│         │                         │                             │
│         │ Opponent Obs History    │ Own Observation             │
│         │ (65 frames)             │                             │
│         ▼                         ▼                             │
│  ┌─────────────────────────────────────────┐                   │
│  │   TRANSFORMER STRATEGY ENCODER          │                   │
│  │   • Processes 65 frames (2.17 seconds)  │                   │
│  │   • Self-attention discovers patterns   │                   │
│  │   • Outputs: 256-dim strategy vector    │                   │
│  └─────────────────────────────────────────┘                   │
│                     │                                           │
│                     │ Strategy Latent (256-dim)                 │
│                     ▼                                           │
│  ┌─────────────────────────────────────────┐                   │
│  │   LSTM POLICY (RecurrentPPO)            │                   │
│  │   • Input: Strategy + Current State     │                   │
│  │   • LSTM remembers past actions         │                   │
│  │   • Actor: Outputs action probabilities │                   │
│  │   • Critic: Estimates state value       │                   │
│  └─────────────────────────────────────────┘                   │
│                     │                                           │
│                     │ Action                                    │
│                     ▼                                           │
│  ┌─────────────────────────────────────────┐                   │
│  │   REWARD SYSTEM                          │                   │
│  │   • Damage dealt/received                │                   │
│  │   • Win/loss bonuses                     │                   │
│  │   • Sparse rewards (COMPETITION MODE)    │                   │
│  └─────────────────────────────────────────┘                   │
│                     │                                           │
│                     │ Reward Signal                             │
│                     ▼                                           │
│  ┌─────────────────────────────────────────┐                   │
│  │   PPO UPDATE                             │                   │
│  │   • Calculate advantages (GAE)           │                   │
│  │   • Update policy (clip gradients)       │                   │
│  │   • Update value function                │                   │
│  │   • Backprop through BOTH transformer    │                   │
│  │     and LSTM (end-to-end learning)       │                   │
│  └─────────────────────────────────────────┘                   │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow Timeline

**Single Training Step:**

1. **Environment Reset** (every episode start)
   - Opponent selected from pool (self-play snapshots + scripted bots)
   - History buffer initialized (65 zeros)

2. **Observation Phase** (every frame, 30 FPS)
   - Environment provides: `obs = [your_state (32-dim), opponent_state (32-dim)]`
   - `OpponentHistoryWrapper` maintains rolling buffer of last 65 opponent observations
   - Agent receives: `(current_obs, opponent_history_65_frames)`

3. **Strategy Recognition** (every frame)
   - Transformer processes opponent_history → 256-dim strategy vector
   - This happens inside `TransformerConditionedExtractor.forward()`
   - **Key insight:** Strategy vector updates EVERY frame but represents ~2 seconds of context

4. **Action Selection** (every frame)
   - LSTM policy receives: `[your_state, opponent_state, strategy_vector]`
   - LSTM hidden state carries memory of previous actions
   - Output: Action probabilities for 10 discrete actions
   - **Training:** Sample from distribution (exploration)
   - **Evaluation:** Take argmax (exploitation)

5. **Rollout Collection** (every 512 steps)
   - Buffer fills with: `(obs, action, reward, done, value_estimate)`
   - When buffer full → trigger PPO update

6. **PPO Update** (every 512 steps)
   - Calculate advantages using GAE (λ=0.95, γ=0.99)
   - 10 epochs of gradient descent (n_epochs=10)
   - 8 mini-batches per epoch (512 steps / 64 batch_size)
   - Update both transformer AND LSTM weights
   - Clip policy updates to prevent catastrophic forgetting

7. **Checkpoint Saving** (every 50k steps for 10M config)
   - Save agent weights
   - Self-play handler can load this snapshot as opponent

---

## 2. Configuration System

### Configuration Hierarchy

The file uses a **shared configuration** pattern to ensure consistency:

```python
_SHARED_AGENT_CONFIG = {
    "type": "transformer_strategy",
    "latent_dim": 256,
    "num_heads": 8,
    "num_layers": 6,
    "sequence_length": 65,  # 2.17 seconds at 30 FPS
    "n_steps": 512,         # Rollout buffer size
    "batch_size": 64,
    "n_epochs": 10,
    "ent_coef": 0.01,       # Entropy for exploration
    "learning_rate": 2.5e-4,
    "clip_range": 0.2,      # PPO clip
    "gamma": 0.99,          # Discount factor
    "gae_lambda": 0.95,
}
```

### Key Configurations

#### 1. **TRAIN_CONFIG_10M** (Competition Training - 10M timesteps)

```python
TRAIN_CONFIG_10M = {
    "agent": _SHARED_AGENT_CONFIG,
    "reward": {"factory": "gen_reward_manager_SPARSE"},  # Sparse rewards
    "self_play": {
        "save_freq": 50_000,      # Save every 50k steps
        "max_saved": 200,         # Keep 200 checkpoints
        "opponent_mix": {
            "self_play": (7.0, None),           # 70% past versions
            "constant_agent": (0.5, ConstantAgent),  # 5% stationary
            "based_agent": (1.5, BasedAgent),   # 15% heuristic
            "random_agent": (1.0, RandomAgent), # 10% random
        }
    }
}
```

**Purpose:** Train a robust competition agent that can handle diverse opponents.

**Opponent Mix Explained:**
- Weights are **relative** (sum to 10.0 → 70%/5%/15%/10% distribution)
- `self_play: 7.0` means 70% of episodes against past versions
- Ensures agent doesn't overfit to specific opponent patterns

#### 2. **TRAIN_CONFIG_CURRICULUM** (Stage 1 - 50k timesteps)

```python
"opponent_mix": {
    "constant_agent": (10, ConstantAgent),  # 100% against stationary target
}
```

**Purpose:** Learn basic combat mechanics before facing complex opponents.

**Success Criteria:** 90%+ win rate vs ConstantAgent, positive damage ratio.

#### 3. **TRAIN_CONFIG_CURRICULUM_STAGE2** (Stage 2 - 50k timesteps)

```python
"opponent_mix": {
    "based_agent": (10, BasedAgent),  # 100% against heuristic AI
}
```

**Purpose:** Learn to counter rule-based strategies.

**Success Criteria:** 70%+ win rate vs BasedAgent.

---

## 3. Agent Architectures

### TransformerStrategyAgent

**Class:** `TransformerStrategyAgent(Agent)`
**Location:** Lines 1057-1400

#### Components

1. **TransformerStrategyEncoder** (Lines 831-960)
   - **Input:** `[batch, 65, 32]` (65 frames of opponent observations)
   - **Output:** `[batch, 256]` (strategy latent vector)
   - **Architecture:**
     - Frame embedding: Linear(32 → 256) + LayerNorm + ReLU
     - Positional encoding (sinusoidal)
     - 6-layer Transformer encoder (8 heads, 256-dim)
     - Attention pooling (learns which frames matter)
     - Strategy refinement (2-layer MLP)

2. **RecurrentPPO Policy** (from sb3-contrib)
   - **Input:** Augmented observation `[your_state + opponent_state + strategy_latent]`
   - **Architecture:**
     - LSTM: 512 hidden units (shared between actor/critic)
     - Actor network: [96, 96] → 10 actions
     - Critic network: [96, 96] → 1 value estimate
   - **Key feature:** LSTM maintains hidden state across frames

#### Memory Architecture

```
┌─────────────────────────────────────────────────────┐
│  TRANSFORMER MEMORY (Short-term, 65 frames)         │
│  • Rolling window of opponent behavior              │
│  • Resets every episode                             │
│  • Purpose: Recognize current strategy              │
└─────────────────────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────┐
│  LSTM MEMORY (Medium-term, hidden state)            │
│  • Remembers agent's own action sequence            │
│  • Persists within episode                          │
│  • Purpose: Context-aware action selection          │
└─────────────────────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────┐
│  POLICY WEIGHTS (Long-term, learned parameters)     │
│  • ~4M parameters (2.5M transformer + 1.5M policy)  │
│  • Persists across episodes                         │
│  • Purpose: General strategy knowledge              │
└─────────────────────────────────────────────────────┘
```

#### Why This Architecture?

**Problem:** Opponent strategy changes within match (e.g., aggressive → defensive).

**Solution:**
- **Transformer:** Fast adaptation (updates every frame with 2-second context)
- **LSTM:** Remembers agent's own action history for coordination
- **Combined:** Can adapt to opponent while maintaining coherent strategy

**Comparison to alternatives:**
- Pure LSTM: Slower adaptation (hidden state accumulates entire episode)
- Pure Transformer: No memory of own actions (incoherent behavior)

---

## 4. Transformer Strategy Recognition

### How It Works

#### Input Processing

```python
# OpponentHistoryWrapper maintains rolling buffer
opponent_history = [
    opponent_obs[t-64],  # Oldest
    opponent_obs[t-63],
    ...
    opponent_obs[t]      # Current frame
]
# Shape: [65, 32]
```

**32-dim opponent observation:**
- Position (x, y)
- Velocity (vx, vy)
- Health, armor, shield
- Equipped weapon
- Animation state (idle, attacking, dashing, etc.)
- Distance to agent
- Angle to agent
- ... (exact breakdown depends on environment observation space)

#### Attention Mechanism

**Self-Attention Discovers Patterns:**

Example learned pattern (hypothetical):
```
Frame 10: opponent_pos = [100, 200], state = "idle"
Frame 20: opponent_pos = [120, 200], state = "dashing"
Frame 30: opponent_pos = [140, 200], state = "attacking"

Attention weights: [0.02, ..., 0.45 (frame 20), ..., 0.48 (frame 30)]
```

The transformer learns: "When dash → attack sequence detected, high attention to those frames."

**Strategy Latent Space:**

- **Continuous 256-dim vector** (NOT discrete categories)
- Example interpretations (network learns these implicitly):
  - Dimension 57 might encode "aggression level" (high = rushes, low = defensive)
  - Dimension 142 might encode "weapon preference" (positive = melee, negative = ranged)
  - Dimensions interact: [57, 142] together encode "aggressive melee rusher"

**Important:** You CANNOT directly interpret dimensions. The network learns its own representation. Visualization (t-SNE, PCA) can reveal clusters.

#### Attention Pooling

After transformer processes all frames:

```python
contextualized_frames = [f0', f1', ..., f64']  # Each frame attended to others
strategy_latent = weighted_sum(contextualized_frames, attention_weights)
```

**Key insight:** Network learns which frames are most "diagnostic" of strategy.

Example: If opponent always attacks after dashing, attention will focus on dash→attack transition frames.

---

## 5. Reward System

### Reward Philosophies

#### SPARSE Rewards (Competition Mode)

**Function:** `gen_reward_manager_SPARSE()` (Lines 2184-2248)

```python
rewards = {
    "damage_dealt": +1.0 per damage,
    "damage_received": -1.0 per damage,
    "win_bonus": +100.0,
    "loss_penalty": -100.0,
    "knockout_bonus": +50.0,
    "knockout_penalty": -50.0,
}
```

**Philosophy:** "Minimal hand-holding, let exploration find strategies."

**Pros:**
- Agent discovers creative strategies
- Doesn't overfit to shaped rewards
- More general/robust

**Cons:**
- Slower initial learning
- Requires good exploration (entropy)
- May get stuck in local optima

#### DENSE Rewards (Curriculum Mode)

**Function:** `gen_reward_manager()` (Lines 2108-2182)

**Additional signals:**
- `danger_zone_reward`: Penalty for low health without shield
- `closing_distance_reward`: Reward for approaching opponent
- `edge_pressure_reward`: Bonus for forcing opponent to edge
- `combo_reward`: Bonus for consecutive hits
- And ~10 more shaped rewards

**Philosophy:** "Guide agent toward good behaviors with training wheels."

**Pros:**
- Faster initial learning
- Good for curriculum stage 1
- Helps overcome sparse reward challenges

**Cons:**
- Agent may overfit to shaped rewards
- Less creative strategies
- Rewards might conflict (e.g., "stay close" vs "maintain distance")

### How Rewards Affect Learning

**Example: Damage Interaction**

```python
def damage_interaction_reward(env, agent):
    p1_damage_dealt = env.damage_dealt[agent]
    p2_damage_dealt = env.damage_dealt[opponent]

    delta = p1_damage_dealt - p2_damage_dealt
    return delta  # Positive if dealing more, negative if receiving more
```

**Training dynamics:**
- Early training: Agent learns "attacking = good" (damage_dealt increases)
- Mid training: Agent learns "defense matters" (damage_received hurts reward)
- Late training: Agent balances aggression/defense based on opponent

**Potential problem:** If agent discovers "safe camping" strategy (stay away, never engage), damage is 0 but reward is higher than aggressive failing strategies. Solution: Add win/loss bonuses to force engagement.

### Signal-Based Rewards

**Concept:** Rewards triggered by environment events, not computed every frame.

**Example: Knockout Reward**

```python
def on_knockout_reward(env, agent):
    """Called when agent knocks out opponent."""
    return 50.0  # One-time bonus
```

**Why signals?**
- Efficient (computed only when event occurs)
- Clear semantics (reward directly tied to outcome)
- Avoids "reward hacking" (agent can't manipulate continuous reward)

**Current signals:**
- `on_win`: Episode ends with victory
- `on_knockout`: Opponent health → 0
- `on_equip`: Agent picks up weapon
- `on_combo`: Hit opponent N times in succession
- `on_drop`: Agent drops weapon

---

## 6. Self-Play & Curriculum Learning

### Self-Play Mechanism

**Key Classes:**
- `SelfPlayHandler` (from environment module)
- `SaveHandler` (from environment module)
- `OpponentsCfg` (from environment module)

#### SaveHandler (Checkpointing)

**Parameters:**
- `save_freq`: Save every N steps (e.g., 50,000)
- `max_saved`: Maximum checkpoints to keep (e.g., 200)
- `mode`: `FORCE` (always save) or `SELECTIVE` (save if performance improves)

**What happens:**

```python
# During training, every 50k steps:
save_handler.save_agent()
# Creates: checkpoints/competition_10M_final/agent_50000.zip
#          checkpoints/competition_10M_final/agent_100000.zip
#          ...
#          checkpoints/competition_10M_final/agent_10000000.zip
```

**Why 200 checkpoints?**
- Diverse opponent pool (early agents play differently than late agents)
- Prevents overfitting to current strategy
- Enables robust strategy discovery

#### SelfPlayHandler (Opponent Loading)

**Types:**
- `SelfPlayRandom`: Randomly sample from past checkpoints
- `SelfPlayRecent`: Prefer recent checkpoints (99% recent, 1% older)

**What happens:**

```python
# At episode start:
opponent = selfplay_handler.get_opponent()
# Loads random checkpoint: agent_1550000.zip
# Agent plays against past version of itself
```

**Training dynamics:**

1. **Steps 0-50k:** No self-play opponents (only scripted bots)
2. **Steps 50k-100k:** Play against checkpoint@50k + scripted bots
3. **Steps 100k-150k:** Play against checkpoints@50k,100k + bots
4. **Steps 10M:** Play against 200 diverse past versions + bots

**Why this works:**
- Agent must maintain robustness to old strategies (can't forget)
- Creates "evolutionary pressure" (strategies that beat old strategies survive)
- Similar to AlphaGo self-play (iterative improvement)

### Opponent Mix Strategy

**Definition:**

```python
opponent_mix = {
    "self_play": (7.0, selfplay_handler),
    "constant_agent": (0.5, ConstantAgent),
    "based_agent": (1.5, BasedAgent),
    "random_agent": (1.0, RandomAgent),
}
```

**Selection algorithm:**

```python
total_weight = 7.0 + 0.5 + 1.5 + 1.0 = 10.0
probabilities = {
    "self_play": 7.0 / 10.0 = 0.70,
    "constant_agent": 0.5 / 10.0 = 0.05,
    "based_agent": 1.5 / 10.0 = 0.15,
    "random_agent": 1.0 / 10.0 = 0.10,
}
# At each episode start: opponent = random.choice(based on probabilities)
```

**Why mix opponents?**
- **Self-play (70%):** Learn counter-strategies, continuous improvement
- **Scripted bots (30%):** Maintain basic competencies, prevent overfitting

**Common mistake:** 100% self-play → agent might learn "meta-strategy" that only works against itself (e.g., exploiting specific weakness), but fails against new opponents.

### Curriculum Learning

**Philosophy:** "Easy tasks → Hard tasks" (like learning math: addition before calculus).

**Stage 1: Beat ConstantAgent (50k steps)**
- Opponent: Stationary target
- Goal: Learn basic actions (move, attack, dash)
- Success: 90%+ win rate

**Stage 2: Beat BasedAgent (50k steps)**
- Opponent: Rule-based AI (chases, attacks when close)
- Goal: Learn positioning, timing
- Success: 70%+ win rate

**Stage 3: Self-play (10M steps)**
- Opponent: Past versions + mix
- Goal: Discover advanced strategies
- Success: High ELO, robust to diverse opponents

**When to advance:**
- Manual: Train each stage separately, evaluate, then proceed
- Automatic: `CurriculumRewardScheduler` can transition based on performance metrics

---

## 7. Training Loop Mechanics

### Step-by-Step Training Execution

**Function:** `run_training_loop()` (Lines 3321-3413)

#### Initialization Phase

```python
# 1. Create environment
base_env = SelfPlayWarehouseBrawl(
    reward_manager=reward_manager,
    opponent_cfg=opponent_cfg,
    save_handler=save_handler,
)

# 2. Wrap with opponent history tracking
env = OpponentHistoryWrapper(
    base_env,
    opponent_obs_dim=32,
    sequence_length=65,
)

# 3. Wrap with monitoring
env = Monitor(env, log_dir)

# 4. Initialize agent
agent.get_env_info(env)  # Sets observation/action spaces
```

#### Training Phase

```python
# Main training call
agent.learn(
    env,
    total_timesteps=10_000_000,
    verbose=1,
    callback=monitor_callback
)
```

**What happens inside `agent.learn()`:**

```python
# Pseudocode of RecurrentPPO.learn()
for step in range(0, total_timesteps, n_steps):
    # 1. ROLLOUT PHASE (collect 512 steps)
    for i in range(512):
        obs = env.current_observation
        action, value = policy.predict(obs)
        next_obs, reward, done, info = env.step(action)
        buffer.add(obs, action, reward, done, value)

        if done:
            obs = env.reset()  # New episode, new opponent

    # 2. ADVANTAGE CALCULATION
    advantages = compute_gae(
        rewards=buffer.rewards,
        values=buffer.values,
        dones=buffer.dones,
        gamma=0.99,
        gae_lambda=0.95
    )

    # 3. PPO UPDATE (10 epochs)
    for epoch in range(10):
        for batch in buffer.get_batches(batch_size=64):
            # Forward pass
            new_values, new_log_probs, entropy = policy(batch.obs)

            # Compute losses
            ratio = torch.exp(new_log_probs - batch.old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1-0.2, 1+0.2)
            policy_loss = -torch.min(
                ratio * batch.advantages,
                clipped_ratio * batch.advantages
            ).mean()
            value_loss = F.mse_loss(new_values, batch.returns)
            entropy_loss = -entropy.mean()

            total_loss = (
                policy_loss
                + 0.5 * value_loss
                + 0.01 * entropy_loss
            )

            # Backward pass (updates BOTH transformer and LSTM)
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

    # 4. CHECKPOINTING
    if step % 50_000 == 0:
        save_handler.save_agent()
```

### PPO Algorithm Details

#### Advantage Calculation (GAE)

**Formula:**

```
δ_t = r_t + γ * V(s_{t+1}) - V(s_t)  # TD error
A_t = δ_t + (γλ) * δ_{t+1} + (γλ)² * δ_{t+2} + ...
```

**Parameters:**
- `γ (gamma) = 0.99`: Discount factor (care about future rewards)
- `λ (gae_lambda) = 0.95`: Bias-variance tradeoff

**What it does:**
- Estimates "how much better was this action than expected?"
- Positive advantage → increase action probability
- Negative advantage → decrease action probability

**Why GAE?**
- Pure TD (λ=0): Low variance, high bias
- Pure Monte Carlo (λ=1): High variance, low bias
- GAE (λ=0.95): Good balance

#### PPO Clipping

**Formula:**

```
L_CLIP = min(
    ratio * A_t,
    clip(ratio, 1-ε, 1+ε) * A_t
)
where ratio = π_new(a|s) / π_old(a|s)
```

**Parameters:**
- `ε (clip_range) = 0.2`: 20% policy change limit

**What it does:**
- Prevents catastrophic policy updates
- If new policy very different from old → clip update
- Ensures training stability

**Why 0.2?**
- 0.1: Too conservative (slow learning)
- 0.3: Too aggressive (unstable)
- 0.2: Sweet spot (standard PPO)

#### Entropy Bonus

**Formula:**

```
H(π) = -Σ π(a|s) * log(π(a|s))
L_total = L_CLIP + c_v * L_value + c_ent * H(π)
```

**Parameters:**
- `c_ent (ent_coef) = 0.01`: Entropy coefficient

**What it does:**
- Encourages exploration (diverse action selection)
- High entropy = uniform distribution (more random)
- Low entropy = peaked distribution (more deterministic)

**Training dynamics:**
- Early training: Entropy naturally high (random policy)
- Mid training: Entropy decreases (policy becomes confident)
- Late training: May need entropy bonus to prevent premature convergence

**Why 0.01?**
- 0.1: Too much exploration (policy stays random)
- 0.001: Too little (premature convergence)
- 0.01: Good balance

### Gradient Flow

**End-to-end learning:**

```
Reward → Advantage → Policy Loss → Gradients → LSTM weights
                                             → Transformer weights
```

**Key insight:** Transformer learns patterns that HELP policy win, not arbitrary patterns.

**Example:**
- If recognizing "dash→attack" pattern helps policy dodge → transformer learns to detect it
- If recognizing "health<20" pattern helps policy retreat → transformer learns to encode it

**Potential problem:** Vanishing gradients through transformer (6 layers) to LSTM.

**Solution:**
- Layer normalization (stabilizes gradients)
- Residual connections in transformer
- Gradient clipping (max norm = 0.5)

---

## 8. Monitoring & Callbacks

### TrainingMonitorCallback

**Class:** `TrainingMonitorCallback` (Lines 2829-2999)

**Purpose:** Enhanced logging during training.

**What it tracks:**

1. **Episode Metrics** (every episode end):
   - Episode reward
   - Episode length
   - Win/loss
   - Damage dealt/received
   - Knockouts

2. **Training Metrics** (every n_steps = 512):
   - Policy loss
   - Value loss
   - Entropy
   - Learning rate
   - Gradient norm

3. **Strategy Metrics** (if transformer agent):
   - Strategy latent statistics (mean, std)
   - Attention entropy
   - Strategy drift (how much strategy changes per episode)

4. **Opponent Metrics**:
   - Which opponent type (self-play checkpoint, scripted bot)
   - Win rate per opponent type

**Output files:**

```
checkpoints/competition_10M_final/
├── agent_50000.zip
├── agent_100000.zip
├── monitor.csv                    # Episode rewards, lengths
├── behavior_metrics.csv           # Damage, win rate, knockouts
├── reward_breakdown.csv           # Individual reward terms
├── strategy_analysis.csv          # Latent statistics (if transformer)
└── training_curves.png            # Auto-generated plots
```

### PerformanceBenchmark

**Class:** `PerformanceBenchmark` (Lines 2386-2481)

**Purpose:** Evaluate agent against test opponents.

**Usage:**

```python
benchmark = PerformanceBenchmark(
    agent=agent,
    opponents={
        "constant": ConstantAgent,
        "based": BasedAgent,
        "random": RandomAgent,
    },
    num_episodes=100
)
results = benchmark.run()
# results = {
#     "constant": {"win_rate": 0.98, "avg_reward": 87.3},
#     "based": {"win_rate": 0.76, "avg_reward": 43.2},
#     "random": {"win_rate": 0.91, "avg_reward": 68.5},
# }
```

**When to use:**
- After curriculum stage (check if ready for next stage)
- After full training (evaluate final performance)
- During development (diagnose issues)

---

## 9. Common Problems & Solutions

### Problem 1: Agent Not Learning (Reward Stuck Near 0)

**Symptoms:**
- `monitor.csv` shows flat episode rewards
- Win rate stays at ~0%
- Agent does random actions

**Possible Causes:**

1. **Sparse rewards + poor exploration**
   - **Check:** Is reward=0 for entire episodes?
   - **Solution:**
     - Increase `ent_coef` to 0.05 (more exploration)
     - Switch to dense rewards (curriculum mode)
     - Add "shaping" rewards (e.g., small bonus for approaching opponent)

2. **Reward scale issues**
   - **Check:** Are rewards too small? (e.g., 0.01 per step)
   - **Solution:** Scale rewards by 10-100x
   - **Check:** Are rewards too large? (e.g., 10000 per win)
   - **Solution:** Normalize rewards (divide by 100)

3. **Observation space issues**
   - **Check:** Are observations normalized? (Should be ~[-1, 1] or [0, 1])
   - **Solution:** Add normalization in `OpponentHistoryWrapper`
   - **Check:** Are observations informative? (Can YOU distinguish strategies from obs?)
   - **Solution:** Add more features (e.g., opponent action history)

4. **Learning rate issues**
   - **Check:** Is policy loss decreasing? (Check logs)
   - **Solution if not:** Increase LR to 5e-4
   - **Check:** Is policy loss exploding? (NaN losses)
   - **Solution if yes:** Decrease LR to 1e-4

**Debugging steps:**

```python
# 1. Test with deterministic opponent
opponent_mix = {"constant_agent": (10, ConstantAgent)}  # Should easily win

# 2. Check reward function
print(env.step(action))  # Should see non-zero rewards after damage

# 3. Visualize observations
plt.plot(opponent_history[0])  # Should see clear patterns, not noise

# 4. Test with pre-trained policy
agent = TransformerStrategyAgent(load_path="known_good_agent.zip")
# Should perform well → confirms environment is correct
```

### Problem 2: Agent Overfits to Self-Play

**Symptoms:**
- 95% win rate against self-play opponents
- 20% win rate against scripted bots
- Discovers "meta-exploit" (e.g., always does X because past versions vulnerable)

**Solution:**

1. **Increase scripted bot weight**
   ```python
   opponent_mix = {
       "self_play": (5.0, None),        # Reduce to 50%
       "constant_agent": (1.0, ...),    # 10%
       "based_agent": (2.0, ...),       # 20%
       "random_agent": (2.0, ...),      # 20%
   }
   ```

2. **Add more diverse scripted bots**
   - Implement aggressive bot, defensive bot, kiting bot
   - Forces agent to generalize

3. **Use recent self-play**
   ```python
   selfplay_handler_cls=SelfPlayRecent
   ```
   - Focuses on beating recent (more skilled) versions
   - Prevents exploiting old (weak) versions

### Problem 3: Training Unstable (Reward Oscillates Wildly)

**Symptoms:**
- Reward: 50 → 100 → 20 → 150 → -30 → ...
- Win rate: 80% → 30% → 90% → 10% → ...

**Possible Causes:**

1. **Opponent diversity too high**
   - Agent plays against snapshot@50k (weak) → easy win
   - Next episode plays against snapshot@5M (strong) → brutal loss
   - **Solution:** Use `SelfPlayRecent` (99% recent, 1% old)

2. **Learning rate too high**
   - Policy changes too fast → forgets previous strategies
   - **Solution:** Reduce LR to 1e-4

3. **PPO clip too large**
   - Policy can make big jumps → unstable
   - **Solution:** Reduce `clip_range` to 0.1

4. **Batch size too small**
   - High variance in gradient estimates
   - **Solution:** Increase `batch_size` to 128 (may reduce `n_steps` to 1024 if memory issues)

5. **n_steps too small**
   - Not enough data per update
   - **Solution:** Increase `n_steps` to 1024 or 2048

**Tuning for stability:**

```python
# Conservative settings (slow but stable)
"learning_rate": 1e-4,
"clip_range": 0.1,
"n_steps": 2048,
"batch_size": 128,
"n_epochs": 5,

# Aggressive settings (fast but unstable)
"learning_rate": 5e-4,
"clip_range": 0.3,
"n_steps": 512,
"batch_size": 64,
"n_epochs": 20,
```

### Problem 4: Transformer Not Learning Strategies

**Symptoms:**
- Strategy latent vector doesn't change across opponents
- Attention weights uniform (no focus on specific frames)
- Agent performs same against all opponents

**Possible Causes:**

1. **Sequence too short**
   - 65 frames might not capture full strategy pattern
   - **Solution:** Increase to 90 or 120 frames (costs memory)

2. **Opponent observations not informative**
   - If opponent_obs only has position → can't distinguish strategies
   - **Solution:** Add action history, animation state, attack cooldowns

3. **Transformer underfitting**
   - Too few layers/heads → can't learn complex patterns
   - **Solution:** Increase `num_layers` to 8 or `num_heads` to 12

4. **Transformer overfitting**
   - Too many parameters → memorizes instead of generalizing
   - **Solution:** Add dropout (increase from 0.1 to 0.2)

5. **End-to-end training issue**
   - Gradients not flowing to transformer (LSTM learns to ignore strategy)
   - **Solution:** Pre-train transformer with auxiliary task (e.g., predict opponent next action)

**Debugging:**

```python
# Visualize strategy latents
from sklearn.manifold import TSNE

latents = []
opponent_types = []
for episode in range(100):
    obs = env.reset()
    latent = agent.extractor.forward(obs)
    latents.append(latent.cpu().numpy())
    opponent_types.append(env.current_opponent_type)

# Should see clusters per opponent type
tsne = TSNE(n_components=2)
latents_2d = tsne.fit_transform(latents)
plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=opponent_types)
```

If clusters overlap → transformer not learning distinct representations.

### Problem 5: GPU Memory Issues

**Symptoms:**
- `RuntimeError: CUDA out of memory`
- Training crashes after N steps

**Solutions (in order of preference):**

1. **Reduce batch_size**
   ```python
   "batch_size": 32  # Down from 64
   ```

2. **Reduce n_steps**
   ```python
   "n_steps": 256  # Down from 512
   ```

3. **Reduce sequence_length**
   ```python
   "sequence_length": 45  # Down from 65
   ```

4. **Reduce transformer size**
   ```python
   "num_layers": 4,  # Down from 6
   "num_heads": 4,   # Down from 8
   "latent_dim": 128,  # Down from 256
   ```

5. **Use gradient accumulation**
   - Simulate larger batch by accumulating gradients
   - Requires code modification (not currently supported)

**Memory calculation:**

```
Transformer memory per batch:
= batch_size * sequence_length * latent_dim * num_layers * 4 bytes
= 64 * 65 * 256 * 6 * 4
≈ 25 MB

LSTM memory per batch:
= batch_size * lstm_hidden_size * num_lstm_layers * 4 bytes
= 64 * 512 * 1 * 4
≈ 0.13 MB

Rollout buffer:
= n_steps * obs_dim * 4 bytes
= 512 * (64 + 65*32) * 4
≈ 0.27 MB

Total: ~30 MB per batch (×8 batches in parallel = 240 MB)
```

### Problem 6: Training Too Slow

**Symptoms:**
- <100 FPS during training (should be ~500-1000 FPS on GPU)
- 10M steps takes >24 hours (should be ~10-12 hours on T4)

**Possible Causes:**

1. **CPU bottleneck**
   - **Check:** GPU utilization <50% (use `nvidia-smi`)
   - **Solution:**
     - Increase `n_steps` to 1024 (fewer environment resets)
     - Use vectorized environments (run 4-8 envs in parallel)

2. **Data transfer overhead**
   - **Check:** Observations moving between CPU/GPU every step
   - **Solution:** Keep observations on GPU (requires environment modification)

3. **Rendering enabled**
   - **Check:** Is `resolution=HIGH` or rendering to screen?
   - **Solution:** Use `resolution=LOW`, disable rendering

4. **Logging overhead**
   - **Check:** Are you logging every step?
   - **Solution:** Log every N steps (N=1000)

5. **Checkpoint saving too frequent**
   - **Check:** `save_freq=1000` (too often)
   - **Solution:** `save_freq=50_000`

**Optimization checklist:**

```python
# Fast training settings
resolution = CameraResolution.LOW  # Minimal rendering
save_freq = 50_000                 # Infrequent saves
train_logging = TrainLogging.PLOT  # Log to CSV, not screen
n_steps = 1024                     # Larger rollouts = fewer resets
```

---

## 10. Model Behavior Predictions

### Early Training (0-100k steps)

**Expected Behavior:**
- Random exploration (entropy high)
- Discovers basic actions: move, attack, dash
- Win rate vs ConstantAgent: 0% → 50% → 90%
- Strategy latents: Noisy, no clear patterns

**Reward progression:**
- Steps 0-10k: Reward ≈ -50 (dies quickly, doesn't attack)
- Steps 10k-30k: Reward ≈ 0 (learns to attack, but trades evenly)
- Steps 30k-50k: Reward ≈ +30 (learns to avoid damage while attacking)
- Steps 50k-100k: Reward ≈ +70 (reliably beats ConstantAgent)

**Common issues:**
- If stuck at reward=-50 at step 50k → Problem with exploration or observation space
- If reward oscillates wildly → Problem with policy stability

### Mid Training (100k-1M steps)

**Expected Behavior:**
- Learns positioning and timing
- Beats BasedAgent 70% of time
- Begins adapting to opponent patterns (transformer activates)
- Strategy latents: Some clustering visible

**Self-play dynamics:**
- Plays against snapshots from 50k-100k (weak opponents)
- Easily wins against old versions
- Develops "foundational strategies" (e.g., "approach, attack, retreat")

**Potential problems:**
- **Forgetfulness:** Forgets how to beat ConstantAgent (too focused on self-play)
  - **Solution:** Increase scripted bot weight
- **Meta-exploit:** Discovers opponent always does X at low health, hardcodes counter
  - **Solution:** Increase opponent diversity

### Late Training (1M-10M steps)

**Expected Behavior:**
- Sophisticated strategies emerge
- Adapts within match (e.g., aggressive → defensive when losing)
- Exploits opponent weaknesses (e.g., kiting if opponent rushes)
- Strategy latents: Clear clusters per opponent type

**Self-play dynamics:**
- Plays against 200 diverse snapshots (50k, 100k, ..., 9.9M)
- Evolutionary pressure: Strategies that beat past strategies survive
- May discover "rock-paper-scissors" dynamics:
  - Strategy A beats B
  - Strategy B beats C
  - Strategy C beats A
  - Agent learns meta-strategy: "Recognize A/B/C, choose counter"

**Advanced behaviors:**
- **Baiting:** Pretends to retreat, then punishes opponent's rush
- **Conditioning:** Establishes pattern (e.g., always dash right), then breaks it
- **Resource management:** Saves dash for critical moments
- **Zoning:** Controls space with weapon range

**Potential problems:**
- **Overfitting to self-play:** Beats self but loses to new opponents
  - **Check:** Test against held-out scripted bots
  - **Solution:** Increase scripted bot weight, add more bot types
- **Forgetting:** Late snapshots forget how to beat early strategies
  - **Check:** Test agent@10M vs agent@50k (should still win)
  - **Solution:** Keep old snapshots in opponent pool

### Final Model (10M steps)

**Expected Performance:**

**vs ConstantAgent:** 99%+ win rate (trivial)
**vs BasedAgent:** 95%+ win rate (easy)
**vs RandomAgent:** 90%+ win rate (randomness helps opponent occasionally)
**vs Self-play snapshots:** 60-70% win rate (fair fights)

**Strategy understanding:**

```python
# Test strategy recognition
opponent_aggressive = AggressiveBot()
obs = env.reset(opponent=opponent_aggressive)
latent_aggressive = agent.get_strategy_latent(obs)

opponent_defensive = DefensiveBot()
obs = env.reset(opponent=opponent_defensive)
latent_defensive = agent.get_strategy_latent(obs)

# Should be different
distance = np.linalg.norm(latent_aggressive - latent_defensive)
assert distance > 0.5  # Threshold depends on latent scale
```

**Failure modes:**

1. **Brittle strategies:** Works in training but fails in competition
   - **Cause:** Overfitting to opponent pool
   - **Prevention:** Maximum opponent diversity during training

2. **Exploitable patterns:** Agent always does X in situation Y
   - **Cause:** Local optimum (strategy works against training opponents but exploitable)
   - **Prevention:** Adversarial testing (find weaknesses, add to opponent pool)

3. **Poor generalization:** Loses to "weird" opponents (e.g., agent that only kicks)
   - **Cause:** Training distribution doesn't cover edge cases
   - **Prevention:** Add diverse scripted bots

---

## Appendix: Quick Reference

### Configuration Cheat Sheet

| Parameter | Default | Fast Learning | Stable Learning |
|-----------|---------|---------------|-----------------|
| `learning_rate` | 2.5e-4 | 5e-4 | 1e-4 |
| `n_steps` | 512 | 256 | 2048 |
| `batch_size` | 64 | 64 | 128 |
| `n_epochs` | 10 | 20 | 5 |
| `ent_coef` | 0.01 | 0.05 | 0.005 |
| `clip_range` | 0.2 | 0.3 | 0.1 |
| `sequence_length` | 65 | 45 | 90 |

### Hyperparameter Effects

| Parameter | Increase Effect | Decrease Effect |
|-----------|----------------|-----------------|
| `learning_rate` | Faster learning, less stable | Slower learning, more stable |
| `n_steps` | Better value estimates, slower updates | Faster updates, higher variance |
| `batch_size` | Lower gradient variance | Higher gradient variance, faster iterations |
| `n_epochs` | More optimization per rollout | Less optimization per rollout |
| `ent_coef` | More exploration | More exploitation |
| `clip_range` | Larger policy updates | Smaller policy updates |
| `gamma` | Care more about future | Care more about immediate rewards |
| `gae_lambda` | Lower bias, higher variance | Higher bias, lower variance |
| `sequence_length` | More temporal context | Less memory, faster computation |
| `latent_dim` | Richer representations | Faster training, less expressive |
| `num_layers` | More complex patterns | Faster training, simpler patterns |

### Debug Commands

```python
# Check if agent is learning
python train_agent.py --mode=train --config=CURRICULUM
# Watch monitor.csv → reward should increase

# Evaluate against test opponents
python train_agent.py --mode=eval --agent_path=checkpoints/.../agent.zip

# Visualize strategy latents
python train_agent.py --mode=analyze_strategy --agent_path=checkpoints/.../agent.zip

# Test single match (real-time visualization)
python train_agent.py --mode=demo --agent_path=checkpoints/.../agent.zip
```

### File Outputs

```
checkpoints/competition_10M_final/
├── agent_50000.zip               # Checkpoint @ 50k steps
├── agent_50000_transformer_encoder.pth  # Transformer weights (separate)
├── monitor.csv                   # Episode rewards, lengths
├── behavior_metrics.csv          # Damage, win rate, knockouts
├── reward_breakdown.csv          # Individual reward terms
├── strategy_analysis.csv         # Strategy latent statistics
└── training_curves.png           # Plots
```

### Common Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| `CUDA out of memory` | Batch too large | Reduce `batch_size` or `n_steps` |
| `Loss is NaN` | Learning rate too high | Reduce `learning_rate` to 1e-4 |
| `No module named 'sb3_contrib'` | Missing dependency | `pip install sb3-contrib` |
| `TypeError: unsupported operand type(s) for *: 'function' and 'Tensor'` | `ent_coef` is callable | Change to float (e.g., 0.01) |
| `RuntimeError: mat1 and mat2 shapes cannot be multiplied` | Observation shape mismatch | Check `opponent_obs_dim` matches environment |

---

## Summary

This training system implements a **transformer-based strategy recognition agent** trained via **self-play** and **curriculum learning** using **RecurrentPPO**.

**Key innovations:**
1. **Transformer encoder:** Recognizes opponent strategies from temporal patterns
2. **LSTM policy:** Maintains agent's action coherence
3. **Self-adversarial training:** Agent plays against past versions
4. **Curriculum learning:** Easy → hard opponent progression
5. **End-to-end learning:** Transformer and policy trained jointly

**Expected training time (T4 GPU):**
- Curriculum Stage 1: 15 min (50k steps)
- Curriculum Stage 2: 15 min (50k steps)
- Full training: 10-12 hours (10M steps)

**Expected performance:**
- vs Scripted bots: 90-95% win rate
- vs Self-play: 60-70% win rate (balanced pool)

**Most common issues:**
1. Sparse rewards + poor exploration → Use curriculum, increase `ent_coef`
2. Overfitting to self-play → Increase scripted bot weight
3. Training instability → Reduce `learning_rate`, increase `n_steps`
4. GPU memory → Reduce `batch_size`, `sequence_length`

**Where to start debugging:**
1. Check `monitor.csv` (is reward increasing?)
2. Check `behavior_metrics.csv` (is win rate increasing?)
3. Check `reward_breakdown.csv` (which rewards dominate?)
4. Visualize strategy latents (are clusters forming?)
5. Test against held-out opponents (generalization check)

Good luck with training! 🚀
