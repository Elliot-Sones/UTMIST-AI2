# Why Is My Model Not Learning? - Complete Checklist

## 🔍 Run This First

```bash
python user/debug_training.py
```

This will test 8 critical systems. Review ALL warnings.

---

## 20 Reasons Why RL Models Fail to Learn

### **CATEGORY 1: Environment Issues** 🎮

#### 1. **Observations Don't Change**
**Symptom**: Model gets same observation regardless of actions
**Check**: Run debug script, look at "Observations vary appropriately"
**Fix**: Verify environment step() function updates state

#### 2. **Actions Have No Effect**
**Symptom**: Random actions produce same outcome as zero action
**Check**: Debug script shows "Actions don't affect observations"
**Fix**: Check if actions are being passed to underlying environment

#### 3. **Environment Not Deterministic**
**Symptom**: Same seed + same actions = different results
**Check**: Debug script "Environment determinism" test
**Fix**: Ensure environment respects seed, no uncontrolled randomness

#### 4. **Observation Space Too Large/Small**
**Symptom**: 1000+ dimensions, or missing critical info
**Check**: Print observation shape
**Fix**: Reduce dimensions (PCA/feature selection) or add missing features

#### 5. **Observations Contain NaN/Inf**
**Symptom**: Crash, or model outputs NaN
**Check**: Debug script checks for NaN/Inf
**Fix**: Add bounds/clipping in observation function

---

### **CATEGORY 2: Reward Signal Issues** 🎯

#### 6. **All Rewards Are Zero**
**Symptom**: Model never gets feedback
**Check**: Debug script shows "All rewards are zero"
**Fix**: Verify reward_manager is connected, functions are called

#### 7. **Rewards Too Sparse** (<5% non-zero)
**Symptom**: Model gets reward once per 1000 steps
**Check**: Debug script shows ">95% of rewards are zero"
**Fix**: Add dense shaping rewards (distance, positioning, etc.)

#### 8. **Reward Scale Wrong** (Too large/small)
**Symptom**: Rewards in range [0.0001, 0.0002] or [10000, 20000]
**Check**: Print reward statistics
**Fix**: Scale rewards to roughly [-10, +10] range

#### 9. **Reward Components Imbalanced**
**Symptom**: One reward dominates all others
**Check**: Look at component breakdown in debug script
**Fix**: Adjust weights so no single component is >80% of total

####10. **Reward Not Aligned with Goal**
**Symptom**: High reward doesn't mean good performance
**Check**: Manually check: does high episode reward = good play?
**Fix**: Redesign reward to match actual objective

---

### **CATEGORY 3: Network Architecture Issues** 🧠

#### 11. **Vanishing Gradients**
**Symptom**: Gradients < 1e-6, model doesn't update
**Check**: Debug script gradient flow test
**Fix**:
- Use ReLU/GELU instead of sigmoid/tanh
- Enable orthogonal initialization
- Reduce network depth
- Increase learning rate

#### 12. **Exploding Gradients**
**Symptom**: Loss becomes NaN, gradients > 100
**Check**: Monitor gradient norms in training
**Fix**:
- Reduce learning rate
- Lower max_grad_norm (try 0.5)
- Add gradient clipping

#### 13. **Network Too Small** (Can't represent policy)
**Symptom**: Loss plateaus high, poor performance
**Check**: Compare network capacity to task complexity
**Fix**: Increase layer sizes, add more layers

#### 14. **Network Too Large** (Overfits, slow learning)
**Symptom**: Perfect train performance, terrible eval
**Check**: Train vs eval performance gap
**Fix**: Reduce network size, add dropout/regularization

#### 15. **LSTM States Not Reset**
**Symptom**: Model "remembers" across episode boundaries
**Check**: Verify episode_start flags are set
**Fix**: Ensure RecurrentPPO gets proper done signals

---

### **CATEGORY 4: Hyperparameter Issues** ⚙️

#### 16. **Learning Rate Wrong**
**Symptom**: No learning (too low) or divergence (too high)
**Check**:
- Too low: loss barely decreases
- Too high: loss oscillates wildly
**Fix**:
- Start with 3e-4
- If no learning after 50k steps → increase 10x
- If NaN/divergence → decrease 10x

#### 17. **Batch Size Wrong**
**Symptom**: Noisy updates (too small) or slow learning (too large)
**Check**: Monitor policy/value loss variance
**Fix**:
- Small batch (<256): increase for stability
- Large batch (>2048): decrease for faster updates

#### 18. **Entropy Too Low** (No exploration)
**Symptom**: Model converges to first working strategy
**Check**: Entropy loss near 0, deterministic actions
**Fix**:
- Increase ent_coef to 0.01-0.03
- Slower entropy decay
- **YOUR ISSUE**: ent_coef=0.005 is too low!

#### 19. **Gamma Too High/Low** (Credit assignment)
**Symptom**: Ignores long-term (low) or short-term (high) rewards
**Check**: Episode length vs gamma
**Fix**:
- Short episodes (<100 steps): gamma=0.95-0.98
- Long episodes (>500 steps): gamma=0.99-0.995

#### 20. **Update Constraints Too Tight**
**Symptom**: Model can't make meaningful policy changes
**Check**: KL divergence hitting target_kl every update
**Fix**:
- Increase target_kl (0.01 → 0.03)
- Increase clip_range (0.1 → 0.2)
- **YOUR ISSUE**: target_kl=0.015 + clip→0.05 is too restrictive!

---

## 🔧 Specific Issues for Your Code

Based on your symptoms, here are the most likely culprits:

### Issue A: **VecNormalize Breaking Things**
**Symptom**: Rewards decrease over time, explained variance drops
**Cause**: VecNormalize running statistics diverge

**Test**:
```python
# In your training, after some steps:
print(f"Reward mean: {vec_env.ret_rms.mean}")
print(f"Reward std: {vec_env.ret_rms.var**0.5}")
print(f"Obs mean: {vec_env.obs_rms.mean[:5]}")
print(f"Obs std: {vec_env.obs_rms.var[:5]**0.5}")
```

**Fix**:
```python
# Option 1: Disable reward normalization
vec_env = VecNormalize(
    vec_env,
    norm_obs=True,
    norm_reward=False,  # DISABLE THIS
    clip_obs=5.0,
)

# Option 2: Less aggressive clipping
vec_env = VecNormalize(
    vec_env,
    norm_obs=True,
    norm_reward=True,
    clip_obs=10.0,  # Increase
    clip_reward=30.0,  # Increase (to not clip win reward)
    gamma=AGENT_CONFIG["gamma"],
)
```

### Issue B: **Win Reward Getting Clipped**
**Symptom**: Win reward = 100, but `clip_reward=10` cuts it to 10
**Result**: Winning gives same reward as dealing 100 damage

**Fix**: Increase clip_reward to at least 150:
```python
vec_env = VecNormalize(
    ...,
    clip_reward=150.0,  # Don't clip win rewards!
)
```

### Issue C: **Observation Normalization Unstable**
**Symptom**: Observations become all zeros or all same value
**Cause**: Running mean/var not initialized properly

**Fix**:
```python
# After creating vec_env, collect some samples first:
print("Collecting initial samples for normalization...")
vec_env.reset()
for _ in range(1000):
    actions = np.array([vec_env.action_space.sample() for _ in range(vec_env.num_envs)])
    vec_env.step(actions)
vec_env.reset()
print("Normalization initialized!")
```

### Issue D: **LSTM States Corrupted**
**Symptom**: Value function predicts nonsense, explained variance < 0
**Cause**: LSTM hidden states from one episode leak into next

**Test**:
```python
# Check if model is resetting LSTM states properly
# In TrainingMonitor._on_step():
if hasattr(self.model, 'policy'):
    lstm_states = self.model.policy._last_lstm_states
    print(f"LSTM state norms: {[s.norm().item() for s in lstm_states.lstm_states]}")
```

**Fix**: Already handled by RecurrentPPO, but verify done signals work

### Issue E: **PPO Update Breaking Policy**
**Symptom**: Model works for 50k steps, then suddenly collapses
**Cause**: One bad PPO update destroys policy

**Signs**:
- Entropy suddenly drops to 0
- Policy loss spikes
- Value loss explodes

**Fix**:
```python
# Add to AGENT_CONFIG:
"target_kl": 0.05,  # Increase (was 0.035)
"clip_range": 0.25,  # Increase starting point
"clip_range_final": 0.15,  # Increase final
```

---

## 🚨 Emergency Diagnostic Procedure

If model STILL doesn't learn after everything above:

### Step 1: Simplify to Minimum Viable Agent

Create a test with ONLY ConstantAgent (easiest opponent):

```python
# test_minimal.py
from functools import partial
from environment.agent import *
from sb3_contrib import RecurrentPPO

# Minimal environment
reward_manager = gen_reward_manager()
opponent_cfg = OpponentsCfg(opponents={
    'constant_agent': (1.0, partial(ConstantAgent)),
})

env = SelfPlayWarehouseBrawl(
    reward_manager=reward_manager,
    opponent_cfg=opponent_cfg,
    save_handler=None,
    resolution=CameraResolution.LOW,
)

# Minimal model
model = RecurrentPPO(
    "MlpLstmPolicy",
    env,
    verbose=1,
    n_steps=512,
    batch_size=128,
    learning_rate=1e-3,  # HIGH LR
    ent_coef=0.05,  # HIGH ENTROPY
    policy_kwargs={
        "lstm_hidden_size": 128,
        "n_lstm_layers": 1,
        "net_arch": dict(pi=[128], vf=[128]),
    },
    device="cpu",
)

print("Training minimal agent vs ConstantAgent...")
print("Should reach 90%+ win rate in 50k steps")

model.learn(total_timesteps=50_000)

# Test
from environment.agent import RecurrentPPOAgent, run_match, Result
model.save("test_minimal.zip")
agent = RecurrentPPOAgent(file_path="test_minimal.zip")
wins = 0
for _ in range(10):
    stats = run_match(agent, partial(ConstantAgent), max_timesteps=1800, train_mode=False)
    if stats.player1_result == Result.WIN:
        wins += 1
print(f"Win rate: {wins}/10")

if wins < 8:
    print("❌ CRITICAL: Can't even beat ConstantAgent!")
    print("   → Environment or reward function is fundamentally broken")
else:
    print("✓ Basic learning works. Issue is with full training setup.")
```

### Step 2: Check Individual Reward Components

Test each reward function in isolation:

```python
# Test damage reward
env.reset()
for _ in range(100):
    # Action that should hit opponent
    action = np.array([...])  # Design action
    obs, reward, done, _, _ = env.step(action)
    print(f"Reward: {reward:.3f}")
```

### Step 3: Visualize What Model Sees

```python
# Save observations to file
import matplotlib.pyplot as plt

env.reset()
observations = []
for _ in range(100):
    action = model.predict(obs)[0]
    obs, reward, done, _, _ = env.step(action)
    observations.append(obs)

    if done:
        break

obs_array = np.array(observations)

# Plot each observation dimension over time
plt.figure(figsize=(15, 10))
for i in range(min(20, obs_array.shape[1])):
    plt.subplot(4, 5, i+1)
    plt.plot(obs_array[:, i])
    plt.title(f'Obs dim {i}')
plt.savefig('observations.png')
print("Saved observations.png - check if they make sense!")
```

---

## ✅ Checklist Before Training

- [ ] Run `python user/debug_training.py` - all tests pass
- [ ] Observations change meaningfully with actions
- [ ] Rewards are non-zero and vary (not all same)
- [ ] Random policy can beat ConstantAgent at least 50% of time
- [ ] Feature extractor passes gradient flow test
- [ ] VecNormalize clip_reward > max reward (e.g., 150 > 100)
- [ ] Entropy coef >= 0.01
- [ ] Learning rate = 3e-4 (not too low/high)
- [ ] Target KL >= 0.03 (not too restrictive)
- [ ] Evaluation runs every 20k steps to track progress

---

## 📊 What Good Training Looks Like

### First 100k steps:
```
Step 20k  | Reward: +2.3  | Entropy: -0.45 | WR: 35% | ExplainedVar: 0.62
Step 40k  | Reward: +4.1  | Entropy: -0.38 | WR: 48% | ExplainedVar: 0.68
Step 60k  | Reward: +5.8  | Entropy: -0.32 | WR: 55% | ExplainedVar: 0.71
Step 80k  | Reward: +7.2  | Entropy: -0.28 | WR: 62% | ExplainedVar: 0.74
Step 100k | Reward: +8.5  | Entropy: -0.24 | WR: 67% | ExplainedVar: 0.76
```

**Key indicators**:
- ✓ Reward steadily increasing
- ✓ Entropy slowly decreasing (not dropping to 0)
- ✓ Win rate improving
- ✓ Explained variance >0.6

### What BAD training looks like:
```
Step 20k  | Reward: +8.2  | Entropy: -0.02 | WR: 100%/0% | ExplainedVar: 0.31
Step 40k  | Reward: +6.1  | Entropy: -0.01 | WR: 100%/0% | ExplainedVar: -0.15
Step 60k  | Reward: +3.5  | Entropy: -0.00 | WR: 100%/0% | ExplainedVar: -0.43
```

**Red flags**:
- ❌ Reward decreasing (getting worse!)
- ❌ Entropy near 0 (stopped exploring)
- ❌ Win rates exactly 0% or 100% (deterministic)
- ❌ Explained variance negative (critic broken)

---

## 🎯 My Top 5 Guesses for Your Issue

Based on "model really not learning":

1. **VecNormalize clipping win rewards** (100 → 10) - VERY LIKELY
2. **Entropy collapsed too fast** (ent_coef too low) - CONFIRMED ISSUE
3. **Observation normalization unstable** - LIKELY
4. **Reward components imbalanced** - POSSIBLE
5. **PPO updates too conservative** (can't escape local minimum) - LIKELY

**Recommended immediate fixes**:
```python
# 1. Don't clip win rewards
clip_reward=150.0  # (was 10.0)

# 2. Higher exploration
ent_coef=0.03  # (was 0.02)

# 3. More flexible updates
target_kl=0.05  # (was 0.035)
clip_range_final=0.15  # (was 0.1)

# 4. Initialize normalization
# Add warmup period before training

# 5. Monitor these metrics
# - Entropy loss (should stay < -0.1)
# - Explained variance (should be > 0.5)
# - Reward before/after normalization
```

---

## 🔬 Advanced Debugging

If nothing above works, there might be a bug in:
- Your custom feature extractor
- Reward manager signal subscription
- LSTM state handling
- Environment observation function
- VecEnv wrapper chain

Run the debug script, save output, and review EVERY warning carefully!
