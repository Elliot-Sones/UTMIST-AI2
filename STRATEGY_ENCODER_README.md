# Strategy-Conditioned Training with Population-Based Self-Play

This system implements an advanced self-adversarial training approach where the agent:
1. **Learns to recognize opponent strategies** through a lightweight 1D CNN encoder
2. **Adapts its playstyle in real-time** using strategy embeddings
3. **Trains against a diverse population** of past agents with different strategies
4. **Maintains robustness** by training against weak/weird agents

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING SYSTEM                          │
└─────────────────────────────────────────────────────────────┘

┌──────────────────────┐         ┌──────────────────────┐
│   Agent Observations │         │   Opponent History   │
│   (52D: pos, state)  │         │   (60 × 13 = 780D)   │
└──────────┬───────────┘         └──────────┬───────────┘
           │                                │
           v                                v
┌──────────────────────┐         ┌──────────────────────┐
│  Feature Extractor   │         │  Strategy Encoder    │
│  (ResidualMLP)       │         │  (1D CNN)            │
│  → 512D features     │         │  → 32D embedding     │
└──────────┬───────────┘         └──────────┬───────────┘
           │                                │
           └────────────┬───────────────────┘
                        │ Concatenate (544D)
                        v
              ┌─────────────────────┐
              │   3-Layer LSTM      │
              │   (512 units/layer) │
              └──────────┬──────────┘
                         │
                ┌────────┴────────┐
                v                 v
        ┌───────────────┐  ┌──────────────┐
        │  Policy Head  │  │  Value Head  │
        │  → Actions    │  │  → V(s)      │
        └───────────────┘  └──────────────┘

┌─────────────────────────────────────────────────────────────┐
│             POPULATION-BASED SELF-PLAY                      │
└─────────────────────────────────────────────────────────────┘

Population (15 agents):
  - 10 strong diverse strategies
  - 3 weak/weird agents (50k, 150k, 300k steps)
  - 2 recent strong performers

Opponent Sampling:
  - 70% from population (weighted by diversity)
  - 30% from scripted agents (baseline skills)
  - 10% with random action noise (robustness)

Population Updates (every 100k steps):
  - Evaluate current agent
  - Add if win_rate > 55% AND strategy is novel
  - Prune least diverse agents if population full
  - Always keep weak agents
```

## Key Components

### 1. Strategy Encoder (`user/models/strategy_encoder.py`)
- **Input:** Last 60 frames of opponent observations (position, velocity, move type, etc.)
- **Architecture:** 3-layer 1D CNN with batch norm and dropout
- **Output:** 32D strategy embedding capturing opponent playstyle
- **Parameters:** ~100K (lightweight, minimal overhead)

### 2. Opponent-Conditioned Policy (`user/models/opponent_conditioned_policy.py`)
- Extends RecurrentPPO with strategy conditioning
- Processes agent obs + opponent history in parallel
- Concatenates features before LSTM
- Allows policy to adapt based on detected strategy

### 3. Population Manager (`user/self_play/population_manager.py`)
- Maintains diverse pool of past agents
- Computes diversity metrics (strategy embedding distance)
- Keeps balance of strong/weak/recent agents
- Saves/loads population to disk

### 4. Diverse Opponent Sampler (`user/self_play/diverse_opponent_sampler.py`)
- Samples opponents from population with diversity weighting
- Falls back to scripted opponents when needed
- Adds random noise for robustness
- Tracks sampling statistics

### 5. Population Update Callback (`user/self_play/population_update_callback.py`)
- Runs during training every 100k steps
- Evaluates current agent and adds to population if diverse
- Force-adds weak agents at 50k, 150k, 300k steps
- Automatically manages population pruning

## How to Use

### Quick Start

```bash
# Run training with default settings
python user/train_with_strategy_encoder.py
```

### Training Configuration

Edit `user/train_with_strategy_encoder.py` to customize:

```python
# Strategy encoder settings
STRATEGY_ENCODER_CONFIG = {
    'input_features': 13,      # Opponent feature dim
    'history_length': 60,      # Timesteps to track (2 seconds)
    'embedding_dim': 32,       # Strategy embedding size
    'dropout': 0.1,
}

# Population settings
POPULATION_CONFIG = {
    "max_population_size": 15,       # Max agents in population
    "num_weak_agents": 3,            # Weak agents for robustness
    "update_frequency": 100_000,     # Update every N steps
    "noise_probability": 0.10,       # 10% opponents get noise
    "use_population_prob": 0.70,     # 70% population, 30% scripted
}

# Training settings
TRAINING_CONFIG = {
    "total_timesteps": 5_000_000,  # Extended for diverse strategies
    "save_freq": 50_000,
    "n_envs": 4,                   # Parallel environments
}
```

### GPU Requirements

- **Recommended:** NVIDIA 4090 (24GB VRAM) or equivalent
- **Memory usage:** ~6-8GB during training
- **Batch size:** 1024 fits comfortably
- **Training time:** ~8-10 hours for 5M steps on 4090

### Monitoring Training

```bash
# View TensorBoard logs
tensorboard --logdir=checkpoints/strategy_encoder_training/tb_logs

# Monitor population
# Population stats are printed every 100k steps
```

### Resume Training

The script automatically resumes from the latest checkpoint if available:
- Looks for `checkpoints/strategy_encoder_training/latest_model.zip`
- Loads population from `checkpoints/strategy_encoder_training/population/population.json`

## Expected Outcomes

After training, the agent should:
1. **Adapt playstyle within 5-10 seconds** of episode start
2. **Handle 20+ distinct opponent strategies** from population
3. **Win >70% against unseen scripted opponents**
4. **Maintain diverse action distributions** (entropy >1.5)
5. **Be robust to weird/random behaviors**

## Evaluation

To evaluate the trained agent:

```python
# Load trained model
from sb3_contrib import RecurrentPPO

model = RecurrentPPO.load(
    "checkpoints/strategy_encoder_training/final_model.zip"
)

# Test against specific opponent
# (Evaluation script coming soon)
```

## File Structure

```
user/
├── models/
│   ├── strategy_encoder.py            # 1D CNN encoder
│   └── opponent_conditioned_policy.py # Custom policy
├── wrappers/
│   ├── opponent_history_wrapper.py    # Tracks opponent behavior
│   └── augmented_obs_wrapper.py       # Adds history to obs
├── self_play/
│   ├── population_manager.py          # Diverse agent pool
│   ├── diverse_opponent_sampler.py    # Opponent sampling
│   └── population_update_callback.py  # Training callback
└── train_with_strategy_encoder.py     # Main training script
```

## Troubleshooting

### Issue: OOM (Out of Memory)

**Solution:** Reduce batch size in AGENT_CONFIG:
```python
"batch_size": 512,  # Down from 1024
```

### Issue: Training too slow

**Solution:** Reduce number of parallel environments:
```python
"n_envs": 2,  # Down from 4
```

### Issue: Population not growing

**Check:**
- Win rate must be >55% to add to population
- Strategy must be diverse (embedding distance > 0.1)
- Wait until 200k steps (min_timesteps_before_add)

### Issue: Agent too conservative

**Solution:** Increase exploration:
```python
"ent_coef": 0.03,  # Up from 0.02
```

## Advanced: Extracting Strategy Embeddings

To visualize learned strategies:

```python
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Extract embeddings from population
embeddings = []
for member in population_manager.population:
    if member.strategy_embedding is not None:
        embeddings.append(member.strategy_embedding)

embeddings = np.array(embeddings)

# Reduce to 2D with t-SNE
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# Plot
plt.figure(figsize=(10, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
plt.title("Strategy Embedding Space")
plt.xlabel("t-SNE Dim 1")
plt.ylabel("t-SNE Dim 2")
plt.show()
```

## Comparison with Original Training

| Feature | Original (train_simplified_OPTIMIZED.py) | Strategy Encoder System |
|---------|------------------------------------------|-------------------------|
| **Opponent Strategy Recognition** | ❌ Implicit only | ✅ Explicit 32D embedding |
| **Real-time Adaptation** | ❌ Slow LSTM learning | ✅ Fast strategy conditioning |
| **Self-Play Diversity** | ⚠ Random past checkpoints | ✅ Population-based (15 diverse agents) |
| **Robustness Training** | ❌ None | ✅ Weak agents + noise |
| **Strategy Embedding Space** | ❌ N/A | ✅ 32D latent space |
| **Population Management** | ❌ Manual | ✅ Automatic pruning + diversity scoring |
| **Handles Unseen Strategies** | ⚠ Limited | ✅ Strategy similarity matching |

## Technical Details

### Why 1D CNN for Strategy Encoding?

- **Temporal patterns:** Detects sequences like "dash → attack → retreat"
- **Efficient:** ~100K parameters, minimal overhead
- **Translation invariant:** Recognizes patterns regardless of timing
- **Fast inference:** Single forward pass per step

### Why Population-Based Self-Play?

- **Diversity:** Prevents convergence to single strategy
- **Robustness:** Weak agents prevent overfitting to optimal play
- **Continual learning:** New strategies added as agent improves
- **Quality diversity:** Balance between strength and novelty

### Opponent History Features (13D)

The encoder tracks these opponent features:
1. Position (x, y) - 2D
2. Velocity (x, y) - 2D
3. Facing direction - 1D
4. Grounded/Aerial status - 2D
5. Jumps remaining - 1D
6. Damage taken - 1D
7. Stocks remaining - 1D
8. Current move type - 1D
9. Current state - 1D
10. Stun frames - 1D

Total: 13 features tracked over 60 timesteps (2 seconds)

## Future Enhancements

Potential improvements:
1. **Transformer encoder** instead of CNN (better long-range dependencies)
2. **Contrastive loss** to explicitly separate strategy clusters
3. **Online evaluation** during training for accurate win rates
4. **Strategy conditioning visualization** in TensorBoard
5. **Multi-agent training** (3+ players)
6. **Hierarchical strategies** (macro + micro level)

## Credits

Built on:
- **Stable-Baselines3** for RecurrentPPO implementation
- **WarehouseBrawl** environment from UTMIST AI²
- **Population-Based Training** concepts from DeepMind

---

For questions or issues, please refer to the main project documentation or create an issue on GitHub.
