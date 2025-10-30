'''
TRAINING: AGENT - COMPLETE TOURNAMENT-OPTIMIZED VERSION

This file contains all Agent classes, Reward Function API, and training configuration
optimized for AI Squared tournament competition.

Key improvements:
- Fixed reward functions with proper signatures
- Added strategic combat rewards (combos, spacing, adaptation)
- Rebalanced weights for aggressive tournament play
- Increased self-play diversity
- Added curriculum learning capability
'''

# -------------------------------------------------------------------
# ----------------------------- IMPORTS -----------------------------
# -------------------------------------------------------------------

import multiprocessing as mp

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed


from enum import Enum
from typing import Optional, Tuple
from functools import partial
from stable_baselines3.common.callbacks import BaseCallback


import torch 
import gymnasium as gym
from torch.nn import functional as F
from torch import nn as nn
import numpy as np
import pygame
from stable_baselines3 import A2C, PPO, SAC, DQN, DDPG, TD3, HER 
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from environment.agent import *
from typing import Optional, Type, List, Tuple
from functools import partial

import csv
import os
from datetime import datetime

'''
TRAINING: AGENT - GPU + MULTI-CORE OPTIMIZED VERSION

Optimizations:
- Multi-process vectorized environments (uses all CPU cores)
- GPU-optimized batch sizes and buffer sizes
- Parallel environment rollouts
- Efficient data transfer between CPU/GPU
- Mixed precision training support
- Optimized PPO hyperparameters for parallel training
'''



# -------------------------------------------------------------------------
# ----------------------- ENVIRONMENT UTILITIES ---------------------------
# -------------------------------------------------------------------------

def make_env(env_class, rank: int, seed: int = 0, **env_kwargs):
    """
    Utility function for multiprocessed env.
    
    :param env_class: the environment class or callable
    :param rank: index of the subprocess
    :param seed: the initial seed for RNG
    """
    def _init():
        if callable(env_class):
            env = env_class(**env_kwargs)
        else:
            env = env_class
        if hasattr(env, 'reset'):
            env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


def create_vectorized_env_from_instance(base_env, n_envs: int, seed: int = 0):
    """
    Create a vectorized environment from an existing environment instance.
    
    :param base_env: existing environment instance
    :param n_envs: number of parallel environments
    :param seed: base seed for RNG
    :return: vectorized environment
    """
    if n_envs == 1:
        # Use DummyVecEnv for single environment (no multiprocessing overhead)
        return DummyVecEnv([lambda: base_env])
    else:
        # For multiple environments, try to recreate similar envs
        try:
            # Try to get environment creation info
            if hasattr(base_env, 'spec') and base_env.spec is not None:
                env_id = base_env.spec.id
                return SubprocVecEnv([make_env(lambda: gym.make(env_id), i, seed) 
                                     for i in range(n_envs)])
            else:
                # Fallback: use single env with DummyVecEnv
                print(f"⚠️  Cannot create multiple environments, using single env")
                return DummyVecEnv([lambda: base_env])
        except Exception as e:
            print(f"⚠️  Error creating vectorized env: {e}, using single env")
            return DummyVecEnv([lambda: base_env])


# -------------------------------------------------------------------------
# ----------------------------- AGENT CLASSES -----------------------------
# -------------------------------------------------------------------------

class OptimizedSB3Agent(Agent):
    """
    Optimized SB3 Agent with multi-core CPU and GPU support.
    Compatible with existing training framework.
    
    Key optimizations:
    - Vectorized environments for parallel rollouts (when possible)
    - GPU-optimized hyperparameters
    - Efficient data transfer
    - Optional mixed precision training
    """
    
    def __init__(
        self,
        sb3_class: Optional[Type[BaseAlgorithm]] = PPO,
        file_path: Optional[str] = None,
        device: str = 'auto',
        n_envs: Optional[int] = None,
        use_vectorized: bool = False,  # Set to True to enable multi-env (may not work with all frameworks)
        use_mixed_precision: bool = False,
        **model_kwargs
    ):
        """
        Initialize the optimized agent.
        
        :param sb3_class: SB3 algorithm class
        :param file_path: path to load existing model
        :param device: 'cuda', 'cpu', or 'auto'
        :param n_envs: number of parallel environments (only used if use_vectorized=True)
        :param use_vectorized: enable multi-process environments (experimental with some frameworks)
        :param use_mixed_precision: enable mixed precision training (faster on modern GPUs)
        :param model_kwargs: additional arguments for the model
        """
        self.sb3_class = sb3_class
        self.use_mixed_precision = use_mixed_precision
        self.use_vectorized = use_vectorized
        self.model_kwargs = model_kwargs
        self._model_initialized = False
        
        # Auto-detect device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Auto-detect CPU cores for parallel environments
        if n_envs is None:
            cpu_count = mp.cpu_count()
            # Use CPU count - 1 to leave one core free, min 1, max 16 for stability
            self.n_envs = min(max(1, cpu_count - 1), 16) if use_vectorized else 1
        else:
            self.n_envs = n_envs if use_vectorized else 1
            
        print(f"🚀 Agent Initialization:")
        print(f"   Device: {self.device}")
        if self.device == 'cpu':
            print(f"   ⚠️  Training on CPU - consider using GPU for 3-5x speedup")
            print(f"   💡 To use GPU: Ensure CUDA is installed and torch.cuda.is_available() returns True")
        if use_vectorized:
            print(f"   Vectorized Envs: {self.n_envs} (experimental)")
        else:
            print(f"   Single Environment Mode (stable, recommended)")
        print(f"   CPU Cores Available: {mp.cpu_count()}")
        if self.device == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        super().__init__(file_path)

    def _initialize(self) -> None:
        """
        Initialize method called by parent Agent class.
        For framework compatibility, actual initialization happens in get_env_info().
        """
        if self.file_path is not None:
            # Load existing model
            self.model = self.sb3_class.load(self.file_path, device=self.device)
            self._model_initialized = True
            print(f"✓ Model loaded from {self.file_path}\n")
        # For new models, wait for get_env_info() to be called by framework

    def _gdown(self) -> str:
        return

    def get_env_info(self, env):
        """
        Called by training framework to set up environment.
        This is where we actually initialize the model.
        """
        # Store the environment
        self.env = env
        
        # Call parent's method if it exists
        try:
            super().get_env_info(env)
        except AttributeError:
            pass  # Parent doesn't have this method
        
        # Initialize model if not already done
        if not self._model_initialized:
            self._initialize_model()

    def _initialize_model(self):
        """
        Actually initialize the model with the environment.
        """
        if not hasattr(self, 'env') or self.env is None:
            print("⚠️  No environment set yet, deferring model initialization")
            return
        
        base_env = self.env
        
        # Decide on vectorization
        if self.use_vectorized and self.n_envs > 1:
            try:
                print(f"   Attempting to create {self.n_envs} parallel environments...")
                vec_env = create_vectorized_env_from_instance(base_env, self.n_envs)
                
                # Check if we actually got multiple environments
                if isinstance(vec_env, SubprocVecEnv):
                    actual_n_envs = self.n_envs
                    print(f"✓ Successfully created {actual_n_envs} parallel environments")
                    print(f"   🚀 This will speed up data collection by ~{actual_n_envs}x")
                else:
                    actual_n_envs = 1
                    print(f"⚠️  Vectorization not available for this environment")
                    print(f"   Using single environment (still optimized!)")
            except Exception as e:
                print(f"⚠️  Could not create parallel environments: {e}")
                print(f"   Using single environment (still optimized!)")
                vec_env = DummyVecEnv([lambda: base_env])
                actual_n_envs = 1
        else:
            # Standard single environment wrapped for compatibility
            vec_env = DummyVecEnv([lambda: base_env])
            actual_n_envs = 1
            if self.device == 'cpu':
                print(f"   💡 Tip: For multi-core CPU training, set use_vectorized=True")
        
        # Update actual number of environments
        self.n_envs = actual_n_envs
        
        if self.device == 'cuda':
            # 64-core CPU + A2000 GPU
            actual_n_envs = 64  # 1 env per CPU core
            n_steps = 2048      # stable large rollout buffer
            batch_size = 2048   # keeps GPU busy but fits 6GB VRAM
            n_epochs = 8        # good tradeoff for PPO stability
            learning_rate = 2.5e-4
            ent_coef = 0.015

            print(f"✓ Optimized for 64-core CPU + A2000 GPU")
        else:
            # CPU-only fallback
            actual_n_envs = 16
            n_steps = 1024
            batch_size = 256
            n_epochs = 4
            learning_rate = 3e-4
            ent_coef = 0.05
            print(f"✓ CPU-only configuration")

        total_buffer_size = n_steps * actual_n_envs
        print(f"   Steps per update: {n_steps}")
        print(f"   Batch size: {batch_size}")
        print(f"   Training epochs: {n_epochs}")
        print(f"   Buffer size: {total_buffer_size}")

        default_config = {
            "policy": "MlpPolicy",
            "env": vec_env,
            "verbose": 1,
            "device": self.device,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "learning_rate": learning_rate,
            "ent_coef": ent_coef,
            "max_grad_norm": 0.5,
            "normalize_advantage": True,
        }

        
        # Merge with user-provided kwargs
        default_config.update(self.model_kwargs)
        
        # Create model
        try:
            self.model = self.sb3_class(**default_config)
            self._model_initialized = True
            
            # Enable mixed precision if requested
            if self.use_mixed_precision and self.device == 'cuda':
                try:
                    from torch.cuda.amp import autocast, GradScaler
                    self.model.policy.use_amp = True
                    print("   Mixed Precision: ENABLED")
                except ImportError:
                    print("   Mixed Precision: NOT AVAILABLE (requires PyTorch 1.6+)")
            
            print(f"✓ Model initialized successfully")
            
            # Store vectorized env
            self.vec_env = vec_env
            
        except Exception as e:
            print(f"❌ Error initializing model: {e}")
            raise

    def predict(self, obs, deterministic: bool = True):
        """
        Predict action for single observation.
        """
        if not hasattr(self, 'model') or self.model is None:
            raise RuntimeError("Model not initialized. Call get_env_info() first.")
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action

    def save(self, file_path: str) -> None:
        """Save model to file."""
        if not hasattr(self, 'model') or self.model is None:
            raise RuntimeError("Model not initialized. Nothing to save.")
        self.model.save(file_path)
        print(f"✓ Model saved to {file_path}")

    def learn(
        self, 
        env,
        total_timesteps: int,
        verbose: int = 1,
        callback: Optional[BaseCallback] = None,
        log_interval: int = 10,
        tb_log_name: str = "run",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False
    ):
        """
        Train the agent with optimized settings.
        Compatible with framework's train() function signature.
        """
        if not hasattr(self, 'model') or self.model is None:
            raise RuntimeError("Model not initialized. Call get_env_info() first.")
        
        # Set the environment
        self.model.set_env(env)
        
        # Set verbosity
        self.model.verbose = verbose
        
        if verbose > 0:
            print(f"\n🎓 Training for {total_timesteps:,} timesteps...")
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar
        )
        
        if verbose > 0:
            print(f"✓ Training batch complete")


# Backward compatibility: keep original class name
class SB3Agent(OptimizedSB3Agent):
    """
    Drop-in replacement for original SB3Agent with optimizations.
    - GPU: Automatic GPU acceleration with optimized hyperparameters
    - CPU: Larger batch sizes and better training configuration
    - Optional: Multi-core parallelism (set ENABLE_MULTICORE=True in environment)
    """
    def __init__(
        self,
        sb3_class: Optional[Type[BaseAlgorithm]] = PPO,
        file_path: Optional[str] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        # Check for environment variable to enable multi-core
        enable_multicore = os.environ.get('ENABLE_MULTICORE', 'False').lower() == 'true'
        
        is_cpu = device == 'cpu' or (device == 'cuda' and not torch.cuda.is_available())
        
        # Call parent with optimized defaults
        super().__init__(
            sb3_class=sb3_class,
            file_path=file_path,
            device=device,
            use_vectorized=enable_multicore and is_cpu,  # Only if explicitly enabled
            n_envs=None if (enable_multicore and is_cpu) else 1,
            use_mixed_precision=False,  # Conservative default
        )
        
        if is_cpu:
            if enable_multicore and self.n_envs > 1:
                print(f"   🚀 Multi-core mode: Using {self.n_envs} parallel environments")
                print(f"   💡 Expected speedup: ~{self.n_envs}x for data collection")
            else:
                print(f"   💡 To enable multi-core: set environment variable ENABLE_MULTICORE=true")
                print(f"   💡 Or use OptimizedSB3Agent(use_vectorized=True) directly")


# -------------------------------------------------------------------------
# ------------------------- PERFORMANCE MONITOR ---------------------------
# -------------------------------------------------------------------------

class PerformanceMonitor(BaseCallback):
    """
    Callback to monitor CPU/GPU usage during training.
    """
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.start_time = None
        
    def _on_training_start(self):
        import datetime
        self.start_time = datetime.datetime.now()
        
    def _on_step(self):
        if self.n_calls % 10000 == 0:
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.memory_allocated() / 1e9
                gpu_cached = torch.cuda.memory_reserved() / 1e9
                print(f"   GPU Memory: {gpu_mem:.2f}GB used, {gpu_cached:.2f}GB cached")
        return True


class RecurrentPPOAgent(Agent):
    '''
    RecurrentPPOAgent:
    - Defines an RL Agent that uses the Recurrent PPO (LSTM+PPO) algorithm
    - Better for learning temporal patterns and combos
    '''
    def __init__(
            self,
            file_path: Optional[str] = None
    ):
        super().__init__(file_path)
        self.lstm_states = None
        self.episode_starts = np.ones((1,), dtype=bool)

    def _initialize(self) -> None:
        if self.file_path is None:
            policy_kwargs = {
                'activation_fn': nn.ReLU,
                'lstm_hidden_size': 512,
                'net_arch': [dict(pi=[64, 64], vf=[64, 64])],  # Increased capacity
                'shared_lstm': True,
                'enable_critic_lstm': False,
                'share_features_extractor': True,
            }
            self.model = RecurrentPPO("MlpLstmPolicy",
                                      self.env,
                                      verbose=0,
                                      n_steps=30*90*20,
                                      batch_size=32,  # Increased from 16
                                      ent_coef=0.03,  # Slightly increased exploration
                                      policy_kwargs=policy_kwargs)
            del self.env
        else:
            self.model = RecurrentPPO.load(self.file_path)

    def reset(self) -> None:
        self.episode_starts = True

    def predict(self, obs):
        action, self.lstm_states = self.model.predict(obs, state=self.lstm_states, episode_start=self.episode_starts, deterministic=True)
        if self.episode_starts: self.episode_starts = False
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path)

    def learn(self, env, total_timesteps, log_interval: int = 2, verbose=0):
        self.model.set_env(env)
        self.model.verbose = verbose
        self.model.learn(total_timesteps=total_timesteps, log_interval=log_interval)

class BasedAgent(Agent):
    '''
    BasedAgent:
    - Hard-coded agent for testing and as opponent
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time = 0

    def predict(self, obs):
        self.time += 1
        pos = self.obs_helper.get_section(obs, 'player_pos')
        opp_pos = self.obs_helper.get_section(obs, 'opponent_pos')
        opp_KO = self.obs_helper.get_section(obs, 'opponent_state') in [5, 11]
        action = self.act_helper.zeros()

        if pos[0] > 10.67/2:
            action = self.act_helper.press_keys(['a'])
        elif pos[0] < -10.67/2:
            action = self.act_helper.press_keys(['d'])
        elif not opp_KO:
            if (opp_pos[0] > pos[0]):
                action = self.act_helper.press_keys(['d'])
            else:
                action = self.act_helper.press_keys(['a'])

        if (pos[1] > 1.6 or pos[1] > opp_pos[1]) and self.time % 2 == 0:
            action = self.act_helper.press_keys(['space'], action)

        if (pos[0] - opp_pos[0])**2 + (pos[1] - opp_pos[1])**2 < 4.0:
            action = self.act_helper.press_keys(['j'], action)
        return action

class UserInputAgent(Agent):
    '''
    UserInputAgent:
    - For manual testing and gameplay
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict(self, obs):
        action = self.act_helper.zeros()
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]: action = self.act_helper.press_keys(['w'], action)
        if keys[pygame.K_a]: action = self.act_helper.press_keys(['a'], action)
        if keys[pygame.K_s]: action = self.act_helper.press_keys(['s'], action)
        if keys[pygame.K_d]: action = self.act_helper.press_keys(['d'], action)
        if keys[pygame.K_SPACE]: action = self.act_helper.press_keys(['space'], action)
        if keys[pygame.K_h]: action = self.act_helper.press_keys(['h'], action)
        if keys[pygame.K_j]: action = self.act_helper.press_keys(['j'], action)
        if keys[pygame.K_k]: action = self.act_helper.press_keys(['k'], action)
        if keys[pygame.K_l]: action = self.act_helper.press_keys(['l'], action)
        if keys[pygame.K_g]: action = self.act_helper.press_keys(['g'], action)
        return action

class ClockworkAgent(Agent):
    '''
    ClockworkAgent:
    - Performs sequential scripted actions
    '''
    def __init__(
            self,
            action_sheet: Optional[List[Tuple[int, List[str]]]] = None,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.steps = 0
        self.current_action_end = 0
        self.current_action_data = None
        self.action_index = 0

        if action_sheet is None:
            self.action_sheet = [
                (10, ['a']),
                (1, ['l']),
                (20, ['a']),
                (3, ['a', 'j']),
                (15, ['space']),
            ]
        else:
            self.action_sheet = action_sheet

    def predict(self, obs):
        if self.steps >= self.current_action_end and self.action_index < len(self.action_sheet):
            hold_time, action_data = self.action_sheet[self.action_index]
            self.current_action_data = action_data
            self.current_action_end = self.steps + hold_time
            self.action_index += 1

        action = self.act_helper.press_keys(self.current_action_data)
        self.steps += 1
        return action

# -------------------------------------------------------------------------
# ----------------------------- CUSTOM POLICY -----------------------------
# -------------------------------------------------------------------------

class MLPPolicy(nn.Module):
    def __init__(self, obs_dim: int = 64, action_dim: int = 10, hidden_dim: int = 128):
        """
        Stable 3-layer MLP policy with layer normalization
        """
        super(MLPPolicy, self).__init__()
        
        # Use smaller hidden dimension to prevent instability
        self.fc1 = nn.Linear(obs_dim, hidden_dim, dtype=torch.float32)
        self.ln1 = nn.LayerNorm(hidden_dim)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
        self.fc3 = nn.Linear(hidden_dim, action_dim, dtype=torch.float32)
        
        # Initialize weights properly to prevent exploding gradients
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier initialization for stable training"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)  # Small gain for stability
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, obs):
        x = self.fc1(obs)
        x = self.ln1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        x = self.ln2(x)
        x = F.relu(x)
        
        x = self.fc3(x)
        return x

class MLPExtractor(BaseFeaturesExtractor):
    '''
    Stable MLP Features Extractor with normalization
    '''
    def __init__(self, observation_space: gym.Space, features_dim: int = 128, hidden_dim: int = 128):
        super(MLPExtractor, self).__init__(observation_space, features_dim)
        self.model = MLPPolicy(
            obs_dim=observation_space.shape[0], 
            action_dim=features_dim,
            hidden_dim=hidden_dim,
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.model(obs)
    
    @classmethod
    def get_policy_kwargs(cls, features_dim: int = 64, hidden_dim: int = 64) -> dict:
        """
        Reduced dimensions for better stability
        """
        return dict(
            features_extractor_class=cls,
            features_extractor_kwargs=dict(features_dim=features_dim, hidden_dim=hidden_dim)
        )

class CustomAgent(Agent):
    '''
    Custom agent with configurable feature extractor and stable hyperparameters
    '''
    def __init__(self, sb3_class: Optional[Type[BaseAlgorithm]] = PPO, file_path: str = None, extractor: BaseFeaturesExtractor = None):
        self.sb3_class = sb3_class
        self.extractor = extractor
        super().__init__(file_path)
    
    def _initialize(self) -> None:
        if self.file_path is None:
            self.model = self.sb3_class(
                "MlpPolicy", 
                self.env, 
                policy_kwargs=self.extractor.get_policy_kwargs() if self.extractor else {}, 
                verbose=0, 
                n_steps=30*90*3, 
                batch_size=128,
                # Stability improvements
                ent_coef=0.01,  # Standard entropy coefficient
                learning_rate=3e-4,  # Standard learning rate
                clip_range=0.2,  # Standard PPO clip range
                max_grad_norm=0.5,  # Gradient clipping for stability
                vf_coef=0.5,  # Value function coefficient
                gae_lambda=0.95,  # GAE lambda
                normalize_advantage=True,  # Normalize advantages
            )
            del self.env
        else:
            self.model = self.sb3_class.load(self.file_path)

    def _gdown(self) -> str:
        return

    def predict(self, obs):
        action, _ = self.model.predict(obs)
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path, include=['num_timesteps'])

    def learn(self, env, total_timesteps, log_interval: int = 1, verbose=0):
        self.model.set_env(env)
        self.model.verbose = verbose
        self.model.learn(
            total_timesteps=total_timesteps,
            log_interval=log_interval,
        )

# --------------------------------------------------------------------------------
# ----------------------------- REWARD FUNCTIONS API -----------------------------
# --------------------------------------------------------------------------------

def base_height_l2(
    env: WarehouseBrawl,
    target_height: float,
    obj_name: str = 'player'
) -> float:
    """Penalize asset height from its target using L2 squared kernel."""
    obj: GameObject = env.objects[obj_name]
    return (obj.body.position.y - target_height)**2

class RewardMode(Enum):
    ASYMMETRIC_OFFENSIVE = 0
    SYMMETRIC = 1
    ASYMMETRIC_DEFENSIVE = 2


def damage_interaction_reward(
    env: 'WarehouseBrawl',
    mode: 'RewardMode' = None,
) -> float:
    """
    Core damage-based reward with multiple modes
    FIX: damage_taken_this_frame is reset in pre_process(), must be called after
    """
    player = env.objects["player"]
    opponent = env.objects["opponent"]

    damage_taken = player.damage_taken_this_frame
    damage_dealt = opponent.damage_taken_this_frame

    if mode == RewardMode.ASYMMETRIC_OFFENSIVE:
        reward = damage_dealt
    elif mode == RewardMode.SYMMETRIC:
        reward = damage_dealt - damage_taken
    elif mode == RewardMode.ASYMMETRIC_DEFENSIVE:
        reward = -damage_taken
    else:
        raise ValueError(f"Invalid mode: {mode}")

    return reward / 140

def danger_zone_reward(
    env: WarehouseBrawl,
    zone_penalty: int = 1,
    zone_height: float = 4.2
) -> float:
    """
    Penalize being too high (falling risk)
    """
    player: Player = env.objects["player"]
    reward = -zone_penalty if player.body.position.y >= zone_height else 0.0
    return reward * env.dt

def in_state_reward(
    env: WarehouseBrawl,
    desired_state: str = "attack",  # Changed to string
) -> float:
    """
    Reward/penalize being in specific states (SAFE VERSION)
    """
    player = env.objects["player"]
    
    if desired_state == "attack":
        reward = 1.0 if is_player_attacking(player) else 0.0
    elif desired_state == "jump":
        reward = 1.0 if is_player_jumping(player) else 0.0
    else:
        return 0.0
    
    return reward * env.dt

def head_to_middle_reward(env: WarehouseBrawl) -> float:
    """
    Encourage moving toward center stage
    """
    player: Player = env.objects["player"]
    multiplier = -1 if player.body.position.x > 0 else 1
    reward = multiplier * (player.body.position.x - player.prev_x)
    return reward

def head_to_opponent(env: WarehouseBrawl) -> float:
    """
    Basic approach reward
    """
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    multiplier = -1 if player.body.position.x > opponent.body.position.x else 1
    reward = multiplier * (player.body.position.x - player.prev_x)
    return reward

# --------------------------------------------
# --- TOURNAMENT-OPTIMIZED REWARD FUNCTIONS ---
# --------------------------------------------

def optimal_distance_reward(env, min_dist=1.5, max_dist=3.5) -> float:
    """
    Reward staying at optimal striking distance
    """
    player = env.objects["player"]
    opponent = env.objects["opponent"]

    dist = np.linalg.norm([
        player.body.position.x - opponent.body.position.x,
        player.body.position.y - opponent.body.position.y
    ])

    if min_dist <= dist <= max_dist:
        return 1.0
    else:
        penalty = abs(dist - ((min_dist + max_dist) / 2))
        return -0.15 * penalty


def burst_damage_reward(env, frame_window=60, threshold=15) -> float:
    """
    Reward sustained aggression and damage bursts
    """
    if not hasattr(env, "damage_window"):
        env.damage_window = []

    opponent = env.objects["opponent"]
    env.damage_window.append(opponent.damage_taken_this_frame)

    if len(env.damage_window) > frame_window:
        env.damage_window.pop(0)

    total_damage = sum(env.damage_window)
    if total_damage >= threshold:
        reward = 10.0 + (total_damage - threshold) * 0.5  # Bonus for exceeding threshold
        env.damage_window.clear()
        return reward
    return 0.0


def edge_guard_reward(env, zone_x=8.0) -> float:
    """
    Reward controlling edges while staying safe
    """
    player = env.objects["player"]
    opponent = env.objects["opponent"]

    if abs(opponent.body.position.x) >= zone_x and abs(player.body.position.x) < zone_x - 1.0:
        return 2.5
    elif abs(player.body.position.x) >= zone_x:
        return -2.5
    return 0.0


def move_to_opponent_reward(env: WarehouseBrawl) -> float:
    """
    Reward approaching opponent strategically
    """
    player = env.objects["player"]
    opponent = env.objects["opponent"]
    
    player_pos = np.array([player.body.position.x, player.body.position.y])
    opponent_pos = np.array([opponent.body.position.x, opponent.body.position.y])
    curr_distance = np.linalg.norm(player_pos - opponent_pos)
    
    # Initialize on first call
    if not hasattr(env, "prev_distance_to_opponent"):
        env.prev_distance_to_opponent = curr_distance
        return 0.0  # No reward on first step
    
    prev_distance = env.prev_distance_to_opponent
    
    # Calculate reward based on distance change
    if curr_distance < prev_distance:
        reward = (prev_distance - curr_distance) * 0.5
    else:
        reward = -0.15 * (curr_distance - prev_distance)
    
    env.prev_distance_to_opponent = curr_distance
    
    # Clip reward to prevent extreme values
    return np.clip(reward, -10.0, 10.0)


def combo_extension_reward(env: WarehouseBrawl) -> float:
    """
    Exponentially reward longer combos
    """
    if not hasattr(env, "combo_counter"):
        env.combo_counter = 0
        env.last_hit_frame = -100
    
    opponent = env.objects["opponent"]
    current_frame = getattr(env, 'frame_count', 0)
    
    # Reset combo if too much time passed
    if current_frame - env.last_hit_frame > 90:
        env.combo_counter = 0
    
    # Increment combo on hit
    if opponent.damage_taken_this_frame > 0:
        env.combo_counter += 1
        env.last_hit_frame = current_frame
        
        # Exponential scaling: 2-hit = 2.8, 3-hit = 5.2, 4-hit = 8, 5-hit = 11.2
        return min(env.combo_counter ** 1.5, 25.0)
    
    return 0.0


def defensive_spacing_reward(env: WarehouseBrawl) -> float:
    """
    Reward defensive play when at health disadvantage
    """
    player = env.objects["player"]
    opponent = env.objects["opponent"]
    
    # Only apply when losing
    if player.damage > opponent.damage:
        dist = np.linalg.norm([
            player.body.position.x - opponent.body.position.x,
            player.body.position.y - opponent.body.position.y
        ])
        
        # Reward medium-far spacing when disadvantaged
        if 3.5 <= dist <= 6.0:
            return 1.0
        elif dist < 2.0:
            return -0.6
    
    return 0.0


def weapon_priority_reward(env: WarehouseBrawl) -> float:
    """
    Encourage weapon pickup and retention
    """
    player = env.objects["player"]
    
    weapon_values = {
        "Hammer": 2.5,
        "Spear": 1.2,
        "Punch": -0.3
    }
    
    current_weapon = getattr(player, 'weapon', "Punch")
    return weapon_values.get(current_weapon, 0.0) * env.dt * 0.15


def punish_predictable_patterns(env: WarehouseBrawl) -> float:
    """
    Penalize repetitive actions to encourage adaptation (updated)
    """
    if not hasattr(env, "action_history"):
        env.action_history = []
    
    player = env.objects["player"]
    
    # Create action signature (which buttons are pressed)
    if hasattr(player, 'cur_action'):
        current_action = tuple((player.cur_action > 0.5).astype(int))
        
        env.action_history.append(current_action)
        
        if len(env.action_history) > 30:
            env.action_history.pop(0)
        
        # Check for excessive repetition
        if len(env.action_history) >= 10:
            recent = env.action_history[-10:]
            repetition_count = recent.count(current_action)
            
            if repetition_count >= 7:
                return -0.4
            elif repetition_count >= 5:
                return -0.2
    
    return 0.0

def dash_usage_reward(env: WarehouseBrawl) -> float:
    """
    Reward using dash for mobility (but not spamming)
    """
    if not hasattr(env, "dash_cooldown"):
        env.dash_cooldown = 0
    
    player = env.objects["player"]
    
    # Reduce cooldown
    if env.dash_cooldown > 0:
        env.dash_cooldown -= 1
    
    # Reward dash usage when cooldown is ready
    if is_player_dashing(player) and env.dash_cooldown == 0:
        env.dash_cooldown = 60  # 60 frames cooldown
        
        # Extra reward if escaping danger
        if player.body.position.y < 1.0 or abs(player.body.position.x) > 7.0:
            return 1.5
        
        return 0.5
    
    # Penalize spamming dash when on cooldown
    if is_player_dashing(player) and env.dash_cooldown > 0:
        return -0.3
    
    return 0.0


def smart_movement_reward(env: WarehouseBrawl) -> float:
    """
    Reward moving toward objectives (opponent or center)
    """
    player = env.objects["player"]
    opponent = env.objects["opponent"]
    
    if not is_player_moving(player):
        return 0.0
    
    # Initialize previous position
    if not hasattr(env, "prev_player_x"):
        env.prev_player_x = player.body.position.x
        return 0.0
    
    # Calculate movement direction
    moved_distance = player.body.position.x - env.prev_player_x
    env.prev_player_x = player.body.position.x
    
    # Reward moving toward opponent (when safe)
    player_safe = 1.0 <= player.body.position.y <= 5.0
    if player_safe and not is_opponent_knocked_out(opponent):
        opponent_direction = 1 if opponent.body.position.x > player.body.position.x else -1
        
        if np.sign(moved_distance) == opponent_direction:
            return 0.3
    
    # Reward moving to center when in danger
    player_danger = abs(player.body.position.x) > 7.0 or player.body.position.y < 1.0
    if player_danger:
        toward_center = -1 if player.body.position.x > 0 else 1
        
        if np.sign(moved_distance) == toward_center:
            return 0.5
    
    return 0.0


def weapon_management_reward(env: 'WarehouseBrawl') -> float:
    """
    FIX: Weapon attribute exists on Player class
    """
    player = env.objects["player"]
    opponent = env.objects["opponent"]
    
    if not is_player_throwing(player):
        return 0.0
    
    # Check current weapon (weapon attribute confirmed in Player class)
    current_weapon = player.weapon
    
    # Reward picking up good weapons
    if current_weapon == "Punch":
        return 0.8
    
    # Reward throwing weapon at opponent
    dist = np.linalg.norm([
        player.body.position.x - opponent.body.position.x,
        player.body.position.y - opponent.body.position.y
    ])
    
    if dist < 3.0:
        return 1.2
    
    # Slight penalty for throwing weapon when far
    if dist > 5.0:
        return -0.2
    
    return 0.0



def stage_control_reward(env: WarehouseBrawl) -> float:
    """
    Reward maintaining center stage advantage
    """
    player = env.objects["player"]
    opponent = env.objects["opponent"]
    
    player_center_dist = abs(player.body.position.x)
    opponent_center_dist = abs(opponent.body.position.x)
    
    if player_center_dist < opponent_center_dist:
        advantage = opponent_center_dist - player_center_dist
        return min(advantage * 0.3, 0.8)
    
    return 0.0


def hit_confirm_reward(env: WarehouseBrawl) -> float:
    """
    Reward successful hit confirms (landing attacks when they connect)
    """
    if not hasattr(env, "attack_frames"):
        env.attack_frames = []
    
    player = env.objects["player"]
    opponent = env.objects["opponent"]
    
    is_attacking = isinstance(player.state, AttackState)
    hit_landed = opponent.damage_taken_this_frame > 0
    
    # Track attack frames
    env.attack_frames.append(is_attacking)
    if len(env.attack_frames) > 10:
        env.attack_frames.pop(0)
    
    # Reward if hit lands during recent attack
    if hit_landed and any(env.attack_frames[-5:]):
        return 3.0
    
    return 0.0


def holding_more_than_3_keys(env: WarehouseBrawl) -> float:
    """
    Penalize input spam (updated with helper)
    """
    player: Player = env.objects["player"]
    active_count = get_active_input_count(player)
    
    if active_count > 3:
        return -env.dt
    return 0.0


def neutral_win_reward(env: WarehouseBrawl) -> float:
    """
    Reward winning neutral game (landing first hit in an exchange)
    """
    if not hasattr(env, "exchange_tracker"):
        env.exchange_tracker = {
            "in_exchange": False,
            "start_frame": 0
        }
    
    player = env.objects["player"]
    opponent = env.objects["opponent"]
    current_frame = getattr(env, 'frame_count', 0)
    
    # Detect start of exchange (both players close)
    dist = abs(player.body.position.x - opponent.body.position.x)
    
    if dist < 3.0 and not env.exchange_tracker["in_exchange"]:
        env.exchange_tracker["in_exchange"] = True
        env.exchange_tracker["start_frame"] = current_frame
    
    # Check for exchange end (someone got hit or moved away)
    if env.exchange_tracker["in_exchange"]:
        if opponent.damage_taken_this_frame > 0:
            env.exchange_tracker["in_exchange"] = False
            return 5.0  # Won neutral
        elif player.damage_taken_this_frame > 0:
            env.exchange_tracker["in_exchange"] = False
            return -2.0  # Lost neutral
        elif dist > 4.5:
            env.exchange_tracker["in_exchange"] = False
    
    return 0.0

# --------------------------------------------------------------------------------
# ----------------------------- SIGNAL-BASED REWARDS -----------------------------
# --------------------------------------------------------------------------------

def on_win_reward(env: WarehouseBrawl, agent: str) -> float:
    """Match win bonus"""
    return 1.0 if agent == 'player' else -1.0

def on_knockout_reward(env: WarehouseBrawl, agent: str) -> float:
    """Knockout event reward"""
    return -1.0 if agent == 'player' else 1.0

def on_combo_reward(env: WarehouseBrawl, agent: str) -> float:
    """Combo hit during stun"""
    return 1.0 if agent == 'opponent' else -1.0
    
def on_equip_reward(env: 'WarehouseBrawl', agent: str) -> float:
    """
    Signal-based reward using weapon_equip_signal
    USAGE: Connect this to env.weapon_equip_signal in your training loop
    Example: env.weapon_equip_signal.connect(on_equip_reward)
    """
    if agent == "player":
        weapon = env.objects["player"].weapon
        if weapon == "Hammer":
            return 2.5
        elif weapon == "Spear":
            return 1.5
        elif weapon == "Punch":
            return 0.0  # Just dropped weapon
    return 0.0


def on_drop_reward(env: 'WarehouseBrawl', agent: str) -> float:
    """
    Signal-based reward using weapon_drop_signal
    USAGE: Connect this to env.weapon_drop_signal in your training loop
    Example: env.weapon_drop_signal.connect(on_drop_reward)
    """
    if agent == "player":
        # Penalty for dropping weapon
        return -1.5
    return 0.0


# --------------------------------------------------------------------------------
# ----------------------------- HELPER FUNCTIONS ---------------------------------
# --------------------------------------------------------------------------------

def is_player_attacking(player) -> bool:
    """
    FIX: Check if player is in AttackState (most reliable)
    Backup: Check action indices
    """
    # Primary method: Check state by class name (no import needed)
    if hasattr(player, 'state'):
        state_name = type(player.state).__name__
        if state_name == 'AttackState':
            return True
    
    # Backup method: Check actions
    if hasattr(player, 'cur_action'):
        action = player.cur_action
        if len(action) > 8:
            light_attack = action[7] > 0.5
            heavy_attack = action[8] > 0.5
            return light_attack or heavy_attack
    
    return False

def is_player_jumping(player) -> bool:
    """
    FIX: Check upward velocity and InAirState, not just space key
    """
    # Check if in air and moving upward
    if hasattr(player, 'body') and hasattr(player.body, 'velocity'):
        if player.body.velocity.y < -0.5:  # Negative Y is up in this system
            return True
    
    # Check space key being pressed
    if hasattr(player, 'cur_action'):
        action = player.cur_action
        if len(action) > 4 and action[4] > 0.5:
            return True
    
    return False

def is_player_dashing(player) -> bool:
    """Helper to check if player is dashing"""
    # Check DashState
    if hasattr(player, 'state'):
        state_name = type(player.state).__name__
        if state_name in ['DashState', 'BackDashState']:
            return True
    
    # Check action
    if hasattr(player, 'cur_action'):
        action = player.cur_action
        if len(action) > 6 and action[6] > 0.5:
            return True
    return False


def is_player_moving(player) -> bool:
    """Helper to check if player is moving horizontally"""
    # Check velocity
    if hasattr(player.body, 'velocity'):
        if abs(player.body.velocity.x) > 0.5:
            return True
    
    # Check action
    if hasattr(player, 'cur_action'):
        action = player.cur_action
        if len(action) > 3:
            moving_left = action[1] > 0.5
            moving_right = action[3] > 0.5
            return moving_left or moving_right
    return False


def is_player_throwing(player) -> bool:
    """Helper to check if player is picking up/throwing weapon"""
    if hasattr(player, 'cur_action'):
        action = player.cur_action
        if len(action) > 5 and action[5] > 0.5:
            return True
    return False


def get_active_input_count(player) -> int:
    """Count how many buttons are being pressed"""
    if hasattr(player, 'cur_action'):
        action = player.cur_action
        return int((action > 0.5).sum())
    return 0

def is_opponent_knocked_out(opponent) -> bool:
    """
    FIX: Check KOState instead of guessing from position
    """
    # Primary check: Is in KO state? (check by class name, no import needed)
    if hasattr(opponent, 'state'):
        state_name = type(opponent.state).__name__
        if state_name == 'KOState':
            return True
    
    # Secondary check: Out of bounds
    opponent_out_of_bounds = (
        abs(opponent.body.position.x) > 14.9 or  # stage_width_tiles/2
        opponent.body.position.y < -8.4 or       # -stage_height_tiles/2
        opponent.body.position.y > 8.4           # stage_height_tiles/2
    )
    
    return opponent_out_of_bounds

# --------------------------------------------------------------------------------
# ----------------------- FIRST HIT REWARD (FRAME-BASED) -------------------------
# --------------------------------------------------------------------------------

def first_hit_reward(env: WarehouseBrawl) -> float:
    """
    First hit of the match bonus - rewards landing first blood
    This is frame-based, not signal-based, so it works without modifying other files
    """
    if not hasattr(env, "first_hit_done"):
        env.first_hit_done = False
        env.first_hit_winner = None

    if not env.first_hit_done:
        player = env.objects["player"]
        opponent = env.objects["opponent"]
        
        # Check if opponent got hit (player landed first hit)
        if opponent.damage_taken_this_frame > 0:
            env.first_hit_done = True
            env.first_hit_winner = "player"
            return 20.0  # Large reward for first blood
        
        # Check if player got hit (lost first hit)
        elif player.damage_taken_this_frame > 0:
            env.first_hit_done = True
            env.first_hit_winner = "opponent"
            return -10.0  # Penalty for getting hit first
    
    return 0.0
def self_preservation_reward(env: 'WarehouseBrawl') -> float:
    """
    FIX: Use actual stage boundaries
    """
    player = env.objects["player"]
    opponent = env.objects["opponent"]
    
    stage_x_limit = env.stage_width_tiles / 2 - 2.0
    
    # Track if both players are in danger
    player_danger = player.body.position.y < 0.5 or abs(player.body.position.x) > stage_x_limit
    opponent_danger = opponent.body.position.y < 0.5 or abs(opponent.body.position.x) > stage_x_limit
    
    # CRITICAL: If player is in danger, penalize being near opponent
    if player_danger:
        dist = np.linalg.norm([
            player.body.position.x - opponent.body.position.x,
            player.body.position.y - opponent.body.position.y
        ])
        
        if dist < 2.5:
            return -8.0
        elif dist < 4.0:
            return -3.0
        else:
            return 1.5
    
    # If opponent is in danger but player is safe, reward pressure
    if opponent_danger and not player_danger:
        return 2.0
    
    return 0.0


def ledge_safety_reward(env: 'WarehouseBrawl') -> float:
    """
    FIX: Use actual stage boundaries from env
    """
    player = env.objects["player"]
    
    # Use actual stage dimensions
    stage_x_limit = env.stage_width_tiles / 2
    stage_y_min = -env.stage_height_tiles / 2
    
    # Distance from edge
    edge_dist_x = stage_x_limit - abs(player.body.position.x)
    edge_dist_y = player.body.position.y - stage_y_min
    
    # Penalize being near edges
    if edge_dist_x < 2.0:
        penalty = -1.5 * (2.0 - edge_dist_x)
        return penalty
    
    if edge_dist_y < 1.5:
        penalty = -2.0 * (1.5 - edge_dist_y)
        return penalty
    
    # Reward staying in safe zone
    if edge_dist_x > 5.0 and edge_dist_y > 3.0:
        return 0.4
    
    return 0.0


def situational_aggression_reward(env: 'WarehouseBrawl') -> float:
    """
    FIX: Use actual stage boundaries from env
    """
    player = env.objects["player"]
    
    # Use actual stage dimensions
    stage_x_limit = env.stage_width_tiles / 2 - 2.0  # Add safety margin
    stage_y_max = env.stage_height_tiles / 2 - 2.0
    
    player_safe = (
        1.0 <= player.body.position.y <= stage_y_max and
        abs(player.body.position.x) < stage_x_limit
    )
    
    if is_player_attacking(player):
        if player_safe:
            return 0.5
        else:
            return -2.5
    
    return 0.0

def respect_knockout_reward(env: WarehouseBrawl) -> float:
    """
    Penalize wasting actions when opponent is knocked out
    """
    opponent = env.objects["opponent"]
    player = env.objects["player"]
    
    if is_opponent_knocked_out(opponent):
        active_inputs = get_active_input_count(player)
        
        if active_inputs > 0:
            return -1.5 * active_inputs
        
        # Reward staying still
        if hasattr(player.body, 'velocity'):
            velocity = np.linalg.norm([player.body.velocity.x, player.body.velocity.y])
            if velocity < 0.1:
                return 0.5
    
    return 0.0


def opportunistic_positioning_reward(env: 'WarehouseBrawl') -> float:
    """
    FIX: Use is_opponent_knocked_out() correctly
    """
    opponent = env.objects["opponent"]
    player = env.objects["player"]
    
    if is_opponent_knocked_out(opponent):
        # Reward moving to center stage
        center_distance = abs(player.body.position.x)
        
        if center_distance < 2.0:
            return 1.0
        elif center_distance < 4.0:
            return 0.5
    
    return 0.0

def platform_awareness_reward(env: 'WarehouseBrawl') -> float:
    """
    FIX: Use is_on_floor() method instead of hardcoded heights
    """
    player = env.objects["player"]
    
    # Check if on floor (includes platform1)
    on_floor = player.is_on_floor()
    
    player_height = player.body.position.y
    
    # Reward being on solid ground
    if on_floor:
        return 0.5
    
    # Strong penalty for being too low (falling)
    if player_height < 0.0:
        return -3.0
    elif player_height < 1.0:
        return -1.0 * (1.0 - player_height)
    
    # Reward jumping when low (trying to recover)
    if player_height < 1.0 and is_player_jumping(player):
        return 1.5
    
    return 0.0

def survive_reward(env: 'WarehouseBrawl') -> float:
    """
    FIX: Use is_on_floor() and actual stage boundaries
    """
    player = env.objects["player"]
    
    # Penalize being too low (about to fall)
    if player.body.position.y < -2.0:
        return -5.0
    elif player.body.position.y < 0.0:
        return -2.0
    
    # Reward staying on platforms
    if player.is_on_floor():
        return 0.3
    
    return 0.0
def penalize_attack_spam(env: WarehouseBrawl) -> float:
    """Penalize excessive attacking"""
    player = env.objects["player"]
    return -0.02 if is_player_attacking(player) else 0.0

# --------------------------------------------------------------------------------
# ----------------------- ATTACK BOUNDARY SAFETY ---------------------------------
# --------------------------------------------------------------------------------

def predict_attack_danger(player, opponent, attack_type: str = "any") -> dict:
    """
    FIX: Use actual stage boundaries and velocity from body
    """
    result = {
        'safe': True,
        'danger_level': 0.0,
        'reason': 'Safe to attack'
    }
    
    player_pos = np.array([player.body.position.x, player.body.position.y])
    player_vel = np.array([player.body.velocity.x, player.body.velocity.y])
    
    # Use env stage boundaries
    STAGE_X_LIMIT = player.env.stage_width_tiles / 2
    STAGE_Y_MIN = -player.env.stage_height_tiles / 2
    
    # Define attack momentum
    attack_momentum = {
        'light': (0.5, 0.2),
        'heavy': (1.2, 0.1),
        'aerial': (0.3, -0.8),
        'dash_attack': (2.0, 0.0)
    }
    
    momentum_x, momentum_y = attack_momentum.get(attack_type, (1.0, 0.0))
    
    # Add attack momentum to current velocity
    predicted_vel_x = player_vel[0] + momentum_x * np.sign(player_vel[0] if abs(player_vel[0]) > 0.1 else 1.0)
    predicted_vel_y = player_vel[1] + momentum_y
    
    # Predict position 30 frames ahead
    frames_ahead = 30
    predicted_x = player_pos[0] + predicted_vel_x * frames_ahead * 0.016
    predicted_y = player_pos[1] + predicted_vel_y * frames_ahead * 0.016
    
    # Check boundaries
    danger_reasons = []
    danger_score = 0.0
    
    # X boundary danger
    if abs(predicted_x) > STAGE_X_LIMIT - 1.0:
        danger_score += 0.5
        danger_reasons.append("Will approach X boundary")
        
    if abs(predicted_x) > STAGE_X_LIMIT:
        danger_score += 0.5
        danger_reasons.append("Will exceed X boundary!")
    
    # Y boundary danger
    if predicted_y < STAGE_Y_MIN + 0.5:
        danger_score += 0.4
        danger_reasons.append("Will fall too low")
        
    if predicted_y < STAGE_Y_MIN:
        danger_score += 0.6
        danger_reasons.append("Will fall off stage!")
    
    # Current position already dangerous
    if abs(player_pos[0]) > STAGE_X_LIMIT - 2.0:
        danger_score += 0.3
        danger_reasons.append("Currently near edge")
    
    if player_pos[1] < STAGE_Y_MIN + 1.0:
        danger_score += 0.3
        danger_reasons.append("Currently near fall zone")
    
    # Final assessment
    result['danger_level'] = min(danger_score, 1.0)
    result['safe'] = danger_score < 0.3
    result['reason'] = "; ".join(danger_reasons) if danger_reasons else "Safe to attack"
    
    return result


def safe_attack_reward(env: WarehouseBrawl) -> float:
    """
    Reward safe attacks, heavily penalize dangerous attacks
    """
    player = env.objects["player"]
    opponent = env.objects["opponent"]
    
    # Only evaluate when attacking
    if not is_player_attacking(player):
        return 0.0
    
    # Determine attack type based on action
    action = player.cur_action if hasattr(player, 'cur_action') else None
    attack_type = "light" if (action is not None and action[7] > 0.5) else "heavy"
    
    # Check if dash attacking
    if is_player_dashing(player) and is_player_attacking(player):
        attack_type = "dash_attack"
    
    # Predict danger
    danger_info = predict_attack_danger(player, opponent, attack_type)
    
    if not danger_info['safe']:
        # Penalize based on danger level
        penalty = -5.0 * danger_info['danger_level']
        
        # Extra penalty for extremely dangerous attacks
        if danger_info['danger_level'] > 0.7:
            penalty *= 2.0
        
        return penalty
    else:
        # Small reward for safe aggression
        return 0.3
    


def momentum_awareness_reward(env: WarehouseBrawl) -> float:
    """
    Teach agent to respect attack momentum and positioning
    """
    player = env.objects["player"]
    
    if not hasattr(player.body, 'velocity'):
        return 0.0
    
    velocity_magnitude = np.linalg.norm([player.body.velocity.x, player.body.velocity.y])
    
    # If moving fast toward edge, penalize attacking
    edge_distance = min(
        10.0 - abs(player.body.position.x),  # Distance to left/right edge
        player.body.position.y - (-1.0)       # Distance to bottom
    )
    
    if edge_distance < 2.0 and velocity_magnitude > 1.0:
        if is_player_attacking(player):
            # Moving fast near edge and attacking = very bad
            return -3.0 * (2.0 - edge_distance)
        else:
            # Moving fast near edge without attacking = still concerning
            return -1.0 * (2.0 - edge_distance)
    
    return 0.0


def recovery_priority_reward(env: 'WarehouseBrawl') -> float:
    """
    FIX: Use actual stage boundaries
    """
    player = env.objects["player"]
    
    stage_x_limit = env.stage_width_tiles / 2 - 2.0
    
    # Check if in danger
    in_danger = (
        player.body.position.y < 0.5 or 
        abs(player.body.position.x) > stage_x_limit
    )
    
    if in_danger:
        # Heavily penalize attacking when in danger
        if is_player_attacking(player):
            return -4.0
        
        # Reward defensive actions
        if is_player_jumping(player):
            return 2.0
        
        if is_player_dashing(player) and not is_player_attacking(player):
            # Check if moving toward center
            if hasattr(player, 'cur_action') and len(player.cur_action) > 3:
                moving_left = player.cur_action[1] > 0.5
                moving_right = player.cur_action[3] > 0.5
                
                moving_toward_center = (
                    (player.body.position.x > 0 and moving_left) or
                    (player.body.position.x < 0 and moving_right)
                )
                if moving_toward_center:
                    return 1.5
    
    return 0.0


def attack_commitment_reward(env: WarehouseBrawl) -> float:
    """
    Reward finishing attack strings, penalize half-committed attacks
    """
    if not hasattr(env, "attack_state"):
        env.attack_state = {
            'in_attack': False,
            'attack_frames': 0,
            'completed_attacks': 0
        }
    
    player = env.objects["player"]
    opponent = env.objects["opponent"]
    
    currently_attacking = is_player_attacking(player)
    
    # Track attack state
    if currently_attacking:
        if not env.attack_state['in_attack']:
            env.attack_state['in_attack'] = True
            env.attack_state['attack_frames'] = 0
        env.attack_state['attack_frames'] += 1
    else:
        if env.attack_state['in_attack']:
            # Just finished attacking
            env.attack_state['in_attack'] = False
            
            # Check if it was a good attack (lasted reasonable duration)
            if env.attack_state['attack_frames'] >= 10:
                # Good commitment
                env.attack_state['completed_attacks'] += 1
                return 0.8
            elif env.attack_state['attack_frames'] < 5:
                # Panic attack / accidental input
                return -0.5
    
    return 0.0

# ==================== GROUNDED PLAY REWARDS ====================

def grounded_combat_reward(env: 'WarehouseBrawl') -> float:
    """
    Reward attacking while grounded on stage
    Fixed to work with single-player reward system
    """
    player = env.objects["player"]
    
    # Check if in AttackState (using class name check)
    is_attacking = False
    if hasattr(player, 'state'):
        state_name = type(player.state).__name__
        is_attacking = (state_name == 'AttackState')
    
    # Check if grounded using player's method
    is_grounded = player.is_on_floor()
    
    if is_attacking and is_grounded:
        return 0.1  # Small constant reward for safe aggression
    
    return 0.0


def excessive_aerial_penalty(env: 'WarehouseBrawl') -> float:
    """
    Penalize staying airborne for too long without purpose
    Fixed to work with single-player reward system
    """
    player = env.objects["player"]
    
    # Track frames spent airborne
    if not hasattr(player, 'airborne_frames'):
        player.airborne_frames = 0
    
    # Check if grounded
    is_grounded = player.is_on_floor()
    
    if not is_grounded:
        player.airborne_frames += 1
    else:
        player.airborne_frames = 0
    
    # Penalize extended aerial time (>60 frames = 2 seconds)
    if player.airborne_frames > 60:
        return -0.05 * (player.airborne_frames - 60) / 30  # Escalating penalty
    
    return 0.0


def aerial_attack_penalty(env: 'WarehouseBrawl') -> float:
    """
    Penalize attacking while far from stage or with bad positioning
    Fixed to work with single-player reward system and use actual stage bounds
    """
    player = env.objects["player"]
    
    # Check if in AttackState (using class name check)
    is_attacking = False
    if hasattr(player, 'state'):
        state_name = type(player.state).__name__
        is_attacking = (state_name == 'AttackState')
    
    is_grounded = player.is_on_floor()
    
    # Penalize attacking while airborne
    if is_attacking and not is_grounded:
        # Use actual stage dimensions
        stage_x_limit = env.stage_width_tiles / 2  # 14.9
        stage_y_safe = env.stage_height_tiles / 2 - 3  # ~5.4
        
        x_pos = abs(player.body.position.x)
        y_pos = player.body.position.y
        
        # Heavy penalty if attacking near blast zones
        if x_pos > stage_x_limit - 2 or y_pos > stage_y_safe:
            return -0.3  # Strong penalty for risky aerial attacks
        elif x_pos > stage_x_limit - 6:
            return -0.15  # Moderate penalty
        else:
            return -0.05  # Small penalty for any aerial attack
    
    return 0.0


def stock_awareness_reward(env: 'WarehouseBrawl') -> float:
    """
    Reward playing more carefully when at low stocks
    Fixed to work with single-player reward system and use actual stage bounds
    """
    player = env.objects["player"]
    opponent = env.objects["opponent"]
    
    # If player has fewer stocks, reward survival positioning
    if player.stocks < opponent.stocks:
        is_grounded = player.is_on_floor()
        x_pos = abs(player.body.position.x)
        
        # Use actual stage center (reasonable safe zone)
        stage_safe_x = env.stage_width_tiles / 2 - 8  # ~6.9
        
        # Reward being grounded and near center
        if is_grounded and x_pos < stage_safe_x:
            return 0.1 * (opponent.stocks - player.stocks)
    
    return 0.0


# ==================== PLATFORM-AWARE REWARDS ====================

def platform_positioning_reward(env: 'WarehouseBrawl') -> float:
    """
    Reward controlling the moving platform for positional advantage
    
    The moving platform (platform1) moves between two waypoints and provides:
    - High ground advantage
    - Momentum for attacks
    - Escape routes
    """
    player = env.objects["player"]
    opponent = env.objects["opponent"]
    platform = env.objects["platform1"]
    
    # Check if player is on the platform
    player_on_platform = (
        player.is_on_floor() and
        player.shape.cache_bb().intersects(platform.shape.cache_bb()) and
        player.body.position.y <= platform.body.position.y - 0.01 and
        (platform.body.position.x - 1.25) <= player.body.position.x <= (platform.body.position.x + 1.25)
    )
    
    # Check if opponent is on the platform
    opponent_on_platform = (
        opponent.is_on_floor() and
        opponent.shape.cache_bb().intersects(platform.shape.cache_bb()) and
        opponent.body.position.y <= platform.body.position.y - 0.01 and
        (platform.body.position.x - 1.25) <= opponent.body.position.x <= (platform.body.position.x + 1.25)
    )
    
    # Reward controlling the platform
    if player_on_platform and not opponent_on_platform:
        # Extra reward if platform gives height advantage
        if player.body.position.y > opponent.body.position.y + 0.5:
            return 1.5
        return 0.8
    
    # Penalty if opponent controls platform
    if opponent_on_platform and not player_on_platform:
        if opponent.body.position.y > player.body.position.y + 0.5:
            return -1.0
        return -0.5
    
    return 0.0


def platform_approach_reward(env: 'WarehouseBrawl') -> float:
    """
    Reward moving toward the platform when opponent controls it
    This encourages contesting platform control
    """
    player = env.objects["player"]
    opponent = env.objects["opponent"]
    platform = env.objects["platform1"]
    
    # Check if opponent is on platform
    opponent_on_platform = (
        opponent.shape.cache_bb().intersects(platform.shape.cache_bb()) and
        opponent.body.position.y <= platform.body.position.y - 0.01
    )
    
    if not opponent_on_platform:
        return 0.0
    
    # Calculate distance to platform
    platform_center = platform.body.position
    player_to_platform = np.linalg.norm([
        player.body.position.x - platform_center[0],
        player.body.position.y - platform_center[1]
    ])
    
    # Initialize tracking
    if not hasattr(env, "prev_platform_distance"):
        env.prev_platform_distance = player_to_platform
        return 0.0
    
    # Reward getting closer to platform
    distance_change = env.prev_platform_distance - player_to_platform
    env.prev_platform_distance = player_to_platform
    
    if distance_change > 0:  # Moving closer
        return 0.5 * distance_change
    
    return 0.0


def platform_timing_reward(env: 'WarehouseBrawl') -> float:
    """
    Reward timing attacks with platform momentum
    
    The platform moves with velocity, so timing attacks when the platform
    moves toward the opponent can add momentum to attacks
    """
    player = env.objects["player"]
    opponent = env.objects["opponent"]
    platform = env.objects["platform1"]
    
    # Check if player is on platform
    player_on_platform = (
        player.shape.cache_bb().intersects(platform.shape.cache_bb()) and
        player.body.position.y <= platform.body.position.y - 0.01
    )
    
    if not player_on_platform:
        return 0.0
    
    # Check if attacking
    if not is_player_attacking(player):
        return 0.0
    
    # Get platform velocity
    platform_vel_x = platform.body.velocity.x
    
    # Calculate if platform is moving toward opponent
    to_opponent_x = opponent.body.position.x - player.body.position.x
    
    # Reward if platform momentum helps the attack
    if np.sign(platform_vel_x) == np.sign(to_opponent_x) and abs(platform_vel_x) > 0.1:
        # Platform is moving toward opponent while attacking
        return 1.2
    
    # Slight penalty if attacking against platform momentum
    if np.sign(platform_vel_x) != np.sign(to_opponent_x) and abs(platform_vel_x) > 0.1:
        return -0.3
    
    return 0.0


def platform_escape_reward(env: 'WarehouseBrawl') -> float:
    """
    Reward using the platform to escape danger
    """
    player = env.objects["player"]
    opponent = env.objects["opponent"]
    platform = env.objects["platform1"]
    
    # Check if player is at health disadvantage
    if player.damage <= opponent.damage:
        return 0.0
    
    # Check if opponent is close (dangerous)
    distance = np.linalg.norm([
        player.body.position.x - opponent.body.position.x,
        player.body.position.y - opponent.body.position.y
    ])
    
    if distance > 4.0:  # Not in danger
        return 0.0
    
    # Check if moving toward platform
    platform_center = platform.body.position
    player_to_platform = np.linalg.norm([
        player.body.position.x - platform_center[0],
        player.body.position.y - platform_center[1]
    ])
    
    # Track distance
    if not hasattr(env, "prev_escape_platform_dist"):
        env.prev_escape_platform_dist = player_to_platform
        return 0.0
    
    # Reward moving toward platform when disadvantaged and pressured
    distance_change = env.prev_escape_platform_dist - player_to_platform
    env.prev_escape_platform_dist = player_to_platform
    
    if distance_change > 0:  # Moving closer to platform
        return 1.0 * distance_change
    
    return 0.0


def platform_contest_reward(env: 'WarehouseBrawl') -> float:
    """
    Reward being willing to fight for platform control
    """
    player = env.objects["player"]
    opponent = env.objects["opponent"]
    platform = env.objects["platform1"]
    
    platform_center = platform.body.position
    
    # Check if both players are near the platform
    player_dist = np.linalg.norm([
        player.body.position.x - platform_center[0],
        player.body.position.y - platform_center[1]
    ])
    
    opponent_dist = np.linalg.norm([
        opponent.body.position.x - platform_center[0],
        opponent.body.position.y - platform_center[1]
    ])
    
    # Both players contesting platform
    if player_dist < 3.0 and opponent_dist < 3.0:
        # Reward attacking during contest
        if is_player_attacking(player):
            return 0.8
        
        # Small reward for just being there
        return 0.2
    
    return 0.0


def platform_prediction_reward(env: 'WarehouseBrawl') -> float:
    """
    Reward predicting where the platform will be
    
    The platform moves between waypoint1 and waypoint2, so smart players
    should move to where it's going, not where it is
    """
    player = env.objects["player"]
    platform = env.objects["platform1"]
    
    # Don't reward if already on platform
    if player.is_on_floor() and player.shape.cache_bb().intersects(platform.shape.cache_bb()):
        return 0.0
    
    # Determine where platform is heading
    target_waypoint = platform.waypoint2 if platform.moving_to_w2 else platform.waypoint1
    
    # Calculate if player is moving toward the target waypoint
    player_to_target = np.array([
        target_waypoint[0] - player.body.position.x,
        target_waypoint[1] - player.body.position.y
    ])
    
    player_velocity = np.array([
        player.body.velocity.x,
        player.body.velocity.y
    ])
    
    # Dot product to see if moving in right direction
    if np.linalg.norm(player_velocity) > 0.5:  # Player is moving
        velocity_normalized = player_velocity / np.linalg.norm(player_velocity)
        target_normalized = player_to_target / max(np.linalg.norm(player_to_target), 0.1)
        
        alignment = np.dot(velocity_normalized, target_normalized)
        
        # Reward if moving toward where platform is going (not where it is now)
        if alignment > 0.7:  # Moving in right direction
            return 0.6
    
    return 0.0

# -------------------------------------------------------------------------
# ----------------------------- REWARD MANAGER -----------------------------
# -------------------------------------------------------------------------

# def gen_reward_manager():
#     """
#     INTELLIGENT AGGRESSION configuration
#     Reward calculated combat, not reckless suicide
#     """
#     reward_functions = {
#         # === CORE COMBAT (HIGH BUT BALANCED) ===
#         'damage_interaction_reward': RewTerm(
#             func=lambda env: damage_interaction_reward(env, mode=RewardMode.SYMMETRIC),
#             weight=3.5
#         ),
#         'combo_extension_reward': RewTerm(func=combo_extension_reward, weight=1.8),
#         'hit_confirm_reward': RewTerm(func=hit_confirm_reward, weight=1.5),
#         'attack_commitment_reward': RewTerm(func=attack_commitment_reward, weight=1.0),  # REDUCED
        
#         # === SMART AGGRESSION ===
#         'situational_aggression_reward': RewTerm(func=situational_aggression_reward, weight=2.0),
#         'neutral_win_reward': RewTerm(func=neutral_win_reward, weight=1.5),
#         'first_hit_reward': RewTerm(func=first_hit_reward, weight=1.2),
        
#         # === GROUNDED PLAY (NEW!) ===
#         # 'grounded_combat_reward': RewTerm(func=grounded_combat_reward, weight=1.5),  # NEW
#         # 'aerial_attack_penalty': RewTerm(func=aerial_attack_penalty, weight=1.0),  # NEW
        
#         # === ATTACK SAFETY ===
#         'safe_attack_reward': RewTerm(func=safe_attack_reward, weight=1.5),
#         'momentum_awareness_reward': RewTerm(func=momentum_awareness_reward, weight=0.8),
#         'recovery_priority_reward': RewTerm(func=recovery_priority_reward, weight=1.2),
        
#         # === SURVIVAL (CRITICAL!) ===
#         'survive_reward': RewTerm(func=survive_reward, weight=2.0),  # INCREASED
#         'self_preservation_reward': RewTerm(func=self_preservation_reward, weight=2.0),  # INCREASED
#         'ledge_safety_reward': RewTerm(func=ledge_safety_reward, weight=1.5),
#         'danger_zone_reward': RewTerm(func=danger_zone_reward, weight=-1.5),
#         # 'stock_awareness_reward': RewTerm(func=stock_awareness_reward, weight=1.2),  # NEW
        
#         # === POSITIONING ===
#         'platform_awareness_reward': RewTerm(func=platform_awareness_reward, weight=1.2),
#         'optimal_distance_reward': RewTerm(func=optimal_distance_reward, weight=1.0),
#         'stage_control_reward': RewTerm(func=stage_control_reward, weight=1.5),
#         'edge_guard_reward': RewTerm(func=edge_guard_reward, weight=0.6),  # REDUCED
#         'opportunistic_positioning_reward': RewTerm(func=opportunistic_positioning_reward, weight=1.0),
        
#         # === AWARENESS ===
#         'respect_knockout_reward': RewTerm(func=respect_knockout_reward, weight=2.0),
#         'defensive_spacing_reward': RewTerm(func=defensive_spacing_reward, weight=0.8),
        
#         # === MOBILITY & WEAPON ===
#         'dash_usage_reward': RewTerm(func=dash_usage_reward, weight=0.8),
#         'smart_movement_reward': RewTerm(func=smart_movement_reward, weight=1.2),
#         'weapon_management_reward': RewTerm(func=weapon_management_reward, weight=1.0),
        
#         # === ANTI-SPAM & BAD HABITS ===
#         'penalize_attack_spam': RewTerm(func=penalize_attack_spam, weight=0.8),
#         'holding_more_than_3_keys': RewTerm(func=holding_more_than_3_keys, weight=0.5),
#         'punish_predictable_patterns': RewTerm(func=punish_predictable_patterns, weight=0.8),
#         # 'excessive_aerial_penalty': RewTerm(func=excessive_aerial_penalty, weight=1.2),  # NEW
#     }

#     signal_subscriptions = {
#         'on_win_reward': ('win_signal', RewTerm(func=on_win_reward, weight=500)),
#         'on_knockout_reward': ('knockout_signal', RewTerm(func=on_knockout_reward, weight=-100)),  # HUGE penalty
#         'on_combo_reward': ('hit_during_stun', RewTerm(func=on_combo_reward, weight=20)),
#         'on_equip_reward': ('weapon_equip_signal', RewTerm(func=on_equip_reward, weight=12)),
#         'on_drop_reward': ('weapon_drop_signal', RewTerm(func=on_drop_reward, weight=-15)),
#     }

#     return RewardManager(reward_functions, signal_subscriptions)


def gen_reward_manager():
    """
    INTELLIGENT AGGRESSION + PLATFORM MASTERY configuration
    Reward calculated combat, platform control, and smart positioning
    """
    reward_functions = {
        # === CORE COMBAT (HIGH BUT BALANCED) ===
        'damage_interaction_reward': RewTerm(
            func=lambda env: damage_interaction_reward(env, mode=RewardMode.SYMMETRIC),
            weight=3.5
        ),
        'combo_extension_reward': RewTerm(func=combo_extension_reward, weight=1.8),
        'hit_confirm_reward': RewTerm(func=hit_confirm_reward, weight=1.5),
        'attack_commitment_reward': RewTerm(func=attack_commitment_reward, weight=1.0),
        
        # === SMART AGGRESSION ===
        'situational_aggression_reward': RewTerm(func=situational_aggression_reward, weight=2.0),
        'neutral_win_reward': RewTerm(func=neutral_win_reward, weight=1.5),
        'first_hit_reward': RewTerm(func=first_hit_reward, weight=1.2),
        
        # === GROUNDED PLAY ===
        'grounded_combat_reward': RewTerm(func=grounded_combat_reward, weight=1.5),
        'aerial_attack_penalty': RewTerm(func=aerial_attack_penalty, weight=1.0),
        'excessive_aerial_penalty': RewTerm(func=excessive_aerial_penalty, weight=1.2),
        
        # === ATTACK SAFETY ===
        'safe_attack_reward': RewTerm(func=safe_attack_reward, weight=1.5),
        'momentum_awareness_reward': RewTerm(func=momentum_awareness_reward, weight=0.8),
        'recovery_priority_reward': RewTerm(func=recovery_priority_reward, weight=1.2),
        
        # === SURVIVAL (CRITICAL!) ===
        'survive_reward': RewTerm(func=survive_reward, weight=2.0),
        'self_preservation_reward': RewTerm(func=self_preservation_reward, weight=2.0),
        'ledge_safety_reward': RewTerm(func=ledge_safety_reward, weight=1.5),
        'danger_zone_reward': RewTerm(func=danger_zone_reward, weight=1.2),
        'stock_awareness_reward': RewTerm(func=stock_awareness_reward, weight=1.2),
        
        # === PLATFORM CONTROL (NEW CATEGORY!) ===
        'platform_positioning_reward': RewTerm(func=platform_positioning_reward, weight=1.8),  # High priority
        'platform_timing_reward': RewTerm(func=platform_timing_reward, weight=1.3),  # Momentum attacks
        'platform_contest_reward': RewTerm(func=platform_contest_reward, weight=1.0),  # Fight for control
        'platform_escape_reward': RewTerm(func=platform_escape_reward, weight=1.2),  # Defensive use
        'platform_approach_reward': RewTerm(func=platform_approach_reward, weight=0.9),  # Contest opponent's control
        'platform_prediction_reward': RewTerm(func=platform_prediction_reward, weight=0.8),  # Advanced play
        
        # === POSITIONING ===
        'platform_awareness_reward': RewTerm(func=platform_awareness_reward, weight=1.2),
        'optimal_distance_reward': RewTerm(func=optimal_distance_reward, weight=1.0),
        'stage_control_reward': RewTerm(func=stage_control_reward, weight=1.3),  # Slightly increased
        'edge_guard_reward': RewTerm(func=edge_guard_reward, weight=0.6),
        'opportunistic_positioning_reward': RewTerm(func=opportunistic_positioning_reward, weight=1.0),
        
        # === AWARENESS ===
        'respect_knockout_reward': RewTerm(func=respect_knockout_reward, weight=2.0),
        'defensive_spacing_reward': RewTerm(func=defensive_spacing_reward, weight=0.8),
        
        # === MOBILITY & WEAPON ===
        'dash_usage_reward': RewTerm(func=dash_usage_reward, weight=0.8),
        'smart_movement_reward': RewTerm(func=smart_movement_reward, weight=1.2),
        'weapon_management_reward': RewTerm(func=weapon_management_reward, weight=1.0),
        
        # === ANTI-SPAM & BAD HABITS ===
        'penalize_attack_spam': RewTerm(func=penalize_attack_spam, weight=0.8),
        'holding_more_than_3_keys': RewTerm(func=holding_more_than_3_keys, weight=0.5),
        'punish_predictable_patterns': RewTerm(func=punish_predictable_patterns, weight=0.8),
    }

    signal_subscriptions = {
        'on_win_reward': ('win_signal', RewTerm(func=on_win_reward, weight=500)),
        'on_knockout_reward': ('knockout_signal', RewTerm(func=on_knockout_reward, weight=-100)),
        'on_combo_reward': ('hit_during_stun', RewTerm(func=on_combo_reward, weight=25)),  # Increased for combo importance
        'on_equip_reward': ('weapon_equip_signal', RewTerm(func=on_equip_reward, weight=12)),
        'on_drop_reward': ('weapon_drop_signal', RewTerm(func=on_drop_reward, weight=-15)),
    }

    return RewardManager(reward_functions, signal_subscriptions)



# -------------------------------------------------------------------------
# ----------------------- CURRICULUM LEARNING SYSTEM ----------------------
# -------------------------------------------------------------------------

class TrainingPhase(Enum):
    """Training curriculum phases"""
    FUNDAMENTALS = 0    # Learn basic movement and attacks
    AGGRESSION = 1      # Learn damage dealing and combos
    ADVANCED = 2        # Learn spacing, adaptation, mind games

class CurriculumManager:
    """
    Manages training curriculum to progressively increase difficulty
    """
    def __init__(self, initial_phase: TrainingPhase = TrainingPhase.FUNDAMENTALS):
        self.current_phase = initial_phase
        self.phase_timesteps = {
            TrainingPhase.FUNDAMENTALS: 1_000_000,
            TrainingPhase.AGGRESSION: 2_500_000,
            TrainingPhase.ADVANCED: float('inf')
        }
        self.total_timesteps = 0
    
    def update(self, timesteps: int) -> bool:
        """
        Update curriculum based on training progress
        Returns True if phase changed
        """
        self.total_timesteps += timesteps
        old_phase = self.current_phase
        
        if self.total_timesteps >= self.phase_timesteps[TrainingPhase.FUNDAMENTALS]:
            if self.total_timesteps >= self.phase_timesteps[TrainingPhase.AGGRESSION]:
                self.current_phase = TrainingPhase.ADVANCED
            else:
                self.current_phase = TrainingPhase.AGGRESSION
        
        return old_phase != self.current_phase
    
    def get_opponent_config(self) -> dict:
        """
        Returns opponent configuration based on current phase
        """
        if self.current_phase == TrainingPhase.FUNDAMENTALS:
            return {
                'self_play': (4, None),  # Less self-play initially
                'constant_agent': (2.0, partial(ConstantAgent)),
                'based_agent': (3.5, partial(BasedAgent)),
            }
        elif self.current_phase == TrainingPhase.AGGRESSION:
            return {
                'self_play': (8, None),  # More self-play
                'constant_agent': (1.0, partial(ConstantAgent)),
                'based_agent': (1.0, partial(BasedAgent)),
            }
        else:  # ADVANCED
            return {
                'self_play': (12, None),  # Maximum self-play diversity
                'constant_agent': (0.5, partial(ConstantAgent)),
                'based_agent': (0.5, partial(BasedAgent)),
            }
    
    def get_phase_message(self) -> str:
        """Returns descriptive message about current phase"""
        messages = {
            TrainingPhase.FUNDAMENTALS: "Phase 1: Learning fundamentals (movement, basic attacks)",
            TrainingPhase.AGGRESSION: "Phase 2: Developing aggression (combos, damage optimization)",
            TrainingPhase.ADVANCED: "Phase 3: Advanced tactics (spacing, adaptation, mind games)"
        }
        return messages[self.current_phase]


# --------------------------------------------------------------------------------
# --------------------------- REWARD TARGET SYSTEM -------------------------------
# --------------------------------------------------------------------------------

class RewardTargets:
    """Expected reward ranges for different performance levels"""
    
    # Per Episode (full match) targets
    EXCELLENT_WIN = 300      # Dominant victory with minimal damage taken
    GOOD_WIN = 200           # Solid win
    CLOSE_WIN = 100          # Narrow victory
    CLOSE_LOSS = -50         # Lost but played well
    BAD_LOSS = -150          # Got dominated
    TERRIBLE = -300          # Died multiple times, played recklessly
    
    # Key milestones
    WIN_BONUS = 250          # From win signal
    DEATH_PENALTY = -50      # From knockout signal
    GOOD_COMBO = 20          # 4-5 hit combo
    FIRST_BLOOD = 20         # First hit bonus
    
    @classmethod
    def get_performance_level(cls, total_reward: float) -> str:
        """Return performance description based on reward"""
        if total_reward >= cls.EXCELLENT_WIN:
            return "EXCELLENT - Dominated opponent!"
        elif total_reward >= cls.GOOD_WIN:
            return "GOOD - Solid victory"
        elif total_reward >= cls.CLOSE_WIN:
            return "OKAY - Close win"
        elif total_reward >= cls.CLOSE_LOSS:
            return "DECENT - Close loss"
        elif total_reward >= cls.BAD_LOSS:
            return "POOR - Got dominated"
        else:
            return "TERRIBLE - Very reckless play"

# --------------------------------------------------------------------------------
# ------------------------- REWARD MONITORING SYSTEM -----------------------------
# --------------------------------------------------------------------------------

class RewardMonitor:
    """
    Monitor, log, and save reward statistics during training
    """
    def __init__(self, window_size: int = 100, log_dir: str = "./checkpoints/logs"):
        self.window_size = window_size
        self.episode_rewards = []
        self.episode_lengths = []
        self.win_count = 0
        self.loss_count = 0
        self.avg_damage_dealt = []
        self.avg_damage_taken = []
        self.death_count = 0

        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, "training_stats.csv")

        # Initialize CSV file if it doesn't exist
        if not os.path.exists(self.log_file):
            with open(self.log_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "avg_reward", "min_reward", "max_reward", "std_reward",
                    "avg_episode_length", "win_rate", "total_games",
                    "deaths", "avg_damage_dealt", "avg_damage_taken", "kd_ratio",
                    "performance_level"
                ])

    def log_episode(self, total_reward: float, episode_length: int, 
                    won: bool, damage_dealt: float, damage_taken: float,
                    died: bool):
        """Log episode statistics"""
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(episode_length)
        self.avg_damage_dealt.append(damage_dealt)
        self.avg_damage_taken.append(damage_taken)
        
        if won:
            self.win_count += 1
        else:
            self.loss_count += 1
            
        if died:
            self.death_count += 1
        
        # Keep only recent episodes
        if len(self.episode_rewards) > self.window_size:
            self.episode_rewards.pop(0)
            self.episode_lengths.pop(0)
            self.avg_damage_dealt.pop(0)
            self.avg_damage_taken.pop(0)
    
    def get_statistics(self) -> dict:
        """Get current statistics"""
        if not self.episode_rewards:
            return {}
        
        total_games = self.win_count + self.loss_count
        win_rate = (self.win_count / total_games * 100) if total_games > 0 else 0
        
        return {
            'avg_reward': np.mean(self.episode_rewards),
            'min_reward': np.min(self.episode_rewards),
            'max_reward': np.max(self.episode_rewards),
            'std_reward': np.std(self.episode_rewards),
            'avg_episode_length': np.mean(self.episode_lengths),
            'win_rate': win_rate,
            'total_games': total_games,
            'deaths': self.death_count,
            'avg_damage_dealt': np.mean(self.avg_damage_dealt),
            'avg_damage_taken': np.mean(self.avg_damage_taken),
            'kd_ratio': np.mean(self.avg_damage_dealt) / max(np.mean(self.avg_damage_taken), 1.0)
        }
    
    def _save_statistics(self, stats, performance):
        """Internal: append stats to CSV"""
        with open(self.log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                stats['avg_reward'],
                stats['min_reward'],
                stats['max_reward'],
                stats['std_reward'],
                stats['avg_episode_length'],
                stats['win_rate'],
                stats['total_games'],
                stats['deaths'],
                stats['avg_damage_dealt'],
                stats['avg_damage_taken'],
                stats['kd_ratio'],
                performance
            ])

    def print_statistics(self):
        """Print formatted statistics and log to file"""
        stats = self.get_statistics()
        if not stats:
            print("No statistics available yet")
            return
        
        print("\n" + "="*70)
        print("TRAINING STATISTICS (Last {} episodes)".format(self.window_size))
        print("="*70)
        print(f"Average Reward:        {stats['avg_reward']:>8.2f}")
        print(f"Reward Range:          {stats['min_reward']:>8.2f} to {stats['max_reward']:>8.2f}")
        print(f"Reward Std Dev:        {stats['std_reward']:>8.2f}")
        print(f"Avg Episode Length:    {stats['avg_episode_length']:>8.0f} steps")
        print(f"\nWin Rate:              {stats['win_rate']:>7.1f}%")
        print(f"Total Games:           {stats['total_games']:>8d}")
        print(f"Deaths:                {stats['deaths']:>8d}")
        print(f"\nAvg Damage Dealt:      {stats['avg_damage_dealt']:>8.2f}")
        print(f"Avg Damage Taken:      {stats['avg_damage_taken']:>8.2f}")
        print(f"K/D Ratio:             {stats['kd_ratio']:>8.2f}")
        print("="*70)
        
        performance = RewardTargets.get_performance_level(stats['avg_reward'])
        
        print(f"Performance Level: {performance}")
        print("="*70 + "\n")

        # Save the same stats to CSV
        self._save_statistics(stats, performance)


# --------------------------------------------------------------------------------
# ------------------------- REWARD MONITORING CALLBACK ---------------------------
# --------------------------------------------------------------------------------

class RewardMonitorCallback(BaseCallback):
    """
    Callback to log episode statistics to RewardMonitor during training
    """
    def __init__(self, reward_monitor: RewardMonitor, print_freq: int = 10000, verbose=0):
        super().__init__(verbose)
        self.reward_monitor = reward_monitor
        self.print_freq = print_freq
        self.last_print = 0
        
        # Track current episode stats
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.current_damage_dealt = 0
        self.current_damage_taken = 0
        
    def _on_step(self) -> bool:
        """Called at each step of training"""
        
        # Accumulate reward and length
        self.current_episode_reward += self.locals.get("rewards", [0])[0]
        self.current_episode_length += 1
        
        # Try to get damage info from the info dict
        infos = self.locals.get("infos", [{}])
        if len(infos) > 0:
            info = infos[0]
            self.current_damage_dealt += info.get("damage_dealt_step", 0)
            self.current_damage_taken += info.get("damage_taken_step", 0)
        
        # Check if episode is done
        dones = self.locals.get("dones", [False])
        if len(dones) > 0 and dones[0]:
            # Episode completed - extract final info
            info = infos[0] if len(infos) > 0 else {}
            
            # Determine if won (you may need to adjust these keys based on your environment)
            won = info.get("winner", False) or info.get("won", False)
            died = info.get("died", False) or info.get("ko", False)
            
            # If damage wasn't tracked per-step, try to get totals from final info
            damage_dealt = self.current_damage_dealt if self.current_damage_dealt > 0 else info.get("total_damage_dealt", 0)
            damage_taken = self.current_damage_taken if self.current_damage_taken > 0 else info.get("total_damage_taken", 0)
            
            # Log to monitor
            self.reward_monitor.log_episode(
                total_reward=self.current_episode_reward,
                episode_length=self.current_episode_length,
                won=won,
                damage_dealt=damage_dealt,
                damage_taken=damage_taken,
                died=died
            )
            
            # Reset episode tracking
            self.current_episode_reward = 0
            self.current_episode_length = 0
            self.current_damage_dealt = 0
            self.current_damage_taken = 0
        
        # Print statistics periodically
        if self.num_timesteps - self.last_print >= self.print_freq:
            self.reward_monitor.print_statistics()
            self.last_print = self.num_timesteps
        
        return True


# -------------------------------------------------------------------------
# --------------------------- HELPER FUNCTIONS ----------------------------
# -------------------------------------------------------------------------

def create_tournament_agent(
    agent_type: str = "simple",
    load_path: Optional[str] = None
) -> Agent:
    """
    Factory function to create tournament-ready agents
    
    Args:
        agent_type: "simple", "custom", "recurrent", or "standard"
        load_path: Path to checkpoint to resume training
    
    Returns:
        Configured agent instance
    """
    if agent_type == "simple":
        # Simple standard agent - most stable option
        return SB3Agent(sb3_class=PPO, file_path=load_path)
    elif agent_type == "custom":
        return CustomAgent(
            sb3_class=PPO, 
            extractor=MLPExtractor,
            file_path=load_path
        )
    elif agent_type == "recurrent":
        return RecurrentPPOAgent(file_path=load_path)
    elif agent_type == "standard":
        return SB3Agent(sb3_class=PPO, file_path=load_path)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}. Choose from: simple, custom, recurrent, standard")


def setup_training_configuration(
    agent: Agent,
    reward_monitor: Optional[RewardMonitor] = None,
    curriculum_enabled: bool = True,
    run_name: str = "tournament_run",
    save_freq: int = 100_000,
    max_checkpoints: int = 500,
    monitor_print_freq: int = 10000
) -> tuple:
    """
    Sets up complete training configuration
    
    Returns:
        (reward_manager, selfplay_handler, save_handler, opponent_cfg, curriculum, callbacks)
    """
    # Reward manager
    reward_manager = gen_reward_manager()
    
    # Curriculum manager
    curriculum = CurriculumManager() if curriculum_enabled else None
    
    # Self-play handler - just pass the agent class
    selfplay_handler = SelfPlayRandom(
        partial(type(agent))  # Just the agent class, no extra kwargs
    )
    
    # Save handler
    save_handler = SaveHandler(
        agent=agent,
        save_freq=save_freq,
        max_saved=max_checkpoints,
        save_path='checkpoints',
        run_name=run_name,
        mode=SaveHandlerMode.FORCE
    )
    
    # Opponent configuration
    if curriculum_enabled:
        opponent_spec = curriculum.get_opponent_config()
        opponent_spec['self_play'] = (opponent_spec['self_play'][0], selfplay_handler)
    else:
        opponent_spec = {
            'self_play': (12, selfplay_handler),
            'constant_agent': (0.5, partial(ConstantAgent)),
            'based_agent': (1.0, partial(BasedAgent)),
        }
    
    opponent_cfg = OpponentsCfg(opponents=opponent_spec)
    
    # Setup callbacks
    callbacks = []
    if reward_monitor:
        callbacks.append(RewardMonitorCallback(reward_monitor, print_freq=monitor_print_freq))
    
    return reward_manager, selfplay_handler, save_handler, opponent_cfg, curriculum, callbacks


# -------------------------------------------------------------------------
# ----------------------------- MAIN FUNCTION -----------------------------
# -------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 70)
    print("AI SQUARED TOURNAMENT - TRAINING CONFIGURATION")
    print("=" * 70)
    
    # ==================== CONFIGURATION ====================
    
    AGENT_TYPE = "simple"
    TOTAL_TRAINING_TIMESTEPS = 8_000_000
    TRAINING_BATCH_SIZE = 100_000
    CURRICULUM_ENABLED = True
    RESUME_FROM = "rl-model.zip"
    # RESUME_FROM = None
    RUN_NAME = "tournament_safe_attack_v1"
    SAVE_FREQUENCY = 100_000
    MAX_CHECKPOINTS = 500
    
    # Monitoring
    ENABLE_MONITORING = True
    MONITOR_INTERVAL = 10_000  # Print stats every 10k steps
    
    print(f"\nConfiguration:")
    print(f"  Agent Type: {AGENT_TYPE}")
    print(f"  Training Timesteps: {TOTAL_TRAINING_TIMESTEPS:,}")
    print(f"  Curriculum Learning: {CURRICULUM_ENABLED}")
    print(f"  Monitoring: {ENABLE_MONITORING}")
    print(f"  Run Name: {RUN_NAME}")
    print("="*70)
    
    # ==================== SETUP ====================
    
    print("\n[1/6] Creating agent...")
    my_agent = create_tournament_agent(
        agent_type=AGENT_TYPE,
        load_path=RESUME_FROM
    )
    print(f"✓ {AGENT_TYPE.capitalize()} agent created")
    
    print("\n[2/6] Setting up monitoring...")
    reward_monitor = RewardMonitor(window_size=100) if ENABLE_MONITORING else None
    if reward_monitor:
        print("✓ Reward monitoring enabled")
        print(f"✓ Statistics will be saved to: {reward_monitor.log_file}")
    
    print("\n[3/6] Setting up training configuration...")
    reward_manager, selfplay_handler, save_handler, opponent_cfg, curriculum, callbacks = setup_training_configuration(
        agent=my_agent,
        reward_monitor=reward_monitor,
        curriculum_enabled=CURRICULUM_ENABLED,
        run_name=RUN_NAME,
                save_freq=SAVE_FREQUENCY,
        max_checkpoints=MAX_CHECKPOINTS,
        monitor_print_freq=MONITOR_INTERVAL
    )
    
    print("✓ Reward manager configured with attack boundary safety")
    print("✓ Self-play handler initialized")
    print("✓ Save handler configured")
    print("✓ Opponent configuration set")
    
    if CURRICULUM_ENABLED:
        print(f"✓ Curriculum learning enabled")
        print(f"  {curriculum.get_phase_message()}")
    
    if callbacks:
        print(f"✓ Monitoring callbacks configured")
    
    # ==================== TRAINING LOOP ====================
    
    print("\n[4/6] Starting training...")
    print("="*70)
    
    if CURRICULUM_ENABLED:
        timesteps_trained = 0
        
        while timesteps_trained < TOTAL_TRAINING_TIMESTEPS:
            remaining = TOTAL_TRAINING_TIMESTEPS - timesteps_trained
            batch_size = min(TRAINING_BATCH_SIZE, remaining)
            
            print(f"\n[Training Progress: {timesteps_trained:,} / {TOTAL_TRAINING_TIMESTEPS:,}]")
            print(f"Current Phase: {curriculum.get_phase_message()}")
            
            # Train for this batch
            try:
                # Try to pass callbacks if train() supports it
                train(
                    my_agent,
                    reward_manager,
                    save_handler,
                    opponent_cfg,
                    CameraResolution.LOW,
                    train_timesteps=batch_size,
                    train_logging=TrainLogging.PLOT,
                    callbacks=callbacks  # Pass callbacks here
                )
            except TypeError:
                # If train() doesn't support callbacks parameter, try without it
                print("Note: train() doesn't support callbacks parameter, using alternative approach")
                train(
                    my_agent,
                    reward_manager,
                    save_handler,
                    opponent_cfg,
                    CameraResolution.LOW,
                    train_timesteps=batch_size,
                    train_logging=TrainLogging.PLOT
                )
            
            timesteps_trained += batch_size
            
            # Update curriculum
            phase_changed = curriculum.update(batch_size)
            
            if phase_changed:
                print(f"\n{'!'*70}")
                print(f"PHASE TRANSITION!")
                print(f"{curriculum.get_phase_message()}")
                print(f"{'!'*70}\n")
                
                # Update opponent configuration for new phase
                new_opponent_spec = curriculum.get_opponent_config()
                new_opponent_spec['self_play'] = (new_opponent_spec['self_play'][0], selfplay_handler)
                opponent_cfg = OpponentsCfg(opponents=new_opponent_spec)
    
    else:
        # Standard training without curriculum
        try:
            train(
                my_agent,
                reward_manager,
                save_handler,
                opponent_cfg,
                CameraResolution.LOW,
                train_timesteps=TOTAL_TRAINING_TIMESTEPS,
                train_logging=TrainLogging.PLOT,
                callbacks=callbacks
            )
        except TypeError:
            print("Note: train() doesn't support callbacks parameter, using alternative approach")
            train(
                my_agent,
                reward_manager,
                save_handler,
                opponent_cfg,
                CameraResolution.LOW,
                train_timesteps=TOTAL_TRAINING_TIMESTEPS,
                train_logging=TrainLogging.PLOT
            )
    
    # ==================== TRAINING COMPLETE ====================
    
    print("\n" + "="*70)
    print("[5/6] Training complete!")
    print("="*70)
    
    if ENABLE_MONITORING:
        print("\nFinal Statistics:")
        reward_monitor.print_statistics()
    
    # ==================== FINAL SAVE ====================
    
    print("\n[6/6] Saving final model...")
    final_path = f"checkpoints/{RUN_NAME}/final_model.zip"
    my_agent.save(final_path)
    print(f"✓ Final model saved to: {final_path}")
    
    # ==================== SUMMARY ====================
    
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"Total timesteps trained: {TOTAL_TRAINING_TIMESTEPS:,}")
    print(f"Checkpoints saved: Every {SAVE_FREQUENCY:,} steps")
    print(f"Final model location: {final_path}")
    
    if CURRICULUM_ENABLED:
        print(f"\nCurriculum Progress:")
        print(f"  Final Phase: {curriculum.get_phase_message()}")
    
    if ENABLE_MONITORING:
        stats = reward_monitor.get_statistics()
        if stats:
            print(f"\nFinal Performance Metrics:")
            print(f"  Average Reward: {stats['avg_reward']:.2f}")
            print(f"  Win Rate: {stats['win_rate']:.1f}%")
            print(f"  K/D Ratio: {stats['kd_ratio']:.2f}")
            print(f"  Performance: {RewardTargets.get_performance_level(stats['avg_reward'])}")
            print(f"\nDetailed logs saved to: {reward_monitor.log_file}")
    
    print("\n" + "="*70)
    print("Training complete! Your agent is ready for tournament play.")
    print("="*70)