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

# -------------------------------------------------------------------------
# ----------------------------- AGENT CLASSES -----------------------------
# -------------------------------------------------------------------------

class SB3Agent(Agent):
    '''
    SB3Agent:
    - Defines an AI Agent that takes an SB3 class input for specific SB3 algorithm (e.g. PPO, SAC)
    '''
    def __init__(
            self,
            sb3_class: Optional[Type[BaseAlgorithm]] = PPO,
            file_path: Optional[str] = None
    ):
        self.sb3_class = sb3_class
        super().__init__(file_path)

    def _initialize(self) -> None:
        if self.file_path is None:
            self.model = self.sb3_class(
                "MlpPolicy", 
                self.env, 
                verbose=0, 
                n_steps=30*90*3, 
                batch_size=128, 
                ent_coef=0.01,
                learning_rate=3e-4,
                max_grad_norm=0.5,  # Prevent exploding gradients
                normalize_advantage=True,  # Normalize advantages for stability
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
    env: WarehouseBrawl,
    mode: RewardMode = RewardMode.SYMMETRIC,
) -> float:
    """
    Core damage-based reward with multiple modes
    """
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]

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


def weapon_management_reward(env: WarehouseBrawl) -> float:
    """
    Reward smart weapon pickup/throw decisions
    """
    player = env.objects["player"]
    opponent = env.objects["opponent"]
    
    if not is_player_throwing(player):
        return 0.0
    
    # Check current weapon (you may need to adjust attribute name)
    current_weapon = getattr(player, 'weapon', "Punch")
    
    # Reward picking up good weapons
    if current_weapon == "Punch":
        # Trying to pick up weapon
        return 0.8
    
    # Reward throwing weapon at opponent
    dist = np.linalg.norm([
        player.body.position.x - opponent.body.position.x,
        player.body.position.y - opponent.body.position.y
    ])
    
    if dist < 3.0:  # Close enough to hit
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
    
def on_equip_reward(env: WarehouseBrawl, agent: str) -> float:
    """Weapon pickup reward"""
    if agent == "player":
        weapon = env.objects["player"].weapon
        if weapon == "Hammer":
            return 2.5
        elif weapon == "Spear":
            return 1.5
    return 0.0

def on_drop_reward(env: WarehouseBrawl, agent: str) -> float:
    """Weapon drop penalty"""
    if agent == "player":
        if env.objects["player"].weapon == "Punch":
            return -1.5
    return 0.0

# --------------------------------------------------------------------------------
# ----------------------------- HELPER FUNCTIONS ---------------------------------
# --------------------------------------------------------------------------------

def is_player_attacking(player) -> bool:
    """
    Helper to check if player is attacking without state dependency
    
    Action indices:
    - 7: 'j' (Light Attack)
    - 8: 'k' (Heavy Attack)
    """
    if hasattr(player, 'cur_action'):
        action = player.cur_action
        if len(action) > 8:
            # Check light attack (j) and heavy attack (k)
            light_attack = action[7] > 0.5
            heavy_attack = action[8] > 0.5
            return light_attack or heavy_attack
    return False

def is_player_jumping(player) -> bool:
    """
    Helper to check if player is jumping
    
    Action index:
    - 4: 'space' (Jump)
    """
    # Method 1: Check space key (index 4)
    if hasattr(player, 'cur_action'):
        action = player.cur_action
        if len(action) > 4 and action[4] > 0.5:
            return True
    
    # Method 2: Check upward velocity
    if hasattr(player.body, 'velocity') and player.body.velocity.y > 0.5:
        return True
    
    return False

def is_player_dashing(player) -> bool:
    """
    Helper to check if player is dashing/dodging
    
    Action index:
    - 6: 'l' (Dash/Dodge)
    """
    if hasattr(player, 'cur_action'):
        action = player.cur_action
        if len(action) > 6 and action[6] > 0.5:
            return True
    return False

def is_player_moving(player) -> bool:
    """
    Helper to check if player is moving horizontally
    
    Action indices:
    - 1: 'a' (Left)
    - 3: 'd' (Right)
    """
    if hasattr(player, 'cur_action'):
        action = player.cur_action
        if len(action) > 3:
            moving_left = action[1] > 0.5
            moving_right = action[3] > 0.5
            return moving_left or moving_right
    return False

def is_player_throwing(player) -> bool:
    """
    Helper to check if player is picking up/throwing weapon
    
    Action index:
    - 5: 'h' (Pickup/Throw)
    """
    if hasattr(player, 'cur_action'):
        action = player.cur_action
        if len(action) > 5 and action[5] > 0.5:
            return True
    return False

def is_opponent_knocked_out(opponent) -> bool:
    """Helper to check if opponent is knocked out or respawning"""
    opponent_out_of_bounds = (
        abs(opponent.body.position.x) > 12.0 or 
        opponent.body.position.y < -3.0 or
        opponent.body.position.y > 10.0
    )
    opponent_stationary = (
        hasattr(opponent.body, 'velocity') and
        abs(opponent.body.velocity.x) < 0.1 and 
        abs(opponent.body.velocity.y) < 0.1 and
        opponent.body.position.y > 5.0
    )
    return opponent_out_of_bounds or opponent_stationary

def get_active_input_count(player) -> int:
    """Count how many buttons are being pressed"""
    if hasattr(player, 'cur_action'):
        action = player.cur_action
        return int((action > 0.5).sum())
    return 0

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
def self_preservation_reward(env: WarehouseBrawl) -> float:
    """
    Heavily penalize actions that lead to mutual deaths
    """
    player = env.objects["player"]
    opponent = env.objects["opponent"]
    
    # Track if both players are in danger
    player_danger = player.body.position.y < 0.5 or abs(player.body.position.x) > 8.0
    opponent_danger = opponent.body.position.y < 0.5 or abs(opponent.body.position.x) > 8.0
    
    # CRITICAL: If player is in danger, penalize being near opponent
    if player_danger:
        dist = np.linalg.norm([
            player.body.position.x - opponent.body.position.x,
            player.body.position.y - opponent.body.position.y
        ])
        
        if dist < 2.5:
            # Very close while in danger = likely mutual death
            return -8.0
        elif dist < 4.0:
            return -3.0
        else:
            # Reward escaping danger
            return 1.5
    
    # If opponent is in danger but player is safe, reward pressure
    if opponent_danger and not player_danger:
        return 2.0
    
    return 0.0


def ledge_safety_reward(env: WarehouseBrawl) -> float:
    """
    Prevent ledge deaths
    """
    player = env.objects["player"]
    
    # Define stage boundaries (adjust based on your stage)
    stage_x_limit = 9.0
    stage_y_limit = -1.0
    
    # Distance from edge
    edge_dist_x = stage_x_limit - abs(player.body.position.x)
    edge_dist_y = player.body.position.y - stage_y_limit
    
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


def situational_aggression_reward(env: WarehouseBrawl) -> float:
    """
    Only be aggressive when safe to do so (SAFE VERSION)
    """
    player = env.objects["player"]
    
    # Check if player is in safe position
    player_safe = (
        1.0 <= player.body.position.y <= 5.0 and
        abs(player.body.position.x) < 7.0
    )
    
    # Check if attacking
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


def opportunistic_positioning_reward(env: WarehouseBrawl) -> float:
    """
    Encourage good positioning during opponent respawn (SAFE VERSION)
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

def platform_awareness_reward(env: WarehouseBrawl) -> float:
    """
    Reward proper platform navigation and jumping (SAFE VERSION)
    """
    player = env.objects["player"]
    
    # Reward staying at platform height
    ideal_height_min = 1.5
    ideal_height_max = 4.5
    
    player_height = player.body.position.y
    
    # Reward being at good height
    if ideal_height_min <= player_height <= ideal_height_max:
        return 0.5
    
    # Strong penalty for being too low (falling)
    if player_height < 0.0:
        return -3.0
    elif player_height < ideal_height_min:
        return -1.0 * (ideal_height_min - player_height)
    
    # Mild penalty for being too high
    if player_height > ideal_height_max + 1.0:
        return -0.3
    
    # Reward jumping when low (trying to recover)
    if player_height < ideal_height_min and is_player_jumping(player):
        return 1.5
    
    return 0.0

def survive_reward(env: WarehouseBrawl) -> float:
    """
    Strong penalty for falling/dying
    """
    player = env.objects["player"]
    
    # Penalize being too low (about to fall)
    if player.body.position.y < -2.0:
        return -5.0
    elif player.body.position.y < 0.0:
        return -2.0
    
    # Reward staying on platforms
    if 1.0 <= player.body.position.y <= 4.0:
        return 0.3
    
    return 0.0

def penalize_attack_spam(env: WarehouseBrawl) -> float:
    """Penalize excessive attacking"""
    player = env.objects["player"]
    return -0.02 if is_player_attacking(player) else 0.0

# -------------------------------------------------------------------------
# ----------------------------- REWARD MANAGER -----------------------------
# -------------------------------------------------------------------------

def gen_reward_manager():
    """
    Enhanced tournament-optimized reward configuration
    With correct action indices
    """
    reward_functions = {
        # === SURVIVAL & SAFETY (HIGHEST PRIORITY) ===
        'survive_reward': RewTerm(func=survive_reward, weight=2.0),
        'self_preservation_reward': RewTerm(func=self_preservation_reward, weight=2.5),
        'ledge_safety_reward': RewTerm(func=ledge_safety_reward, weight=1.8),
        'danger_zone_reward': RewTerm(func=danger_zone_reward, weight=-2.0),
        
        # === PLATFORM AWARENESS ===
        'platform_awareness_reward': RewTerm(func=platform_awareness_reward, weight=1.5),
        
        # === SITUATIONAL AWARENESS ===
        'respect_knockout_reward': RewTerm(func=respect_knockout_reward, weight=1.2),
        'opportunistic_positioning_reward': RewTerm(func=opportunistic_positioning_reward, weight=0.8),
        
        # === SMART COMBAT ===
        'damage_interaction_reward': RewTerm(
            func=lambda env: damage_interaction_reward(env, mode=RewardMode.SYMMETRIC),
            weight=2.0
        ),
        'situational_aggression_reward': RewTerm(func=situational_aggression_reward, weight=1.0),
        
        # === COMBO & SKILL ===
        'combo_extension_reward': RewTerm(func=combo_extension_reward, weight=0.8),
        'hit_confirm_reward': RewTerm(func=hit_confirm_reward, weight=0.5),
        'neutral_win_reward': RewTerm(func=neutral_win_reward, weight=0.6),
        
        # === POSITIONING ===
        'optimal_distance_reward': RewTerm(func=optimal_distance_reward, weight=0.4),
        'stage_control_reward': RewTerm(func=stage_control_reward, weight=0.3),
        'edge_guard_reward': RewTerm(func=edge_guard_reward, weight=0.4),
        
        # === DEFENSIVE INTELLIGENCE ===
        'defensive_spacing_reward': RewTerm(func=defensive_spacing_reward, weight=0.4),
        
        # === MOBILITY & WEAPON ===
        'dash_usage_reward': RewTerm(func=dash_usage_reward, weight=0.6),
        'smart_movement_reward': RewTerm(func=smart_movement_reward, weight=0.4),
        'weapon_management_reward': RewTerm(func=weapon_management_reward, weight=0.5),
        
        # === ANTI-SPAM ===
        'penalize_attack_spam': RewTerm(func=penalize_attack_spam, weight=1.0),
        'holding_more_than_3_keys': RewTerm(func=holding_more_than_3_keys, weight=1.0),
        'punish_predictable_patterns': RewTerm(func=punish_predictable_patterns, weight=1.0),
        
        # === FIRST BLOOD ===
        'first_hit_reward': RewTerm(func=first_hit_reward, weight=0.8),
    }

    signal_subscriptions = {
        'on_win_reward': ('win_signal', RewTerm(func=on_win_reward, weight=250)),
        'on_knockout_reward': ('knockout_signal', RewTerm(func=on_knockout_reward, weight=-50)),
        'on_combo_reward': ('hit_during_stun', RewTerm(func=on_combo_reward, weight=12)),
        'on_equip_reward': ('weapon_equip_signal', RewTerm(func=on_equip_reward, weight=10)),
        'on_drop_reward': ('weapon_drop_signal', RewTerm(func=on_drop_reward, weight=-6)),
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
    curriculum_enabled: bool = True,
    run_name: str = "tournament_run",
    save_freq: int = 100_000,
    max_checkpoints: int = 500
) -> tuple:
    """
    Sets up complete training configuration
    
    Returns:
        (reward_manager, selfplay_handler, save_handler, opponent_cfg, curriculum)
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
    
    return reward_manager, selfplay_handler, save_handler, opponent_cfg, curriculum


# -------------------------------------------------------------------------
# ----------------------------- MAIN FUNCTION -----------------------------
# -------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 70)
    print("AI SQUARED TOURNAMENT - TRAINING CONFIGURATION")
    print("=" * 70)
    
    # ==================== CONFIGURATION ====================
    
    # Choose agent type: "custom", "recurrent", "standard", or "simple"
    # Use "simple" if you encounter NaN errors with "custom"
    AGENT_TYPE = "simple"  # Changed to simple for better stability
    
    # Training settings
    TOTAL_TRAINING_TIMESTEPS = 8_000_000  # 5M total timesteps
    TRAINING_BATCH_SIZE = 100_000  # Train in batches for curriculum updates
    
    # Enable curriculum learning (recommended)
    CURRICULUM_ENABLED = True
    
    # Resume from checkpoint (set to None to start fresh)
    RESUME_FROM = None  # e.g., 'checkpoints/tournament_run/rl_model_1200000_steps.zip'
    
    # Experiment name
    RUN_NAME = "tournament_elite_v1"
    
    # Save settings
    SAVE_FREQUENCY = 100_000
    MAX_CHECKPOINTS = 500
    
    print(f"\nConfiguration:")
    print(f"  Agent Type: {AGENT_TYPE}")
    print(f"  Training Timesteps: {TOTAL_TRAINING_TIMESTEPS:,}")
    print(f"  Curriculum Learning: {CURRICULUM_ENABLED}")
    print(f"  Resume From: {RESUME_FROM or 'Starting fresh'}")
    print(f"  Run Name: {RUN_NAME}")
    print("=" * 70)
    
    # ==================== AGENT CREATION ====================
    
    print("\n[1/5] Creating agent...")
    my_agent = create_tournament_agent(
        agent_type=AGENT_TYPE,
        load_path=RESUME_FROM
    )
    print(f"✓ {AGENT_TYPE.capitalize()} agent created")
    
    # ==================== TRAINING SETUP ====================
    
    print("\n[2/5] Setting up training configuration...")
    reward_manager, selfplay_handler, save_handler, opponent_cfg, curriculum = setup_training_configuration(
        agent=my_agent,
        curriculum_enabled=CURRICULUM_ENABLED,
        run_name=RUN_NAME,
        save_freq=SAVE_FREQUENCY,
        max_checkpoints=MAX_CHECKPOINTS
    )
    print("✓ Reward manager configured with 13 reward functions")
    print("✓ Self-play handler initialized")
    print("✓ Save handler configured")
    print("✓ Opponent configuration set")
    
    if CURRICULUM_ENABLED:
        print(f"✓ Curriculum learning enabled")
        print(f"  {curriculum.get_phase_message()}")
    
    # ==================== TRAINING LOOP ====================
    
    print("\n[3/5] Starting training...")
    print("=" * 70)
    
    if CURRICULUM_ENABLED:
        # Train with curriculum updates
        timesteps_trained = 0
        
        while timesteps_trained < TOTAL_TRAINING_TIMESTEPS:
            remaining = TOTAL_TRAINING_TIMESTEPS - timesteps_trained
            batch_size = min(TRAINING_BATCH_SIZE, remaining)
            
            print(f"\n[Training Progress: {timesteps_trained:,} / {TOTAL_TRAINING_TIMESTEPS:,}]")
            print(f"Current Phase: {curriculum.get_phase_message()}")
            
            # Train for this batch
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
                print(f"\n{'!' * 70}")
                print(f"PHASE TRANSITION!")
                print(f"{curriculum.get_phase_message()}")
                print(f"{'!' * 70}\n")
                
                # Update opponent configuration for new phase
                new_opponent_spec = curriculum.get_opponent_config()
                new_opponent_spec['self_play'] = (new_opponent_spec['self_play'][0], selfplay_handler)
                opponent_cfg = OpponentsCfg(opponents=new_opponent_spec)
    
    else:
        # Standard training without curriculum
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
    
    print("\n" + "=" * 70)
    print("[4/5] Training complete!")
    print("=" * 70)
    
    # ==================== FINAL SAVE ====================
    
    print("\n[5/5] Saving final model...")
    final_path = f"checkpoints/{RUN_NAME}/final_model.zip"
    my_agent.save(final_path)
    print(f"✓ Final model saved to: {final_path}")
    
    # ==================== SUMMARY ====================
    
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"Total timesteps trained: {TOTAL_TRAINING_TIMESTEPS:,}")
    print(f"Checkpoints saved: Every {SAVE_FREQUENCY:,} steps")
    print(f"Final model location: {final_path}")
    
    if CURRICULUM_ENABLED:
        print(f"\nCurriculum Progress:")
        print(f"  Final Phase: {curriculum.get_phase_message()}")