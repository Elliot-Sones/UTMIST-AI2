from typing import Optional

from environment.environment import RenderMode, CameraResolution
from environment.agent import run_real_time_match, RandomAgent
from user.train_agent import (
    UserInputAgent,
    BasedAgent,
    ConstantAgent,
    ClockworkAgent,
    SB3Agent,
)  # add anymore custom Agents (from train_agent.py) here as needed
from user.my_agent import SubmittedAgent
import pygame
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))


pygame.init()

# ---------------------------------------------------------------------------
# Pick how each side should behave: "human", "rules", "rl", or "random".
# Supply a checkpoint path only when using the RL controller.
# ---------------------------------------------------------------------------
AGENT_1_TYPE = "rl"
AGENT_1_MODEL_PATH = "/rl_model_700000_steps.zip"
AGENT_2_TYPE = "rules"
AGENT_2_MODEL_PATH = None


def build_agent(agent_type: str, model_path: Optional[str] = None):
    agent_type = agent_type.lower()
    if agent_type == "human":
        return UserInputAgent()
    if agent_type == "rules":
        return SubmittedAgent(mode="rules")
    if agent_type == "rl":
        return SubmittedAgent(file_path=model_path, mode="rl")
    if agent_type == "random":
        return RandomAgent()
    raise ValueError(f"Unsupported agent type '{agent_type}'.")


agent_1 = build_agent(AGENT_1_TYPE, model_path=AGENT_1_MODEL_PATH)
agent_2 = build_agent(AGENT_2_TYPE, model_path=AGENT_2_MODEL_PATH)

match_time = 99999

# Run a single real-time match
print(f"agent_1: {AGENT_1_TYPE}")
print(f"agent_2: {AGENT_2_TYPE}")
run_real_time_match(
    agent_1=agent_1,
    agent_2=agent_2,
    max_timesteps=30 * match_time,  # Match time in frames (adjust as needed)
    resolution=CameraResolution.LOW,
)
