from typing import Optional

from environment.environment import RenderMode, CameraResolution
from environment.agent import run_match
from user.train_agent import (
    UserInputAgent,
    BasedAgent,
    ConstantAgent,
    ClockworkAgent,
    SB3Agent,
)  # add anymore custom Agents (from train_agent.py) here as needed
from user.my_agent import SubmittedAgent
import pygame

pygame.init()

# ---------------------------------------------------------------------------
# Configure which controller each slot should use.
# Options: "human" (keyboard), "rules" (heuristic agent), "rl" (PPO/learned).
# ---------------------------------------------------------------------------
PLAYER_AGENT_TYPE = "human"
OPPONENT_AGENT_TYPE = "rl"
OPPONENT_MODEL_PATH = "checkpoints/experiment_9/rl_model_56700_steps.zip"  # only used when type == "rl"


def build_agent(agent_type: str, model_path: Optional[str] = None):
    agent_type = agent_type.lower()
    if agent_type == "human":
        return UserInputAgent()
    if agent_type == "rules":
        return SubmittedAgent(mode="rules")
    if agent_type == "rl":
        return SubmittedAgent(file_path=model_path, mode="rl")
    raise ValueError(f"Unsupported agent type '{agent_type}'.")


agent_1 = build_agent(PLAYER_AGENT_TYPE)
agent_2 = build_agent(OPPONENT_AGENT_TYPE, model_path=OPPONENT_MODEL_PATH)

match_time = 99999

# Run a single real-time match
run_match(
    agent_1=agent_1,
    agent_2=agent_2,
    max_timesteps=30 * match_time,  # Match time in frames (adjust as needed)
    resolution=CameraResolution.LOW,
    video_path="tt_agent.mp4",  # NOTE: you can change the save path of the video here
)
