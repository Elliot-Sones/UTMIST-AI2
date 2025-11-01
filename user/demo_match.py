from typing import Optional
import argparse

import pygame

from environment.environment import CameraResolution
from environment.agent import run_match
from user.train_agent import (
    UserInputAgent,
    BasedAgent,
    ConstantAgent,
    ClockworkAgent,
    SB3Agent,
)  # add anymore custom Agents (from train_agent.py) here as needed
from user.my_agent import SubmittedAgent

pygame.init()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run an offline WarehouseBrawl match and optionally record it."
    )
    parser.add_argument(
        "--player-agent",
        default="human",
        choices=["human", "rules", "rl"],
        help="Controller for the player slot.",
    )
    parser.add_argument(
        "--player-model",
        default=None,
        help="Checkpoint path when --player-agent=rl (e.g. rl_model_300000_steps.zip).",
    )
    parser.add_argument(
        "--opponent-agent",
        default="rl",
        choices=["human", "rules", "rl"],
        help="Controller for the opponent slot.",
    )
    parser.add_argument(
        "--opponent-model",
        default=None,
        help="Checkpoint path when --opponent-agent=rl.",
    )
    parser.add_argument(
        "--match-seconds",
        type=int,
        default=180,
        help="Match length in seconds (frames = 30 × seconds).",
    )
    parser.add_argument(
        "--resolution",
        default="low",
        choices=["low", "medium", "high"],
        help="Rendering resolution preset.",
    )
    parser.add_argument(
        "--video-path",
        default="tt_agent.mp4",
        help="Output video file; pass '-' to disable recording.",
    )
    return parser.parse_args()


def build_agent(agent_type: str, model_path: Optional[str] = None):
    agent_type = agent_type.lower()
    if agent_type == "human":
        return UserInputAgent()
    if agent_type == "rules":
        return SubmittedAgent(mode="rules")
    if agent_type == "rl":
        return SubmittedAgent(file_path=model_path, mode="rl")
    raise ValueError(f"Unsupported agent type '{agent_type}'.")


def main():
    args = parse_args()
    resolution_map = {
        "low": CameraResolution.LOW,
        "medium": CameraResolution.MEDIUM,
        "high": CameraResolution.HIGH,
    }

    agent_1 = build_agent(args.player_agent, model_path=args.player_model)
    agent_2 = build_agent(args.opponent_agent, model_path=args.opponent_model)

    print(f"player: {args.player_agent} ({args.player_model or 'no checkpoint'})")
    print(f"opponent: {args.opponent_agent} ({args.opponent_model or 'no checkpoint'})")
    video_path = None if args.video_path == "-" else args.video_path

    run_match(
        agent_1=agent_1,
        agent_2=agent_2,
        max_timesteps=30 * args.match_seconds,
        resolution=resolution_map[args.resolution],
        video_path=video_path,
    )


if __name__ == "__main__":
    main()
