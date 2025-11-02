"""
Video Generation Callback: Periodic video generation for visual verification.

This callback generates videos of agent gameplay at regular intervals during training,
allowing you to visually verify that the agent is improving over time.
"""

import subprocess
from pathlib import Path
from stable_baselines3.common.callbacks import BaseCallback


class VideoGenerationCallback(BaseCallback):
    """
    Callback that generates gameplay videos at regular intervals.

    Generates short videos of the agent playing to verify behavior improvement.
    Useful for overnight training runs to visually confirm progress.
    """

    def __init__(
        self,
        video_freq: int = 500_000,  # Generate video every N steps
        video_dir: str = None,  # Directory to save videos
        num_episodes: int = 1,  # Number of episodes to record per checkpoint
        verbose: int = 1,
    ):
        super().__init__(verbose)

        self.video_freq = video_freq
        self.video_dir = Path(video_dir) if video_dir else Path("./videos")
        self.num_episodes = num_episodes
        self.last_video_step = 0

    def _on_training_start(self) -> None:
        """Called at the beginning of training."""
        self.video_dir.mkdir(parents=True, exist_ok=True)
        if self.verbose:
            print(f"✓ Video generation enabled: Recording {self.num_episodes} episode(s) every {self.video_freq:,} steps")
            print(f"  Videos will be saved to: {self.video_dir}")

    def _on_step(self) -> bool:
        """Called at each environment step."""
        # Check if we should generate a video
        if self.num_timesteps - self.last_video_step >= self.video_freq:
            self._generate_video()
            self.last_video_step = self.num_timesteps

        return True  # Continue training

    def _generate_video(self):
        """Generate a video of the current agent playing."""
        if self.verbose:
            print(f"\n📹 Generating video at {self.num_timesteps:,} steps...")

        # Save current model temporarily
        temp_model_path = self.video_dir / f"temp_model_{self.num_timesteps}.zip"
        self.model.save(temp_model_path)

        # Video output path
        video_path = self.video_dir / f"agent_gameplay_{self.num_timesteps:08d}.mp4"

        try:
            # Use the validation script to generate video
            # Note: This assumes validation script accepts model path and generates video
            result = subprocess.run(
                [
                    "python", "user/validate.py",
                    "--model", str(temp_model_path),
                    "--output", str(video_path),
                    "--episodes", str(self.num_episodes),
                ],
                capture_output=True,
                text=True,
                timeout=120,  # 2 minute timeout
            )

            if result.returncode == 0:
                if self.verbose:
                    print(f"  ✓ Video saved: {video_path}")
            else:
                if self.verbose:
                    print(f"  ⚠ Video generation failed: {result.stderr}")

        except subprocess.TimeoutExpired:
            if self.verbose:
                print(f"  ⚠ Video generation timed out")
        except Exception as e:
            if self.verbose:
                print(f"  ⚠ Video generation error: {e}")
        finally:
            # Clean up temporary model
            if temp_model_path.exists():
                temp_model_path.unlink()


def create_video_generation_callback(
    video_freq: int = 500_000,
    video_dir: str = None,
    num_episodes: int = 1,
    verbose: int = 1,
) -> VideoGenerationCallback:
    """
    Factory function to create VideoGenerationCallback.

    Args:
        video_freq: Generate video every N steps (default: 500,000)
        video_dir: Directory to save videos (default: ./videos)
        num_episodes: Number of episodes to record (default: 1)
        verbose: Verbosity level (default: 1)

    Returns:
        VideoGenerationCallback instance
    """
    return VideoGenerationCallback(
        video_freq=video_freq,
        video_dir=video_dir,
        num_episodes=num_episodes,
        verbose=verbose,
    )
