# # SUBMISSION: Agent
# This will be the Agent class we run in the 1v1. We've started you off with a functioning RL agent (`SB3Agent(Agent)`) and if-statement agent (`BasedAgent(Agent)`). Feel free to copy either to `SubmittedAgent(Agent)` then begin modifying.
# 
# Requirements:
# - Your submission **MUST** be of type `SubmittedAgent(Agent)`
# - Any instantiated classes **MUST** be defined within and below this code block.
# 
# Remember, your agent can be either machine learning, OR if-statement based. I've seen many successful agents arising purely from if-statements - give them a shot as well, if ML is too complicated at first!!
# 
# Also PLEASE ask us questions in the Discord server if any of the API is confusing. We'd be more than happy to clarify and get the team on the right track.
# Requirements:
# - **DO NOT** import any modules beyond the following code block. They will not be parsed and may cause your submission to fail validation.
# - Only write imports that have not been used above this code block
# - Only write imports that are from libraries listed here
# We're using PPO by default, but feel free to experiment with other Stable-Baselines 3 algorithms!

import os
import gdown
from typing import Optional, Literal
from environment.agent import Agent
from stable_baselines3 import PPO, A2C # Sample RL Algo imports
from sb3_contrib import RecurrentPPO # Importing an LSTM

# To run the sample TTNN model, you can uncomment the 2 lines below: 
# import ttnn
# from user.my_agent_tt import TTMLPPolicy


class RuleBasedAgent(Agent):
    """Deterministic fallback agent driven by simple heuristics."""

    def __init__(self):
        super().__init__(file_path=None)
        self._ticks = 0
        self._stage_half_width = 10.67 / 2  # keep players inside the arena

    def _gdown(self) -> Optional[str]:
        # No assets required for this agent.
        return None

    def _initialize(self) -> None:
        # Rule agent keeps no additional state.
        return

    def predict(self, obs):
        self._ticks += 1
        action = self.act_helper.zeros()

        # Pull structured data from the observation helper.
        my_pos = self.obs_helper.get_section(obs, "player_pos")
        opp_pos = self.obs_helper.get_section(obs, "opponent_pos")
        opp_state = self.obs_helper.get_section(obs, "opponent_state")

        # If we're drifting off stage, prioritise recovering.
        if my_pos[0] > self._stage_half_width:
            action = self.act_helper.press_keys(["a"], action)
        elif my_pos[0] < -self._stage_half_width:
            action = self.act_helper.press_keys(["d"], action)
        else:
            opponent_down = opp_state in (5, 11)
            if not opponent_down:
                horiz_gap = opp_pos[0] - my_pos[0]
                action = self.act_helper.press_keys(["d" if horiz_gap > 0 else "a"], action)

        # Soft jump cadence to contest air or recover from drops.
        should_jump = my_pos[1] > 1.6 or opp_pos[1] > my_pos[1]
        if should_jump and self._ticks % 8 == 0:
            action = self.act_helper.press_keys(["space"], action)

        # Throw quick attacks when in melee range.
        if abs(opp_pos[0] - my_pos[0]) < 1.5 and abs(opp_pos[1] - my_pos[1]) < 1.0:
            action = self.act_helper.press_keys(["j"], action)

        return action


class SubmittedAgent(Agent):
    """
    Hybrid tournament agent that can run PPO or a rule-based policy.

    What: wraps Stable-Baselines3 PPO for RL use and exposes a deterministic
    heuristic fallback in the same interface.
    Why: lets evaluation scripts flip between learning-based and handcrafted
    behaviour without touching call sites.
    How: build the requested controller during `_initialize` and delegate
    `predict` (and `learn`/`save` when relevant) to either SB3 or the rule agent.
    """

    # === Initialization & configuration =====================================
    def __init__(
        self,
        file_path: Optional[str] = None,
        mode: Literal["rl", "rules"] = "rl",
    ):
        self.mode = mode.lower()
        if self.mode not in {"rl", "rules"}:
            raise ValueError(f"Unsupported agent mode '{mode}'")

        super().__init__(file_path if self.mode == "rl" else None)

        # Stable-Baselines3 model handle (created in `_initialize`)
        self.model: Optional[PPO] = None
        self._rule_agent: Optional[RuleBasedAgent] = None

        # Default PPO network shape: deeper than stock to capture the game's dynamics.
        self._policy_name: str = "MlpPolicy"
        self._policy_kwargs = dict(
            net_arch=[dict(pi=[256, 256], vf=[256, 256])],
        )

        # Default PPO hyperparameters; tweak here before long training runs.
        self._training_defaults = dict(
            n_steps=30 * 90 * 4,      # four rounds of 3-second (90 frame) rollouts
            batch_size=256,           # high enough for stable gradient estimates
            gamma=0.99,               # standard discount for long-term rewards
            gae_lambda=0.95,          # balances bias/variance for GAE
            ent_coef=0.01,            # keeps exploration alive early on
            learning_rate=3e-4,       # reliable PPO learning rate
            verbose=0,
        )

        # Optional TTNN hook (left disabled by default for clarity)
        # self.mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))

    # === SB3 lifecycle hooks =================================================
    def _initialize(self) -> None:
        """
        Called once the base `Agent` knows about the environment interfaces.
        """
        if self.mode == "rules":
            self._rule_agent = RuleBasedAgent()
            self._rule_agent.get_env_info(self.env)
            return

        if self.file_path:
            self.model = self._load_trained_model(self.file_path)
        else:
            self.model = self._build_default_model()

        # Uncomment the block below to swap the SB3 MLP for a TTNN version.
        # mlp_state_dict = self.model.policy.features_extractor.model.state_dict()
        # self.tt_model = TTMLPPolicy(mlp_state_dict, self.mesh_device)
        # self.model.policy.features_extractor.model = self.tt_model
        # self.model.policy.vf_features_extractor.model = self.tt_model
        # self.model.policy.pi_features_extractor.model = self.tt_model

    def _build_default_model(self) -> PPO:
        """
        Create a fresh PPO policy bound to the current environment.
        """
        assert self.env is not None, "Environment must be set before building the model."

        model = PPO(
            policy=self._policy_name,
            env=self.env,
            policy_kwargs=self._policy_kwargs,
            **self._training_defaults,
        )

        # The underlying environment reference is no longer needed once PPO stores it.
        del self.env
        return model

    def _load_trained_model(self, file_path: str) -> PPO:
        """
        Load a previously trained PPO policy from disk.

        We avoid binding the live environment here so inference works without
        wrapping the custom Malachite env in Gym/Gymnasium shims.
        """
        return PPO.load(file_path)

    # === Runtime / competition interface ====================================
    def _gdown(self) -> Optional[str]:
        if self.mode == "rules":
            return None
        data_path = "rl-model.zip"
        if not os.path.isfile(data_path):
            print(f"Downloading {data_path}...")
            url = "https://drive.google.com/file/d/1JIokiBOrOClh8piclbMlpEEs6mj3H1HJ/view?usp=sharing"
            gdown.download(url, output=data_path, fuzzy=True)
        return data_path

    def predict(self, obs):
        """
        Choose an action for the current observation (deterministic by default).
        """
        if self.mode == "rules":
            assert self._rule_agent is not None, "Rule agent must be initialized."
            return self._rule_agent.predict(obs)

        action, _ = self.model.predict(obs, deterministic=True)
        return action

    def save(self, file_path: str) -> None:
        """
        Persist the full PPO model so it can be reloaded with `_load_trained_model`.
        """
        if self.mode == "rules":
            raise RuntimeError("Rule-based agent does not support saving checkpoints.")
        self.model.save(file_path)

    def learn(self, env, total_timesteps, log_interval: int = 4):
        """
        Train (or continue training) using the provided environment handle.
        """
        if self.mode == "rules":
            raise RuntimeError("Rule-based agent cannot be trained.")

        self.model.set_env(env)
        self.model.learn(total_timesteps=total_timesteps, log_interval=log_interval)
