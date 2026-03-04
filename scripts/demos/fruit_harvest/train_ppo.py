# Copyright (c) 2024-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train a PPO agent for the fruit harvesting task using skrl.

This script trains a Proximal Policy Optimization (PPO) agent to control a
robot arm that harvests fruit from a simulated plant.  The environment is a
DirectRL environment built on top of IsaacLab.

The network architecture uses separate policy (Gaussian) and value
(deterministic) heads with three hidden layers (256-128-64) and ELU
activations.

Usage (from the IsaacLab root):
    isaaclab.bat -p scripts/demos/fruit_harvest/train_ppo.py --num_envs 64
    isaaclab.bat -p scripts/demos/fruit_harvest/train_ppo.py --num_envs 128 --max_iterations 2000

Launch Isaac Sim Simulator first.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

# -- CLI arguments -----------------------------------------------------------
parser = argparse.ArgumentParser(description="Train a PPO agent for fruit harvesting.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments.")
parser.add_argument("--max_iterations", type=int, default=1000, help="Maximum training iterations.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during training."
)
parser.add_argument(
    "--video_length", type=int, default=200, help="Length of the recorded video (in steps)."
)
parser.add_argument(
    "--video_interval", type=int, default=2000, help="Interval between video recordings (in steps)."
)
# Append AppLauncher CLI args (--headless, --device, etc.)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Always enable cameras when recording video
if args_cli.video:
    args_cli.enable_cameras = True

# Launch the Omniverse application
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time
from datetime import datetime

import gymnasium as gym
import torch
import torch.nn as nn

# -- Import the fruit-harvest task so that gym.register() is triggered -------
import isaaclab_tasks.direct.fruit_harvest  # noqa: F401

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

# -- skrl imports ------------------------------------------------------------
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed


# ---------------------------------------------------------------------------
# Neural-network definitions
# ---------------------------------------------------------------------------

class GaussianPolicy(GaussianMixin, Model):
    """Stochastic Gaussian policy network.

    Three hidden layers (256 -> 128 -> 64) with ELU activations.  The log
    standard deviation is a learnable parameter (state-independent).
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions: bool = False,
        clip_log_std: bool = True,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        reduction: str = "sum",
    ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, self.num_actions),
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), self.log_std_parameter, {}


class DeterministicValue(DeterministicMixin, Model):
    """Deterministic state-value network.

    Same hidden-layer structure as the policy (256 -> 128 -> 64) but outputs
    a single scalar value estimate.
    """

    def __init__(self, observation_space, action_space, device, clip_actions: bool = False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 1),
        )

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def main():
    """Set up environment, agent, and trainer, then run PPO training."""

    # -- Environment configuration -------------------------------------------
    env_cfg = parse_env_cfg(
        "Isaac-Fruit-Harvest-Direct-v0",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )

    # Create the gymnasium environment
    env = gym.make(
        "Isaac-Fruit-Harvest-Direct-v0",
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )

    # Wrap for video recording if requested
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join("runs", "fruit_harvest", "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Wrap the environment for skrl
    env = wrap_env(env, wrapper="isaaclab")

    device = env.device

    # -- Reproducibility -----------------------------------------------------
    set_seed(args_cli.seed)

    # -- Rollout memory ------------------------------------------------------
    rollout_steps = 24
    memory = RandomMemory(memory_size=rollout_steps, num_envs=env.num_envs, device=device)

    # -- Instantiate models --------------------------------------------------
    models = {
        "policy": GaussianPolicy(env.observation_space, env.action_space, device),
        "value": DeterministicValue(env.observation_space, env.action_space, device),
    }

    # -- PPO hyper-parameters ------------------------------------------------
    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg["rollouts"] = rollout_steps
    cfg["learning_epochs"] = 8
    cfg["mini_batches"] = 4
    cfg["discount_factor"] = 0.99
    cfg["lambda"] = 0.95
    cfg["learning_rate"] = 3e-4
    cfg["learning_rate_scheduler"] = "KLAdaptiveLR"
    cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
    cfg["grad_norm_clip"] = 1.0
    cfg["ratio_clip"] = 0.2
    cfg["value_clip"] = 0.2
    cfg["clip_predicted_values"] = True
    cfg["entropy_loss_scale"] = 0.0
    cfg["value_loss_scale"] = 2.0
    cfg["kl_threshold"] = 0.0
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
    cfg["value_preprocessor"] = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}

    # -- Logging / checkpointing ---------------------------------------------
    log_dir = os.path.join("runs", "fruit_harvest")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = f"{timestamp}_ppo_seed{args_cli.seed}"

    cfg["experiment"]["write_interval"] = 50
    cfg["experiment"]["checkpoint_interval"] = 500
    cfg["experiment"]["directory"] = log_dir
    cfg["experiment"]["experiment_name"] = experiment_name

    print(f"[INFO] Logging experiment in directory: {os.path.abspath(log_dir)}")
    print(f"[INFO] Experiment name: {experiment_name}")

    # -- Create the PPO agent ------------------------------------------------
    agent = PPO(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    # -- Trainer -------------------------------------------------------------
    total_timesteps = args_cli.max_iterations * rollout_steps * env.num_envs
    trainer_cfg = {
        "timesteps": total_timesteps,
        "headless": True,
    }
    trainer = SequentialTrainer(env=env, agents=agent, cfg=trainer_cfg)

    print(f"[INFO] Starting PPO training for {args_cli.max_iterations} iterations "
          f"({total_timesteps} total timesteps).")
    start_time = time.time()

    # -- Train ---------------------------------------------------------------
    trainer.train()

    elapsed = time.time() - start_time
    print(f"[INFO] Training completed in {elapsed:.1f} seconds.")

    # -- Cleanup -------------------------------------------------------------
    env.close()


if __name__ == "__main__":
    # Run the main training function
    main()
    # Close the simulator
    simulation_app.close()
