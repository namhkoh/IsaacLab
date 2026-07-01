# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Standalone test runner for the Tomato Harvest environment.

Runs the environment with random, zero, or scripted actions to verify:
- Plant geometry renders correctly
- Spring physics keeps tomatoes in place
- State machine transitions work
- Metrics are logged correctly

Usage:
    isaaclab.bat -p .../run_tomato_harvest.py --num_envs 4 --max_steps 200
    isaaclab.bat -p .../run_tomato_harvest.py --num_envs 1 --max_steps 500 --zero_actions
    isaaclab.bat -p .../run_tomato_harvest.py --num_envs 1 --max_steps 500 --save_video
    isaaclab.bat -p .../run_tomato_harvest.py --num_envs 4 --max_steps 500 --scripted
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test the Tomato Harvest environment.")
parser.add_argument("--task", type=str, default="Isaac-Tomato-Harvest-Direct-v0", help="Gym env ID.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments.")
parser.add_argument("--max_steps", type=int, default=2000, help="Max steps per episode.")
parser.add_argument("--num_episodes", type=int, default=3, help="Number of episodes to run.")
parser.add_argument("--zero_actions", action="store_true", help="Use zero actions instead of scripted.")
parser.add_argument("--random_actions", action="store_true", help="Use random actions instead of scripted.")
parser.add_argument("--scripted", action="store_true", default=True, help="Use scripted heuristic policy (DEFAULT).")
parser.add_argument("--save_video", action="store_true", help="Save wrist camera as MP4.")
parser.add_argument("--video_path", type=str, default="tomato_harvest_test.mp4", help="Output video path.")
parser.add_argument("--target_tomato_idx", type=int, default=None, help="Optional fixed target tomato index for scripted runs.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest follows after AppLauncher."""

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectRLEnvCfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


class ScriptedHarvestPolicy:
    """Direct IK target controller for tomato harvesting.

    Sets exact world-space IK targets via env.set_ik_target_world() —
    the same approach used by the reference fruit_harvest_scene.py demo.
    No deltas, no scaling, no accumulation. The arm converges directly
    to the exact waypoint position via per-substep IK.

    Phase transitions are convergence-based: the arm moves to each
    waypoint and advances when it arrives (within threshold).

    Phases:
        0 - PRE_APPROACH: Standoff point above and in front of fruit.
        1 - APPROACH:     Exact fruit center position.
        2 - GRASP:        Hold at fruit, close gripper.
        3 - PULL:         Pull along the peduncle axis until the fruit detaches.
        4 - RETRACT:      Move the detached fruit away from the plant.
        5 - DELIVER:      Above basket rim.
        6 - RELEASE:      Open gripper to drop fruit.
    """

    PHASES = [
        # (name, gripper_open, converge_dist, min_steps)
        ("PRE_APPROACH", True,   0.025, 10),
        ("APPROACH",     True,   0.012, 10),
        ("GRASP",        False,  0.0,  130),
        ("PULL",         False,  0.0,   30),
        ("RETRACT",      False,  0.025, 20),
        ("DELIVER",      False,  0.025, 15),
        ("RELEASE",      True,   0.0,   60),
    ]
    GRASP_TIMEOUT_STEPS = 420
    PULL_TIMEOUT_STEPS = 360

    def __init__(self, env):
        self.env_unwrapped = env.unwrapped
        self.device = self.env_unwrapped.device
        self.num_envs = self.env_unwrapped.num_envs
        N = self.num_envs

        self.phase_idx = torch.zeros(N, device=self.device, dtype=torch.long)
        self.phase_step = torch.zeros(N, device=self.device, dtype=torch.long)
        self._last_target_w = None  # cache to avoid redundant set_ik_target calls

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.phase_idx[env_ids] = 0
        self.phase_step[env_ids] = 0
        self._last_target_w = None

    @property
    def scripted_phase(self):
        return self.phase_idx

    def compute(self) -> torch.Tensor:
        """Set IK target directly and return gripper-only actions."""
        uw = self.env_unwrapped
        N = self.num_envs
        actions = torch.zeros(N, 7, device=self.device)
        self.phase_step += 1

        # Current fingertip position in world frame
        ft_pos = uw._fingertip_world()

        # Use the fruit rest pose from the plant layout, not the live fruit pose.
        # Pull along the stem axis so detach comes from actual peduncle tension.
        env_ids = torch.arange(N, device=self.device)
        target_idx = uw.target_tomato_idx
        fruit_rest_w = uw.tomato_init_positions[target_idx] + uw.scene.env_origins
        fruit_live_w = uw._get_target_tomato_positions()
        anchor_w = uw.tomato_anchor_pos[env_ids, target_idx]
        basket_w = uw._get_basket_centers_w()

        pull_dir = fruit_rest_w - anchor_w
        pull_dir = pull_dir / pull_dir.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        phase_progress = torch.clamp(self.phase_step.float().unsqueeze(-1), max=180.0) / 180.0
        pull_distance = 0.035 + 0.020 * phase_progress

        # Build waypoints (exact world-space positions)
        standoff_wp = fruit_rest_w.clone()
        standoff_wp[:, 0] += 0.15   # 15cm toward robot
        standoff_wp[:, 2] += 0.05   # 5cm above fruit

        approach_wp = fruit_live_w.clone()
        grasp_wp = fruit_live_w.clone()
        grasp_wp[:, 0] -= 0.004     # slight squeeze bias toward the palm

        pull_wp = fruit_live_w + pull_distance * pull_dir

        retract_wp = pull_wp.clone()
        retract_wp[:, 0] += 0.10    # move detached fruit away from the truss
        retract_wp[:, 2] += 0.05

        above_basket = basket_w.clone()
        above_basket[:, 2] += 0.12  # above basket rim

        waypoints = [standoff_wp, approach_wp, grasp_wp, pull_wp, retract_wp, above_basket, above_basket]

        # Determine the active target per env and set IK target directly
        target_w = torch.zeros(N, 3, device=self.device)
        gripper_open = torch.ones(N, dtype=torch.bool, device=self.device)
        target_attached = uw.tomato_attached[env_ids, target_idx]

        for pi, (name, grip_open, conv_dist, min_steps) in enumerate(self.PHASES):
            mask = self.phase_idx == pi
            if not mask.any():
                continue

            target_w[mask] = waypoints[pi][mask]
            gripper_open[mask] = grip_open

            # Phase transition
            delta_w = waypoints[pi] - ft_pos
            dist = delta_w.norm(dim=-1)
            past_min = mask & (self.phase_step >= min_steps)

            grasped_now = uw._tomato_grasped
            if name == "GRASP":
                advance = past_min & grasped_now
                retry = mask & (self.phase_step >= self.GRASP_TIMEOUT_STEPS) & ~grasped_now
                if retry.any():
                    print("  [PHASE] GRASP retry -> PRE_APPROACH | grasp not confirmed")
                    self.phase_idx[retry] = 0
                    self.phase_step[retry] = 0
                    target_w[retry] = waypoints[0][retry]
                    gripper_open[retry] = True
                    continue
            elif name == "PULL":
                advance = past_min & ~target_attached
                retry = mask & (self.phase_step >= self.PULL_TIMEOUT_STEPS) & target_attached & ~grasped_now
                if retry.any():
                    print("  [PHASE] PULL retry -> PRE_APPROACH | lost grasp before detach")
                    self.phase_idx[retry] = 0
                    self.phase_step[retry] = 0
                    target_w[retry] = waypoints[0][retry]
                    gripper_open[retry] = True
                    continue
            elif conv_dist > 0:
                advance = past_min & (dist < conv_dist)
            else:
                advance = past_min

            if advance.any():
                next_pi = min(pi + 1, len(self.PHASES) - 1)
                next_name = self.PHASES[next_pi][0]
                d0 = dist[0].item() if dist.numel() > 0 else 0
                finger_gap = uw._robot.data.joint_pos[0, uw.finger_joint_ids].sum().item()
                grasped = uw._tomato_grasped[0].item()
                print(f"  [PHASE] {name} -> {next_name} | dist={d0:.4f} finger_gap={finger_gap:.4f} grasped={grasped}")
                self.phase_idx[advance] = next_pi
                self.phase_step[advance] = 0

        # Set the IK target directly on the env (bypasses delta action pipeline)
        uw.set_ik_target_world(target_w)

        # Actions: only gripper matters (position is handled by direct IK target)
        actions[:, 6] = torch.where(gripper_open, torch.ones(N, device=self.device),
                                     -torch.ones(N, device=self.device))
        return actions


def print_training_guide():
    """Print a guide on how to train policies for this environment."""
    guide = """
============================================================
HOW TO TRAIN A HARVEST POLICY
============================================================

1. VLA (Vision-Language-Action) — RECOMMENDED
   Uses a pretrained foundation model (pi0.5, OpenVLA, etc.)
   that takes camera images + language instruction and outputs
   7-D actions. No sim training needed.

   a) Run the VLA inference server (see run_pi05.py pattern):
      - Send wrist_rgb (128x128) + language prompt to VLA
      - VLA returns 7-D action (dx, dy, dz, drx, dry, drz, grip)
      - Prompt: "pick the ripe tomato and place it in the basket"

   b) The env already outputs camera observations:
      - obs["wrist_rgb"]  — 128x128 wrist-mounted camera
      - obs["scene_rgb"]  — 256x256 overview camera

   c) Fine-tuning (optional):
      - Collect demos with --scripted --save_video
      - Fine-tune VLA on (image, instruction, action) tuples
      - Use LeRobot or Open X-Embodiment data format

2. STATE-BASED RL (fastest to converge in sim)
   Uses the 36-D observation vector, no images.

   isaaclab.bat -p scripts/reinforcement_learning/skrl/train.py \\
     --task Isaac-Tomato-Harvest-Direct-v0 --num_envs 1024

   Config: agents/skrl_ppo_cfg.yaml (PPO with MLP policy)

3. VISION-BASED RL (CNN policy trained in sim)
   Uses wrist_rgb camera as input, slower to converge.

   - Create agents/skrl_camera_ppo_cfg.yaml with CNN encoder
   - skrl and rl_games both support CNN policies natively
   - The env observation dict already provides camera images

============================================================
"""
    print(guide)


def main():
    env_cfg: DirectRLEnvCfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)

    # Video recording
    video_writer = None
    if args_cli.save_video:
        try:
            import imageio
            video_writer = imageio.get_writer(args_cli.video_path, fps=30, codec="libx264")
            print(f"[Video] Recording to {args_cli.video_path}")
        except ImportError:
            print("[Video] imageio not available, skipping.")
            args_cli.save_video = False

    # Determine action mode: scripted is default unless overridden
    if args_cli.random_actions:
        action_mode = "random"
    elif args_cli.zero_actions:
        action_mode = "zero"
    else:
        action_mode = "scripted"

    # Scripted policy
    scripted_policy = None
    if action_mode == "scripted":
        scripted_policy = ScriptedHarvestPolicy(env)
        print("=" * 60)
        print("  SCRIPTED MODE ACTIVE — direct IK target policy")
        print("  The arm should converge to the exact fruit position.")
        print("  If using random/zero, pass --random_actions or --zero_actions")
        print("=" * 60)

    # Accumulate episode stats
    all_success = []
    all_damage = []
    all_steps = []
    all_peak_force = []

    for episode in range(args_cli.num_episodes):
        print(f"\n{'=' * 60}")
        print(f"Episode {episode + 1}/{args_cli.num_episodes}")
        print(f"  Task:       {args_cli.task}")
        print(f"  Num envs:   {args_cli.num_envs}")
        print(f"  Actions:    {action_mode}")
        print(f"  Max steps:  {args_cli.max_steps}")
        print(f"{'=' * 60}\n")

        obs, info = env.reset()
        if args_cli.target_tomato_idx is not None:
            target_count = int(env.unwrapped.tomato_init_positions.shape[0])
            forced_target = max(0, min(args_cli.target_tomato_idx, target_count - 1))
            env.unwrapped.target_tomato_idx[:] = forced_target
            print(f"  Fixed target tomato: {forced_target}")
        elif scripted_policy is not None and args_cli.num_envs == 1:
            robot_pos = env.unwrapped._robot.data.root_pose_w[0, :3]
            tomato_w = env.unwrapped.tomato_init_positions + env.unwrapped.scene.env_origins[0].unsqueeze(0)
            forced_target = int(torch.argmin(torch.norm(tomato_w - robot_pos.unsqueeze(0), dim=-1)).item())
            env.unwrapped.target_tomato_idx[:] = forced_target
            print(f"  Scripted target tomato: {forced_target}")
        if scripted_policy is not None:
            scripted_policy.reset()
        total_reward = 0.0
        done = False

        for step in range(args_cli.max_steps):
            if scripted_policy is not None:
                action = scripted_policy.compute()
            elif args_cli.zero_actions:
                action = torch.zeros(args_cli.num_envs, env.action_space.shape[-1], device=env.unwrapped.device)
            else:
                action = torch.randn(args_cli.num_envs, env.action_space.shape[-1], device=env.unwrapped.device) * 0.5

            obs, reward, terminated, truncated, info = env.step(action)

            step_reward = reward.mean().item() if torch.is_tensor(reward) else float(reward)
            total_reward += step_reward

            # Log every 20 steps
            if step % 20 == 0 and "log" in info:
                log = info["log"]
                phase_str = ""
                if scripted_policy is not None:
                    pi = int(scripted_policy.scripted_phase[0].item())
                    phase_name = scripted_policy.PHASES[min(pi, len(scripted_policy.PHASES) - 1)][0]
                    phase_str = f"{phase_name:>13s} | "
                ft_dist = log.get('mean_dist_to_tomato', 0)
                print(
                    f"  Step {step:4d} | reward: {step_reward:+.4f} | "
                    f"{phase_str}"
                    f"dist: {ft_dist:.4f} | "
                    f"grasp: {log.get('grasp_rate', 0):.2f} | "
                    f"harvest: {log.get('harvest_success_rate', 0):.2f} | "
                    f"grip_F: {log.get('peak_grip_force', 0):.2f}N"
                )

            # Save video frame
            if video_writer is not None and "wrist_rgb" in obs:
                frame = obs["wrist_rgb"]
                if torch.is_tensor(frame):
                    if frame.dim() == 4:
                        frame = frame[0]
                    frame = frame[:, :, :3]
                    if frame.dtype == torch.uint8:
                        frame = frame.cpu().numpy()
                    else:
                        frame = (frame.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                video_writer.append_data(frame)

            if torch.is_tensor(terminated):
                done = (terminated | truncated).any().item()
            else:
                done = terminated or truncated
            if done:
                break

        # Episode summary
        print(f"\n  Episode {episode + 1} finished at step {step + 1}. Total reward: {total_reward:.4f}")
        if "log" in info:
            log = info["log"]
            success = float(log.get("harvest_success_rate", 0))
            damage = float(log.get("damage_rate", 0))
            peak_f = float(log.get("peak_grip_force", 0))
            all_success.append(success)
            all_damage.append(damage)
            all_steps.append(step + 1)
            all_peak_force.append(peak_f)
            print(f"  Success rate: {success:.2%}")
            print(f"  Damage rate:  {damage:.2%}")
            print(f"  Peak grip F:  {peak_f:.2f} N")
            print(f"  Detach force: {float(log.get('mean_detach_force', 0)):.2f} N")

    # Final summary
    print(f"\n{'=' * 60}")
    print("BENCHMARK SUMMARY")
    print(f"{'=' * 60}")
    if all_success:
        print(f"  Episodes:          {args_cli.num_episodes}")
        print(f"  Mean success rate: {np.mean(all_success):.2%}")
        print(f"  Mean damage rate:  {np.mean(all_damage):.2%}")
        print(f"  Mean cycle steps:  {np.mean(all_steps):.1f}")
        print(f"  Mean peak grip F:  {np.mean(all_peak_force):.2f} N")
    print(f"{'=' * 60}\n")

    if video_writer is not None:
        video_writer.close()
        import os
        print(f"[Video] Saved to {os.path.abspath(args_cli.video_path)}")

    # Print training guide after scripted run
    if args_cli.scripted:
        print_training_guide()

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
