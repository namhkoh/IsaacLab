# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""pi0.5 inference on the fruit harvest environment.

Communicates with a pi0.5 server (openpi) running in WSL2 over WebSocket.
The server handles model inference (JAX/GPU), this script handles
environment stepping and observation/action bridging.

Supports two modes:
    - LIBERO (default): EE delta actions (7D), --env LIBERO on server
    - DROID: joint velocity actions (8D), --env DROID on server

Prerequisites:
    1. Start the pi0.5 server in WSL2:
       LIBERO: cd ~/openpi && uv run scripts/serve_policy.py --env LIBERO --port 8000
       DROID:  cd ~/openpi && uv run scripts/serve_policy.py --env DROID --port 8000

    2. Install client dependencies in Isaac Sim Python:
       python.bat -m pip install websockets msgpack

Usage (LIBERO):
    isaaclab.bat -p .../run_pi05.py --max_steps 500 --replan_steps 5

Usage (DROID):
    isaaclab.bat -p .../run_pi05.py --mode droid --action_repeat 4 --max_steps 500
"""

from __future__ import annotations

import argparse
import os

# -- Step 1: Parse args and launch SimulationApp BEFORE any isaaclab imports --
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run pi0.5 inference on fruit harvest env.")
parser.add_argument("--task", type=str, default="Isaac-Fruit-Harvest-Direct-v0", help="Gym env ID.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--host", type=str, default="localhost", help="pi0.5 server hostname.")
parser.add_argument("--port", type=int, default=8000, help="pi0.5 server port.")
parser.add_argument(
    "--instruction",
    type=str,
    default="pick the red sphere fruit from the plant and carefully place it into the purple basket",
    help="Task instruction for pi0.5 prompt.",
)
parser.add_argument("--max_steps", type=int, default=500, help="Max steps per episode.")
parser.add_argument("--replan_steps", type=int, default=5, help="Actions to execute per server query.")
parser.add_argument("--rotate_image", action="store_true", help="Rotate images 180 deg (LIBERO convention).")
parser.add_argument("--action_scale", type=float, default=1.0, help="Additional multiplier on action deltas.")
parser.add_argument("--mode", type=str, choices=["libero", "droid"], default="libero", help="pi0.5 model mode: libero (EE delta) or droid (joint velocity).")
parser.add_argument("--action_repeat", type=int, default=1, help="Repeat each model action N times (e.g. 4 for DROID 15Hz -> 60Hz env).")
parser.add_argument("--save_video", action="store_true", help="Save camera feed as MP4.")
parser.add_argument("--video_path", type=str, default="pi05_rollout.mp4", help="Output video path.")
parser.add_argument("--num_episodes", type=int, default=1, help="Number of episodes to run.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Force-enable cameras
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

# -- Step 2: Now safe to import isaaclab / isaaclab_tasks --
import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectRLEnvCfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# Try to import OpenCV for live camera view
_HAS_CV2 = False
try:
    import cv2

    _test_img = np.zeros((10, 10, 3), dtype=np.uint8)
    cv2.imshow("_test", _test_img)
    cv2.destroyAllWindows()
    _HAS_CV2 = True
except Exception:
    pass
if not _HAS_CV2:
    print("[Warning] cv2.imshow not available -- saving camera frames to disk instead.")


def _display_frame(obs: dict, cam_key: str, step: int, video_writer=None, save_video: bool = False):
    """Display or save camera frame."""
    if cam_key not in obs:
        return
    frame = obs[cam_key]
    if torch.is_tensor(frame):
        if frame.dim() == 4:
            frame = frame[0]
        frame_rgb = frame[:, :, :3]
        if frame_rgb.dtype == torch.uint8:
            frame = frame_rgb.cpu().numpy()
        else:
            frame = (frame_rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

    if _HAS_CV2:
        display = cv2.resize(frame, (384, 384), interpolation=cv2.INTER_NEAREST)
        display = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)
        cv2.imshow(f"pi0.5 - {cam_key}", display)
        cv2.waitKey(1)
    elif step % 50 == 0:
        try:
            from PIL import Image as _PILImage

            cam_dir = os.path.join(os.path.dirname(__file__), "camera_frames")
            os.makedirs(cam_dir, exist_ok=True)
            _PILImage.fromarray(np.ascontiguousarray(frame[:, :, :3], dtype=np.uint8)).save(
                os.path.join(cam_dir, f"{cam_key}_latest.png")
            )
        except Exception as e:
            if step == 0:
                print(f"[Warning] Could not save {cam_key} frame: {e}")

    if save_video and video_writer is not None and cam_key == "wrist_rgb":
        video_writer.append_data(frame)


def main():
    # -- Create environment --
    env_cfg: DirectRLEnvCfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    if args_cli.mode == "droid":
        env_cfg.control_mode = "joint_vel"
    env = gym.make(args_cli.task, cfg=env_cfg)

    # -- Create pi0.5 wrapper --
    from isaaclab_tasks.direct.fruit_harvest.pi05_wrapper import Pi05Wrapper

    wrapper = Pi05Wrapper(
        host=args_cli.host,
        port=args_cli.port,
        instruction=args_cli.instruction,
        ik_command_scale=env_cfg.ik_command_scale,
        replan_steps=args_cli.replan_steps,
        rotate_image_180=args_cli.rotate_image,
        action_scale_factor=args_cli.action_scale,
        mode=args_cli.mode,
        action_repeat=args_cli.action_repeat,
    )

    # Connect to server (lazy -- actual WebSocket connects on first infer)
    try:
        wrapper.connect()
    except Exception as e:
        print(f"[FATAL] Cannot create pi0.5 client: {e}")
        env.close()
        simulation_app.close()
        return

    # -- Video recording setup --
    video_writer = None
    if args_cli.save_video:
        try:
            import imageio

            video_writer = imageio.get_writer(args_cli.video_path, fps=30, codec="libx264")
            print(f"[Video] Recording to {args_cli.video_path}")
        except ImportError:
            print("[Video] imageio not available, skipping video recording.")
            args_cli.save_video = False

    # -- Create live viewport windows --
    try:
        from omni.kit.viewport.utility import create_viewport_window

        vp_wrist = create_viewport_window("Wrist Camera", width=400, height=400)
        vp_wrist.viewport_api.set_active_camera("/World/envs/env_0/Robot/panda_hand/wrist_cam")
        vp_scene = create_viewport_window("Scene Camera", width=400, height=400)
        vp_scene.viewport_api.set_active_camera("/World/envs/env_0/scene_cam")
        print("[Viewport] Created live camera windows in Isaac Sim UI")
    except Exception as e:
        print(f"[Viewport] Could not create viewport windows: {e}")

    # -- Run episodes --
    import sys

    for episode in range(args_cli.num_episodes):
        print(f"\n{'=' * 60}", flush=True)
        print(f"Episode {episode + 1}/{args_cli.num_episodes}", flush=True)
        print(f"  Task:          {args_cli.task}", flush=True)
        print(f"  Server:        ws://{args_cli.host}:{args_cli.port}", flush=True)
        print(f"  Mode:          {args_cli.mode}", flush=True)
        print(f"  Instruction:   {args_cli.instruction}", flush=True)
        print(f"  Replan steps:  {args_cli.replan_steps}", flush=True)
        print(f"  Action scale:  {args_cli.action_scale}", flush=True)
        print(f"  Action repeat: {args_cli.action_repeat}", flush=True)
        print(f"  Rotate image:  {args_cli.rotate_image}", flush=True)
        print(f"  Max steps:     {args_cli.max_steps}", flush=True)
        print(f"{'=' * 60}\n", flush=True)

        # Reset
        print("[run_pi05] Resetting environment...", flush=True)
        obs, info = env.reset()
        wrapper.reset_action_queue()

        # Diagnostic: check camera data
        for cam_key in ["wrist_rgb", "scene_rgb"]:
            if cam_key in obs:
                t = obs[cam_key]
                tf = t.float()
                print(
                    f"[run_pi05] {cam_key}: shape={t.shape}, dtype={t.dtype}, "
                    f"min={t.min().item()}, max={t.max().item()}, mean={tf.mean().item():.2f}",
                    flush=True,
                )
            else:
                print(f"[run_pi05] {cam_key}: NOT in obs", flush=True)

        total_reward = 0.0
        action_history = []
        done = False

        for step in range(args_cli.max_steps):
            try:
                action = wrapper.predict_action(obs, step=step)
                action = action.to(env.unwrapped.device)
            except Exception as e:
                print(f"[run_pi05] predict_action error at step {step}: {e}", flush=True)
                import traceback

                traceback.print_exc()
                sys.stdout.flush()
                break

            # Track action diversity
            action_np = action.cpu().numpy().flatten()
            action_history.append(action_np.copy())

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)

            step_reward = reward.item() if torch.is_tensor(reward) else float(reward)
            total_reward += step_reward

            # Log every 10 steps
            if step % 10 == 0:
                print(
                    f"Step {step:4d} | reward: {step_reward:+.4f} | "
                    f"total: {total_reward:+.4f} | "
                    f"action: [{', '.join(f'{a:.3f}' for a in action_np)}]",
                    flush=True,
                )

            # Display/save camera frames
            for cam_key in ["wrist_rgb", "scene_rgb"]:
                _display_frame(obs, cam_key, step, video_writer, args_cli.save_video)

            # Check done
            done = terminated or truncated
            if torch.is_tensor(done):
                done = done.any().item()
            if done:
                print(f"\nEpisode ended at step {step + 1}. Total reward: {total_reward:.4f}", flush=True)
                break

        if not done:
            print(f"\nMax steps ({args_cli.max_steps}) reached. Total reward: {total_reward:.4f}", flush=True)

        # -- Action diversity analysis --
        if action_history:
            actions_arr = np.array(action_history)
            action_std = actions_arr.std(axis=0)
            action_range = actions_arr.max(axis=0) - actions_arr.min(axis=0)
            print(f"\n--- Action Diversity (Episode {episode + 1}) ---", flush=True)
            print(f"  STD per dim:   [{', '.join(f'{s:.4f}' for s in action_std)}]", flush=True)
            print(f"  Range per dim: [{', '.join(f'{r:.4f}' for r in action_range)}]", flush=True)
            print(f"  Mean STD:      {action_std.mean():.4f}", flush=True)
            if action_std.mean() < 0.01:
                print("  [WARNING] Very low action diversity -- model may not be responding to observations!", flush=True)
            else:
                print("  [OK] Actions show meaningful variation across steps.", flush=True)

    # -- Cleanup --
    if _HAS_CV2:
        cv2.destroyAllWindows()
    if video_writer is not None:
        video_writer.close()
        print(f"[Video] Saved to {os.path.abspath(args_cli.video_path)}")

    wrapper.close()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
