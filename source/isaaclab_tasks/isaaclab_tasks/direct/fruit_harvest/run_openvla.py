# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Zero-shot OpenVLA inference on the fruit harvest environment.

Usage:
    isaaclab.bat -p source/isaaclab_tasks/isaaclab_tasks/direct/fruit_harvest/run_openvla.py \
        --task Isaac-Fruit-Harvest-Direct-v0 \
        --num_envs 1 \
        --model openvla/openvla-7b \
        --instruction "pick the red sphere fruit from the plant and carefully place it into the purple basket" \
        --image_source wrist_rgb \
        --max_steps 500
"""

from __future__ import annotations

import argparse
import os

# -- Step 1: Parse args and launch SimulationApp BEFORE any isaaclab imports --
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run OpenVLA zero-shot inference on fruit harvest env.")
parser.add_argument("--task", type=str, default="Isaac-Fruit-Harvest-Direct-v0", help="Gym env ID.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--model", type=str, default="openvla/openvla-7b", help="OpenVLA model name or path.")
parser.add_argument("--instruction", type=str, default="pick the red sphere fruit from the plant and carefully place it into the purple basket", help="Task instruction for VLA prompt.")
parser.add_argument("--image_source", type=str, default="wrist_rgb", choices=["wrist_rgb", "scene_rgb"], help="Which camera to use.")
parser.add_argument("--max_steps", type=int, default=500, help="Max steps per episode.")
parser.add_argument("--device_vla", type=str, default="cuda:0", help="Device for VLA model.")
parser.add_argument("--unnorm_key", type=str, default="bridge_orig", help="Action un-normalization key.")
parser.add_argument("--save_video", action="store_true", help="Save camera feed as MP4.")
parser.add_argument("--video_path", type=str, default="openvla_rollout.mp4", help="Output video path.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Force-enable cameras — required for wrist/scene camera sensors
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
# NOTE: Isaac Sim bundles a headless cv2 (4.11) in omni.pip.compute that lacks imshow.
# We force-load the full cv2 from site-packages if available.
_HAS_CV2 = False
try:
    import importlib
    import cv2
    # Check if imshow actually works (headless builds raise error)
    _test_img = np.zeros((10, 10, 3), dtype=np.uint8)
    cv2.imshow("_test", _test_img)
    cv2.destroyAllWindows()
    _HAS_CV2 = True
except Exception:
    # Try loading from site-packages directly, bypassing bundled headless version
    try:
        import importlib.util
        _sp = os.path.join(os.path.dirname(os.path.abspath(importlib.util.find_spec("cv2").origin)), "..")
        # If that doesn't work, just skip
        _HAS_CV2 = False
    except Exception:
        pass
    if not _HAS_CV2:
        print("[Warning] cv2.imshow not available — saving camera frames to disk instead.")


def main():
    # -- Create environment --
    env_cfg: DirectRLEnvCfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)

    # -- Load OpenVLA --
    from isaaclab_tasks.direct.fruit_harvest.openvla_wrapper import OpenVLAWrapper

    vla_wrapper = OpenVLAWrapper(
        model_name=args_cli.model,
        instruction=args_cli.instruction,
        image_source=args_cli.image_source,
        device=args_cli.device_vla,
        unnorm_key=args_cli.unnorm_key,
        ik_command_scale=env_cfg.ik_command_scale,
    )
    vla_wrapper.load_model()

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

    # -- Create live viewport windows for each camera --
    try:
        from omni.kit.viewport.utility import create_viewport_window
        vp_wrist = create_viewport_window("Wrist Camera", width=400, height=400)
        vp_wrist.viewport_api.set_active_camera("/World/envs/env_0/Robot/panda_hand/wrist_cam")
        vp_scene = create_viewport_window("Scene Camera", width=400, height=400)
        vp_scene.viewport_api.set_active_camera("/World/envs/env_0/scene_cam")
        print("[Viewport] Created live camera windows in Isaac Sim UI")
    except Exception as e:
        print(f"[Viewport] Could not create viewport windows: {e}")

    # -- Rollout loop --
    import sys

    print("[run_openvla] Resetting environment...", flush=True)
    obs, info = env.reset()
    print(f"[run_openvla] Obs keys: {list(obs.keys()) if isinstance(obs, dict) else type(obs)}", flush=True)
    # Diagnostic: check camera data
    for cam_key in ["wrist_rgb", "scene_rgb"]:
        if cam_key in obs:
            t = obs[cam_key]
            tf = t.float()  # convert to float for mean() — uint8 doesn't support it
            print(f"[run_openvla] {cam_key}: shape={t.shape}, dtype={t.dtype}, "
                  f"min={t.min().item()}, max={t.max().item()}, mean={tf.mean().item():.2f}", flush=True)
        else:
            print(f"[run_openvla] {cam_key}: NOT in obs", flush=True)
    total_reward = 0.0

    print(f"\n{'='*60}", flush=True)
    print(f"Running OpenVLA zero-shot inference", flush=True)
    print(f"  Task:        {args_cli.task}", flush=True)
    print(f"  Model:       {args_cli.model}", flush=True)
    print(f"  Instruction: {args_cli.instruction}", flush=True)
    print(f"  Camera:      {args_cli.image_source}", flush=True)
    print(f"  Max steps:   {args_cli.max_steps}", flush=True)
    print(f"{'='*60}\n", flush=True)

    done = False
    for step in range(args_cli.max_steps):
        try:
            # VLA inference
            action = vla_wrapper.predict_action(obs, step=step)
            action = action.to(env.unwrapped.device)
        except Exception as e:
            print(f"[run_openvla] predict_action error at step {step}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            break

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)

        step_reward = reward.item() if torch.is_tensor(reward) else float(reward)
        total_reward += step_reward

        # Log every 10 steps
        if step % 10 == 0:
            action_np = action.cpu().numpy().flatten()
            print(
                f"Step {step:4d} | reward: {step_reward:+.4f} | "
                f"total: {total_reward:+.4f} | "
                f"action: [{', '.join(f'{a:.3f}' for a in action_np)}]",
                flush=True,
            )

        # -- Live camera view + video recording --
        for cam_key in ["wrist_rgb", "scene_rgb"]:
            if cam_key not in obs:
                continue
            frame = obs[cam_key]
            if torch.is_tensor(frame):
                if frame.dim() == 4:
                    frame = frame[0]
                frame_rgb = frame[:, :, :3]
                # Handle both uint8 [0,255] and float32 [0,1] camera outputs
                if frame_rgb.dtype == torch.uint8:
                    frame = frame_rgb.cpu().numpy()
                else:
                    frame = (frame_rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

            # Live OpenCV window (if GUI cv2 is available)
            if _HAS_CV2:
                import cv2
                display = cv2.resize(frame, (384, 384), interpolation=cv2.INTER_NEAREST)
                display = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)
                cv2.imshow(f"OpenVLA - {cam_key}", display)
                cv2.waitKey(1)
            else:
                # Save latest frame to disk for external viewing (every 50 steps to avoid I/O overhead)
                if step % 50 == 0:
                    try:
                        from PIL import Image as _PILImage
                        _cam_dir = os.path.join(os.path.dirname(__file__), "camera_frames")
                        os.makedirs(_cam_dir, exist_ok=True)
                        _frame_save = np.ascontiguousarray(frame[:, :, :3], dtype=np.uint8)
                        _PILImage.fromarray(_frame_save).save(os.path.join(_cam_dir, f"{cam_key}_latest.png"))
                    except Exception as e:
                        if step == 0:
                            print(f"[Warning] Could not save {cam_key} frame: {e}")

            # Record video (wrist camera only)
            if cam_key == args_cli.image_source and args_cli.save_video and video_writer is not None:
                video_writer.append_data(frame)

        # Check done
        done = terminated or truncated
        if torch.is_tensor(done):
            done = done.any().item()
        if done:
            print(f"\nEpisode ended at step {step + 1}. Total reward: {total_reward:.4f}", flush=True)
            break

    if not done:
        print(f"\nMax steps ({args_cli.max_steps}) reached. Total reward: {total_reward:.4f}", flush=True)

    if _HAS_CV2:
        cv2.destroyAllWindows()

    if video_writer is not None:
        video_writer.close()
        print(f"[Video] Saved to {os.path.abspath(args_cli.video_path)}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
