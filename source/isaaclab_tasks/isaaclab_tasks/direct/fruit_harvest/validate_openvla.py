# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""OpenVLA zero-shot validation experiments for the fruit harvest environment.

Runs a battery of diagnostic tests to determine if OpenVLA can work zero-shot:
  1. Visual grounding: Does the model respond differently to different images?
  2. Token inspection: What bins is the model predicting? Confident or uncertain?
  3. Prompt sweep: Simple vs complex instructions
  4. Unnorm key comparison: bridge_orig vs Franka-specific datasets
  5. Sampling: Greedy vs temperature sampling
  6. Camera source: Wrist vs scene (BridgeData uses third-person)

Usage:
    isaaclab.bat -p source/isaaclab_tasks/isaaclab_tasks/direct/fruit_harvest/validate_openvla.py \
        --task Isaac-Fruit-Harvest-Direct-v0 --num_envs 1 --max_steps 20
"""
from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Validate OpenVLA zero-shot on fruit harvest env.")
parser.add_argument("--task", type=str, default="Isaac-Fruit-Harvest-Direct-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--model", type=str, default="openvla/openvla-7b")
parser.add_argument("--max_steps", type=int, default=20)
parser.add_argument("--device_vla", type=str, default="cuda:0")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from PIL import Image

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectRLEnvCfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def load_vla(model_name: str, device: str):
    """Load OpenVLA model and processor."""
    # Patch compat
    from isaaclab_tasks.direct.fruit_harvest.openvla_wrapper import OpenVLAWrapper
    wrapper = OpenVLAWrapper(model_name=model_name, device=device)
    wrapper._patch_transformers_compat()

    from transformers import AutoConfig, AutoProcessor
    try:
        from transformers import AutoModelForVision2Seq as AutoVLAModel
    except ImportError:
        from transformers import AutoModelForImageTextToText as AutoVLAModel

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if hasattr(config, "auto_map"):
        if "AutoModelForVision2Seq" in config.auto_map and "AutoModelForImageTextToText" not in config.auto_map:
            config.auto_map["AutoModelForImageTextToText"] = config.auto_map["AutoModelForVision2Seq"]

    print(f"[VLA] Loading {model_name}...")
    sys.stdout.flush()
    vla = AutoVLAModel.from_pretrained(
        model_name, config=config, dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True, attn_implementation="eager",
    ).to(device)
    print("[VLA] Model loaded.")
    sys.stdout.flush()
    return vla, processor


def predict_raw(vla, processor, image: Image.Image, prompt: str, device: str, unnorm_key: str):
    """Run inference and return raw action + normalized action + token IDs."""
    inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)

    # Get raw generated tokens with scores
    generated = vla.generate(
        inputs["input_ids"],
        pixel_values=inputs.get("pixel_values"),
        max_new_tokens=7,
        do_sample=False,
        output_scores=True,
        return_dict_in_generate=True,
    )

    action_token_ids = generated.sequences[0, -7:].cpu().numpy()

    # Decode tokens to normalized actions
    vocab_size = vla.vocab_size
    discretized = vocab_size - action_token_ids
    discretized = np.clip(discretized - 1, a_min=0, a_max=vla.bin_centers.shape[0] - 1)
    normalized = vla.bin_centers[discretized]

    # Unnormalize
    raw_action = vla.predict_action(
        **inputs, unnorm_key=unnorm_key, do_sample=False
    )

    # Compute entropy of first action token's distribution
    scores = generated.scores  # tuple of 7 tensors, each (1, vocab_size_full)
    entropies = []
    top_bins = []
    for i, score in enumerate(scores):
        probs = torch.softmax(score[0].float(), dim=-1)
        # Action tokens are at the END of vocab
        # Get entropy over action-relevant range
        action_probs = probs[-256:]
        entropy = -(action_probs * (action_probs + 1e-10).log()).sum().item()
        top5 = torch.topk(action_probs, 5)
        entropies.append(entropy)
        top_bins.append((top5.indices.cpu().numpy(), top5.values.cpu().numpy()))

    return {
        "token_ids": action_token_ids,
        "bin_indices": discretized,
        "normalized": normalized,
        "unnormalized": raw_action,
        "entropies": entropies,
        "top_bins": top_bins,
    }


def predict_sampled(vla, processor, image: Image.Image, prompt: str, device: str,
                    unnorm_key: str, temperature: float, n_samples: int = 5):
    """Run multiple stochastic predictions and return action statistics."""
    inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)
    actions = []
    for _ in range(n_samples):
        action = vla.predict_action(
            **inputs, unnorm_key=unnorm_key, do_sample=True, temperature=temperature
        )
        actions.append(action)
    actions = np.stack(actions)
    return actions.mean(axis=0), actions.std(axis=0), actions


# ==============================================================================
# Experiments
# ==============================================================================

def exp1_visual_grounding(vla, processor, device, sim_image=None):
    """Test: Does the model respond differently to different images?"""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Visual Grounding — Does model respond to image changes?")
    print("=" * 70)

    prompt = "In: What action should the robot take to pick up the red ball?\nOut:"
    unnorm_key = "bridge_orig"

    test_images = {
        "solid_red":   Image.new("RGB", (224, 224), (255, 0, 0)),
        "solid_green": Image.new("RGB", (224, 224), (0, 255, 0)),
        "solid_grey":  Image.new("RGB", (224, 224), (128, 128, 128)),
        "noise":       Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)),
    }
    if sim_image is not None:
        test_images["sim_frame"] = sim_image.resize((224, 224))

    results = {}
    for name, img in test_images.items():
        r = predict_raw(vla, processor, img, prompt, device, unnorm_key)
        results[name] = r["unnormalized"]
        print(f"  {name:12s} | bins={r['bin_indices']} | "
              f"action=[{', '.join(f'{a:.5f}' for a in r['unnormalized'])}]")

    actions_arr = np.stack(list(results.values()))
    per_dim_std = actions_arr.std(axis=0)
    print(f"\n  Per-dim STD across images: [{', '.join(f'{s:.6f}' for s in per_dim_std)}]")
    responsive = per_dim_std.max() > 0.001
    print(f"  VERDICT: {'RESPONSIVE — model differentiates images' if responsive else 'NOT RESPONSIVE — model ignores image content'}")
    return responsive


def exp2_token_inspection(vla, processor, device, sim_image):
    """Test: What tokens/bins is the model predicting? How confident?"""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Token Inspection — Bin predictions and confidence")
    print("=" * 70)

    prompt = "In: What action should the robot take to pick up the red ball?\nOut:"
    img = sim_image.resize((224, 224)) if sim_image else Image.new("RGB", (224, 224), (128, 128, 128))

    r = predict_raw(vla, processor, img, prompt, device, "bridge_orig")

    print(f"  Token IDs:    {r['token_ids']}")
    print(f"  Bin indices:  {r['bin_indices']}")
    print(f"  Normalized:   [{', '.join(f'{v:.4f}' for v in r['normalized'])}]")
    print(f"  Unnormalized: [{', '.join(f'{v:.5f}' for v in r['unnormalized'])}]")

    dim_names = ["dx", "dy", "dz", "drx", "dry", "drz", "grip"]
    center_bin = len(vla.bin_centers) // 2  # ~127
    print(f"\n  Per-dimension analysis (center bin = {center_bin}):")
    for i, name in enumerate(dim_names):
        bins_idx, bins_prob = r['top_bins'][i]
        entropy = r['entropies'][i]
        is_center = abs(r['bin_indices'][i] - center_bin) < 5
        print(f"    {name:4s}: bin={r['bin_indices'][i]:3d} {'(CENTER!)' if is_center else '         '} "
              f"entropy={entropy:.2f}  top5_bins={bins_idx}  top5_probs=[{', '.join(f'{p:.3f}' for p in bins_prob)}]")

    avg_entropy = np.mean(r['entropies'])
    print(f"\n  Average entropy: {avg_entropy:.2f}")
    if avg_entropy > 4.0:
        print("  VERDICT: HIGH ENTROPY — model is very uncertain (flat distribution)")
    elif avg_entropy > 2.0:
        print("  VERDICT: MODERATE ENTROPY — model has some preference but not strong")
    else:
        print("  VERDICT: LOW ENTROPY — model is confident in its predictions")


def exp3_prompt_sweep(vla, processor, device, sim_image):
    """Test: Do simpler prompts produce better/different actions?"""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Prompt Sweep — Simple vs complex instructions")
    print("=" * 70)

    img = sim_image.resize((224, 224)) if sim_image else Image.new("RGB", (224, 224), (128, 128, 128))
    unnorm_key = "bridge_orig"

    prompts = {
        "simple_pick":    "In: What action should the robot take to pick up the red ball?\nOut:",
        "simple_grasp":   "In: What action should the robot take to grasp the red object?\nOut:",
        "move_forward":   "In: What action should the robot take to move forward?\nOut:",
        "move_down":      "In: What action should the robot take to move down?\nOut:",
        "close_gripper":  "In: What action should the robot take to close the gripper?\nOut:",
        "complex_full":   "In: What action should the robot take to pick the red sphere fruit from the plant and carefully place it into the purple basket?\nOut:",
    }

    for name, prompt in prompts.items():
        r = predict_raw(vla, processor, img, prompt, device, unnorm_key)
        print(f"  {name:16s} | bins={r['bin_indices']} | "
              f"action=[{', '.join(f'{a:.5f}' for a in r['unnormalized'])}]")


def exp4_unnorm_key_comparison(vla, processor, device, sim_image):
    """Test: Do different unnorm_keys produce more meaningful actions?"""
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Unnorm Key Comparison — bridge_orig vs Franka datasets")
    print("=" * 70)

    img = sim_image.resize((224, 224)) if sim_image else Image.new("RGB", (224, 224), (128, 128, 128))
    prompt = "In: What action should the robot take to pick up the red ball?\nOut:"

    keys = [
        "bridge_orig",
        "nyu_franka_play_dataset_converted_externally_to_rlds",
        "furniture_bench_dataset_converted_externally_to_rlds",
        "fractal20220817_data",
    ]

    for key in keys:
        try:
            r = predict_raw(vla, processor, img, prompt, device, key)
            action_mag = np.abs(r['unnormalized'][:6]).mean()
            print(f"  {key:55s} | pos_mag={action_mag:.5f} | "
                  f"action=[{', '.join(f'{a:.5f}' for a in r['unnormalized'])}]")
        except Exception as e:
            print(f"  {key:55s} | ERROR: {e}")


def exp5_sampling_sweep(vla, processor, device, sim_image):
    """Test: Does stochastic sampling produce more varied/meaningful actions?"""
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Sampling Sweep — Greedy vs temperature sampling")
    print("=" * 70)

    img = sim_image.resize((224, 224)) if sim_image else Image.new("RGB", (224, 224), (128, 128, 128))
    prompt = "In: What action should the robot take to pick up the red ball?\nOut:"
    unnorm_key = "bridge_orig"

    # Greedy baseline
    r = predict_raw(vla, processor, img, prompt, device, unnorm_key)
    print(f"  {'greedy':12s} | action=[{', '.join(f'{a:.5f}' for a in r['unnormalized'])}]")

    # Temperature sweep
    for temp in [0.1, 0.3, 0.5, 0.7, 1.0]:
        mean, std, all_actions = predict_sampled(
            vla, processor, img, prompt, device, unnorm_key, temperature=temp, n_samples=5
        )
        print(f"  T={temp:.1f} (5 samples) | mean=[{', '.join(f'{a:.5f}' for a in mean)}] | "
              f"std=[{', '.join(f'{s:.5f}' for s in std)}]")


def exp6_camera_source(env, vla, processor, device):
    """Test: Wrist vs scene camera — which gives better actions?"""
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: Camera Source — Wrist (EEF) vs Scene (3rd person)")
    print("=" * 70)

    obs, _ = env.reset()
    # Step a few times to let cameras initialize
    zero_action = torch.zeros(1, 7, device=env.unwrapped.device)
    for _ in range(5):
        obs, _, _, _, _ = env.step(zero_action)

    prompt = "In: What action should the robot take to pick up the red ball?\nOut:"
    unnorm_key = "bridge_orig"

    for cam_key in ["wrist_rgb", "scene_rgb"]:
        if cam_key not in obs:
            print(f"  {cam_key}: NOT AVAILABLE")
            continue
        frame = obs[cam_key]
        if frame.dim() == 4:
            frame = frame[0]
        frame_rgb = frame[:, :, :3]
        if frame_rgb.dtype == torch.uint8:
            img_np = frame_rgb.cpu().numpy()
        else:
            img_np = (frame_rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

        pil_img = Image.fromarray(img_np, mode="RGB").resize((224, 224))

        # Save for reference
        debug_dir = os.path.join(os.path.dirname(__file__), "camera_frames")
        os.makedirs(debug_dir, exist_ok=True)
        pil_img.save(os.path.join(debug_dir, f"validate_{cam_key}.png"))

        r = predict_raw(vla, processor, pil_img, prompt, device, unnorm_key)
        print(f"  {cam_key:10s} | img_mean={img_np.mean():.1f} img_std={img_np.std():.1f} | "
              f"bins={r['bin_indices']} | "
              f"action=[{', '.join(f'{a:.5f}' for a in r['unnormalized'])}]")
        print(f"             | entropies=[{', '.join(f'{e:.2f}' for e in r['entropies'])}]")


# ==============================================================================
# Main
# ==============================================================================

def main():
    env_cfg: DirectRLEnvCfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)

    vla, processor = load_vla(args_cli.model, args_cli.device_vla)

    # Get a sim frame for experiments
    obs, _ = env.reset()
    zero_action = torch.zeros(1, 7, device=env.unwrapped.device)
    for _ in range(5):
        obs, _, _, _, _ = env.step(zero_action)

    sim_image = None
    for cam_key in ["wrist_rgb", "scene_rgb"]:
        if cam_key in obs:
            frame = obs[cam_key]
            if frame.dim() == 4:
                frame = frame[0]
            frame_rgb = frame[:, :, :3]
            if frame_rgb.dtype == torch.uint8:
                img_np = frame_rgb.cpu().numpy()
            else:
                img_np = (frame_rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            sim_image = Image.fromarray(img_np, mode="RGB")
            break

    print("\n" + "#" * 70)
    print("# OpenVLA ZERO-SHOT VALIDATION EXPERIMENTS")
    print("#" * 70)

    # Run all experiments
    exp1_visual_grounding(vla, processor, args_cli.device_vla, sim_image)
    exp2_token_inspection(vla, processor, args_cli.device_vla, sim_image)
    exp3_prompt_sweep(vla, processor, args_cli.device_vla, sim_image)
    exp4_unnorm_key_comparison(vla, processor, args_cli.device_vla, sim_image)
    exp5_sampling_sweep(vla, processor, args_cli.device_vla, sim_image)
    exp6_camera_source(env, vla, processor, args_cli.device_vla)

    print("\n" + "#" * 70)
    print("# VALIDATION COMPLETE")
    print("#" * 70)
    print("\nKey questions answered:")
    print("  1. Does the model respond to different images? (Exp 1)")
    print("  2. Is it predicting center bins confidently or uncertainly? (Exp 2)")
    print("  3. Do simpler prompts help? (Exp 3)")
    print("  4. Do Franka-specific unnorm keys change behavior? (Exp 4)")
    print("  5. Does stochastic sampling produce varied actions? (Exp 5)")
    print("  6. Is scene camera (3rd person, like BridgeData) better? (Exp 6)")
    print("\nSaved diagnostic images to: camera_frames/validate_*.png")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
