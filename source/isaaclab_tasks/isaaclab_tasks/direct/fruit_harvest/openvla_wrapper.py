# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""OpenVLA wrapper for fruit harvest environment — bridges VLA model to env actions."""

from __future__ import annotations

import glob
import os
import re

import numpy as np
import torch
from PIL import Image


class OpenVLAWrapper:
    """Handles OpenVLA model loading, image preprocessing, and action prediction.

    Converts Isaac Lab observations (RGBA float32 tensors) into OpenVLA inputs,
    runs inference, and maps raw 7D model outputs back to env action space.
    """

    def __init__(
        self,
        model_name: str = "openvla/openvla-7b",
        instruction: str = "pick the red sphere fruit from the plant and carefully place it into the purple basket",
        image_source: str = "wrist_rgb",
        device: str = "cuda:0",
        unnorm_key: str = "bridge_orig",
        ik_command_scale: float = 0.05,
    ):
        self.model_name = model_name
        self.instruction = instruction
        self.image_source = image_source
        self.device = device
        self.unnorm_key = unnorm_key
        self.ik_command_scale = ik_command_scale

        self.vla = None
        self.processor = None

    @staticmethod
    def _patch_transformers_compat():
        """Patch transformers 5.x compat: fix cached OpenVLA source files that import from old locations.

        OpenVLA's processing_prismatic.py does:
            from transformers.tokenization_utils import PaddingStrategy, ...
        but transformers 5.x moved these to tokenization_utils_base.

        We fix this by rewriting the cached source file directly, since setattr patching
        doesn't work with importlib's fresh imports used by transformers.dynamic_module_utils.
        """
        import transformers

        tv = getattr(transformers, "__version__", "0")
        if not tv.startswith("5"):
            return

        # Find all cached processing_prismatic.py files for OpenVLA
        hf_cache = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "modules", "transformers_modules")
        pattern = os.path.join(hf_cache, "openvla", "**", "processing_prismatic.py")
        for fpath in glob.glob(pattern, recursive=True):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    src = f.read()
                old_import = "from transformers.tokenization_utils import"
                new_import = "from transformers.tokenization_utils_base import"
                if old_import in src:
                    src = src.replace(old_import, new_import)
                    with open(fpath, "w", encoding="utf-8") as f:
                        f.write(src)
                    print(f"[OpenVLA] Patched {fpath} for transformers 5.x compat")
            except OSError:
                pass

        # Patch modeling_prismatic.py for transformers 5.x + timm 1.x compat
        pattern_model = os.path.join(hf_cache, "openvla", "**", "modeling_prismatic.py")
        for fpath in glob.glob(pattern_model, recursive=True):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    src = f.read()
                changed = False
                # Relax timm version check (timm 1.x is compatible)
                old_check = 'if timm.__version__ not in {"0.9.10", "0.9.11", "0.9.12", "0.9.16"}'
                if old_check in src:
                    src = re.sub(
                        r'if timm\.__version__ not in \{.*?\}:\s*raise NotImplementedError\([^)]*\)',
                        'pass  # timm version check relaxed for 1.x compat',
                        src,
                        flags=re.DOTALL,
                    )
                    changed = True
                # Fix tie_weights signature for transformers 5.x (adds recompute_mapping kwarg)
                if "def tie_weights(self) -> None:" in src:
                    src = src.replace(
                        "def tie_weights(self) -> None:",
                        "def tie_weights(self, **kwargs) -> None:",
                    )
                    changed = True
                if changed:
                    with open(fpath, "w", encoding="utf-8") as f:
                        f.write(src)
                    print(f"[OpenVLA] Patched {fpath} for transformers 5.x / timm 1.x compat")
            except OSError:
                pass

        # Also patch via setattr as a belt-and-suspenders approach
        try:
            import transformers.tokenization_utils as _tok_utils
            from transformers.tokenization_utils_base import (
                PaddingStrategy,
                PreTokenizedInput,
                TextInput,
                TruncationStrategy,
            )
            for name, obj in [
                ("PaddingStrategy", PaddingStrategy),
                ("PreTokenizedInput", PreTokenizedInput),
                ("TextInput", TextInput),
                ("TruncationStrategy", TruncationStrategy),
            ]:
                if not hasattr(_tok_utils, name):
                    setattr(_tok_utils, name, obj)
        except Exception:
            pass

    def load_model(self):
        """Load OpenVLA model and processor. Requires ~16GB VRAM for BF16."""
        # Patch transformers compat BEFORE any model code is loaded
        self._patch_transformers_compat()

        from transformers import AutoConfig, AutoProcessor

        # transformers 5.x renamed AutoModelForVision2Seq -> AutoModelForImageTextToText
        try:
            from transformers import AutoModelForVision2Seq as AutoVLAModel
        except ImportError:
            from transformers import AutoModelForImageTextToText as AutoVLAModel

        print(f"[OpenVLA] Loading model: {self.model_name} on {self.device} ...")
        self.processor = AutoProcessor.from_pretrained(
            self.model_name, trust_remote_code=True
        )

        # Load config first so we can patch auto_map for transformers 5.x compat.
        # OpenVLA's config registers with "AutoModelForVision2Seq" (transformers 4.x)
        # but transformers 5.x renamed it to "AutoModelForImageTextToText".
        config = AutoConfig.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        if hasattr(config, "auto_map"):
            old_key = "AutoModelForVision2Seq"
            new_key = "AutoModelForImageTextToText"
            if old_key in config.auto_map and new_key not in config.auto_map:
                config.auto_map[new_key] = config.auto_map[old_key]

        # Force eager attention — OpenVLA's model class (transformers 4.x era) lacks
        # the _supports_sdpa / _supports_flash_attn_2 attributes that transformers 5.x checks.
        print("[OpenVLA] Downloading/loading weights into CPU RAM...")
        import sys
        sys.stdout.flush()
        self.vla = AutoVLAModel.from_pretrained(
            self.model_name,
            config=config,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation="eager",
        )
        print(f"[OpenVLA] Weights loaded on CPU. Moving to {self.device}...")
        sys.stdout.flush()
        self.vla = self.vla.to(self.device)
        print("[OpenVLA] Model loaded on GPU.")
        sys.stdout.flush()

    def _obs_to_pil(self, obs: dict) -> Image.Image:
        """Extract camera image from obs dict and convert to 224x224 RGB PIL Image.

        Isaac Lab cameras may output either:
          - RGBA float32 tensors of shape (N, H, W, 4) in [0, 1]
          - RGBA uint8 tensors of shape (N, H, W, 4) in [0, 255]
        We take env 0, strip alpha, convert to uint8 if needed, and resize.
        """
        key = self.image_source
        if key not in obs:
            raise KeyError(
                f"Image source '{key}' not found in obs. Available: {list(obs.keys())}"
            )
        img_tensor = obs[key]
        # Take first env, strip alpha channel
        if img_tensor.dim() == 4:
            img_tensor = img_tensor[0]  # (H, W, 4)
        img_rgb = img_tensor[:, :, :3]  # (H, W, 3)
        # Handle both uint8 [0,255] and float32 [0,1] camera outputs
        if img_rgb.dtype == torch.uint8:
            img_np = img_rgb.cpu().numpy()
        else:
            img_np = (img_rgb.clamp(0.0, 1.0).cpu().numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np, mode="RGB")
        return pil_img.resize((224, 224), Image.LANCZOS)

    def _build_prompt(self) -> str:
        return f"In: What action should the robot take to {self.instruction}?\nOut:"

    def predict_action(self, obs: dict, step: int = -1) -> torch.Tensor:
        """Run VLA inference on current observation.

        Args:
            obs: Observation dict with camera images.
            step: Current step number (for debug saving). -1 = don't save.

        Returns:
            Action tensor of shape (1, 7) suitable for env.step().
        """
        pil_image = self._obs_to_pil(obs)
        prompt = self._build_prompt()

        # Save the exact 224x224 image fed to the VLA for debugging
        if step >= 0 and step % 10 == 0:
            debug_dir = os.path.join(os.path.dirname(__file__), "camera_frames")
            os.makedirs(debug_dir, exist_ok=True)
            pil_image.save(os.path.join(debug_dir, f"vla_input_step{step:04d}.png"))

        inputs = self.processor(prompt, pil_image).to(self.device, dtype=torch.bfloat16)
        raw_action = self.vla.predict_action(
            **inputs, unnorm_key=self.unnorm_key, do_sample=False
        )  # numpy array, shape (7,)

        # Log raw model output and image stats for debugging
        if step >= 0 and step % 10 == 0:
            img_np = np.array(pil_image)
            print(f"  [VLA] raw_action: [{', '.join(f'{a:.5f}' for a in raw_action)}]"
                  f"  img_mean={img_np.mean():.1f} img_std={img_np.std():.1f}", flush=True)

        return self._scale_action_to_env(raw_action)

    def _scale_action_to_env(self, raw_action: np.ndarray) -> torch.Tensor:
        """Map OpenVLA raw 7D output to env action space [-1, 1].

        OpenVLA outputs: [dx, dy, dz, drx, dry, drz, gripper]
        - Position/orientation deltas: scale by 1/ik_command_scale to undo env's scaling
        - Gripper: [0, 1] -> [-1, 1]  (>0 = open, <0 = close in env)
        """
        action = np.array(raw_action, dtype=np.float32)

        # Scale pose deltas: raw values are in meters/radians,
        # env multiplies by ik_command_scale, so we divide to get [-1, 1] range
        action[:6] = action[:6] / self.ik_command_scale
        action[:6] = np.clip(action[:6], -1.0, 1.0)

        # Gripper: OpenVLA outputs ~0 (close) to ~1 (open)
        # Env expects >0 = open, <0 = close
        action[6] = action[6] * 2.0 - 1.0
        action[6] = np.clip(action[6], -1.0, 1.0)

        return torch.tensor(action, dtype=torch.float32).unsqueeze(0)  # (1, 7)
