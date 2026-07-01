# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""pi0.5 wrapper for fruit harvest environment.

Bridges openpi server (running in WSL2 with JAX) to Isaac Sim env actions
via WebSocket. Supports both the official openpi-client package and a
bundled minimal WebSocket client.

Architecture:
    Isaac Sim (Windows) <--WebSocket--> openpi server (WSL2/Linux)

The wrapper handles:
    - Observation mapping: Isaac Sim tensors -> pi0.5 LIBERO format
    - Action chunk caching: execute N actions before re-querying server
    - Action scaling: pi0.5 raw deltas -> env [-1, 1] action space
"""

from __future__ import annotations

import functools
import logging
import os
import sys
import threading
import time
from typing import Optional

import msgpack
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


# =============================================================================
# Bundled minimal WebSocket client for openpi protocol
# =============================================================================
# Adapted from openpi's msgpack_numpy.py and websocket_client_policy.py.
# Uses dict-based numpy serialization with b"__ndarray__" key and
# synchronous websockets (websockets.sync.client).
#
# Protocol:
#   1. Client connects via WebSocket
#   2. Server sends metadata dict as first message
#   3. Client sends msgpack-encoded observation dicts
#   4. Server responds with msgpack-encoded action dicts
#   5. If server sends a string (not bytes), it's an error traceback


def _pack_numpy(obj):
    """Serialize numpy arrays for msgpack (openpi wire format)."""
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported numpy dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def _unpack_numpy(obj):
    """Deserialize numpy arrays from msgpack (openpi wire format)."""
    if b"__ndarray__" in obj:
        return np.ndarray(
            buffer=obj[b"data"],
            dtype=np.dtype(obj[b"dtype"]),
            shape=obj[b"shape"],
        )
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


_packb = functools.partial(msgpack.packb, default=_pack_numpy)
_unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_numpy)


class _BundledWebsocketClient:
    """Minimal WebSocket client compatible with openpi serve_policy.py.

    Uses synchronous websockets API (websockets.sync.client) matching
    the official openpi-client implementation.
    """

    def __init__(self, host: str = "localhost", port: int = 8000):
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack.Packer(default=_pack_numpy)
        self._ws = None
        self._server_metadata = None
        self._lock = threading.Lock()

    def _connect(self):
        """Connect to server and receive initial metadata."""
        import websockets.sync.client

        logger.info(f"Connecting to pi0.5 server at {self._uri}...")
        while True:
            try:
                self._ws = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                )
                # Server sends metadata as the first message
                metadata_raw = self._ws.recv()
                self._server_metadata = _unpackb(metadata_raw)
                logger.info(f"Connected. Server metadata keys: {list(self._server_metadata.keys())}")
                return
            except ConnectionRefusedError:
                logger.info("Server not ready, retrying in 5s...")
                time.sleep(5)
            except Exception as e:
                raise ConnectionError(
                    f"Cannot connect to pi0.5 server at {self._uri}. "
                    f"Is the server running? Start it with:\n"
                    f"  cd ~/openpi && uv run scripts/serve_policy.py --env LIBERO --port 8000\n"
                    f"Error: {e}"
                ) from e

    def get_server_metadata(self) -> dict:
        return self._server_metadata or {}

    def infer(self, observation: dict) -> dict:
        """Send observation dict, receive action dict from server."""
        with self._lock:
            if self._ws is None:
                self._connect()

            data = self._packer.pack(observation)
            self._ws.send(data)
            response = self._ws.recv()

            # If server sends a string, it's an error traceback
            if isinstance(response, str):
                raise RuntimeError(f"Error in pi0.5 inference server:\n{response}")

            return _unpackb(response)

    def close(self):
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None


def _get_client(host: str, port: int):
    """Try official openpi-client first, fall back to bundled implementation."""
    try:
        from openpi_client.websocket_client_policy import WebsocketClientPolicy

        print("[pi0.5] Using official openpi-client package", flush=True)
        return WebsocketClientPolicy(host=host, port=port)
    except ImportError:
        print(
            "[pi0.5] openpi-client not installed, using bundled WebSocket client.\n"
            "  (Optional: pip install openpi-client  for the official client)",
            flush=True,
        )
        return _BundledWebsocketClient(host=host, port=port)


# =============================================================================
# Utility functions
# =============================================================================


def _quat_wxyz_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to axis-angle representation (3D).

    Args:
        quat: Quaternion array [w, x, y, z].

    Returns:
        Axis-angle vector (3D) where direction = rotation axis, magnitude = angle.
    """
    w, x, y, z = quat.astype(np.float64)
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-10:
        return np.zeros(3, dtype=np.float32)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    # Use shorter rotation path
    if w < 0:
        w, x, y, z = -w, -x, -y, -z

    angle = 2.0 * np.arccos(np.clip(w, -1.0, 1.0))
    sin_half = np.sqrt(1.0 - w * w)
    if sin_half < 1e-10:
        return np.zeros(3, dtype=np.float32)
    axis = np.array([x, y, z]) / sin_half
    return (axis * angle).astype(np.float32)


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions in (w, x, y, z) format."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float64)


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate (inverse for unit quaternions) in (w, x, y, z) format."""
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


# Robot base pose in world frame (from fruit_harvest_env.py FruitHarvestEnvCfg):
#   pos = (0.6, 0.0, 0.7)  — on table surface
#   rot = (0, 0, 0, 1)     — 180° around Z (w,x,y,z format) to face plant
_ROBOT_BASE_POS = np.array([0.6, 0.0, 0.7], dtype=np.float64)
_ROBOT_BASE_QUAT = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)  # (w,x,y,z)
_ROBOT_BASE_QUAT_INV = _quat_conjugate(_ROBOT_BASE_QUAT)

# Rotation matrix for 180° around Z (used for position transform)
# R_180z = [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
# R_180z_inv = R_180z (self-inverse)

# LIBERO normalization stats (from checkpoint norm_stats.json) for reference
_LIBERO_STATE_MEAN = np.array([-0.0436, 0.0353, 0.7637, 2.9674, -0.2108, -0.1298, 0.0278, -0.0280])
_LIBERO_STATE_STD = np.array([0.1034, 0.1519, 0.3815, 0.3545, 0.9292, 0.3307, 0.0141, 0.0140])
_LIBERO_STATE_Q01 = np.array([-0.3524, -0.2682, 0.0408, 1.5318, -2.7152, -1.0765, 0.0017, -0.0400])
_LIBERO_STATE_Q99 = np.array([0.1389, 0.3252, 1.2569, 3.2628, 2.4437, 0.5638, 0.0403, -0.0017])


def _rgba_tensor_to_uint8(tensor: torch.Tensor) -> np.ndarray:
    """Convert RGBA float32/uint8 tensor (N,H,W,4) -> RGB uint8 numpy (H,W,3).

    Handles both Isaac Sim camera output formats:
      - float32 in [0, 1] with alpha channel
      - uint8 in [0, 255] with alpha channel
    """
    if tensor.dim() == 4:
        tensor = tensor[0]
    rgb = tensor[:, :, :3]
    if rgb.dtype == torch.uint8:
        return rgb.cpu().numpy()
    return (rgb.clamp(0.0, 1.0).cpu().numpy() * 255).astype(np.uint8)


def _resize_image(img: np.ndarray, target_size: int = 224) -> np.ndarray:
    """Resize RGB uint8 image to target_size x target_size."""
    pil = Image.fromarray(img)
    pil = pil.resize((target_size, target_size), Image.LANCZOS)
    return np.array(pil)


# =============================================================================
# Pi05Wrapper
# =============================================================================


class Pi05Wrapper:
    """Wraps pi0.5 server inference for the fruit harvest environment.

    Same interface as OpenVLAWrapper (predict_action takes obs dict, returns
    action tensor), but communicates with a remote pi0.5 server over WebSocket
    instead of running a local model.

    Supports two modes:

    LIBERO mode (mode="libero", default):
        Observation mapping (Isaac Sim -> pi0.5 LIBERO):
            scene_rgb (256x256 RGBA)     -> observation/image (224x224 RGB uint8)
            wrist_rgb (128x128 RGBA)     -> observation/wrist_image (224x224 RGB uint8)
            policy[9:12] (EE position)   -> observation/state[0:3]
            policy[12:16] (EE quat wxyz) -> observation/state[3:6] (axis-angle)
            policy[23:25] (finger state) -> observation/state[6:8]
            instruction string           -> prompt
        Action mapping (pi0.5 -> env):
            action[0:3]: position delta  -> scaled by 1/ik_command_scale, clipped [-1,1]
            action[3:6]: rotation delta  -> scaled by 1/ik_command_scale, clipped [-1,1]
            action[6]:   gripper         -> binarized: >0.5 = open (1.0), else close (-1.0)

    DROID mode (mode="droid"):
        Observation mapping (Isaac Sim -> pi0.5 DROID):
            scene_rgb                    -> observation/exterior_image_1_left (224x224 RGB uint8)
            wrist_rgb                    -> observation/wrist_image_left (224x224 RGB uint8)
            raw_joint_pos (7D)           -> observation/joint_position
            raw_gripper_pos[0] (1D)      -> observation/gripper_position
            instruction string           -> prompt
        Action mapping (pi0.5 -> env):
            action[0:7]: joint velocities -> clipped [-1,1], scaled by action_scale_factor
            action[7]:   gripper          -> binarized: >0.5 = open (1.0), else close (-1.0)
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        instruction: str = "pick the red sphere fruit from the plant and carefully place it into the purple basket",
        ik_command_scale: float = 0.05,
        replan_steps: int = 5,
        rotate_image_180: bool = False,
        action_scale_factor: float = 1.0,
        mode: str = "libero",
        action_repeat: int = 1,
    ):
        """Initialize wrapper.

        Args:
            host: pi0.5 server hostname (WSL2 is accessible at localhost).
            port: pi0.5 server port.
            instruction: Task instruction prompt for pi0.5.
            ik_command_scale: Env's IK command scale (used to map raw deltas to [-1,1]).
            replan_steps: Number of actions to execute from each chunk before re-querying.
            rotate_image_180: Whether to rotate images 180 deg (LIBERO training convention).
            action_scale_factor: Additional multiplier on position/rotation deltas.
            mode: "libero" for LIBERO pi0.5 (7D EE delta) or "droid" for DROID pi0.5 (8D joint vel).
            action_repeat: How many env steps to repeat each model action (e.g. 4 for DROID 15Hz -> 60Hz env).
        """
        self.host = host
        self.port = port
        self.instruction = instruction
        self.ik_command_scale = ik_command_scale
        self.replan_steps = replan_steps
        self.rotate_image_180 = rotate_image_180
        self.action_scale_factor = action_scale_factor
        self.mode = mode
        self.action_repeat = action_repeat

        self.client: Optional[_BundledWebsocketClient] = None
        self.action_queue: list[np.ndarray] = []
        self._step_count = 0
        self._total_inferences = 0

    def connect(self):
        """Connect to pi0.5 server via WebSocket."""
        print(f"[pi0.5] Connecting to ws://{self.host}:{self.port} ...", flush=True)
        self.client = _get_client(self.host, self.port)
        print(f"[pi0.5] Client ready. Connection established on first inference.", flush=True)

    def predict_action(self, obs: dict, step: int = -1) -> torch.Tensor:
        """Get next action from pi0.5, using action chunk caching.

        On first call (or when chunk is exhausted), queries the server for a
        new action chunk and caches `replan_steps` actions. Returns one action
        per call.

        Args:
            obs: Observation dict from env (must contain scene_rgb, wrist_rgb, policy).
            step: Current step number for logging. -1 = no logging.

        Returns:
            Action tensor of shape (1, 7) in [-1, 1] suitable for env.step().
        """
        if self.client is None:
            raise RuntimeError("Call connect() before predict_action()")

        action_dim = 8 if self.mode == "droid" else 7

        if not self.action_queue:
            # Build observation for pi0.5
            if self.mode == "droid":
                pi05_obs = self._build_obs_droid(obs, step=step)
            else:
                pi05_obs = self._build_obs(obs, step=step)

            if step >= 0 and step % 10 == 0:
                print(f"  [pi0.5] Querying server (step {step}, inference #{self._total_inferences}, mode={self.mode})...", flush=True)
                sys.stdout.flush()

            try:
                result = self.client.infer(pi05_obs)
            except Exception as e:
                print(f"  [pi0.5] Server inference failed: {e}", flush=True)
                # Return zero action on failure
                return torch.zeros(1, action_dim, dtype=torch.float32)

            self._total_inferences += 1

            # Extract action chunk from response
            chunk = self._extract_actions(result)

            # Queue replan_steps actions
            n_to_queue = min(self.replan_steps, len(chunk))
            queued = [chunk[i].copy() for i in range(n_to_queue)]

            # Apply action repeat (e.g. DROID 15Hz -> 60Hz env = repeat 4x)
            if self.action_repeat > 1:
                expanded = []
                for action in queued:
                    expanded.extend([action.copy() for _ in range(self.action_repeat)])
                self.action_queue = expanded
            else:
                self.action_queue = queued

            if step >= 0 and step % 10 == 0:
                print(
                    f"  [pi0.5] Got chunk: shape={chunk.shape}, queued {len(self.action_queue)} actions"
                    f" (repeat={self.action_repeat})",
                    flush=True,
                )

        # Pop next action from queue
        raw_action = self.action_queue.pop(0)
        self._step_count += 1

        # Log raw action
        if step >= 0 and step % 10 == 0:
            print(
                f"  [pi0.5] raw_action: [{', '.join(f'{a:.5f}' for a in raw_action)}]",
                flush=True,
            )

        if self.mode == "droid":
            return self._scale_action_to_env_droid(raw_action)
        return self._scale_action_to_env(raw_action)

    def _extract_actions(self, result: dict) -> np.ndarray:
        """Extract action array from server response dict.

        The server may return actions under different keys depending on config.
        """
        for key in ["actions", "action"]:
            if key in result:
                arr = np.asarray(result[key], dtype=np.float32)
                if arr.ndim == 1:
                    arr = arr[np.newaxis, :]
                return arr

        raise KeyError(
            f"No 'actions' or 'action' in server response. "
            f"Available keys: {list(result.keys())}"
        )

    def _build_obs(self, obs: dict, step: int = -1) -> dict:
        """Build observation dict in pi0.5 LIBERO format.

        Maps Isaac Sim observation tensors to the format expected by the
        pi0.5 LIBERO checkpoint. Key transformations:
        1. EE pose: world frame -> robot base frame (LIBERO convention)
        2. Gripper: normalized [0,1] -> raw joint positions [0, 0.04] (LIBERO scale)
        3. Images: RGBA float/uint8 -> 224x224 RGB uint8
        """
        # -- Images --
        scene_img = _rgba_tensor_to_uint8(obs["scene_rgb"])
        scene_img = _resize_image(scene_img, 224)

        wrist_img = _rgba_tensor_to_uint8(obs["wrist_rgb"])
        wrist_img = _resize_image(wrist_img, 224)

        if self.rotate_image_180:
            scene_img = np.rot90(scene_img, k=2).copy()
            wrist_img = np.rot90(wrist_img, k=2).copy()

        # Save debug images periodically
        if step >= 0 and step % 50 == 0:
            self._save_debug_images(scene_img, wrist_img, step)

        # -- State vector --
        # Policy tensor layout (Franka, 9 joints):
        #   [0:9]   scaled joint positions
        #   [9:12]  EE position (3D, world frame)
        #   [12:16] EE quaternion (4D, w,x,y,z)
        #   [16:19] target fruit position
        #   [19:22] fruit-to-EE vector
        #   [22:23] distance to target
        #   [23:25] finger state (2D, normalized 0-1)
        #   [25:26] grasped state
        #   [26:27] harvested state
        policy = obs["policy"]
        if torch.is_tensor(policy):
            policy = policy.cpu().numpy()
        if policy.ndim == 2:
            policy = policy[0]  # Take first env

        ee_pos_w = policy[9:12].astype(np.float64)     # EE position (world frame)
        ee_quat_w = policy[12:16].astype(np.float64)   # EE quaternion (w,x,y,z, world frame)
        finger_norm = policy[23:25].astype(np.float32)  # finger state normalized [0,1]

        # --- Transform EE pose from world frame to robot base frame ---
        # Robot base: pos=(0.6, 0, 0.7), rot=180° around Z
        # 1. Translate: subtract base position
        ee_pos_delta = ee_pos_w - _ROBOT_BASE_POS
        # 2. Rotate by inverse of base rotation (180° around Z is self-inverse)
        #    R_180z: x -> -x, y -> -y, z -> z
        ee_pos_local = np.array([
            -ee_pos_delta[0],
            -ee_pos_delta[1],
            ee_pos_delta[2],
        ], dtype=np.float64)

        # 3. Transform quaternion to base frame: q_local = q_base_inv * q_world
        ee_quat_local = _quat_multiply(_ROBOT_BASE_QUAT_INV, ee_quat_w)
        # Normalize
        ee_quat_local = ee_quat_local / np.linalg.norm(ee_quat_local)

        # Convert local-frame quaternion to axis-angle
        ee_axis_angle = _quat_wxyz_to_axis_angle(ee_quat_local)

        # --- Map gripper to LIBERO scale ---
        # Our finger_state is normalized [0, 1] (0=closed, 1=open)
        # LIBERO uses raw joint positions: finger[0] in [0, 0.04], finger[1] in [-0.04, 0]
        # gripper_open_value = 0.04m for Franka
        gripper_raw = np.array([
            finger_norm[0] * 0.04,    # [0, 1] -> [0, 0.04]
            -finger_norm[1] * 0.04,   # [0, 1] -> [0, -0.04] (LIBERO convention: second finger is negative)
        ], dtype=np.float32)

        # LIBERO state: [ee_pos(3), ee_rot_axis_angle(3), gripper(2)] = 8D
        ee_pos_f32 = ee_pos_local.astype(np.float32)
        state = np.concatenate([ee_pos_f32, ee_axis_angle, gripper_raw]).astype(np.float32)

        # Log state and normalization quality
        if step >= 0 and step % 10 == 0:
            normalized = (state - _LIBERO_STATE_MEAN) / _LIBERO_STATE_STD
            max_sigma = np.max(np.abs(normalized))
            print(
                f"  [pi0.5] state (base frame): "
                f"ee_pos=[{ee_pos_f32[0]:.3f},{ee_pos_f32[1]:.3f},{ee_pos_f32[2]:.3f}] "
                f"ee_aa=[{ee_axis_angle[0]:.3f},{ee_axis_angle[1]:.3f},{ee_axis_angle[2]:.3f}] "
                f"grip=[{gripper_raw[0]:.4f},{gripper_raw[1]:.4f}]",
                flush=True,
            )
            print(
                f"  [pi0.5] normalized σ: [{', '.join(f'{n:.1f}' for n in normalized)}] "
                f"max={max_sigma:.1f}σ",
                flush=True,
            )

        return {
            "observation/image": scene_img,
            "observation/wrist_image": wrist_img,
            "observation/state": state,
            "prompt": self.instruction,
        }

    def _scale_action_to_env(self, raw_action: np.ndarray) -> torch.Tensor:
        """Map pi0.5 raw 7D output to env action space [-1, 1].

        pi0.5 outputs:
            [dx, dy, dz, drx, dry, drz, gripper]
            - Position/rotation deltas in meters/radians
            - Gripper: continuous (0=close, 1=open) or similar

        Env expects:
            (1, 7) tensor in [-1, 1]
            - [:, 0:6] scaled by ik_command_scale (0.05)
            - [:, 6] gripper: >0 = open, <0 = close
        """
        action = np.array(raw_action, dtype=np.float32)

        # Ensure 7D
        if len(action) > 7:
            action = action[:7]
        elif len(action) < 7:
            action = np.pad(action, (0, 7 - len(action)), constant_values=0.0)

        # Apply optional scale factor
        action[:6] *= self.action_scale_factor

        # Scale pose deltas: raw meters/radians -> [-1, 1] via ik_command_scale
        action[:6] = action[:6] / self.ik_command_scale
        action[:6] = np.clip(action[:6], -1.0, 1.0)

        # Binarize gripper: >0.5 = open (1.0), <=0.5 = close (-1.0)
        action[6] = 1.0 if action[6] > 0.5 else -1.0

        return torch.tensor(action, dtype=torch.float32).unsqueeze(0)  # (1, 7)

    def _build_obs_droid(self, obs: dict, step: int = -1) -> dict:
        """Build observation dict in pi0.5 DROID format.

        DROID expects:
            observation/exterior_image_1_left: (224,224,3) uint8
            observation/wrist_image_left: (224,224,3) uint8
            observation/joint_position: (7,) float32 raw joint positions
            observation/gripper_position: (1,) float32 first finger raw position
            prompt: instruction string
        """
        # -- Images --
        scene_img = _rgba_tensor_to_uint8(obs["scene_rgb"])
        scene_img = _resize_image(scene_img, 224)

        wrist_img = _rgba_tensor_to_uint8(obs["wrist_rgb"])
        wrist_img = _resize_image(wrist_img, 224)

        if self.rotate_image_180:
            scene_img = np.rot90(scene_img, k=2).copy()
            wrist_img = np.rot90(wrist_img, k=2).copy()

        # Save debug images periodically
        if step >= 0 and step % 50 == 0:
            self._save_debug_images(scene_img, wrist_img, step)

        # -- Joint state (raw, unscaled) --
        raw_joint_pos = obs["raw_joint_pos"]
        if torch.is_tensor(raw_joint_pos):
            raw_joint_pos = raw_joint_pos.cpu().numpy()
        if raw_joint_pos.ndim == 2:
            raw_joint_pos = raw_joint_pos[0]
        joint_pos = raw_joint_pos.astype(np.float32)  # (7,)

        raw_gripper_pos = obs["raw_gripper_pos"]
        if torch.is_tensor(raw_gripper_pos):
            raw_gripper_pos = raw_gripper_pos.cpu().numpy()
        if raw_gripper_pos.ndim == 2:
            raw_gripper_pos = raw_gripper_pos[0]
        gripper_pos = np.array([raw_gripper_pos[0]], dtype=np.float32)  # (1,) first finger

        if step >= 0 and step % 10 == 0:
            print(
                f"  [pi0.5/droid] joint_pos: [{', '.join(f'{j:.3f}' for j in joint_pos)}] "
                f"gripper: {gripper_pos[0]:.4f}",
                flush=True,
            )

        return {
            "observation/exterior_image_1_left": scene_img,
            "observation/wrist_image_left": wrist_img,
            "observation/joint_position": joint_pos,
            "observation/gripper_position": gripper_pos,
            "prompt": self.instruction,
        }

    def _scale_action_to_env_droid(self, raw_action: np.ndarray) -> torch.Tensor:
        """Map pi0.5 DROID 8D output to env action space [-1, 1].

        DROID outputs:
            [joint_vel_1..7, gripper_position]
            - Joint velocities in rad/s
            - Gripper: continuous position (>0.5 = open)

        Env expects (joint_vel mode):
            (1, 8) tensor in [-1, 1]
            - [:, 0:7] joint velocities
            - [:, 7] gripper: >0 = open, <0 = close
        """
        action = np.array(raw_action, dtype=np.float32)

        # Ensure 8D
        if len(action) > 8:
            action = action[:8]
        elif len(action) < 8:
            action = np.pad(action, (0, 8 - len(action)), constant_values=0.0)

        # Scale and clip joint velocities
        action[:7] *= self.action_scale_factor
        action[:7] = np.clip(action[:7], -1.0, 1.0)

        # Binarize gripper: >0.5 = open (1.0), <=0.5 = close (-1.0)
        action[7] = 1.0 if action[7] > 0.5 else -1.0

        return torch.tensor(action, dtype=torch.float32).unsqueeze(0)  # (1, 8)

    def _save_debug_images(self, scene_img: np.ndarray, wrist_img: np.ndarray, step: int):
        """Save camera frames to disk for debugging."""
        try:
            debug_dir = os.path.join(os.path.dirname(__file__), "camera_frames")
            os.makedirs(debug_dir, exist_ok=True)
            Image.fromarray(scene_img).save(os.path.join(debug_dir, f"pi05_scene_step{step:04d}.png"))
            Image.fromarray(wrist_img).save(os.path.join(debug_dir, f"pi05_wrist_step{step:04d}.png"))
        except Exception:
            pass

    def reset_action_queue(self):
        """Clear cached actions (call on env reset)."""
        self.action_queue.clear()

    def close(self):
        """Clean up WebSocket connection."""
        if self.client is not None and hasattr(self.client, "close"):
            try:
                self.client.close()
            except Exception:
                pass
        self.client = None
