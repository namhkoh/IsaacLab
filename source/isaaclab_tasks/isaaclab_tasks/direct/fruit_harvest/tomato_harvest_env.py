# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Realistic tomato harvesting environment with spring-based peduncle attachment.

Benchmark for PhD research targeting ICRA/CoRL/RSS venues.
Key features:
- Procedural tomato plant geometry (stem, branches, trusses, peduncles, calyxes)
- Spring-based peduncle attachment with configurable break force (literature: 9.7-28.1N)
- Sub-task decomposition: APPROACH -> GRASP -> DETACH -> TRANSPORT -> PLACE
- Benchmark metrics: success rate, cycle time, grip force, damage rate
- Physical parameters from agricultural robotics literature
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.utils.math import (
    combine_frame_transforms,
    matrix_from_quat,
    quat_inv,
    sample_uniform,
    skew_symmetric_matrix,
    subtract_frame_transforms,
)


# ---------------------------------------------------------------------------
# Plant topology constants
# ---------------------------------------------------------------------------
NUM_TOMATOES = 6  # first 6 of 8 fruits from fruit-001.json

PLANT_OFFSET = (0.0, 0.0, 0.7)  # table height (matches reference fruit_harvest_env)
TOMATO_RADIUS = 0.0215           # exact from fruit-001.json fruit_radii
EE_BODY_OFFSET = 0.107           # panda_hand -> fingertip center (metres, along hand Z)
_TABLE_Z = PLANT_OFFSET[2]       # 0.7

PLANT_JSON_DEFAULT = "D:/research/Plant_Robot_Interaction-master/data/plant/fruit-001.json"

# Keep the original plant topology from fruit-001.json, but fan the fruit
# slightly outward and down so the scripted Franka policy can isolate one
# tomato without inventing unrealistically long stems.
_FALLBACK_TOMATO_POSITIONS: list[tuple[float, float, float]] = [
    (0.02, -0.18, 1.26),
    (0.14, -0.18, 1.25),
    (0.02, -0.09, 1.26),
    (0.14, -0.09, 1.25),
    (0.02, 0.00, 1.26),
    (0.14, 0.00, 1.25),
]
_DEFAULT_TOMATO_HANG_DISTANCES = (0.060, 0.055, 0.070, 0.050, 0.060, 0.055)
_DEFAULT_TOMATO_OUTWARD_SPREAD = 0.025
TOMATO_MASSES: list[float] = [0.022, 0.020, 0.018, 0.020, 0.022, 0.018]
TOMATO_COLORS: list[tuple[float, float, float]] = [
    (0.86, 0.12, 0.08),
    (0.84, 0.14, 0.09),
    (0.80, 0.16, 0.10),
    (0.85, 0.13, 0.08),
    (0.88, 0.11, 0.08),
    (0.83, 0.13, 0.09),
]


def _load_reference_tomato_layout(
    plant_json_path: str = PLANT_JSON_DEFAULT,
) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]]]:
    """Derive fruit rest positions from the source plant topology."""
    if not Path(plant_json_path).is_file():
        return list(_FALLBACK_TOMATO_POSITIONS), list(_FALLBACK_TOMATO_POSITIONS)

    with open(plant_json_path) as f:
        plant = json.load(f)

    positions = np.array(plant["positions"], dtype=np.float64)
    positions[:, 2] += _TABLE_Z
    fathers = plant["fathers"]
    fruit_radii = np.array(plant["fruit_radii"], dtype=np.float64)
    fruit_nodes = [i for i, radius in enumerate(fruit_radii) if radius > 0.0][:NUM_TOMATOES]
    if len(fruit_nodes) < NUM_TOMATOES:
        return list(_FALLBACK_TOMATO_POSITIONS), list(_FALLBACK_TOMATO_POSITIONS)

    anchors = np.array([positions[fathers[node]] for node in fruit_nodes], dtype=np.float64)
    cluster_center_xy = anchors[:, :2].mean(axis=0)

    tomato_positions: list[tuple[float, float, float]] = []
    for i, anchor in enumerate(anchors):
        radial_xy = anchor[:2] - cluster_center_xy
        radial_norm = float(np.linalg.norm(radial_xy))
        if radial_norm < 1e-6:
            direction_xy = np.array([1.0 if i % 2 == 0 else -1.0, 0.0], dtype=np.float64)
        else:
            direction_xy = radial_xy / radial_norm

        xy = anchor[:2] + _DEFAULT_TOMATO_OUTWARD_SPREAD * direction_xy
        z = anchor[2] - _DEFAULT_TOMATO_HANG_DISTANCES[i]
        tomato_positions.append((float(xy[0]), float(xy[1]), float(z)))

    branch_centers = [tuple(float(v) for v in anchor.tolist()) for anchor in anchors]
    return branch_centers, tomato_positions


_BRANCH_CENTERS_REF, TOMATO_POSITIONS = _load_reference_tomato_layout()

# Sub-task phase constants
PHASE_APPROACH = 0
PHASE_GRASP = 1
PHASE_DETACH = 2
PHASE_TRANSPORT = 3
PHASE_PLACE = 4
NUM_PHASES = 5
MIN_BRANCH_RADIUS = 0.003  # m — visual minimum branch radius (matches plant_physics.py)


# ---------------------------------------------------------------------------
# Tomato RigidObject config (dynamic, not kinematic)
# ---------------------------------------------------------------------------

def _make_tomato_cfg(index: int) -> RigidObjectCfg:
    """Create a RigidObjectCfg for a single dynamic tomato sphere."""
    pos = TOMATO_POSITIONS[index]
    return RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/Tomato{index}",
        spawn=sim_utils.SphereCfg(
            radius=TOMATO_RADIUS,
            mass_props=sim_utils.MassPropertiesCfg(mass=TOMATO_MASSES[index]),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                linear_damping=4.0,
                angular_damping=12.0,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=0.2,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005,
                rest_offset=0.001,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=5.0,
                dynamic_friction=5.0,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=TOMATO_COLORS[index],
                roughness=0.35,
                metallic=0.0,
                opacity=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
    )


# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------

@configclass
class TomatoHarvestEnvCfg(DirectRLEnvCfg):
    """Config for the realistic tomato harvesting environment."""

    # env
    episode_length_s = 50.0  # 3000 steps at decimation=2, dt=1/120
    decimation = 2
    action_space = 7   # delta_pos(3) + delta_orient(3) + gripper(1)
    observation_space = 36
    state_space = 0
    control_mode: str = "ik"

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=3.0, replicate_physics=True, clone_in_fabric=False
    )

    # -- Robot-agnostic joint/body config --
    arm_joint_names_expr: list[str] = ["panda_joint.*"]
    finger_joint_names_expr: list[str] = ["panda_finger_joint.*"]
    ee_body_name: str = "panda_hand"
    gripper_open_value: float = 0.04
    gripper_close_value: float = 0.0205  # slight preload for a secure grasp without crushing the fruit

    # -- Cosserat rod plant visual --
    plant_json_path: str = PLANT_JSON_DEFAULT
    plant_rod_scale: float = 3.0
    plant_offset: tuple[float, float, float] = PLANT_OFFSET
    dynamic_peduncle_visual_max_envs: int = 32

    # -- Harvest basket --
    basket_center: tuple[float, float] = (0.45, 0.15)
    basket_bottom_z: float = 0.80
    basket_wall_h: float = 0.15
    basket_inner: float = 0.13
    basket_thick: float = 0.008

    # -- Spring peduncle model (literature: 9.7-28.1N pull break force) --
    spring_stiffness: float = 300.0       # N/m
    spring_damping: float = 10.0          # N*s/m
    spring_break_distance: float = 0.06   # m - catastrophic stretch before forced detach
    spring_break_force: float = 15.0      # N - literature midpoint for a hard yank
    spring_yield_force: float = 8.0       # N - sustained pull needed before stem damage accumulates
    spring_damage_rate: float = 3.5       # 1/s - axial overload accumulation rate
    spring_recovery_rate: float = 2.0     # 1/s - unloading lets the peduncle recover
    spring_damage_exponent: float = 1.5   # >1 reduces single-frame brittleness
    spring_min_axial_stretch: float = 0.008  # m - grasping alone should not detach the fruit
    spring_catastrophic_force_ratio: float = 1.35
    grasp_settle_steps: int = 8           # stable grasp steps before detach logic activates
    peduncle_anchor_noise: float = 0.001  # m - reset jitter without changing stem regime

    # -- Transport spring (fruit tracking fingertip during RETRACT/DELIVER) --
    transport_spring_k: float = 100.0     # N/m
    transport_damping: float = 8.0        # Ns/m
    transport_max_force: float = 3.0      # N - caps total force magnitude
    transport_cutoff: float = 0.30        # m - soft tracking length scale
    transport_pose_alpha: float = 0.35    # post-physics correction for grasped detached fruit
    transport_speed_cap: float = 0.75     # m/s - suppress ballistic detach failures
    hold_alpha: float = 0.92              # attached cluster should remain visually connected under contact
    robot_contact_friction: float = 3.5   # more real holding friction during pull
    grasp_velocity_damping: float = 0.50  # suppress impulsive fruit motion at first contact
    grasp_linear_speed_cap: float = 0.10  # m/s - attached fruit should not ballistic-launch
    grasp_angular_speed_cap: float = 1.5  # rad/s - keeps calyx rotation believable

    # -- Damage thresholds (literature: <8N non-destructive) --
    max_grip_force: float = 8.0           # N - above = fruit damage
    grip_force_k: float = 2000.0          # N/m - effective fruit compression stiffness

    # -- Sub-task reward scales --
    approach_reward_scale: float = 1.0
    grasp_reward_scale: float = 2.0
    detach_reward_scale: float = 10.0
    transport_reward_scale: float = 2.0
    place_reward_scale: float = 15.0
    damage_penalty_scale: float = 5.0
    action_penalty_scale: float = 0.01
    time_penalty_scale: float = 0.001

    # grasp thresholds: generous approach radius, but tighter latch criteria
    grasp_thresh: float = 0.04
    grasp_latch_thresh: float = 0.028
    grasp_latch_speed_max: float = 0.18
    grasp_maintain_thresh: float = 0.042
    grasp_release_gap_margin: float = 0.010

    # tomato positions (for reset convenience)
    tomato_positions: list[tuple[float, float, float]] = list(TOMATO_POSITIONS)

    # robot - Franka Panda with high PD gains for IK tracking
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": -0.569,
                "panda_joint3": 0.0,
                "panda_joint4": -2.810,
                "panda_joint5": 0.0,
                "panda_joint6": 3.037,
                "panda_joint7": 0.741,
                "panda_finger_joint.*": 0.04,
            },
            pos=(0.50, -0.09, 0.50),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit_sim=87.0,
                stiffness=400.0,
                damping=80.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit_sim=12.0,
                stiffness=400.0,
                damping=80.0,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit_sim=110.0,
                stiffness=1.5e3,
                damping=1.5e2,
            ),
        },
    )

    # tomatoes
    tomato_cfgs: list[RigidObjectCfg] = [_make_tomato_cfg(i) for i in range(NUM_TOMATOES)]

    # ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # IK controller scale for delta commands (metres per unit action)
    ik_command_scale = 0.10

    # -- Cameras for VLA training --
    wrist_camera: CameraCfg = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/panda_hand/wrist_cam",
        update_period=0.0,
        height=128,
        width=128,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 2.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.13, 0.0, -0.15),
            rot=(-0.70614, 0.03701, 0.03701, -0.70614),
            convention="ros",
        ),
    )

    scene_camera: CameraCfg = CameraCfg(
        prim_path="/World/envs/env_.*/scene_cam",
        update_period=0.0,
        height=256,
        width=256,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 20.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="world",
        ),
    )


# ---------------------------------------------------------------------------
# Environment implementation
# ---------------------------------------------------------------------------


class TomatoHarvestEnv(DirectRLEnv):
    """Realistic tomato harvesting environment with spring-based peduncle physics.

    Sub-task pipeline: APPROACH -> GRASP -> DETACH -> TRANSPORT -> PLACE
    Actions: delta EE pose (6D) + gripper open/close (1D) = 7D
    Observations: 36D (see _get_observations)
    """

    cfg: TomatoHarvestEnvCfg

    def __init__(self, cfg: TomatoHarvestEnvCfg, render_mode: str | None = None, **kwargs):
        if cfg.control_mode == "joint_vel":
            cfg.action_space = 8  # 7 joint velocities + 1 gripper
        self._dynamic_peduncle_visuals = cfg.scene.num_envs <= cfg.dynamic_peduncle_visual_max_envs
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation
        N = self.num_envs

        # -------------------------------------------------------------------
        # Robot joint / body indices
        # -------------------------------------------------------------------
        self.arm_joint_ids = self._robot.find_joints(self.cfg.arm_joint_names_expr)[0]
        self.finger_joint_ids = self._robot.find_joints(self.cfg.finger_joint_names_expr)[0]
        self.hand_body_idx = self._robot.find_bodies(self.cfg.ee_body_name)[0][0]
        self.ee_jacobi_idx = self.hand_body_idx - 1

        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        # Startup verification
        arm_names = self._robot.find_joints(self.cfg.arm_joint_names_expr)[1]
        print(f"[ENV-INIT] arm_joint_ids={self.arm_joint_ids} names={arm_names}")
        print(f"[ENV-INIT] hand_body_idx={self.hand_body_idx} ee_jacobi_idx={self.ee_jacobi_idx}")
        print(f"[ENV-INIT] num_joints={self._robot.num_joints} num_bodies={self._robot.num_bodies}")
        print(f"[ENV-INIT] tomato positions (world, env0):")
        for i, tp in enumerate(TOMATO_POSITIONS):
            print(f"  fruit {i}: ({tp[0]:.4f}, {tp[1]:.4f}, {tp[2]:.4f})")

        # -------------------------------------------------------------------
        # Differential IK controller
        # -------------------------------------------------------------------
        diff_ik_cfg = DifferentialIKControllerCfg(
            command_type="position",
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": 0.05},
        )
        self.diff_ik_controller = DifferentialIKController(
            diff_ik_cfg, num_envs=N, device=self.device
        )

        # -------------------------------------------------------------------
        # Action / control buffers
        # -------------------------------------------------------------------
        self.robot_dof_targets = torch.zeros((N, self._robot.num_joints), device=self.device)
        self.actions = torch.zeros((N, self.cfg.action_space), device=self.device)
        self.gripper_open = torch.ones(N, dtype=torch.bool, device=self.device)
        self._first_step_after_reset = torch.ones(N, dtype=torch.bool, device=self.device)
        # Persistent IK target in base frame (absolute position)
        self._ik_target_pos_b = torch.zeros((N, 3), device=self.device)
        self._direct_ik_target_set = False  # True when set_ik_target_world() was called

        # -------------------------------------------------------------------
        # Sub-task state machine buffers
        # -------------------------------------------------------------------
        self.sub_task_phase = torch.zeros(N, dtype=torch.long, device=self.device)
        self.target_tomato_idx = torch.zeros(N, dtype=torch.long, device=self.device)
        self.tomato_attached = torch.ones(N, NUM_TOMATOES, dtype=torch.bool, device=self.device)
        self.tomato_anchor_pos = torch.zeros(N, NUM_TOMATOES, 3, device=self.device)
        self.peduncle_rest_length = torch.zeros(N, NUM_TOMATOES, device=self.device)
        self.peduncle_rest_axis = torch.zeros(N, NUM_TOMATOES, 3, device=self.device)
        self.peduncle_damage = torch.zeros(N, NUM_TOMATOES, device=self.device)
        self._tomato_grasped = torch.zeros(N, dtype=torch.bool, device=self.device)
        self._grasp_hold_steps = torch.zeros(N, dtype=torch.long, device=self.device)
        self.tomato_harvested = torch.zeros(N, dtype=torch.bool, device=self.device)

        # -------------------------------------------------------------------
        # Metric buffers
        # -------------------------------------------------------------------
        self.peak_grip_force = torch.zeros(N, device=self.device)
        self.detach_force = torch.zeros(N, device=self.device)
        self.tomato_damaged = torch.zeros(N, dtype=torch.bool, device=self.device)
        self._episode_steps = torch.zeros(N, dtype=torch.long, device=self.device)
        self._ik_substep_count = 0  # global substep counter for diagnostics

        # Initial tomato positions as tensor
        self.tomato_init_positions = torch.tensor(
            self.cfg.tomato_positions, device=self.device, dtype=torch.float32
        )

        # Body-offset: panda_hand -> fingertip center (for body-offset Jacobian IK)
        self._body_offset_pos = torch.tensor(
            [[0.0, 0.0, EE_BODY_OFFSET]], device=self.device, dtype=torch.float32
        ).expand(N, -1).contiguous()
        self._body_offset_rot = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]], device=self.device, dtype=torch.float32
        ).expand(N, -1).contiguous()

        # Basket center in local coords (world-space version computed after scene setup)
        self._basket_center_local = torch.tensor(
            [self.cfg.basket_center[0], self.cfg.basket_center[1],
             self.cfg.basket_bottom_z + self.cfg.basket_wall_h / 2.0],
            device=self.device, dtype=torch.float32,
        )
        # Per-env world-space basket center — set after scene.env_origins is available
        self._basket_center_w: torch.Tensor | None = None
        self._dynamic_peduncle_visuals = self.num_envs <= self.cfg.dynamic_peduncle_visual_max_envs

        self._tune_robot_contact_materials()

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        # Spawn tomato rigid bodies (dynamic)
        self._tomatoes: list[RigidObject] = []
        for i, tomato_cfg in enumerate(self.cfg.tomato_cfgs):
            tomato = RigidObject(tomato_cfg)
            self._tomatoes.append(tomato)
            self.scene.rigid_objects[f"tomato{i}"] = tomato

        self._add_tomato_detail_visuals()

        # Cameras
        self._wrist_camera = Camera(self.cfg.wrist_camera)
        self.scene.sensors["wrist_camera"] = self._wrist_camera
        self._scene_camera = Camera(self.cfg.scene_camera)
        self.scene.sensors["scene_camera"] = self._scene_camera

        # --- Per-env geometry (spawned BEFORE clone so every env gets a copy) ---
        ENV = "/World/envs/env_.*"

        # Tables (matching reference: 0.7m table height) — with collision so arm can't pass through
        table_cfg = sim_utils.CuboidCfg(
            size=(0.35, 0.35, 0.7),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.30, 0.15)),
        )
        table_cfg.func(f"{ENV}/PlantTable", table_cfg, translation=(0.0, 0.0, 0.35))
        pedestal_cfg = sim_utils.CuboidCfg(
            size=(0.22, 0.28, 0.50),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.30, 0.15)),
        )
        pedestal_cfg.func(f"{ENV}/RobotStand", pedestal_cfg, translation=(0.50, -0.09, 0.25))

        # Cosserat rod plant (visual only, per-env)
        self._spawn_cosserat_plant()

        # Harvest basket (with collision, per-env)
        self._spawn_basket()

        # Ground plane
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # --- Post-clone: global-only geometry (lighting) ---
        light_cfg = sim_utils.DomeLightCfg(intensity=5000.0, color=(0.8, 0.8, 0.8))
        light_cfg.func("/World/Light", light_cfg)

        self._scene_cam_initialized = False

    def _tune_robot_contact_materials(self):
        """Increase robot contact friction to match the stable demo setup."""
        try:
            mats = self._robot.root_physx_view.get_material_properties()
            mats[..., 0] = self.cfg.robot_contact_friction
            mats[..., 1] = self.cfg.robot_contact_friction
            mats[..., 2] = 0.0
            self._robot.root_physx_view.set_material_properties(
                mats, torch.arange(self.num_envs, device="cpu")
            )
        except Exception as exc:
            print(f"[TomatoHarvestEnv] Robot contact tuning skipped: {exc}")

    def _add_tomato_detail_visuals(self):
        """Attach a tomato-style calyx and an oblate visual shell to each fruit."""
        from pxr import Gf, Sdf, UsdGeom, UsdShade

        stage = self.scene.stage
        looks_root = stage.DefinePrim("/World/TomatoLooks", "Xform")
        _ = looks_root

        calyx_mat = UsdShade.Material.Define(stage, "/World/TomatoLooks/CalyxMat")
        calyx_sh = UsdShade.Shader.Define(stage, "/World/TomatoLooks/CalyxMat/Shader")
        calyx_sh.CreateIdAttr("UsdPreviewSurface")
        calyx_sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.18, 0.45, 0.10))
        calyx_sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.65)
        calyx_mat.CreateSurfaceOutput().ConnectToSource(calyx_sh.ConnectableAPI(), "surface")

        for t_idx in range(NUM_TOMATOES):
            fruit_prim_path = f"/World/envs/env_0/Tomato{t_idx}"
            fruit_prim = stage.GetPrimAtPath(fruit_prim_path)
            if not fruit_prim.IsValid():
                continue

            tomato_mat = UsdShade.Material.Define(stage, f"/World/TomatoLooks/TomatoMat{t_idx}")
            tomato_sh = UsdShade.Shader.Define(stage, f"/World/TomatoLooks/TomatoMat{t_idx}/Shader")
            tomato_sh.CreateIdAttr("UsdPreviewSurface")
            tomato_sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(*TOMATO_COLORS[t_idx])
            )
            tomato_sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.28)
            tomato_mat.CreateSurfaceOutput().ConnectToSource(tomato_sh.ConnectableAPI(), "surface")

            shell = UsdGeom.Sphere.Define(stage, f"{fruit_prim_path}/TomatoVisual")
            shell.GetRadiusAttr().Set(TOMATO_RADIUS)
            shell_xf = UsdGeom.XformCommonAPI(shell)
            shell_xf.SetScale(Gf.Vec3f(1.06, 1.02, 0.90))
            shell_xf.SetTranslate(Gf.Vec3d(0.0, 0.0, -TOMATO_RADIUS * 0.03))
            UsdShade.MaterialBindingAPI(shell.GetPrim()).Bind(tomato_mat)

            num_sepals = 5
            sepal_tilt = 65.0
            sepal_len = TOMATO_RADIUS * 0.80
            sepal_base_r = TOMATO_RADIUS * 0.15
            for si in range(num_sepals):
                angle = si * 72.0
                sepal = UsdGeom.Cone.Define(stage, f"{fruit_prim_path}/Sepal{si}")
                sepal.GetRadiusAttr().Set(sepal_base_r)
                sepal.GetHeightAttr().Set(sepal_len)
                sepal.GetAxisAttr().Set("Z")
                sepal_xf = UsdGeom.XformCommonAPI(sepal)
                sepal_xf.SetScale(Gf.Vec3f(1.0, 0.35, 1.0))
                sepal_xf.SetRotate(Gf.Vec3f(0.0, -sepal_tilt, angle))
                sepal_xf.SetTranslate(Gf.Vec3d(0.0, 0.0, TOMATO_RADIUS * 0.86))
                UsdShade.MaterialBindingAPI(sepal.GetPrim()).Bind(calyx_mat)

            stem = UsdGeom.Cylinder.Define(stage, f"{fruit_prim_path}/CalyxStem")
            stem.GetRadiusAttr().Set(0.0015)
            stem.GetHeightAttr().Set(TOMATO_RADIUS * 0.35)
            stem.GetAxisAttr().Set("Z")
            stem_xf = UsdGeom.XformCommonAPI(stem)
            stem_xf.SetTranslate(Gf.Vec3d(0.0, 0.0, TOMATO_RADIUS * 1.08))
            UsdShade.MaterialBindingAPI(stem.GetPrim()).Bind(calyx_mat)

    # ------------------------------------------------------------------
    # Cosserat rod plant geometry (visual only)
    # ------------------------------------------------------------------
    def _spawn_cosserat_plant(self):
        """Spawn Cosserat rod plant from Plant_Robot_Interaction JSON data.

        Creates capsule segments for each rod edge in the tree topology
        (rounded ends give smooth joints — no branch-point spheres needed).
        Uses max(parent, child) radius per segment for organic taper.
        Color gradient from brown (trunk) to green (canopy).
        Also spawns peduncle segments and calyxes connecting each tomato
        to its branch attachment point on the plant.
        Falls back to a simple stem if JSON file is not found.
        """
        if not self.cfg.plant_json_path or not os.path.isfile(self.cfg.plant_json_path):
            self._spawn_simple_plant_fallback()
            return

        with open(self.cfg.plant_json_path) as f:
            plant = json.load(f)

        positions = np.array(plant["positions"], dtype=np.float64)
        fathers = plant["fathers"]
        radii_arr = np.array(plant["radii"], dtype=np.float64)
        fruit_radii_arr = np.array(plant["fruit_radii"], dtype=np.float64)

        # Apply offset only (no scaling — raw positions are in meters)
        offset = np.array(self.cfg.plant_offset, dtype=np.float64)
        positions = positions + offset

        z_min = float(positions[:, 2].min())
        z_range = max(float(positions[:, 2].max()) - z_min, 0.01)

        # Identify fruit nodes and their parent (branch attachment) positions
        fruit_nodes = [i for i in range(len(positions)) if fruit_radii_arr[i] > 0]

        ENV = "/World/envs/env_.*"
        seg_count = 0
        for i in range(len(positions)):
            fi = fathers[i]
            if fi == -1:
                continue  # root node has no parent segment
            if fruit_radii_arr[i] > 0:
                continue  # fruit nodes handled by RigidObject tomatoes

            p1 = positions[fi]
            p2 = positions[i]
            direction = p2 - p1
            length = float(np.linalg.norm(direction))
            if length < 1e-6:
                continue

            midpoint = (p1 + p2) / 2.0
            # Use max(parent, child) radius for smooth taper (matches plant_physics.py)
            rod_radius = float(max(radii_arr[fi], radii_arr[i], MIN_BRANCH_RADIUS))

            # Color gradient: brown trunk -> green canopy
            z_frac = (p2[2] - z_min) / z_range
            color = self._plant_color_by_height(z_frac)

            quat = self._direction_to_quat_z(direction)

            # Capsule geometry — rounded ends give smooth joints between segments
            cap_cfg = sim_utils.CapsuleCfg(
                radius=rod_radius,
                height=length,
                axis="Z",
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )
            cap_cfg.func(
                f"{ENV}/CosseratPlant_Seg{seg_count}",
                cap_cfg,
                translation=tuple(midpoint.tolist()),
                orientation=quat,
            )
            seg_count += 1

        # --- Peduncles + calyxes: connect each tomato to its branch attachment ---
        peduncle_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.20, 0.55, 0.15), roughness=0.7
        )
        attach_points: list[tuple[float, float, float]] = []
        for t_idx in range(min(NUM_TOMATOES, len(fruit_nodes))):
            fruit_node = fruit_nodes[t_idx]
            parent_node = fathers[fruit_node]
            attach_pos = positions[parent_node]  # branch attachment point (scaled)
            tomato_pos = np.array(TOMATO_POSITIONS[t_idx], dtype=np.float64)
            attach_points.append(tuple(attach_pos.tolist()))

            # Peduncle: thin green cylinder from branch to tomato top
            tomato_top = tomato_pos + np.array([0.0, 0.0, TOMATO_RADIUS])
            self._spawn_cylinder_segment(
                f"{ENV}/Peduncle{t_idx}",
                attach_pos, tomato_top, 0.002, peduncle_material,
            )

        # Remaining tomatoes (if more than fruit nodes): use position directly above
        for t_idx in range(len(fruit_nodes), NUM_TOMATOES):
            tp = np.array(TOMATO_POSITIONS[t_idx], dtype=np.float64)
            ap = (tp[0], tp[1], tp[2] + TOMATO_RADIUS + 0.03)
            attach_points.append(ap)
            self._spawn_cylinder_segment(
                f"{ENV}/Peduncle{t_idx}",
                ap, (tp[0], tp[1], tp[2] + TOMATO_RADIUS), 0.002, peduncle_material,
            )
        # Store branch attachment points for spring physics
        self._peduncle_attach_points = attach_points

        print(
            f"[TomatoHarvestEnv] Spawned Cosserat rod plant: {seg_count} capsule segments, "
            f"{len(attach_points)} peduncles"
        )

    def _spawn_simple_plant_fallback(self):
        """Minimal fallback plant: single green stem if JSON not found."""
        ENV = "/World/envs/env_.*"
        stem_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.20, 0.55, 0.15), roughness=0.7
        )
        self._spawn_cylinder_segment(
            f"{ENV}/PlantStem", (0.0, 0.0, _TABLE_Z), (0.0, 0.0, _TABLE_Z + 0.40), 0.010, stem_material
        )
        # No peduncle data — anchor will fall back to tomato positions
        self._peduncle_attach_points = []
        print("[TomatoHarvestEnv] Fallback: simple plant stem (JSON not found)")

    @staticmethod
    def _plant_color_by_height(z_frac: float) -> tuple[float, float, float]:
        """Return (R,G,B) color for a plant segment based on normalized height [0,1]."""
        if z_frac < 0.25:
            return (0.40, 0.26, 0.13)  # dark brown trunk
        elif z_frac < 0.50:
            t = (z_frac - 0.25) / 0.25
            return (0.40 - 0.20 * t, 0.26 + 0.24 * t, 0.13)  # brown -> green
        else:
            return (0.18, 0.52, 0.18)  # green canopy

    def _spawn_cylinder_segment(self, path: str, p1, p2, radius: float, material):
        """Spawn a cylinder segment between two 3D points."""
        p1 = np.array(p1, dtype=np.float64)
        p2 = np.array(p2, dtype=np.float64)
        direction = p2 - p1
        length = float(np.linalg.norm(direction))
        if length < 1e-6:
            return
        midpoint = (p1 + p2) / 2.0
        quat = self._direction_to_quat_z(direction)

        cyl_cfg = sim_utils.CylinderCfg(
            radius=radius,
            height=length,
            axis="Z",
            visual_material=material,
        )
        cyl_cfg.func(path, cyl_cfg, translation=tuple(midpoint.tolist()), orientation=quat)

    @staticmethod
    def _direction_to_quat_z(direction) -> tuple[float, float, float, float]:
        """Compute quaternion (w,x,y,z) to rotate Z-axis to align with direction."""
        d = np.array(direction, dtype=np.float64)
        length = np.linalg.norm(d)
        if length < 1e-12:
            return (1.0, 0.0, 0.0, 0.0)
        d = d / length
        z_axis = np.array([0.0, 0.0, 1.0])
        dot = float(np.dot(z_axis, d))
        if dot > 0.99999:
            return (1.0, 0.0, 0.0, 0.0)
        if dot < -0.99999:
            return (0.0, 1.0, 0.0, 0.0)
        axis = np.cross(z_axis, d)
        axis = axis / np.linalg.norm(axis)
        angle = math.acos(max(-1.0, min(1.0, dot)))
        w = math.cos(angle / 2.0)
        s = math.sin(angle / 2.0)
        return (w, float(axis[0] * s), float(axis[1] * s), float(axis[2] * s))

    @staticmethod
    def _quat_from_z_axis_torch(direction: torch.Tensor) -> torch.Tensor:
        """Return quaternions (w,x,y,z) rotating +Z onto the given direction."""
        d = direction / direction.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        z_axis = torch.tensor([0.0, 0.0, 1.0], device=d.device, dtype=d.dtype).expand_as(d)
        dot = (z_axis * d).sum(dim=-1, keepdim=True)

        quat = torch.zeros(d.shape[0], 4, device=d.device, dtype=d.dtype)
        same = dot.squeeze(-1) > 0.99999
        opposite = dot.squeeze(-1) < -0.99999
        general = ~(same | opposite)

        quat[same, 0] = 1.0
        quat[opposite, 1] = 1.0

        if general.any():
            axis = torch.cross(z_axis[general], d[general], dim=-1)
            axis = axis / axis.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            angle = torch.acos(dot[general].clamp(-1.0, 1.0))
            half = 0.5 * angle
            quat[general, 0] = torch.cos(half).squeeze(-1)
            quat[general, 1:] = axis * torch.sin(half)

        return quat

    def _update_peduncle_visuals(
        self,
        fruit_pos_w: torch.Tensor | None = None,
        fruit_quat_w: torch.Tensor | None = None,
    ) -> None:
        """Keep peduncle visuals attached to the live fruit pose for small-env runs."""
        if not getattr(self, "_dynamic_peduncle_visuals", False):
            return
        if not hasattr(self, "_peduncle_attach_points") or len(self._peduncle_attach_points) != NUM_TOMATOES:
            return
        if not hasattr(self, "tomato_attached") or not hasattr(self, "tomato_anchor_pos"):
            return

        from pxr import Gf, UsdGeom

        stage = self.scene.stage
        if fruit_pos_w is None:
            fruit_pos_w = torch.stack([t.data.root_pos_w[:, :3] for t in self._tomatoes], dim=1)
        if fruit_quat_w is None:
            fruit_quat_w = torch.stack([t.data.root_state_w[:, 3:7] for t in self._tomatoes], dim=1)

        rot = matrix_from_quat(fruit_quat_w.reshape(-1, 4)).reshape(self.num_envs, NUM_TOMATOES, 3, 3)
        local_top = torch.tensor([0.0, 0.0, TOMATO_RADIUS * 0.92], device=self.device, dtype=torch.float32)
        tomato_top = fruit_pos_w + torch.einsum("ntij,j->nti", rot, local_top)

        for env_id in range(self.num_envs):
            for t_idx in range(NUM_TOMATOES):
                prim_path = f"/World/envs/env_{env_id}/Peduncle{t_idx}"
                prim = stage.GetPrimAtPath(prim_path)
                if not prim.IsValid():
                    continue
                mesh_prim = stage.GetPrimAtPath(f"{prim_path}/geometry/mesh")

                imageable = UsdGeom.Imageable(prim)
                attached = bool(self.tomato_attached[env_id, t_idx].item())
                if not attached:
                    imageable.GetVisibilityAttr().Set("invisible")
                    continue
                imageable.GetVisibilityAttr().Set("inherited")

                p1 = self.tomato_anchor_pos[env_id, t_idx].detach().cpu().numpy()
                p2 = tomato_top[env_id, t_idx].detach().cpu().numpy()
                direction = p2 - p1
                length = float(np.linalg.norm(direction))
                if length < 1e-6:
                    imageable.GetVisibilityAttr().Set("invisible")
                    continue

                midpoint = (p1 + p2) / 2.0
                quat = self._direction_to_quat_z(direction)

                xform = UsdGeom.Xformable(prim)
                translate_op = None
                orient_op = None
                for op in xform.GetOrderedXformOps():
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate and translate_op is None:
                        translate_op = op
                    elif op.GetOpType() == UsdGeom.XformOp.TypeOrient and orient_op is None:
                        orient_op = op
                if translate_op is None:
                    translate_op = xform.AddTranslateOp()
                if orient_op is None:
                    orient_op = xform.AddOrientOp()
                translate_op.Set(Gf.Vec3d(float(midpoint[0]), float(midpoint[1]), float(midpoint[2])))
                orient_op.Set(Gf.Quatd(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])))

                if mesh_prim.IsValid() and mesh_prim.GetTypeName() == "Cylinder":
                    UsdGeom.Cylinder(mesh_prim).GetHeightAttr().Set(length)

    # ------------------------------------------------------------------
    # Harvest basket (reused from FruitHarvestEnv)
    # ------------------------------------------------------------------
    def _spawn_basket(self):
        """Spawn a harvest basket with floor + 4 walls (per-env, with collision)."""
        ENV = "/World/envs/env_.*"
        bx, by = self.cfg.basket_center
        bz = self.cfg.basket_bottom_z
        wall_h = self.cfg.basket_wall_h
        inner = self.cfg.basket_inner
        thick = self.cfg.basket_thick
        outer = inner + 2 * thick
        wall_z = bz + thick + wall_h / 2.0
        half_off = inner / 2.0 + thick / 2.0

        basket_color = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.5))

        stand_cfg = sim_utils.CuboidCfg(
            size=(outer + 0.02, outer + 0.02, bz),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.30, 0.15)),
        )
        stand_cfg.func(f"{ENV}/BasketStand", stand_cfg, translation=(bx, by, bz / 2.0))

        pieces = [
            ("BasketFloor",     (bx, by, bz + thick / 2.0),       (outer, outer, thick)),
            ("BasketWallFront", (bx, by + half_off, wall_z),       (outer, thick, wall_h)),
            ("BasketWallBack",  (bx, by - half_off, wall_z),       (outer, thick, wall_h)),
            ("BasketWallLeft",  (bx - half_off, by, wall_z),       (thick, inner, wall_h)),
            ("BasketWallRight", (bx + half_off, by, wall_z),       (thick, inner, wall_h)),
        ]
        for name, pos, size in pieces:
            cfg = sim_utils.CuboidCfg(
                size=size,
                visual_material=basket_color,
                collision_props=sim_utils.CollisionPropertiesCfg(),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            )
            cfg.func(f"{ENV}/{name}", cfg, translation=pos)

    # ------------------------------------------------------------------
    # Pre-physics step (IK / joint vel)
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone().clamp(-1.0, 1.0)
        if self.cfg.control_mode == "joint_vel":
            self._pre_physics_step_joint_vel()
        else:
            self._pre_physics_step_ik()

    def _pre_physics_step_joint_vel(self):
        """Joint-velocity control mode: actions[:, :7] = joint vel, actions[:, 7] = gripper."""
        joint_vel = self.actions[:, :7]
        gripper_actions = self.actions[:, 7]
        self.gripper_open = gripper_actions > 0.0

        arm_joint_pos = self._robot.data.joint_pos[:, self.arm_joint_ids]
        arm_joint_pos_des = arm_joint_pos + joint_vel * self.dt
        arm_joint_pos_des[self._first_step_after_reset] = self._robot.data.default_joint_pos[
            self._first_step_after_reset
        ][:, self.arm_joint_ids]
        self._first_step_after_reset[:] = False

        arm_lower = self.robot_dof_lower_limits[self.arm_joint_ids]
        arm_upper = self.robot_dof_upper_limits[self.arm_joint_ids]
        arm_joint_pos_des = torch.clamp(arm_joint_pos_des, arm_lower, arm_upper)
        self.robot_dof_targets[:, self.arm_joint_ids] = arm_joint_pos_des

        finger_target = torch.where(
            self.gripper_open.unsqueeze(-1),
            torch.full((self.num_envs, len(self.finger_joint_ids)), self.cfg.gripper_open_value, device=self.device),
            torch.full((self.num_envs, len(self.finger_joint_ids)), self.cfg.gripper_close_value, device=self.device),
        )
        self.robot_dof_targets[:, self.finger_joint_ids] = finger_target

    def _pre_physics_step_ik(self):
        """Set the IK target for this env step (absolute or delta mode).

        Supports two modes for setting the IK target:
        1. Direct world-space target (scripted): call set_ik_target_world()
           before env.step() to set an exact world-space position. The arm
           converges to it directly — no deltas, no scaling, no accumulation.
        2. Delta actions (RL/VLA): actions[:, :3] are position deltas that
           nudge a persistent target. Fallback when no direct target is set.

        The actual IK solve runs every substep in _solve_ik_substep().
        """
        gripper_actions = self.actions[:, 6]
        self.gripper_open = gripper_actions > 0.0

        ft_pos_b, _ = self._fingertip_in_base()

        if self._direct_ik_target_set:
            # Direct target mode: _ik_target_pos_b was already set by
            # set_ik_target_world() — use it as-is. Skip the first-step
            # initialization (which would overwrite the direct target).
            self._direct_ik_target_set = False
            self._first_step_after_reset[:] = False
        else:
            # First step after reset (delta mode only): initialize target
            # to current fingertip so deltas start from a valid position.
            first = self._first_step_after_reset
            if first.any():
                self._ik_target_pos_b[first] = ft_pos_b[first]
            self._first_step_after_reset[:] = False
            # Delta mode (RL/VLA): accumulate delta onto persistent target
            pos_delta = self.actions[:, :3] * self.cfg.ik_command_scale
            self._ik_target_pos_b = self._ik_target_pos_b + pos_delta

            # Safety clamp: target can't drift more than 50cm from fingertip
            disp = self._ik_target_pos_b - ft_pos_b
            dist = disp.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            max_target_dist = 0.50
            self._ik_target_pos_b = torch.where(
                dist > max_target_dist,
                ft_pos_b + disp * (max_target_dist / dist),
                self._ik_target_pos_b,
            )

        # Workspace Z-floor: keep IK target above the table top.
        # Table top = 0.70m world, robot base Z=0.50m → 0.20m in base frame.
        min_z_base = 0.22
        self._ik_target_pos_b[:, 2] = torch.clamp(
            self._ik_target_pos_b[:, 2], min=min_z_base
        )

        # Gripper targets
        finger_target = torch.where(
            self.gripper_open.unsqueeze(-1),
            torch.full((self.num_envs, len(self.finger_joint_ids)), self.cfg.gripper_open_value, device=self.device),
            torch.full((self.num_envs, len(self.finger_joint_ids)), self.cfg.gripper_close_value, device=self.device),
        )
        self.robot_dof_targets[:, self.finger_joint_ids] = finger_target

    def set_ik_target_world(self, target_world: torch.Tensor):
        """Set an absolute IK target in world coordinates.

        Call this BEFORE env.step() to command the arm to go to an exact
        world-space position. The arm converges toward it using per-substep
        IK — exactly like the reference demo's set_ik_target().

        Args:
            target_world: (num_envs, 3) fingertip target in world frame.
        """
        rp = self._robot.data.root_pose_w[:, :3]
        rq = self._robot.data.root_pose_w[:, 3:7]
        dummy_q = self._body_offset_rot
        target_b, _ = subtract_frame_transforms(rp, rq, target_world, dummy_q)
        self._ik_target_pos_b = target_b
        self._direct_ik_target_set = True

        # Diagnostic: print on first few calls
        step = int(self._episode_steps[0].item())
        if step < 3:
            tw = target_world[0]
            tb = target_b[0]
            ft_w = self._fingertip_world()[0]
            reach = target_b[0].norm().item()
            print(f"[IK-SET] step={step} target_world=({tw[0]:.4f},{tw[1]:.4f},{tw[2]:.4f}) "
                  f"target_base=({tb[0]:.4f},{tb[1]:.4f},{tb[2]:.4f}) "
                  f"ft_world=({ft_w[0]:.4f},{ft_w[1]:.4f},{ft_w[2]:.4f}) "
                  f"reach={reach:.4f}m")

    def _fingertip_in_base(self):
        """Return (ft_pos_b, ft_quat_b) — fingertip position/orientation in robot base frame."""
        root_pose_w = self._robot.data.root_pose_w
        rp_w = root_pose_w[:, 0:3]
        rq_w = root_pose_w[:, 3:7]
        ee_pose_w = self._robot.data.body_pose_w[:, self.hand_body_idx]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(rp_w, rq_w, ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        ft_pos_b, ft_quat_b = combine_frame_transforms(
            ee_pos_b, ee_quat_b, self._body_offset_pos, self._body_offset_rot
        )
        return ft_pos_b, ft_quat_b

    def _solve_ik_substep(self):
        """Recompute IK from current joint state toward the persistent absolute target.

        Called every substep (inside _apply_action), matching the reference demo
        where IK runs every physics step. The absolute target (_ik_target_pos_b)
        stays fixed across substeps, so the error naturally decreases as the arm
        converges — no oscillation.
        """
        root_pose_w = self._robot.data.root_pose_w
        rp_w = root_pose_w[:, 0:3]
        rq_w = root_pose_w[:, 3:7]

        ee_pose_w = self._robot.data.body_pose_w[:, self.hand_body_idx]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(rp_w, rq_w, ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        ft_pos_b, ft_quat_b = combine_frame_transforms(
            ee_pos_b, ee_quat_b, self._body_offset_pos, self._body_offset_rot
        )

        # Set absolute IK target (same target across substeps — convergence!)
        self.diff_ik_controller.set_command(self._ik_target_pos_b, ee_quat=ft_quat_b)

        # Jacobian: world frame -> base frame (matches reference demo exactly)
        jacobian_w = self._robot.root_physx_view.get_jacobians()[:, self.ee_jacobi_idx, :, self.arm_joint_ids]
        qc = rq_w.clone()
        qc[:, 1:] *= -1.0  # quaternion conjugate (same as reference demo)
        base_rot_matrix = matrix_from_quat(qc)
        jacobian_b = jacobian_w.clone()
        jacobian_b[:, :3, :] = torch.bmm(base_rot_matrix, jacobian_w[:, :3, :])
        jacobian_b[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian_w[:, 3:, :])

        # Body offset Jacobian correction (panda_hand -> fingertip)
        ee_rot_b = matrix_from_quat(ee_quat_b)
        offset_in_base = torch.bmm(ee_rot_b, self._body_offset_pos.unsqueeze(-1)).squeeze(-1)
        jacobian_b[:, 0:3, :] += torch.bmm(
            -skew_symmetric_matrix(offset_in_base), jacobian_b[:, 3:, :]
        )

        # Solve IK (position-only: top 3 Jacobian rows)
        arm_joint_pos = self._robot.data.joint_pos[:, self.arm_joint_ids]
        arm_joint_pos_des = self.diff_ik_controller.compute(
            ft_pos_b, ft_quat_b, jacobian_b, arm_joint_pos
        )

        # One-time Jacobian health check
        if self._ik_substep_count == 0:
            J_sv = torch.linalg.svdvals(jacobian_b[0, :3, :])
            print(f"[IK-INIT] Jacobian(pos) SVD: {J_sv.tolist()}")
            print(f"[IK-INIT] Jacobian shape: {jacobian_b.shape}")
            err0 = self._ik_target_pos_b[0] - ft_pos_b[0]
            print(f"[IK-INIT] initial error: ({err0[0]:.4f},{err0[1]:.4f},{err0[2]:.4f}) |err|={err0.norm():.4f}")

        # Safety: cap per-substep joint delta norm, then clamp to limits
        # (matches reference demo order: delta-scale first, joint-clamp last)
        delta = arm_joint_pos_des - arm_joint_pos
        norm = delta.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = torch.clamp(0.2 / norm, max=1.0)
        arm_joint_pos_des = arm_joint_pos + delta * scale
        arm_lower = self.robot_dof_lower_limits[self.arm_joint_ids]
        arm_upper = self.robot_dof_upper_limits[self.arm_joint_ids]
        arm_joint_pos_des = torch.clamp(arm_joint_pos_des, arm_lower, arm_upper)

        self.robot_dof_targets[:, self.arm_joint_ids] = arm_joint_pos_des

        # Diagnostic logging (first 10 substeps then every 200)
        self._ik_substep_count += 1
        sc = self._ik_substep_count
        if sc <= 10 or sc % 200 == 0:
            err = self._ik_target_pos_b[0] - ft_pos_b[0]
            err_norm = err.norm().item()
            tb = self._ik_target_pos_b[0]
            fb = ft_pos_b[0]
            dn = norm[0].item()
            print(f"[IK-SUB] substep={sc} "
                  f"tgt_b=({tb[0]:.4f},{tb[1]:.4f},{tb[2]:.4f}) "
                  f"ft_b=({fb[0]:.4f},{fb[1]:.4f},{fb[2]:.4f}) "
                  f"|err|={err_norm:.4f} |delta_q|={dn:.4f}")

    # ------------------------------------------------------------------
    # Apply action: IK solve + spring physics + state machine
    # ------------------------------------------------------------------
    def _apply_action(self):
        # Alpha-blend hold: corrects idle fruit positions after previous
        # substep's physics (matches reference demo's post_step_hold pattern).
        self._post_physics_hold_fruits()

        # 1. Recompute IK from current state toward persistent target (per-substep)
        if self.cfg.control_mode != "joint_vel":
            self._solve_ik_substep()

        # 2. Set joint targets
        self._robot.set_joint_position_target(
            self.robot_dof_targets[:, self.arm_joint_ids], joint_ids=self.arm_joint_ids
        )
        self._robot.set_joint_position_target(
            self.robot_dof_targets[:, self.finger_joint_ids], joint_ids=self.finger_joint_ids
        )

        # Increment episode step counter
        self._episode_steps += 1

        # 3. PRE-physics forces: peduncle stretch for grasped+attached
        self._apply_spring_forces()

        # -- Grasp detection (based on ACTUAL finger position, not command) --
        ft_pos = self._fingertip_world()
        target_pos = self._get_target_tomato_positions()
        dist = torch.norm(ft_pos - target_pos, dim=-1)

        # Check actual finger gap (sum of both finger joint positions).
        # Open: 2*0.04=0.08m. At fruit contact: 2*TOMATO_RADIUS=0.043m.
        # Latch only when the target fruit is centered, mostly stationary, and
        # the fingers are actually closed around it.
        finger_pos = self._robot.data.joint_pos[:, self.finger_joint_ids]
        finger_gap = finger_pos.sum(dim=-1)  # total gap (both fingers)
        target_speed = torch.norm(self._get_target_tomato_linear_velocities(), dim=-1)
        gripper_actually_closed = finger_gap < (2.0 * TOMATO_RADIUS + 0.004)
        close_enough = dist < self.cfg.grasp_latch_thresh
        target_stable = target_speed < self.cfg.grasp_latch_speed_max

        newly_grasped = close_enough & gripper_actually_closed & target_stable & ~self._tomato_grasped & ~self.tomato_harvested
        self._tomato_grasped = self._tomato_grasped | newly_grasped

        # Grasp must be maintained, not just latched once. If the fruit center is
        # no longer near the fingertip or the fingers are no longer closed around
        # it, release the controller-side grasp so the fruit cannot stay glued.
        maintain_closed = finger_gap < (2.0 * TOMATO_RADIUS + self.cfg.grasp_release_gap_margin)
        maintain_contact = dist < self.cfg.grasp_maintain_thresh
        maintained_grasp = self._tomato_grasped & maintain_closed & maintain_contact

        released = self.gripper_open & self._tomato_grasped
        slipped = self._tomato_grasped & ~maintained_grasp & ~released
        self._tomato_grasped = maintained_grasp & ~released
        self._grasp_hold_steps = torch.where(
            self._tomato_grasped,
            self._grasp_hold_steps + 1,
            torch.zeros_like(self._grasp_hold_steps),
        )
        if slipped.any():
            self.sub_task_phase[slipped & (self.sub_task_phase >= PHASE_TRANSPORT)] = PHASE_APPROACH

        # -- Grip force estimation (Hooke's law from finger displacement) --
        self._estimate_grip_force()

        # -- State machine transitions --
        gripper_closed = ~self.gripper_open
        self._update_state_machine(ft_pos, target_pos, dist, gripper_closed)

        # 4. PRE-physics forces: spring-damper transport for grasped+detached
        self._apply_transport_forces(ft_pos)

        # -- Scene camera --
        if hasattr(self, "_scene_camera"):
            scene_eye = torch.tensor([[2.5, 1.3, 1.8]], device=self.device).expand(self.num_envs, -1)
            scene_target = torch.tensor([[0.2, 0.0, 0.90]], device=self.device).expand(self.num_envs, -1)
            self._scene_camera.set_world_poses_from_view(scene_eye, scene_target)

    def _post_physics_hold_fruits(self):
        """Blend idle fruit toward rest while damping unrealistic grasp-time spin.

        The original reference demo could hard-reset orientation because the
        fruit visuals were spheres. With a tomato-shaped visual shell and
        calyx, we must preserve the simulated quaternion and only damp
        implausibly large velocities.
        """
        env_origins = self.scene.env_origins
        ft_pos = self._fingertip_world()
        HOLD_ALPHA = self.cfg.hold_alpha
        TRANSPORT_ALPHA = self.cfg.transport_pose_alpha
        visual_pos = torch.stack([t.data.root_pos_w[:, :3].clone() for t in self._tomatoes], dim=1)
        visual_quat = torch.stack([t.data.root_state_w[:, 3:7].clone() for t in self._tomatoes], dim=1)

        def cap_speed(vec: torch.Tensor, max_speed: float) -> torch.Tensor:
            speed = vec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            return vec * torch.clamp(max_speed / speed, max=1.0)

        for t_idx in range(NUM_TOMATOES):
            tomato = self._tomatoes[t_idx]
            attached = self.tomato_attached[:, t_idx]
            is_target = self.target_tomato_idx == t_idx

            # Only hold idle fruits (attached and not grasped)
            being_grasped = is_target & self._tomato_grasped
            idle = attached & ~being_grasped
            if idle.any():
                hold_ids = idle.nonzero(as_tuple=False).squeeze(-1)
                rest_pos = self.tomato_init_positions[t_idx].unsqueeze(0) + env_origins[hold_ids]

                actual_pos = tomato.data.root_pos_w[hold_ids, :3]
                corrected_pos = actual_pos + HOLD_ALPHA * (rest_pos - actual_pos)

                actual_vel = tomato.data.root_vel_w[hold_ids].clone()
                damped_vel = actual_vel * (1.0 - HOLD_ALPHA)
                damped_vel[:, :3] = cap_speed(damped_vel[:, :3], self.cfg.grasp_linear_speed_cap)
                damped_vel[:, 3:] = cap_speed(damped_vel[:, 3:], self.cfg.grasp_angular_speed_cap)
                stem_quat = self._quat_from_z_axis_torch(self.tomato_anchor_pos[hold_ids, t_idx, :] - corrected_pos)

                pose = torch.cat([corrected_pos, stem_quat], dim=-1)
                visual_pos[hold_ids, t_idx, :] = corrected_pos
                visual_quat[hold_ids, t_idx, :] = stem_quat
                tomato.write_root_pose_to_sim(pose, env_ids=hold_ids)
                tomato.write_root_velocity_to_sim(damped_vel, env_ids=hold_ids)

            # While the fruit is still attached, only damp unrealistic grasp impulses.
            grasped_attached = attached & being_grasped
            if grasped_attached.any():
                grasp_ids = grasped_attached.nonzero(as_tuple=False).squeeze(-1)
                curr_pos = tomato.data.root_pos_w[grasp_ids, :3]
                stem_quat = self._quat_from_z_axis_torch(self.tomato_anchor_pos[grasp_ids, t_idx, :] - curr_pos)
                damped_vel = tomato.data.root_vel_w[grasp_ids].clone()
                damped_vel = damped_vel * (1.0 - self.cfg.grasp_velocity_damping)
                damped_vel[:, :3] = cap_speed(damped_vel[:, :3], self.cfg.grasp_linear_speed_cap)
                damped_vel[:, 3:] = cap_speed(damped_vel[:, 3:], self.cfg.grasp_angular_speed_cap)
                pose = torch.cat([curr_pos, stem_quat], dim=-1)
                visual_pos[grasp_ids, t_idx, :] = curr_pos
                visual_quat[grasp_ids, t_idx, :] = stem_quat
                tomato.write_root_pose_to_sim(pose, env_ids=grasp_ids)
                tomato.write_root_velocity_to_sim(damped_vel, env_ids=grasp_ids)

            # Detached grasped fruit gets a soft post-physics correction toward the fingertip.
            transport = ~attached & being_grasped
            if transport.any():
                transport_ids = transport.nonzero(as_tuple=False).squeeze(-1)
                target_pos = ft_pos[transport_ids]
                actual_pos = tomato.data.root_pos_w[transport_ids, :3]
                corrected_pos = actual_pos + TRANSPORT_ALPHA * (target_pos - actual_pos)

                actual_vel = tomato.data.root_vel_w[transport_ids].clone()
                damped_vel = actual_vel * (1.0 - TRANSPORT_ALPHA)
                damped_vel[:, :3] = cap_speed(damped_vel[:, :3], self.cfg.transport_speed_cap)
                damped_vel[:, 3:] = cap_speed(damped_vel[:, 3:], self.cfg.grasp_angular_speed_cap)
                actual_quat = tomato.data.root_state_w[transport_ids, 3:7]

                pose = torch.cat([corrected_pos, actual_quat], dim=-1)
                visual_pos[transport_ids, t_idx, :] = corrected_pos
                visual_quat[transport_ids, t_idx, :] = actual_quat
                tomato.write_root_pose_to_sim(pose, env_ids=transport_ids)
                tomato.write_root_velocity_to_sim(damped_vel, env_ids=transport_ids)

        self._update_peduncle_visuals(visual_pos, visual_quat)

    def _apply_spring_forces(self):
        """Integrate peduncle damage from sustained fruit-side stem tension.

        Detachment should only happen after the grasped fruit is actually
        pulled away from its branch anchor for long enough. Grasping alone
        must not break the stem.
        """
        env_origins = self.scene.env_origins
        dt = float(self.cfg.sim.dt)

        for t_idx in range(NUM_TOMATOES):
            attached = self.tomato_attached[:, t_idx]

            # All attached peduncles slowly recover when unloaded. Detached
            # fruit has no remaining stem damage to integrate.
            damage = self.peduncle_damage[:, t_idx]
            damage = torch.where(
                attached,
                torch.clamp(damage - self.cfg.spring_recovery_rate * dt, min=0.0),
                torch.zeros_like(damage),
            )
            self.peduncle_damage[:, t_idx] = damage

            if not attached.any():
                continue

            # Only the actively grasped target can accumulate detach damage.
            is_target = self.target_tomato_idx == t_idx
            being_grasped = is_target & self._tomato_grasped & attached
            stable_grasp = being_grasped & (self._grasp_hold_steps >= self.cfg.grasp_settle_steps)
            if not stable_grasp.any():
                continue

            grasped_ids = stable_grasp.nonzero(as_tuple=False).squeeze(-1)
            tomato = self._tomatoes[t_idx]
            curr_pos = tomato.data.root_pos_w[grasped_ids, :3]
            curr_vel = tomato.data.root_vel_w[grasped_ids, :3]

            anchor = self.tomato_anchor_pos[grasped_ids, t_idx, :]
            rest_len = self.peduncle_rest_length[grasped_ids, t_idx]
            rest_axis = self.peduncle_rest_axis[grasped_ids, t_idx, :]
            rest_pos = self.tomato_init_positions[t_idx].unsqueeze(0) + env_origins[grasped_ids]

            stem_vec = curr_pos - anchor
            stem_len = torch.norm(stem_vec, dim=-1).clamp(min=1e-6)
            total_stretch = torch.clamp(stem_len - rest_len, min=0.0)

            disp_from_rest = curr_pos - rest_pos
            axial_stretch = torch.clamp((disp_from_rest * rest_axis).sum(dim=-1), min=0.0)
            axial_vel = torch.clamp((curr_vel * rest_axis).sum(dim=-1), min=0.0)
            tension_force = self.cfg.spring_stiffness * axial_stretch + self.cfg.spring_damping * axial_vel

            active_pull = axial_stretch > self.cfg.spring_min_axial_stretch
            overload_denom = max(self.cfg.spring_break_force - self.cfg.spring_yield_force, 1e-6)
            overload = torch.clamp((tension_force - self.cfg.spring_yield_force) / overload_denom, min=0.0)
            damage_gain = self.cfg.spring_damage_rate * torch.pow(overload, self.cfg.spring_damage_exponent) * dt

            local_damage = self.peduncle_damage[grasped_ids, t_idx]
            local_damage = torch.where(active_pull, local_damage + damage_gain, local_damage)
            self.peduncle_damage[grasped_ids, t_idx] = local_damage

            catastrophic_break = (
                (total_stretch >= self.cfg.spring_break_distance)
                | (tension_force >= self.cfg.spring_break_force * self.cfg.spring_catastrophic_force_ratio)
            )
            should_break_local = active_pull & ((local_damage >= 1.0) | catastrophic_break)

            if should_break_local.any():
                break_ids = grasped_ids[should_break_local]
                self.detach_force[break_ids] = tension_force[should_break_local]
                self.tomato_attached[break_ids, t_idx] = False
                self.peduncle_damage[break_ids, t_idx] = 0.0

    def _estimate_grip_force(self):
        """Estimate grip force from finger joint displacement using Hooke's law."""
        finger_pos = self._robot.data.joint_pos[:, self.finger_joint_ids]
        finger_gap = finger_pos.sum(dim=-1)
        fruit_diameter = 2.0 * TOMATO_RADIUS
        fruit_compression = torch.clamp(fruit_diameter - finger_gap, min=0.0)
        grip_force = torch.clamp(self.cfg.grip_force_k * fruit_compression, min=0.0)

        # Track peak grip force when grasping
        grasping = self._tomato_grasped
        self.peak_grip_force = torch.where(
            grasping & (grip_force > self.peak_grip_force),
            grip_force, self.peak_grip_force
        )
        # Flag damage
        self.tomato_damaged = self.tomato_damaged | (grasping & (grip_force > self.cfg.max_grip_force))

    def _apply_transport_forces(self, ft_pos: torch.Tensor):
        """Apply external forces to tomatoes. Matches reference demo pattern.

        Idle fruits get ZERO external force -- their positions are enforced
        by _post_physics_hold_fruits() via alpha-blend position writes.

        Only active forces:
        - Grasped + detached (transport): spring-damper tracking fingertip
          + gravity comp. Matches reference demo K=100, D=5, MAX_FORCE=2N.
        - Grasped + attached: peduncle anchor spring resisting retraction.
        - All others: zero force.
        """
        for t_idx in range(NUM_TOMATOES):
            tomato = self._tomatoes[t_idx]
            force_buf = torch.zeros(self.num_envs, 1, 3, device=self.device)
            torque_buf = torch.zeros(self.num_envs, 1, 3, device=self.device)
            grav_comp = TOMATO_MASSES[t_idx] * 9.81

            is_target = self.target_tomato_idx == t_idx
            attached = self.tomato_attached[:, t_idx]

            # --- Grasped + detached: transport spring toward fingertip ---
            transport = is_target & self._tomato_grasped & ~attached
            if transport.any():
                t_ids = transport.nonzero(as_tuple=False).squeeze(-1)
                f_pos = tomato.data.root_pos_w[t_ids, :3]
                f_vel = tomato.data.root_vel_w[t_ids, :3]
                disp = ft_pos[t_ids] - f_pos
                dist_mag = disp.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                tracking_scale = torch.clamp(self.cfg.transport_cutoff / dist_mag, max=1.0)
                spring = self.cfg.transport_spring_k * disp * tracking_scale
                damper = -self.cfg.transport_damping * f_vel
                total = spring + damper
                total[:, 2] += grav_comp
                mag = total.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                total = total * torch.clamp(self.cfg.transport_max_force / mag, max=1.0)
                force_buf[t_ids, 0, :] = total

            # --- Grasped + attached: peduncle tension from the source anchor ---
            grasped_attached = is_target & self._tomato_grasped & attached
            if grasped_attached.any():
                ga_ids = grasped_attached.nonzero(as_tuple=False).squeeze(-1)
                curr_pos = tomato.data.root_pos_w[ga_ids, :3]
                curr_vel = tomato.data.root_vel_w[ga_ids, :3]
                anchor = self.tomato_anchor_pos[ga_ids, t_idx, :]
                rest_len = self.peduncle_rest_length[ga_ids, t_idx].unsqueeze(-1)
                anchor_to_fruit = curr_pos - anchor
                curr_len = anchor_to_fruit.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                direction = anchor_to_fruit / curr_len
                stretch = torch.clamp(curr_len - rest_len, min=0.0)
                ped_force = -self.cfg.spring_stiffness * stretch * direction
                ped_force = ped_force - self.cfg.spring_damping * curr_vel
                mag = ped_force.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                ped_force = ped_force * torch.clamp(
                    self.cfg.spring_break_force / mag, max=1.0
                )
                ped_force[:, 2] += grav_comp
                force_buf[ga_ids, 0, :] = ped_force

            # Idle attached: zero force. Held by position writes in
            # _post_physics_hold_fruits() (alpha-blend, matching reference demo).

            tomato.set_external_force_and_torque(force_buf, torque_buf, is_global=True)

    def _update_state_machine(self, ee_pos, target_pos, dist, gripper_closed):
        """Vectorized state machine transitions across all envs."""
        target_idx = self.target_tomato_idx  # (N,)
        # Gather attachment state for target tomato
        target_attached = self.tomato_attached[
            torch.arange(self.num_envs, device=self.device), target_idx
        ]

        # APPROACH(0) -> GRASP(1): EE near target (gripper may still be open)
        to_grasp = (self.sub_task_phase == PHASE_APPROACH) & (dist < self.cfg.grasp_thresh)
        self.sub_task_phase[to_grasp] = PHASE_GRASP

        # GRASP(1) -> DETACH(2): target tomato detached from vine
        to_detach = (self.sub_task_phase == PHASE_GRASP) & ~target_attached
        self.sub_task_phase[to_detach] = PHASE_DETACH

        # DETACH(2) -> TRANSPORT(3): immediate (tomato still grasped after detach)
        to_transport = (self.sub_task_phase == PHASE_DETACH) & self._tomato_grasped
        self.sub_task_phase[to_transport] = PHASE_TRANSPORT

        # TRANSPORT(3) -> PLACE(4): tomato inside basket
        basket_w = self._get_basket_centers_w()  # (N, 3)
        in_basket_xy = (
            (torch.abs(target_pos[:, 0] - basket_w[:, 0]) < self.cfg.basket_inner / 2.0)
            & (torch.abs(target_pos[:, 1] - basket_w[:, 1]) < self.cfg.basket_inner / 2.0)
        )
        env_basket_z = self.scene.env_origins[:, 2] + self.cfg.basket_bottom_z
        in_basket_z = (
            (target_pos[:, 2] > env_basket_z)
            & (target_pos[:, 2] < env_basket_z + self.cfg.basket_wall_h + 0.05)
        )
        in_basket = in_basket_xy & in_basket_z
        to_place = (self.sub_task_phase == PHASE_TRANSPORT) & in_basket & ~self._tomato_grasped
        self.sub_task_phase[to_place] = PHASE_PLACE
        self.tomato_harvested[to_place] = True

    # ------------------------------------------------------------------
    # Observations (36D)
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        joint_pos = self._robot.data.joint_pos

        dof_pos_scaled = (
            2.0 * (joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )

        ft_pos_w = self._fingertip_world()
        ee_quat_w = self._robot.data.body_pose_w[:, self.hand_body_idx, 3:7]

        target_tomato_pos = self._get_target_tomato_positions()
        fruit_to_ft = target_tomato_pos - ft_pos_w
        dist_to_target = torch.norm(fruit_to_ft, dim=-1, keepdim=True)

        finger_state = joint_pos[:, self.finger_joint_ids] / max(self.cfg.gripper_open_value, 1e-6)

        grasped_state = self._tomato_grasped.unsqueeze(-1).float()
        harvested_state = self.tomato_harvested.unsqueeze(-1).float()

        # Sub-task phase one-hot (5 phases)
        phase_onehot = torch.zeros(self.num_envs, NUM_PHASES, device=self.device)
        phase_onehot.scatter_(1, self.sub_task_phase.unsqueeze(-1), 1.0)

        # Target attached state
        target_attached = self.tomato_attached[
            torch.arange(self.num_envs, device=self.device), self.target_tomato_idx
        ].unsqueeze(-1).float()

        # Basket direction (fingertip -> basket center, in world coords)
        basket_direction = self._get_basket_centers_w() - ft_pos_w

        obs_parts = [
            dof_pos_scaled,       # 9  (7 arm + 2 finger)
            ft_pos_w,             # 3  (fingertip position)
            ee_quat_w,            # 4
            target_tomato_pos,    # 3
            fruit_to_ft,          # 3
            dist_to_target,       # 1
            finger_state,         # 2
            grasped_state,        # 1
            harvested_state,      # 1
            phase_onehot,         # 5
            target_attached,      # 1
            basket_direction,     # 3
        ]                         # Total: 36
        obs = torch.cat(obs_parts, dim=-1)

        # Pad/truncate to observation_space
        obs_dim = obs.shape[-1]
        if obs_dim < self.cfg.observation_space:
            pad = torch.zeros(self.num_envs, self.cfg.observation_space - obs_dim, device=self.device)
            obs = torch.cat([obs, pad], dim=-1)
        elif obs_dim > self.cfg.observation_space:
            obs = obs[:, :self.cfg.observation_space]

        obs_dict: dict = {"policy": torch.clamp(obs, -5.0, 5.0)}
        obs_dict["raw_joint_pos"] = self._robot.data.joint_pos[:, self.arm_joint_ids]
        obs_dict["raw_gripper_pos"] = self._robot.data.joint_pos[:, self.finger_joint_ids]

        if "rgb" in self._wrist_camera.data.output:
            obs_dict["wrist_rgb"] = self._wrist_camera.data.output["rgb"]
        if "rgb" in self._scene_camera.data.output:
            obs_dict["scene_rgb"] = self._scene_camera.data.output["rgb"]

        return obs_dict

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        ft_pos_w = self._fingertip_world()
        target_pos = self._get_target_tomato_positions()

        dist = torch.norm(ft_pos_w - target_pos, dim=-1)
        dist_to_basket = torch.norm(target_pos - self._get_basket_centers_w(), dim=-1)

        # Dense approach reward: 1/(1+d^2)^2
        approach_reward = 1.0 / (1.0 + dist ** 2)
        approach_reward = approach_reward * approach_reward

        # Sparse grasp reward
        grasp_reward = (self.sub_task_phase >= PHASE_GRASP).float()

        # Sparse detach reward (BIG)
        detach_reward = (self.sub_task_phase >= PHASE_DETACH).float()

        # Dense transport reward (distance to basket, only when transporting)
        transport_shaping = 1.0 / (1.0 + dist_to_basket ** 2)
        transport_shaping = transport_shaping * transport_shaping
        transport_reward = transport_shaping * (self.sub_task_phase >= PHASE_TRANSPORT).float()

        # Sparse place reward (BIGGEST)
        place_reward = self.tomato_harvested.float()

        # Damage penalty
        damage_penalty = self.tomato_damaged.float()

        # Action penalty (smooth motion)
        action_penalty = torch.sum(self.actions ** 2, dim=-1)

        # Time penalty (efficiency)
        time_penalty = self._episode_steps.float() / self.max_episode_length

        rewards = (
            self.cfg.approach_reward_scale * approach_reward
            + self.cfg.grasp_reward_scale * grasp_reward
            + self.cfg.detach_reward_scale * detach_reward
            + self.cfg.transport_reward_scale * transport_reward
            + self.cfg.place_reward_scale * place_reward
            - self.cfg.damage_penalty_scale * damage_penalty
            - self.cfg.action_penalty_scale * action_penalty
            - self.cfg.time_penalty_scale * time_penalty
        )

        # -- Logged metrics --
        self.extras["log"] = {
            "harvest_success_rate": self.tomato_harvested.float().mean(),
            "grasp_rate": self._tomato_grasped.float().mean(),
            "detach_rate": (self.sub_task_phase >= PHASE_DETACH).float().mean(),
            "damage_rate": self.tomato_damaged.float().mean(),
            "mean_sub_task_phase": self.sub_task_phase.float().mean(),
            "mean_dist_to_tomato": dist.mean(),
            "mean_dist_to_basket": dist_to_basket.mean(),
            "peak_grip_force": self.peak_grip_force.mean(),
            "mean_detach_force": self.detach_force.mean(),
            "approach_reward": (self.cfg.approach_reward_scale * approach_reward).mean(),
            "grasp_reward": (self.cfg.grasp_reward_scale * grasp_reward).mean(),
            "detach_reward": (self.cfg.detach_reward_scale * detach_reward).mean(),
            "transport_reward": (self.cfg.transport_reward_scale * transport_reward).mean(),
            "place_reward": (self.cfg.place_reward_scale * place_reward).mean(),
            "damage_penalty": (-self.cfg.damage_penalty_scale * damage_penalty).mean(),
            "action_penalty": (-self.cfg.action_penalty_scale * action_penalty).mean(),
        }

        return rewards

    # ------------------------------------------------------------------
    # Terminations
    # ------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = self.tomato_harvested.clone()
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        N = len(env_ids)

        # Randomize target tomato (0..5)
        self.target_tomato_idx[env_ids] = torch.randint(0, NUM_TOMATOES, (N,), device=self.device)

        # Reset sub-task state
        self.sub_task_phase[env_ids] = PHASE_APPROACH
        self._tomato_grasped[env_ids] = False
        self.tomato_harvested[env_ids] = False
        self.tomato_attached[env_ids] = True
        self._first_step_after_reset[env_ids] = True

        # Reset anchor positions in WORLD coordinates.
        # Use branch attachment points (peduncle origin) if available,
        # otherwise fall back to tomato positions.
        env_origins = self.scene.env_origins[env_ids]  # (N, 3)
        noise_mag = self.cfg.peduncle_anchor_noise
        noise = sample_uniform(-noise_mag, noise_mag, (N, NUM_TOMATOES, 3), self.device)
        if hasattr(self, "_peduncle_attach_points") and len(self._peduncle_attach_points) == NUM_TOMATOES:
            attach_local = torch.tensor(
                self._peduncle_attach_points, device=self.device, dtype=torch.float32
            )  # (NUM_TOMATOES, 3)
        else:
            attach_local = self.tomato_init_positions
        anchors = (
            attach_local.unsqueeze(0).expand(N, -1, -1)
            + env_origins.unsqueeze(1)
            + noise
        )
        self.tomato_anchor_pos[env_ids] = anchors

        # Peduncle rest pose: natural stem vector (anchor → fruit rest position).
        fruit_rest_w = (
            self.tomato_init_positions.unsqueeze(0).expand(N, -1, -1)
            + env_origins.unsqueeze(1)
        )  # (N, NUM_TOMATOES, 3)
        rest_vec = fruit_rest_w - anchors
        rest_len = torch.norm(rest_vec, dim=-1).clamp(min=1e-6)
        self.peduncle_rest_length[env_ids] = rest_len
        self.peduncle_rest_axis[env_ids] = rest_vec / rest_len.unsqueeze(-1)
        self.peduncle_damage[env_ids] = 0.0

        # Reset metrics
        self.peak_grip_force[env_ids] = 0.0
        self.detach_force[env_ids] = 0.0
        self.tomato_damaged[env_ids] = False
        self._grasp_hold_steps[env_ids] = 0
        self._episode_steps[env_ids] = 0

        self._update_peduncle_visuals()

        # Reset robot joints with small noise
        joint_pos = self._robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.125, 0.125, (N, self._robot.num_joints), self.device,
        )
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot_dof_targets[env_ids] = joint_pos

        # Reset all tomato states to their spawn (rest) positions
        rest_noise = sample_uniform(-0.003, 0.003, (N, NUM_TOMATOES, 3), self.device)
        rest_positions = (
            self.tomato_init_positions.unsqueeze(0).expand(N, -1, -1)
            + env_origins.unsqueeze(1)
            + rest_noise
        )
        for t_idx, tomato in enumerate(self._tomatoes):
            default_state = tomato.data.default_root_state[env_ids].clone()
            default_state[:, :3] = rest_positions[:, t_idx, :]
            default_state[:, 7:] = 0.0  # zero velocity
            tomato.write_root_state_to_sim(default_state, env_ids=env_ids)

        self.diff_ik_controller.reset(env_ids)
        self._ik_substep_count = 0

        # Diagnostic: print setup for env 0
        if 0 in env_ids:
            rp = self._robot.data.root_pose_w[0, :3]
            rq = self._robot.data.root_pose_w[0, 3:7]
            ti = int(self.target_tomato_idx[0].item())
            tp = self.tomato_init_positions[ti]
            reach_est = tp.norm().item()  # rough estimate (local frame ≈ base frame offset)
            print(f"[RESET] env0: robot=({rp[0]:.3f},{rp[1]:.3f},{rp[2]:.3f}) "
                  f"rot=({rq[0]:.3f},{rq[1]:.3f},{rq[2]:.3f},{rq[3]:.3f}) "
                  f"target_tomato={ti} local_pos=({tp[0]:.4f},{tp[1]:.4f},{tp[2]:.4f})")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _fingertip_world(self) -> torch.Tensor:
        """Compute fingertip center position in world frame (EE + body offset)."""
        ee_pose = self._robot.data.body_pose_w[:, self.hand_body_idx]
        R = matrix_from_quat(ee_pose[:, 3:7])
        offset = self._body_offset_pos.unsqueeze(-1)  # (N, 3, 1)
        return ee_pose[:, :3] + torch.bmm(R, offset).squeeze(-1)

    def _get_basket_centers_w(self) -> torch.Tensor:
        """Return (num_envs, 3) world-space basket centers (lazily computed)."""
        if self._basket_center_w is None:
            self._basket_center_w = self.scene.env_origins + self._basket_center_local.unsqueeze(0)
        return self._basket_center_w

    def _get_target_tomato_positions(self) -> torch.Tensor:
        """Return (num_envs, 3) world positions of each env's targeted tomato."""
        all_pos = torch.stack([t.data.root_pos_w for t in self._tomatoes], dim=0)  # (6, N, 3)
        idx = self.target_tomato_idx.unsqueeze(0).unsqueeze(-1).expand(1, -1, 3)
        target_pos = torch.gather(all_pos, dim=0, index=idx).squeeze(0)
        return target_pos

    def _get_target_tomato_linear_velocities(self) -> torch.Tensor:
        """Return (num_envs, 3) world linear velocities of each env's target fruit."""
        all_vel = torch.stack([t.data.root_vel_w[:, :3] for t in self._tomatoes], dim=0)  # (6, N, 3)
        idx = self.target_tomato_idx.unsqueeze(0).unsqueeze(-1).expand(1, -1, 3)
        target_vel = torch.gather(all_vel, dim=0, index=idx).squeeze(0)
        return target_vel
