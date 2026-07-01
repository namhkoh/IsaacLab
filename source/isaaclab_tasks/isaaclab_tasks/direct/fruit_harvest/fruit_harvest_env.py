# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json
import math
import os

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
from isaaclab.utils.math import matrix_from_quat, quat_inv, sample_uniform, subtract_frame_transforms


##
# Environment configuration
##

NUM_FRUITS = 8

# Fruit positions (already include table height offset of 0.7m)
FRUIT_POSITIONS: list[tuple[float, float, float]] = [
    (0.06488, -0.11387, 1.41291),
    (0.05581, -0.08732, 1.41244),
    (0.08848, -0.10604, 1.40838),
    (0.07937, -0.07918, 1.40886),
    (0.11194, -0.09819, 1.40172),
    (0.10269, -0.07105, 1.40172),
    (0.13503, -0.09047, 1.39083),
    (0.12568, -0.06305, 1.39083),
]


def _make_fruit_cfg(index: int) -> RigidObjectCfg:
    """Create a RigidObjectCfg for a single fruit sphere (kinematic — stays in place)."""
    pos = FRUIT_POSITIONS[index]
    return RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/Fruit{index}",
        spawn=sim_utils.SphereCfg(
            radius=0.0215,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
    )


@configclass
class FruitHarvestEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 8.3333  # 500 timesteps
    decimation = 2
    action_space = 7  # delta_pos(3) + delta_orient(3) + gripper(1)
    observation_space = 27
    state_space = 0
    control_mode: str = "ik"  # "ik" for EE delta (LIBERO), "joint_vel" for joint velocity (DROID)

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
    gripper_close_value: float = 0.0
    enable_greenhouse: bool = False

    # -- Harvest basket --
    basket_center: tuple[float, float] = (0.45, 0.15)
    basket_bottom_z: float = 0.80
    basket_wall_h: float = 0.15
    basket_inner: float = 0.13
    basket_thick: float = 0.008

    # -- Cosserat rod plant visualization --
    # Pre-generated watertight USD mesh (best visual quality, single prim).
    # Generated from fruit-001.json via generate_plant_mesh.py + marching cubes.
    plant_usd_path: str | None = "D:/research/IsaacLab/scripts/demos/fruit_harvest/usd/plant_branches.usd"
    # Fallback: build from raw Cosserat rod JSON (cylinder segments per edge).
    plant_json_path: str | None = "D:/research/Plant_Robot_Interaction-master/data/plant/fruit-001.json"
    plant_offset: tuple[float, float, float] = (0.0, 0.0, 0.7)  # table height offset
    plant_rod_scale: float = 3.0  # visual scale for thin rod radii

    # robot — Franka Panda with high PD gains for IK tracking
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
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
            pos=(0.6, 0.0, 0.7),  # elevated to table height
            rot=(0.0, 0.0, 0.0, 1.0),  # 180° around Z to face the plant
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
                effort_limit_sim=200.0,
                stiffness=2e3,
                damping=1e2,
            ),
        },
    )

    # fruits
    fruit_cfgs: list[RigidObjectCfg] = [_make_fruit_cfg(i) for i in range(NUM_FRUITS)]

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

    # IK controller scale for delta commands
    ik_command_scale = 0.05

    # reward scales
    dist_reward_scale = 1.0
    grasp_reward_scale = 1.0
    harvest_reward_scale = 10.0
    action_penalty_scale = 0.01

    # harvest threshold — fruit must be lifted this far above its initial z to count
    harvest_height_thresh = 0.1

    # grasp proximity threshold (metres)
    grasp_thresh = 0.04

    # fruit positions (for reset convenience)
    fruit_positions: list[tuple[float, float, float]] = list(FRUIT_POSITIONS)

    # -- Cameras for VLA training --
    # Wrist camera (eye-in-hand) — rigidly mounted on panda_hand.
    # Offset and rotation from Isaac Lab visuomotor stack example.
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

    # Third-person scene camera — fixed view of the workspace.
    # Positioned to see robot, plant, and fruit cluster.
    # Orientation is set at runtime via set_world_poses.
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


@configclass
class FrankaGreenhouseFruitHarvestEnvCfg(FruitHarvestEnvCfg):
    """Franka Panda in a greenhouse scene."""
    enable_greenhouse: bool = True


@configclass
class UR5eFruitHarvestEnvCfg(FruitHarvestEnvCfg):
    """UR5e + UMI gripper in a greenhouse scene."""
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="D:/research/ddai_sim/Assets/manipulator/Universal_Robots/UMI.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
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
                "shoulder_pan_joint": 0.0,
                "shoulder_lift_joint": -1.5708,
                "elbow_joint": 1.5708,
                "wrist_1_joint": -1.5708,
                "wrist_2_joint": -1.5708,
                "wrist_3_joint": 0.0,
                "PrismaticJoint_Right": 0.0,
                "PrismaticJoint_Left": 0.0,
            },
            pos=(0.6, 0.0, 0.7),
        ),
        actuators={
            "ur5e_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint"],
                effort_limit_sim=150.0,
                stiffness=400.0,
                damping=80.0,
            ),
            "ur5e_wrist": ImplicitActuatorCfg(
                joint_names_expr=["wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
                effort_limit_sim=28.0,
                stiffness=400.0,
                damping=80.0,
            ),
            "umi_gripper": ImplicitActuatorCfg(
                joint_names_expr=["PrismaticJoint_.*"],
                effort_limit_sim=200.0,
                stiffness=2e3,
                damping=1e2,
            ),
        },
    )
    arm_joint_names_expr: list[str] = [
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
    ]
    finger_joint_names_expr: list[str] = ["PrismaticJoint_.*"]
    ee_body_name: str = "tool0"
    gripper_open_value: float = 0.04
    gripper_close_value: float = 0.0
    enable_greenhouse: bool = True


##
# Environment implementation
##


class FruitHarvestEnv(DirectRLEnv):
    """Fruit harvesting environment with a Franka Panda robot.

    The robot must reach toward fruit spheres attached to a plant,
    grasp them, and lift them above a height threshold to "harvest" them.

    Actions: delta EE pose (6D) + gripper open/close (1D) = 7D
    Observations: joint state + EE pose + fruit target info = 27D
    """

    cfg: FruitHarvestEnvCfg

    def __init__(self, cfg: FruitHarvestEnvCfg, render_mode: str | None = None, **kwargs):
        if cfg.control_mode == "joint_vel":
            cfg.action_space = 8  # 7 joint velocities + 1 gripper
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # -------------------------------------------------------------------
        # Robot joint / body indices (robot-agnostic via config)
        # -------------------------------------------------------------------
        self.arm_joint_ids = self._robot.find_joints(self.cfg.arm_joint_names_expr)[0]
        self.finger_joint_ids = self._robot.find_joints(self.cfg.finger_joint_names_expr)[0]

        # End-effector body index
        self.hand_body_idx = self._robot.find_bodies(self.cfg.ee_body_name)[0][0]
        # For fixed-base robots the jacobian body index is shifted by -1
        self.ee_jacobi_idx = self.hand_body_idx - 1

        # Joint limits
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        # -------------------------------------------------------------------
        # Differential IK controller
        # -------------------------------------------------------------------
        diff_ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method="dls",
            ik_params={"lambda_val": 0.05},
        )
        self.diff_ik_controller = DifferentialIKController(
            diff_ik_cfg, num_envs=self.num_envs, device=self.device
        )

        # -------------------------------------------------------------------
        # Buffers
        # -------------------------------------------------------------------
        self.target_fruit_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.fruit_harvested = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.gripper_open = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        # Initial fruit positions as tensor (NUM_FRUITS, 3)
        self.fruit_init_positions = torch.tensor(self.cfg.fruit_positions, device=self.device, dtype=torch.float32)
        # Skip IK on first step after reset (jacobians not yet valid)
        self._first_step_after_reset = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        # Grasp tracking: whether the target fruit is currently grasped
        self._fruit_grasped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        # Spawn fruit rigid bodies
        self._fruits: list[RigidObject] = []
        for i, fruit_cfg in enumerate(self.cfg.fruit_cfgs):
            fruit = RigidObject(fruit_cfg)
            self._fruits.append(fruit)
            self.scene.rigid_objects[f"fruit{i}"] = fruit

        # Greenhouse geometry (if enabled)
        if self.cfg.enable_greenhouse:
            self._spawn_greenhouse()

        # Cameras — spawned BEFORE clone so prims get cloned to all envs
        self._wrist_camera = Camera(self.cfg.wrist_camera)
        self.scene.sensors["wrist_camera"] = self._wrist_camera
        self._scene_camera = Camera(self.cfg.scene_camera)
        self.scene.sensors["scene_camera"] = self._scene_camera

        # Ground plane
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # --- Post-clone: world-space geometry (lights, tables, plant) ---
        light_cfg = sim_utils.DomeLightCfg(intensity=5000.0, color=(0.8, 0.8, 0.8))
        light_cfg.func("/World/Light", light_cfg)

        # Tables / pedestals (visual only — match demo scene layout)
        table_cfg = sim_utils.CuboidCfg(
            size=(0.35, 0.35, 0.7),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.30, 0.15)),
        )
        table_cfg.func("/World/Tables/PlantTable", table_cfg, translation=(0.0, 0.0, 0.35))
        pedestal_cfg = sim_utils.CuboidCfg(
            size=(0.22, 0.28, 0.7),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.30, 0.15)),
        )
        pedestal_cfg.func("/World/Tables/RobotStand", pedestal_cfg, translation=(0.6, 0.0, 0.35))

        # Cosserat rod plant — created in world space (/World/Plant/)
        # Uses the same build_plant_with_physics() as the demo scene
        self._spawn_plant_visual()

        # Harvest basket
        self._spawn_basket()

        # Camera poses are set in _apply_action() (sensors not initialized until sim.reset())
        self._scene_cam_initialized = False

    # ------------------------------------------------------------------
    # Plant visual geometry
    # ------------------------------------------------------------------
    def _spawn_plant_visual(self):
        """Spawn plant visual geometry — physics plant > JSON cylinders > simple fallback."""
        if self.cfg.plant_json_path and os.path.isfile(self.cfg.plant_json_path):
            self._spawn_cosserat_plant_physics()
        else:
            self._spawn_simple_plant_visual()

    def _spawn_cosserat_plant_physics(self):
        """Spawn the full Cosserat rod plant using build_plant_with_physics().

        Uses the exact same function as the demo scene (fruit_harvest_scene.py)
        to create articulated capsule geometry with collision and green material.
        Prims are created under /World/Plant/ (world space, not per-env).
        """
        import sys
        plant_physics_dir = os.path.normpath(os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "..", "..", "scripts", "demos", "fruit_harvest"
        ))
        if plant_physics_dir not in sys.path:
            sys.path.insert(0, plant_physics_dir)

        try:
            from plant_physics import build_plant_with_physics
        except ImportError:
            print(f"[FruitHarvestEnv] Could not import plant_physics from {plant_physics_dir}")
            print("[FruitHarvestEnv] Falling back to cylinder-based plant.")
            self._spawn_cosserat_plant()
            return

        table_height = self.cfg.plant_offset[2]  # 0.7m
        stage = sim_utils.SimulationContext.instance().stage
        build_plant_with_physics(stage, self.cfg.plant_json_path, table_height)
        print("[FruitHarvestEnv] Spawned Cosserat rod plant via build_plant_with_physics()")

    def _spawn_simple_plant_visual(self):
        """Fallback: simple brown stem + green canopy sphere."""
        fp = self.cfg.fruit_positions
        cx = sum(p[0] for p in fp) / len(fp)
        cy = sum(p[1] for p in fp) / len(fp)
        fruit_z_min = min(p[2] for p in fp)
        fruit_z_max = max(p[2] for p in fp)
        table_z = 0.7
        stem_height = fruit_z_min - table_z - 0.02
        stem_center_z = table_z + stem_height / 2.0

        stem_cfg = sim_utils.CuboidCfg(
            size=(0.02, 0.02, stem_height),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.40, 0.26, 0.13)),
        )
        stem_cfg.func("/World/envs/env_0/Plant/Stem", stem_cfg, translation=(cx, cy, stem_center_z))

        canopy_cfg = sim_utils.SphereCfg(
            radius=0.12,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.55, 0.15), opacity=0.45),
        )
        canopy_cfg.func(
            "/World/envs/env_0/Plant/Canopy", canopy_cfg,
            translation=(cx, cy, (fruit_z_min + fruit_z_max) / 2.0),
        )

    def _spawn_cosserat_plant(self):
        """Spawn Cosserat rod plant geometry from Plant_Robot_Interaction JSON data.

        Creates cylinder segments for each rod edge in the tree topology,
        with color gradient from brown (trunk) to green (canopy).
        Branch points get small spheres for visual smoothness.
        """
        with open(self.cfg.plant_json_path) as f:
            plant = json.load(f)

        positions = np.array(plant["positions"], dtype=np.float64)
        fathers = plant["fathers"]
        radii_arr = np.array(plant["radii"], dtype=np.float64)
        fruit_radii_arr = np.array(plant["fruit_radii"], dtype=np.float64)

        offset = np.array(self.cfg.plant_offset, dtype=np.float64)
        positions = positions + offset

        z_min = float(positions[:, 2].min())
        z_range = max(float(positions[:, 2].max()) - z_min, 0.01)

        # Count children per node (for branch point detection)
        child_count: dict[int, int] = {}
        for fi in fathers:
            if fi >= 0:
                child_count[fi] = child_count.get(fi, 0) + 1

        seg_count = 0
        for i in range(len(positions)):
            fi = fathers[i]
            if fi == -1:
                continue  # root node has no parent segment
            if fruit_radii_arr[i] > 0:
                continue  # fruit nodes handled by RigidObjectCfg

            p1 = positions[fi]
            p2 = positions[i]
            direction = p2 - p1
            length = float(np.linalg.norm(direction))
            if length < 1e-6:
                continue

            midpoint = (p1 + p2) / 2.0
            rod_radius = float(max(radii_arr[i] * self.cfg.plant_rod_scale, 0.002))

            # Color gradient: brown trunk → green canopy
            z_frac = (p2[2] - z_min) / z_range
            color = self._plant_color_by_height(z_frac)

            # Quaternion to orient Z-axis cylinder along segment direction
            quat = self._direction_to_quat_z(direction)

            cyl_cfg = sim_utils.CylinderCfg(
                radius=rod_radius,
                height=length,
                axis="Z",
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )
            cyl_cfg.func(
                f"/World/envs/env_0/CosseratPlant/Seg{seg_count}",
                cyl_cfg,
                translation=tuple(midpoint.tolist()),
                orientation=quat,
            )
            seg_count += 1

        # Add small spheres at branch points for smooth visual joints
        joint_count = 0
        for node_idx, count in child_count.items():
            if count > 1 and fruit_radii_arr[node_idx] == 0:
                pos = positions[node_idx]
                r = float(max(radii_arr[node_idx] * self.cfg.plant_rod_scale * 1.3, 0.003))
                z_frac = (pos[2] - z_min) / z_range
                color = self._plant_color_by_height(z_frac)

                sphere_cfg = sim_utils.SphereCfg(
                    radius=r,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                )
                sphere_cfg.func(
                    f"/World/envs/env_0/CosseratPlant/Joint{joint_count}",
                    sphere_cfg,
                    translation=tuple(pos.tolist()),
                )
                joint_count += 1

        print(f"[FruitHarvestEnv] Spawned Cosserat rod plant: {seg_count} segments, {joint_count} branch joints")

    @staticmethod
    def _look_at_quat(eye: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute quaternion (w,x,y,z) for a camera at *eye* looking at *target*.

        Uses OpenGL convention: camera looks along its local -Z axis, Y is up.
        Args:
            eye: (N, 3) camera positions.
            target: (N, 3) look-at positions.
        Returns:
            (N, 4) quaternions in (w, x, y, z) order.
        """
        forward = target - eye
        forward = forward / forward.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        world_up = torch.tensor([[0.0, 0.0, 1.0]], device=eye.device).expand_as(forward)

        # Camera -Z = forward, so camera Z = -forward
        cam_z = -forward
        right = torch.cross(world_up, cam_z, dim=-1)
        right_norm = right.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        right = right / right_norm
        cam_y = torch.cross(cam_z, right, dim=-1)

        # Build rotation matrix [right, cam_y, cam_z] as columns -> (N, 3, 3)
        R = torch.stack([right, cam_y, cam_z], dim=-1)  # (N, 3, 3)

        # Robust rotation matrix -> quaternion (Shepperd's method)
        # Handles all cases including negative trace
        batch = R.shape[0]
        quat = torch.zeros(batch, 4, device=eye.device)
        for i in range(batch):
            m = R[i]
            tr = m[0, 0] + m[1, 1] + m[2, 2]
            if tr > 0:
                s = torch.sqrt(tr + 1.0) * 2.0
                quat[i, 0] = 0.25 * s
                quat[i, 1] = (m[2, 1] - m[1, 2]) / s
                quat[i, 2] = (m[0, 2] - m[2, 0]) / s
                quat[i, 3] = (m[1, 0] - m[0, 1]) / s
            elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
                s = torch.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
                quat[i, 0] = (m[2, 1] - m[1, 2]) / s
                quat[i, 1] = 0.25 * s
                quat[i, 2] = (m[0, 1] + m[1, 0]) / s
                quat[i, 3] = (m[0, 2] + m[2, 0]) / s
            elif m[1, 1] > m[2, 2]:
                s = torch.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
                quat[i, 0] = (m[0, 2] - m[2, 0]) / s
                quat[i, 1] = (m[0, 1] + m[1, 0]) / s
                quat[i, 2] = 0.25 * s
                quat[i, 3] = (m[1, 2] + m[2, 1]) / s
            else:
                s = torch.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
                quat[i, 0] = (m[1, 0] - m[0, 1]) / s
                quat[i, 1] = (m[0, 2] + m[2, 0]) / s
                quat[i, 2] = (m[1, 2] + m[2, 1]) / s
                quat[i, 3] = 0.25 * s

        quat = quat / quat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return quat

    @staticmethod
    def _plant_color_by_height(z_frac: float) -> tuple[float, float, float]:
        """Return (R,G,B) color for a plant segment based on normalized height [0,1]."""
        if z_frac < 0.25:
            return (0.40, 0.26, 0.13)  # dark brown trunk
        elif z_frac < 0.50:
            t = (z_frac - 0.25) / 0.25
            return (0.40 - 0.20 * t, 0.26 + 0.24 * t, 0.13)  # brown → green
        else:
            return (0.18, 0.52, 0.18)  # green canopy

    @staticmethod
    def _direction_to_quat_z(direction) -> tuple[float, float, float, float]:
        """Compute quaternion (w,x,y,z) to rotate the Z-axis to align with direction.

        CylinderCfg with axis="Z" creates a cylinder along the Z-axis.
        This returns the quaternion needed to orient that cylinder along ``direction``.
        """
        d = np.array(direction, dtype=np.float64)
        length = np.linalg.norm(d)
        if length < 1e-12:
            return (1.0, 0.0, 0.0, 0.0)
        d = d / length
        z_axis = np.array([0.0, 0.0, 1.0])

        dot = float(np.dot(z_axis, d))
        if dot > 0.99999:  # already aligned
            return (1.0, 0.0, 0.0, 0.0)
        if dot < -0.99999:  # anti-parallel — rotate 180° around X
            return (0.0, 1.0, 0.0, 0.0)

        axis = np.cross(z_axis, d)
        axis = axis / np.linalg.norm(axis)
        angle = math.acos(max(-1.0, min(1.0, dot)))

        w = math.cos(angle / 2.0)
        s = math.sin(angle / 2.0)
        return (w, float(axis[0] * s), float(axis[1] * s), float(axis[2] * s))

    # ------------------------------------------------------------------
    # Harvest basket
    # ------------------------------------------------------------------
    def _spawn_basket(self):
        """Spawn a wicker-brown harvest basket with floor + 4 walls (with collision)."""
        bx, by = self.cfg.basket_center
        bz = self.cfg.basket_bottom_z
        wall_h = self.cfg.basket_wall_h
        inner = self.cfg.basket_inner
        thick = self.cfg.basket_thick
        outer = inner + 2 * thick
        wall_z = bz + thick + wall_h / 2.0
        half_off = inner / 2.0 + thick / 2.0

        basket_color = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.5))

        # Stand / pedestal under the basket
        stand_cfg = sim_utils.CuboidCfg(
            size=(outer + 0.02, outer + 0.02, bz),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.30, 0.15)),
        )
        stand_cfg.func("/World/Basket/Stand", stand_cfg, translation=(bx, by, bz / 2.0))

        # Floor + 4 walls
        pieces = [
            ("Floor",     (bx, by, bz + thick / 2.0),       (outer, outer, thick)),
            ("WallFront", (bx, by + half_off, wall_z),       (outer, thick, wall_h)),
            ("WallBack",  (bx, by - half_off, wall_z),       (outer, thick, wall_h)),
            ("WallLeft",  (bx - half_off, by, wall_z),       (thick, inner, wall_h)),
            ("WallRight", (bx + half_off, by, wall_z),       (thick, inner, wall_h)),
        ]
        for name, pos, size in pieces:
            cfg = sim_utils.CuboidCfg(
                size=size,
                visual_material=basket_color,
                collision_props=sim_utils.CollisionPropertiesCfg(),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            )
            cfg.func(f"/World/Basket/{name}", cfg, translation=pos)

        print(f"[FruitHarvestEnv] Basket spawned at ({bx}, {by}, {bz})")

    # ------------------------------------------------------------------
    # Greenhouse scene geometry
    # ------------------------------------------------------------------
    def _spawn_greenhouse(self):
        """Spawn greenhouse walls, floor, and plant table as visual/collision geometry."""
        # Earthy brown floor (4x4m)
        floor_cfg = sim_utils.CuboidCfg(
            size=(4.0, 4.0, 0.02),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.32, 0.18)),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        )
        floor_cfg.func("/World/envs/env_.*/Greenhouse/Floor", floor_cfg, translation=(0.0, 0.0, -0.01))

        # Back wall (translucent green, 2m high)
        wall_cfg = sim_utils.CuboidCfg(
            size=(4.0, 0.05, 2.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.6, 0.3), opacity=0.4),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        )
        wall_cfg.func("/World/envs/env_.*/Greenhouse/BackWall", wall_cfg, translation=(0.0, -2.0, 1.0))

        # Left wall
        side_wall_cfg = sim_utils.CuboidCfg(
            size=(0.05, 4.0, 2.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.6, 0.3), opacity=0.4),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        )
        side_wall_cfg.func("/World/envs/env_.*/Greenhouse/LeftWall", side_wall_cfg, translation=(-2.0, 0.0, 1.0))

        # Right wall
        side_wall_cfg.func("/World/envs/env_.*/Greenhouse/RightWall", side_wall_cfg, translation=(2.0, 0.0, 1.0))

        # Plant table surface
        table_cfg = sim_utils.CuboidCfg(
            size=(0.8, 0.6, 0.02),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.35, 0.15)),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        )
        table_cfg.func("/World/envs/env_.*/Greenhouse/Table", table_cfg, translation=(0.0, -0.1, 0.69))

    # ------------------------------------------------------------------
    # Pre-physics step
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone().clamp(-1.0, 1.0)
        if self.cfg.control_mode == "joint_vel":
            self._pre_physics_step_joint_vel()
        else:
            self._pre_physics_step_ik()

    def _pre_physics_step_joint_vel(self):
        """Joint-velocity control mode (DROID): actions[:, :7] = joint vel, actions[:, 7] = gripper."""
        joint_vel = self.actions[:, :7]  # rad/s in [-1, 1]
        gripper_actions = self.actions[:, 7]

        self.gripper_open = gripper_actions > 0.0

        # Integrate: new_pos = current_pos + vel * dt
        arm_joint_pos = self._robot.data.joint_pos[:, self.arm_joint_ids]
        arm_joint_pos_des = arm_joint_pos + joint_vel * self.dt

        # Skip on first step after reset (use defaults)
        arm_joint_pos_des[self._first_step_after_reset] = self._robot.data.default_joint_pos[
            self._first_step_after_reset
        ][:, self.arm_joint_ids]
        self._first_step_after_reset[:] = False

        # Clamp to joint limits
        arm_lower = self.robot_dof_lower_limits[self.arm_joint_ids]
        arm_upper = self.robot_dof_upper_limits[self.arm_joint_ids]
        arm_joint_pos_des = torch.clamp(arm_joint_pos_des, arm_lower, arm_upper)

        self.robot_dof_targets[:, self.arm_joint_ids] = arm_joint_pos_des

        # Gripper targets
        finger_target = torch.where(
            self.gripper_open.unsqueeze(-1),
            torch.full((self.num_envs, len(self.finger_joint_ids)), self.cfg.gripper_open_value, device=self.device),
            torch.full((self.num_envs, len(self.finger_joint_ids)), self.cfg.gripper_close_value, device=self.device),
        )
        self.robot_dof_targets[:, self.finger_joint_ids] = finger_target

    def _pre_physics_step_ik(self):
        """IK control mode (LIBERO): actions[:, :6] = EE delta pose, actions[:, 6] = gripper."""
        # Split actions: arm delta pose (6) + gripper (1)
        arm_actions = self.actions[:, :6] * self.cfg.ik_command_scale
        gripper_actions = self.actions[:, 6]

        # Gripper: >0 = open, <0 = close
        self.gripper_open = gripper_actions > 0.0

        # ------------------------------------------------------------------
        # Differential IK
        # ------------------------------------------------------------------
        ee_pose_w = self._robot.data.body_pose_w[:, self.hand_body_idx]
        ee_pos_w = ee_pose_w[:, 0:3]
        ee_quat_w = ee_pose_w[:, 3:7]

        root_pose_w = self._robot.data.root_pose_w
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pos_w, ee_quat_w
        )

        self.diff_ik_controller.set_command(arm_actions, ee_pos=ee_pos_b, ee_quat=ee_quat_b)

        # Jacobian: world -> base frame
        jacobian_w = self._robot.root_physx_view.get_jacobians()[:, self.ee_jacobi_idx, :, self.arm_joint_ids]
        base_rot = root_pose_w[:, 3:7]
        base_rot_matrix = matrix_from_quat(quat_inv(base_rot))
        jacobian_b = jacobian_w.clone()
        jacobian_b[:, :3, :] = torch.bmm(base_rot_matrix, jacobian_b[:, :3, :])
        jacobian_b[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian_b[:, 3:, :])

        arm_joint_pos = self._robot.data.joint_pos[:, self.arm_joint_ids]
        arm_joint_pos_des = self.diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian_b, arm_joint_pos)

        # Skip IK on first step after reset
        arm_joint_pos_des[self._first_step_after_reset] = self._robot.data.default_joint_pos[
            self._first_step_after_reset
        ][:, self.arm_joint_ids]
        self._first_step_after_reset[:] = False

        # Clamp to joint limits
        arm_lower = self.robot_dof_lower_limits[self.arm_joint_ids]
        arm_upper = self.robot_dof_upper_limits[self.arm_joint_ids]
        arm_joint_pos_des = torch.clamp(arm_joint_pos_des, arm_lower, arm_upper)

        self.robot_dof_targets[:, self.arm_joint_ids] = arm_joint_pos_des

        # Gripper targets (values from config for robot-agnostic support)
        finger_target = torch.where(
            self.gripper_open.unsqueeze(-1),
            torch.full((self.num_envs, len(self.finger_joint_ids)), self.cfg.gripper_open_value, device=self.device),
            torch.full((self.num_envs, len(self.finger_joint_ids)), self.cfg.gripper_close_value, device=self.device),
        )
        self.robot_dof_targets[:, self.finger_joint_ids] = finger_target

    def _apply_action(self):
        self._robot.set_joint_position_target(
            self.robot_dof_targets[:, self.arm_joint_ids], joint_ids=self.arm_joint_ids
        )
        self._robot.set_joint_position_target(
            self.robot_dof_targets[:, self.finger_joint_ids], joint_ids=self.finger_joint_ids
        )

        # -- Grasp detection and fruit tracking --
        ee_pos = self._robot.data.body_pose_w[:, self.hand_body_idx, 0:3]
        ee_quat = self._robot.data.body_pose_w[:, self.hand_body_idx, 3:7]
        target_pos = self._get_target_fruit_positions()
        dist = torch.norm(ee_pos - target_pos, dim=-1)

        gripper_closed = ~self.gripper_open
        close_enough = dist < self.cfg.grasp_thresh

        # Grasp: EE near fruit + gripper closed + not already harvested
        newly_grasped = close_enough & gripper_closed & ~self._fruit_grasped & ~self.fruit_harvested
        self._fruit_grasped = self._fruit_grasped | newly_grasped

        # Release: gripper opens while grasped
        released = self.gripper_open & self._fruit_grasped
        self._fruit_grasped = self._fruit_grasped & ~released

        # -- Scene camera: fixed third-person view, set every step --
        if hasattr(self, "_scene_camera"):
            scene_eye = torch.tensor([[2.5, 1.3, 1.8]], device=self.device).expand(self.num_envs, -1)
            scene_target = torch.tensor([[0.3, 0.0, 0.85]], device=self.device).expand(self.num_envs, -1)
            self._scene_camera.set_world_poses_from_view(scene_eye, scene_target)

        # Wrist camera is rigidly mounted on panda_hand — no runtime update needed

        # Move grasped fruits to follow end-effector
        if self._fruit_grasped.any():
            offset = torch.tensor([[0.0, 0.0, -0.04]], device=self.device, dtype=torch.float32)
            for fruit_idx in range(NUM_FRUITS):
                mask = self._fruit_grasped & (self.target_fruit_idx == fruit_idx)
                if not mask.any():
                    continue
                env_ids = mask.nonzero(as_tuple=False).squeeze(-1)
                fruit = self._fruits[fruit_idx]
                new_pos = ee_pos[env_ids] + offset
                new_pose = torch.cat([new_pos, ee_quat[env_ids]], dim=-1)
                fruit.write_root_pose_to_sim(new_pose, env_ids=env_ids)

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        joint_pos = self._robot.data.joint_pos

        # Scale joint positions to [-1, 1]
        dof_pos_scaled = (
            2.0 * (joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )

        ee_pose_w = self._robot.data.body_pose_w[:, self.hand_body_idx]
        ee_pos_w = ee_pose_w[:, 0:3]
        ee_quat_w = ee_pose_w[:, 3:7]

        target_fruit_pos = self._get_target_fruit_positions()
        fruit_to_ee = target_fruit_pos - ee_pos_w
        dist_to_target = torch.norm(fruit_to_ee, dim=-1, keepdim=True)

        # Normalized finger state (0=closed, 1=open)
        finger_state = joint_pos[:, self.finger_joint_ids] / max(self.cfg.gripper_open_value, 1e-6)

        harvested = self.fruit_harvested.unsqueeze(-1).float()
        grasped_state = self._fruit_grasped.unsqueeze(-1).float()

        # Build obs and pad to fixed 27D regardless of arm joint count
        obs_parts = [
            dof_pos_scaled,      # (num_envs, num_joints)
            ee_pos_w,            # (num_envs, 3)
            ee_quat_w,           # (num_envs, 4)
            target_fruit_pos,    # (num_envs, 3)
            fruit_to_ee,         # (num_envs, 3)
            dist_to_target,      # (num_envs, 1)
            finger_state,        # (num_envs, num_fingers)
            grasped_state,       # (num_envs, 1)
            harvested,           # (num_envs, 1)
        ]
        obs = torch.cat(obs_parts, dim=-1)
        # Pad to observation_space (27) if fewer joints (e.g. UR5e has 6 arm + 2 finger = 8 vs Franka 9)
        obs_dim = obs.shape[-1]
        if obs_dim < self.cfg.observation_space:
            pad = torch.zeros(self.num_envs, self.cfg.observation_space - obs_dim, device=self.device)
            obs = torch.cat([obs, pad], dim=-1)
        elif obs_dim > self.cfg.observation_space:
            obs = obs[:, : self.cfg.observation_space]
        obs_dict: dict = {"policy": torch.clamp(obs, -5.0, 5.0)}

        # Raw joint state (for DROID mode and other joint-space policies)
        obs_dict["raw_joint_pos"] = self._robot.data.joint_pos[:, self.arm_joint_ids]
        obs_dict["raw_gripper_pos"] = self._robot.data.joint_pos[:, self.finger_joint_ids]

        # Include camera images when available (for VLA training)
        if "rgb" in self._wrist_camera.data.output:
            obs_dict["wrist_rgb"] = self._wrist_camera.data.output["rgb"]
        if "rgb" in self._scene_camera.data.output:
            obs_dict["scene_rgb"] = self._scene_camera.data.output["rgb"]

        return obs_dict

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        ee_pos_w = self._robot.data.body_pose_w[:, self.hand_body_idx, 0:3]
        target_fruit_pos = self._get_target_fruit_positions()

        dist = torch.norm(ee_pos_w - target_fruit_pos, dim=-1)

        # Distance reward: shaped as 1/(1+d^2)^2
        dist_reward = 1.0 / (1.0 + dist ** 2)
        dist_reward = dist_reward * dist_reward
        dist_reward = torch.where(dist <= 0.02, dist_reward * 2.0, dist_reward)

        # Grasp reward
        gripper_closed = ~self.gripper_open
        near_fruit = dist < self.cfg.grasp_thresh
        grasp_reward = (gripper_closed & near_fruit).float()

        # Harvest reward: fruit currently grasped AND lifted above threshold
        target_init_z = self.fruit_init_positions[self.target_fruit_idx, 2]
        target_cur_z = target_fruit_pos[:, 2]
        fruit_lifted = (target_cur_z - target_init_z) > self.cfg.harvest_height_thresh
        self.fruit_harvested = self.fruit_harvested | (fruit_lifted & self._fruit_grasped)
        harvest_reward = self.fruit_harvested.float()

        # Bonus for actively grasping (encourages holding)
        grasping_reward = self._fruit_grasped.float() * 0.5

        # Action penalty
        action_penalty = torch.sum(self.actions ** 2, dim=-1)

        rewards = (
            self.cfg.dist_reward_scale * dist_reward
            + self.cfg.grasp_reward_scale * grasp_reward
            + self.cfg.grasp_reward_scale * grasping_reward
            + self.cfg.harvest_reward_scale * harvest_reward
            - self.cfg.action_penalty_scale * action_penalty
        )

        self.extras["log"] = {
            "dist_reward": (self.cfg.dist_reward_scale * dist_reward).mean(),
            "grasp_reward": (self.cfg.grasp_reward_scale * grasp_reward).mean(),
            "harvest_reward": (self.cfg.harvest_reward_scale * harvest_reward).mean(),
            "action_penalty": (-self.cfg.action_penalty_scale * action_penalty).mean(),
            "mean_dist_to_fruit": dist.mean(),
            "grasping_reward": (self.cfg.grasp_reward_scale * grasping_reward).mean(),
            "harvest_rate": self.fruit_harvested.float().mean(),
            "grasp_rate": self._fruit_grasped.float().mean(),
        }

        return rewards

    # ------------------------------------------------------------------
    # Terminations
    # ------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = self.fruit_harvested.clone()
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)

        # Randomize target fruit
        self.target_fruit_idx[env_ids] = torch.randint(0, NUM_FRUITS, (len(env_ids),), device=self.device)
        self.fruit_harvested[env_ids] = False
        self._fruit_grasped[env_ids] = False
        self._first_step_after_reset[env_ids] = True

        # Reset robot joints with small noise
        joint_pos = self._robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.125, 0.125, (len(env_ids), self._robot.num_joints), self.device,
        )
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot_dof_targets[env_ids] = joint_pos

        # Reset fruit positions
        for fruit in self._fruits:
            default_state = fruit.data.default_root_state[env_ids].clone()
            fruit.write_root_state_to_sim(default_state, env_ids=env_ids)

        self.diff_ik_controller.reset(env_ids)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_target_fruit_positions(self) -> torch.Tensor:
        """Return (num_envs, 3) world positions of each env's targeted fruit."""
        all_fruit_pos = torch.stack([fruit.data.root_pos_w for fruit in self._fruits], dim=0)
        idx = self.target_fruit_idx.unsqueeze(0).unsqueeze(-1).expand(1, -1, 3)
        target_pos = torch.gather(all_fruit_pos, dim=0, index=idx).squeeze(0)
        return target_pos
