# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
UR5 Pick and Place Simulation
=============================

This script demonstrates a pick-and-place task using a UR5 robotic arm in Isaac Sim
with IsaacLab. The robot uses differential inverse kinematics (IK) to follow a
sequence of waypoints that execute:

    1. Move to home position
    2. Approach above the object
    3. Descend to grasp height
    4. Grasp the object (simulated via rigid attachment)
    5. Lift the object
    6. Transport to the place location
    7. Lower the object
    8. Release the object
    9. Retreat and return home

Usage:
    ./isaaclab.sh -p scripts/demos/ur5_pick_and_place.py --num_envs 1

"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

# -- CLI arguments --
parser = argparse.ArgumentParser(description="UR5 Pick and Place Demo")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# -- Launch Omniverse application --
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- Everything below runs after the Omniverse app is started ----

import torch
import enum

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import (
    Articulation,
    ArticulationCfg,
    AssetBaseCfg,
    RigidObject,
    RigidObjectCfg,
)
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import subtract_frame_transforms

# =============================================================================
# UR5 Robot Configuration
# =============================================================================
# The UR5 USD is available on the Isaac Sim Nucleus server.
# Joint structure: shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3

UR5_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/UniversalRobots/ur5/ur5.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=12,
            solver_velocity_iteration_count=1,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.5708,
            "elbow_joint": 1.5708,
            "wrist_1_joint": -1.5708,
            "wrist_2_joint": -1.5708,
            "wrist_3_joint": 0.0,
        },
    ),
    actuators={
        "shoulder": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_.*"],
            stiffness=800.0,
            damping=40.0,
        ),
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=["elbow_joint"],
            stiffness=800.0,
            damping=40.0,
        ),
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=["wrist_.*"],
            stiffness=800.0,
            damping=40.0,
        ),
    },
)


# =============================================================================
# State Machine
# =============================================================================

class PickPlaceState(enum.IntEnum):
    """States for the pick-and-place state machine."""
    HOME = 0
    APPROACH = 1
    DESCEND = 2
    GRASP = 3
    LIFT = 4
    TRANSPORT = 5
    LOWER = 6
    RELEASE = 7
    RETREAT = 8


class PickPlaceStateMachine:
    """A waypoint-based state machine for pick-and-place tasks.

    Each state drives the end-effector toward a target 7-D pose
    (position + quaternion) using differential IK.  When the EE is
    close enough to the target the machine advances to the next state.

    Grasp / release are simulated by teleporting the cube to the
    end-effector frame (``_attach_object``) and releasing it back to
    free-body dynamics (``_detach_object``).
    """

    def __init__(
        self,
        num_envs: int,
        device: str,
        pick_pos: torch.Tensor | None = None,
        place_pos: torch.Tensor | None = None,
    ):
        self.num_envs = num_envs
        self.device = device

        # -- pick / place positions (in robot root frame) --
        if pick_pos is None:
            pick_pos = torch.tensor([0.4, 0.0, 0.05], device=device)
        if place_pos is None:
            place_pos = torch.tensor([0.0, 0.4, 0.05], device=device)

        self.pick_pos = pick_pos
        self.place_pos = place_pos

        # End-effector orientation: pointing straight down  (w, x, y, z)
        self.ee_quat_down = torch.tensor([0.0, 1.0, 0.0, 0.0], device=device)

        # Height offsets
        self.approach_height = 0.25
        self.grasp_height = 0.04
        self.lift_height = 0.30
        self.transport_height = 0.30

        # State tracking per environment
        self.state = torch.full((num_envs,), PickPlaceState.HOME, dtype=torch.int32, device=device)
        self.wait_counter = torch.zeros(num_envs, dtype=torch.int32, device=device)

        # Whether the object is currently attached (grasped)
        self.object_attached = torch.zeros(num_envs, dtype=torch.bool, device=device)

        # Build the pose table  (num_states, 7)  [px, py, pz, qw, qx, qy, qz]
        self._build_waypoints()

    # --------------------------------------------------------------------- #
    # Waypoint table
    # --------------------------------------------------------------------- #
    def _build_waypoints(self):
        """Pre-compute the target EE poses for every state."""
        q = self.ee_quat_down  # orientation pointing down

        home_pos = torch.tensor([0.3, 0.0, 0.30], device=self.device)
        approach_pick = torch.tensor(
            [self.pick_pos[0], self.pick_pos[1], self.approach_height], device=self.device
        )
        descend_pick = torch.tensor(
            [self.pick_pos[0], self.pick_pos[1], self.grasp_height], device=self.device
        )
        grasp_pos = descend_pick.clone()
        lift_pos = torch.tensor(
            [self.pick_pos[0], self.pick_pos[1], self.lift_height], device=self.device
        )
        transport_pos = torch.tensor(
            [self.place_pos[0], self.place_pos[1], self.transport_height], device=self.device
        )
        lower_pos = torch.tensor(
            [self.place_pos[0], self.place_pos[1], self.grasp_height], device=self.device
        )
        release_pos = lower_pos.clone()
        retreat_pos = torch.tensor(
            [self.place_pos[0], self.place_pos[1], self.transport_height], device=self.device
        )

        poses = []
        for p in [home_pos, approach_pick, descend_pick, grasp_pos, lift_pos,
                   transport_pos, lower_pos, release_pos, retreat_pos]:
            poses.append(torch.cat([p, q]))

        # (num_states, 7)
        self.waypoints = torch.stack(poses, dim=0)

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def get_ee_target(self) -> torch.Tensor:
        """Return current target pose (num_envs, 7) based on per-env state."""
        targets = self.waypoints[self.state.long()]  # (N, 7)
        return targets

    def update(
        self,
        ee_pos_b: torch.Tensor,
        ee_quat_b: torch.Tensor,
    ) -> tuple[bool, bool]:
        """Advance the state machine.  Returns (grasp_trigger, release_trigger)."""
        targets = self.get_ee_target()
        pos_err = torch.norm(targets[:, :3] - ee_pos_b, dim=-1)

        grasp_triggered = False
        release_triggered = False

        # Threshold to consider "arrived"
        pos_threshold = 0.02

        arrived = pos_err < pos_threshold

        for i in range(self.num_envs):
            if not arrived[i]:
                self.wait_counter[i] = 0
                continue

            self.wait_counter[i] += 1

            # GRASP and RELEASE states need a short dwell before transitioning
            current = self.state[i].item()

            if current == PickPlaceState.GRASP:
                if self.wait_counter[i] > 10:
                    grasp_triggered = True
                    self.object_attached[i] = True
                    self.state[i] = PickPlaceState.LIFT
                    self.wait_counter[i] = 0
            elif current == PickPlaceState.RELEASE:
                if self.wait_counter[i] > 10:
                    release_triggered = True
                    self.object_attached[i] = False
                    self.state[i] = PickPlaceState.RETREAT
                    self.wait_counter[i] = 0
            elif current == PickPlaceState.RETREAT:
                if self.wait_counter[i] > 20:
                    # Cycle back to HOME -> APPROACH for continuous demo
                    self.state[i] = PickPlaceState.HOME
                    self.wait_counter[i] = 0
            elif current == PickPlaceState.HOME:
                if self.wait_counter[i] > 30:
                    self.state[i] = PickPlaceState.APPROACH
                    self.wait_counter[i] = 0
            else:
                # For all other states just advance immediately
                if self.wait_counter[i] > 5:
                    next_state = min(current + 1, PickPlaceState.RETREAT)
                    self.state[i] = next_state
                    self.wait_counter[i] = 0

        return grasp_triggered, release_triggered

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.state[env_ids] = PickPlaceState.HOME
        self.wait_counter[env_ids] = 0
        self.object_attached[env_ids] = False


# =============================================================================
# Scene Configuration
# =============================================================================

@configclass
class PickPlaceSceneCfg(InteractiveSceneCfg):
    """Scene with a UR5, a table, a cube to pick, and a target marker."""

    # -- ground plane --
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # -- lighting --
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75)),
    )

    # -- table / mount --
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd",
            scale=(2.0, 2.0, 2.0),
        ),
    )

    # -- UR5 robot --
    robot = UR5_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # -- cube to pick --
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.04, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=1.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.8, 0.2),  # green cube
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.4, 0.0, 0.02),  # on the table surface
        ),
    )


# =============================================================================
# Main simulation loop
# =============================================================================

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Run the pick-and-place simulation."""

    robot: Articulation = scene["robot"]
    cube: RigidObject = scene["cube"]

    # -- Differential IK controller --
    ik_cfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls",
    )
    ik_controller = DifferentialIKController(ik_cfg, num_envs=scene.num_envs, device=sim.device)

    # -- Resolve robot entity for IK --
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=[".*"], body_names=["ee_link"])
    robot_entity_cfg.resolve(scene)

    if robot.is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0]

    # -- Visualisation markers --
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.08, 0.08, 0.08)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    # -- State machine --
    sm = PickPlaceStateMachine(
        num_envs=scene.num_envs,
        device=sim.device,
        pick_pos=torch.tensor([0.4, 0.0, 0.05], device=sim.device),
        place_pos=torch.tensor([0.0, 0.4, 0.05], device=sim.device),
    )

    # -- Buffers --
    ik_commands = torch.zeros(scene.num_envs, ik_controller.action_dim, device=sim.device)
    joint_pos_des = robot.data.default_joint_pos[:, robot_entity_cfg.joint_ids].clone()

    # -- Place marker (red sphere) to show where to place the cube --
    place_marker_cfg = FRAME_MARKER_CFG.copy()
    place_marker_cfg.markers["frame"].scale = (0.12, 0.12, 0.12)
    place_marker = VisualizationMarkers(place_marker_cfg.replace(prim_path="/Visuals/place_target"))

    sim_dt = sim.get_physics_dt()
    count = 0

    print("[INFO] UR5 Pick-and-Place simulation starting...")
    print("[INFO] The robot will pick the green cube and place it at the place marker.")
    print("[INFO] Press Ctrl+C or close the window to exit.")

    # Initial scene update to populate data buffers
    scene.update(sim_dt)

    while simulation_app.is_running():
        # ---------------------------------------------------------------- #
        # 1. Get current EE pose in robot-root frame
        # ---------------------------------------------------------------- #
        ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
        root_pose_w = robot.data.root_pose_w
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3], ee_pose_w[:, 3:7],
        )

        # ---------------------------------------------------------------- #
        # 2. State machine update
        # ---------------------------------------------------------------- #
        grasp_trigger, release_trigger = sm.update(ee_pos_b, ee_quat_b)

        # Handle grasp: disable gravity on cube and we will manually track it
        if grasp_trigger:
            pass  # tracking is handled below

        # Handle release: re-enable cube free dynamics
        if release_trigger:
            pass  # cube will just stay where it is placed

        # ---------------------------------------------------------------- #
        # 3. Compute IK toward the current waypoint
        # ---------------------------------------------------------------- #
        target_pose = sm.get_ee_target()
        ik_commands[:] = target_pose
        ik_controller.reset()
        ik_controller.set_command(ik_commands)

        # Jacobian and current joint positions
        jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
        joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]

        # Compute desired joint positions
        joint_pos_des = ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

        # ---------------------------------------------------------------- #
        # 4. Apply joint targets
        # ---------------------------------------------------------------- #
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        scene.write_data_to_sim()

        # ---------------------------------------------------------------- #
        # 5. If object attached, teleport cube to EE
        # ---------------------------------------------------------------- #
        for i in range(scene.num_envs):
            if sm.object_attached[i]:
                # Place cube at the EE position (slightly below)
                ee_world_pos = ee_pose_w[i, 0:3].clone()
                ee_world_pos[2] -= 0.03  # offset below EE
                cube_pose = torch.zeros(1, 7, device=sim.device)
                cube_pose[0, 0:3] = ee_world_pos
                cube_pose[0, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=sim.device)
                cube.write_root_pose_to_sim(
                    cube_pose,
                    env_ids=torch.tensor([i], device=sim.device),
                )

        # ---------------------------------------------------------------- #
        # 6. Step physics
        # ---------------------------------------------------------------- #
        sim.step()
        count += 1
        scene.update(sim_dt)

        # ---------------------------------------------------------------- #
        # 7. Update visual markers
        # ---------------------------------------------------------------- #
        ee_state_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        ee_marker.visualize(ee_state_w[:, 0:3], ee_state_w[:, 3:7])
        goal_marker.visualize(
            ik_commands[:, 0:3] + scene.env_origins,
            ik_commands[:, 3:7],
        )
        # Place target marker
        place_world = sm.place_pos.unsqueeze(0).expand(scene.num_envs, -1) + scene.env_origins
        place_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=sim.device).expand(scene.num_envs, -1)
        place_marker.visualize(place_world, place_quat)

        # ---------------------------------------------------------------- #
        # 8. Print state for debugging (every 100 steps)
        # ---------------------------------------------------------------- #
        if count % 100 == 0:
            state_names = [s.name for s in PickPlaceState]
            for i in range(min(scene.num_envs, 4)):
                st = sm.state[i].item()
                print(
                    f"  Env {i}: state={state_names[st]:>10s}  "
                    f"ee_pos=({ee_pos_b[i, 0]:.3f}, {ee_pos_b[i, 1]:.3f}, {ee_pos_b[i, 2]:.3f})  "
                    f"target=({ik_commands[i, 0]:.3f}, {ik_commands[i, 1]:.3f}, {ik_commands[i, 2]:.3f})"
                )


def main():
    """Entry point."""
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    # Camera looking at the workspace
    sim.set_camera_view(eye=[1.2, 1.2, 1.0], target=[0.3, 0.2, 0.0])

    # Build scene
    scene_cfg = PickPlaceSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5)
    scene = InteractiveScene(scene_cfg)

    # Reset simulation
    sim.reset()
    print("[INFO] Scene setup complete. Starting pick-and-place loop...")

    # Run
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()