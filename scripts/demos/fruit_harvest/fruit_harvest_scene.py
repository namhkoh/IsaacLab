# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Fruit-harvesting demo -- GPU-accelerated, scripted multi-fruit version.

The robot sits on a 0.5 m pedestal, rotated 180 deg around Z to face the plant.
Position-only IK controls the fingertip center (via body-offset-adjusted
Jacobian), with no orientation constraint.  The Jacobian is properly
transformed from world frame to base frame before IK computation.

The robot sequentially harvests multiple fruits (configurable via
HARVEST_INDICES).  Each fruit is a separate dynamic rigid body held at its
branch by a spring-damper force with gravity compensation.  The robot
completes APPROACH -> GRASP -> RETRACT -> DELIVER -> RELEASE for each fruit,
then moves to the next.

Grasping is physics-based:
  - Each fruit is a dynamic rigid body with gravity always ON.  An upward
    gravity-compensation force cancels gravity while on the branch.
  - A spring-damper force anchors each fruit at its branch position.
  - Gripper physically closes around the fruit.
  - At RETRACT the spring target switches from branch -> fingertip, and the
    arm pulls the fruit off the branch.  The visual stem is hidden once the
    fruit displaces > 3 cm from the branch.
  - The spring-damper keeps the fruit at the fingertip during transport.
  - At RELEASE the force is turned off and the fruit drops into the basket.
  - All other fruits remain anchored at their branches via their own springs
    and react physically to gripper contact.
  - Previously harvested fruits remain in the basket under gravity.
  - Fruit-fruit collision is active (needed for basket stacking).  On-branch
    overlap is managed by soft position correction (HOLD_ALPHA).  All fruits
    also collide with the robot, basket, and ground.

Usage:
    isaaclab.bat -p scripts/demos/fruit_harvest/fruit_harvest_scene.py
"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Fruit-harvesting demo with Franka Panda.")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import os

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    combine_frame_transforms,
    matrix_from_quat,
    skew_symmetric_matrix,
    subtract_frame_transforms,
)

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG  # isort:skip

# ---------------------------------------------------------------------------
# Paths & metadata
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLANT_JSON = os.path.join(SCRIPT_DIR, "..", "..", "..", "..",
                          "Plant_Robot_Interaction-master", "data", "plant", "fruit-001.json")
METADATA_JSON = os.path.join(SCRIPT_DIR, "plant_metadata.json")

with open(METADATA_JSON, "r") as f:
    META = json.load(f)

# Multi-fruit harvest config
HARVEST_INDICES = [6, 4, 2, 0]    # metadata fruit indices, in harvest order
NUM_FRUITS = len(HARVEST_INDICES)

# Branch attachment points (where stems connect to the plant)
BRANCH_POSITIONS = [tuple(META["fruit_centers"][i]) for i in HARVEST_INDICES]

# Hang distances below branch (meters) -- varied for natural cluster look
HANG_DISTANCES = [0.06, 0.05, 0.08, 0.04]

# Per-fruit variations: (radius, mass, diffuse_color)
# Ordered by HARVEST_INDICES = [6, 4, 2, 0]
FRUIT_SPECS = [
    {"radius": 0.030, "mass": 0.022, "color": (0.85, 0.10, 0.08)},   # fruit 0: large ripe red
    {"radius": 0.026, "mass": 0.016, "color": (0.90, 0.35, 0.08)},   # fruit 1: medium orange-red
    {"radius": 0.023, "mass": 0.012, "color": (0.80, 0.55, 0.10)},   # fruit 2: small turning orange
    {"radius": 0.028, "mass": 0.019, "color": (0.70, 0.15, 0.08)},   # fruit 3: medium-large dark red
]

# Actual fruit positions: hanging below branches like a tomato cluster
FRUIT_POSITIONS = [
    (bp[0], bp[1], bp[2] - hd)
    for bp, hd in zip(BRANCH_POSITIONS, HANG_DISTANCES)
]

TABLE_HEIGHT = META["table_height"]           # 0.7 -- plant table
ROBOT_HEIGHT = 0.50                           # default EE lands near fruit height


# ---------------------------------------------------------------------------
# Fruit spawn helper
# ---------------------------------------------------------------------------
def _fruit_spawn(idx):
    """Spawn config for fruit at index *idx* (per-fruit radius/mass/color)."""
    spec = FRUIT_SPECS[idx]
    return sim_utils.SphereCfg(
        radius=spec["radius"],
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=False,
            disable_gravity=False,              # gravity ON -- compensated by upward force
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_depenetration_velocity=0.5,     # gentle collision separation
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=spec["mass"]),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.005,
            rest_offset=0.001,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=4.0,
            dynamic_friction=4.0,
            restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=spec["color"]),
    )


# ---------------------------------------------------------------------------
# Scene config -- 4 physics fruit rigid bodies
# ---------------------------------------------------------------------------
@configclass
class FruitHarvestSceneCfg(InteractiveSceneCfg):

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=5000.0, color=(0.8, 0.8, 0.8)),
    )

    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.init_state.pos = (0.50, -0.09, ROBOT_HEIGHT)
    robot.init_state.rot = (0.0, 0.0, 0.0, 1.0)  # 180 deg Z -- faces the plant

    fruit_0: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Fruit_0",
        spawn=_fruit_spawn(0),
        init_state=RigidObjectCfg.InitialStateCfg(pos=FRUIT_POSITIONS[0]),
    )
    fruit_1: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Fruit_1",
        spawn=_fruit_spawn(1),
        init_state=RigidObjectCfg.InitialStateCfg(pos=FRUIT_POSITIONS[1]),
    )
    fruit_2: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Fruit_2",
        spawn=_fruit_spawn(2),
        init_state=RigidObjectCfg.InitialStateCfg(pos=FRUIT_POSITIONS[2]),
    )
    fruit_3: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Fruit_3",
        spawn=_fruit_spawn(3),
        init_state=RigidObjectCfg.InitialStateCfg(pos=FRUIT_POSITIONS[3]),
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EE_BODY_OFFSET = 0.107     # panda_hand -> fingertip center (metres, along hand Z)
GRIPPER_OPEN = 0.04
GRIPPER_CLOSE = 0.018        # must be < smallest fruit radius (0.023) for contact

WARMUP_STEPS = 100
APPROACH_CONVERGE_DIST = 0.015   # 15 mm
MAX_STEP_NORM = 0.2              # rad -- safety cap on per-step joint delta norm

# Transport spring (current fruit tracking fingertip during RETRACT/DELIVER only)
K_SPRING = 100.0                 # N/m
D_DAMP = 5.0                     # Ns/m
MAX_FORCE = 2.0                  # N
GRAVITY_COMPS = [spec["mass"] * 9.81 for spec in FRUIT_SPECS]
SPRING_CUTOFF = 0.15             # m

# Soft position correction for held fruits -- allows physics reactions
# while preventing permanent drift.  0.0 = free, 1.0 = hard lock.
HOLD_ALPHA = 0.8

# Basket geometry
BASKET_CENTER = (0.45, 0.15)
BASKET_BOTTOM_Z = 0.80          # floor of basket
BASKET_WALL_H = 0.15            # wall height
BASKET_INNER = 0.13             # inner square side length
BASKET_THICK = 0.008            # wall / floor thickness

SCRIPT = [
    ("APPROACH", 1500, GRIPPER_OPEN),
    ("GRASP",     400, GRIPPER_CLOSE),
    ("RETRACT",   800, GRIPPER_CLOSE),   # pull fruit off branch toward robot
    ("DELIVER",   800, GRIPPER_CLOSE),   # carry to basket
    ("RELEASE",   300, GRIPPER_OPEN),    # drop into basket
]


# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------
def run_sim(sim, scene, stage):
    device = sim.device
    num_envs = scene.cfg.num_envs
    robot = scene["robot"]

    # All fruit rigid objects
    fruits = [scene[f"fruit_{i}"] for i in range(NUM_FRUITS)]

    fruit_positions_t = [
        torch.tensor([FRUIT_POSITIONS[i]], device=device, dtype=torch.float32).expand(num_envs, -1)
        for i in range(NUM_FRUITS)
    ]

    # Force buffers -- one per fruit (shape: num_envs x 1 body x 3)
    force_bufs = [torch.zeros(num_envs, 1, 3, device=device) for _ in range(NUM_FRUITS)]
    torque_bufs = [torch.zeros(num_envs, 1, 3, device=device) for _ in range(NUM_FRUITS)]

    # Pre-built pose tensors for position correction (held fruits)
    identity_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).expand(num_envs, -1)
    zero_vel = torch.zeros(num_envs, 6, device=device)
    held_poses = [
        torch.cat([fruit_positions_t[i], identity_quat], dim=-1)   # (num_envs, 7)
        for i in range(NUM_FRUITS)
    ]

    # -- Resolve joints --
    arm_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"],
                             body_names=["panda_hand"])
    arm_cfg.resolve(scene)
    ee_body_idx = arm_cfg.body_ids[0]
    ee_jac_idx = ee_body_idx - 1   # PhysX drops the fixed base from Jacobian rows

    finger_cfg = SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"])
    finger_cfg.resolve(scene)
    finger_ids = list(finger_cfg.joint_ids)
    arm_ids = list(arm_cfg.joint_ids)

    # -- Position-only IK (no orientation constraint) --
    ik_cfg = DifferentialIKControllerCfg(
        command_type="position", use_relative_mode=False,
        ik_method="dls", ik_params={"lambda_val": 0.05},
    )
    ik = DifferentialIKController(ik_cfg, num_envs=num_envs, device=device)
    j_lo = robot.data.soft_joint_pos_limits[0, arm_ids, 0]
    j_hi = robot.data.soft_joint_pos_limits[0, arm_ids, 1]

    # -- Body offset: panda_hand -> fingertip center --
    body_off_pos = torch.tensor([[0.0, 0.0, EE_BODY_OFFSET]], device=device).expand(num_envs, -1)
    body_off_rot = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).expand(num_envs, -1)

    # IK command buffer (position-only, N x 3)
    ik_cmd = torch.zeros(num_envs, 3, device=device)

    default_arm_pos = robot.data.default_joint_pos[:, arm_ids].clone()

    # -- Helpers --
    def set_gripper(val):
        robot.set_joint_position_target(
            torch.full((num_envs, len(finger_ids)), val, device=device),
            joint_ids=finger_ids)

    def set_ik_target(pos_w):
        """Set fingertip target (world frame). Internally transforms to base frame."""
        rp = robot.data.root_pose_w[:, :3]
        rq = robot.data.root_pose_w[:, 3:7]
        dummy_q = body_off_rot  # identity quat
        pb, _ = subtract_frame_transforms(rp, rq, pos_w, dummy_q)
        ik_cmd[:] = pb
        ik.set_command(ik_cmd, ee_quat=body_off_rot)

    def fingertip_world():
        """Fingertip position in world frame (uses actual EE orientation)."""
        ee = robot.data.body_pose_w[:, ee_body_idx]
        R = matrix_from_quat(ee[:, 3:7])                         # (N,3,3)
        off = body_off_pos.unsqueeze(-1)                          # (N,3,1)
        return ee[:, :3] + torch.bmm(R, off).squeeze(-1)          # (N,3)

    # -- Per-fruit waypoint builder --
    basket_rim_z = BASKET_BOTTOM_Z + BASKET_THICK + BASKET_WALL_H
    wp_deliver = torch.tensor(
        [[BASKET_CENTER[0], BASKET_CENTER[1], basket_rim_z + 0.06]],
        device=device).expand(num_envs, -1)

    def make_waypoints(fruit_idx):
        """Build waypoint dict for the fruit at index fruit_idx."""
        fp = fruit_positions_t[fruit_idx]
        wp_approach = fp.clone()
        wp_retract = fp.clone()
        wp_retract[:, 0] += 0.15    # 15cm toward robot
        wp_retract[:, 2] -= 0.10    # 10cm down (easier retraction path)
        return {
            "APPROACH": wp_approach,
            "GRASP":    wp_approach,
            "RETRACT":  wp_retract,
            "DELIVER":  wp_deliver,
            "RELEASE":  wp_deliver,
        }

    # -- Pre-step: set external forces --
    def pre_step_forces(cur_fruit, transport_active, released):
        """Set external forces before physics step.

        Only the current fruit during transport (RETRACT/DELIVER) gets a
        spring-damper force tracking the fingertip.  All other fruits get
        zero external force -- their positions are enforced by post_step_hold.
        """
        for i in range(NUM_FRUITS):
            if i == cur_fruit and transport_active and not released[i]:
                # Spring-damper tracking fingertip
                ft_pos = fingertip_world()
                f_pos = fruits[i].data.root_pos_w[:, :3]
                f_vel = fruits[i].data.root_vel_w[:, :3]
                displacement = ft_pos - f_pos
                dist = displacement.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                active = (dist < SPRING_CUTOFF).float()
                spring = K_SPRING * displacement * active
                damper = -D_DAMP * f_vel * active
                total = spring + damper
                total[:, 2] += GRAVITY_COMPS[cur_fruit]
                mag = total.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                total = total * torch.clamp(MAX_FORCE / mag, max=1.0)
                force_bufs[i][:, 0, :] = total
            else:
                force_bufs[i][:] = 0
            fruits[i].set_external_force_and_torque(
                force_bufs[i], torque_bufs[i], is_global=True)

    # -- Post-step: soft-correct held fruits toward their target positions --
    def post_step_hold(cur_fruit, transport_active, released):
        """After physics step: blend held fruits back toward target positions.

        Uses HOLD_ALPHA (0.8) to pull 80% back toward target each step.
        This allows fruits to physically react to gripper contact and
        fruit-fruit collisions (the remaining 20% displacement is visible
        as a brief wobble) while preventing permanent drift or flyaway.
        """
        for i in range(NUM_FRUITS):
            if released[i]:
                continue
            if i == cur_fruit and transport_active:
                continue   # transport fruit moves freely via spring

            # Blend position toward target
            actual_pos = fruits[i].data.root_pos_w[:, :3]
            corrected_pos = actual_pos + HOLD_ALPHA * (fruit_positions_t[i] - actual_pos)

            # Damp velocity proportionally
            actual_vel = fruits[i].data.root_vel_w
            damped_vel = actual_vel * (1.0 - HOLD_ALPHA)

            pose = torch.cat([corrected_pos, identity_quat], dim=-1)
            fruits[i].write_root_pose_to_sim(pose)
            fruits[i].write_root_velocity_to_sim(damped_vel)

    # -- Warm-up: hold all fruits at branch positions --
    dt = sim.get_physics_dt()
    released = [False] * NUM_FRUITS

    print(f"[WARMUP] {WARMUP_STEPS} steps ...")
    for _ in range(WARMUP_STEPS):
        robot.set_joint_position_target(default_arm_pos, joint_ids=arm_ids)
        set_gripper(GRIPPER_OPEN)
        pre_step_forces(-1, False, released)
        scene.write_data_to_sim(); sim.step()
        try:
            scene.update(dt)
        except Exception:
            break
        post_step_hold(-1, False, released)

    ft0 = fingertip_world()[0]
    rp0 = robot.data.root_pose_w[0]
    print(f"[DEBUG] Robot base  = ({rp0[0]:.3f}, {rp0[1]:.3f}, {rp0[2]:.3f})")
    print(f"[DEBUG] Fingertip   = ({ft0[0]:.3f}, {ft0[1]:.3f}, {ft0[2]:.3f})")
    for fi in range(NUM_FRUITS):
        fp = fruits[fi].data.root_pos_w[0]
        bp = fruit_positions_t[fi][0]
        drift = torch.norm(fp[:3] - bp).item()
        print(f"[DEBUG] Fruit {fi} (idx {HARVEST_INDICES[fi]}) "
              f"pos=({fp[0]:.3f}, {fp[1]:.3f}, {fp[2]:.3f}) drift={drift:.4f}m")
    print("[WARMUP] Done")

    # -- State --
    current_fruit = 0        # index into fruits[] / HARVEST_INDICES
    phase_idx = 0
    phase_step = 0
    force_active = False     # True after RETRACT -- spring targets fingertip
    stem_hidden = False      # True once visual stem is hidden

    waypoints = make_waypoints(current_fruit)
    phase_name, phase_duration, gripper_val = SCRIPT[phase_idx]
    set_ik_target(waypoints[phase_name])
    set_gripper(gripper_val)
    print(f"[SCRIPT] === FRUIT {current_fruit}/{NUM_FRUITS} (idx {HARVEST_INDICES[current_fruit]}) ===")
    print(f"[SCRIPT] Phase 0: {phase_name} ({phase_duration} steps)")

    while simulation_app.is_running():
        # ---- IK ----
        jpos = robot.data.joint_pos[:, arm_ids]
        if torch.isnan(jpos).any():
            robot.write_joint_state_to_sim(
                robot.data.default_joint_pos, robot.data.default_joint_vel)
            ik.set_command(ik_cmd)
            pre_step_forces(current_fruit, force_active, released)
            scene.write_data_to_sim(); sim.step()
            try: scene.update(dt)
            except Exception: pass
            post_step_hold(current_fruit, force_active, released)
            continue

        # 1. Raw Jacobian (world frame) -> base frame
        J_w = robot.root_physx_view.get_jacobians()[:, ee_jac_idx, :, arm_ids]
        rp_w = robot.data.root_pose_w
        qc = rp_w[:, 3:7].clone()
        qc[:, 1:] *= -1.0                                # quat conjugate
        R_inv = matrix_from_quat(qc)                      # (N,3,3)
        J = J_w.clone()
        J[:, :3, :] = torch.bmm(R_inv, J_w[:, :3, :])
        J[:, 3:, :] = torch.bmm(R_inv, J_w[:, 3:, :])

        # 2. EE pose in base frame
        ee_w = robot.data.body_pose_w[:, ee_body_idx]
        ee_b, eq_b = subtract_frame_transforms(
            rp_w[:, :3], rp_w[:, 3:7], ee_w[:, :3], ee_w[:, 3:7])

        # 3. Apply body offset to Jacobian -- offset must be in BASE frame
        ee_rot_b = matrix_from_quat(eq_b)                        # (N,3,3)
        offset_in_base = torch.bmm(
            ee_rot_b, body_off_pos.unsqueeze(-1)).squeeze(-1)     # (N,3)
        J[:, 0:3, :] += torch.bmm(
            -skew_symmetric_matrix(offset_in_base), J[:, 3:, :])

        # 4. Fingertip position in base frame (for IK error computation)
        ee_b_ft, eq_b_ft = combine_frame_transforms(
            ee_b, eq_b, body_off_pos, body_off_rot)

        # 5. Compute position-only IK
        try:
            jd = ik.compute(ee_b_ft, eq_b_ft, J, jpos)
        except RuntimeError:
            jd = jpos.clone()
        if torch.isnan(jd).any():
            jd = jpos.clone()

        # 6. Safety limiter
        delta = jd - jpos
        norm = delta.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = torch.clamp(MAX_STEP_NORM / norm, max=1.0)
        jd = jpos + delta * scale
        jd = torch.clamp(jd, j_lo.unsqueeze(0), j_hi.unsqueeze(0))

        robot.set_joint_position_target(jd, joint_ids=arm_ids)
        set_gripper(gripper_val)

        # ---- Timeline ----
        phase_step += 1
        ft = fingertip_world()[0]
        tgt = waypoints[phase_name][0]
        dist = torch.norm(ft - tgt).item()

        # First-step diagnostics
        if phase_step == 1 and phase_name == "APPROACH":
            err = ik_cmd[0] - ee_b_ft[0]
            print(f"[IK-DBG] fingertip(base)=({ee_b_ft[0,0]:.4f},{ee_b_ft[0,1]:.4f},{ee_b_ft[0,2]:.4f})")
            print(f"[IK-DBG] target(base)   =({ik_cmd[0,0]:.4f},{ik_cmd[0,1]:.4f},{ik_cmd[0,2]:.4f})")
            print(f"[IK-DBG] pos error(base)=({err[0]:.4f},{err[1]:.4f},{err[2]:.4f}) |err|={err.norm():.4f}")
            J_sv = torch.linalg.svdvals(J[0, :3, :])
            print(f"[IK-DBG] J_pos singular vals={J_sv.tolist()}")

        log_interval = 50 if phase_name == "APPROACH" else 100
        if phase_step % log_interval == 1:
            fing = robot.data.joint_pos[0, finger_ids]
            cf = fruits[current_fruit]
            fp = cf.data.root_pos_w[0]
            print(f"  [{phase_name} {phase_step:4d}/{phase_duration}] "
                  f"FT=({ft[0]:.3f},{ft[1]:.3f},{ft[2]:.3f}) "
                  f"dist={dist:.4f} fing={fing[0]:.4f} "
                  f"fruit=({fp[0]:.3f},{fp[1]:.3f},{fp[2]:.3f})")

        if phase_name == "APPROACH" and phase_step > 50 and dist < APPROACH_CONVERGE_DIST:
            print(f"  >>> APPROACH converged at step {phase_step} (dist={dist:.4f})")
            phase_step = phase_duration

        if phase_step >= phase_duration:
            phase_idx += 1
            if phase_idx >= len(SCRIPT):
                # ---- Harvest complete for this fruit -- advance ----
                released[current_fruit] = True
                current_fruit += 1
                if current_fruit >= NUM_FRUITS:
                    print("[SCRIPT] ALL FRUITS HARVESTED -- idling")
                    while simulation_app.is_running():
                        robot.set_joint_position_target(jd, joint_ids=arm_ids)
                        set_gripper(GRIPPER_OPEN)
                        pre_step_forces(-1, False, released)
                        scene.write_data_to_sim(); sim.step()
                        try: scene.update(dt)
                        except Exception: break
                        post_step_hold(-1, False, released)
                    break

                # Reset for next fruit
                phase_idx = 0
                phase_step = 0
                force_active = False
                stem_hidden = False
                waypoints = make_waypoints(current_fruit)
                phase_name, phase_duration, gripper_val = SCRIPT[0]
                set_ik_target(waypoints[phase_name])
                set_gripper(gripper_val)
                print(f"[SCRIPT] === FRUIT {current_fruit}/{NUM_FRUITS} "
                      f"(idx {HARVEST_INDICES[current_fruit]}) ===")
                print(f"[SCRIPT] Phase 0: {phase_name} ({phase_duration} steps)")
            else:
                phase_name, phase_duration, gripper_val = SCRIPT[phase_idx]
                phase_step = 0
                set_ik_target(waypoints[phase_name])
                set_gripper(gripper_val)
                print(f"[SCRIPT] Phase {phase_idx}: {phase_name} ({phase_duration} steps)")

                # RETRACT start: switch to spring-based fingertip tracking
                if phase_name == "RETRACT" and not force_active:
                    force_active = True
                    fp = fruits[current_fruit].data.root_pos_w[0]
                    print(f"  >>> RETRACT -- spring -> fingertip")
                    print(f"      fruit=({fp[0]:.4f},{fp[1]:.4f},{fp[2]:.4f})")

                # RELEASE start: turn off all forces on current fruit
                if phase_name == "RELEASE":
                    released[current_fruit] = True
                    fp = fruits[current_fruit].data.root_pos_w[0]
                    print(f"  >>> RELEASE -- force OFF, fruit at ({fp[0]:.3f},{fp[1]:.3f},{fp[2]:.3f})")

        # ---- Pre-step: external forces ----
        pre_step_forces(current_fruit, force_active, released)

        scene.write_data_to_sim()
        sim.step()
        try:
            scene.update(dt)
        except Exception:
            break

        # ---- Post-step: lock held fruits at target positions ----
        post_step_hold(current_fruit, force_active, released)

        # ---- Stem hiding: once fruit moves > 3cm from branch, hide stem ----
        if not stem_hidden and force_active:
            fruit_actual = fruits[current_fruit].data.root_pos_w[:, :3]
            disp = torch.norm(fruit_actual[0] - fruit_positions_t[current_fruit][0]).item()
            if disp > 0.03:  # 3cm threshold
                stem_hidden = True
                from pxr import UsdGeom as _UsdGeom
                stem_prim = stage.GetPrimAtPath(f"/World/Plant/fruit_stem_{current_fruit}")
                if stem_prim.IsValid():
                    _UsdGeom.Imageable(stem_prim).MakeInvisible()
                print(f"  >>> Fruit {current_fruit} detached -- stem hidden (disp={disp:.3f}m)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    sim_cfg = sim_utils.SimulationCfg(
        dt=0.005,
        device=args_cli.device,
        use_fabric=True,
        physx=sim_utils.PhysxCfg(
            solver_type=1,
            max_position_iteration_count=16,
            max_velocity_iteration_count=1,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            gpu_max_rigid_contact_count=2**20,
            gpu_max_rigid_patch_count=2**18,
            gpu_found_lost_pairs_capacity=2**20,
            gpu_total_aggregate_pairs_capacity=2**20,
            gpu_collision_stack_size=2**26,
            gpu_max_num_partitions=8,
        ),
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[0.85, 0.60, 1.50], target=[0.30, 0.0, 1.20])

    scene_cfg = FruitHarvestSceneCfg(num_envs=args_cli.num_envs, env_spacing=3.0)
    scene = InteractiveScene(scene_cfg)

    stage = sim.stage
    import sys
    if SCRIPT_DIR not in sys.path:
        sys.path.insert(0, SCRIPT_DIR)
    from plant_physics import build_plant_with_physics
    build_plant_with_physics(stage, PLANT_JSON, TABLE_HEIGHT,
                             target_fruit_indices=HARVEST_INDICES,
                             fruit_world_positions=FRUIT_POSITIONS)

    from pxr import UsdPhysics

    # Collision groups: fruits collide with each other, robot, basket, ground.
    # Filtered against branches only (fruits are held by soft position correction).
    fcg = stage.GetPrimAtPath("/World/Plant/FruitCollisionGroup")
    bcg = stage.GetPrimAtPath("/World/Plant/BranchCollisionGroup")
    if fcg.IsValid():
        fcg_includes = fcg.GetRelationship("collection:colliders:includes")
        for fi in range(NUM_FRUITS):
            fcg_includes.AddTarget(f"/World/envs/env_0/Fruit_{fi}")
        UsdPhysics.CollisionGroup(fcg).GetFilteredGroupsRel().AddTarget(
            "/World/Plant/BranchCollisionGroup")
    if bcg.IsValid():
        UsdPhysics.CollisionGroup(bcg).GetFilteredGroupsRel().AddTarget(
            "/World/Plant/FruitCollisionGroup")

    from pxr import Gf, Sdf, UsdGeom, UsdShade

    mat = UsdShade.Material.Define(stage, "/World/Tables/Mat")
    sh = UsdShade.Shader.Define(stage, "/World/Tables/Mat/Sh")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.45, 0.3, 0.15))
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")

    for name, pos, sc in [
        ("PlantTable", (0, 0, TABLE_HEIGHT / 2), (0.35, 0.35, TABLE_HEIGHT)),
        ("RobotStand", (0.50, -0.09, ROBOT_HEIGHT / 2), (0.22, 0.28, ROBOT_HEIGHT)),
    ]:
        xf = UsdGeom.Xform.Define(stage, f"/World/Tables/{name}")
        c = UsdGeom.Cube.Define(stage, f"/World/Tables/{name}/c")
        c.GetSizeAttr().Set(1.0)
        UsdGeom.XformCommonAPI(xf).SetTranslate(Gf.Vec3d(*pos))
        UsdGeom.XformCommonAPI(xf).SetScale(Gf.Vec3f(*sc))
        UsdShade.MaterialBindingAPI(c.GetPrim()).Bind(mat)

    # -- Harvest basket --
    bk_mat = UsdShade.Material.Define(stage, "/World/Basket/Mat")
    bk_sh = UsdShade.Shader.Define(stage, "/World/Basket/Mat/Sh")
    bk_sh.CreateIdAttr("UsdPreviewSurface")
    bk_sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.55, 0.35, 0.12))   # wicker brown
    bk_mat.CreateSurfaceOutput().ConnectToSource(bk_sh.ConnectableAPI(), "surface")

    bx, by = BASKET_CENTER
    outer = BASKET_INNER + 2 * BASKET_THICK
    wall_z = BASKET_BOTTOM_Z + BASKET_THICK + BASKET_WALL_H / 2.0
    half_off = BASKET_INNER / 2.0 + BASKET_THICK / 2.0

    # Stand / pedestal under the basket
    stand_xf = UsdGeom.Xform.Define(stage, "/World/Basket/Stand")
    stand_c = UsdGeom.Cube.Define(stage, "/World/Basket/Stand/c")
    stand_c.GetSizeAttr().Set(1.0)
    UsdGeom.XformCommonAPI(stand_xf).SetTranslate(
        Gf.Vec3d(bx, by, BASKET_BOTTOM_Z / 2.0))
    UsdGeom.XformCommonAPI(stand_xf).SetScale(
        Gf.Vec3f(outer + 0.02, outer + 0.02, float(BASKET_BOTTOM_Z)))
    UsdShade.MaterialBindingAPI(stand_c.GetPrim()).Bind(mat)   # same wood as tables

    # Floor + 4 walls (with collision so the fruit stays inside)
    basket_pieces = [
        ("Floor",     (bx, by, BASKET_BOTTOM_Z + BASKET_THICK / 2.0),
                      (outer, outer, BASKET_THICK)),
        ("WallFront", (bx, by + half_off, wall_z),
                      (outer, BASKET_THICK, BASKET_WALL_H)),
        ("WallBack",  (bx, by - half_off, wall_z),
                      (outer, BASKET_THICK, BASKET_WALL_H)),
        ("WallLeft",  (bx - half_off, by, wall_z),
                      (BASKET_THICK, BASKET_INNER, BASKET_WALL_H)),
        ("WallRight", (bx + half_off, by, wall_z),
                      (BASKET_THICK, BASKET_INNER, BASKET_WALL_H)),
    ]
    for bname, bpos, bscale in basket_pieces:
        bxf = UsdGeom.Xform.Define(stage, f"/World/Basket/{bname}")
        bc = UsdGeom.Cube.Define(stage, f"/World/Basket/{bname}/c")
        bc.GetSizeAttr().Set(1.0)
        UsdGeom.XformCommonAPI(bxf).SetTranslate(Gf.Vec3d(*bpos))
        UsdGeom.XformCommonAPI(bxf).SetScale(Gf.Vec3f(*bscale))
        UsdPhysics.CollisionAPI.Apply(bc.GetPrim())
        UsdShade.MaterialBindingAPI(bc.GetPrim()).Bind(bk_mat)

    # -- Calyx (star-shaped sepals) on each fruit -- visual only, no collision --
    # IMPORTANT: Must be created BEFORE sim.reset() so Fabric/PhysX sees the
    # final USD stage.  Adding child prims to rigid bodies after reset can
    # corrupt the Fabric simulation state and break physics behaviour.
    calyx_mat = UsdShade.Material.Define(stage, "/World/CalyxMat")
    calyx_sh = UsdShade.Shader.Define(stage, "/World/CalyxMat/Sh")
    calyx_sh.CreateIdAttr("UsdPreviewSurface")
    calyx_sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.18, 0.45, 0.10))   # dark green
    calyx_mat.CreateSurfaceOutput().ConnectToSource(calyx_sh.ConnectableAPI(), "surface")

    NUM_SEPALS = 5
    SEPAL_TILT = 65.0   # degrees from vertical -- sepals radiate outward

    for fi in range(NUM_FRUITS):
        r = FRUIT_SPECS[fi]["radius"]
        fruit_prim_path = f"/World/envs/env_0/Fruit_{fi}"

        sepal_len = r * 0.75        # length of each pointed sepal
        sepal_base_r = r * 0.14     # base radius (narrow pointed cone)

        # 5 sepals arranged as a star, each tilted outward
        for si in range(NUM_SEPALS):
            angle = si * 72.0       # 360 / 5 = 72 degrees apart
            sepal_path = f"{fruit_prim_path}/sepal_{si}"
            sepal = UsdGeom.Cone.Define(stage, sepal_path)
            sepal.GetRadiusAttr().Set(sepal_base_r)
            sepal.GetHeightAttr().Set(sepal_len)
            sepal.GetAxisAttr().Set("Z")
            sx = UsdGeom.XformCommonAPI(sepal)
            sx.SetScale(Gf.Vec3f(1.0, 0.35, 1.0))    # flatten for leaf shape
            sx.SetRotate(Gf.Vec3f(0, -SEPAL_TILT, angle))
            sx.SetTranslate(Gf.Vec3d(0, 0, r * 0.88))
            UsdShade.MaterialBindingAPI(sepal.GetPrim()).Bind(calyx_mat)

        # Tiny central stem nub
        stem_path = f"{fruit_prim_path}/calyx_stem"
        stem = UsdGeom.Cylinder.Define(stage, stem_path)
        stem.GetRadiusAttr().Set(0.0015)           # 1.5 mm thin
        stem.GetHeightAttr().Set(r * 0.3)
        stem.GetAxisAttr().Set("Z")
        stem_xf = UsdGeom.XformCommonAPI(stem)
        stem_xf.SetTranslate(Gf.Vec3d(0, 0, r * 1.1))
        UsdShade.MaterialBindingAPI(stem.GetPrim()).Bind(calyx_mat)

    sim.reset()
    scene.update(sim_cfg.dt)

    num_envs = args_cli.num_envs
    robot = scene["robot"]
    mats = robot.root_physx_view.get_material_properties()
    mats[..., 0] = 2.0
    mats[..., 1] = 2.0
    mats[..., 2] = 0.0
    robot.root_physx_view.set_material_properties(
        mats, torch.arange(num_envs, device="cpu"))

    run_sim(sim, scene, stage)


if __name__ == "__main__":
    main()
    simulation_app.close()
