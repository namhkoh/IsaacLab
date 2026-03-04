"""Multi-articulation plant physics using Featherstone reduced-coordinate solver.

Partitions the plant's tree topology into sub-articulations of ≤52 capsule links
each (PhysX limit: 64).  Each sub-tree root is fixed to the
world.  This lets the Featherstone solver handle the full tree exactly in O(N)
time — no convergence issues regardless of chain depth.

Creates BOTH geometry and physics in a single pass so that every capsule is a
child of its articulation Xform from the start.

Usage (called from fruit_harvest_scene.py):
    from plant_physics import build_plant_with_physics
    build_plant_with_physics(stage, plant_json_path, table_height)
"""

from __future__ import annotations

import json
import math

import numpy as np


# ---------------------------------------------------------------------------
# Tunable parameters
# ---------------------------------------------------------------------------
STIFFNESS_SCALE = 200000.0    # Cosserat Ks -> N·m/rad (high enough to resist gravity droop)
DAMPING_RATIO   = 3.0         # overdamped to prevent oscillation
MASS_SCALE      = 5e4         # microgram -> gram scale
MIN_MASS        = 0.005       # 5 grams floor
MAX_PARTITION   = 52          # capsule links per articulation (leave room for fruit links)
MIN_RADIUS      = 0.003       # visual minimum (3 mm)


# ---------------------------------------------------------------------------
# Quaternion helpers (w, x, y, z convention)
# ---------------------------------------------------------------------------
def _quat_z_to(direction: np.ndarray):
    """Return quaternion (w, x, y, z) rotating Z-axis onto direction."""
    d = direction / (np.linalg.norm(direction) + 1e-12)
    dot = float(d[2])
    if dot > 0.9999:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if dot < -0.9999:
        return np.array([0.0, 1.0, 0.0, 0.0])
    axis = np.array([-d[1], d[0], 0.0])
    axis = axis / np.linalg.norm(axis)
    half = math.acos(max(-1.0, min(1.0, dot))) / 2.0
    s = math.sin(half)
    return np.array([math.cos(half), axis[0] * s, axis[1] * s, axis[2] * s])


def _quat_inv(q):
    """Inverse of quaternion (w, x, y, z)."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _quat_mul(a, b):
    """Multiply two quaternions (w, x, y, z)."""
    return np.array([
        a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3],
        a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2],
        a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1],
        a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0],
    ])


def _quat_rotate(q, v):
    """Rotate vector v by quaternion q (w, x, y, z)."""
    qv = np.array([q[1], q[2], q[3]])
    t = 2.0 * np.cross(qv, v)
    return v + q[0] * t + np.cross(qv, t)


# ---------------------------------------------------------------------------
# Tree partitioning
# ---------------------------------------------------------------------------
def _partition_tree(fathers, capsule_nodes, max_size=MAX_PARTITION):
    """Partition capsule_nodes into sub-trees of ≤max_size links each.

    Uses DFS with subtree-size checks to *strictly* enforce the size limit.
    Before including a child's entire subtree, we verify it fits within the
    remaining budget.  If not, the child becomes the root of a new partition.

    Returns list of dicts: [{"root_node": int, "nodes": [int, ...]}]
    """
    capsule_set = set(capsule_nodes)

    # Build children map: capsule endpoint -> list of capsules starting there.
    # Capsule cn spans from fathers[cn] to cn.  A child capsule cc starts at
    # fathers[cc]; if fathers[cc] is itself a capsule endpoint, cc is a child
    # of that capsule.
    children_map = {}  # capsule_node -> [child capsule_nodes]
    root_capsules = []

    for cn in capsule_nodes:
        children_map[cn] = []

    for cn in capsule_nodes:
        parent_node = fathers[cn]
        if parent_node in capsule_set:
            children_map[parent_node].append(cn)
        else:
            root_capsules.append(cn)

    # Pre-compute subtree sizes (bottom-up)
    subtree_size = {}

    def _compute_size(node):
        s = 1
        for child in children_map[node]:
            s += _compute_size(child)
        subtree_size[node] = s
        return s

    for root in root_capsules:
        _compute_size(root)

    # DFS partitioning — strictly respects max_size
    partitions = []

    def _dfs_partition(start):
        """Build one partition from start.  Returns (nodes, deferred_roots).

        At each node we check every child: if the child's full subtree fits
        within the remaining budget we recurse into it; otherwise the child
        is deferred as a new partition root.  This guarantees the partition
        never exceeds max_size.
        """
        partition = []
        deferred = []

        def _visit(node):
            partition.append(node)
            # Process smaller subtrees first so we pack the partition tightly
            children = sorted(children_map[node],
                              key=lambda c: subtree_size[c])
            for child in children:
                if len(partition) + subtree_size[child] <= max_size:
                    _visit(child)
                else:
                    deferred.append(child)

        _visit(start)
        return partition, deferred

    to_process = list(root_capsules)
    while to_process:
        start = to_process.pop(0)
        nodes, deferred = _dfs_partition(start)
        partitions.append({"root_node": nodes[0], "nodes": nodes})
        to_process.extend(deferred)

    return partitions


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------
def build_plant_with_physics(stage, plant_json_path: str, table_height: float,
                             target_fruit_index=None,
                             target_fruit_indices=None,
                             fruit_world_positions=None):
    """Create plant geometry AND physics as multi-articulation sub-trees.

    Each sub-tree is an independent ArticulationRootAPI with ≤60 links,
    solved exactly by PhysX's Featherstone reduced-coordinate solver.

    Args:
        target_fruit_index: single fruit index (backward compat)
        target_fruit_indices: list of fruit indices to create stems for
        fruit_world_positions: optional list of (x,y,z) world positions for
            each target fruit (same order as target_fruit_indices).  When
            provided, stems extend from branch attachment to these positions
            instead of the default plant-data positions.
    """
    # Backward compatibility: single index -> list
    if target_fruit_indices is None and target_fruit_index is not None:
        target_fruit_indices = [target_fruit_index]
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade, PhysxSchema

    # ------------------------------------------------------------------
    # Load plant data
    # ------------------------------------------------------------------
    with open(plant_json_path, "r") as f:
        plant = json.load(f)

    positions = np.array(plant["positions"], dtype=np.float64)
    positions[:, 2] += table_height
    fathers = np.array(plant["fathers"], dtype=int)
    fruit_radii = np.array(plant["fruit_radii"], dtype=np.float64)
    radii = np.array(plant["radii"], dtype=np.float64)
    ks_data = plant["Ks"]
    masses = plant["masses"]
    fixeds = plant["fixeds"]
    n_nodes = len(positions)

    # ------------------------------------------------------------------
    # Build capsule list and orientation lookup
    # ------------------------------------------------------------------
    capsule_nodes = []
    capsule_quat = {}  # node_index -> world orientation quaternion (w,x,y,z)

    for i in range(n_nodes):
        if fathers[i] < 0:
            continue
        if fruit_radii[i] > 0:
            continue
        p0 = positions[fathers[i]]
        p1 = positions[i]
        d = p1 - p0
        if np.linalg.norm(d) < 1e-6:
            continue
        capsule_nodes.append(i)
        capsule_quat[i] = _quat_z_to(d)

    capsule_set = set(capsule_nodes)

    print(f"[PlantPhysics] {len(capsule_nodes)} branch capsules from {n_nodes} nodes")

    # ------------------------------------------------------------------
    # Partition tree into sub-articulations
    # ------------------------------------------------------------------
    partitions = _partition_tree(fathers, capsule_nodes, MAX_PARTITION)
    total_assigned = sum(len(p["nodes"]) for p in partitions)
    max_part_size = max(len(p["nodes"]) for p in partitions)
    print(f"[PlantPhysics] Partitioned into {len(partitions)} sub-articulations "
          f"({total_assigned} nodes assigned, max partition={max_part_size}):")
    node_to_artic = {}
    for k, part in enumerate(partitions):
        tag = " ** OVER LIMIT **" if len(part["nodes"]) > MAX_PARTITION else ""
        print(f"  artic_{k}: root={part['root_node']}, links={len(part['nodes'])}{tag}")
        for nd in part["nodes"]:
            node_to_artic[nd] = k

    # ------------------------------------------------------------------
    # Root xform + material
    # ------------------------------------------------------------------
    UsdGeom.Xform.Define(stage, "/World/Plant")

    branch_mat = UsdShade.Material.Define(stage, "/World/Plant/BranchMaterial")
    branch_sh = UsdShade.Shader.Define(stage, "/World/Plant/BranchMaterial/Shader")
    branch_sh.CreateIdAttr("UsdPreviewSurface")
    branch_sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.18, 0.55, 0.12)
    )
    branch_sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.85)
    branch_mat.CreateSurfaceOutput().ConnectToSource(
        branch_sh.ConnectableAPI(), "surface"
    )

    # ------------------------------------------------------------------
    # Collision group: branches don't collide with each other but DO
    # collide with the robot.
    # ------------------------------------------------------------------
    group_path = "/World/Plant/BranchCollisionGroup"
    col_group = UsdPhysics.CollisionGroup.Define(stage, group_path)
    col_group.CreateFilteredGroupsRel().SetTargets([group_path])
    group_prim = col_group.GetPrim()
    group_prim.CreateAttribute(
        "collection:colliders:expansionRule", Sdf.ValueTypeNames.Token
    ).Set("expandPrims")
    col_includes = group_prim.CreateRelationship("collection:colliders:includes")

    # Fruit collision group — NO self-filter so fruits collide with each other.
    # Filtered against BranchCollisionGroup (set in scene script) so fruits
    # don't collide with plant branches.
    fruit_group_path = "/World/Plant/FruitCollisionGroup"
    fruit_col_group = UsdPhysics.CollisionGroup.Define(stage, fruit_group_path)
    fruit_col_group.CreateFilteredGroupsRel()
    fruit_group_prim = fruit_col_group.GetPrim()
    fruit_group_prim.CreateAttribute(
        "collection:colliders:expansionRule", Sdf.ValueTypeNames.Token
    ).Set("expandPrims")
    fruit_group_prim.CreateRelationship("collection:colliders:includes")

    # ------------------------------------------------------------------
    # Build parent-capsule lookup: for a capsule cn, its parent capsule is
    # the capsule whose endpoint == fathers[cn].
    # ------------------------------------------------------------------
    def find_parent_capsule(child_node):
        parent_node = fathers[child_node]
        if parent_node < 0:
            return None
        if parent_node in capsule_set:
            return parent_node
        return None

    # ------------------------------------------------------------------
    # Build each sub-articulation
    # ------------------------------------------------------------------
    total_links = 0
    total_joints = 0

    for k, part in enumerate(partitions):
        artic_path = f"/World/Plant/artic_{k}"
        nodes_in_part = part["nodes"]
        nodes_set = set(nodes_in_part)

        # --- Create articulation container ---
        artic_xform = UsdGeom.Xform.Define(stage, artic_path)
        artic_prim = artic_xform.GetPrim()
        UsdPhysics.ArticulationRootAPI.Apply(artic_prim)
        physx_artic = PhysxSchema.PhysxArticulationAPI.Apply(artic_prim)
        physx_artic.CreateEnabledSelfCollisionsAttr().Set(False)

        # --- Create body + geometry for each capsule node ---
        for cn in nodes_in_part:
            parent_node = fathers[cn]
            p0 = positions[parent_node]
            p1 = positions[cn]
            direction = p1 - p0
            seg_len = float(np.linalg.norm(direction))
            if seg_len < 1e-6:
                continue

            midpoint = (p0 + p1) / 2.0
            q = capsule_quat[cn]
            r = max(float(radii[parent_node]), float(radii[cn]), MIN_RADIUS)
            half_len = seg_len / 2.0

            # Body xform
            body_path = f"{artic_path}/b_{cn:04d}"
            body_xform = UsdGeom.Xform.Define(stage, body_path)
            body_xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
                Gf.Vec3d(float(midpoint[0]), float(midpoint[1]), float(midpoint[2]))
            )
            body_xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
                Gf.Quatd(float(q[0]), float(q[1]), float(q[2]), float(q[3]))
            )

            body_prim = body_xform.GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(body_prim)

            # Mass + inertia (solid cylinder approximation)
            mass_val = float(masses[cn]) * MASS_SCALE
            mass_val = max(mass_val, MIN_MASS)
            seg_r = max(float(radii[cn]), MIN_RADIUS)
            i_perp = mass_val * (3.0 * seg_r**2 + seg_len**2) / 12.0
            i_axial = mass_val * seg_r**2 / 2.0

            mass_api = UsdPhysics.MassAPI.Apply(body_prim)
            mass_api.CreateMassAttr().Set(mass_val)
            mass_api.CreateDiagonalInertiaAttr().Set(
                Gf.Vec3f(float(i_perp), float(i_perp), float(i_axial))
            )

            # Capsule geometry child
            capsule_path = f"{body_path}/capsule"
            capsule = UsdGeom.Capsule.Define(stage, capsule_path)
            capsule.GetRadiusAttr().Set(r)
            capsule.GetHeightAttr().Set(seg_len)
            capsule.GetAxisAttr().Set("Z")
            UsdShade.MaterialBindingAPI(capsule.GetPrim()).Bind(branch_mat)

            # Collision on capsule child
            UsdPhysics.CollisionAPI.Apply(capsule.GetPrim())
            col_includes.AddTarget(capsule_path)

            total_links += 1

        # --- Create joints ---
        for cn in nodes_in_part:
            parent_node = fathers[cn]
            p0 = positions[parent_node]
            p1 = positions[cn]
            seg_len = float(np.linalg.norm(p1 - p0))
            if seg_len < 1e-6:
                continue
            child_half = seg_len / 2.0
            q_child = capsule_quat[cn]
            body_path = f"{artic_path}/b_{cn:04d}"

            parent_capsule = find_parent_capsule(cn)

            # Is this the root link of the partition?
            is_partition_root = (cn == nodes_in_part[0])
            # Or the parent capsule is outside this partition
            if parent_capsule is not None and parent_capsule not in nodes_set:
                is_partition_root = True

            joint_path = f"{artic_path}/j_{cn:04d}"

            if is_partition_root:
                # Fix to world — anchor at the base of this capsule
                anchor_pos = p0
                joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
                joint.CreateBody1Rel().SetTargets([body_path])
                # body0 omitted -> world
                joint.CreateLocalPos0Attr().Set(
                    Gf.Vec3f(float(anchor_pos[0]), float(anchor_pos[1]), float(anchor_pos[2]))
                )
                joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, -child_half))
                joint.CreateLocalRot0Attr().Set(
                    Gf.Quatf(float(q_child[0]), float(q_child[1]),
                             float(q_child[2]), float(q_child[3]))
                )
                joint.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
                total_joints += 1

            elif parent_capsule is None:
                # No parent capsule at all — fix to world
                anchor_pos = p0
                joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
                joint.CreateBody1Rel().SetTargets([body_path])
                joint.CreateLocalPos0Attr().Set(
                    Gf.Vec3f(float(anchor_pos[0]), float(anchor_pos[1]), float(anchor_pos[2]))
                )
                joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, -child_half))
                joint.CreateLocalRot0Attr().Set(
                    Gf.Quatf(float(q_child[0]), float(q_child[1]),
                             float(q_child[2]), float(q_child[3]))
                )
                joint.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
                total_joints += 1

            else:
                # Fixed joint to parent — keeps plant rigid so fruit
                # positions match the JSON data exactly.
                parent_body_path = f"{artic_path}/b_{parent_capsule:04d}"
                pp = fathers[parent_capsule]
                par_len = float(np.linalg.norm(positions[parent_capsule] - positions[pp]))
                par_half = par_len / 2.0 if par_len > 1e-6 else 0.01

                q_parent = capsule_quat[parent_capsule]
                q_rel = _quat_mul(_quat_inv(q_child), q_parent)

                joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
                joint.CreateBody0Rel().SetTargets([parent_body_path])
                joint.CreateBody1Rel().SetTargets([body_path])
                joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, par_half))
                joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, -child_half))
                joint.CreateLocalRot0Attr().Set(Gf.Quatf(1, 0, 0, 0))
                joint.CreateLocalRot1Attr().Set(
                    Gf.Quatf(float(q_rel[0]), float(q_rel[1]),
                             float(q_rel[2]), float(q_rel[3]))
                )
                total_joints += 1

    print(f"[PlantPhysics] Created {len(partitions)} articulations "
          f"with {total_links} total links and {total_joints} joints")

    # ------------------------------------------------------------------
    # Create visual stems for all target fruits.
    # Each fruit sphere is a dynamic RigidObject in the scene script;
    # here we add thin brown stems connecting them to the branch.
    # Stems are named fruit_stem_0, fruit_stem_1, ... (by list position).
    # ------------------------------------------------------------------
    if target_fruit_indices:
        # Stem material (brown) — shared across all stems
        stem_mat = UsdShade.Material.Define(stage, "/World/Plant/StemMaterial")
        stem_sh = UsdShade.Shader.Define(stage, "/World/Plant/StemMaterial/Shader")
        stem_sh.CreateIdAttr("UsdPreviewSurface")
        stem_sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(0.30, 0.20, 0.08)
        )
        stem_sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
        stem_mat.CreateSurfaceOutput().ConnectToSource(
            stem_sh.ConnectableAPI(), "surface"
        )

        target_set = set(target_fruit_indices)
        fruit_idx = 0
        for i in range(n_nodes):
            if fruit_radii[i] <= 0:
                continue
            if fruit_idx in target_set:
                list_position = target_fruit_indices.index(fruit_idx)
                parent_node = fathers[i]
                if parent_node >= 0:
                    p_parent = positions[parent_node]
                    # Use custom world position if provided (hanging cluster)
                    if fruit_world_positions is not None:
                        p_fruit = np.array(fruit_world_positions[list_position],
                                           dtype=np.float64)
                    else:
                        p_fruit = positions[i]
                    direction = p_fruit - p_parent
                    seg_len = float(np.linalg.norm(direction))
                    if seg_len > 1e-6:
                        midpoint = (p_parent + p_fruit) / 2.0
                        q_stem = _quat_z_to(direction)

                        stem_path = f"/World/Plant/fruit_stem_{list_position}"
                        stem_xform = UsdGeom.Xform.Define(stage, stem_path)
                        stem_xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
                            Gf.Vec3d(float(midpoint[0]), float(midpoint[1]),
                                     float(midpoint[2]))
                        )
                        stem_xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
                            Gf.Quatd(float(q_stem[0]), float(q_stem[1]),
                                      float(q_stem[2]), float(q_stem[3]))
                        )
                        stem_cap = UsdGeom.Capsule.Define(stage, f"{stem_path}/capsule")
                        stem_cap.GetRadiusAttr().Set(0.002)
                        stem_cap.GetHeightAttr().Set(seg_len)
                        stem_cap.GetAxisAttr().Set("Z")
                        UsdShade.MaterialBindingAPI(stem_cap.GetPrim()).Bind(stem_mat)
                        print(f"[PlantPhysics] Created stem {list_position} for fruit {fruit_idx}")
            fruit_idx += 1

    print(f"[PlantPhysics] Grand total: {total_links} links, {total_joints} joints")

    # ------------------------------------------------------------------
    # Compute fruit attachment info for all target fruits
    # ------------------------------------------------------------------
    fruit_attachments = []
    if target_fruit_indices:
        target_set = set(target_fruit_indices)
        fruit_idx = 0
        for i in range(n_nodes):
            if fruit_radii[i] <= 0:
                continue
            if fruit_idx in target_set:
                parent_node = fathers[i]
                attachment = None
                if parent_node >= 0 and parent_node in capsule_set:
                    artic_k = node_to_artic[parent_node]
                    parent_body_path = f"/World/Plant/artic_{artic_k}/b_{parent_node:04d}"
                    pp = fathers[parent_node]
                    par_len = float(np.linalg.norm(positions[parent_node] - positions[pp]))
                    par_half = par_len / 2.0 if par_len > 1e-6 else 0.01
                    q_parent = capsule_quat[parent_node]
                    offset = positions[parent_node] - positions[i]
                    attachment = {
                        "fruit_meta_index": fruit_idx,
                        "parent_body_path": parent_body_path,
                        "parent_half_height": par_half,
                        "parent_quat": q_parent.tolist(),
                        "fruit_world_pos": positions[i].tolist(),
                        "offset_fruit_to_branch_tip": offset.tolist(),
                    }
                fruit_attachments.append(attachment)
            fruit_idx += 1

    return {"fruit_attachments": fruit_attachments}
