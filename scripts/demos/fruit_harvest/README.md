# Fruit Harvest Demo

A GPU-accelerated fruit-harvesting simulation using Isaac Lab. A Franka Panda robot sequentially picks tomato-like fruits from a Cosserat-rod plant and places them in a basket.

## Prerequisites

- **NVIDIA GPU** with CUDA support (tested with CUDA 12.8)
- **Isaac Sim 4.x or 5.x** pre-built binaries ([download](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/download.html))
- **Miniconda** ([install](https://www.anaconda.com/docs/getting-started/miniconda/main/))
- **Plant data**: The demo loads plant geometry from the [Plant-Robot-Interaction](https://github.com/namhkoh/-Plant-Robot-Interaction) repo. Clone it alongside IsaacLab so that the relative path resolves:
  ```
  research/
  ├── IsaacLab/                         # this repo
  │   └── scripts/demos/fruit_harvest/  # this directory
  └── Plant_Robot_Interaction-master/
      └── data/plant/fruit-001.json     # plant data
  ```

## Environment Setup

### 1. Install Isaac Sim

Download and unzip the Isaac Sim pre-built binaries. We'll refer to the unzipped location as `ISAACSIM_PATH`.

```batch
:: Windows — set the path to your Isaac Sim install
set ISAACSIM_PATH=C:\isaacsim
```
```bash
# Linux
export ISAACSIM_PATH="${HOME}/isaacsim"
```

Verify it works:

```batch
:: Windows
"%ISAACSIM_PATH%\python.bat" -c "print('Isaac Sim OK')"
```
```bash
# Linux
${ISAACSIM_PATH}/python.sh -c "print('Isaac Sim OK')"
```

### 2. Clone IsaacLab and create the symlink

```batch
:: Windows (run Command Prompt as Administrator for mklink)
cd D:\research
git clone https://github.com/namhkoh/IsaacLab.git
cd IsaacLab
git checkout koh-dev/tomato
mklink /D _isaac_sim %ISAACSIM_PATH%
```
```bash
# Linux
cd ~/research
git clone https://github.com/namhkoh/IsaacLab.git
cd IsaacLab
git checkout koh-dev/tomato
ln -s ${ISAACSIM_PATH} _isaac_sim
```

### 3. Create the `env_isaaclab` conda environment

Isaac Lab provides a one-command conda setup. The Python version must match Isaac Sim (3.10 for Isaac Sim 4.x, 3.11 for Isaac Sim 5.x).

```batch
:: Windows
isaaclab.bat --conda env_isaaclab
```
```bash
# Linux
./isaaclab.sh --conda env_isaaclab
```

Activate it:

```bash
conda activate env_isaaclab
```

### 4. Install Isaac Lab extensions

This installs all Isaac Lab source extensions (including `isaaclab_assets` which provides the Franka Panda config) and optionally RL frameworks:

```batch
:: Windows — install all extensions + RL frameworks
isaaclab.bat --install
```
```bash
# Linux
./isaaclab.sh --install
```

To install only specific frameworks (e.g. for PPO training):

```batch
isaaclab.bat --install rl_games
```

### 5. Clone Plant-Robot-Interaction data

```bash
cd D:\research   # or ~/research on Linux
git clone https://github.com/namhkoh/-Plant-Robot-Interaction.git Plant_Robot_Interaction-master
```

The fruit harvest script expects plant data at `../../../Plant_Robot_Interaction-master/data/plant/fruit-001.json` relative to its own location.

### 6. Verify the setup

```batch
:: Windows (from IsaacLab root, with env_isaaclab activated)
isaaclab.bat -p scripts/demos/fruit_harvest/fruit_harvest_scene.py
```
```bash
# Linux
./isaaclab.sh -p scripts/demos/fruit_harvest/fruit_harvest_scene.py
```

You should see the Franka Panda robot, a green plant with tomato fruits, and the harvest sequence begin.

## Quick Start

From the IsaacLab root directory:

```bash
# Windows
isaaclab.bat -p scripts/demos/fruit_harvest/fruit_harvest_scene.py

# Linux
./isaaclab.sh -p scripts/demos/fruit_harvest/fruit_harvest_scene.py
```

The simulation will:
1. Build the plant (Cosserat rod articulations with ~200 branch capsules)
2. Spawn 4 tomato fruits at branch positions with varied sizes, colors, and star-shaped calyxes
3. Warm up physics (100 steps to stabilize positions)
4. Sequentially harvest each fruit: **APPROACH → GRASP → RETRACT → DELIVER → RELEASE**
5. Idle with all fruits in the basket

## Command-Line Options

| Flag | Default | Description |
|------|---------|-------------|
| `--num_envs` | `1` | Number of parallel environments |
| `--headless` | off | Run without GUI |
| `--device` | `cuda:0` | Simulation device |

Example headless run:
```bash
isaaclab.bat -p scripts/demos/fruit_harvest/fruit_harvest_scene.py --headless
```

## Files

| File | Description |
|------|-------------|
| `fruit_harvest_scene.py` | Main simulation script — scene config, IK loop, force model, basket/calyx geometry |
| `plant_physics.py` | Builds plant as multi-articulation Featherstone sub-trees (≤52 links each) |
| `plant_metadata.json` | Pre-computed fruit centers, bounding box, mesh stats from plant data |
| `generate_plant_mesh.py` | Utility to regenerate OBJ meshes and metadata from plant JSON |
| `train_ppo.py` | PPO reinforcement learning training script for the harvest task |
| `fruit_*.obj` | Per-fruit OBJ meshes |
| `plant_branches.obj` | Branch geometry mesh |
| `plant_fruits.obj` | Combined fruit geometry mesh |

## Harvest Sequence

Each fruit goes through 5 phases:

1. **APPROACH** (up to 1500 steps) — IK drives the fingertip toward the fruit. Converges early if within 15mm.
2. **GRASP** (400 steps) — Gripper closes around the fruit. Physics-based contact holds it.
3. **RETRACT** (800 steps) — Spring-damper force switches to track the fingertip. The arm pulls the fruit off the branch. The visual stem hides once displacement exceeds 3cm.
4. **DELIVER** (800 steps) — Arm carries the fruit to above the basket rim.
5. **RELEASE** (300 steps) — Gripper opens, spring force deactivates, fruit drops into the basket under gravity.

After all 4 fruits are harvested, the simulation idles.

## Key Parameters

Tunable constants in `fruit_harvest_scene.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `HARVEST_INDICES` | `[6, 4, 2, 0]` | Which metadata fruits to harvest, in order |
| `FRUIT_SPECS` | per-fruit | Radius (0.023–0.030m), mass (0.012–0.022kg), color |
| `GRIPPER_CLOSE` | `0.018` | Finger position when closed (must be < smallest fruit radius) |
| `K_SPRING` | `100.0` | Spring stiffness for fruit transport (N/m) |
| `HOLD_ALPHA` | `0.8` | Position correction strength for held fruits (0=free, 1=locked) |
| `BASKET_CENTER` | `(0.45, 0.15)` | Basket XY position in world frame |

## Architecture

- **Position-only IK** via `DifferentialIKController` (DLS method, no orientation constraint)
- **Body-offset Jacobian** adjustment for fingertip center (panda_hand → fingertip = 0.107m)
- **World→base frame** Jacobian transformation for correct IK in non-identity root poses
- **Spring-damper transport**: `F = K*(target - pos) - D*vel + gravity_comp`, clamped to MAX_FORCE
- **Soft position hold**: Non-transported fruits blended 80% toward branch positions each step
- **Collision groups**: Fruits filtered against branches, active against robot/basket/ground/each other
- **Featherstone articulations**: Plant partitioned into sub-trees of ≤52 links for PhysX solver limits
