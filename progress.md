# Running Isaac Lab Headless on odin2 Server

This document outlines the complete setup process and experiments run with Isaac Lab inside a Docker container on the odin2 server in headless mode.

## Server Environment

- **Server**: odin2
- **OS**: Ubuntu 24.04.2 LTS (Noble Numbat)
- **Kernel**: 5.14.0-427.40.1.el9_4.x86_64
- **CPU**: AMD EPYC 9654 96-Core Processor
- **GPU**: 3x NVIDIA A100 80GB PCIe
- **Isaac Sim Version**: 5.1.0
- **NVIDIA Driver**: 570.211.01

## Minimum Requirements

- **NVIDIA Driver**: >= 570.169
- **Docker**: With nvidia-container-toolkit configured
- **GPU**: NVIDIA GPU with Vulkan 1.1 support

---

## Complete Setup Instructions

### Phase 1: System Prerequisites (One-time)

#### 1.1 Check NVIDIA Driver Version

```bash
nvidia-smi
```

If driver version is < 570.169, upgrade is required for Isaac Sim 5.1.

#### 1.2 Install/Configure nvidia-container-toolkit

```bash
# Add NVIDIA container toolkit repository (RHEL/CentOS)
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo

# Install nvidia-container-toolkit
sudo dnf install -y nvidia-container-toolkit

# Configure the runtime
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker
sudo systemctl restart docker
```

### Phase 2: Isaac Lab Docker Setup

#### 2.1 Clone the Repository

```bash
git clone https://github.com/isaac-sim/IsaacLab.git /ext_hdd2/nhkoh/IsaacLab
cd /ext_hdd2/nhkoh/IsaacLab
```

#### 2.2 Configure for Headless Operation

Create/edit `docker/.container.cfg`:

```ini
[X11]
x11_forwarding_enabled = 0
```

#### 2.3 Build and Start the Docker Container

```bash
cd /ext_hdd2/nhkoh/IsaacLab
python docker/container.py start
```

This command:
- Pulls the Isaac Sim base image if not present
- Builds the `isaac-lab-base` Docker image
- Starts the container in detached mode

#### 2.4 Enter the Container

```bash
python docker/container.py enter
```

---

## Experiments Run

### Experiment 1: Cartpole (Classic Control)

**Training:**
```bash
isaaclab -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Cartpole-v0 \
    --num_envs 4096 \
    --max_iterations 150 \
    --headless
```

**Configuration:**
- Environment: Classic cartpole balance
- Parallel Environments: 4096
- Iterations: 149
- Algorithm: PPO (RSL-RL)

**Output:** `logs/rsl_rl/cartpole/2026-01-18_02-34-08/`

---

### Experiment 2: Franka Reach

**Training:**
```bash
isaaclab -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Reach-Franka-v0 \
    --num_envs 64 \
    --max_iterations 100 \
    --headless
```

**Configuration:**
- Environment: Franka arm reach to target pose
- Parallel Environments: 64
- Iterations: 99
- Algorithm: PPO (RSL-RL)

**Output:** `logs/rsl_rl/franka_reach/2026-01-18_05-34-53/`

---

### Experiment 3: Franka Lift Cube (Short Run)

**Training:**
```bash
isaaclab -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Lift-Cube-Franka-v0 \
    --num_envs 512 \
    --max_iterations 500 \
    --headless
```

**Configuration:**
- Environment: Franka arm lift cube to target position
- Parallel Environments: 512
- Iterations: 499
- Algorithm: PPO (RSL-RL)

**Output:** `logs/rsl_rl/franka_lift/2026-01-18_05-39-30/`

---

### Experiment 4: Franka Lift Cube (Extended Run)

**Training:**
```bash
isaaclab -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Lift-Cube-Franka-v0 \
    --num_envs 512 \
    --max_iterations 3000 \
    --headless
```

**Configuration:**
- Environment: Franka arm lift cube to target position
- Parallel Environments: 512
- Iterations: 2999
- Algorithm: PPO (RSL-RL)
- Checkpoints saved every 50 iterations

**Output:** `logs/rsl_rl/franka_lift/2026-01-18_05-48-55/`

---

## Evaluation and Video Recording

### Play/Evaluate a Trained Policy

```bash
isaaclab -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Lift-Cube-Franka-Play-v0 \
    --num_envs 1 \
    --video \
    --video_length 500 \
    --headless \
    --enable_cameras
```

**Key flags:**
- `--headless`: Run without GUI
- `--enable_cameras`: Enable offscreen rendering (required for video in headless mode)
- `--video`: Enable video recording
- `--video_length 500`: Number of frames to record

### Copy Outputs from Container to Host

From the host machine (outside container):

```bash
# Copy Franka Lift results
docker cp isaac-lab-base:/workspace/isaaclab/logs/rsl_rl/franka_lift/ /ext_hdd2/nhkoh/IsaacLab/output_videos/franka_lift/

# Copy Franka Reach results
docker cp isaac-lab-base:/workspace/isaaclab/logs/rsl_rl/franka_reach/ /ext_hdd2/nhkoh/IsaacLab/output_videos/franka_reach/

# Copy Cartpole results
docker cp isaac-lab-base:/workspace/isaaclab/logs/rsl_rl/cartpole/ /ext_hdd2/nhkoh/IsaacLab/output_videos/
```

---

## Container Management

### Stop the Container

```bash
python docker/container.py stop
```

### Re-enter Existing Container

```bash
python docker/container.py enter
```

### Check Container Status

```bash
docker ps -a | grep isaac
```

---

## Output File Structure

| Content | Container Path | Host Path After Copy |
|---------|----------------|---------------------|
| Training logs | `/workspace/isaaclab/logs/rsl_rl/<task>/<timestamp>/` | `output_videos/<task>/<timestamp>/` |
| Model checkpoints | `.../<timestamp>/model_*.pt` | Same structure |
| Exported policies | `.../<timestamp>/exported/policy.pt, policy.onnx` | Same structure |
| Videos | `.../<timestamp>/videos/play/` | Same structure |
| TensorBoard events | `.../<timestamp>/events.out.tfevents.*` | Same structure |
| Config files | `.../<timestamp>/params/env.yaml, agent.yaml` | Same structure |

---

## Experience Files Reference

Isaac Lab uses different experience (.kit) files based on the mode:

| Mode | Experience File |
|------|-----------------|
| GUI | `isaaclab.python.kit` |
| GUI + Cameras | `isaaclab.python.rendering.kit` |
| Headless (no render) | `isaaclab.python.headless.kit` |
| Headless + Cameras | `isaaclab.python.headless.rendering.kit` |

The correct experience file is automatically selected based on the `--headless` and `--enable_cameras` flags.

---

## Troubleshooting

### Issue 1: X11 Forwarding Error

**Error:**
```
KeyError: 'DISPLAY'
```

**Cause:** The container startup script tried to configure X11 forwarding, but no DISPLAY environment variable was set on the headless server.

**Solution:** Disable X11 forwarding in `docker/.container.cfg`:
```ini
[X11]
x11_forwarding_enabled = 0
```

### Issue 2: Vulkan Driver Incompatibility

**Error:**
```
VkResult: ERROR_INCOMPATIBLE_DRIVER
vkCreateInstance failed. Vulkan 1.1 is not supported, or your driver requires an update.
```

**Cause:** The host NVIDIA driver version (560.35.03) was older than the minimum required version (570.169) for Isaac Sim 5.1.

**Solution:** Upgrade the NVIDIA driver to version 570.211.01 or later.

### Issue 3: Docker GPU Access Lost After Driver Upgrade

**Error:**
```
Error response from daemon: could not select device driver "nvidia" with capabilities: [[gpu]]
```

**Cause:** After upgrading the NVIDIA driver, the nvidia-container-toolkit needed to be reconfigured.

**Solution:**
```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Issue 4: Video Recording Failed in Headless Mode

**Error:**
```
RuntimeError: Cannot render 'rgb_array' when the simulation render mode is 'NO_GUI_OR_RENDERING'.
```

**Cause:** The `--enable_cameras` flag was not passed, which is required for rendering in headless mode.

**Solution:** Always use `--enable_cameras` together with `--video` and `--headless`.
