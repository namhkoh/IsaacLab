# Running Isaac Lab Headless on odin2 Server

This document outlines the steps taken to successfully run Isaac Lab inside a Docker container on the odin2 server in headless mode with video recording.

## Server Environment

- **Server**: odin2
- **OS**: Ubuntu 24.04.2 LTS (Noble Numbat)
- **Kernel**: 5.14.0-427.40.1.el9_4.x86_64
- **CPU**: AMD EPYC 9654 96-Core Processor
- **GPU**: 3x NVIDIA A100 80GB PCIe
- **Isaac Sim Version**: 5.1.0

## Issues Encountered and Solutions

### 1. X11 Forwarding Error

**Error:**
```
KeyError: 'DISPLAY'
```

**Cause:** The container startup script tried to configure X11 forwarding, but no DISPLAY environment variable was set on the headless server.

**Solution:** Disable X11 forwarding in the container configuration file:

```bash
# Edit docker/.container.cfg
[X11]
x11_forwarding_enabled = 0
```

### 2. Vulkan Driver Incompatibility

**Error:**
```
VkResult: ERROR_INCOMPATIBLE_DRIVER
vkCreateInstance failed. Vulkan 1.1 is not supported, or your driver requires an update.
```

**Cause:** The host NVIDIA driver version (560.35.03) was older than the minimum required version (570.169) for Isaac Sim 5.1.

**Solution:** Upgrade the NVIDIA driver to version 570.211.01 or later.

### 3. Docker GPU Access Lost After Driver Upgrade

**Error:**
```
Error response from daemon: could not select device driver "nvidia" with capabilities: [[gpu]]
```

**Cause:** After upgrading the NVIDIA driver, the nvidia-container-toolkit needed to be reconfigured.

**Solution:**
```bash
# Add NVIDIA container toolkit repository (if not present)
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo

# Install nvidia-container-toolkit
sudo dnf install -y nvidia-container-toolkit

# Configure the runtime
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker
sudo systemctl restart docker
```

### 4. Video Recording Failed in Headless Mode

**Error:**
```
RuntimeError: Cannot render 'rgb_array' when the simulation render mode is 'NO_GUI_OR_RENDERING'.
```

**Cause:** The `--enable_cameras` flag was not passed, which is required for rendering in headless mode.

**Solution:** Use the correct combination of flags for headless video recording.

## Working Commands

### Start the Container

```bash
# From the IsaacLab directory
python docker/container.py start
```

### Enter the Container

```bash
python docker/container.py enter
```

### Run Simulation with Video Recording (Headless)

```bash
# Inside the container
isaaclab -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Cartpole-v0 \
    --num_envs 1 \
    --video \
    --video_length 500 \
    --headless \
    --enable_cameras
```

Key flags:
- `--headless`: Run without GUI
- `--enable_cameras`: Enable offscreen rendering (required for video in headless mode)
- `--video`: Enable video recording
- `--video_length 500`: Record 500 frames

### Copy Videos from Container to Host

```bash
# From the host machine
docker cp isaac-lab-base:/workspace/isaaclab/logs/rsl_rl/cartpole/ /ext_hdd2/nhkoh/IsaacLab/output_videos/
```

## Output Locations

- **Videos**: `/workspace/isaaclab/logs/rsl_rl/<task>/<timestamp>/videos/play/`
- **Logs**: `/tmp/isaaclab/logs/`
- **Model Checkpoints**: `/workspace/isaaclab/logs/rsl_rl/<task>/<timestamp>/`

## Experience Files

Isaac Lab uses different experience (.kit) files based on the mode:

| Mode | Experience File |
|------|-----------------|
| GUI | `isaaclab.python.kit` |
| GUI + Cameras | `isaaclab.python.rendering.kit` |
| Headless (no render) | `isaaclab.python.headless.kit` |
| Headless + Cameras | `isaaclab.python.headless.rendering.kit` |

The correct experience file is automatically selected based on the `--headless` and `--enable_cameras` flags.

## Minimum Requirements

- **NVIDIA Driver**: >= 570.169
- **Docker**: With nvidia-container-toolkit configured
- **GPU**: NVIDIA GPU with Vulkan 1.1 support
