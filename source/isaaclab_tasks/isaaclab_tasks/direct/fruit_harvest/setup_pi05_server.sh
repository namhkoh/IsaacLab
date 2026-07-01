#!/bin/bash
# ==============================================================================
# Setup script for pi0.5 (openpi) server in WSL2 Ubuntu 22.04.
#
# This script:
#   1. Clones the openpi repository
#   2. Installs dependencies via uv
#   3. Starts the LIBERO policy server on port 8000
#
# The server will be accessible from Windows at localhost:8000
# (WSL2 automatically forwards ports).
#
# Usage (run inside WSL2):
#   chmod +x setup_pi05_server.sh
#   ./setup_pi05_server.sh [--env LIBERO|DROID] [--port 8000]
#
# Requirements:
#   - WSL2 Ubuntu 22.04 with NVIDIA GPU drivers (nvidia-smi must work)
#   - At least 16GB free disk space (checkpoint ~14GB)
#   - At least 8GB VRAM
# ==============================================================================

set -euo pipefail

# Defaults
ENV_NAME="${1:---env}"
ENV_VALUE="LIBERO"
PORT=8000

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --env)
            ENV_VALUE="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

echo "============================================"
echo "  pi0.5 Server Setup (openpi)"
echo "  Environment: ${ENV_VALUE}"
echo "  Port:        ${PORT}"
echo "============================================"

# Check GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "[ERROR] nvidia-smi not found. Install NVIDIA GPU drivers for WSL2."
    echo "  See: https://docs.nvidia.com/cuda/wsl-user-guide/"
    exit 1
fi
echo "[OK] GPU detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "[Setup] Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    # Also add to bashrc for future sessions
    if ! grep -q 'astral' ~/.bashrc 2>/dev/null; then
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    fi
fi
echo "[OK] uv version: $(uv --version)"

# Clone openpi if not already present
OPENPI_DIR="$HOME/openpi"
if [ ! -d "$OPENPI_DIR" ]; then
    echo "[Setup] Cloning openpi repository..."
    GIT_LFS_SKIP_SMUDGE=1 git clone --recurse-submodules \
        https://github.com/Physical-Intelligence/openpi.git "$OPENPI_DIR"
else
    echo "[OK] openpi already cloned at $OPENPI_DIR"
    cd "$OPENPI_DIR"
    echo "[Setup] Pulling latest changes..."
    GIT_LFS_SKIP_SMUDGE=1 git pull --recurse-submodules || true
fi

cd "$OPENPI_DIR"

# Install dependencies
echo "[Setup] Installing Python dependencies (this may take a few minutes)..."
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

echo ""
echo "============================================"
echo "  Starting pi0.5 server"
echo "  Environment: ${ENV_VALUE}"
echo "  Port:        ${PORT}"
echo "  Checkpoint will auto-download (~14GB)"
echo "============================================"
echo ""
echo "Server will be accessible from Windows at:"
echo "  ws://localhost:${PORT}"
echo ""
echo "Press Ctrl+C to stop the server."
echo ""

# Start the server
# The checkpoint will be auto-downloaded on first run
uv run scripts/serve_policy.py --env "$ENV_VALUE" --port "$PORT"
