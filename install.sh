#!/usr/bin/env bash
# Install PyTorch matching the host's CUDA driver, then scgg in editable mode.
#
# Pip cannot express "pick the matching CUDA wheel" in pyproject.toml, so we
# detect the driver here and install torch from the right wheel index BEFORE
# resolving scgg's other deps.
#
# Usage:
#     bash install.sh                       # auto-detect CUDA, install scgg -e
#     bash install.sh --cuda 12.4           # force a specific CUDA wheel
#     bash install.sh --cpu                 # CPU-only torch
#     bash install.sh --skip-torch          # assume torch is already installed
#
# Exit on any error.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

FORCE_CUDA=""
SKIP_TORCH="0"
USE_CPU="0"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cuda) FORCE_CUDA="$2"; shift 2 ;;
        --cpu) USE_CPU="1"; shift ;;
        --skip-torch) SKIP_TORCH="1"; shift ;;
        -h|--help)
            sed -n '/^# Usage:/,/^# Exit/p' "$0" | sed 's/^# //'
            exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

# Build the wheel index URL.
if [[ "$SKIP_TORCH" == "1" ]]; then
    echo "[install.sh] --skip-torch set; assuming torch is already installed."
elif [[ "$USE_CPU" == "1" ]]; then
    INDEX="https://download.pytorch.org/whl/cpu"
    TAG="cpu"
elif [[ -n "$FORCE_CUDA" ]]; then
    case "$FORCE_CUDA" in
        12.1|121) INDEX="https://download.pytorch.org/whl/cu121"; TAG="cu121" ;;
        12.4|124) INDEX="https://download.pytorch.org/whl/cu124"; TAG="cu124" ;;
        12.6|126) INDEX="https://download.pytorch.org/whl/cu126"; TAG="cu126" ;;
        12.8|128) INDEX="https://download.pytorch.org/whl/cu128"; TAG="cu128" ;;
        *) echo "[install.sh] unsupported --cuda value: $FORCE_CUDA"; exit 1 ;;
    esac
else
    # Auto-detect.
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "[install.sh] nvidia-smi not found — falling back to CPU torch."
        INDEX="https://download.pytorch.org/whl/cpu"
        TAG="cpu"
    else
        # Parse "CUDA Version: 12.4" from the top-right of nvidia-smi output.
        CUDA_VER="$(nvidia-smi | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | head -n1 | awk '{print $3}')"
        if [[ -z "$CUDA_VER" ]]; then
            echo "[install.sh] could not parse CUDA version from nvidia-smi — falling back to CPU."
            INDEX="https://download.pytorch.org/whl/cpu"
            TAG="cpu"
        else
            # Pick the highest pytorch wheel CUDA <= driver CUDA.
            MAJOR=${CUDA_VER%%.*}
            MINOR=${CUDA_VER##*.}
            VAL=$((MAJOR * 10 + MINOR))
            if   (( VAL >= 128 )); then TAG="cu128"
            elif (( VAL >= 126 )); then TAG="cu126"
            elif (( VAL >= 124 )); then TAG="cu124"
            elif (( VAL >= 121 )); then TAG="cu121"
            else
                echo "[install.sh] driver CUDA $CUDA_VER is below the lowest supported wheel (12.1)."
                echo "[install.sh] falling back to CPU torch."
                TAG="cpu"
            fi
            INDEX="https://download.pytorch.org/whl/${TAG}"
            echo "[install.sh] detected driver CUDA $CUDA_VER -> torch wheel index: $TAG"
        fi
    fi
fi

if [[ "$SKIP_TORCH" != "1" ]]; then
    echo "[install.sh] installing torch from $INDEX"
    pip install --upgrade pip setuptools wheel
    pip install --upgrade --index-url "$INDEX" torch
fi

echo "[install.sh] installing scgg (editable) from $REPO_DIR"
pip install -e "$REPO_DIR"

echo "[install.sh] verifying torch + scgg ..."
python - <<'PY'
import torch
print(f"torch       : {torch.__version__}")
print(f"cuda built  : {torch.version.cuda}")
print(f"cuda runtime: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device count: {torch.cuda.device_count()}")
    print(f"device name : {torch.cuda.get_device_name(0)}")
import scgg
print(f"scgg        : {scgg.__version__}")
PY

echo "[install.sh] done."
