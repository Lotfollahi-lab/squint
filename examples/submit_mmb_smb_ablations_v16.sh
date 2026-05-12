#!/usr/bin/env bash
# submit_mmb_smb_ablations_v16.sh
# -----------------------------------------------------------------------------
# Submits all 8 variants in
# `DATASET_VARIANTS["mmb0-1b_smb1-1b_1p-ablations-v16"]` as separate LSF
# jobs (one bsub per variant, one GPU exclusive each).
#
# Defaults: --queue training-parallel --group s10396.
#
# v16 design: ENCODER / DECODER DIM GRID with codebook FIXED at
# (30, 30, 30). 4 hidden dims (<256) × 2 encoder depths = 8 variants.
# All variants use `_patch_dual_small(hidden=N)` which sets encoder
# MLP + GNN + BOTH decoders to the same width N — uniform-width
# architectures, parameter-counts comparable across the grid.
#
# Concrete variant list (in submission order):
#   1. +small-h32                  # h=32,  1-layer MLP
#   2. +small-h64                  # h=64,  1-layer MLP
#   3. +small-h128                 # h=128, 1-layer MLP
#   4. +small-h192                 # h=192, 1-layer MLP
#   5. +small-h32  +enc-deeper     # h=32,  3-layer MLP [32, 32, 32]
#   6. +small-h64  +enc-deeper     # h=64,  3-layer MLP [64, 64, 64]
#   7. +small-h128 +enc-deeper     # h=128, 3-layer MLP [128, 128, 128]
#   8. +small-h192 +enc-deeper     # h=192, 3-layer MLP [192, 192, 192]
#
# Codebook is held at (30, 30, 30) symmetric 3-level RVQ on both
# branches — v16 deliberately does NOT ablate the codebook (that's v13).
#
# Usage:
#   bash examples/submit_mmb_smb_ablations_v16.sh [OPTIONS]
#
# Options:
#   --queue / -q QUEUE   LSF queue (default: training-parallel).
#   --group / -g GROUP   LSF cost-code group (default: s10396).
#   --help  / -h         Print this usage block and exit.
# -----------------------------------------------------------------------------

set -euo pipefail

DEFAULT_QUEUE="training-parallel"
DEFAULT_GROUP="s10396"

QUEUE="${LSF_QUEUE:-$DEFAULT_QUEUE}"
GROUP="${LSF_GROUP:-$DEFAULT_GROUP}"

PASSTHROUGH_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --queue|-q)
            shift
            if [[ $# -eq 0 ]]; then
                echo "ERROR: --queue / -q requires a value." >&2
                exit 2
            fi
            QUEUE="$1"
            shift
            ;;
        --queue=*)
            QUEUE="${1#--queue=}"
            shift
            ;;
        --group|-g)
            shift
            if [[ $# -eq 0 ]]; then
                echo "ERROR: --group / -g requires a value." >&2
                exit 2
            fi
            GROUP="$1"
            shift
            ;;
        --group=*)
            GROUP="${1#--group=}"
            shift
            ;;
        --help|-h)
            awk '/^[^#]/ {exit} {print}' "$0"
            exit 0
            ;;
        *)
            PASSTHROUGH_ARGS+=("$1")
            shift
            ;;
    esac
done

export LSF_QUEUE="$QUEUE"
export LSF_GROUP="$GROUP"

echo "[v16 ablations] LSF_QUEUE = $LSF_QUEUE"
echo "[v16 ablations] LSF_GROUP = $LSF_GROUP"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
exec bash "$SCRIPT_DIR/submit_dataset_sweep.sh" \
    "mmb0-1b_smb1-1b_1p-ablations-v16" \
    ${PASSTHROUGH_ARGS[@]+"${PASSTHROUGH_ARGS[@]}"}
