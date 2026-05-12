#!/usr/bin/env bash
# submit_mmb_smb_ablations_v13.sh
# -----------------------------------------------------------------------------
# Submits all 8 variants in
# `DATASET_VARIANTS["mmb0-1b_smb1-1b_1p-ablations-v13"]` as separate LSF
# jobs (one bsub per variant, one GPU exclusive each).
#
# Defaults: --queue training-parallel --group s10396.
#
# v13 design: CODEBOOK STRUCTURE exploration off the v3 spine
# `dualvq+wide+rvq-both+decoder-cov+adv+enc-deeper+adv-warmup10`.
# Probes RVQ depth (2 vs 3 levels), level-1+ size growth patterns,
# asymmetric cell-vs-niche depth, single-level VQ on cell, and
# composite VQ on cell.
#
# DESIGN CONSTRAINT: level-0 codebook size is FIXED at 30 across
# every variant (both branches). v13 deliberately does NOT ablate
# level-0 — it ablates everything ELSE about the codebook structure.
#
# Concrete variant list (in submission order):
#   1. +vq-cell-30   +rvq-niche-30-60-120     # single-level cell K=30
#   2. +rvq-cell-30-60   +rvq-niche-30-60-120 # 2-level cell modest L1
#   3. +rvq-cell-30-200  +rvq-niche-30-60-120 # 2-level cell BIG L1
#   4. +rvq-cell-30-30   +rvq-niche-30-60     # 2-level cell no L1 growth
#   5. +rvq-cell-30-90-270 +rvq-niche-30-60-120 # 3-level cell symmetric niche
#   6. +rvq-both-3level-30-30-30              # symmetric 3-level no growth
#   7. +cvq-cell-30-10   +rvq-niche-30-60-120 # composite VQ cell
#   8. +rvq-cell-30-90-270 +rvq-niche-30-90   # 3-level cell asymmetric niche
#
# Usage:
#   bash examples/submit_mmb_smb_ablations_v13.sh [OPTIONS]
#
# Options:
#   --queue / -q QUEUE   LSF queue (default: training-parallel).
#   --group / -g GROUP   LSF cost-code group (default: s10396).
#   --help  / -h         Print this usage block and exit.
#
# Other env overrides from submit_dataset_sweep.sh work too
# (DRY_RUN=1, LSF_WALL=..., LSF_MEM_MB=..., SQUINT_WITH_PEARSON=1, etc.).
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

echo "[v13 ablations] LSF_QUEUE = $LSF_QUEUE"
echo "[v13 ablations] LSF_GROUP = $LSF_GROUP"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
exec bash "$SCRIPT_DIR/submit_dataset_sweep.sh" \
    "mmb0-1b_smb1-1b_1p-ablations-v13" \
    ${PASSTHROUGH_ARGS[@]+"${PASSTHROUGH_ARGS[@]}"}
