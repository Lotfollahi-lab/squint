#!/usr/bin/env bash
# submit_mmb_smb_ablations_v18.sh
# -----------------------------------------------------------------------------
# Submits all 8 variants in
# `DATASET_VARIANTS["mmb0-1b_smb1-1b_1p-ablations-v18"]` as separate LSF
# jobs (one bsub per variant, one GPU exclusive each).
#
# Defaults: --queue training-parallel --group s10396.
#
# Sweep 18 design: SMALL-ARCH + NO-WARMUP + SAMPLER-{4,8} grid.
# 2 × 2 × 2 = 8 variants:
#   hidden   ∈ {32, 64}
#   codebook ∈ {(30, 30) 2-level, (30, 30, 30) 3-level}
#   sampler  ∈ {[4], [8]}
#
# All variants share a strictly-unsupervised spine with NO adversarial
# warmup (`+adv` only, alpha=1.0 from epoch 0), `_patch_dual_small`
# for uniform-width encoder MLP + GNN + both decoders, and codebook
# level-0 fixed at 30.
#
# Goal: find the smallest working architecture before scaling up. If
# the s18_v<N>_* numbers look promising, the next sweep can scale up
# hidden / codebook depth from a confirmed working baseline; if not,
# we know SQUINT needs more capacity than 64-d.
#
# Variant naming follows the `s<sweep>_v<variant>_<base>` convention
# (see the "Variant-name prefix convention" comment in run_squint.py).
#
# Concrete variant list (in submission order):
#   s18_v1_  small-h32  +rvq-both-30-30        +sampler4   # smallest
#   s18_v2_  small-h64  +rvq-both-30-30        +sampler4
#   s18_v3_  small-h32  +rvq-both-3level-30-30-30 +sampler4
#   s18_v4_  small-h64  +rvq-both-3level-30-30-30 +sampler4
#   s18_v5_  small-h32  +rvq-both-30-30        +sampler8
#   s18_v6_  small-h64  +rvq-both-30-30        +sampler8
#   s18_v7_  small-h32  +rvq-both-3level-30-30-30 +sampler8
#   s18_v8_  small-h64  +rvq-both-3level-30-30-30 +sampler8   # largest
#
# Usage:
#   bash examples/submit_mmb_smb_ablations_v18.sh [OPTIONS]
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

echo "[v18 ablations] LSF_QUEUE = $LSF_QUEUE"
echo "[v18 ablations] LSF_GROUP = $LSF_GROUP"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
exec bash "$SCRIPT_DIR/submit_dataset_sweep.sh" \
    "mmb0-1b_smb1-1b_1p-ablations-v18" \
    ${PASSTHROUGH_ARGS[@]+"${PASSTHROUGH_ARGS[@]}"}
