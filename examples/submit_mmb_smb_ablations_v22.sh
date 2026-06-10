#!/usr/bin/env bash
# submit_mmb_smb_ablations_v22.sh
# -----------------------------------------------------------------------------
# Submits all 8 variants in
# `DATASET_VARIANTS["mmb0-1b_smb1-1b_1p-ablations-v22"]` as separate LSF
# jobs (one bsub per variant, one GPU exclusive each).
#
# Defaults: --queue training-parallel --group s10396.
#
# Sweep 22 design: BATCH_SIZE × LEARNING_RATE grid on the s17_v2 spine
# (`+wide+rvq-both+decoder-cov+adv`, niche NMI 0.695 at the historic
# default batch=256 / lr=5e-4). Tests whether SQUINT can train at
# bigger batch sizes without quality regression.
#
# Grid (8 variants):
#
#   batch\lr        sqrt-scaled lr     linear-scaled lr
#   ------------    --------------     ----------------
#    256 (default)   5e-4    (v1)       5e-4             (= baseline)
#   1024            1e-3    (v2)       2e-3    (v5)
#   2048            1.4e-3  (v3)       4e-3    (v6)
#   4096            2e-3    (v4)       8e-3    (v7)
#
#   + v8: batch=256, lr=1e-3 (LR-only control — does the extra LR
#         help even WITHOUT a bigger batch?)
#
# Reading the grid:
#   - v1 → v2 → v3 → v4 traces sqrt-scaling at growing batch. If
#     niche NMI tracks v1 across this diagonal, SQUINT scales cleanly.
#   - v5/v6/v7 are linear-scaling pairs of v2/v3/v4. Compare side by
#     side to pick the safer LR rule for SQUINT's adversarial loop.
#   - v8 isolates "does 2x lr alone help at default batch?" — useful
#     for separating "bigger batches help" from "we were under-lr'd".
#
# Common failure mode: linear scaling at 16x batch (v7) destabilises
# the adversarial discriminator, val_loss spikes, early stopping fires.
# If v7 collapses, that's the lr cliff — try a longer warmup or
# smaller scaling rule.
#
# Usage:
#   bash examples/submit_mmb_smb_ablations_v22.sh [OPTIONS]
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

echo "[v22 ablations] LSF_QUEUE = $LSF_QUEUE"
echo "[v22 ablations] LSF_GROUP = $LSF_GROUP"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
exec bash "$SCRIPT_DIR/submit_dataset_sweep.sh" \
    "mmb0-1b_smb1-1b_1p-ablations-v22" \
    ${PASSTHROUGH_ARGS[@]+"${PASSTHROUGH_ARGS[@]}"}
