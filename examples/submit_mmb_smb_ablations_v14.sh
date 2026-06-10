#!/usr/bin/env bash
# submit_mmb_smb_ablations_v14.sh
# -----------------------------------------------------------------------------
# Submits all 8 variants in
# `DATASET_VARIANTS["mmb0-1b_smb1-1b_1p-ablations-v14"]` as separate LSF
# jobs (one bsub per variant, one GPU exclusive each).
#
# Defaults: --queue training-parallel --group s10396.
#
# v14 design: LOSS-WEIGHT TUNING (UNSUPERVISED) off the v3 spine
# `dualvq+wide+rvq-both+decoder-cov+adv+enc-deeper+adv-warmup10`.
# Scans the adjacency-BCE weight, adversarial schedule (warmup
# duration + weight), and MMD batch-integration alternatives.
#
# Concrete variant list (in submission order):
#   1. +adj-w250                # lighter adjacency
#   2. +adj-w500                # mid-point adjacency
#   3. +adj-w5000               # heavier adjacency (vs existing w3000)
#   4. +adv-warmup5             # shorter adv warmup
#   5. +adv +mmd-w50            # light MMD on top of adv spine
#   6. +mmd-w100 (no adv)       # pure MMD batch integration
#   7. +adv-w500                # 3.3x adv weight
#   8. +adv-warmup20            # 20-epoch warmup (longer than spine)
#
# Usage:
#   bash examples/submit_mmb_smb_ablations_v14.sh [OPTIONS]
#
# Options:
#   --queue / -q QUEUE   LSF queue (default: training-parallel).
#   --group / -g GROUP   LSF cost-code group (default: s10396).
#   --help  / -h         Print this usage block and exit.
#
# All 8 variants are STRICTLY UNSUPERVISED — none consume the
# `cell_type` label at training time. (An earlier draft included
# `+ce-w*` cross-entropy variants for an oracle/upper-bound point;
# those were dropped when SQUINT was committed to the unsupervised
# design.)
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

echo "[v14 ablations] LSF_QUEUE = $LSF_QUEUE"
echo "[v14 ablations] LSF_GROUP = $LSF_GROUP"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
exec bash "$SCRIPT_DIR/submit_dataset_sweep.sh" \
    "mmb0-1b_smb1-1b_1p-ablations-v14" \
    ${PASSTHROUGH_ARGS[@]+"${PASSTHROUGH_ARGS[@]}"}
