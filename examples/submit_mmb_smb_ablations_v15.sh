#!/usr/bin/env bash
# submit_mmb_smb_ablations_v15.sh
# -----------------------------------------------------------------------------
# Submits all 8 variants in
# `DATASET_VARIANTS["mmb0-1b_smb1-1b_1p-ablations-v15"]` as separate LSF
# jobs (one bsub per variant, one GPU exclusive each).
#
# Defaults: --queue training-parallel --group s10396.
#
# v15 design: ARCHITECTURE exploration off the v3 spine
# `dualvq+wide+rvq-both+decoder-cov+adv+enc-deeper+adv-warmup10`.
# Probes GNN/MLP width, latent dim, dropout, GNN depth, and attention.
#
# Concrete variant list (in submission order):
#   1. +gnn-h512                  # wider GNN (256 -> 512)
#   2. +mlp-h512                  # wider MLP (400 -> 512)
#   3. +mlp-h128                  # narrower MLP (400 -> 128)
#   4. +dropout-p0.1              # light regularisation
#   5. +dropout-p0.3              # heavy regularisation
#   6. +small-latent-128          # latent 256 -> 128
#   7. +gnn2                      # 2-layer GNN (with sampler=[8,8])
#   8. +gatv2                     # attention instead of mean aggregation
#
# Usage:
#   bash examples/submit_mmb_smb_ablations_v15.sh [OPTIONS]
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

echo "[v15 ablations] LSF_QUEUE = $LSF_QUEUE"
echo "[v15 ablations] LSF_GROUP = $LSF_GROUP"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
exec bash "$SCRIPT_DIR/submit_dataset_sweep.sh" \
    "mmb0-1b_smb1-1b_1p-ablations-v15" \
    ${PASSTHROUGH_ARGS[@]+"${PASSTHROUGH_ARGS[@]}"}
