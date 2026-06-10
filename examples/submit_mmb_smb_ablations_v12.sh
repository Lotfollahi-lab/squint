#!/usr/bin/env bash
# submit_mmb_smb_ablations_v12.sh
# -----------------------------------------------------------------------------
# Submits all 8 variants in
# `DATASET_VARIANTS["mmb0-1b_smb1-1b_1p-ablations-v12"]` as separate LSF
# jobs (one bsub per variant, one GPU exclusive each). Queue +
# cost-code group are accepted as flags so you can switch between
# (queue, group) pairings without editing the script.
#
# Defaults: --queue training-parallel --group s10396 (the AI/GPU queue
# with the team-project cost code).
#
# v12 design: COMMIT-LOSS exploration off the v3 spine
# `dualvq+wide+rvq-both+decoder-cov+adv+enc-deeper+adv-warmup10`.
# The v6b `commit-cell-w2` data point lifted cell-NMI from 0.495 to
# 0.503 — v12 maps the rest of the (wt_commit_cell, wt_commit_niche)
# plane.
#
#   wt_commit_cell  ∈ { 0.5, 3, 5 }    (3 variants)
#   wt_commit_niche ∈ { 0.5, 3, 5 }    (3 variants)
#   asymmetric combinations: cell-w3+niche-w0.5, cell-w0.5+niche-w3
#                            (2 variants)
#
# Concrete variant list (in submission order):
#   1. +commit-cell-w0.5                # looser cell VQ
#   2. +commit-cell-w3                  # sharper cell VQ
#   3. +commit-cell-w5                  # aggressive sharper
#   4. +commit-niche-w0.5               # looser niche VQ
#   5. +commit-niche-w3                 # sharper niche VQ
#   6. +commit-niche-w5                 # aggressive sharper
#   7. +commit-cell-w3 +commit-niche-w0.5  # cell sharp / niche loose
#   8. +commit-cell-w0.5 +commit-niche-w3  # cell loose / niche sharp
#
# Usage:
#   bash examples/submit_mmb_smb_ablations_v12.sh [OPTIONS]
#
# Options:
#   --queue / -q QUEUE   LSF queue (default: training-parallel).
#                        Also overridable via LSF_QUEUE env var.
#   --group / -g GROUP   LSF cost-code group (default: s10396).
#                        Also overridable via LSF_GROUP env var.
#   --help  / -h         Print this usage block and exit.
#
# Precedence: CLI flag > env var > script default.
#
# Other env-var overrides from `submit_dataset_sweep.sh` work too:
#   DRY_RUN=1, LSF_WALL=..., LSF_MEM_MB=..., SQUINT_WITH_PEARSON=1, etc.
#
# Logs / artifacts: same convention as v9/v10/v11.
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

echo "[v12 ablations] LSF_QUEUE = $LSF_QUEUE"
echo "[v12 ablations] LSF_GROUP = $LSF_GROUP"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
exec bash "$SCRIPT_DIR/submit_dataset_sweep.sh" \
    "mmb0-1b_smb1-1b_1p-ablations-v12" \
    ${PASSTHROUGH_ARGS[@]+"${PASSTHROUGH_ARGS[@]}"}
