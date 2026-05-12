#!/usr/bin/env bash
# submit_mmb_smb_ablations_v20.sh
# -----------------------------------------------------------------------------
# Submits all 4 variants in
# `DATASET_VARIANTS["mmb0-1b_smb1-1b_1p-ablations-v20"]` as separate LSF
# jobs (one bsub per variant, one GPU exclusive each).
#
# Defaults: --queue training-parallel --group s10396.
#
# Sweep 20 design: ISOLATE THE S18 REGRESSION. s17_v2 (+wide+rvq-both
# +decoder-cov+adv, niche NMI 0.695) → s18 variants (niche NMI 0.57-
# 0.61) changed THREE knobs simultaneously:
#
#   - architecture:  +wide (h=256)        → +small-h32/64
#   - codebook:      (30, 90) 2-level     → (30, 30) or (30, 30, 30)
#   - sampler:       [8] (default)        → [4] in half of s18
#
# Sweep 20 holds two knobs at the s17 reference and varies the third,
# so each variant's niche-NMI delta is directly attributable to that
# one knob. v1 re-runs the exact s17_v2 baseline as a within-sweep
# sanity check (same hardware, same code).
#
# Concrete variant list:
#   s20_v1_+wide+rvq-both+adv                # baseline (s17_v2 re-run)
#   s20_v2_+wide+rvq-both-3level-30-30-30+adv # only codebook changed
#   s20_v3_+small-h64+rvq-both+adv            # only encoder size changed
#   s20_v4_+wide+rvq-both+adv+sampler4        # only sampler changed
#
# Interpretation cheat-sheet:
#   - v1 ≈ 0.69, others split below: all 3 knobs contribute.
#   - One of v2/v3/v4 drops sharply, others ≈ v1: THAT knob is the
#     culprit; the others are safe to vary in future sweeps.
#   - v1 itself drops well below 0.69: something else regressed
#     between then and now (data, env, dependency versions) —
#     investigate before reading v2/v3/v4 deltas.
#
# Usage:
#   bash examples/submit_mmb_smb_ablations_v20.sh [OPTIONS]
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

echo "[v20 ablations] LSF_QUEUE = $LSF_QUEUE"
echo "[v20 ablations] LSF_GROUP = $LSF_GROUP"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
exec bash "$SCRIPT_DIR/submit_dataset_sweep.sh" \
    "mmb0-1b_smb1-1b_1p-ablations-v20" \
    ${PASSTHROUGH_ARGS[@]+"${PASSTHROUGH_ARGS[@]}"}
