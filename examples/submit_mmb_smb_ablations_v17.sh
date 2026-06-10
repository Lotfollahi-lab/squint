#!/usr/bin/env bash
# submit_mmb_smb_ablations_v17.sh
# -----------------------------------------------------------------------------
# Submits all 8 variants in
# `DATASET_VARIANTS["mmb0-1b_smb1-1b_1p-ablations-v17"]` as separate LSF
# jobs (one bsub per variant, one GPU exclusive each).
#
# Defaults: --queue training-parallel --group s10396.
#
# v17 purpose: REGRESSION-FIX VERIFICATION. The 2026-05-12 sweeps
# (v12 / v13 / v14 / v15 / v16) all regressed ~0.03-0.05 on niche
# NMI because the base config had `max_epochs=200` while early
# stopping monitored `val_loss` — over-trained adversarial pushed
# the niche encoder past peak NMI. The fix was to revert
# `max_epochs` to 80 (the historic value).
#
# v17 takes the TOP-8 niche NMI variants from `summary_long.csv`
# (pre-05-12 timestamps) and re-runs them as `s17_v<N>_*` aliases
# on the now-reverted base config. The name pattern is
# `s<sweep>_v<variant>_<base>`:
#   - s17  = this sweep's number (matches the DATASET_VARIANTS key
#            `mmb...-ablations-v17`).
#   - v<N> = the variant's per-sweep index (v1..v8 here, sorted by
#            historic best niche NMI descending).
# See the "Variant-name prefix convention" comment block in
# run_squint.py for the full rationale + the
# `_register_alias(...)` mechanism. Each alias points at the SAME
# build lambda as its original variant — only the artefact
# directory name changes, so the test is a direct
# apples-to-apples re-run.
#
# Concrete variant list (in submission order):
#   s17_v1_+mmd-w100                            # historic 0.6950
#   s17_v2_+adv (bare, no dataset tag)          # historic 0.6950
#   s17_v3_+adv+enc-deeper                      # historic 0.6944
#   s17_v4_+adv+nbr-hops-3                      # historic 0.6867
#   s17_v5_+adv+adj-w3000                       # historic 0.6834
#   s17_v6_+adv-w50+adv-warmup10                # historic 0.6824
#   s17_v7_+rvq-both-3level+mmd-w50             # historic 0.6811
#   s17_v8_+rvq-both-3level-20-40-80+enc-deeper # historic 0.6778
#
# Expected outcome: every s17_v<N>_* run should land within ~0.01
# niche NMI of its historic score. If so, the regression is fixed
# and v12-v16 can be re-submitted on the same base config.
#
# Usage:
#   bash examples/submit_mmb_smb_ablations_v17.sh [OPTIONS]
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

echo "[v17 ablations] LSF_QUEUE = $LSF_QUEUE"
echo "[v17 ablations] LSF_GROUP = $LSF_GROUP"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
exec bash "$SCRIPT_DIR/submit_dataset_sweep.sh" \
    "mmb0-1b_smb1-1b_1p-ablations-v17" \
    ${PASSTHROUGH_ARGS[@]+"${PASSTHROUGH_ARGS[@]}"}
