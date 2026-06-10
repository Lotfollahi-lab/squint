#!/usr/bin/env bash
# submit_mmb_smb_ablations_v19.sh
# -----------------------------------------------------------------------------
# Submits all 4 variants in
# `DATASET_VARIANTS["mmb0-1b_smb1-1b_1p-ablations-v19"]` as separate LSF
# jobs (one bsub per variant, one GPU exclusive each).
#
# Defaults: --queue training-parallel --group s10396.
#
# Sweep 19 design: SPATIAL-LOSS ABLATION. 2x2 grid over
# {adj-BCE on/off} × {nbr-NB on/off} = 4 variants on the historic
# top-niche-NMI spine (`+wide+rvq-both+decoder-cov+adv`).
#
#                 adj-BCE ON          adj-BCE OFF
#   nbr-NB ON     s19_v1 baseline     s19_v2 +no-adj
#   nbr-NB OFF    s19_v3 +no-nbr      s19_v4 +no-spatial
#
# Question being answered: do the spatial supervision losses
# (adjacency BCE + neighborhood-NB gene-expression reconstruction)
# help the CELL branch learn cell-type-discriminative codes? The
# cell branch shares `z_mlp` with the niche GNN (via the sampled-
# neighbor cells), so spatial-side losses backprop into z_mlp via
# the shared trunk — they MAY help or hurt cell-NMI depending on
# whether spatial structure correlates with cell-type identity on
# this dataset.
#
# Concrete variant list (in submission order):
#   s19_v1_+adv                # baseline (adj ON, nbr-NB ON)
#   s19_v2_+adv+no-adj         # drop adj only
#   s19_v3_+adv+no-nbr         # drop nbr-NB only (keep adj)
#   s19_v4_+adv+no-spatial     # drop both (pure cell-side training)
#
# Interpretation cheat-sheet:
#   - If v4 ≥ v1 on cell-NMI: spatial losses HURT the cell branch.
#   - If v4 << v1 on cell-NMI: spatial losses HELP the cell branch.
#   - v2 vs v3 isolates which of the two is doing the work.
#
# Niche-NMI will obviously degrade in v2/v3/v4 since the niche
# branch loses its supervision — that's expected, not a regression.
# The headline result is the CELL-side delta.
#
# Usage:
#   bash examples/submit_mmb_smb_ablations_v19.sh [OPTIONS]
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

echo "[v19 ablations] LSF_QUEUE = $LSF_QUEUE"
echo "[v19 ablations] LSF_GROUP = $LSF_GROUP"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
exec bash "$SCRIPT_DIR/submit_dataset_sweep.sh" \
    "mmb0-1b_smb1-1b_1p-ablations-v19" \
    ${PASSTHROUGH_ARGS[@]+"${PASSTHROUGH_ARGS[@]}"}
