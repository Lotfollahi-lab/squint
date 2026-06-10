#!/usr/bin/env bash
# submit_mmb_smb_ablations_v21.sh
# -----------------------------------------------------------------------------
# Submits all 4 variants in
# `DATASET_VARIANTS["mmb0-1b_smb1-1b_1p-ablations-v21"]` as separate LSF
# jobs (one bsub per variant, one GPU exclusive each).
#
# Defaults: --queue training-parallel --group s10396.
#
# Sweep 21 design: BATCH-INTEGRATION LOSS & NICHE-NECK ablation. All
# 4 variants share the s17_v2 spine (`+wide+rvq-both+decoder-cov+adv`,
# niche NMI 0.695); v2/v3/v4 each change one axis to test 3 distinct
# hypotheses.
#
# Concrete variant list:
#   s21_v1_+adv                                # baseline (re-run of s17_v2)
#   s21_v2_+no-batch-int                       # drop adv, no MMD
#   s21_v3_+mmd-w100                           # MMD instead of adv
#   s21_v4_+adv+niche-neck                     # spine + new pre-GNN MLP
#
# What each variant tests:
#
#   v1 (baseline): re-establishes the s17_v2 reference number on the
#     current hardware + code. Anchor for v2/v3/v4 deltas.
#
#   v2 (no batch-int loss): does SQUINT need an encoder-side
#     batch-correction signal at all, or is `+decoder-cov` (the
#     NicheCompass-style covariate concat to the decoder) sufficient?
#     If v2 ≈ v1, the adversarial / MMD encoder pressure is not
#     load-bearing for this dataset.
#
#   v3 (MMD instead of adv): replaces the GRL adversary with a
#     deterministic kernel-MMD pressure. If v3 ≈ v1 with simpler
#     training dynamics (no min-max game), MMD is the easier batch-
#     integration loss to ship.
#
#   v4 (niche-neck MLP): inserts a NEW 256-d projection between z_mlp
#     and the GNN. Cell-VQ still takes z_mlp directly; only the GNN
#     sees niche_neck(z_mlp). This DECOUPLES the cell-VQ input from
#     niche-side gradients (adj-BCE, nbr-NB, niche-commit) that
#     currently flow back into the shared z_mlp. If v4 lifts cell-NMI
#     WITHOUT hurting niche-NMI, the shared trunk was a bottleneck —
#     niche-neck becomes a candidate addition to the spine.
#
# Note on v4: this is the first sweep that exercises a NEW
# architecture knob (`niche_neck_params` in VQNiche_Dual_Encoder).
# The default is `None` (legacy behaviour, identical to before),
# so all earlier variants continue to produce byte-identical models
# — only this sweep's v4 turns it on.
#
# Usage:
#   bash examples/submit_mmb_smb_ablations_v21.sh [OPTIONS]
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

echo "[v21 ablations] LSF_QUEUE = $LSF_QUEUE"
echo "[v21 ablations] LSF_GROUP = $LSF_GROUP"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
exec bash "$SCRIPT_DIR/submit_dataset_sweep.sh" \
    "mmb0-1b_smb1-1b_1p-ablations-v21" \
    ${PASSTHROUGH_ARGS[@]+"${PASSTHROUGH_ARGS[@]}"}
