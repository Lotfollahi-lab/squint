#!/usr/bin/env bash
# submit_s57_crossbatch_ablations_multiseed.sh
# -----------------------------------------------------------------------------
# Run the COMPLETE SELF-CONTAINED s57 paper set multi-seed (29 variants).
#
# DEFAULT / reference model = cell-cond niche FiLM scale-only (+ cross-batch MNN),
# == s57_v19. Everything the paper needs lives under s57_v*:
#   v19         reference (FiLM scale-only)
#   v1-v18      component ablations, REBUILT on the FiLM spine (ablate the FiLM
#               model: adjacency / decoder-cov / GNN depth / neighbours /
#               cell+niche codebook L1 & L0)
#   v20/v21     contrastive axis (within-batch+FiLM / none+FiLM)
#   v22-v25     coupling-mechanism comparison vs FiLM (coupled / cross-stitch /
#               affine / decoupled)
#   v26/v27     two-separate-models baseline (cell-only / niche-only)
#   v28/v29     continuous-vs-VQ (continuous / discrete VQ ref)
# Each -> 5 seed jobs + 1 aggregator via submit_multi_seed.sh (29 x 5 = 145).
# NOTE: s57 now subsumes the old s63 (FiLM-spine ablations) — don't also run s63.
#
# Usage:
#   bash examples/submit_s57_crossbatch_ablations_multiseed.sh [SEEDS]
#   DRY_RUN=1 bash examples/submit_s57_crossbatch_ablations_multiseed.sh
#
# Env overrides forwarded to submit_multi_seed.sh: SEEDS, LSF_QUEUE, LSF_GROUP, ...
# -----------------------------------------------------------------------------
set -euo pipefail

SEEDS="${1:-${SEEDS:-0,1,2,3,4}}"
DRY_RUN="${DRY_RUN:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# s57 variants are generated programmatically in run_squint.py — read them from
# the registry so this launcher stays in sync (no hard-coded key list).
mapfile -t VARIANTS < <(python -c "
import sys; sys.path.insert(0, '$SCRIPT_DIR')
import run_squint
for k in run_squint.VARIANTS:
    if k.startswith('s57_v'):
        print(k)
")

if [[ "${#VARIANTS[@]}" -eq 0 ]]; then
    echo "No s57 variants found — is run_squint.py synced and importable in this env?" >&2
    exit 1
fi

echo "s57 cross-batch ablation sweep"
echo "  variants : ${#VARIANTS[@]}   seeds: $SEEDS   DRY_RUN: $DRY_RUN"
echo "  (reference s55_v3 + contrastive s51_v1/s51_v2 + coupled s56_v1 run separately)"
echo

for V in "${VARIANTS[@]}"; do
    echo ">>> $V"
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "    DRY: bash $SCRIPT_DIR/submit_multi_seed.sh \"$V\" \"$SEEDS\""
    else
        bash "$SCRIPT_DIR/submit_multi_seed.sh" "$V" "$SEEDS" || \
            echo "    !! submit FAILED for $V (continuing)"
    fi
done

echo
echo "Done (DRY_RUN=$DRY_RUN). ${#VARIANTS[@]} variants x $(echo "$SEEDS" | tr ',' ' ' | wc -w) seeds."
