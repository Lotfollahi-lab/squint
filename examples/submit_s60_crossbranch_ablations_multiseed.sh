#!/usr/bin/env bash
# submit_s60_crossbranch_ablations_multiseed.sh
# -----------------------------------------------------------------------------
# Run ALL s60 ablations multi-seed — NOVEL cross-branch couplings (none overlap
# s56/s58/s59). On the s55_v3 cross-batch-MNN decoupled spine; 8 variants:
#   v1 shared-token (additive shared codebook)
#   v2 cross-branch residual VQ
#   v3 shared codebook (tied)
#   v4 cell-conditioned niche (bias)
#   v5 cell-conditioned niche (FiLM)
#   v6 niche attends cell codebook
#   v7 mutual alignment loss (wt=100)
#   v8 BANKSY/CellCharter neighbour-expression augmentation
# Each -> 5 seed jobs + 1 aggregator via submit_multi_seed.sh.
#
# Reference (decoupled) = s55_v3 — already run; NOT relaunched.
# REQUIRES the synced vqniche code on the farm (new encoder cross-branch paths +
# cell_niche_alignment_loss) before launching.
#
# Usage:
#   bash examples/submit_s60_crossbranch_ablations_multiseed.sh [SEEDS]
#   DRY_RUN=1 bash examples/submit_s60_crossbranch_ablations_multiseed.sh
# -----------------------------------------------------------------------------
set -euo pipefail

SEEDS="${1:-${SEEDS:-0,1,2,3,4}}"
DRY_RUN="${DRY_RUN:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mapfile -t VARIANTS < <(python -c "
import sys; sys.path.insert(0, '$SCRIPT_DIR')
import run_squint
for k in run_squint.VARIANTS:
    if k.startswith('s60_v'):
        print(k)
")

if [[ "${#VARIANTS[@]}" -eq 0 ]]; then
    echo "No s60 variants found — is run_squint.py synced and importable in this env?" >&2
    exit 1
fi

echo "s60 cross-branch coupling sweep"
echo "  variants : ${#VARIANTS[@]}   seeds: $SEEDS   DRY_RUN: $DRY_RUN"
echo "  (reference s55_v3 (decoupled) run separately as part of the s55 sweep)"
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
