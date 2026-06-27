#!/usr/bin/env bash
# submit_s55_crossbatch_multiseed.sh
# -----------------------------------------------------------------------------
# Launch the s55 CROSS-BATCH MNN contrastive sweep with 5 seeds each, by
# calling `submit_multi_seed.sh <variant> <seeds>` per variant.
#
# s55 = the s49_v23 spine with the within-batch contrastive loss replaced by
# contrastive_cell_attribute_cross_batch_mnn_loss (within-batch NT-Xent +
# cross-batch mutual-NN pure-attraction). Goal: keep cell-type resolution while
# improving batch integration (iLISI/MMD). Sweep over wt_cross / k_cross / floor:
#
#   s55_v1  wt_cross=1   k_cross=1                 (conservative)
#   s55_v2  wt_cross=5   k_cross=1
#   s55_v3  wt_cross=10  k_cross=1                 (cross weight = within weight)
#   s55_v4  wt_cross=5   k_cross=2                 (more cross matches)
#   s55_v5  wt_cross=2   k_cross=1  mnn_floor=0.5  (high-confidence matches only)
#
# REFERENCE: s49_v23 (== within-batch only, i.e. wt_cross=0). It's already run
# multi-seed, so it is NOT relaunched here — compare the s55 sweep against it.
#
# Usage:
#   bash examples/submit_s55_crossbatch_multiseed.sh [SEEDS]
#   DRY_RUN=1 bash examples/submit_s55_crossbatch_multiseed.sh
#
# Env overrides forwarded to submit_multi_seed.sh: SEEDS, LSF_QUEUE, LSF_GROUP,
# LSF_MEM_MB, LSF_WALL, ...   (DRY_RUN=1 prints the calls instead of running).
# -----------------------------------------------------------------------------
set -euo pipefail

SEEDS="${1:-${SEEDS:-0,1,2,3,4}}"
DRY_RUN="${DRY_RUN:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

_SPINE="dualvq+rvq-both+decoder-cov+no-batch-int+enc-deeper+dec-w32+knn16+sampler16+cell-w1+bs512+lr7e-4+within-sec+decoupled-enc+diversity-w10+contrastWB-w10-k5"
VARIANTS=(
  "s55_v1_${_SPINE}+crossmnn-wt1-k1+mmb0-1b_smb1-1b_1p"
  "s55_v2_${_SPINE}+crossmnn-wt5-k1+mmb0-1b_smb1-1b_1p"
  "s55_v3_${_SPINE}+crossmnn-wt10-k1+mmb0-1b_smb1-1b_1p"
  "s55_v4_${_SPINE}+crossmnn-wt5-k2+mmb0-1b_smb1-1b_1p"
  "s55_v5_${_SPINE}+crossmnn-wt2-k1-floor0.5+mmb0-1b_smb1-1b_1p"
)

echo "s55 cross-batch MNN multi-seed sweep"
echo "  seeds   : $SEEDS"
echo "  variants: ${#VARIANTS[@]}   DRY_RUN=$DRY_RUN"
echo "  (reference s49_v23 is already run — not relaunched)"
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
echo "Done (DRY_RUN=$DRY_RUN)."
