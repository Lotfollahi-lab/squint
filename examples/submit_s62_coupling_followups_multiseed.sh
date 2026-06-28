#!/usr/bin/env bash
# submit_s62_coupling_followups_multiseed.sh
# -----------------------------------------------------------------------------
# Run the s62 follow-ups multi-seed — tweaks of the s56/s60 threads that looked
# on-par / slightly better:
#   v1-v3 Y-shape smaller shared trunk (48/208, 32/224, 16/240)
#   v4-v6 cell-conditioned-niche FiLM (pre-VQ, scale-only, continuous)
#   v7    multi-head (heads=4) niche->cell-codebook attention
#   v8    Y-shape 64/192 + cell-cond FiLM combo
# All on the s55_v3 spine. Each -> 5 seed jobs + 1 aggregator.
#
# Reference = s55_v3 (already run). REQUIRES synced vqniche on the farm.
#
# Usage:
#   bash examples/submit_s62_coupling_followups_multiseed.sh [SEEDS]
#   DRY_RUN=1 bash examples/submit_s62_coupling_followups_multiseed.sh
# -----------------------------------------------------------------------------
set -euo pipefail

SEEDS="${1:-${SEEDS:-0,1,2,3,4}}"
DRY_RUN="${DRY_RUN:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mapfile -t VARIANTS < <(python -c "
import sys; sys.path.insert(0, '$SCRIPT_DIR')
import run_squint
for k in run_squint.VARIANTS:
    if k.startswith('s62_v'):
        print(k)
")

if [[ "${#VARIANTS[@]}" -eq 0 ]]; then
    echo "No s62 variants found — is run_squint.py synced and importable in this env?" >&2
    exit 1
fi

echo "s62 coupling follow-ups"
echo "  variants : ${#VARIANTS[@]}   seeds: $SEEDS   DRY_RUN: $DRY_RUN"
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
