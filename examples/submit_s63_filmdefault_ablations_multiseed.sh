#!/usr/bin/env bash
# submit_s63_filmdefault_ablations_multiseed.sh
# -----------------------------------------------------------------------------
# Run ALL s63 ablations multi-seed = every standard ablation re-run on the NEW
# DEFAULT spine (cross-batch MNN + cell-cond niche FiLM scale-only, == s62_v5).
# Mirrors s57 but with the FiLM scale-only coupling as default. 18 variants
# (generated programmatically in run_squint.py). Each -> 5 seed jobs + 1
# aggregator via submit_multi_seed.sh.
#
# Reference (un-ablated default) = s62_v5 (already run as part of s62).
# REQUIRES the synced vqniche code on the farm.
#
# !! LARGE re-run (18 x 5 = 90 jobs). RECOMMENDED to first replicate the FiLM
#    scale-only integration gain on a 2nd dataset before committing this.
#
# Usage:
#   bash examples/submit_s63_filmdefault_ablations_multiseed.sh [SEEDS]
#   DRY_RUN=1 bash examples/submit_s63_filmdefault_ablations_multiseed.sh
# -----------------------------------------------------------------------------
set -euo pipefail

SEEDS="${1:-${SEEDS:-0,1,2,3,4}}"
DRY_RUN="${DRY_RUN:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mapfile -t VARIANTS < <(python -c "
import sys; sys.path.insert(0, '$SCRIPT_DIR')
import run_squint
for k in run_squint.VARIANTS:
    if k.startswith('s63_v'):
        print(k)
")

if [[ "${#VARIANTS[@]}" -eq 0 ]]; then
    echo "No s63 variants found — is run_squint.py synced and importable in this env?" >&2
    exit 1
fi

echo "s63 FiLM-scale-only default ablation sweep"
echo "  variants : ${#VARIANTS[@]}   seeds: $SEEDS   DRY_RUN: $DRY_RUN"
echo "  (reference / un-ablated default = s62_v5, already run)"
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
