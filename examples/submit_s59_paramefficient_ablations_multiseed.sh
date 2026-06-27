#!/usr/bin/env bash
# submit_s59_paramefficient_ablations_multiseed.sh
# -----------------------------------------------------------------------------
# Run ALL s59 ablations multi-seed. Follow-ups to the s56 coupling table:
#   A. soft-L2 weight sweep (v1 1e-4, v2 1e-2, v3 1e-1; 1e-3 == s56_v8)
#   B. parameter-efficient coupling = shared trunk + per-branch adapter
#      (v4 affine ~50%, v5 lora-r16 ~52%, v6 mlp ~80% of decoupled enc params)
# All on the s55_v3 cross-batch-MNN spine (defined in run_squint.py).
# Each -> 5 seed jobs + 1 aggregator via submit_multi_seed.sh.
#
# Reference (decoupled) = s55_v3 — already run; NOT relaunched.
# REQUIRES the synced vqniche code on the farm (BranchAdapter + encoder
# branch_adapter path) before launching.
#
# Usage:
#   bash examples/submit_s59_paramefficient_ablations_multiseed.sh [SEEDS]
#   DRY_RUN=1 bash examples/submit_s59_paramefficient_ablations_multiseed.sh
# -----------------------------------------------------------------------------
set -euo pipefail

SEEDS="${1:-${SEEDS:-0,1,2,3,4}}"
DRY_RUN="${DRY_RUN:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mapfile -t VARIANTS < <(python -c "
import sys; sys.path.insert(0, '$SCRIPT_DIR')
import run_squint
for k in run_squint.VARIANTS:
    if k.startswith('s59_v'):
        print(k)
")

if [[ "${#VARIANTS[@]}" -eq 0 ]]; then
    echo "No s59 variants found — is run_squint.py synced and importable in this env?" >&2
    exit 1
fi

echo "s59 soft-L2 sweep + parameter-efficient coupling"
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
