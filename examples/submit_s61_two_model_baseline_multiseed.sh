#!/usr/bin/env bash
# submit_s61_two_model_baseline_multiseed.sh
# -----------------------------------------------------------------------------
# Run the s61 TWO-SEPARATE-MODELS baseline (reviewer: "is the joint model better
# than two strong separate single-task models?"). Same decoupled spine as SQUINT
# (s55_v3), each variant single-task:
#   v1 cell-only   (drop niche/spatial losses)  -> read CELL codes
#   v2 niche-only  (drop cell losses + contrastive) -> read NICHE codes
# Each -> 5 seed jobs + 1 aggregator via submit_multi_seed.sh.
#
# Then compare:  python ../squint-reproducibility/analysis/ablations/summarize_ablation_multiseed.py --set joint_vs_separate
#   SQUINT(decoupled) cell vs s61 cell-only ; SQUINT niche vs s61 niche-only
#   (per-task parity) + the shared-encoder efficiency frontier (s56_v1 / s59_v4).
#
# Usage:
#   bash examples/submit_s61_two_model_baseline_multiseed.sh [SEEDS]
#   DRY_RUN=1 bash examples/submit_s61_two_model_baseline_multiseed.sh
# -----------------------------------------------------------------------------
set -euo pipefail

SEEDS="${1:-${SEEDS:-0,1,2,3,4}}"
DRY_RUN="${DRY_RUN:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mapfile -t VARIANTS < <(python -c "
import sys; sys.path.insert(0, '$SCRIPT_DIR')
import run_squint
for k in run_squint.VARIANTS:
    if k.startswith('s61_v'):
        print(k)
")

if [[ "${#VARIANTS[@]}" -eq 0 ]]; then
    echo "No s61 variants found — is run_squint.py synced and importable in this env?" >&2
    exit 1
fi

echo "s61 two-separate-models baseline"
echo "  variants : ${#VARIANTS[@]}   seeds: $SEEDS   DRY_RUN: $DRY_RUN"
echo "  (joint reference = SQUINT decoupled s55_v3, already run)"
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
