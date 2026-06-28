#!/usr/bin/env bash
# submit_s57_residual_vq_ablation_multiseed.sh
# -----------------------------------------------------------------------------
# Run ONLY the residual-VQ depth ablation (reviewer: "no validation of the
# residual vector quantization, L=1 vs L=2"). 3 NEW s57 variants — L=1
# single-level VQ at three codebook sizes, on the FiLM-scale default spine:
#   s57_v30  L=1 VQ K=2700  (capacity-matched, = 30x90 effective)
#   s57_v31  L=1 VQ K=120   (parameter-matched, = 30+90 vectors)
#   s57_v32  L=1 VQ K=30    (drop-the-level, just the coarse L0)
# Compared against the L=2 residual (30,90) default = s57_v19 (already run).
# Each -> 5 seed jobs + 1 aggregator via submit_multi_seed.sh.
#
# Usage:
#   bash examples/submit_s57_residual_vq_ablation_multiseed.sh [SEEDS]
#   DRY_RUN=1 bash examples/submit_s57_residual_vq_ablation_multiseed.sh
# -----------------------------------------------------------------------------
set -euo pipefail

SEEDS="${1:-${SEEDS:-0,1,2,3,4}}"
DRY_RUN="${DRY_RUN:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# only the three residual-VQ-depth variants (v30/v31/v32), from the registry.
mapfile -t VARIANTS < <(python -c "
import sys; sys.path.insert(0, '$SCRIPT_DIR')
import run_squint
for k in run_squint.VARIANTS:
    if k.startswith(('s57_v30_','s57_v31_','s57_v32_')):
        print(k)
")

if [[ "${#VARIANTS[@]}" -eq 0 ]]; then
    echo "No s57 v30/v31/v32 variants found — is run_squint.py synced/importable?" >&2
    exit 1
fi

echo "s57 residual-VQ depth ablation (L=1 vs the L=2 default s57_v19)"
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
