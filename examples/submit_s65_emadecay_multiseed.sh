#!/usr/bin/env bash
# submit_s65_emadecay_multiseed.sh
# -----------------------------------------------------------------------------
# EMA-decay (beta) SENSITIVITY SWEEP on the headline reference (s57_v19).
# Reviewer: "EMA decay beta = 0.8 still asserted, not swept ... a short beta
# sweep would preempt the obvious question."
# (Family s65 — s64 is already taken by the squint_hln hyperparameter sweep.)
#
# 5 NEW s65 variants — the s57_v19 FiLM-scale reference (cross-batch MNN +
# cell-cond FiLM scale-only on the s51_v1 RVQ(30,90) spine) with ONLY the
# codebook EMA decay changed on both branches:
#   s65_v1  beta=0.5     s65_v2  beta=0.7     s65_v3  beta=0.9
#   s65_v4  beta=0.95    s65_v5  beta=0.99
# beta=0.8 is the reference itself (s57_v19, already run) — NOT re-run here.
# Full sweep = {0.5, 0.7, 0.8(=ref), 0.9, 0.95, 0.99}.
# Each -> 5 seed jobs + 1 aggregator via submit_multi_seed.sh.
#
# Evaluate CODEBOOK HEALTH across beta (the reviewer's actual concern):
#   for V in <the 5 s65 variants> s57_v19_reference-filmscale+mmb0-1b_smb1-1b_1p; do
#     bash examples/submit_codebook_usage.sh "$V" -- --all-seeds \
#       --codebook-sizes-cell 30,90 --codebook-sizes-niche 30,90
#   done
# (+ the usual downstream NMI/ARI/iLISI/MMD via the multiseed metrics.)
#
# Usage:
#   bash examples/submit_s65_emadecay_multiseed.sh [SEEDS]
#   DRY_RUN=1 bash examples/submit_s65_emadecay_multiseed.sh
# -----------------------------------------------------------------------------
set -euo pipefail

SEEDS="${1:-${SEEDS:-0,1,2,3,4}}"
DRY_RUN="${DRY_RUN:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The five EMA-decay-sweep variants (s65_v1..v5), pulled from the registry.
mapfile -t VARIANTS < <(python -c "
import sys; sys.path.insert(0, '$SCRIPT_DIR')
import run_squint
for k in run_squint.VARIANTS:
    if k.startswith('s65_v'):
        print(k)
")

if [[ "${#VARIANTS[@]}" -eq 0 ]]; then
    echo "No s65_v* variants found — is run_squint.py synced/importable?" >&2
    exit 1
fi

echo "s65 EMA-decay (beta) sweep on the s57_v19 reference (beta=0.8 = ref, not re-run)"
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
