#!/usr/bin/env bash
# submit_s58_infoflow_ablations_multiseed.sh
# -----------------------------------------------------------------------------
# Run ALL s58 ablations (INFORMATION-FLOW / COMPLEMENTARITY coupling) multi-seed.
#
# The s56 trunk-sharing sweep tied with the decoupled baseline, so s58 changes
# WHAT flows between the cell & niche branches instead of how the trunk is
# shared. On the s55_v3 cross-batch-MNN decoupled spine (defined in
# run_squint.py); 7 variants:
#   v1  compose cell->niche (z_q_cell, detached)            #1, lead
#   v2  compose cell->niche (z_q_cell, detached, proj 64)   #1
#   v3  compose cell->niche (z_mlp_cell continuous)         #1 ablation
#   v4  compose cell->niche (z_q_cell, NO detach)           #1 (protection test)
#   v5  disentanglement penalty (wt=100)                    #2
#   v6  disentanglement penalty (wt=1000)                   #2
#   v7  compose (z_q_cell) + disentangle (wt=100)           #1+#2 combined
# Each -> 5 seed jobs + 1 aggregator via submit_multi_seed.sh.
#
# Reference (decoupled) = s55_v3 — already run; NOT relaunched here.
# REQUIRES the synced vqniche code on the farm (new encoder cell_to_niche path +
# disentangle_cell_niche_loss) before launching.
#
# Usage:
#   bash examples/submit_s58_infoflow_ablations_multiseed.sh [SEEDS]
#   DRY_RUN=1 bash examples/submit_s58_infoflow_ablations_multiseed.sh
# -----------------------------------------------------------------------------
set -euo pipefail

SEEDS="${1:-${SEEDS:-0,1,2,3,4}}"
DRY_RUN="${DRY_RUN:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# s58 variants are defined in run_squint.py — read them from the registry so this
# launcher stays in sync (no hard-coded key list).
mapfile -t VARIANTS < <(python -c "
import sys; sys.path.insert(0, '$SCRIPT_DIR')
import run_squint
for k in run_squint.VARIANTS:
    if k.startswith('s58_v'):
        print(k)
")

if [[ "${#VARIANTS[@]}" -eq 0 ]]; then
    echo "No s58 variants found — is run_squint.py synced and importable in this env?" >&2
    exit 1
fi

echo "s58 information-flow / complementarity coupling sweep"
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
