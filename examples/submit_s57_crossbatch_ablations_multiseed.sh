#!/usr/bin/env bash
# submit_s57_crossbatch_ablations_multiseed.sh
# -----------------------------------------------------------------------------
# Run ALL s57 ablations (the CROSS-BATCH MNN contrastive spine) multi-seed.
#
# s57 = every s51/s52/s54 ablation re-run with the within-batch contrastive loss
# swapped for the cross-batch MNN loss (wt_cross=10, k_cross=1 == s55_v3). 18 NEW
# variants (generated programmatically in run_squint.py). Each -> 5 seed jobs +
# 1 aggregator via submit_multi_seed.sh.
#
# NOT relaunched (already run / run separately):
#   * cross-batch reference (s49_v23 + cross) = s55_v3
#   * contrastive-axis comparators            = s51_v1 (within) + s51_v2 (none)
#   * encoder cell/niche coupling axis        = s56 (8 variants; reference s55_v3),
#       run via:  bash examples/submit_s56_coupling_ablations_multiseed.sh
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
