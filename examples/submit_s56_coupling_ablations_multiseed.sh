#!/usr/bin/env bash
# submit_s56_coupling_ablations_multiseed.sh
# -----------------------------------------------------------------------------
# Run ALL s56 ablations (the ENCODER CELL/NICHE COUPLING-METHOD sweep) multi-seed.
#
# s56 sweeps how the cell-intrinsic and spatial-niche encoders share parameters,
# on the cross-batch MNN spine (== s55_v3). 8 variants (defined in run_squint.py):
#   v1  coupled (shared trunk)            — hard parameter sharing
#   v2  Y-shape shared192/path64          — partial sharing (mostly shared)
#   v3  Y-shape shared128/path128         — partial sharing (balanced)
#   v4  Y-shape shared64/path192          — partial sharing (mostly specialised)
#   v5  cross-stitch, one 2x2 (scalar)    — learned soft sharing (Misra 2016)
#   v6  cross-stitch, per-channel 2x2     — learned soft sharing (Misra 2016)
#   v7  stop-gradient on coupled trunk    — gradient-level coupling
#   v8  soft L2 penalty between trunks    — soft parameter sharing (Duong 2015)
# Each -> 5 seed jobs + 1 aggregator via submit_multi_seed.sh.
#
# Reference (decoupled endpoint) = s55_v3 — already run as part of the s55 sweep;
# NOT relaunched here.
#
# Usage:
#   bash examples/submit_s56_coupling_ablations_multiseed.sh [SEEDS]
#   DRY_RUN=1 bash examples/submit_s56_coupling_ablations_multiseed.sh
#
# Env overrides forwarded to submit_multi_seed.sh: SEEDS, LSF_QUEUE, LSF_GROUP, ...
# -----------------------------------------------------------------------------
set -euo pipefail

SEEDS="${1:-${SEEDS:-0,1,2,3,4}}"
DRY_RUN="${DRY_RUN:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# s56 variants are defined in run_squint.py — read them from the registry so this
# launcher stays in sync (no hard-coded key list).
mapfile -t VARIANTS < <(python -c "
import sys; sys.path.insert(0, '$SCRIPT_DIR')
import run_squint
for k in run_squint.VARIANTS:
    if k.startswith('s56_v'):
        print(k)
")

if [[ "${#VARIANTS[@]}" -eq 0 ]]; then
    echo "No s56 variants found — is run_squint.py synced and importable in this env?" >&2
    exit 1
fi

echo "s56 encoder-coupling ablation sweep"
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
