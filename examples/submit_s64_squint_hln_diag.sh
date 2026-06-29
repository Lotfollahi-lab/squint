#!/usr/bin/env bash
# submit_s64_squint_hln_diag.sh
# -----------------------------------------------------------------------------
# Submit the s64 squint_hln DIAGNOSTIC sweep (20 variants, each ONE change vs the
# FiLM-scale squint_hln reference) to find why HLN cell/niche resolution is poor.
# SINGLE-SEED by default (seed 0); aggregator skipped (SUBMIT_AGGREGATOR=0) since
# one seed needs no across-seed concat. All variants use the existing squint_hln
# (full-panel) blob — make sure it was REBUILT with labels (cell_type / niche)
# first; no new blob is needed.
#
# Reads the s64 keys straight from the VARIANTS registry (so it can't drift from
# run_squint.py), then calls submit_multi_seed.sh per variant.
#
# Usage:
#   bash examples/submit_s64_squint_hln_diag.sh [SEEDS]
#     SEEDS  comma list (default: 0). e.g. 0,1,2 to add seeds later.
#
# Env overrides (forwarded / mirrored from submit_multi_seed.sh):
#   VENV_PATH   /nfs/team361/sb75/.venvs/squint
#   SQUINT_REPO <auto: this script's repo root>
#   LSF_QUEUE   gpu-lotfollahi   LSF_GROUP team361   (GPU training)
#   DRY_RUN     0  (1 = list the variants + the submit commands, submit nothing)
# -----------------------------------------------------------------------------
set -euo pipefail

SEEDS="${1:-0}"
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SQUINT_REPO="${SQUINT_REPO:-"$( cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd )"}"
VENV_PATH="${VENV_PATH:-/nfs/team361/sb75/.venvs/squint}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
    echo "ERROR: VENV_PATH=$VENV_PATH is not a venv (no bin/activate)." >&2; exit 1
fi
# shellcheck disable=SC1091
source "$VENV_PATH/bin/activate"

# Enumerate s64 variant keys from the registry (sorted by version index).
# (while-read, not mapfile, for bash 3.2 portability.)
KEYS_RAW="$(python - <<PY
import sys, re
sys.path.insert(0, "$SQUINT_REPO/examples")
from run_squint import VARIANTS
ks = [k for k in VARIANTS if k.startswith("s64_v")]
ks.sort(key=lambda k: int(re.match(r"s64_v(\d+)_", k).group(1)))
print("\n".join(ks))
PY
)"
N="$(printf '%s\n' "$KEYS_RAW" | grep -c .)"
if [[ "$N" -eq 0 ]]; then
    echo "ERROR: no s64_v* variants found in the registry (sync run_squint.py?)." >&2
    exit 1
fi

echo "=========================================================="
echo "s64 squint_hln diagnostic sweep: $N variants, seeds=$SEEDS"
echo "  repo=$SQUINT_REPO  venv=$VENV_PATH  DRY_RUN=$DRY_RUN"
echo "=========================================================="

i=0
while IFS= read -r key; do
    [[ -z "$key" ]] && continue
    i=$((i + 1))
    echo "[$i/$N] $key"
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "    SUBMIT_AGGREGATOR=0 bash examples/submit_multi_seed.sh '$key' '$SEEDS'"
    else
        SUBMIT_AGGREGATOR=0 bash "$SQUINT_REPO/examples/submit_multi_seed.sh" "$key" "$SEEDS"
    fi
done <<< "$KEYS_RAW"

echo "----------------------------------------------------------"
echo "Submitted $N single-seed diagnostic jobs."
[[ "$DRY_RUN" == "1" ]] && echo "(DRY_RUN: nothing was actually submitted)"
