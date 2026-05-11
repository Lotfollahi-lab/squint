#!/usr/bin/env bash
# submit_mmb_smb_ablations_v9.sh
# -----------------------------------------------------------------------------
# Submits all 8 variants in
# `DATASET_VARIANTS["mmb0-1b_smb1-1b_1p-ablations-v9"]` as separate LSF
# jobs (one bsub per variant, one GPU exclusive each). Queue +
# cost-code group are accepted as flags so you can switch between
# (queue, group) pairings without editing the script.
#
# Defaults: --queue gpu-lotfollahi --group team361 (the AI/GPU queue
# with the team-project cost code).
#
# v9 design: SMALLER + FASTER architecture, 2 x 4 grid.
#
#   architectural size   ∈ { "compact"  (h32 + codebook (30, 10, 10)),
#                            "medium"   (h64 + codebook (30, 20, 10)) }
#
#   batch integration    ∈ { adv (wt=150),         # default
#                            adv-w300,             # boosted adv
#                            adv + mmd-w50,        # combo (adv + MMD)
#                            mmd-w100 (no adv) }   # MMD only
#
# Shared spine (NO +wide so 1-hop GNN/sampler=[8]/nbr_hops=1):
#   +small(hidden=N)                       # encoder + GNN + decoders -> N
#   +rvq(branch=both, levels=[30, L1, L2]) # 3-level RVQ symmetric
#   +decoder_covariate                     # NicheCompass-shape batch slot
#
# Concrete variant list (in submission order):
#   Compact (h32 + (30, 10, 10)):
#     1. +adv                            # default pressure
#     2. +adv-w300                       # boosted adv
#     3. +adv +mmd-w50                   # combo
#     4. +mmd-w100                       # MMD only (no adv)
#   Medium (h64 + (30, 20, 10)):
#     5. +adv                            # default pressure
#     6. +adv-w300                       # boosted adv
#     7. +adv +mmd-w50                   # combo
#     8. +mmd-w100                       # MMD only (no adv)
#
# Usage:
#   bash examples/submit_mmb_smb_ablations_v9.sh [OPTIONS]
#
# Options:
#   --queue / -q QUEUE   LSF queue (default: gpu-lotfollahi).
#                        Also overridable via LSF_QUEUE env var.
#   --group / -g GROUP   LSF cost-code group (default: team361).
#                        Also overridable via LSF_GROUP env var.
#   --help  / -h         Print this usage block and exit.
#
# Precedence: CLI flag > env var > script default.
#
# Other env-var overrides from `submit_dataset_sweep.sh` work too:
#
#   # Dry-run first (recommended) — prints bsub commands without submitting:
#   DRY_RUN=1 bash examples/submit_mmb_smb_ablations_v9.sh
#
#   # Force a longer wallclock or more memory:
#   LSF_WALL=48:00 LSF_MEM_MB=200000 \
#       bash examples/submit_mmb_smb_ablations_v9.sh
#
#   # Opt into per-epoch Pearson metrics (~10-20x slower epochs):
#   SQUINT_WITH_PEARSON=1 bash examples/submit_mmb_smb_ablations_v9.sh
#
#   # Switch queue + group at submission time:
#   bash examples/submit_mmb_smb_ablations_v9.sh \
#       --queue training-parallel --group s10396
#
# Logs land at:
#   /nfs/team361/sb75/squint-reproducibility/artifacts/logs/
#       mmb0-1b_smb1-1b_1p-ablations-v9/<variant>.{out,err}
# Per-job artifacts at:
#   /nfs/team361/sb75/squint-reproducibility/artifacts/
#       mmb0-1b_smb1-1b_1p/<variant>/<timestamp>/
# -----------------------------------------------------------------------------

set -euo pipefail

# Defaults — same (queue, group) pair as the v10 / v11 wrappers.
DEFAULT_QUEUE="gpu-lotfollahi"
DEFAULT_GROUP="team361"

# Seed working values from env vars (env still works for the legacy
# call-style), then let CLI flags override below.
QUEUE="${LSF_QUEUE:-$DEFAULT_QUEUE}"
GROUP="${LSF_GROUP:-$DEFAULT_GROUP}"

PASSTHROUGH_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --queue|-q)
            shift
            if [[ $# -eq 0 ]]; then
                echo "ERROR: --queue / -q requires a value." >&2
                exit 2
            fi
            QUEUE="$1"
            shift
            ;;
        --queue=*)
            QUEUE="${1#--queue=}"
            shift
            ;;
        --group|-g)
            shift
            if [[ $# -eq 0 ]]; then
                echo "ERROR: --group / -g requires a value." >&2
                exit 2
            fi
            GROUP="$1"
            shift
            ;;
        --group=*)
            GROUP="${1#--group=}"
            shift
            ;;
        --help|-h)
            awk '/^[^#]/ {exit} {print}' "$0"
            exit 0
            ;;
        *)
            PASSTHROUGH_ARGS+=("$1")
            shift
            ;;
    esac
done

export LSF_QUEUE="$QUEUE"
export LSF_GROUP="$GROUP"

echo "[v9 ablations] LSF_QUEUE = $LSF_QUEUE"
echo "[v9 ablations] LSF_GROUP = $LSF_GROUP"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
exec bash "$SCRIPT_DIR/submit_dataset_sweep.sh" \
    "mmb0-1b_smb1-1b_1p-ablations-v9" \
    ${PASSTHROUGH_ARGS[@]+"${PASSTHROUGH_ARGS[@]}"}
