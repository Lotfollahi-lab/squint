#!/usr/bin/env bash
# submit_mmb_smb_ablations_v10.sh
# -----------------------------------------------------------------------------
# Submits all 4 variants in
# `DATASET_VARIANTS["mmb0-1b_smb1-1b_1p-ablations-v10"]` as separate LSF
# jobs. Queue + cost-code group are accepted as flags so you can switch
# between (queue, group) pairings without editing the script.
#
# Defaults: --queue gpu-lotfollahi --group team361 (the AI/GPU queue the
# user has access to with the team-project cost code).
#
# v10 = "is spatial supervision hurting cell-type NMI?" DIAGNOSTIC.
#
# 4 variants on a fixed MEDIUM architecture
# (small(hidden=64), codebook (30, 20, 10), decoder-covariate),
# crossed over:
#
#   batch integration       ∈ { adv (wt=150),  mmd-w100 (no adv) }
#   spatial-loss treatment  ∈ { +no-spatial,  +no-adj }
#
# Spatial-loss treatments:
#   - `+no-spatial`: drops EVERY spatial supervision term — niche NB
#                    reconstruction, niche VQ commit, AND adjacency BCE.
#                    Niche-side modules still run forward but receive
#                    zero gradient. Trains as a pure cell-branch VQ-VAE
#                    + (adversarial or MMD) batch integration, matching
#                    scVI / Harmony shape on the cell side.
#   - `+no-adj`:     keeps niche NB reconstruction + niche VQ commit,
#                    drops ONLY the cosine adjacency BCE. The niche
#                    branch still gets gradient from neighborhood
#                    reconstruction — only the topology-encoding
#                    pressure is removed.
#
# Architectural size is held constant (h64 + (30, 20, 10)) so the
# diagnostic isolates the effect of LOSSES on cell-type NMI. The size
# was chosen to be small enough to be fast, but not so small that
# capacity becomes the limiting factor.
#
# Interpretation of the resulting cell-NMI:
#   - +no-adj alone recovers cell-NMI -> adjacency BCE is the specific
#       culprit; niche NB recon is fine.
#   - Only +no-spatial recovers -> ALL spatial supervision contributes;
#       the full two-stage train is the right next step.
#   - Neither recovers -> bottleneck is elsewhere (codebook structure,
#       encoder capacity, adversary calibration). Don't build two-stage.
#
# Concrete variant list (in submission order):
#   1. h64 + (30,20,10) + adv      + no-spatial
#   2. h64 + (30,20,10) + mmd-w100 + no-spatial
#   3. h64 + (30,20,10) + adv      + no-adj
#   4. h64 + (30,20,10) + mmd-w100 + no-adj
#
# Usage:
#   bash examples/submit_mmb_smb_ablations_v10.sh [OPTIONS]
#
# Options:
#   --queue / -q QUEUE   LSF queue to submit to.
#                        Default: gpu-lotfollahi.
#                        Also overridable via LSF_QUEUE env var.
#   --group / -g GROUP   LSF cost-code group (`bsub -G`).
#                        Default: team361.
#                        Also overridable via LSF_GROUP env var.
#   --help  / -h         Print this usage block and exit.
#
# Precedence for the queue / group values (highest wins):
#   1. --queue / --group flag on the CLI
#   2. LSF_QUEUE / LSF_GROUP env var
#   3. The defaults in this script
#
# Any flag this script doesn't recognise is forwarded verbatim to
# `submit_dataset_sweep.sh`, so the other env-var-driven knobs from
# that script (LSF_WALL, LSF_MEM_MB, DRY_RUN, RAPIDS_ENV, ...) still
# work — pass them as env vars, e.g.:
#
#   DRY_RUN=1 bash examples/submit_mmb_smb_ablations_v10.sh
#
#   LSF_WALL=48:00 LSF_MEM_MB=200000 \
#       bash examples/submit_mmb_smb_ablations_v10.sh
#
#   # Switch queue + group at submission time:
#   bash examples/submit_mmb_smb_ablations_v10.sh \
#       --queue inference --group s10396
#
#   # Or via env vars (equivalent to the flags above):
#   LSF_QUEUE=inference LSF_GROUP=s10396 \
#       bash examples/submit_mmb_smb_ablations_v10.sh
#
# Logs land at:
#   /nfs/team361/sb75/squint-reproducibility/artifacts/logs/
#       mmb0-1b_smb1-1b_1p-ablations-v10/<variant>.{out,err}
# Per-job artifacts at:
#   /nfs/team361/sb75/squint-reproducibility/artifacts/
#       mmb0-1b_smb1-1b_1p/<variant>/<timestamp>/
# -----------------------------------------------------------------------------

set -euo pipefail

# Defaults — chosen for the user's standard team361 / gpu-lotfollahi
# pairing. The cluster rejects sXXXX-group jobs on `gpu-basement` (it's
# not an AI-acceleration queue), so we default to a (queue, group) pair
# that's actually valid end-to-end rather than a queue alone.
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
            # Print everything from the first '#' line up to (but not
            # including) the first non-comment line, so users get the
            # usage block without having to open the file.
            awk '/^[^#]/ {exit} {print}' "$0"
            exit 0
            ;;
        *)
            # Any other argument is forwarded verbatim to
            # submit_dataset_sweep.sh — keeps the door open for
            # future positional / flag args added downstream without
            # breaking this wrapper.
            PASSTHROUGH_ARGS+=("$1")
            shift
            ;;
    esac
done

# Export so submit_dataset_sweep.sh's `${LSF_QUEUE:-...}` / `${LSF_GROUP:-...}`
# fallbacks pick them up. We always export both even when they match the
# defaults so the chosen values are visible in the submitted job's
# environment (useful for debugging).
export LSF_QUEUE="$QUEUE"
export LSF_GROUP="$GROUP"

echo "[v10] LSF_QUEUE = $LSF_QUEUE"
echo "[v10] LSF_GROUP = $LSF_GROUP"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# `exec bash submit_dataset_sweep.sh DATASET_KEY [PASSTHROUGH...]`. The
# array-expansion idiom `${arr[@]+"${arr[@]}"}` is the bash-set-u-safe
# way to expand a possibly-empty array — a bare `"${arr[@]}"` would
# trip `set -u` when the array has no elements.
exec bash "$SCRIPT_DIR/submit_dataset_sweep.sh" \
    "mmb0-1b_smb1-1b_1p-ablations-v10" \
    ${PASSTHROUGH_ARGS[@]+"${PASSTHROUGH_ARGS[@]}"}
