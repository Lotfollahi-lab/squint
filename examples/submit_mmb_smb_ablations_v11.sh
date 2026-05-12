#!/usr/bin/env bash
# submit_mmb_smb_ablations_v11.sh
# -----------------------------------------------------------------------------
# Submits the single variant in
# `DATASET_VARIANTS["mmb0-1b_smb1-1b_1p-ablations-v11"]` as one LSF job.
# Queue + cost-code group are accepted as flags so you can switch
# between (queue, group) pairings without editing the script.
#
# Defaults: --queue training-parallel --group s10396.
#
# v11 = SMOKE-TEST for the updated DataLoader config:
#
#   Base config now sets in `make_train_config_dualvq().datamodule.loader_params`:
#     - num_workers        = 8        (was implicitly 0)
#     - persistent_workers = True
#     - pin_memory         = True
#
#   And the LSF submitters now request:
#     - LSF_CORES = 10                (was 6; 8 workers + main + headroom)
#
# Purpose of this sweep: run ONE representative variant to confirm that
#   - the DataLoader kwargs are accepted by PyG's NeighborLoader (no
#     `TypeError: __init__() got an unexpected keyword argument` etc.),
#   - 8 async worker processes can fork without OOM under the standard
#     LSF_MEM_MB=128000 budget,
#   - GPU utilization actually flattens out (check `nvidia-smi -l 1`
#     while the job is running),
#   - training completes within wallclock and produces the expected
#     metrics CSVs.
#
# Once v11 passes, re-submit the heavier sweeps (v7/v8/v9/v10) with
# confidence that the DataLoader path won't regress.
#
# Variant picked (incidental — any variant would do; this one is the
# v9 medium baseline):
#   dualvq+small-h64+rvq-both-3level-30-20-10+decoder-cov+adv+mmb0-1b_smb1-1b_1p
#
# Usage:
#   bash examples/submit_mmb_smb_ablations_v11.sh [OPTIONS]
#
# Options:
#   --queue / -q QUEUE   LSF queue to submit to.
#                        Default: training-parallel.
#                        Also overridable via LSF_QUEUE env var.
#   --group / -g GROUP   LSF cost-code group (`bsub -G`).
#                        Default: s10396.
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
# that script (LSF_WALL, LSF_MEM_MB, LSF_CORES, DRY_RUN, ...) still
# work — pass them as env vars, e.g.:
#
#   # Dry-run first (recommended for a smoke test):
#   DRY_RUN=1 bash examples/submit_mmb_smb_ablations_v11.sh
#
#   # Tighter wallclock (smoke test should finish quickly):
#   LSF_WALL=6:00 bash examples/submit_mmb_smb_ablations_v11.sh
#
#   # Bump cores further if you want to test more workers later:
#   LSF_CORES=16 bash examples/submit_mmb_smb_ablations_v11.sh
#
#   # Switch queue + group at submission time:
#   bash examples/submit_mmb_smb_ablations_v11.sh \
#       --queue inference --group s10396
#
# Verifying the DataLoader change took effect:
#   1. While the job is running, `bjobs -l <JOBID>` will show -n 10.
#   2. ssh to the compute node, `nvidia-smi -l 1` — utilization should
#      stay 70-95% (vs the spikes-to-0 pattern under num_workers=0).
#   3. After the run finishes, check `<run_dir>/user_specified_config.yaml`
#      — the loader_params block should contain num_workers/persistent_workers/pin_memory.
#
# Logs land at:
#   /nfs/team361/sb75/squint-reproducibility/artifacts/logs/
#       mmb0-1b_smb1-1b_1p-ablations-v11/<variant>.{out,err}
# Per-job artifacts at:
#   /nfs/team361/sb75/squint-reproducibility/artifacts/
#       mmb0-1b_smb1-1b_1p/<variant>/<timestamp>/
# -----------------------------------------------------------------------------

set -euo pipefail

# Defaults — same (queue, group) pair as the v10 wrapper because that's
# the combo that's valid end-to-end on this cluster (sXXXX groups are
# rejected on non-AI queues).
DEFAULT_QUEUE="training-parallel"
DEFAULT_GROUP="s10396"

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
            # including) the first non-comment line.
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
# fallbacks pick them up. Always export so the resolved values are
# visible in the submitted job's environment (useful for debugging).
export LSF_QUEUE="$QUEUE"
export LSF_GROUP="$GROUP"

echo "[v11 smoke-test] LSF_QUEUE = $LSF_QUEUE"
echo "[v11 smoke-test] LSF_GROUP = $LSF_GROUP"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# `exec bash submit_dataset_sweep.sh DATASET_KEY [PASSTHROUGH...]`. The
# array-expansion idiom `${arr[@]+"${arr[@]}"}` is the bash-set-u-safe
# way to expand a possibly-empty array — a bare `"${arr[@]}"` would
# trip `set -u` when the array has no elements.
exec bash "$SCRIPT_DIR/submit_dataset_sweep.sh" \
    "mmb0-1b_smb1-1b_1p-ablations-v11" \
    ${PASSTHROUGH_ARGS[@]+"${PASSTHROUGH_ARGS[@]}"}
