#!/usr/bin/env bash
# submit_dataset_sweep.sh
# -----------------------------------------------------------------------------
# Submit one LSF job per variant in a DATASET_VARIANTS group, so an entire
# sweep / ablation matrix runs in parallel across nodes (one GPU exclusive
# per job).
#
# Usage:
#   bash examples/submit_dataset_sweep.sh <DATASET_KEY> [OPTIONS]
#
# Required:
#   DATASET_KEY   — A key from DATASET_VARIANTS in run_squint.py.
#                   See `python examples/run_squint.py --list-dataset-sweeps`.
#                   Examples:
#                     mmb0-1b_smb1-1b_1p-ablations          (8 variants on the 1-layer spine)
#                     mmb0-1b_smb1-1b_1p         (10-axis sweep on mmb-smb)
#                     chl59-8b_1p                (10-axis sweep on chl59)
#                     mmb0-1b_smb1-20b_1p        (10-axis sweep on mmb20)
#
# Optional environment overrides (override defaults on the command line):
#   VENV_PATH     — Python venv to activate inside each job
#                   (default: /nfs/team361/sb75/.venvs/squint)
#   SQUINT_REPO   — squint repo root (auto-detected from this script's dir)
#   LOG_ROOT      — Where to write LSF stdout/stderr per variant
#                   (default: /nfs/team361/sb75/squint-reproducibility/artifacts/logs)
#   LSF_GROUP     — bsub -G group              (default: s10396)
#   LSF_QUEUE     — bsub -q queue              (default: training-parallel)
#   LSF_CORES     — bsub -n / -R span ptile    (default: 20)
#   LSF_MEM_MB    — bsub -M / mem rusage in MB (default: 128000 = 128 GB)
#   LSF_GPU       — bsub -gpu spec             (default: mode=exclusive_process:num=1:block=yes)
#   LSF_WALL      — bsub -W wallclock          (default: 24:00 = 24 h)
#   DRY_RUN       — set to "1" to print bsub commands without submitting
#
# Examples:
#   # Submit the 8 mmb-smb ablations:
#   bash examples/submit_dataset_sweep.sh mmb0-1b_smb1-1b_1p-ablations
#
#   # Submit the chl59 10-axis sweep with 200 GB and 48 h walltime:
#   LSF_MEM_MB=200000 LSF_WALL=48:00 \
#     bash examples/submit_dataset_sweep.sh chl59-8b_1p
#
#   # Dry-run to see what will be submitted:
#   DRY_RUN=1 bash examples/submit_dataset_sweep.sh mmb0-1b_smb1-1b_1p-ablations
#
# Logs:
#   stdout -> $LOG_ROOT/$DATASET_KEY/<variant>.out
#   stderr -> $LOG_ROOT/$DATASET_KEY/<variant>.err
# Per-job artifacts (checkpoints, predicted_adata.h5ad, plots, metrics) go
# to <ARTIFACTS_DIR>/<dataset_tag>/<variant>/<timestamp>/ as configured by
# run_squint.py itself.
# -----------------------------------------------------------------------------

set -euo pipefail

DATASET_KEY="${1:-}"

if [[ -z "$DATASET_KEY" ]]; then
    sed -n '4,40p' "$0"   # echo the usage block above
    exit 1
fi

# Auto-detect the squint repo root from this script's location, but allow
# the caller to override (useful for testing from a different checkout).
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SQUINT_REPO="${SQUINT_REPO:-"$( cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd )"}"

VENV_PATH="${VENV_PATH:-/nfs/team361/sb75/.venvs/squint}"
LOG_ROOT="${LOG_ROOT:-/nfs/team361/sb75/squint-reproducibility/artifacts/logs}"

LSF_GROUP="${LSF_GROUP:-s10396}"
LSF_QUEUE="${LSF_QUEUE:-training-parallel}"
# 20 cores per job: leaves room for the 16 DataLoader workers configured
# in `make_train_config_dualvq().datamodule.loader_params.num_workers`,
# the main training thread, plus ~3 cores of headroom for the BLAS /
# NCCL threads PyTorch spins up. Drop to 16 (workers + main + 0 spare)
# or even 10 (legacy default) if a queue is selective about job size.
LSF_CORES="${LSF_CORES:-20}"
LSF_MEM_MB="${LSF_MEM_MB:-128000}"
LSF_GPU="${LSF_GPU:-mode=exclusive_process:num=1:block=yes}"
LSF_WALL="${LSF_WALL:-24:00}"
DRY_RUN="${DRY_RUN:-0}"

# --- Resolve the variant list -------------------------------------------------
# `--variants-for-dataset` prints one variant per line, machine-readable.
# We must source the venv to invoke run_squint.py (it imports torch, anndata,
# etc. at module load time).
if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
    echo "ERROR: VENV_PATH=$VENV_PATH is not a Python venv (no bin/activate)." >&2
    exit 1
fi

# shellcheck disable=SC1091
source "$VENV_PATH/bin/activate"
VARIANTS="$(python "$SQUINT_REPO/examples/run_squint.py" \
                --variants-for-dataset "$DATASET_KEY")"
deactivate

if [[ -z "$VARIANTS" ]]; then
    echo "ERROR: No variants returned for DATASET_KEY=$DATASET_KEY." >&2
    echo "Run: python examples/run_squint.py --list-dataset-sweeps" >&2
    exit 1
fi

N_VARIANTS=$(echo "$VARIANTS" | wc -l | tr -d ' ')
LOG_DIR="$LOG_ROOT/$DATASET_KEY"
mkdir -p "$LOG_DIR"

echo "================================================================"
echo "Dataset key : $DATASET_KEY"
echo "Variants    : $N_VARIANTS"
echo "Repo        : $SQUINT_REPO"
echo "Venv        : $VENV_PATH"
echo "Log dir     : $LOG_DIR"
echo "Resources   : -G $LSF_GROUP -q $LSF_QUEUE -n $LSF_CORES "
echo "              -M $LSF_MEM_MB (rusage[mem=$LSF_MEM_MB]) -W $LSF_WALL"
echo "              -gpu '$LSF_GPU'"
echo "Dry run     : $DRY_RUN"
echo "================================================================"

# --- Submit one bsub per variant ---------------------------------------------
# The job body sources the venv inside the job (LSF jobs start with a clean
# shell) and invokes run_squint.py --all so every variant goes through the
# full train -> predict -> plots -> metrics pipeline, with all artifacts
# landing in the run_dir for that variant.
i=0
while IFS= read -r V; do
    [[ -z "$V" ]] && continue
    i=$((i + 1))
    JOB_NAME="squint-${V}"
    LOG_OUT="$LOG_DIR/${V}.out"
    LOG_ERR="$LOG_DIR/${V}.err"

    BSUB_CMD=(
        bsub
        -G "$LSF_GROUP"
        -q "$LSF_QUEUE"
        -n "$LSF_CORES"
        -M "$LSF_MEM_MB"
        -R "select[mem>$LSF_MEM_MB] rusage[mem=$LSF_MEM_MB]"
        -R "span[ptile=$LSF_CORES]"
        -gpu "$LSF_GPU"
        -W "$LSF_WALL"
        -J "$JOB_NAME"
        -o "$LOG_OUT"
        -e "$LOG_ERR"
    )

    # The job body lives in a separate runner script (_run_one_variant.sh)
    # rather than being inlined here. Inlining via `bsub ... bash -c "<body>"`
    # mangles single quotes inside the body (e.g. `$(date '+%F %T')`),
    # which broke earlier attempts. Keeping it in a real file dodges that
    # entirely — bsub just gets argv pointing at the script + variant name.
    RUNNER="$SCRIPT_DIR/_run_one_variant.sh"
    # We invoke as `bash $RUNNER`, so the file just needs to exist —
    # the executable bit (`+x`) is not required and may be lost across
    # cluster transfers (rsync/scp without `-p`, fresh git clones, etc.).
    if [[ ! -f "$RUNNER" ]]; then
        echo "ERROR: missing runner script at $RUNNER" >&2
        exit 1
    fi

    echo "[$i/$N_VARIANTS] $V"
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '    %q ' "${BSUB_CMD[@]}"
        printf '%s %q %q %q\n' "bash" "$RUNNER" "$V" "$VENV_PATH"
    else
        "${BSUB_CMD[@]}" bash "$RUNNER" "$V" "$VENV_PATH" "$SQUINT_REPO"
    fi
done <<< "$VARIANTS"

echo "================================================================"
echo "Submitted $i jobs."
echo "Watch with: bjobs"
echo "Logs       : $LOG_DIR/<variant>.{out,err}"
echo "================================================================"
