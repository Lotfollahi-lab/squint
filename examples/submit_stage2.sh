#!/usr/bin/env bash
# submit_stage2.sh
# -----------------------------------------------------------------------------
# Submit a single GPU LSF job that trains the SQUINT stage-2 spatial code prior
# (vqniche.stage2) on a frozen predicted_adata.h5ad, then runs a region-holdout
# in-painting evaluation. Mirrors the env / LSF conventions of
# submit_multi_seed.sh (venv at /nfs/team361/sb75/.venvs/squint, gpu-lotfollahi,
# group team361, one exclusive GPU).
#
# Usage:
#   bash examples/submit_stage2.sh <PREDICTED_ADATA> [-- <run_stage2.py args>]
#
#   bash examples/submit_stage2.sh \
#     /nfs/team361/sb75/squint-reproducibility/artifacts/mmb0-1b_smb1-1b_1p/<variant>/<TS>_seed0/predicted_adata.h5ad \
#     -- --smoke
#
# Everything after `--` is forwarded verbatim to run_stage2.py, e.g.
#   -- --max-steps 30000 --d-model 256 --patch-size 1024 --batch-size 8
#   -- --smoke                     (tiny fast end-to-end sanity run)
#
# Environment overrides (same names as submit_multi_seed.sh):
#   VENV_PATH    /nfs/team361/sb75/.venvs/squint
#   SQUINT_REPO  <auto: this script's repo root>
#   LOG_ROOT     /nfs/team361/sb75/squint-reproducibility/artifacts/logs
#   LSF_GROUP    team361        (-G; project cost-code group for the GPU queue)
#   LSF_QUEUE    gpu-lotfollahi (-q)
#   LSF_CORES    16
#   LSF_MEM_MB   128000
#   LSF_GPU      mode=exclusive_process:num=1:block=yes
#   LSF_WALL     12:00
#   DRY_RUN      0  (set 1 to print the bsub instead of submitting)
# -----------------------------------------------------------------------------
set -euo pipefail

if [[ $# -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    sed -n '3,40p' "$0"
    exit 0
fi

PREDICTED_ADATA="$1"; shift
# allow an optional `--` separator before forwarded args
if [[ "${1:-}" == "--" ]]; then shift; fi
FORWARD_ARGS=("$@")

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SQUINT_REPO="${SQUINT_REPO:-"$( cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd )"}"

VENV_PATH="${VENV_PATH:-/nfs/team361/sb75/.venvs/squint}"
LOG_ROOT="${LOG_ROOT:-/nfs/team361/sb75/squint-reproducibility/artifacts/logs}"

LSF_GROUP="${LSF_GROUP:-team361}"
LSF_QUEUE="${LSF_QUEUE:-gpu-lotfollahi}"
LSF_CORES="${LSF_CORES:-16}"
LSF_MEM_MB="${LSF_MEM_MB:-128000}"
LSF_GPU="${LSF_GPU:-mode=exclusive_process:num=1:block=yes}"
LSF_WALL="${LSF_WALL:-12:00}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -f "$PREDICTED_ADATA" ]]; then
    echo "ERROR: predicted_adata not found: $PREDICTED_ADATA" >&2
    exit 1
fi
if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
    echo "ERROR: VENV_PATH=$VENV_PATH is not a Python venv (no bin/activate)." >&2
    exit 1
fi

TS="$(date +%Y%m%d_%H%M%S)"
LSF_LOG_DIR="$LOG_ROOT/stage2/$TS"
mkdir -p "$LSF_LOG_DIR"
JOB_NAME="squint-stage2-$TS"
LOG_OUT="$LSF_LOG_DIR/stage2.out"
LOG_ERR="$LSF_LOG_DIR/stage2.err"

# The job command: activate venv, then run the trainer.
read -r -d '' JOB_CMD <<EOF || true
set -euo pipefail
source "$VENV_PATH/bin/activate"
cd "$SQUINT_REPO"
python examples/run_stage2.py --predicted-adata "$PREDICTED_ADATA" ${FORWARD_ARGS[*]:-}
EOF

echo "=========================================================="
echo "Stage-2 training submission"
echo "  predicted_adata : $PREDICTED_ADATA"
echo "  forwarded args  : ${FORWARD_ARGS[*]:-<none>}"
echo "  repo            : $SQUINT_REPO"
echo "  venv            : $VENV_PATH"
echo "  LSF             : -G $LSF_GROUP -q $LSF_QUEUE -n $LSF_CORES -M $LSF_MEM_MB"
echo "                    -gpu '$LSF_GPU' -W $LSF_WALL"
echo "  logs            : $LOG_OUT"
echo "=========================================================="

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

if [[ "$DRY_RUN" == "1" ]]; then
    printf '%q ' "${BSUB_CMD[@]}"; printf 'bash -lc %q\n' "$JOB_CMD"
else
    "${BSUB_CMD[@]}" bash -lc "$JOB_CMD"
    echo "[submitted] $JOB_NAME  (tail -f $LOG_OUT)"
fi
