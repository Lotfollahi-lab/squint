#!/usr/bin/env bash
# _run_one_variant.sh
# -----------------------------------------------------------------------------
# Runner script invoked by submit_dataset_sweep.sh inside each LSF job.
# Activates the squint venv and runs `run_squint.py --all --variant $1`,
# with diagnostic prints so each job's stdout makes it obvious which
# interpreter and host ran the job.
#
# Lives as a separate file (rather than inlined in submit_dataset_sweep.sh)
# because LSF's `bsub <cmd> "<inline shell>"` invocation mangles single
# quotes inside the body — putting the script in a file dodges that
# entirely.
#
# Usage (normally invoked by submit_dataset_sweep.sh, but works standalone):
#   bash examples/_run_one_variant.sh <VARIANT> [VENV_PATH] [SQUINT_REPO]
#
# Args:
#   VARIANT      — required. A registered variant name from VARIANTS in
#                  run_squint.py.
#   VENV_PATH    — optional. Path to the venv to activate.
#                  Default: /nfs/team361/sb75/.venvs/squint
#   SQUINT_REPO  — optional. Path to the squint repo root.
#                  Default: /nfs/team361/sb75/squint
# -----------------------------------------------------------------------------

set -euo pipefail

VARIANT="${1:?usage: $0 VARIANT [VENV_PATH] [SQUINT_REPO]}"
VENV_PATH="${2:-/nfs/team361/sb75/.venvs/squint}"
SQUINT_REPO="${3:-/nfs/team361/sb75/squint}"

# Defensive: scrub PYTHONPATH/PYTHONHOME from the parent environment before
# activating the venv. Stale values (e.g. from a conda init in the user's
# .bashrc, or a module that pre-sets PYTHONPATH) would silently shadow the
# venv's site-packages and break `import torch` even after activation.
unset PYTHONPATH PYTHONHOME

# ----------------------------------------------------------------------------
# Redirect wandb / tmp caches off $HOME — HPC home dirs are typically
# quota-limited (10–50 GB) and the run config has `log_model=True`, so
# Lightning hands wandb every "best" checkpoint to upload as an artifact.
# Wandb STAGES each .ckpt under $WANDB_CACHE_DIR (default
# $HOME/.cache/wandb/artifacts/staging/) before transmitting. Across a
# multi-variant sweep that overflows a typical home quota with
# `OSError: [Errno 28] No space left on device` deep inside
# `wandb/sdk/artifacts/artifact.py:_add_local_file`.
#
# Point all of wandb's caches + the generic TMPDIR at the project NFS
# (overridable via env). The locations:
#   - WANDB_DIR           : where wandb writes run dirs (logs, config)
#   - WANDB_CACHE_DIR     : staging area for artifacts (the offender)
#   - WANDB_ARTIFACT_DIR  : where downloaded artifacts cache
#   - TMPDIR              : torch / python temp files
# ----------------------------------------------------------------------------
WANDB_BASE="${WANDB_BASE:-/nfs/team361/sb75/squint-reproducibility/artifacts/wandb_cache}"
mkdir -p "$WANDB_BASE/run" "$WANDB_BASE/cache" "$WANDB_BASE/artifacts" "$WANDB_BASE/tmp"
export WANDB_DIR="${WANDB_DIR:-$WANDB_BASE/run}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$WANDB_BASE/cache}"
export WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-$WANDB_BASE/artifacts}"
export TMPDIR="${TMPDIR:-$WANDB_BASE/tmp}"

echo "[$(date '+%F %T')] activating venv: $VENV_PATH"
# shellcheck disable=SC1091
source "$VENV_PATH/bin/activate"
echo "  which python       : $(which python)"
echo "  python -V          : $(python --version 2>&1)"
echo "  WANDB_DIR          : $WANDB_DIR"
echo "  WANDB_CACHE_DIR    : $WANDB_CACHE_DIR"
echo "  WANDB_ARTIFACT_DIR : $WANDB_ARTIFACT_DIR"
echo "  TMPDIR             : $TMPDIR"

cd "$SQUINT_REPO"
echo "[$(date '+%F %T')] starting variant: $VARIANT"
echo "  host         : $(hostname)"
echo "  CUDA devices : ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  cwd          : $(pwd)"

python "$SQUINT_REPO/examples/run_squint.py" --all --variant "$VARIANT"

echo "[$(date '+%F %T')] finished variant: $VARIANT"
