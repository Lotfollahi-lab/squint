#!/usr/bin/env bash
# _run_one_seed.sh
# -----------------------------------------------------------------------------
# Runner invoked by `submit_multi_seed.sh` inside each LSF job. One bsub
# per (variant, seed). Activates the squint venv and calls
# `_run_one_seed.py` with the seed override.
#
# Same env scaffolding as `_run_one_variant.sh` (PYTHONPATH scrub, TMPDIR
# off NFS, wandb cache redirect, torch CUDA-lib discovery) — the docstring
# in that sibling file has the full rationale.
#
# Usage (normally invoked by submit_multi_seed.sh, but works standalone):
#   bash examples/_run_one_seed.sh <VARIANT> <SEED> <OUT_DIR> \
#                                  [VENV_PATH] [SQUINT_REPO]
#
# Args:
#   VARIANT      — required. Registered variant name.
#   SEED         — required. Integer seed for this LSF job.
#   OUT_DIR      — required. Sweep-level output dir created by the
#                  submitter; each seed writes stamp files into
#                  `$OUT_DIR/seed_runs/seed_<N>_*.txt`.
#   VENV_PATH    — optional. Default: /nfs/team361/sb75/.venvs/squint
#   SQUINT_REPO  — optional. Default: /nfs/team361/sb75/squint
#
# Extra flags forwarded to `_run_one_seed.py` (env-driven so the submitter
# can pass them through without quoting magic):
#   SEED_EXTRA_ARGS — space-separated extra CLI args
#                     (e.g. "--skip-umap --skip-svg-plots").
# -----------------------------------------------------------------------------

set -euo pipefail

VARIANT="${1:?usage: $0 VARIANT SEED OUT_DIR [VENV_PATH] [SQUINT_REPO]}"
SEED="${2:?usage: $0 VARIANT SEED OUT_DIR [VENV_PATH] [SQUINT_REPO]}"
OUT_DIR="${3:?usage: $0 VARIANT SEED OUT_DIR [VENV_PATH] [SQUINT_REPO]}"
VENV_PATH="${4:-/nfs/team361/sb75/.venvs/squint}"
SQUINT_REPO="${5:-/nfs/team361/sb75/squint}"

# Scrub stale Python env (see _run_one_variant.sh).
unset PYTHONPATH PYTHONHOME

# TMPDIR off NFS — multiprocessing tempdirs hit `.nfsXXXX` silly-rename
# races on shared filesystems, which break train/predict at exit time
# even when the actual work succeeded.
export TMPDIR="/tmp/${USER:-$(id -un)}"
mkdir -p "$TMPDIR"

# Redirect wandb caches off $HOME — see _run_one_variant.sh for full
# rationale (HPC home quotas + log_model=True + per-seed checkpoint
# staging will overflow a typical 10–50 GB home quota).
WANDB_BASE="${WANDB_BASE:-/nfs/team361/sb75/squint-reproducibility/artifacts/wandb_cache}"
mkdir -p "$WANDB_BASE/run" "$WANDB_BASE/cache" "$WANDB_BASE/artifacts" "$WANDB_BASE/tmp"
export WANDB_DIR="${WANDB_DIR:-$WANDB_BASE/run}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$WANDB_BASE/cache}"
export WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-$WANDB_BASE/artifacts}"
export TMPDIR="${TMPDIR:-$WANDB_BASE/tmp}"

echo "[$(date '+%F %T')] activating venv: $VENV_PATH"
# shellcheck disable=SC1091
source "$VENV_PATH/bin/activate"

# Make torch's bundled CUDA shared libs findable; see _run_one_variant.sh.
NVIDIA_LIB_DIRS=$(python - <<'PY'
import glob, os, site
sp = site.getsitepackages()[0]
dirs = sorted(glob.glob(os.path.join(sp, "nvidia", "*", "lib")))
print(":".join(dirs))
PY
)
if [[ -n "$NVIDIA_LIB_DIRS" ]]; then
    export LD_LIBRARY_PATH="${NVIDIA_LIB_DIRS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

echo "  which python       : $(which python)"
echo "  python -V          : $(python --version 2>&1)"
echo "  WANDB_DIR          : $WANDB_DIR"
echo "  TMPDIR             : $TMPDIR"
echo "  LD_LIBRARY_PATH    : $LD_LIBRARY_PATH"

cd "$SQUINT_REPO"
echo "[$(date '+%F %T')] starting seed: variant=$VARIANT seed=$SEED"
echo "  host         : $(hostname)"
echo "  CUDA devices : ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  cwd          : $(pwd)"
echo "  out_dir      : $OUT_DIR"
echo "  extra args   : ${SEED_EXTRA_ARGS:-<none>}"

# Unquoted on purpose — splits SEED_EXTRA_ARGS into argv tokens.
# shellcheck disable=SC2086
python "$SQUINT_REPO/examples/_run_one_seed.py" \
    "$VARIANT" "$SEED" "$OUT_DIR" \
    ${SEED_EXTRA_ARGS:-}

echo "[$(date '+%F %T')] finished seed: variant=$VARIANT seed=$SEED"
