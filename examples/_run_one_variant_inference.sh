#!/usr/bin/env bash
# _run_one_variant_inference.sh
# -----------------------------------------------------------------------------
# Runner for re-running inference (steps 2-6 of the run_squint.py pipeline)
# against an already-trained checkpoint. Sibling of `_run_one_variant.sh`,
# which does the FULL train+inference pipeline.
#
# Activates the squint venv, sets up the same wandb / TMPDIR / LD_LIBRARY_PATH
# scaffolding as `_run_one_variant.sh`, then calls:
#
#     python examples/run_inference.py --variant <V> --timestamp <TS> $EXTRA
#
# Usage (normally invoked by submit_inference_for_existing_runs.sh, but
# works standalone):
#   bash examples/_run_one_variant_inference.sh <VARIANT> <TIMESTAMP> [VENV_PATH] [SQUINT_REPO]
#
# Args:
#   VARIANT      — required. Registered variant name (must exist in VARIANTS).
#   TIMESTAMP    — required. Run timestamp (`YYYYMMDD_HHMMSS` folder name
#                  under <ARTIFACTS>/<dataset_tag>/<variant>/).
#   VENV_PATH    — optional. Default: /nfs/team361/sb75/.venvs/squint
#   SQUINT_REPO  — optional. Default: /nfs/team361/sb75/squint
#
# Extra flags forwarded to run_inference.py (env-driven so the LSF
# submitter can wire them through without rewriting this file):
#   INFERENCE_EXTRA_ARGS — space-separated list of additional CLI args
#                          (e.g. "--skip-umap --silver-dir /alt/path").
# -----------------------------------------------------------------------------

set -euo pipefail

VARIANT="${1:?usage: $0 VARIANT TIMESTAMP [VENV_PATH] [SQUINT_REPO]}"
TIMESTAMP="${2:?usage: $0 VARIANT TIMESTAMP [VENV_PATH] [SQUINT_REPO]}"
VENV_PATH="${3:-/nfs/team361/sb75/.venvs/squint}"
SQUINT_REPO="${4:-/nfs/team361/sb75/squint}"

# Defensive: scrub PYTHONPATH/PYTHONHOME (see _run_one_variant.sh).
unset PYTHONPATH PYTHONHOME

# Force TMPDIR to node-local /tmp/$USER. Multiprocessing tempdirs on
# NFS hit `.nfsXXXX` silly-rename races at process exit ("OSError:
# [Errno 16] Device or resource busy"), which clobber the inference
# loop with cleanup tracebacks even when the actual work succeeded.
# Setting TMPDIR here (BEFORE the WANDB_BASE/tmp default below) means
# every subprocess (DataLoader workers, rapids UMAP, etc.) puts its
# tempdirs on local SSD instead of NFS.
export TMPDIR="/tmp/${USER:-$(id -un)}"
mkdir -p "$TMPDIR"

# Wandb / TMPDIR redirects identical to the train runner. predict() doesn't
# upload artifacts, but the rapids UMAP subprocess (when not skipped) and
# subprocess scripts can still write to TMPDIR; keep them off $HOME.
WANDB_BASE="${WANDB_BASE:-/nfs/team361/sb75/squint-reproducibility/artifacts/wandb_cache}"
mkdir -p "$WANDB_BASE/run" "$WANDB_BASE/cache" "$WANDB_BASE/artifacts" "$WANDB_BASE/tmp"
export WANDB_DIR="${WANDB_DIR:-$WANDB_BASE/run}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$WANDB_BASE/cache}"
export WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-$WANDB_BASE/artifacts}"
export TMPDIR="${TMPDIR:-$WANDB_BASE/tmp}"

echo "[$(date '+%F %T')] activating venv: $VENV_PATH"
# shellcheck disable=SC1091
source "$VENV_PATH/bin/activate"

# torch's bundled CUDA libs need to be on LD_LIBRARY_PATH (see
# _run_one_variant.sh for full rationale).
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
echo "[$(date '+%F %T')] starting inference: variant=$VARIANT timestamp=$TIMESTAMP"
echo "  host         : $(hostname)"
echo "  CUDA devices : ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  cwd          : $(pwd)"
echo "  extra args   : ${INFERENCE_EXTRA_ARGS:-<none>}"

# `INFERENCE_EXTRA_ARGS` is intentionally unquoted so it splits into argv
# tokens on whitespace.
# shellcheck disable=SC2086
python "$SQUINT_REPO/examples/run_inference.py" \
    --variant "$VARIANT" \
    --timestamp "$TIMESTAMP" \
    ${INFERENCE_EXTRA_ARGS:-}

echo "[$(date '+%F %T')] finished inference: variant=$VARIANT timestamp=$TIMESTAMP"
