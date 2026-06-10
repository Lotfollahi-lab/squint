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
#
# Env-var toggles (set in the calling shell or in the bsub command):
#   SQUINT_WITH_PLOTS=1   re-enable the code-index spatial / SVG-recon /
#                          latent-UMAP plot stages (skipped by default for
#                          ablation throughput). The metrics step always
#                          runs regardless.
#   SQUINT_WITH_PEARSON=1  run the Pearson metrics list every val epoch
#                          during training (forwarded to run_squint.py).
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

# Force TMPDIR to node-local /tmp/$USER. Multiprocessing tempdirs on
# NFS hit `.nfsXXXX` silly-rename races at process exit ("OSError:
# [Errno 16] Device or resource busy"), which clobber the training
# loop with cleanup tracebacks even after early-stopping fired and
# the run completed cleanly. Setting TMPDIR here (BEFORE the
# WANDB_BASE/tmp default below) means every subprocess (DataLoader
# workers, etc.) puts its tempdirs on local SSD instead of NFS.
# Sibling fix to the same patch in `_run_one_variant_inference.sh`.
export TMPDIR="/tmp/${USER:-$(id -un)}"
mkdir -p "$TMPDIR"

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

# ----------------------------------------------------------------------------
# Make torch's bundled CUDA shared libraries findable. Pip-installed torch +
# rapids ship the CUDA toolkit pieces under
# <venv>/lib/python*/site-packages/nvidia/<lib>/lib/, but the dynamic linker
# only finds them if the dirs are on LD_LIBRARY_PATH. Interactive shells on
# HPC usually inherit a populated LD_LIBRARY_PATH from `module load cuda/...`
# in ~/.bashrc; LSF jobs start clean so torch can't find e.g.
# libcusparse.so.12 and `import torch` (transitively triggered by
# `import anndata` in compute_inference_metrics.py) fails with:
#   ImportError: libcusparse.so.12: cannot open shared object file
# Discover the nvidia/*/lib dirs from the active venv and prepend them.
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
echo "  WANDB_CACHE_DIR    : $WANDB_CACHE_DIR"
echo "  WANDB_ARTIFACT_DIR : $WANDB_ARTIFACT_DIR"
echo "  TMPDIR             : $TMPDIR"
echo "  LD_LIBRARY_PATH    : $LD_LIBRARY_PATH"

cd "$SQUINT_REPO"
echo "[$(date '+%F %T')] starting variant: $VARIANT"
echo "  host         : $(hostname)"
echo "  CUDA devices : ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  cwd          : $(pwd)"

# Ablation sweeps consume the per-variant metric CSVs, not the
# spatial / SVG / UMAP plots. Skip those by default — UMAP alone
# typically takes 5-15 min per run and the plot rendering adds
# another 1-3 min, which compounds across a 24-variant sweep.
# Opt back in for individual debugging runs:
#   SQUINT_WITH_PLOTS=1 bash examples/_run_one_variant.sh <V>
# (or invoke `run_squint.py --all --variant <V>` directly without
# --metrics-only).
EXTRA_ARGS=()
if [[ "${SQUINT_WITH_PLOTS:-0}" == "1" ]]; then
    echo "  plots        : ENABLED (SQUINT_WITH_PLOTS=1)"
else
    echo "  plots        : skipped (--metrics-only); set SQUINT_WITH_PLOTS=1 to enable"
    EXTRA_ARGS+=( --metrics-only )
fi

python "$SQUINT_REPO/examples/run_squint.py" \
    --all --variant "$VARIANT" \
    "${EXTRA_ARGS[@]}"

echo "[$(date '+%F %T')] finished variant: $VARIANT"
