#!/usr/bin/env bash
# _run_umap_rapids.sh
# -----------------------------------------------------------------------------
# Run plot_latent_umap.py inside a separate conda env that has
# rapids-singlecell installed (GPU UMAP), while the rest of the pipeline
# stays in the squint uv venv. Invoked by run_inference_and_analysis()
# in run_squint.py for the [5/6] plot_latent_umap step.
#
# This wrapper exists because:
#   - The squint training/predict env is a uv-managed venv pinned to
#     torch 2.2 + pyg 2.5; mixing rapids-singlecell into it is fragile
#     (CUDA-lib version conflicts).
#   - `conda activate` requires `module load cellgen/conda` first on
#     this cluster — that won't happen automatically inside an LSF job
#     or a clean subshell.
#   - LSF jobs start with empty modulefiles state.
#
# Usage (mostly invoked by run_squint.py, but works standalone):
#   bash examples/_run_umap_rapids.sh <args-forwarded-to-plot_latent_umap.py>
#
# Env overrides:
#   RAPIDS_ENV       — full path to the conda env prefix
#                      (default: /nfs/team361/sb75/ENVS/rapids-singlecell)
#   CONDA_PKGS_DIRS  — conda package cache dir
#                      (default: /nfs/team361/sb75/.conda_pkgs)
# -----------------------------------------------------------------------------

set -euo pipefail

RAPIDS_ENV="${RAPIDS_ENV:-/nfs/team361/sb75/ENVS/rapids-singlecell}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/nfs/team361/sb75/.conda_pkgs}"

if [[ ! -d "$RAPIDS_ENV" ]]; then
    echo "ERROR: rapids env not found at $RAPIDS_ENV" >&2
    echo "       create it via:" >&2
    echo "         source /etc/profile.d/modules.sh" >&2
    echo "         module load cellgen/conda" >&2
    echo "         conda env create -f rsc_rapids_25.10.yml -p $RAPIDS_ENV" >&2
    exit 1
fi

# Bring in the cluster's `module` function + the conda hook.
# shellcheck disable=SC1091
source /etc/profile.d/modules.sh
module load cellgen/conda

# `conda activate` needs the shell hook initialised. `module load
# cellgen/conda` should source `conda init` for this shell. If the
# function isn't on $PATH afterwards, fall back to the env's bin/python
# directly (loses any per-env activation hooks but usually still works
# because the conda env's lib/ is on LD_LIBRARY_PATH via the activate
# script which conda runs internally on `conda activate`).
if command -v conda &>/dev/null; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$RAPIDS_ENV"
fi

echo "[$(date '+%F %T')] rapids env: $RAPIDS_ENV"
echo "  which python : $(which python)"
echo "  python -V    : $(python --version 2>&1)"

# Forward all CLI args to plot_latent_umap.py.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
exec python "$SCRIPT_DIR/plot_latent_umap.py" "$@"
