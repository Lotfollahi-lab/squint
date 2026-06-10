#!/usr/bin/env bash
# submit_single_run.sh
# -----------------------------------------------------------------------------
# Submit a SINGLE LSF job for one SQUINT variant, with ALL analysis
# steps enabled (UMAP, code-index spatial, SVG reconstruction, metrics).
#
# Sibling of:
#   - submit_dataset_sweep.sh : one job per variant in a DATASET_VARIANTS
#                                sweep, with --metrics-only by default
#                                (UMAP / SVG / code-index plots SKIPPED).
#   - submit_multi_seed.sh    : N jobs per variant across SEEDS + an
#                                aggregator; per-seed plots also SKIPPED.
#
# This script is for "I want every diagnostic plot for ONE variant, ONE
# seed, ONE run." Reuses `_run_one_variant.sh` as the runner with the
# `SQUINT_WITH_PLOTS=1` env-var toggle that file already supports.
#
# Usage:
#   bash examples/submit_single_run.sh <VARIANT> [OPTIONS]
#
# Required:
#   VARIANT  — A registered variant name from VARIANTS in run_squint.py.
#              Use `python examples/run_squint.py --list-variants` to list.
#
# Optional flags:
#   --queue / -q QUEUE   LSF queue (default: training-parallel).
#                        Also overridable via LSF_QUEUE env var.
#   --group / -g GROUP   LSF cost-code group (default: s10396).
#                        Also overridable via LSF_GROUP env var.
#   --help  / -h         Print this usage block and exit.
#
# Optional env-var overrides:
#   VENV_PATH         /nfs/team361/sb75/.venvs/squint
#   SQUINT_REPO       <auto: this script's repo root>
#   LOG_ROOT          /nfs/team361/sb75/squint-reproducibility/artifacts/logs
#   LSF_GROUP         s10396             (also: --group / -g)
#   LSF_QUEUE         training-parallel  (also: --queue / -q)
#   LSF_CORES         24
#   LSF_MEM_MB        128000             (heavier datasets — chl59,
#                                          mmb20 — may need 200000+)
#   LSF_GPU           mode=exclusive_process:num=1:block=yes
#   LSF_WALL          24:00              (full pipeline w/ UMAP: ~1-2h
#                                          on mmb-smb, ~2-4h on chl59;
#                                          24h is a safe ceiling)
#   DRY_RUN           0                  set to 1 to print bsub
#                                          without submitting
#
# What runs:
#   The job invokes `run_squint.py --all --variant $VARIANT` with the
#   FULL 6-stage pipeline:
#     1. train()
#     2. predict()
#     3. plot_code_indices_spatial.py
#     4. plot_svg_reconstruction.py
#     5. plot_latent_umap.py
#     6. compute_inference_metrics.py
#   No `--metrics-only`, no `--skip-*`. All plots land in the run dir
#   under <ARTIFACTS>/<dataset_tag>/<variant>/<timestamp>/.
#
# Examples:
#   # The chl59 winner port with full diagnostics (default resources):
#   bash examples/submit_single_run.sh \
#       dualvq+rvq-both+decoder-cov+adv+enc-deeper+dec-w32+chl59-8b_1p
#
#   # chl59 with bumped memory (the 6-section Lung graph is heavier):
#   LSF_MEM_MB=200000 LSF_WALL=48:00 \
#     bash examples/submit_single_run.sh \
#       dualvq+rvq-both+decoder-cov+adv+enc-deeper+dec-w32+chl59-8b_1p
#
#   # Different queue / group:
#   bash examples/submit_single_run.sh <VARIANT> \
#       --queue gpu-lotfollahi --group team361
#
#   # Dry-run to inspect the bsub command:
#   DRY_RUN=1 bash examples/submit_single_run.sh <VARIANT>
#
# Logs:
#   Stdout/stderr go to:
#     $LOG_ROOT/single/<variant>__<TS>/single.out
#     $LOG_ROOT/single/<variant>__<TS>/single.err
#   Per-run artifacts (checkpoints, predicted_adata.h5ad, plots,
#   metrics) land in the standard run dir:
#     <ARTIFACTS>/<dataset_tag>/<variant>/<TS>/
# -----------------------------------------------------------------------------

set -euo pipefail

# --- Defaults ---------------------------------------------------------------
DEFAULT_QUEUE="training-parallel"
DEFAULT_GROUP="s10396"

QUEUE_ARG="${LSF_QUEUE:-$DEFAULT_QUEUE}"
GROUP_ARG="${LSF_GROUP:-$DEFAULT_GROUP}"

# --- Parse CLI flags --------------------------------------------------------
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --queue|-q)
            shift
            if [[ $# -eq 0 ]]; then
                echo "ERROR: --queue / -q requires a value." >&2
                exit 2
            fi
            QUEUE_ARG="$1"
            shift
            ;;
        --queue=*)
            QUEUE_ARG="${1#--queue=}"
            shift
            ;;
        --group|-g)
            shift
            if [[ $# -eq 0 ]]; then
                echo "ERROR: --group / -g requires a value." >&2
                exit 2
            fi
            GROUP_ARG="$1"
            shift
            ;;
        --group=*)
            GROUP_ARG="${1#--group=}"
            shift
            ;;
        --help|-h)
            awk '/^[^#]/ {exit} {print}' "$0"
            exit 0
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done
set -- ${POSITIONAL_ARGS[@]+"${POSITIONAL_ARGS[@]}"}

VARIANT="${1:-}"
if [[ -z "$VARIANT" ]]; then
    sed -n '4,75p' "$0"
    exit 1
fi

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SQUINT_REPO="${SQUINT_REPO:-"$( cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd )"}"

VENV_PATH="${VENV_PATH:-/nfs/team361/sb75/.venvs/squint}"
LOG_ROOT="${LOG_ROOT:-/nfs/team361/sb75/squint-reproducibility/artifacts/logs}"

# --- LSF resources ----------------------------------------------------------
LSF_GROUP="$GROUP_ARG"
LSF_QUEUE="$QUEUE_ARG"
LSF_CORES="${LSF_CORES:-24}"
LSF_MEM_MB="${LSF_MEM_MB:-128000}"
LSF_GPU="${LSF_GPU:-mode=exclusive_process:num=1:block=yes}"
LSF_WALL="${LSF_WALL:-24:00}"

DRY_RUN="${DRY_RUN:-0}"

# --- Per-run log dir --------------------------------------------------------
# Use a timestamp so re-submitting the same variant doesn't clobber the
# previous run's logs. The actual training artifacts (checkpoints,
# predicted_adata.h5ad, plots, metrics) land at the standard location
# managed by run_squint.py train() itself.
RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$LOG_ROOT/single/${VARIANT}__${RUN_TS}"
mkdir -p "$LOG_DIR"

echo "================================================================"
echo "Variant       : $VARIANT"
echo "Repo          : $SQUINT_REPO"
echo "Venv          : $VENV_PATH"
echo "Log dir       : $LOG_DIR"
echo "All plots     : ENABLED (UMAP + code-index + SVG + metrics)"
echo "Resources     : -G $LSF_GROUP -q $LSF_QUEUE -n $LSF_CORES"
echo "                -M $LSF_MEM_MB (rusage[mem=$LSF_MEM_MB]) -W $LSF_WALL"
echo "                -gpu '$LSF_GPU'"
echo "Dry run       : $DRY_RUN"
echo "================================================================"

# --- Verify runner script exists --------------------------------------------
RUNNER="$SCRIPT_DIR/_run_one_variant.sh"
# Invoked as `bash $RUNNER`, so the file just needs to exist (executable
# bit not required and may be lost across cluster transfers).
if [[ ! -f "$RUNNER" ]]; then
    echo "ERROR: missing runner at $RUNNER" >&2
    exit 1
fi

# --- Submit -----------------------------------------------------------------
JOB_NAME="squint-single-${VARIANT}-${RUN_TS}"
LOG_OUT="$LOG_DIR/single.out"
LOG_ERR="$LOG_DIR/single.err"

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
    # Inject SQUINT_WITH_PLOTS=1 into the job env. `_run_one_variant.sh`
    # reads this and omits `--metrics-only`, so the full 6-stage
    # pipeline runs (train, predict, code-index plots, SVG plots,
    # latent UMAP, metrics).
    -env "all, SQUINT_WITH_PLOTS=1"
)

echo "Submitting:"
if [[ "$DRY_RUN" == "1" ]]; then
    printf '    %q ' "${BSUB_CMD[@]}"
    printf '%s %q %q %q %q\n' "bash" "$RUNNER" "$VARIANT" "$VENV_PATH" "$SQUINT_REPO"
else
    "${BSUB_CMD[@]}" bash "$RUNNER" "$VARIANT" "$VENV_PATH" "$SQUINT_REPO"
fi

echo
echo "================================================================"
echo "Submitted single-run job for $VARIANT."
echo "Watch with    : bjobs"
echo "Logs          : $LOG_OUT"
echo "              : $LOG_ERR"
echo "Per-run dir   : <ARTIFACTS>/<dataset_tag>/$VARIANT/<TS>/"
echo "                (set by run_squint.py train() itself)"
echo "================================================================"
