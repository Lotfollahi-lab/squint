#!/usr/bin/env bash
# rerun_missing_metrics.sh
# -----------------------------------------------------------------------------
# Walk an artifacts dataset folder, find every variant whose latest
# timestamp is missing the canonical metrics CSVs, and (re)run inference
# for it — picking the cheapest valid set of `--skip-*` flags based on
# what's already on disk:
#
#   - If `predicted_adata.h5ad` already exists at the latest timestamp,
#     run with `--skip-predict --skip-umap --skip-svg-plots
#     --skip-code-index-plots` (metrics-only ~30-90s/variant).
#
#   - If `predicted_adata.h5ad` is missing, run full inference (predict +
#     metrics, no UMAP/SVG/code-index plots — those are slow and
#     usually not needed for re-runs).
#
# Variants whose latest TS already has BOTH `niche_identification_metrics.csv`
# AND `batch_integration_metrics.csv` are skipped.
#
# Usage:
#   bash examples/rerun_missing_metrics.sh [DATASET_DIR]
#
# DATASET_DIR (positional, optional):
#   Path to <ARTIFACTS>/<dataset_tag>/. Defaults to mmb0-1b_smb1-1b_1p.
#
# Environment overrides:
#   VENV_PATH         /nfs/team361/sb75/.venvs/squint
#   SQUINT_REPO       <auto: this script's repo root>
#   LOG_ROOT          /nfs/team361/sb75/squint-reproducibility/artifacts/logs
#   TIMESTAMP_STRATEGY  latest | all       (default: latest)
#   DRY_RUN           0|1                  (default: 0; 1 = print what
#                                           would run)
#   CONTINUE_ON_ERROR 0|1                  (default: 1; do NOT abort
#                                           the loop when one variant
#                                           fails. Override with =0 if
#                                           you want fail-fast.)
#   SKIP_UMAP         0|1                  (default: 1; UMAPs are slow)
#   SKIP_CODE_INDEX_PLOTS / SKIP_SVG_PLOTS  0|1 (default: 1 each)
#   FORCE_REPREDICT   0|1                  (default: 0; if 1, re-run
#                                           predict even when
#                                           predicted_adata.h5ad
#                                           already exists. Useful
#                                           after model code changes.)
#   INFERENCE_EXTRA_ARGS   free-form extra args appended to
#                          run_inference.py.
#
# Examples:
#   # Default: scan mmb0-1b_smb1-1b_1p, fix every missing-metrics variant
#   bash examples/rerun_missing_metrics.sh
#
#   # Dry-run first to see what would run:
#   DRY_RUN=1 bash examples/rerun_missing_metrics.sh
#
#   # Different dataset:
#   bash examples/rerun_missing_metrics.sh \
#       /nfs/team361/sb75/squint-reproducibility/artifacts/chl59-8b_1p
#
#   # Force re-prediction even where predicted_adata.h5ad already exists
#   # (e.g. after a model-code change):
#   FORCE_REPREDICT=1 bash examples/rerun_missing_metrics.sh
#
# Logs:
#   Per-variant logs at $LOG_ROOT/inference/<variant>__<timestamp>.{out,err}
#   Live console output is also tee'd to those files.
# -----------------------------------------------------------------------------

set -eo pipefail

DATASET_DIR_DEFAULT="/nfs/team361/sb75/squint-reproducibility/artifacts/mmb0-1b_smb1-1b_1p"
DATASET_DIR="${1:-$DATASET_DIR_DEFAULT}"

if [[ ! -d "$DATASET_DIR" ]]; then
    echo "ERROR: DATASET_DIR=$DATASET_DIR does not exist or is not a directory." >&2
    exit 1
fi

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SQUINT_REPO="${SQUINT_REPO:-"$( cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd )"}"
VENV_PATH="${VENV_PATH:-/nfs/team361/sb75/.venvs/squint}"
LOG_ROOT="${LOG_ROOT:-/nfs/team361/sb75/squint-reproducibility/artifacts/logs}"
LOG_DIR="$LOG_ROOT/inference"
mkdir -p "$LOG_DIR"

TIMESTAMP_STRATEGY="${TIMESTAMP_STRATEGY:-latest}"
DRY_RUN="${DRY_RUN:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
SKIP_UMAP="${SKIP_UMAP:-1}"
SKIP_CODE_INDEX_PLOTS="${SKIP_CODE_INDEX_PLOTS:-1}"
SKIP_SVG_PLOTS="${SKIP_SVG_PLOTS:-1}"
FORCE_REPREDICT="${FORCE_REPREDICT:-0}"
INFERENCE_EXTRA_ARGS_USER="${INFERENCE_EXTRA_ARGS:-}"

RUNNER="$SCRIPT_DIR/_run_one_variant_inference.sh"
if [[ ! -f "$RUNNER" ]]; then
    echo "ERROR: missing runner script at $RUNNER" >&2
    exit 1
fi

# Force TMPDIR to local /tmp/$USER so multiprocessing tempdirs don't
# land on NFS (avoids the OSError: [Errno 16] silly-rename problem).
# `_run_one_variant_inference.sh` does this too, but we set it here as
# well so any python imports the parent shell does (e.g. for variant
# enumeration below) also use local tmpdir.
export TMPDIR="/tmp/${USER:-$(id -un)}"
mkdir -p "$TMPDIR"

# --- Build the list of registered variants (filter foreign subdirs) ---
if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
    echo "ERROR: VENV_PATH=$VENV_PATH is not a Python venv." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$VENV_PATH/bin/activate"
REGISTERED_VARIANTS="$(python "$SQUINT_REPO/examples/run_squint.py" --list-variants 2>/dev/null \
    | grep -E '^  \S' | awk '{print $1}' | sort -u)"
deactivate

if [[ -z "$REGISTERED_VARIANTS" ]]; then
    echo "ERROR: Could not enumerate registered variants." >&2
    exit 1
fi

# --- Walk the dataset folder + classify each (variant, ts) -------------
# Categories:
#   COMPLETE          — both metrics CSVs present, skip
#   METRICS_ONLY      — predicted_adata.h5ad present, metrics missing
#                       -> re-run with --skip-predict
#   FULL              — predicted_adata.h5ad missing
#                       -> full inference
#   NO_CHECKPOINT     — no .ckpt anywhere in the run dir
#                       -> can't run, log and continue
#   NOT_REGISTERED    — variant not in --list-variants, skip
#   NO_TIMESTAMP      — no timestamp subdir, skip

declare -a JOBS_METRICS=()    # entries: "variant|timestamp"
declare -a JOBS_FULL=()       # entries: "variant|timestamp"
declare -a SKIPPED_COMPLETE=()
declare -a SKIPPED_NOT_REG=()
declare -a SKIPPED_NO_TS=()
declare -a SKIPPED_NO_CKPT=()

for variant_dir in "$DATASET_DIR"/*/; do
    [[ ! -d "$variant_dir" ]] && continue
    variant="$(basename "$variant_dir")"

    # 1. Filter by registered variants.
    if ! grep -Fxq "$variant" <<< "$REGISTERED_VARIANTS"; then
        SKIPPED_NOT_REG+=("$variant")
        continue
    fi

    # 2. Pick timestamp(s).
    timestamps=()
    while IFS= read -r ts_path; do
        [[ -z "$ts_path" ]] && continue
        timestamps+=("$(basename "$ts_path")")
    done < <(find "$variant_dir" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort)

    if [[ "${#timestamps[@]}" -eq 0 ]]; then
        SKIPPED_NO_TS+=("$variant")
        continue
    fi

    case "$TIMESTAMP_STRATEGY" in
        latest)
            selected=("${timestamps[$((${#timestamps[@]} - 1))]}")
            ;;
        all)
            selected=("${timestamps[@]}")
            ;;
        *)
            echo "ERROR: TIMESTAMP_STRATEGY must be 'latest' or 'all', got: $TIMESTAMP_STRATEGY" >&2
            exit 1
            ;;
    esac

    for ts in "${selected[@]}"; do
        run_dir="$variant_dir$ts"

        # 3. Need at least one .ckpt.
        ckpt_count="$(find "$run_dir" -type f -name '*.ckpt' 2>/dev/null | wc -l | tr -d ' ')"
        if [[ "$ckpt_count" -eq 0 ]]; then
            SKIPPED_NO_CKPT+=("$variant/$ts")
            continue
        fi

        # 4. Classify by what's on disk at the run dir.
        has_pred=0
        [[ -f "$run_dir/predicted_adata.h5ad" ]] && has_pred=1
        has_niche=0
        [[ -f "$run_dir/metrics/niche_identification_metrics.csv" ]] && has_niche=1
        has_batch=0
        [[ -f "$run_dir/metrics/batch_integration_metrics.csv" ]] && has_batch=1

        if [[ "$has_niche" -eq 1 && "$has_batch" -eq 1 ]]; then
            SKIPPED_COMPLETE+=("$variant/$ts")
            continue
        fi

        if [[ "$has_pred" -eq 1 && "$FORCE_REPREDICT" -ne 1 ]]; then
            JOBS_METRICS+=("$variant|$ts")
        else
            JOBS_FULL+=("$variant|$ts")
        fi
    done
done

N_METRICS="${#JOBS_METRICS[@]}"
N_FULL="${#JOBS_FULL[@]}"
N_TOTAL=$((N_METRICS + N_FULL))

echo "================================================================"
echo "Dataset dir            : $DATASET_DIR"
echo "Strategy               : $TIMESTAMP_STRATEGY"
echo "Repo                   : $SQUINT_REPO"
echo "Venv                   : $VENV_PATH"
echo "Log dir                : $LOG_DIR"
echo "TMPDIR                 : $TMPDIR"
echo "Force re-predict       : $FORCE_REPREDICT"
echo "Skip UMAP/SVG/code-idx : $SKIP_UMAP/$SKIP_SVG_PLOTS/$SKIP_CODE_INDEX_PLOTS"
echo "Continue on error      : $CONTINUE_ON_ERROR"
echo "Dry run                : $DRY_RUN"
echo "----------------------------------------------------------------"
echo "JOBS — metrics-only    : $N_METRICS  (predicted_adata.h5ad reused)"
for j in "${JOBS_METRICS[@]}"; do echo "    $j"; done
echo "JOBS — full inference  : $N_FULL"
for j in "${JOBS_FULL[@]}"; do echo "    $j"; done
echo "----------------------------------------------------------------"
echo "Skipped — complete            : ${#SKIPPED_COMPLETE[@]}"
for v in "${SKIPPED_COMPLETE[@]}"; do echo "    $v"; done
echo "Skipped — not-registered      : ${#SKIPPED_NOT_REG[@]}"
for v in "${SKIPPED_NOT_REG[@]}"; do echo "    $v"; done
echo "Skipped — no checkpoint       : ${#SKIPPED_NO_CKPT[@]}"
for v in "${SKIPPED_NO_CKPT[@]}"; do echo "    $v"; done
echo "Skipped — no timestamps       : ${#SKIPPED_NO_TS[@]}"
for v in "${SKIPPED_NO_TS[@]}"; do echo "    $v"; done
echo "================================================================"

if [[ "$N_TOTAL" -eq 0 ]]; then
    echo "Nothing to do — every registered variant has metrics already."
    exit 0
fi

# --- Build the EXTRA_ARGS once per category ----------------------------
# Skip the slow plotting / code-index steps regardless — they're not on
# the metrics critical path.
build_extra_args() {
    local skip_predict="$1"
    local args=()
    [[ "$skip_predict"          == "1" ]] && args+=("--skip-predict")
    [[ "$SKIP_CODE_INDEX_PLOTS" == "1" ]] && args+=("--skip-code-index-plots")
    [[ "$SKIP_SVG_PLOTS"        == "1" ]] && args+=("--skip-svg-plots")
    [[ "$SKIP_UMAP"             == "1" ]] && args+=("--skip-umap")
    if [[ -n "$INFERENCE_EXTRA_ARGS_USER" ]]; then
        # shellcheck disable=SC2206
        args+=($INFERENCE_EXTRA_ARGS_USER)
    fi
    echo "${args[*]:-}"
}

EXTRA_METRICS_ONLY="$(build_extra_args 1)"
EXTRA_FULL_INFERENCE="$(build_extra_args 0)"

# --- Run each selected job sequentially --------------------------------
# Tee per-variant logs so failures are inspectable.

declare -a FAILED_JOBS=()
i=0

run_job() {
    local entry="$1"; local extra="$2"; local mode="$3"
    local V="${entry%%|*}"
    local TS="${entry##*|}"
    local LOG_OUT="$LOG_DIR/${V}__${TS}.out"
    local LOG_ERR="$LOG_DIR/${V}__${TS}.err"

    i=$((i + 1))
    echo
    echo "================================================================"
    echo "[$i/$N_TOTAL] $mode  : $V  ($TS)"
    echo "  log out : $LOG_OUT"
    echo "  log err : $LOG_ERR"
    echo "  extra   : ${extra:-<none>}"
    echo "================================================================"

    if [[ "$DRY_RUN" == "1" ]]; then
        printf '    INFERENCE_EXTRA_ARGS=%q ' "$extra"
        printf '%s %q %q %q %q %q\n' "bash" "$RUNNER" "$V" "$TS" "$VENV_PATH" "$SQUINT_REPO"
        return 0
    fi

    set +e
    INFERENCE_EXTRA_ARGS="$extra" \
        bash "$RUNNER" "$V" "$TS" "$VENV_PATH" "$SQUINT_REPO" \
        > >(tee "$LOG_OUT") 2> >(tee "$LOG_ERR" >&2)
    rc=$?
    set -e

    if [[ "$rc" -ne 0 ]]; then
        FAILED_JOBS+=("[$mode] $V/$TS (exit $rc; see $LOG_ERR)")
        if [[ "$CONTINUE_ON_ERROR" == "1" ]]; then
            echo "  ! variant failed (exit $rc); continuing." >&2
        else
            echo "  ! variant failed (exit $rc); aborting loop." >&2
            return 1
        fi
    fi
}

# Run metrics-only jobs first (cheap, fast); then full-inference jobs.
echo
echo "================================================================"
echo "PHASE 1 — METRICS-ONLY ($N_METRICS variants, ~30-90s each)"
echo "================================================================"
for entry in "${JOBS_METRICS[@]}"; do
    run_job "$entry" "$EXTRA_METRICS_ONLY" "METRICS-ONLY" || break
done

echo
echo "================================================================"
echo "PHASE 2 — FULL INFERENCE ($N_FULL variants, ~5-30min each)"
echo "================================================================"
for entry in "${JOBS_FULL[@]}"; do
    run_job "$entry" "$EXTRA_FULL_INFERENCE" "FULL" || break
done

# --- Summary ------------------------------------------------------------
echo
echo "================================================================"
if [[ "${#FAILED_JOBS[@]}" -eq 0 ]]; then
    echo "ALL DONE — $i/$N_TOTAL variants completed successfully."
else
    echo "DONE WITH FAILURES — $((i - ${#FAILED_JOBS[@]}))/$N_TOTAL succeeded."
    echo "Failures (${#FAILED_JOBS[@]}):"
    for f in "${FAILED_JOBS[@]}"; do
        echo "    - $f"
    done
fi
echo "Logs: $LOG_DIR/<variant>__<timestamp>.{out,err}"
echo "================================================================"
