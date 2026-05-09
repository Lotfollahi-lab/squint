#!/usr/bin/env bash
# submit_inference_for_existing_runs.sh
# -----------------------------------------------------------------------------
# Walk an artifacts dataset folder, find every (variant, timestamp) that has
# a trained checkpoint, and submit one LSF job per pair to re-run inference
# (`run_inference.py`). Sibling of `submit_dataset_sweep.sh` (which trains
# from scratch + runs inference); this script ONLY re-runs steps 2-6
# against existing checkpoints.
#
# Use case: a metrics / predict-pipeline fix landed and you want to
# regenerate `predicted_adata.h5ad` + metrics for every variant you
# already trained, without retraining.
#
# Usage:
#   bash examples/submit_inference_for_existing_runs.sh [DATASET_DIR] [OPTIONS]
#
# DATASET_DIR (positional, optional)
#   Path to <ARTIFACTS>/<dataset_tag>/. Defaults to
#   /nfs/team361/sb75/squint-reproducibility/artifacts/mmb0-1b_smb1-1b_1p
#
# Selection rules:
#   - For each <variant>/<timestamp>/ subdir whose name matches a registered
#     variant in VARIANTS, the LATEST timestamp (lexicographic, == calendar
#     for YYYYMMDD_HHMMSS) is selected. Override with TIMESTAMP_STRATEGY=all
#     to submit every timestamp instead.
#   - The selected timestamp must contain at least one .ckpt under
#     `checkpoints/` (or `files/checkpoints/` for legacy wandb layout);
#     otherwise the variant is skipped with a warning.
#
# Environment overrides (same names + defaults as `submit_dataset_sweep.sh`):
#   VENV_PATH         /nfs/team361/sb75/.venvs/squint
#   SQUINT_REPO       <auto: this script's repo root>
#   LOG_ROOT          /nfs/team361/sb75/squint-reproducibility/artifacts/logs
#   LSF_GROUP         s10396
#   LSF_QUEUE         training-parallel
#   LSF_CORES         6
#   LSF_MEM_MB        128000
#   LSF_GPU           mode=exclusive_process:num=1:block=yes
#   LSF_WALL          12:00          # inference is much shorter than training
#   DRY_RUN           0              # 1 = print commands without running
#   LOCAL             0              # 1 = run sequentially in the current
#                                    #     shell (no bsub). Use on an
#                                    #     interactive GPU node when you
#                                    #     want live stdout and the ability
#                                    #     to Ctrl-C the loop. Logs are
#                                    #     also written to LOG_DIR via
#                                    #     `tee` so failures are inspectable.
#
# Inference-specific overrides:
#   TIMESTAMP_STRATEGY        latest | all       (default: latest)
#   SKIP_UMAP                 0 | 1              (default: 1; UMAPs are slow)
#   SKIP_CODE_INDEX_PLOTS     0 | 1              (default: 0)
#   SKIP_SVG_PLOTS            0 | 1              (default: 0)
#   SKIP_METRICS              0 | 1              (default: 0)
#   SKIP_PREDICT              0 | 1              (default: 0; set to 1 to
#                                                 reuse existing
#                                                 predicted_adata.h5ad)
#   INFERENCE_EXTRA_ARGS      free-form extra CLI args appended to
#                             run_inference.py (e.g. "--silver-dir /alt/path").
#
# Examples:
#   # Default: every variant under mmb-smb, latest timestamp each, skip UMAPs.
#   bash examples/submit_inference_for_existing_runs.sh
#
#   # Dry-run first to see what would submit:
#   DRY_RUN=1 bash examples/submit_inference_for_existing_runs.sh
#
#   # Different dataset (chl59):
#   bash examples/submit_inference_for_existing_runs.sh \
#       /nfs/team361/sb75/squint-reproducibility/artifacts/chl59-8b_1p
#
#   # Re-run metrics only (predict + plots already up-to-date):
#   SKIP_PREDICT=1 SKIP_CODE_INDEX_PLOTS=1 SKIP_SVG_PLOTS=1 SKIP_UMAP=1 \
#       bash examples/submit_inference_for_existing_runs.sh
#
#   # Run sequentially on an interactive GPU node (no LSF):
#   LOCAL=1 bash examples/submit_inference_for_existing_runs.sh
#
# Logs:
#   LSF mode  : stdout/stderr -> $LOG_ROOT/inference/<variant>__<timestamp>.{out,err}
#   LOCAL=1   : same files via `tee`, AND mirrored live to your terminal.
# -----------------------------------------------------------------------------

set -euo pipefail

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

LSF_GROUP="${LSF_GROUP:-s10396}"
LSF_QUEUE="${LSF_QUEUE:-training-parallel}"
LSF_CORES="${LSF_CORES:-6}"
LSF_MEM_MB="${LSF_MEM_MB:-128000}"
LSF_GPU="${LSF_GPU:-mode=exclusive_process:num=1:block=yes}"
LSF_WALL="${LSF_WALL:-12:00}"
DRY_RUN="${DRY_RUN:-0}"
LOCAL="${LOCAL:-0}"

TIMESTAMP_STRATEGY="${TIMESTAMP_STRATEGY:-latest}"
SKIP_UMAP="${SKIP_UMAP:-1}"
SKIP_CODE_INDEX_PLOTS="${SKIP_CODE_INDEX_PLOTS:-0}"
SKIP_SVG_PLOTS="${SKIP_SVG_PLOTS:-0}"
SKIP_METRICS="${SKIP_METRICS:-0}"
SKIP_PREDICT="${SKIP_PREDICT:-0}"
INFERENCE_EXTRA_ARGS_USER="${INFERENCE_EXTRA_ARGS:-}"

# Build the --skip-* flag list once for all jobs.
EXTRA_ARGS=()
[[ "$SKIP_PREDICT"           == "1" ]] && EXTRA_ARGS+=("--skip-predict")
[[ "$SKIP_CODE_INDEX_PLOTS"  == "1" ]] && EXTRA_ARGS+=("--skip-code-index-plots")
[[ "$SKIP_SVG_PLOTS"         == "1" ]] && EXTRA_ARGS+=("--skip-svg-plots")
[[ "$SKIP_UMAP"              == "1" ]] && EXTRA_ARGS+=("--skip-umap")
[[ "$SKIP_METRICS"           == "1" ]] && EXTRA_ARGS+=("--skip-metrics")
if [[ -n "$INFERENCE_EXTRA_ARGS_USER" ]]; then
    # Split on whitespace so the user can pass multi-arg overrides.
    # shellcheck disable=SC2206
    EXTRA_ARGS+=($INFERENCE_EXTRA_ARGS_USER)
fi
EXTRA_ARGS_STR="${EXTRA_ARGS[*]:-}"

# --- Validate venv (we need it to enumerate VARIANTS for the filter) -------
if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
    echo "ERROR: VENV_PATH=$VENV_PATH is not a Python venv." >&2
    exit 1
fi

# Pull the registered variant set from run_squint.py — used to filter
# subdirs (someone may have created stray dirs that don't correspond to a
# variant; skipping them with a warning is friendlier than crashing).
# shellcheck disable=SC1091
source "$VENV_PATH/bin/activate"
# `--list-variants` prints lines like:
#     "  <variant>"          (2-space indent, the name we want)
#     "    description: ..." (4-space indent, ignore)
#     "    patches:     ..." (4-space indent, ignore)
# The `^  \S` regex matches only the 2-space-indent variant lines.
REGISTERED_VARIANTS="$(python "$SQUINT_REPO/examples/run_squint.py" --list-variants 2>/dev/null \
    | grep -E '^  \S' | awk '{print $1}' | sort -u)"
deactivate

if [[ -z "$REGISTERED_VARIANTS" ]]; then
    echo "ERROR: Could not enumerate registered variants from run_squint.py." >&2
    exit 1
fi

# --- Walk the dataset folder for (variant, timestamp) pairs -----------------
LOG_DIR="$LOG_ROOT/inference"
mkdir -p "$LOG_DIR"

declare -a JOBS=()      # entries: "<variant>|<timestamp>"
declare -a SKIPPED=()   # entries: "<reason>: <variant>"

for variant_dir in "$DATASET_DIR"/*/; do
    [[ ! -d "$variant_dir" ]] && continue
    variant="$(basename "$variant_dir")"

    # Filter by registered variants. Use grep -Fx for exact-match safety
    # (variant names contain `+` / `[` chars that confuse `=~`).
    if ! grep -Fxq "$variant" <<< "$REGISTERED_VARIANTS"; then
        SKIPPED+=("not-registered: $variant")
        continue
    fi

    # Pick timestamp(s) under this variant. `sort` (lexicographic) on
    # YYYYMMDD_HHMMSS == calendar order, so the last entry == latest.
    # We avoid `mapfile` (bash 4+) and `find -printf` (GNU find) so the
    # script is portable to macOS / BSD environments too.
    timestamps=()
    while IFS= read -r ts_path; do
        [[ -z "$ts_path" ]] && continue
        timestamps+=("$(basename "$ts_path")")
    done < <(find "$variant_dir" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort)

    if [[ "${#timestamps[@]}" -eq 0 ]]; then
        SKIPPED+=("no-timestamps: $variant")
        continue
    fi

    case "$TIMESTAMP_STRATEGY" in
        latest)
            # ${arr[-1]} requires bash 4.3+. Use the explicit
            # last-index form for older-bash portability.
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
        # Look for ANY .ckpt anywhere under the run dir (covers
        # checkpoints/*.ckpt and the legacy files/checkpoints/*.ckpt
        # layout in one find call).
        ckpt_count="$(find "$run_dir" -type f -name '*.ckpt' 2>/dev/null | wc -l | tr -d ' ')"
        if [[ "$ckpt_count" -eq 0 ]]; then
            SKIPPED+=("no-checkpoint: $variant/$ts")
            continue
        fi
        JOBS+=("$variant|$ts")
    done
done

N_JOBS="${#JOBS[@]}"

echo "================================================================"
echo "Dataset dir       : $DATASET_DIR"
echo "Strategy          : $TIMESTAMP_STRATEGY"
echo "Repo              : $SQUINT_REPO"
echo "Venv              : $VENV_PATH"
echo "Log dir           : $LOG_DIR"
echo "Inference flags   : ${EXTRA_ARGS_STR:-<none>}"
echo "Mode              : $([[ "$LOCAL" == "1" ]] && echo "LOCAL (sequential, no LSF)" || echo "LSF")"
if [[ "$LOCAL" != "1" ]]; then
    echo "Resources         : -G $LSF_GROUP -q $LSF_QUEUE -n $LSF_CORES "
    echo "                    -M $LSF_MEM_MB (rusage[mem=$LSF_MEM_MB]) -W $LSF_WALL"
    echo "                    -gpu '$LSF_GPU'"
fi
echo "Dry run           : $DRY_RUN"
echo "Submitting        : $N_JOBS job(s)"
if [[ "${#SKIPPED[@]}" -gt 0 ]]; then
    echo "Skipped entries   : ${#SKIPPED[@]}"
    for s in "${SKIPPED[@]}"; do
        echo "    - $s"
    done
fi
echo "================================================================"

if [[ "$N_JOBS" -eq 0 ]]; then
    echo "Nothing to submit." >&2
    exit 0
fi

RUNNER="$SCRIPT_DIR/_run_one_variant_inference.sh"
if [[ ! -f "$RUNNER" ]]; then
    echo "ERROR: missing runner script at $RUNNER" >&2
    exit 1
fi

# --- Submission / execution loop --------------------------------------------
# Two modes:
#   LSF (default)  : one bsub per (variant, timestamp). Returns immediately
#                    after queuing every job.
#   LOCAL=1        : run sequentially in the current shell. `tee` to the
#                    same log files LSF would write, AND to the user's
#                    terminal. Ctrl-C aborts the current variant via the
#                    EXIT trap, then the outer `set -e` halts the loop.
#
# Failure handling in LOCAL mode: by default `set -e` halts the whole
# loop on the first variant that errors, so you don't silently miss
# failures while inference looks like it's running. Override with
# CONTINUE_ON_ERROR=1 to log the error and move on to the next variant.
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"

# Track failures for the final summary in LOCAL mode.
declare -a FAILED_JOBS=()

i=0
for entry in "${JOBS[@]}"; do
    i=$((i + 1))
    V="${entry%%|*}"
    TS="${entry##*|}"
    JOB_NAME="squint-inf-${V}-${TS}"
    LOG_OUT="$LOG_DIR/${V}__${TS}.out"
    LOG_ERR="$LOG_DIR/${V}__${TS}.err"

    if [[ "$LOCAL" == "1" ]]; then
        echo "[$i/$N_JOBS] $V  ($TS)  -> $LOG_OUT"
        if [[ "$DRY_RUN" == "1" ]]; then
            printf '    INFERENCE_EXTRA_ARGS=%q ' "$EXTRA_ARGS_STR"
            printf '%s %q %q %q %q %q\n' "bash" "$RUNNER" "$V" "$TS" "$VENV_PATH" "$SQUINT_REPO"
            continue
        fi
        # Run sequentially. `tee` mirrors stdout/stderr to per-variant log
        # files so failures are inspectable later, while the user still
        # sees live progress in the terminal. The runner inherits this
        # shell's environment (active venv, CUDA_VISIBLE_DEVICES, …).
        # `set +e` around the call so a single variant failure doesn't
        # abort the whole loop when CONTINUE_ON_ERROR=1.
        set +e
        INFERENCE_EXTRA_ARGS="$EXTRA_ARGS_STR" \
            bash "$RUNNER" "$V" "$TS" "$VENV_PATH" "$SQUINT_REPO" \
            > >(tee "$LOG_OUT") 2> >(tee "$LOG_ERR" >&2)
        rc=$?
        set -e
        if [[ "$rc" -ne 0 ]]; then
            FAILED_JOBS+=("$V/$TS (exit $rc; see $LOG_ERR)")
            if [[ "$CONTINUE_ON_ERROR" == "1" ]]; then
                echo "    ! variant failed (exit $rc); continuing (CONTINUE_ON_ERROR=1)" >&2
            else
                echo "    ! variant failed (exit $rc); aborting loop. Set CONTINUE_ON_ERROR=1 to skip past failures." >&2
                break
            fi
        fi
        continue
    fi

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
        # Inject INFERENCE_EXTRA_ARGS into the job's environment so the
        # runner can forward them to run_inference.py without us having
        # to compose argv strings (which would require quoting magic).
        -env "all, INFERENCE_EXTRA_ARGS=$EXTRA_ARGS_STR"
    )

    echo "[$i/$N_JOBS] $V  ($TS)"
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '    %q ' "${BSUB_CMD[@]}"
        printf '%s %q %q %q %q %q\n' "bash" "$RUNNER" "$V" "$TS" "$VENV_PATH" "$SQUINT_REPO"
    else
        "${BSUB_CMD[@]}" bash "$RUNNER" "$V" "$TS" "$VENV_PATH" "$SQUINT_REPO"
    fi
done

echo "================================================================"
if [[ "$LOCAL" == "1" ]]; then
    echo "Ran $i variant(s) sequentially."
    if [[ "${#FAILED_JOBS[@]}" -gt 0 ]]; then
        echo "Failures: ${#FAILED_JOBS[@]}"
        for f in "${FAILED_JOBS[@]}"; do
            echo "    - $f"
        done
    else
        echo "All variants completed successfully."
    fi
else
    echo "Submitted $i job(s)."
    echo "Watch with: bjobs"
fi
echo "Logs      : $LOG_DIR/<variant>__<timestamp>.{out,err}"
echo "================================================================"
