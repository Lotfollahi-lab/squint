#!/usr/bin/env bash
# submit_multi_seed.sh
# -----------------------------------------------------------------------------
# Submit one LSF job per seed for a SQUINT variant, plus an aggregator
# job (with -w dependency) that concatenates the per-seed metric CSVs
# into the same long-format `per_seed_*.csv` files the baseline
# benchmark runners produce. Sibling of `submit_dataset_sweep.sh` (which
# parallelises across VARIANTS); this one parallelises across SEEDS.
#
# Output layout (created upfront by this script, populated by the jobs):
#
#   <ARTIFACTS_DIR>/<dataset_tag>/<variant>__multiseed/<sweep_TS>/
#       manifest.yaml                          # variant, seeds, queue, ...
#       seed_runs/
#           seed_<N>_run_dir.txt               # written by each seed job
#           seed_<N>_runtime_seconds.txt
#           seed_<N>_status.txt
#       metrics/                                # written by aggregator
#           per_seed_niche_identification.csv
#           per_seed_batch_integration.csv
#           per_seed_pearson_reconstruction.csv
#           per_seed_runtimes.csv
#           runtime_summary.csv
#       seed_run_index.csv
#
# Per-seed SQUINT run dirs themselves land at the standard location
# `<ARTIFACTS_DIR>/<dataset_tag>/<variant>/<TS_per_seed>/` (because
# `train()` writes there); the stamp files above are the link between
# the sweep dir and each seed's actual run dir.
#
# Usage:
#   bash examples/submit_multi_seed.sh <VARIANT> [SEEDS] [OPTIONS]
#
# Required:
#   VARIANT  — Registered variant name. Use
#              `python examples/run_squint.py --list-variants` to list.
#
# Optional positional:
#   SEEDS    — Comma-separated seed list. Default: '0,1,2,3,4'.
#
# Optional CLI flags:
#   --queue / -q QUEUE   LSF queue (default: training-parallel).
#                        Per-seed jobs AND the chained aggregator both
#                        land on this queue (unless AGG_QUEUE env
#                        override is set). Also overridable via the
#                        LSF_QUEUE env var.
#   --group / -g GROUP   LSF cost-code group (default: s10396).
#                        Also overridable via LSF_GROUP env var.
#   --batch-size / -b N  Override `loader_params.batch_size` for this
#                        seed sweep. Default: inherit from the variant
#                        (the base dual config sets 256). Pair with a
#                        rescaled --lr — see below.
#   --lr LR              Override `optimizer_params.lr` for this seed
#                        sweep. Default: inherit from the variant
#                        (the base dual config sets 5e-4 at batch=256).
#                        When you bump --batch-size, scale --lr too:
#                          - sqrt-scaling (recommended start):
#                              lr ≈ 5e-4 * sqrt(batch / 256)
#                              batch=1024 -> lr=1e-3
#                              batch=4096 -> lr=2e-3
#                          - linear-scaling (more aggressive):
#                              lr =   5e-4 * (batch / 256)
#                              batch=4096 -> lr=8e-3
#                          Linear scaling usually destabilises the
#                          adversarial loop at this batch size; sqrt
#                          is the safer starting point.
#   --help  / -h         Print this usage block and exit.
#
# Precedence for queue/group/batch-size/lr: CLI flag > env (if any) >
# variant's build.
#
# Optional environment overrides:
#   VENV_PATH         /nfs/team361/sb75/.venvs/squint
#   SQUINT_REPO       <auto: this script's repo root>
#   LOG_ROOT          /nfs/team361/sb75/squint-reproducibility/artifacts/logs
#   OUT_DIR           <auto: <ARTIFACTS>/<dataset_tag>/<variant>__multiseed/<TS>/>
#   LSF_GROUP         team361             (also: --group / -g)
#   LSF_QUEUE         gpu-lotfollahi      (also: --queue / -q)
#   LSF_CORES         24  (22 DataLoader workers + 1 main + 1 spare;
#                          bumped from 20 / 16 workers after wandb
#                          showed GPU util at 5-10% — see the
#                          `num_workers` comment in run_squint.py's
#                          base config for the rationale)
#   LSF_MEM_MB        128000
#   LSF_GPU           mode=exclusive_process:num=1:block=yes
#   LSF_WALL          24:00
#   AGG_QUEUE         <mirrors LSF_QUEUE>  # override to send the aggregator
#                                          # to a different queue than the
#                                          # per-seed jobs. The aggregator
#                                          # is CPU-only work but most
#                                          # queues require a GPU spec —
#                                          # AGG_GPU below provides one
#                                          # (set AGG_GPU="" to opt out
#                                          # on flexible queues).
#   AGG_CORES         2
#   AGG_MEM_MB        16000
#   AGG_WALL          1:00
#   AGG_GPU           mode=exclusive_process:num=1:block=yes
#   SKIP_UMAP                 1         # default ON (matches user recommendation)
#   SKIP_CODE_INDEX_PLOTS     1         # default ON
#   SKIP_SVG_PLOTS            1         # default ON
#   SKIP_METRICS              0         # never skip — that's the whole point
#   SUBMIT_AGGREGATOR         1         # chain aggregator after seeds
#   DRY_RUN                   0
#
# Examples:
#   # Default: 5 seeds, queue=gpu-lotfollahi, group=team361, plots skipped.
#   bash examples/submit_multi_seed.sh \\
#       dualvq+rvq-both+decoder-cov+adv+enc-deeper+mmb0-1b_smb1-1b_1p
#
#   # Custom seed list:
#   bash examples/submit_multi_seed.sh \\
#       dualvq+rvq-both+decoder-cov+adv+enc-deeper+mmb0-1b_smb1-1b_1p 0,1,2
#
#   # Switch queue + group at submission time:
#   bash examples/submit_multi_seed.sh <VARIANT> \\
#       --queue inference --group s10396
#
#   # Dry-run to inspect what would submit:
#   DRY_RUN=1 bash examples/submit_multi_seed.sh dualvq+...
#
#   # Re-aggregate existing per-seed runs (no new training):
#   #   1. Skip the per-seed bsubs by passing an empty SEEDS list.
#   #   2. Point OUT_DIR at the existing sweep dir.
#   #   3. Just run the aggregator yourself:
#   #        python examples/run_squint_multi_seed.py --out-dir <OUT_DIR>
# -----------------------------------------------------------------------------

set -euo pipefail

# --- Defaults ---------------------------------------------------------------
# Match the (queue, group) pairing used by v9 / v10 / v11 wrappers.
# Env vars still work for backward compatibility; CLI flags below win.
DEFAULT_QUEUE="training-parallel"
DEFAULT_GROUP="s10396"

QUEUE_ARG="${LSF_QUEUE:-$DEFAULT_QUEUE}"
GROUP_ARG="${LSF_GROUP:-$DEFAULT_GROUP}"
# Aggregator queue: if the user explicitly set AGG_QUEUE via env, honour
# that. Otherwise it mirrors the per-seed queue (resolved below after
# flag parsing) so the aggregator always lands on a queue the user can
# actually submit to.
AGG_QUEUE_OVERRIDE="${AGG_QUEUE:-}"

# Runtime config overrides (forwarded to _run_one_seed.py).
# Empty default = inherit from the variant's build (the
# `loader_params.batch_size` / `optimizer_params.lr` set in
# `_make_default_dual_config` / per-variant patches).
BATCH_SIZE_ARG=""
LR_ARG=""

# --- Parse CLI flags --------------------------------------------------------
# Accept --queue / -q and --group / -g anywhere in argv; everything else
# is kept in positional order so VARIANT (=$1) and SEEDS_CSV (=$2) still
# work after.
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
        --batch-size|-b)
            shift
            if [[ $# -eq 0 ]]; then
                echo "ERROR: --batch-size / -b requires a value." >&2
                exit 2
            fi
            BATCH_SIZE_ARG="$1"
            shift
            ;;
        --batch-size=*)
            BATCH_SIZE_ARG="${1#--batch-size=}"
            shift
            ;;
        --lr)
            shift
            if [[ $# -eq 0 ]]; then
                echo "ERROR: --lr requires a value." >&2
                exit 2
            fi
            LR_ARG="$1"
            shift
            ;;
        --lr=*)
            LR_ARG="${1#--lr=}"
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
# Restore positionals so VARIANT / SEEDS_CSV indexing below stays simple.
set -- ${POSITIONAL_ARGS[@]+"${POSITIONAL_ARGS[@]}"}

VARIANT="${1:-}"
if [[ -z "$VARIANT" ]]; then
    sed -n '4,80p' "$0"
    exit 1
fi
SEEDS_CSV="${2:-0,1,2,3,4}"

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SQUINT_REPO="${SQUINT_REPO:-"$( cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd )"}"

VENV_PATH="${VENV_PATH:-/nfs/team361/sb75/.venvs/squint}"
LOG_ROOT="${LOG_ROOT:-/nfs/team361/sb75/squint-reproducibility/artifacts/logs}"

# --- Per-seed (GPU) LSF resources -------------------------------------------
# Apply the resolved queue / group (flag > env > default).
LSF_GROUP="$GROUP_ARG"
LSF_QUEUE="$QUEUE_ARG"
LSF_CORES="${LSF_CORES:-24}"
LSF_MEM_MB="${LSF_MEM_MB:-128000}"
LSF_GPU="${LSF_GPU:-mode=exclusive_process:num=1:block=yes}"
LSF_WALL="${LSF_WALL:-24:00}"

# --- Aggregator LSF resources -----------------------------------------------
# CPU-only work (CSV concat). Mirror the per-seed queue by default so the
# aggregator lands on a queue the user can actually submit to. If the queue
# requires a GPU spec (e.g. gpu-lotfollahi), AGG_GPU below provides one;
# set AGG_GPU="" to opt out on flexible queues.
AGG_QUEUE="${AGG_QUEUE_OVERRIDE:-$LSF_QUEUE}"
AGG_CORES="${AGG_CORES:-2}"
AGG_MEM_MB="${AGG_MEM_MB:-16000}"
AGG_WALL="${AGG_WALL:-1:00}"
AGG_GPU="${AGG_GPU:-mode=exclusive_process:num=1:block=yes}"
SUBMIT_AGGREGATOR="${SUBMIT_AGGREGATOR:-1}"

echo "[multi-seed] LSF_QUEUE = $LSF_QUEUE"
echo "[multi-seed] LSF_GROUP = $LSF_GROUP"
echo "[multi-seed] AGG_QUEUE = $AGG_QUEUE"
echo "[multi-seed] batch_size override = ${BATCH_SIZE_ARG:-<inherit from variant>}"
echo "[multi-seed] lr         override = ${LR_ARG:-<inherit from variant>}"

# Step skipping (forwarded to _run_one_seed.py via SEED_EXTRA_ARGS).
SKIP_UMAP="${SKIP_UMAP:-1}"
SKIP_CODE_INDEX_PLOTS="${SKIP_CODE_INDEX_PLOTS:-1}"
SKIP_SVG_PLOTS="${SKIP_SVG_PLOTS:-1}"
SKIP_METRICS="${SKIP_METRICS:-0}"

DRY_RUN="${DRY_RUN:-0}"

# Build the per-seed extra-args list once.
SEED_EXTRA_ARGS=()
[[ "$SKIP_UMAP"             == "1" ]] && SEED_EXTRA_ARGS+=("--skip-umap")
[[ "$SKIP_CODE_INDEX_PLOTS" == "1" ]] && SEED_EXTRA_ARGS+=("--skip-code-index-plots")
[[ "$SKIP_SVG_PLOTS"        == "1" ]] && SEED_EXTRA_ARGS+=("--skip-svg-plots")
[[ "$SKIP_METRICS"          == "1" ]] && SEED_EXTRA_ARGS+=("--skip-metrics")
# Runtime config overrides — forwarded to _run_one_seed.py which
# layers them on top of the variant's build via `_patch_seed`.
[[ -n "$BATCH_SIZE_ARG" ]] && SEED_EXTRA_ARGS+=("--batch-size" "$BATCH_SIZE_ARG")
[[ -n "$LR_ARG"         ]] && SEED_EXTRA_ARGS+=("--lr"         "$LR_ARG")
SEED_EXTRA_ARGS_STR="${SEED_EXTRA_ARGS[*]:-}"

# Resolve dataset_tag for the variant (so we can place OUT_DIR + log dir
# under the right dataset). This must run inside the venv because
# run_squint.py imports torch/anndata at module load.
if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
    echo "ERROR: VENV_PATH=$VENV_PATH is not a Python venv (no bin/activate)." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$VENV_PATH/bin/activate"
DATASET_TAG="$(python - <<PY
import sys
sys.path.insert(0, "$SQUINT_REPO/examples")
from run_squint import VARIANTS, DATASET_NAME
v = "$VARIANT"
if v not in VARIANTS:
    print("ERROR_UNKNOWN_VARIANT", file=sys.stderr)
    raise SystemExit(2)
cfg = VARIANTS[v]["build"]()
tag = cfg["dataset"].get("dataset_tag",
                        cfg["dataset"].get("dataset_name", DATASET_NAME))
print(tag)
PY
)"
ARTIFACTS_DIR="$(python - <<PY
import sys
sys.path.insert(0, "$SQUINT_REPO/examples")
from run_squint import ARTIFACTS_DIR
print(ARTIFACTS_DIR)
PY
)"
deactivate

if [[ -z "$DATASET_TAG" || -z "$ARTIFACTS_DIR" ]]; then
    echo "ERROR: could not resolve dataset_tag / ARTIFACTS_DIR for variant=$VARIANT." >&2
    exit 1
fi

# Build OUT_DIR (sweep dir). Each invocation gets a fresh timestamp so
# repeat sweeps of the same variant don't collide.
SWEEP_TS="${SWEEP_TS:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-$ARTIFACTS_DIR/$DATASET_TAG/${VARIANT}__multiseed/$SWEEP_TS}"
mkdir -p "$OUT_DIR/seed_runs"

# Where LSF stdout/stderr land for this sweep.
LSF_LOG_DIR="$LOG_ROOT/multiseed/${VARIANT}__$SWEEP_TS"
mkdir -p "$LSF_LOG_DIR"

# Parse SEEDS into an array.
IFS=',' read -r -a SEEDS_ARR <<< "$SEEDS_CSV"
N_SEEDS=${#SEEDS_ARR[@]}

# Write the manifest so the aggregator + a future you can reconstruct
# the sweep parameters without having to look at LSF logs.
cat > "$OUT_DIR/manifest.yaml" <<EOF
variant: $VARIANT
dataset_tag: $DATASET_TAG
seeds: [$(IFS=,; echo "${SEEDS_ARR[*]}")]
sweep_timestamp: $SWEEP_TS
out_dir: $OUT_DIR
queue:
  per_seed: $LSF_QUEUE
  aggregator: $AGG_QUEUE
seed_extra_args: "$SEED_EXTRA_ARGS_STR"
submit_aggregator: $SUBMIT_AGGREGATOR
EOF

echo "================================================================"
echo "Variant       : $VARIANT"
echo "Dataset tag   : $DATASET_TAG"
echo "Seeds         : ${SEEDS_ARR[*]}  ($N_SEEDS total)"
echo "Sweep dir     : $OUT_DIR"
echo "LSF log dir   : $LSF_LOG_DIR"
echo "Venv          : $VENV_PATH"
echo "Repo          : $SQUINT_REPO"
echo "Per-seed      : -G $LSF_GROUP -q $LSF_QUEUE -n $LSF_CORES "
echo "                -M $LSF_MEM_MB (rusage[mem=$LSF_MEM_MB]) -W $LSF_WALL"
echo "                -gpu '$LSF_GPU'"
echo "Aggregator    : -q $AGG_QUEUE -n $AGG_CORES -M $AGG_MEM_MB -W $AGG_WALL"
echo "                -gpu '${AGG_GPU:-<none>}'"
echo "                submit chained job: $SUBMIT_AGGREGATOR"
echo "Step flags    : ${SEED_EXTRA_ARGS_STR:-<none>}"
echo "Dry run       : $DRY_RUN"
echo "================================================================"

SEED_RUNNER="$SCRIPT_DIR/_run_one_seed.sh"
if [[ ! -f "$SEED_RUNNER" ]]; then
    echo "ERROR: missing runner $SEED_RUNNER" >&2
    exit 1
fi

# Submit one bsub per seed. Job names are unique per (variant, seed,
# sweep_ts) so multiple multiseed sweeps for the same variant don't
# collide on the LSF dependency selector.
JOB_NAMES=()
i=0
for SEED in "${SEEDS_ARR[@]}"; do
    [[ -z "$SEED" ]] && continue
    i=$((i + 1))
    JOB_NAME="squint-ms-${VARIANT}-s${SEED}-${SWEEP_TS}"
    LOG_OUT="$LSF_LOG_DIR/seed_${SEED}.out"
    LOG_ERR="$LSF_LOG_DIR/seed_${SEED}.err"

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
        # Inject SEED_EXTRA_ARGS into the job env so the runner can
        # forward them to _run_one_seed.py without quoting magic.
        -env "all, SEED_EXTRA_ARGS=$SEED_EXTRA_ARGS_STR"
    )

    echo "[$i/$N_SEEDS] seed=$SEED  $JOB_NAME"
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '    %q ' "${BSUB_CMD[@]}"
        printf '%s %q %q %q %q %q %q\n' "bash" "$SEED_RUNNER" \
            "$VARIANT" "$SEED" "$OUT_DIR" "$VENV_PATH" "$SQUINT_REPO"
    else
        "${BSUB_CMD[@]}" bash "$SEED_RUNNER" \
            "$VARIANT" "$SEED" "$OUT_DIR" "$VENV_PATH" "$SQUINT_REPO"
    fi
    JOB_NAMES+=("$JOB_NAME")
done

# Submit the aggregator as a dependent job (only fires once every seed
# job has FINISHED — `ended()` matches both success and failure, so the
# aggregator runs even if some seeds crashed; it'll just include whatever
# completed successfully).
if [[ "$SUBMIT_AGGREGATOR" == "1" && "${#JOB_NAMES[@]}" -gt 0 ]]; then
    # Build the -w dependency: `ended(job1) && ended(job2) && ...`.
    DEP_EXPR=""
    for n in "${JOB_NAMES[@]}"; do
        if [[ -z "$DEP_EXPR" ]]; then
            DEP_EXPR="ended(\"$n\")"
        else
            DEP_EXPR="$DEP_EXPR && ended(\"$n\")"
        fi
    done

    AGG_JOB="squint-ms-aggregate-${VARIANT}-${SWEEP_TS}"
    AGG_OUT="$LSF_LOG_DIR/aggregate.out"
    AGG_ERR="$LSF_LOG_DIR/aggregate.err"

    BSUB_AGG=(
        bsub
        -G "$LSF_GROUP"
        -q "$AGG_QUEUE"
        -n "$AGG_CORES"
        -M "$AGG_MEM_MB"
        -R "select[mem>$AGG_MEM_MB] rusage[mem=$AGG_MEM_MB]"
        -W "$AGG_WALL"
        -J "$AGG_JOB"
        -o "$AGG_OUT"
        -e "$AGG_ERR"
        -w "$DEP_EXPR"
    )
    # The aggregator is CPU-only work, but the inference queue requires
    # a GPU spec. Add `-gpu` unless the user explicitly set AGG_GPU=""
    # to opt out (only safe on a queue that accepts CPU-only jobs).
    if [[ -n "$AGG_GPU" ]]; then
        BSUB_AGG+=(-gpu "$AGG_GPU")
    fi

    # Aggregator command: source the venv, then call run_squint_multi_seed.py.
    AGG_CMD=$(cat <<EOF
set -euo pipefail
unset PYTHONPATH PYTHONHOME
source "$VENV_PATH/bin/activate"
cd "$SQUINT_REPO"
python "$SQUINT_REPO/examples/run_squint_multi_seed.py" \
    --out-dir "$OUT_DIR" \
    --method-label "$VARIANT"
EOF
)

    echo
    echo "Aggregator job  : $AGG_JOB"
    echo "Aggregator deps : $DEP_EXPR"
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '    %q ' "${BSUB_AGG[@]}"
        printf 'bash -c %q\n' "$AGG_CMD"
    else
        "${BSUB_AGG[@]}" bash -c "$AGG_CMD"
    fi
fi

echo
echo "================================================================"
echo "Submitted $i seed job(s)."
if [[ "$SUBMIT_AGGREGATOR" == "1" ]]; then
    echo "Aggregator will run automatically once all seeds finish."
    echo "Aggregated CSVs land at: $OUT_DIR/metrics/"
else
    echo "Aggregator NOT chained (SUBMIT_AGGREGATOR=0)."
    echo "Run manually after seeds finish:"
    echo "  python $SQUINT_REPO/examples/run_squint_multi_seed.py \\"
    echo "      --out-dir $OUT_DIR --method-label '$VARIANT'"
fi
echo "Watch with    : bjobs"
echo "Per-seed logs : $LSF_LOG_DIR/seed_<N>.{out,err}"
if [[ "$SUBMIT_AGGREGATOR" == "1" ]]; then
    echo "Aggregator log: $LSF_LOG_DIR/aggregate.{out,err}"
fi
echo "================================================================"
