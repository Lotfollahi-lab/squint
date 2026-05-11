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
# Optional environment overrides:
#   VENV_PATH         /nfs/team361/sb75/.venvs/squint
#   SQUINT_REPO       <auto: this script's repo root>
#   LOG_ROOT          /nfs/team361/sb75/squint-reproducibility/artifacts/logs
#   OUT_DIR           <auto: <ARTIFACTS>/<dataset_tag>/<variant>__multiseed/<TS>/>
#   LSF_GROUP         s10396
#   LSF_QUEUE         inference         # USER REQUEST: inference queue by default
#   LSF_CORES         6
#   LSF_MEM_MB        128000
#   LSF_GPU           mode=exclusive_process:num=1:block=yes
#   LSF_WALL          24:00
#   AGG_QUEUE         inference         # only one queue available on this cluster
#   AGG_CORES         2
#   AGG_MEM_MB        16000
#   AGG_WALL          1:00
#   AGG_GPU           mode=exclusive_process:num=1:block=yes
#                                       # the aggregator is CPU-only work, but
#                                       # the inference queue requires a GPU
#                                       # spec — request one anyway so the
#                                       # job gets accepted. Set AGG_GPU=""
#                                       # to skip the -gpu flag (only safe
#                                       # on a queue that permits CPU jobs).
#   SKIP_UMAP                 1         # default ON (matches user recommendation)
#   SKIP_CODE_INDEX_PLOTS     1         # default ON
#   SKIP_SVG_PLOTS            1         # default ON
#   SKIP_METRICS              0         # never skip — that's the whole point
#   SUBMIT_AGGREGATOR         1         # chain aggregator after seeds
#   DRY_RUN                   0
#
# Examples:
#   # Default: 5 seeds, queue=inference, plots skipped, aggregator chained.
#   bash examples/submit_multi_seed.sh \\
#       dualvq+rvq-both+decoder-cov+adv+enc-deeper+mmb0-1b_smb1-1b_1p
#
#   # Custom seed list:
#   bash examples/submit_multi_seed.sh \\
#       dualvq+rvq-both+decoder-cov+adv+enc-deeper+mmb0-1b_smb1-1b_1p 0,1,2
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

# Per-seed (GPU) LSF resources
LSF_GROUP="${LSF_GROUP:-s10396}"
LSF_QUEUE="${LSF_QUEUE:-inference}"
LSF_CORES="${LSF_CORES:-6}"
LSF_MEM_MB="${LSF_MEM_MB:-128000}"
LSF_GPU="${LSF_GPU:-mode=exclusive_process:num=1:block=yes}"
LSF_WALL="${LSF_WALL:-24:00}"

# Aggregator LSF resources. There's only ONE queue on this cluster
# ("inference"), and it requires a GPU spec — so the aggregator runs
# on the same queue as the per-seed jobs and asks for a GPU even
# though it doesn't use one (the alternative is a queue rejection).
# Resources are kept minimal — 2 cores, 16 GB, 1 h wall — because the
# work is just CSV concatenation.
AGG_QUEUE="${AGG_QUEUE:-inference}"
AGG_CORES="${AGG_CORES:-2}"
AGG_MEM_MB="${AGG_MEM_MB:-16000}"
AGG_WALL="${AGG_WALL:-1:00}"
AGG_GPU="${AGG_GPU:-mode=exclusive_process:num=1:block=yes}"
SUBMIT_AGGREGATOR="${SUBMIT_AGGREGATOR:-1}"

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
