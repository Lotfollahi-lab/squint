#!/usr/bin/env bash
# submit_stage2_mc_sweep.sh
# -----------------------------------------------------------------------------
# DECODE-ONLY Monte-Carlo sweep over --decode-samples K for an already-trained
# stage-2 variant (default: decode-temp05-nonoise). It RE-DECODES that variant's
# SAVED codes (stage2_predicted_codes.npz) at several K — NO retraining, NO
# re-in-painting — writing each K into its own `<BASE>-mc<K>` variant dir so
# `rank_stage2_ablations.py` shows the RMSE-vs-K convergence curve.
#
# WHY this is decode-only: MC-averaging samples K code configs from the SAVED
# per-level posteriors (probs_cell_* / probs_niche_*) and averages the decoded
# count profiles ~= E[profile|context] (the RMSE-optimal predictor). The codes +
# posteriors are already in the npz, so varying K just re-runs the cheap
# codes->expression decode. Higher K -> lower RMSE, converging (~20-50); but it
# also pulls toward the MEAN (blurs), so watch the DISTRIBUTIONAL metrics
# (zero/nonzero AUROC, sharpness) in the full panel, not just RMSE.
#
# The BASE variant itself (hard/argmax decode, K=1-equivalent) stays as the
# reference point already in the ranking — compare <BASE> vs <BASE>-mc{K}.
#
# Requires: the BASE npz carries probs_cell_* (and probs_niche_* for the niche
# branch), i.e. it was produced with save_soft=True (run_stage2 default). If a
# variant's npz lacks them, stage2_decode_pearson exits with a clear message.
#
# Usage:
#   bash examples/submit_stage2_mc_sweep.sh                       # base=decode-temp05-nonoise, K=5 10 20 50
#   KS="10 20 50 100" bash examples/submit_stage2_mc_sweep.sh
#   BASE=decode-greedy-nonoise bash examples/submit_stage2_mc_sweep.sh
#   DRY_RUN=1 bash examples/submit_stage2_mc_sweep.sh             # print bsubs
#
# Env overrides: BASE, KS, NBR_NEIGHS, STAGE1_VARIANT_DIR, SEEDS, ABL_ROOT,
#   VENV_PATH, SQUINT_REPO, LSF_GROUP, LSF_QUEUE, LSF_GPU, LSF_WALL, LSF_MEM_MB.
# -----------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SQUINT_REPO="${SQUINT_REPO:-"$( cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd )"}"
VENV_PATH="${VENV_PATH:-/nfs/team361/sb75/.venvs/squint}"
ART="${ART:-/nfs/team361/sb75/squint-reproducibility/artifacts}"
DATASET="${DATASET:-mmb0-1b_smb1-1b_1p}"

# Frozen stage-1 (region-holdout multiseed) — same default as submit_stage2_ablation.sh.
STAGE1_VARIANT_DIR="${STAGE1_VARIANT_DIR:-$ART/$DATASET/dualvq+rvq-both+decoder-cov+no-batch-int+enc-deeper+dec-w32+knn16+sampler16+cell-w1+bs512+lr7e-4+within-sec+decoupled-enc+diversity-w10+filmscale+crossmnn-wt10-k1+region-holdout+mmb0-1b_smb1-1b_1p}"
SEEDS="${SEEDS:-20260629_023326_seed0 20260629_024131_seed1 20260629_025158_seed2 20260629_025456_seed3 20260629_025518_seed4}"
ABL_ROOT="${ABL_ROOT:-$ART/$DATASET/stage2-ablation}"
LOG_ROOT="${LOG_ROOT:-$ART/logs/$DATASET/stage2-mc-sweep}"

BASE="${BASE:-decode-temp05-nonoise}"     # trained variant whose npz we re-decode
KS="${KS:-5 10 20 30 50}"                  # --decode-samples (MC) sweep; 30 included to match GeST's neighbor count
SMOOTH_KS="${SMOOTH_KS:-30}"               # --smooth-neighs (GeST-style spatial smoothing); 30 = GeST neighbors_k. "" to skip.
# --niche-from-cell (fair-comparison): build the NICHE branch by graph-smoothing
# the CELL prediction (like GeST/kNN) instead of SQUINT's native niche decoder.
# Space-separated list of MC K's to apply it to; "0" (or "hard") = the base
# hard/K=1 decode. Empty = off. E.g. NICHE_FROM_CELL_KS="0 1000" -> two arms:
# <BASE>-nichecell (No-MC) + <BASE>-mc1000-nichecell (MC).
NICHE_FROM_CELL_KS="${NICHE_FROM_CELL_KS:-}"
# ONLY_NICHE_FROM_CELL=1 -> run ONLY the niche-from-cell arms (skip the MC +
# smooth sweeps). Needed because `${KS:-default}` treats an empty KS as unset
# and re-applies the default, so `KS=""` alone won't suppress those arms.
ONLY_NICHE_FROM_CELL="${ONLY_NICHE_FROM_CELL:-}"
if [[ -n "$ONLY_NICHE_FROM_CELL" ]]; then KS=""; SMOOTH_KS=""; fi
NBR_NEIGHS="${NBR_NEIGHS:-16}"             # niche-branch aggregation graph

LSF_GROUP="${LSF_GROUP:-s10396}"
LSF_QUEUE="${LSF_QUEUE:-training-parallel}"
LSF_GPU="${LSF_GPU:-mode=exclusive_process:num=1:block=yes}"
LSF_WALL="${LSF_WALL:-12:00}"
LSF_MEM_MB="${LSF_MEM_MB:-128000}"
LSF_CORES="${LSF_CORES:-8}"
DRY_RUN="${DRY_RUN:-0}"

echo "=========================================================="
echo "Stage-2 decode sweep (DECODE-ONLY, no retrain)"
echo "  base variant  : $BASE   (re-decoding its saved npz)"
echo "  MC K values    : $KS         (--decode-samples: posterior-sample averaging)"
echo "  smooth K values: ${SMOOTH_KS:-<none>}   (--smooth-neighs: GeST-style spatial smoothing)"
echo "  nbr-neighs    : $NBR_NEIGHS"
echo "  stage1        : $STAGE1_VARIANT_DIR"
echo "  seeds         : $SEEDS"
echo "  abl root      : $ABL_ROOT"
echo "  LSF           : -G $LSF_GROUP -q $LSF_QUEUE -gpu '$LSF_GPU' -W $LSF_WALL"
echo "  DRY_RUN=$DRY_RUN"
echo "=========================================================="

[[ "$DRY_RUN" == "1" ]] || mkdir -p "$LOG_ROOT"
n_sub=0

# Submit ONE decode-only re-decode job (loops seeds inside) for a variant.
#   $1 = variant dir name   $2 = job/log tag   $3 = extra decode flag(s)
# Submit-time vars expand now; the loop var + per-seed paths stay literal (\$..).
submit_redecode() {
    local var="$1" tag="$2" extra="$3"
    local log_out="$LOG_ROOT/${var}.out" log_err="$LOG_ROOT/${var}.err"
    local JOB
    read -r -d '' JOB <<EOF || true
set -euo pipefail
source "$VENV_PATH/bin/activate"
cd "$SQUINT_REPO"
for seed in $SEEDS; do
  pred="$STAGE1_VARIANT_DIR/\$seed/predicted_adata.h5ad"
  codes="$ABL_ROOT/$BASE/\$seed/stage2_predicted_codes.npz"
  out="$ABL_ROOT/$var/\$seed"
  seednum="\${seed##*_seed}"
  if [[ ! -f "\$codes" ]]; then echo "MISSING codes (skip): \$codes" >&2; continue; fi
  if [[ ! -f "\$pred"  ]]; then echo "MISSING pred  (skip): \$pred"  >&2; continue; fi
  echo "[$tag] decode \$seed (seednum=\$seednum) -> \$out"
  python examples/stage2_decode_pearson.py --predicted-adata "\$pred" \\
    --stage2-codes "\$codes" --out-metrics-dir "\$out/metrics" \\
    --nbr-neighs $NBR_NEIGHS $extra --seed "\$seednum"
done
echo "[$tag] DONE"
EOF
    local BSUB=( bsub -G "$LSF_GROUP" -q "$LSF_QUEUE" -n "$LSF_CORES" -M "$LSF_MEM_MB"
                 -R "select[mem>$LSF_MEM_MB] rusage[mem=$LSF_MEM_MB]" -R "span[ptile=$LSF_CORES]"
                 -gpu "$LSF_GPU" -W "$LSF_WALL" -J "s2mc-$tag" -o "$log_out" -e "$log_err" )
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '%q ' "${BSUB[@]}"; printf 'bash -lc %q\n\n' "$JOB"
    else
        if [[ ! -d "$ABL_ROOT/$BASE" ]]; then
            echo "MISSING base variant dir: $ABL_ROOT/$BASE — nothing to re-decode." >&2
            exit 1
        fi
        "${BSUB[@]}" bash -lc "$JOB"
    fi
    n_sub=$((n_sub + 1))
}

# MC (posterior-sample averaging) arms:  <BASE>-mc<K>
for K in $KS; do
    submit_redecode "${BASE}-mc${K}" "mc${K}" "--decode-samples $K"
done
# GeST-style spatial-smoothing arms:  <BASE>-smooth<K>  (hard decode + smooth)
for SK in $SMOOTH_KS; do
    submit_redecode "${BASE}-smooth${SK}" "sm${SK}" "--smooth-neighs $SK"
done
# Niche-from-cell (fair-comparison) arms: <BASE>[-mc<K>]-nichecell — niche branch
# = graph-smoothed cell prediction (same as GeST/kNN), NOT the native niche
# decoder. "0"/"hard" = base K=1 decode; a number = that MC K (--decode-samples).
for NK in $NICHE_FROM_CELL_KS; do
    if [[ "$NK" == "0" || "$NK" == "hard" ]]; then
        submit_redecode "${BASE}-nichecell" "ncell" "--niche-from-cell"
    else
        submit_redecode "${BASE}-mc${NK}-nichecell" "ncell${NK}" \
            "--decode-samples $NK --niche-from-cell"
    fi
done

echo "Submitted $n_sub decode-sweep job(s) for base=$BASE  (MC K=$KS ; smooth K=${SMOOTH_KS:-none} ; niche-from-cell K=${NICHE_FROM_CELL_KS:-none})."
echo "Rank when done:  python analysis/ablations/rank_stage2_ablations.py --include '${BASE}*'"
