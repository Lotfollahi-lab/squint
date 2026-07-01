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
KS="${KS:-5 10 20 50}"                     # --decode-samples values to sweep
NBR_NEIGHS="${NBR_NEIGHS:-16}"             # niche-branch aggregation graph

LSF_GROUP="${LSF_GROUP:-s10396}"
LSF_QUEUE="${LSF_QUEUE:-training-parallel}"
LSF_GPU="${LSF_GPU:-mode=exclusive_process:num=1:block=yes}"
LSF_WALL="${LSF_WALL:-12:00}"
LSF_MEM_MB="${LSF_MEM_MB:-128000}"
LSF_CORES="${LSF_CORES:-8}"
DRY_RUN="${DRY_RUN:-0}"

echo "=========================================================="
echo "Stage-2 Monte-Carlo decode sweep (DECODE-ONLY, no retrain)"
echo "  base variant : $BASE   (re-decoding its saved npz)"
echo "  K values     : $KS"
echo "  nbr-neighs   : $NBR_NEIGHS"
echo "  stage1       : $STAGE1_VARIANT_DIR"
echo "  seeds        : $SEEDS"
echo "  abl root     : $ABL_ROOT"
echo "  LSF          : -G $LSF_GROUP -q $LSF_QUEUE -gpu '$LSF_GPU' -W $LSF_WALL"
echo "  DRY_RUN=$DRY_RUN"
echo "=========================================================="

[[ "$DRY_RUN" == "1" ]] || mkdir -p "$LOG_ROOT"
n_sub=0
for K in $KS; do
    var="${BASE}-mc${K}"
    job="s2mc-${K}"
    log_out="$LOG_ROOT/${var}.out"; log_err="$LOG_ROOT/${var}.err"

    # One job per K, looping seeds inside. Submit-time vars expand now; the loop
    # variable + per-seed paths stay literal (\$...) so they resolve on the node.
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
  echo "[mc$K] decode \$seed (seednum=\$seednum) -> \$out"
  python examples/stage2_decode_pearson.py --predicted-adata "\$pred" \\
    --stage2-codes "\$codes" --out-metrics-dir "\$out/metrics" \\
    --decode-samples $K --nbr-neighs $NBR_NEIGHS --seed "\$seednum"
done
echo "[mc$K] DONE"
EOF

    BSUB=( bsub -G "$LSF_GROUP" -q "$LSF_QUEUE" -n "$LSF_CORES" -M "$LSF_MEM_MB"
           -R "select[mem>$LSF_MEM_MB] rusage[mem=$LSF_MEM_MB]" -R "span[ptile=$LSF_CORES]"
           -gpu "$LSF_GPU" -W "$LSF_WALL" -J "$job" -o "$log_out" -e "$log_err" )
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
done

echo "Submitted $n_sub MC-sweep job(s) (K = $KS) for base=$BASE."
echo "Rank when done:  python analysis/ablations/rank_stage2_ablations.py --include '${BASE}*'"
