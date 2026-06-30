#!/usr/bin/env bash
# submit_stage2_ablation.sh
# -----------------------------------------------------------------------------
# Stage-2 imputation ablation across ALL stage-1 seeds. For each (stage-1 seed,
# stage-2 variant) it submits ONE GPU job that:
#   1. trains stage-2 on that seed's frozen predicted_adata (--holdout-split test),
#   2. decodes the predicted codes through the frozen stage-1 decoder -> Pearson
#      (stage2_decode_pearson.py),
# writing into  <ABL_ROOT>/<variant>/<seed>/  so you get 5 results per variant.
#
# VARIANTS (one factor each vs the baseline):
#   base       current stage-2 (defaults).
#   capacity   bigger model + more context (d512/L12, patch 2048, more steps, k=48).
#   l0w4       up-weight the L0 (coarse) code CE 4x (--l0-weight 4.0).
# (soft-decode and decoded-expression-loss arms need extra code paths and a
#  single-seed smoke first; add them to variant_*_flags once they land.)
#
# Usage:
#   bash examples/submit_stage2_ablation.sh                 # all variants x 5 seeds
#   VARIANTS="base capacity" bash examples/submit_stage2_ablation.sh
#   SMOKE=1 bash examples/submit_stage2_ablation.sh         # seed0 only, tiny, fast
#   DRY_RUN=1 bash examples/submit_stage2_ablation.sh       # print bsubs, don't submit
#
# Env overrides: STAGE1_VARIANT_DIR, SEEDS, ABL_ROOT, VENV_PATH, LSF_GROUP,
#   LSF_QUEUE, LSF_GPU, LSF_WALL, LSF_MEM_MB.
# -----------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SQUINT_REPO="${SQUINT_REPO:-"$( cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd )"}"
VENV_PATH="${VENV_PATH:-/nfs/team361/sb75/.venvs/squint}"
ART="${ART:-/nfs/team361/sb75/squint-reproducibility/artifacts}"
DATASET="${DATASET:-mmb0-1b_smb1-1b_1p}"

STAGE1_VARIANT_DIR="${STAGE1_VARIANT_DIR:-$ART/$DATASET/dualvq+rvq-both+decoder-cov+no-batch-int+enc-deeper+dec-w32+knn16+sampler16+cell-w1+bs512+lr7e-4+within-sec+decoupled-enc+diversity-w10+filmscale+crossmnn-wt10-k1+region-holdout+mmb0-1b_smb1-1b_1p}"
SEEDS="${SEEDS:-20260629_023326_seed0 20260629_024131_seed1 20260629_025158_seed2 20260629_025456_seed3 20260629_025518_seed4}"
ABL_ROOT="${ABL_ROOT:-$ART/$DATASET/stage2-ablation}"
LOG_ROOT="${LOG_ROOT:-$ART/logs/$DATASET/stage2-ablation}"

LSF_GROUP="${LSF_GROUP:-s10396}"
LSF_QUEUE="${LSF_QUEUE:-training-parallel}"
LSF_GPU="${LSF_GPU:-mode=exclusive_process:num=1:block=yes}"
LSF_WALL="${LSF_WALL:-24:00}"
LSF_MEM_MB="${LSF_MEM_MB:-128000}"
LSF_CORES="${LSF_CORES:-8}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"

# variant -> stage-2 TRAIN flags. Return non-zero for an unknown variant.
# Flag-only arms (no new code) are runnable today; new-code arms (soft / density
# / distbias / maskcontig / softlabels / exprloss / gestarch) are TODO.
variant_train_flags() {
    case "$1" in
        base)      echo "" ;;
        capacity)  echo "--d-model 512 --n-layers 12 --patch-size 2048 --max-steps 40000 --neighbors-k 48 --eval-chunk-size 256" ;;
        l0w4)      echo "--l0-weight 4.0" ;;
        # --- flag-only architectural/config arms (zero new code) ---
        maskmatch) echo "--mask-frac-min 0.20 --mask-frac-max 0.45" ;;   # tighten train masks toward the eval ~25% contiguous hole
        decode20)  echo "--decode-steps 20" ;;                            # more MaskGIT unmask iterations (decode-side, but set at train cfg)
        nohier)    echo "--no-hierarchical" ;;                            # ablate the cell->niche head conditioning
        bigctx)    echo "--neighbors-k 48 --eval-chunk-size 256 --patch-size 2048" ;;  # more observed context, same model size
        soft)      echo "" ;;                         # same train as base; soft decode (probs saved by run_stage2)
        # --- ARCHITECTURE ablation: swappable stage-2 backbone (--arch) -----
        # Same embedding / heads / masked-CE / MaskGIT decode; only the body
        # differs (vqniche.stage2.backbones). 'base' is the transformer arch.
        arch-gnn)        echo "--arch gnn" ;;          # spatial-kNN message passing (GraphSAGE)
        arch-labelprop)  echo "--arch labelprop" ;;    # APPNP propagation + MLP correction (smoothness ceiling)
        arch-graphmae)   echo "--arch graphmae" ;;     # GNN enc -> re-mask -> GNN dec (GraphMAE)
        arch-gps)        echo "--arch gps" ;;          # MPNN + global attention hybrid (GraphGPS)
        arch-diffusion)  echo "--arch diffusion --decode-steps 20" ;;  # timestep-conditioned denoiser (discrete diffusion)
        # --- DECODE-side sweep around the winning decode20 (decode-only knobs:
        #     the base model trains identically; only MaskGIT inference differs) ---
        decode16)        echo "--decode-steps 16" ;;
        decode24)        echo "--decode-steps 24" ;;
        decode28)        echo "--decode-steps 28" ;;
        decode32)        echo "--decode-steps 32" ;;
        decode40)        echo "--decode-steps 40" ;;
        decode-linear)   echo "--decode-steps 20 --decode-schedule linear" ;;       # linear unmask schedule
        decode-temp05)   echo "--decode-steps 20 --decode-temperature 0.5" ;;       # sharper sampling
        decode-greedy)   echo "--decode-steps 20 --decode-temperature 0.0" ;;       # argmax commits
        decode-ctx3)     echo "--decode-steps 20 --decode-context-radius-mult 3.0" ;; # wider observed context ring
        decode-nonoise)  echo "--decode-steps 20 --decode-noise-anneal 0.0" ;;      # no confidence noise
        # --- COMBINED deterministic decode (temperature [code choice] + noise_anneal
        #     [commit order] are orthogonal, so they stack). The 3 winners pointed
        #     the same way: less stochasticity -> better imputation (point estimate). ---
        decode-greedy-nonoise)     echo "--decode-steps 20 --decode-temperature 0.0 --decode-noise-anneal 0.0" ;;
        decode-temp05-nonoise)     echo "--decode-steps 20 --decode-temperature 0.5 --decode-noise-anneal 0.0" ;;
        decode-greedy-nonoise-s40) echo "--decode-steps 40 --decode-temperature 0.0 --decode-noise-anneal 0.0" ;;
        # --- new-code arms (need the knobs implemented first) ---
        # density)   echo "--density-aware" ;;        # conditioning agent C2
        # distbias)  echo "--attn-dist-exponent 1.5";# attention agent A1
        # softlabels) echo "--soft-label-temp 0.5" ;; # loss agent L2
        # exprloss)  echo "--expr-loss-weight 1.0" ;; # loss agent L1 (frozen decoder in training)
        *) return 1 ;;
    esac
}
# variant -> stage2_decode_pearson DECODE flags (hard for all current arms).
variant_decode_flags() {
    case "$1" in
        soft) echo "--decode-mode soft" ;;
        *) echo "" ;;
    esac
}

VARIANTS="${VARIANTS:-base capacity l0w4}"
[[ "$SMOKE" == "1" ]] && SEEDS="$(echo "$SEEDS" | awk '{print $1}')"   # seed0 only

echo "=========================================================="
echo "Stage-2 imputation ablation"
echo "  stage1 variant : $STAGE1_VARIANT_DIR"
echo "  seeds          : $SEEDS"
echo "  variants       : $VARIANTS"
echo "  abl root       : $ABL_ROOT"
echo "  LSF            : -G $LSF_GROUP -q $LSF_QUEUE -gpu '$LSF_GPU' -W $LSF_WALL"
echo "  SMOKE=$SMOKE  DRY_RUN=$DRY_RUN"
echo "=========================================================="

[[ "$DRY_RUN" == "1" ]] || mkdir -p "$LOG_ROOT"
n_sub=0
for seed in $SEEDS; do
    run_dir="$STAGE1_VARIANT_DIR/$seed"
    pred="$run_dir/predicted_adata.h5ad"
    for var in $VARIANTS; do
        if ! tflags="$(variant_train_flags "$var")"; then
            echo "WARN: unknown variant '$var' — skip" >&2; continue
        fi
        dflags="$(variant_decode_flags "$var")"
        out="$ABL_ROOT/$var/$seed"
        smoke_flag=""; [[ "$SMOKE" == "1" ]] && smoke_flag="--smoke"
        job="s2abl-$var-$seed"
        log_out="$LOG_ROOT/${var}__${seed}.out"; log_err="$LOG_ROOT/${var}__${seed}.err"

        read -r -d '' JOB <<EOF || true
set -euo pipefail
source "$VENV_PATH/bin/activate"
cd "$SQUINT_REPO"
echo "[abl] TRAIN  $var / $seed -> $out"
python examples/run_stage2.py --predicted-adata "$pred" --holdout-split test \\
  --out-dir "$out" $tflags $smoke_flag
echo "[abl] DECODE $var / $seed"
python examples/stage2_decode_pearson.py --predicted-adata "$pred" \\
  --stage2-codes "$out/stage2_predicted_codes.npz" \\
  --out-metrics-dir "$out/metrics" $dflags
echo "[abl] DONE $var / $seed"
EOF

        BSUB=( bsub -G "$LSF_GROUP" -q "$LSF_QUEUE" -n "$LSF_CORES" -M "$LSF_MEM_MB"
               -R "select[mem>$LSF_MEM_MB] rusage[mem=$LSF_MEM_MB]" -R "span[ptile=$LSF_CORES]"
               -gpu "$LSF_GPU" -W "$LSF_WALL" -J "$job" -o "$log_out" -e "$log_err" )
        if [[ "$DRY_RUN" == "1" ]]; then
            printf '%q ' "${BSUB[@]}"; printf 'bash -lc %q\n\n' "$JOB"
        else
            if [[ ! -f "$pred" ]]; then echo "MISSING predicted_adata: $pred — skip" >&2; continue; fi
            "${BSUB[@]}" bash -lc "$JOB"
        fi
        n_sub=$((n_sub + 1))
    done
done
echo "[abl] ${n_sub} job(s) $( [[ "$DRY_RUN" == "1" ]] && echo rendered || echo submitted )."
echo "Outputs: $ABL_ROOT/<variant>/<seed>/metrics/per_seed_pearson_reconstruction.csv"
