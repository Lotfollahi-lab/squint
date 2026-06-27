#!/usr/bin/env bash
# submit_all_ablation_multiseed.sh
# -----------------------------------------------------------------------------
# Launch a 5-seed multi-seed sweep for EVERY ablation variant behind the
# ablation figures (Fig 5 + Fig S3) on mmb0-1b_smb1-1b_1p, by calling
# `submit_multi_seed.sh <variant> <seeds>` once per variant. Each call submits
# N seed jobs + 1 aggregator, writing per_seed_*.csv under
#   <ARTIFACTS>/mmb0-1b_smb1-1b_1p/<variant>__multiseed/<TS>/metrics/
#
# Axis -> variant map (default/reference in *italics* in the comments):
#   Fig 5:
#     adjacency reconstruction : s51_v1 (with adj, ref) vs s51_v3 (no adj)
#     contrastive cell loss    : s51_v1 (with)          vs s51_v2 (without)
#     decoder covariate        : s51_v1 (with, ref)     vs s51_v4 (no dec-cov)
#   Fig S3:
#     GNN depth                : s51_v1 (1 layer, ref)  vs s51_v11 (2 layers)
#     number of neighbours     : s51_v8 (8) / s51_v9 (16, 8-sampled) / s51_v1 (16, ref) / s51_v10 (24)
#     cell codebook L1         : s51_v6 (30,10) / s51_v5 (30,30) / s51_v1 (30,90, ref) / s51_v7 (30,300)
#     niche codebook L1        : s52_v1 (30,10) / s52_v2 (30,30) / s51_v1 (30,90, ref) / s52_v3 (30,300)
#   NEW (Fig S3 extension) — L0 codebook size (L1 held at 30):
#     cell  codebook L0        : s54_v1 (10,30) / s51_v5 (30,30, centre) / s54_v2 (90,30) / s54_v3 (300,30)
#     niche codebook L0        : s54_v4 (10,30) / s52_v2 (30,30, centre) / s54_v5 (90,30) / s54_v6 (300,30)
#
# NB on the reference: s51_v1 (== the s49_v23 benchmark winner, WITH the
# within-batch contrastive loss) is the clean reference for every axis — every
# comparator variant ALSO has the contrastive loss, so each axis differs in
# exactly one factor. (s51_v2 DROPS contrastive and is only the second point of
# the contrastive axis; do NOT use it as the cross-axis reference.)
#
# s51_v1 is config-identical to the already-running s49_v23 multiseed sweep; it
# is listed here for completeness but can be skipped (set SKIP_S51_V1=1) if you
# point the ablation plot at the s49_v23 sweep for the reference slot.
#
# Usage:
#   bash examples/submit_all_ablation_multiseed.sh [SEEDS]
#   DRY_RUN=1 bash examples/submit_all_ablation_multiseed.sh     # preview only
#
# Env overrides (forwarded to submit_multi_seed.sh):
#   SEEDS         0,1,2,3,4
#   LSF_QUEUE / LSF_GROUP / LSF_MEM_MB / LSF_WALL ...   (see submit_multi_seed.sh)
#   SKIP_S51_V1   0   (set 1 to skip the s49_v23-equivalent reference)
#   DRY_RUN       0   (set 1 to print the submit_multi_seed.sh calls, not run)
# -----------------------------------------------------------------------------
set -euo pipefail

SEEDS="${1:-${SEEDS:-0,1,2,3,4}}"
SKIP_S51_V1="${SKIP_S51_V1:-0}"
DRY_RUN="${DRY_RUN:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# All 20 ablation variants (union across the 7 axes + the new L0 axes).
VARIANTS=(
  # --- reference (== s49_v23) ---
  "s51_v1_dualvq+rvq-both+decoder-cov+no-batch-int+enc-deeper+dec-w32+knn16+sampler16+cell-w1+bs512+lr7e-4+within-sec+decoupled-enc+diversity-w10+contrastWB-w10-k5+mmb0-1b_smb1-1b_1p"
  # --- Fig 5 ---
  "s51_v2_dualvq+rvq-both+decoder-cov+no-batch-int+enc-deeper+dec-w32+knn16+sampler16+cell-w1+bs512+lr7e-4+within-sec+decoupled-enc+diversity-w10+mmb0-1b_smb1-1b_1p"
  "s51_v3_dualvq+rvq-both+decoder-cov+no-batch-int+enc-deeper+dec-w32+knn16+sampler16+cell-w1+bs512+lr7e-4+within-sec+decoupled-enc+diversity-w10+contrastWB-w10-k5+no-adj+mmb0-1b_smb1-1b_1p"
  "s51_v4_dualvq+rvq-both+no-batch-int+enc-deeper+dec-w32+knn16+sampler16+cell-w1+bs512+lr7e-4+within-sec+decoupled-enc+diversity-w10+contrastWB-w10-k5+mmb0-1b_smb1-1b_1p"
  # --- Fig S3: GNN depth ---
  "s51_v11_dualvq+rvq-both+decoder-cov+no-batch-int+enc-deeper+dec-w32+knn16+sampler16-16+gnn-l2+cell-w1+bs512+lr7e-4+within-sec+decoupled-enc+diversity-w10+contrastWB-w10-k5+mmb0-1b_smb1-1b_1p"
  # --- Fig S3: number of neighbours ---
  "s51_v8_dualvq+rvq-both+decoder-cov+no-batch-int+enc-deeper+dec-w32+knn8+sampler8+cell-w1+bs512+lr7e-4+within-sec+decoupled-enc+diversity-w10+contrastWB-w10-k5+mmb0-1b_smb1-1b_1p"
  "s51_v9_dualvq+rvq-both+decoder-cov+no-batch-int+enc-deeper+dec-w32+knn16+sampler8+cell-w1+bs512+lr7e-4+within-sec+decoupled-enc+diversity-w10+contrastWB-w10-k5+mmb0-1b_smb1-1b_1p"
  "s51_v10_dualvq+rvq-both+decoder-cov+no-batch-int+enc-deeper+dec-w32+knn24+sampler24+cell-w1+bs512+lr7e-4+within-sec+decoupled-enc+diversity-w10+contrastWB-w10-k5+mmb0-1b_smb1-1b_1p"
  # --- Fig S3: cell codebook L1 ---
  "s51_v6_dualvq+rvq-cell-30-10+rvq-niche-30-90+decoder-cov+no-batch-int+enc-deeper+dec-w32+knn16+sampler16+cell-w1+bs512+lr7e-4+within-sec+decoupled-enc+diversity-w10+contrastWB-w10-k5+mmb0-1b_smb1-1b_1p"
  "s51_v5_dualvq+rvq-cell-30-30+rvq-niche-30-90+decoder-cov+no-batch-int+enc-deeper+dec-w32+knn16+sampler16+cell-w1+bs512+lr7e-4+within-sec+decoupled-enc+diversity-w10+contrastWB-w10-k5+mmb0-1b_smb1-1b_1p"
  "s51_v7_dualvq+rvq-cell-30-300+rvq-niche-30-90+decoder-cov+no-batch-int+enc-deeper+dec-w32+knn16+sampler16+cell-w1+bs512+lr7e-4+within-sec+decoupled-enc+diversity-w10+contrastWB-w10-k5+mmb0-1b_smb1-1b_1p"
  # --- Fig S3: niche codebook L1 ---
  "s52_v1_dualvq+rvq-cell-30-90+rvq-niche-30-10+decoder-cov+no-batch-int+enc-deeper+dec-w32+knn16+sampler16+cell-w1+bs512+lr7e-4+within-sec+decoupled-enc+diversity-w10+contrastWB-w10-k5+mmb0-1b_smb1-1b_1p"
  "s52_v2_dualvq+rvq-cell-30-90+rvq-niche-30-30+decoder-cov+no-batch-int+enc-deeper+dec-w32+knn16+sampler16+cell-w1+bs512+lr7e-4+within-sec+decoupled-enc+diversity-w10+contrastWB-w10-k5+mmb0-1b_smb1-1b_1p"
  "s52_v3_dualvq+rvq-cell-30-90+rvq-niche-30-300+decoder-cov+no-batch-int+enc-deeper+dec-w32+knn16+sampler16+cell-w1+bs512+lr7e-4+within-sec+decoupled-enc+diversity-w10+contrastWB-w10-k5+mmb0-1b_smb1-1b_1p"
  # --- NEW: cell codebook L0 (L1=30) ---
  "s54_v1_dualvq+rvq-cell-10-30+rvq-niche-30-90+decoder-cov+no-batch-int+enc-deeper+dec-w32+knn16+sampler16+cell-w1+bs512+lr7e-4+within-sec+decoupled-enc+diversity-w10+contrastWB-w10-k5+mmb0-1b_smb1-1b_1p"
  "s54_v2_dualvq+rvq-cell-90-30+rvq-niche-30-90+decoder-cov+no-batch-int+enc-deeper+dec-w32+knn16+sampler16+cell-w1+bs512+lr7e-4+within-sec+decoupled-enc+diversity-w10+contrastWB-w10-k5+mmb0-1b_smb1-1b_1p"
  "s54_v3_dualvq+rvq-cell-300-30+rvq-niche-30-90+decoder-cov+no-batch-int+enc-deeper+dec-w32+knn16+sampler16+cell-w1+bs512+lr7e-4+within-sec+decoupled-enc+diversity-w10+contrastWB-w10-k5+mmb0-1b_smb1-1b_1p"
  # --- NEW: niche codebook L0 (L1=30) ---
  "s54_v4_dualvq+rvq-cell-30-90+rvq-niche-10-30+decoder-cov+no-batch-int+enc-deeper+dec-w32+knn16+sampler16+cell-w1+bs512+lr7e-4+within-sec+decoupled-enc+diversity-w10+contrastWB-w10-k5+mmb0-1b_smb1-1b_1p"
  "s54_v5_dualvq+rvq-cell-30-90+rvq-niche-90-30+decoder-cov+no-batch-int+enc-deeper+dec-w32+knn16+sampler16+cell-w1+bs512+lr7e-4+within-sec+decoupled-enc+diversity-w10+contrastWB-w10-k5+mmb0-1b_smb1-1b_1p"
  "s54_v6_dualvq+rvq-cell-30-90+rvq-niche-300-30+decoder-cov+no-batch-int+enc-deeper+dec-w32+knn16+sampler16+cell-w1+bs512+lr7e-4+within-sec+decoupled-enc+diversity-w10+contrastWB-w10-k5+mmb0-1b_smb1-1b_1p"
)

echo "Ablation multi-seed launcher"
echo "  seeds        : $SEEDS"
echo "  variants     : ${#VARIANTS[@]}  (each -> $(echo "$SEEDS" | tr ',' ' ' | wc -w) seed jobs + 1 aggregator)"
echo "  skip s51_v1  : $SKIP_S51_V1"
echo "  DRY_RUN      : $DRY_RUN"
echo

n_sub=0
for V in "${VARIANTS[@]}"; do
    if [[ "$SKIP_S51_V1" == "1" && "$V" == s51_v1_* ]]; then
        echo "[skip] $V (SKIP_S51_V1=1; reuse the s49_v23 sweep for the reference)"
        continue
    fi
    echo ">>> $V"
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "    DRY: bash $SCRIPT_DIR/submit_multi_seed.sh \"$V\" \"$SEEDS\""
    else
        bash "$SCRIPT_DIR/submit_multi_seed.sh" "$V" "$SEEDS" || \
            echo "    !! submit FAILED for $V (continuing)"
        n_sub=$((n_sub + 1))
    fi
done

echo
echo "Done. Submitted multi-seed sweeps for $n_sub variant(s) (DRY_RUN=$DRY_RUN)."
