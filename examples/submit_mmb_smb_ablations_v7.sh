#!/usr/bin/env bash
# submit_mmb_smb_ablations_v7.sh
# -----------------------------------------------------------------------------
# Convenience wrapper: submits all 8 variants in
# `DATASET_VARIANTS["mmb0-1b_smb1-1b_1p-ablations-v7"]` as separate LSF
# jobs (one bsub per variant, one GPU exclusive each).
#
# v7 motivation: the v6/v6b adv-warmup10 family was the best cell-NMI
# winner on paper but post-warmup the loss spiked dramatically — the
# late-arriving adversary tried to undo features the encoder had
# committed during the alpha=0 phase, causing the optimization to go
# unstable. v7 drops warmup entirely and asks a different question:
# can we recover cell-NMI by varying the CELL codebook structure
# while NICHE stays fixed at 3-level RVQ (30, 60, 120)?
#
# Spine (all 8 share):
#   +wide                                 # 2-layer GNN + sampler [8,8] + 2-hop nbr
#   +rvq(niche, levels=[30, 60, 120])     # 3-level RVQ, FIXED
#   +decoder-cov                          # NicheCompass-style covariate decoder
#   +adv(alpha=1.0, wt=150.0)             # NO warmup
#   +enc-deeper                           # mlp=[400, 400, 256]
#
# Cell codebook varies (level 1 = 30 always, matches niche):
#   1. vq-cell-30           single-level VQ k=30 (tightest)
#   2. vq-cell-60           single-level VQ k=60
#   3. vq-cell-200          single-level VQ k=200 (widest single-level)
#   4. rvq-cell-30-60       2-level RVQ (30, 60)
#   5. rvq-cell-30-90       2-level RVQ (30, 90) [legacy v3 cell shape]
#   6. rvq-cell-30-200      2-level RVQ (30, 200) [_default_rvq_params]
#   7. rvq-cell-30-60-120   3-level matches niche [v5 anchor cell shape]
#   8. rvq-cell-30-90-270   3-level wider than niche
#
# All env-var overrides from `submit_dataset_sweep.sh` work here too,
# e.g.:
#
#   # Dry-run first (recommended) -- prints bsub commands without submitting:
#   DRY_RUN=1 bash examples/submit_mmb_smb_ablations_v7.sh
#
#   # Force a longer wallclock or more memory:
#   LSF_WALL=48:00 LSF_MEM_MB=200000 \
#       bash examples/submit_mmb_smb_ablations_v7.sh
#
#   # Use a different rapids env for the UMAP step:
#   RAPIDS_ENV=/path/to/conda/env \
#       bash examples/submit_mmb_smb_ablations_v7.sh
#
# Logs land at:
#   /nfs/team361/sb75/squint-reproducibility/artifacts/logs/
#       mmb0-1b_smb1-1b_1p-ablations-v7/<variant>.{out,err}
# Per-job artifacts (checkpoints, predicted_adata.h5ad, plots, metrics) at:
#   /nfs/team361/sb75/squint-reproducibility/artifacts/
#       mmb0-1b_smb1-1b_1p/<variant>/<timestamp>/
# -----------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
exec bash "$SCRIPT_DIR/submit_dataset_sweep.sh" "mmb0-1b_smb1-1b_1p-ablations-v7" "$@"
