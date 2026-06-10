#!/usr/bin/env bash
# submit_mmb_smb_ablations_v4.sh
# -----------------------------------------------------------------------------
# Convenience wrapper: submits all 8 variants in
# `DATASET_VARIANTS["mmb0-1b_smb1-1b_1p-ablations-v4"]` as separate LSF
# jobs (one bsub per variant, one GPU exclusive each).
#
# v4 spine: `+wide+rvq-both+decoder-cov+adv` (the v3 winner on niche
# NMI). The 8 variants hybridise this spine with the strongest
# non-wide signals from v1 (`+enc-deeper`, `+adj-w3000`, `+adv-w50`)
# and introduce ONE new axis — adversarial warmup — to recover the
# cell-NMI / cell-pearson regression that `+wide` introduced.
#
# Three independently-promising axes:
#   1. Deeper encoder MLP (`+enc-deeper`, [400, 400, 256])
#         -> empirical winner on every batch-integration metric in v1.
#   2. Higher adjacency BCE weight (`+adj-w3000`)
#         -> co-optimises niche NMI + cell iLISI in v1.
#   3. Adversarial warmup (`+adv-warmup10`, NEW)
#         -> alpha=0 for the first 10 epochs so codes settle on biology
#            before batch-invariance pressure kicks in.
#
# This is just `submit_dataset_sweep.sh` with the right key — provided
# as a 1-liner so you don't have to remember which group key to use.
# All env-var overrides from `submit_dataset_sweep.sh` work here too,
# e.g.:
#
#   # Dry-run first (recommended) — prints bsub commands without submitting:
#   DRY_RUN=1 bash examples/submit_mmb_smb_ablations_v4.sh
#
#   # Force a longer wallclock or more memory:
#   LSF_WALL=48:00 LSF_MEM_MB=200000 \
#       bash examples/submit_mmb_smb_ablations_v4.sh
#
#   # Use a different rapids env for the UMAP step:
#   RAPIDS_ENV=/path/to/conda/env \
#       bash examples/submit_mmb_smb_ablations_v4.sh
#
# Logs land at:
#   /nfs/team361/sb75/squint-reproducibility/artifacts/logs/
#       mmb0-1b_smb1-1b_1p-ablations-v4/<variant>.{out,err}
# Per-job artifacts (checkpoints, predicted_adata.h5ad, plots, metrics) at:
#   /nfs/team361/sb75/squint-reproducibility/artifacts/
#       mmb0-1b_smb1-1b_1p/<variant>/<timestamp>/
# -----------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
exec bash "$SCRIPT_DIR/submit_dataset_sweep.sh" "mmb0-1b_smb1-1b_1p-ablations-v4" "$@"
