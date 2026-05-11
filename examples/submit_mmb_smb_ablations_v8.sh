#!/usr/bin/env bash
# submit_mmb_smb_ablations_v8.sh
# -----------------------------------------------------------------------------
# Convenience wrapper: submits all 8 variants in
# `DATASET_VARIANTS["mmb0-1b_smb1-1b_1p-ablations-v8"]` as separate LSF
# jobs (one bsub per variant, one GPU exclusive each).
#
# v8 design: two groups of 4 on the v5-anchor spine
# (+wide +rvq-both-3level(30,60,120) +decoder-cov +adv, no warmup,
# no +enc-deeper — the v5 default minus the deeper encoder so the
# encoder-size axis can vary cleanly).
#
# Group A — encoder/decoder MLP size sweep:
#   1. enc-shallow:   encoder 2-layer [400, 256] -> 1-layer [256]
#                     (depth axis, width preserved)
#   2. mlp-h256:      encoder [400, 256] -> [256, 256];
#                     cell + niche decoders [400, 400] -> [256, 256]
#                     (width axis, depth preserved)
#   3. small-latent:  encoder [400, 256] -> [400, 128]; GNN hidden 256
#                     -> 128; codebook embedding 256 -> 128
#                     (latent-dim axis, encoder depth + decoder unchanged)
#   4. small:         scvi-style compact (hidden=128 everywhere). Drops
#                     +wide because +small re-sets GNN num_layers=1.
#                     The most aggressive capacity ablation.
#
# Group B — MMD + adversarial batch-integration sweep (2x2 grid):
#   5. mmd-w50, NO adv:     MMD-only, weak
#   6. mmd-w200, NO adv:    MMD-only, strong
#   7. adv + mmd-w50:       Combo, weak MMD supplement
#   8. adv + mmd-w200:      Combo, strong MMD supplement
#
# Group B replaces or supplements the default adversary with kernel MMD
# on the cell-token input (z_mlp[:B]) — MMD applies a deterministic
# distribution-matching pressure every step (no min-max instability
# from the GRL), so it doesn't suffer the warmup pathology that bit
# v6/v6b. All four B variants share the v5 spine.
#
# All env-var overrides from `submit_dataset_sweep.sh` work here too,
# e.g.:
#
#   # Dry-run first (recommended) -- prints bsub commands without submitting:
#   DRY_RUN=1 bash examples/submit_mmb_smb_ablations_v8.sh
#
#   # Force a longer wallclock or more memory:
#   LSF_WALL=48:00 LSF_MEM_MB=200000 \
#       bash examples/submit_mmb_smb_ablations_v8.sh
#
#   # Use a different rapids env for the UMAP step:
#   RAPIDS_ENV=/path/to/conda/env \
#       bash examples/submit_mmb_smb_ablations_v8.sh
#
# Logs land at:
#   /nfs/team361/sb75/squint-reproducibility/artifacts/logs/
#       mmb0-1b_smb1-1b_1p-ablations-v8/<variant>.{out,err}
# Per-job artifacts (checkpoints, predicted_adata.h5ad, plots, metrics) at:
#   /nfs/team361/sb75/squint-reproducibility/artifacts/
#       mmb0-1b_smb1-1b_1p/<variant>/<timestamp>/
# -----------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
exec bash "$SCRIPT_DIR/submit_dataset_sweep.sh" "mmb0-1b_smb1-1b_1p-ablations-v8" "$@"
