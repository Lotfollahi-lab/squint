#!/usr/bin/env bash
# submit_mmb_smb_ablations_v9.sh
# -----------------------------------------------------------------------------
# Convenience wrapper: submits all 8 variants in
# `DATASET_VARIANTS["mmb0-1b_smb1-1b_1p-ablations-v9"]` as separate LSF
# jobs (one bsub per variant, one GPU exclusive each).
#
# v9 design: SMALLER + FASTER architecture, 2 x 4 grid.
#
#   architectural size   ∈ { "compact"  (h32 + codebook (30, 10, 10)),
#                            "medium"   (h64 + codebook (30, 20, 10)) }
#
#   batch integration    ∈ { adv (wt=150),         # default
#                            adv-w300,             # boosted adv
#                            adv + mmd-w50,        # combo (adv + MMD)
#                            mmd-w100 (no adv) }   # MMD only
#
# Shared spine (NO +wide so 1-hop GNN/sampler=[8]/nbr_hops=1; 1 GNN
# layer is sufficient and we want the runtime gain from staying narrow):
#   +small(hidden=N)                       # encoder + GNN + decoders -> N
#   +rvq(branch=both, levels=[30, L1, L2]) # 3-level RVQ symmetric
#   +decoder_covariate                     # NicheCompass-shape batch slot
#   (batch-integration loss combo varies per variant)
#
# Effective codebook sizes (vs v5 anchor 216k codes):
#   (30, 10, 10) -> 3,000 codes
#   (30, 20, 10) -> 6,000 codes
#
# Motivation: smaller / shallower encoders helped niche-NMI in past
# ablations but hurt iLISI. The four batch-integration strategies span
# the space from default adv pressure to MMD-only — pairing each
# strategy at TWO size scales lets us read "does compact architecture
# work AND how much batch-integration pressure does it need?" off
# the resulting 4 x 2 metric grid.
#
# Concrete variant list (in submission order):
#   Compact (h32 + (30, 10, 10)):
#     1. +adv                            # default pressure
#     2. +adv-w300                       # boosted adv
#     3. +adv +mmd-w50                   # combo
#     4. +mmd-w100                       # MMD only (no adv)
#   Medium (h64 + (30, 20, 10)):
#     5. +adv                            # default pressure
#     6. +adv-w300                       # boosted adv
#     7. +adv +mmd-w50                   # combo
#     8. +mmd-w100                       # MMD only (no adv)
#
# All env-var overrides from `submit_dataset_sweep.sh` work here too:
#
#   # Dry-run first (recommended) — prints bsub commands without submitting:
#   DRY_RUN=1 bash examples/submit_mmb_smb_ablations_v9.sh
#
#   # Force a longer wallclock or more memory:
#   LSF_WALL=48:00 LSF_MEM_MB=200000 \
#       bash examples/submit_mmb_smb_ablations_v9.sh
#
#   # Use a different rapids env for the UMAP step:
#   RAPIDS_ENV=/path/to/conda/env \
#       bash examples/submit_mmb_smb_ablations_v9.sh
#
# Logs land at:
#   /nfs/team361/sb75/squint-reproducibility/artifacts/logs/
#       mmb0-1b_smb1-1b_1p-ablations-v9/<variant>.{out,err}
# Per-job artifacts (checkpoints, predicted_adata.h5ad, plots, metrics) at:
#   /nfs/team361/sb75/squint-reproducibility/artifacts/
#       mmb0-1b_smb1-1b_1p/<variant>/<timestamp>/
# -----------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
exec bash "$SCRIPT_DIR/submit_dataset_sweep.sh" "mmb0-1b_smb1-1b_1p-ablations-v9" "$@"
