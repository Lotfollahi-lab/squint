#!/usr/bin/env bash
# submit_mmb_smb_ablations_v5.sh
# -----------------------------------------------------------------------------
# Convenience wrapper: submits all 8 variants in
# `DATASET_VARIANTS["mmb0-1b_smb1-1b_1p-ablations-v5"]` as separate LSF
# jobs (one bsub per variant, one GPU exclusive each).
#
# v5 spine: `+wide+rvq-both-3level+decoder-cov+adv+enc-deeper` -- the
# new project default and v4 winner. Combines the three independently-
# winning axes from prior rounds:
#   - `+wide`              -> niche NMI (v1 winner)
#   - `+enc-deeper`        -> batch integration (v4 winner)
#   - `+rvq-both-3level`   -> Pearson reconstruction (v3 winner)
#
# v5 holds those constant and varies ONLY the codebook structure, to
# isolate the codebook-capacity signal:
#
#   1. Anchor (the new default).
#   2. RVQ-3-level scaled symmetric (50, 100, 200) -- ~5x capacity.
#   3. RVQ-3-level scaled symmetric (80, 160, 320) -- ~19x capacity.
#   4. RVQ-3-level scaled DOWN (20, 40, 80) -- 0.3x capacity, useful
#      smaller-is-enough negative result for the paper.
#   5. MLP narrow: encoder + decoder hidden 400 -> 256 (lockstep).
#   6. MLP wide:   encoder + decoder hidden 400 -> 512 (lockstep).
#   7. CVQ-3-level (30, 10, 5) -- tree partitioning at matched depth.
#      `ConditionalVQ` was extended to 3 levels for v5.
#   8. CVQ-3-level (50, 15, 5) -- larger tree, A/B vs #7.
#
# All env-var overrides from `submit_dataset_sweep.sh` work here too,
# e.g.:
#
#   # Dry-run first (recommended) -- prints bsub commands without submitting:
#   DRY_RUN=1 bash examples/submit_mmb_smb_ablations_v5.sh
#
#   # Force a longer wallclock or more memory:
#   LSF_WALL=48:00 LSF_MEM_MB=200000 \
#       bash examples/submit_mmb_smb_ablations_v5.sh
#
#   # Use a different rapids env for the UMAP step:
#   RAPIDS_ENV=/path/to/conda/env \
#       bash examples/submit_mmb_smb_ablations_v5.sh
#
# Logs land at:
#   /nfs/team361/sb75/squint-reproducibility/artifacts/logs/
#       mmb0-1b_smb1-1b_1p-ablations-v5/<variant>.{out,err}
# Per-job artifacts (checkpoints, predicted_adata.h5ad, plots, metrics) at:
#   /nfs/team361/sb75/squint-reproducibility/artifacts/
#       mmb0-1b_smb1-1b_1p/<variant>/<timestamp>/
# -----------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
exec bash "$SCRIPT_DIR/submit_dataset_sweep.sh" "mmb0-1b_smb1-1b_1p-ablations-v5" "$@"
