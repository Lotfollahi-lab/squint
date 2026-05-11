"""
Example: run SQUINT on two AnnData sections from different platforms.

Data assumed at:
    /lustre/scratch126/cellgen/lotfollahi/DATASETS/silver/mmb0-1b_smb1-1b_1p_coord_aligned/
        ├── harmonised_merfish_mouse_brain_239-batch_batch82_shared_genes.h5ad
        └── harmonised_starmap_plus_mouse_cns_batch15_shared_genes.h5ad

Both files share the same gene panel (the "shared_genes" suffix). They are two
batches of one combined dataset called  `mmb0-1b_smb1-1b_1p_coord_aligned`  in
SQUINT's naming scheme:
    mmb0-1b   = MERFISH Mouse Brain dataset 0,    1 batch (batch82)
    smb1-1b   = STARmap+ Mouse Brain dataset 1,   1 batch (batch15)
    1p        = single shared gene panel
    coord_aligned = the xy_coordinates of the two sections have been aligned

All training artifacts (model checkpoints, configs, inference AnnData files,
metrics, etc.) are written under
    /nfs/team361/sb75/squint-reproducibility/artifacts
Training metrics are logged to wandb under the project "squint".


==============================================================================
                          USAGE  (4 commands)
==============================================================================

  # 0) one-time only — patch .uns metadata of the two .h5ad files if they
  #    don't yet contain the keys SQUINT needs (see "AnnData expectations"
  #    below). Skip if your files already have them.
  python examples/run_squint.py --patch-uns

  # 1) one-time only — preprocess the two .h5ad files into a cached PyG
  #    DatasetBlob.  Output:
  #        <silver-root>/gold/in-memory-PyG-dataset-blob/<DATASET_NAME>/
  python examples/run_squint.py --build-blob

  # 2) train SQUINT on the WHOLE dataset (both batches, every cell).
  #    --variant <name>  picks the ablation. Use --list-variants to see all.
  #
  #    Key variants:
  #      recon-cell        per-cell NB reconstruction only        (start here)
  #      recon-nbr         neighbourhood NB reconstruction only
  #      recon-both        per-cell + neighbourhood NB (two loss terms)
  #      recon-cell+adj    cell NB + adjacency BCE
  #      recon-nbr+adj     neighbourhood NB + adjacency BCE
  #      recon-both+adj    both NB + adjacency BCE
  #      recon-cell+film   cell NB + FiLM batch conditioning
  #      recon-cell+mask   cell NB + MAE feature masking
  #      recon-cell+mhvq   cell NB + multi-head VQ (10 heads, k=5000)
  #      recon-both+mhvq   cell+nbr NB + multi-head VQ (10 heads, k=5000)
  #
  #      RVQ ablations (Residual VQ, levels=[30, 200]):
  #        recon-cell+rvq, recon-nbr+rvq, recon-both+rvq,
  #        recon-cell+rvq+adj, recon-nbr+rvq+adj, recon-both+rvq+adj,
  #        recon-cell+rvq+film, recon-cell+rvq+mask
  #
  #      CVQ ablations (Conditional / Tree VQ, K1=30, K2=10):
  #        recon-cell+cvq, recon-nbr+cvq, recon-both+cvq,
  #        recon-cell+cvq+adj, recon-nbr+cvq+adj, recon-both+cvq+adj,
  #        recon-cell+cvq+film, recon-cell+cvq+mask
  #
  #      full              all components composed
  #
  #    Output (flat run dir, ablation-aware):
  #      <ARTIFACTS_DIR>/<DATASET_NAME>/<VARIANT>/<YYYYMMDD_HHMMSS>/
  #          user_specified_config.yaml      <- full materialised config
  #          ablation_summary.yaml           <- variant + description + patches
  #          checkpoints/<best>.ckpt
  #          wandb/run-<id>/...              (wandb local cache; cloud logging
  #                                           still goes to project "squint")
  python examples/run_squint.py --train --variant recon-cell
  python examples/run_squint.py --train --variant recon-both+adj

  # 3) inference on a folder of .h5ad files. Returns a single AnnData with
  #    the SQUINT outputs in .obsm / .obs / .layers, saved as .h5ad inside
  #    the run_dir itself (predicted_adata.h5ad alongside checkpoints and
  #    user_specified_config.yaml). --run-dir is the directory printed at
  #    the end of step 2.
  python examples/run_squint.py --predict \
      --run-dir <ARTIFACTS_DIR>/<DATASET_NAME>/<VARIANT>/<TIMESTAMP> \
      --silver-dir /lustre/scratch126/cellgen/lotfollahi/DATASETS/silver/mmb0-1b_smb1-1b_1p_coord_aligned

  # (you can omit --silver-dir to default to the path used during training.)


==============================================================================
                  Layout of the inference AnnData (step 3)
==============================================================================

  adata.X                  raw input counts (concatenated, sparse if input was)
  adata.var.index          gene symbols (shared panel)
  adata.obs['cell_type']           cell-type labels (string)
  adata.obs['adata_batch_id']      0 / 1 -> position of the source AnnData
  adata.obs['source_file']         filename of the source .h5ad
  adata.obs['code_index']          (only when codebook has 1 head) per-cell
                                   codebook index assigned by VQ
  adata.obsm['spatial']            (n_cells, 2) xy coordinates
  adata.obsm['X_squint']           SQUINT continuous latent (encoder output)
  adata.obsm['X_squint_quantized'] SQUINT VQ-snapped latent (codebook embed)
  adata.obsm['X_squint_adj']       latent used for adjacency reconstruction
  adata.obsm['code_indices']       (n_cells, n_heads) codebook indices, when
                                   the codebook has more than one head
  adata.layers['X_hat']            reconstructed counts (NB rate * read depth)
  adata.layers['X_hat_nbr']        reconstructed 1-hop neighborhood mean
  adata.uns['squint']              {codebook_size, num_heads, separate_codebook,
                                   run_dir, ckpt}


==============================================================================
                       AnnData expectations  (input)
==============================================================================

Each .h5ad must contain:
    .X                   raw integer counts (sparse CSR is fine)
    .var.index           gene symbols, identical across files
    .obs['cell_type']    cell-type labels (string)
    .obs['cell_id']      unique cell IDs that include "batch82" / "batch15"
                         (used by SQUINT to build per-cell batch one-hots
                         for FiLM conditioning)
    .obsm['spatial']     (n_cells, 2) array of xy coordinates
    .uns['batch']        e.g. "batch82" or "batch15"  (int(s[5:]) must work)
    .uns['dataset_id']   string, e.g. "mmb0" / "smb1"
    .uns['tissue']       string, e.g. "mouse_brain"
    .uns['species']      string, e.g. "mouse"

If any of these are missing, run `--patch-uns` once.
"""
import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import List, Optional, Sequence

# Silence two upstream FutureWarnings that fire at import time of our deps:
#   1. dask: legacy DataFrame deprecation ("Set dataframe.query-planning=True").
#      Triggered transitively (anndata -> ... -> dask) before any user code
#      runs. Prefer turning planning on directly when dask is available;
#      fall back to a warnings filter so older dask versions don't break.
#   2. anndata: "Importing read_text from `anndata` is deprecated. Import
#      anndata.io.read_text instead." Triggered by a transitive dep
#      (likely scvi-tools / scanpy / squidpy) using the old import path —
#      we don't import read_text ourselves. Filter by message text since
#      `category=FutureWarning` alone is too broad and would hide
#      genuine deprecations from our own code.
# Both filters must run BEFORE the dependencies are imported to take effect.
warnings.filterwarnings(
    "ignore",
    message=r".*Importing read_text from `anndata` is deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*legacy Dask DataFrame implementation is deprecated.*",
    category=FutureWarning,
)
try:
    import dask
    dask.config.set({"dataframe.query-planning": True})
except Exception:  # noqa: BLE001
    # dask not installed or older version without query-planning -- the
    # warnings filter above will catch the FutureWarning anyway.
    pass

import yaml


# ---------------------------------------------------------------------------
# Configuration paths (edit these if you move the data or repos)
# ---------------------------------------------------------------------------

# Parent directory that holds  silver/  and  gold/  subfolders.
DATA_ROOT = Path("/lustre/scratch126/cellgen/lotfollahi/DATASETS")

# Dataset name = subfolder name under  silver/
DATASET_NAME = "mmb0-1b_smb1-1b_1p_coord_aligned"

# Path to your local clones of the two repos on the cluster
REPRO_REPO = Path("/nfs/team361/sb75/squint-reproducibility")
SQUINT_PKG = Path("/nfs/team361/sb75/squint")

# Single root for everything we produce: configs, wandb logs, checkpoints,
# inference outputs.
ARTIFACTS_DIR = Path("/nfs/team361/sb75/squint-reproducibility/artifacts")

# Where this example writes its tailored config files.
CONFIG_OUT_DIR = ARTIFACTS_DIR / "configs"

# Where wandb writes run directories. The training script's
# `set_wandb_experiment_dir` will append <dataset>/standalone/<model>/...
# under here.
LOG_ROOT = ARTIFACTS_DIR / "wandb"

# Inference results (predicted_adata.h5ad + plots + metrics) are written
# directly into the run_dir
# (`ARTIFACTS_DIR/<dataset>/<variant>/<timestamp>/`) so everything for one
# training run lives in a single folder.

# Project name in wandb.
WANDB_PROJECT = "squint"


# ---------------------------------------------------------------------------
# In-training metric toggle (Pearson correlations)
# ---------------------------------------------------------------------------
# The model's `on_validation_epoch_end` hook calls
# `compute_benchmarking_metrics(adata)` for every entry in the config's
# `train_metrics_list` (which is also used for val mode, see
# `BaseModel.compute_metrics`). Each Pearson metric reconstructs the full
# val AnnData from the inference cache and runs a cells x genes
# correlation, costing 5-15 s per metric on the 100k-cell datasets.
# With the 14-entry list below that compounds to 100-200 s of pure CPU
# work per val epoch, completely starving the GPU.
#
# DEFAULT: empty lists -> losses-only tracking during training. Final
# Pearson metrics for the benchmark CSVs are produced by the
# `compute_inference_metrics.py` step at the end of the pipeline,
# which is the source of truth that benchmark figure scripts read
# from. The per-epoch values were only ever for live wandb diagnostics.
#
# OPT-IN: set environment variable `SQUINT_WITH_PEARSON=1` (or pass
# `--with-pearson` to `run_squint.py`) to populate train/test
# `*_metrics_list` with the 14 Pearson variants. Use this for one-off
# diagnostic runs when investigating training instabilities — expect
# 10-20x slower epochs.
_PEARSON_METRICS_LIST = [
    "codebook_utilization",
    # cell branch — gene-wise
    "pearson_gene_wise_log1p",
    "pearson_gene_wise_log1p_median",
    "pearson_gene_wise_hvg50_log1p",
    "pearson_gene_wise_hvg50_log1p_median",
    # cell branch — cell-wise
    "pearson_cell_wise_log1p",
    "pearson_cell_wise_log1p_median",
    # neighbourhood branch — gene-wise
    "pearson_gene_wise_1hop_nbr_log1p",
    "pearson_gene_wise_1hop_nbr_log1p_median",
    "pearson_gene_wise_hvg50_1hop_nbr_log1p",
    "pearson_gene_wise_hvg50_1hop_nbr_log1p_median",
    # neighbourhood branch — cell-wise
    "pearson_cell_wise_1hop_nbr_log1p",
    "pearson_cell_wise_1hop_nbr_log1p_median",
    # legacy raw-count metric
    "pearson_gene_wise",
]


def _resolve_metrics_list() -> list:
    """Return the in-training metrics list for the current run.

    - `SQUINT_WITH_PEARSON=1` (or `--with-pearson` on the CLI, which
      exports this env var in `main()`): returns the full Pearson
      list. Per-epoch Pearson appears in wandb again at the cost of
      10-20x slower epochs.
    - Anything else (including unset): returns `[]`. Training runs
      losses-only; final Pearson breakdown still comes from
      `compute_inference_metrics.py` at the end of the pipeline.

    Evaluated at config-build time (inside `make_train_config*`), so
    the env var must be set BEFORE `train(variant)` is called. `main()`
    handles that for the `--with-pearson` flag; manual env-var
    invocations also work (`SQUINT_WITH_PEARSON=1 python ...`).
    """
    if os.environ.get("SQUINT_WITH_PEARSON", "0") == "1":
        return list(_PEARSON_METRICS_LIST)
    return []


# ---------------------------------------------------------------------------
# 1. Dataset-blob config (one-time preprocessing)
# ---------------------------------------------------------------------------

def make_dataset_blob_config_chl59() -> dict:
    """
    Dataset-blob build config for the CosMx Lung dataset
    (`/nfs/team361/sb75/DATASETS/silver/chl59-8b_1p`, 8 AnnDatas).

    Differences from `make_dataset_blob_config`:
      - root data path -> /nfs/team361/sb75/DATASETS
      - dataset name -> chl59-8b_1p
      - label_names empty by default — the Lung AnnDatas may not have
        an `obs['cell_type']` column. The blob loader is now defensive
        (skips missing label columns with a warning), so you can put
        labels back if your AnnDatas DO have them under a different
        name, e.g. `label_names=["cell_types=celltype"]`.
      - `graph_kwargs.batch_key` -> 'batch'. If your Lung AnnDatas don't
        have `obs['batch']` (the user confirmed batch info is encoded
        in `obs['cell_id']` strings as '..._batchN_...'), the loader
        will silently fall back to parsing `cell_id` for the batch
        one-hot. So this default is safe even when obs['batch'] is
        absent.
    """
    return {
        "experiment": {
            "name": "in_memory_dataset_blob",
            "description": "CosMx Lung — 8 samples (Lung5/6/9/12/13 incl. replicates)",
        },
        "dataset": {
            "name": "chl59-8b_1p",
            "feature_names": ["cell_gene_counts"],
            "label_names": [],   # set to ["cell_types=<your_obs_key>"] if available
            "graph_kwargs": {
                "coord_type":         "generic",
                "spatial_key":        "spatial",
                "n_neighs_list":      [8],
                "radius_list":        None,
                "include_self_loop":  True,
                "batch_key":          "batch",     # falls back to cell_id if absent
                "k": {
                    "lm_eigvecs": 128,
                },
            },
            "data_directory_path": "/nfs/team361/sb75/DATASETS",
            "pre_transform": None,
            "pre_filter":    None,
            "overwrite":     True,
        },
        "software_paths": {
            "deepwalk": "",
            "gosh":     "",
        },
    }


def make_dataset_blob_config_mmb20() -> dict:
    """
    Dataset-blob build config for the cross-platform retrieval dataset
    (`/lustre/.../silver/mmb0-1b_smb1-20b_1p_shared_genes`, 21 AnnDatas:
    1 MERFISH mouse-brain section + 20 STARmap+ mouse-CNS sections, all
    reindexed to the MERFISH gene panel via examples/harmonize_mmb_smb20_panels.py).

    Run order (one-time):
        1. python examples/harmonize_mmb_smb20_panels.py
           -> writes harmonised AnnDatas to <DATA_ROOT>/silver/<dataset>_shared_genes/
        2. python examples/run_squint.py --build-blob --build-blob-dataset mmb20
           -> reads from the dir written in step 1, builds dataset_blob.pt

    Notes:
      - root data path -> /lustre/scratch126/cellgen/lotfollahi/DATASETS
        (where the original mmb0-1b_smb1-20b_1p folder lives).
      - dataset name  -> mmb0-1b_smb1-20b_1p_shared_genes (matches the
        output dir of harmonize_mmb_smb20_panels.py).
      - label_names empty: STARmap obs columns vary (some have
        Sub_molecular_tissue_region, others may have cell_type only); the
        blob loader is defensive and skips missing label columns. The
        per-cell labels are still propagated to inference via
        `obs_per_batch_id` (we now carry the full obs DataFrames) — no
        need to register them as supervised y_* tensors here.
      - graph_kwargs.batch_key -> 'batch'. The harmonize script stamps
        every AnnData with obs['batch'] = 'batchN', so
        `build_batch_one_hot_from_obs` densifies cleanly.
    """
    return {
        "experiment": {
            "name": "in_memory_dataset_blob",
            "description": (
                "MERFISH mouse-brain (1 section) + STARmap+ mouse-CNS "
                "(20 sections) on the shared MERFISH gene panel."
            ),
        },
        "dataset": {
            "name": "mmb0-1b_smb1-20b_1p_shared_genes",
            "feature_names": ["cell_gene_counts"],
            "label_names": [],
            "graph_kwargs": {
                "coord_type":         "generic",
                "spatial_key":        "spatial",
                "n_neighs_list":      [8],
                "radius_list":        None,
                "include_self_loop":  True,
                "batch_key":          "batch",
                "k": {
                    "lm_eigvecs": 128,
                },
            },
            "data_directory_path": str(DATA_ROOT),
            "pre_transform": None,
            "pre_filter":    None,
            "overwrite":     True,
        },
        "software_paths": {
            "deepwalk": "",
            "gosh":     "",
        },
    }


def make_dataset_blob_config() -> dict:
    """
    Mirrors  config/create_in_memory_dataset_blob/mmb0-4b_1p.yaml  but for
    our two-section dataset. We turn off DeepWalk/GOSH (which need extra
    binaries) and only build Laplacian eigenvectors. We also build the 8-NN
    spatial graph that SQUINT uses by default.
    """
    return {
        "experiment": {
            "name": "in_memory_dataset_blob",
            "description": "MERFISH MB batch82 + STARmap+ MB batch15 (shared gene panel, coord aligned)",
        },
        "dataset": {
            "name": DATASET_NAME,
            "feature_names": ["cell_gene_counts"],
            "label_names": [
                "cell_types=cell_type",
                # If your AnnData also has e.g. region labels in obs['region'],
                # add: "niche_types=region"
            ],
            "graph_kwargs": {
                "coord_type": "generic",
                "spatial_key": "spatial",     # adata.obsm['spatial']
                "n_neighs_list": [8],         # SQUINT default 8-NN graph
                "radius_list": None,
                "include_self_loop": True,
                "k": {
                    # Spectral embeddings — cheap, optional.
                    "lm_eigvecs": 128,
                    # DeepWalk / GOSH skipped (require external binaries).
                },
            },
            "data_directory_path": str(DATA_ROOT),
            "pre_transform": None,
            "pre_filter": None,
            "overwrite": True,
        },
        "software_paths": {
            # not used because we omit deepwalk/gosh from k above
            "deepwalk": "",
            "gosh": "",
        },
    }


# ---------------------------------------------------------------------------
# 2a. Training config — minimal POC (Graph VQ-VAE baseline)
# ---------------------------------------------------------------------------

def make_train_config_poc() -> dict:
    """
    Right-sized Graph VQ-VAE baseline for 431-gene / ~100k-cell spatial data.

    Design choices:
    - No HVG subsetting (`apply_hvg=False`): the gene panel is already curated;
      set apply_hvg=True (and adjust n_hvg) for larger multi-panel datasets.
    - Encoder: 431 -> MLP[400,256] -> GNN[128, 1 layer, 8-NN] -> VQ[30 codes].
      Latent dim = 128 keeps the VQ cosine-similarity manifold tractable for
      30 codes.  A 30-code codebook with 400-dim embeddings is over-parametrised
      and slows EMA convergence.
    - Decoder: latent[128] -> MLP[400] -> softmax * read_depth -> NB mean.
    - Adjacency decoder: kept at low capacity (128 -> MLP[64]) to avoid the
      BCE adj loss dominating when it's added.
    - Dropout: removed.  With a 30-code bottleneck the model has limited spare
      capacity and dropout further destabilises VQ code assignment.
    - Sampler: 1 hop, 8 neighbors — matches GNN num_layers=1.  Sampling 2 hops
      for a 1-layer GNN is wasted compute.
    - LR: 5e-4 (conservative; 1e-3 tends to cause sharp loss spikes on VQ init).
    - Batch: 256.  ~100k cells -> ~390 batches/epoch, sufficient gradient
      diversity for EMA codebook updates.
    - Epochs: 80.  EMA VQ needs ~40-60 epochs to stabilise dead-code rate.
    - Metrics: pearson_gene_wise_log1p is the primary checkpoint signal.
      It is the metric reported by scVI-tools, NicheCompass, and spatial-
      imputation papers (Tangram, SpaGE, STAGATE).  Per-gene Pearson on
      log1p(counts) assigns equal weight to all genes and is insensitive to
      the mean-count dominance that makes raw-count cell-wise Pearson decrease
      early in training.  pearson_cell_wise_log1p and pearson_gene_wise are
      tracked as secondary metrics.

    recon_mode is injected by patch helpers and defaults to 'cell' here.
    """
    return {
        "experiment": {"seed": 0, "mode": "standalone"},
        "logging": {
            "root_log_dir": str(LOG_ROOT),
            # Don't upload checkpoints to wandb. Lightning's
            # ModelCheckpoint already writes the best ckpt to
            # `<run_dir>/checkpoints/` on NFS, which is the canonical
            # location predict() reads from. Wandb upload of large
            # tensors stages each ckpt into $WANDB_CACHE_DIR (default
            # $HOME/.cache/wandb/artifacts/staging/) and across a
            # multi-variant sweep that overflows typical HPC home quotas.
            "log_model": False,
            "offline": False,
            "enabled": True,
        },
        "dataset": {
            "dataset_name": DATASET_NAME,
            # `dataset_tag` is the SHORT identifier used for artifact /
            # run directory paths (`<ARTIFACTS_DIR>/<dataset_tag>/<variant>/
            # <timestamp>/`). It strips the technical suffixes that live
            # in `dataset_name` (`_coord_aligned`, `_shared_genes`) — those
            # belong to the silver / gold blob paths but make on-disk
            # artifact dirs unwieldy. If a future variant doesn't set
            # `dataset_tag`, the artifact path falls back to
            # `dataset_name`.
            "dataset_tag": "mmb0-1b_smb1-1b_1p",
            # adata_batch_idx are *positions* in the sorted DatasetBlob:
            #   idx 0 -> batch15 (STARmap+)   idx 1 -> batch82 (MERFISH)
            "adata_batch_idx": [0, 1],
            "root_data_dir": str(DATA_ROOT),
            # HVG subsetting — OFF for this dataset (431 curated genes).
            # Switch apply_hvg=True and set n_hvg to a smaller number when
            # running on larger multi-panel datasets where all genes are NOT
            # equally informative.
            "apply_hvg": False,
            "n_hvg": 2000,                    # only used when apply_hvg=True
            "gene_count_transform_names": [], # populated by train() based on apply_hvg
            "gene_count_transform_params": {"n_genes": 2000},
            "graph_params": {
                "spatial_key": "spatial",
                "delaunay": False,
                "n_neighs": 8,
                "radius": None,
                # Source for per-cell batch labels used by FiLM batch-correction
                # and by `build_batch_one_hot_from_obs`. When this column is
                # present in adata.obs at blob-build time, its values are
                # stored per-cell and used downstream (densified to contiguous
                # indices). When absent, the loader falls back to parsing
                # "batchN" substrings out of `obs['cell_id']`.
                "batch_key": "batch",
            },
            "feature_names": ["X"],
            "label_name": "cell_types",
            "train_transform_names": ["SpatialBatchSplit"],
            "train_transform_params": {
                "region": None,
                "train_batches": [15, 82],
                "val_batches": [],
                "test_batches": [],
                "xy_key": "xy_coordinates",
                # Cell-level holdout for in-distribution early stopping +
                # best-checkpoint-on-val. ONLY applied when `val_batches`
                # is empty (no whole-section holdout configured). When
                # `val_batches` is non-empty, that section IS the val set
                # — we do NOT additionally sub-sample training cells.
                # Adversarial training can be unstable late; a 10%
                # in-section val sample gives a stable in-distribution
                # signal that early stopping and best-ckpt-on-val use.
                # Set to 0.0 to disable.
                "train_val_cell_split": 0.10,
                "cell_split_seed":      0,
            },
        },
        "datamodule": {
            "loader_name": "NeighborLoader",
            "loader_params": {
                # batch_size bumped from 256 -> 1024 after wandb showed
                # the GPU peaking at ~25 % utilization with batch=256:
                # the model is too cheap per step to saturate a modern
                # GPU at that batch size, and the GPU memory was only
                # ~5 % allocated, so there's plenty of headroom. With
                # batch=1024 the GPU does 4x more work per forward pass,
                # raising the duty cycle without hitting OOM.
                # NOTE: batch_size CHANGES training dynamics (effective
                # learning rate scales with batch size). Pre-v11 runs at
                # batch=256 are NOT directly comparable to post-v11
                # results at batch=1024 — re-benchmark before drawing
                # cross-run conclusions. If you want apples-to-apples,
                # override via a patch (e.g. `_patch_dual_batch_size(256)`)
                # for the comparison runs.
                "batch_size":         1024,
                # 16 workers to feed the 4x larger batches. The
                # DataModule's auto-derivation
                # (`LSB_DJOB_NUMPROC // 2`) yields ~10 at LSF_CORES=20;
                # explicit override here pins it at 16 so the GPU
                # stays fed even on hosts where LSB_DJOB_NUMPROC
                # reports less than expected. `pin_memory=True`,
                # `persistent_workers=True`, and `prefetch_factor=4`
                # are added automatically by
                # `InMemoryDataModule._loader_perf_kwargs()` — don't
                # set them here, that produces a `got multiple values
                # for keyword argument` TypeError on the loader call.
                "num_workers":        16,
            },
            "sampler_name": "NeighborSampler",
            "sampler_params": {"num_neighbors": [8]},  # 1 hop = num_layers
            # Flipped from False to True (was the GPU-utilization
            # bottleneck): the val + test + inference loaders all share
            # this flag, and `False` made them sample `num_neighbors=[-1]`
            # (every neighbor) — for a 100k-cell graph with ~10-50
            # neighbors per node, that's 10-100x more sampler work per
            # batch than training (which uses the `[8]`-fanout
            # configured below). With val running every epoch, the
            # heavy val pass stalled the GPU between train epochs,
            # producing the spike-then-long-gap utilization pattern.
            # `True` makes val/test/inference use the same `[8]`
            # sampler as training — apples-to-apples, much faster.
            #
            # Implications worth knowing:
            #   - val_loss numerical values shift slightly (val sees a
            #     sparser neighborhood). Early stopping still works,
            #     but the absolute val_loss curves are not comparable
            #     to pre-v11 runs.
            #   - predict()-time inference also uses `[8]`: each cell
            #     sees ONE random sample of 8 neighbors at predict
            #     time. Per-cell X_hat is now stochastic across
            #     repeated predict() calls (was deterministic before).
            #     For benchmarks across seeds this just adds a small
            #     amount of additional variance to NMI / iLISI / Pearson
            #     — within the existing inter-seed spread.
            #   - If you specifically need deterministic, all-context
            #     inference for a paper figure, override per-variant via
            #     `cfg["datamodule"]["inference_params"]["sample_neighbors_for_inference"] = False`.
            "inference_params": {"sample_neighbors_for_inference": True},
        },
        "model": {
            "model_name": "VQNiche",
            "encoder_name": "VQNiche_Encoder",
            "attribute_decoder_name": "MLPSoftmax",
            "adjacency_decoder_name": "MLP_AdjacencyDecoder",
            "predictor_name": "Linear",
            "imputation_params": {
                "mask_strategy": "original",
                "base_mask_ratio": 0.0,
                "final_mask_ratio": 0.0,
                "warmup_epochs": 0,
                "deterministic_masking": False,
                "compute_mask_input_diversity": False,
                "mask_token_eps": 0.0001,
            },
            # All Pearson metrics are reported on every variant.
            # Neighbourhood metrics (1hop_nbr) are silently skipped when
            # X_nbr / X_hat_nbr are absent (recon_mode='cell' variants).
            # In-training metric computation is OFF by default. Both
            # lists are populated by `_resolve_metrics_list()`, which
            # returns `[]` unless `SQUINT_WITH_PEARSON=1` is set
            # (passing `--with-pearson` to `main()` exports that env
            # var automatically). See the module-level docstring for
            # `_resolve_metrics_list()` for the full rationale.
            #
            # Why both lists use the same source:
            #   - train_metrics_list is consumed by
            #     BaseModel.compute_metrics(mode='train' | 'val'),
            #     fired from on_validation_epoch_end.
            #   - test_metrics_list is consumed by the same helper
            #     with mode='test', fired from trainer.test().
            # The standard pipeline doesn't call trainer.test() (we
            # use predict + compute_inference_metrics.py instead), so
            # test_metrics_list is essentially dormant. But aligning
            # it with train avoids surprise behaviour if anyone wires
            # in a test loop later, and keeps a single switch for
            # "Pearson on / off everywhere during PL hooks".
            "train_metrics_list": _resolve_metrics_list(),
            "test_metrics_list":  _resolve_metrics_list(),
            "encoder_params": {
                "gnn_name": "SAGEConv",
                "mlp_params": {
                    "hidden_channels": [400, 256],
                    "dropout": 0.0,
                    "act": "relu",
                    "norm": None,
                },
                "gnn_params": {
                    # Latent dim doubled from 128 -> 256: this is the codebook
                    # embedding dim for ALL VQ variants. With codebook=30 and
                    # 431 input genes, 128 was a tight bottleneck; 256 gives
                    # the codebook ~2x more representational capacity.
                    "hidden_channels": 256,
                    "num_layers": 1,
                    "act_first": True,
                    "activation": "relu",
                    "norm": None,
                    "dropout": 0.0,
                    "init_method": "kaiming_uniform",
                },
                # POC: no FiLM conditioning
                "vq_params": {
                    "vq_name": "VectorQuantize",
                    "freeze_codebook": False,
                    "use_cosine_sim": True,
                    "ema_update": True,
                    "manual_ema_update": False,
                    "threshold_ema_dead_code": 2,
                    "manual_in_place_optimizer_update": False,
                    "learnable_codebook": False,
                    "codebook_size": 30,
                    "heads": 1,
                    "separate_codebook_per_head": False,
                    "decay": 0.8,
                    "eps": 0.00001,
                    "kmeans_init": True,
                    "kmeans_iters": 10,
                    "sync_kmeans": True,
                    "sample_codebook_temp": 0.0,
                    "commitment_weight": 0.0,
                    "commitment_use_cross_entropy_loss": False,
                    "codebook_diversity_loss_weight": 0.0,
                    "codebook_diversity_temperature": 100.0,
                    "orthogonal_reg_weight": 0.0,
                    "orthogonal_reg_max_codes": None,
                    "orthogonal_reg_active_codes_only": False,
                },
            },
            "attribute_decoder_params": {
                "apply_conditioning": None,
                "mlp_params": {
                    # Decoder depth bumped from [400] -> [400, 400] to give
                    # the NB-mean head more capacity to expand the 256-dim
                    # latent into a 431-gene rate vector.
                    "hidden_channels": [400, 400],
                    "dropout": 0.0,
                    "act": "gelu",
                    "norm": "layer_norm",
                },
            },
            "adjacency_decoder_params": {
                "out_channels": 128,
                "mlp_params": {
                    "hidden_channels": [64],
                    "dropout": 0.0,
                    "act": "relu",
                    "norm": "layer_norm",
                },
            },
            "optimizer_params": {
                "optimizer_name": "adam",
                "lr": 0.0005,
                "weight_decay": 0.001,
                "mask_lr_scale": 1.0,
            },
            "loss_params": {
                "loss_names": [
                    "nb_attribute_reconstruction_loss",
                    "mse_commit_loss",
                ],
                "loss_kwargs": {
                    # recon_mode controls which NB targets are computed:
                    #   "cell" -> per-cell (default baseline)
                    #   "nbr"  -> 1-hop neighbourhood mean (niche-aware signal)
                    #   "both" -> cell + neighbourhood (requires adding
                    #             nb_attribute_reconstruction_loss_nbr to loss_names)
                    "recon_mode": "cell",
                    "only_masked": False,
                    "edge_sampling_ratio": 2,
                    "use_pos_weight": True,
                    "estimate_adj_kwargs": {"nonlinearity": "sigmoid", "k": 8},
                    "wt_attr_reconstr": 1.0,
                    "wt_attr_reconstr_nbr": 1.0,
                    # Code-conditional dispersion. When enabled, a small MLP
                    # maps z_q -> log(theta) per-cell-per-gene, replacing the
                    # global per-gene scalar parameter. Default OFF; ablate
                    # via a future +ccd patch.
                    # Format options:
                    #   False                              -> off
                    #   True                               -> on, linear head (no hidden)
                    #   {"enabled": True, "hidden_channels": [128]}
                    "code_conditional_dispersion": False,
                    # Adjacency-loss input mode.
                    # False (default): edge logits = h_adj_i^T h_adj_j where
                    #     h_adj is the output of the MLP adjacency_decoder.
                    # True            : edge logits = z_q_i^T z_q_j (the
                    #     quantized latent itself).  This is the
                    #     NicheCompass / standard graph-VAE formulation —
                    #     forces the codebook embeddings to directly encode
                    #     pairwise spatial proximity, which is exactly the
                    #     bias we want when codes should capture niches.
                    "bypass_adj_decoder": False,
                    "wt_commit": 1.0,
                },
            },
        },
        "trainer": {
            # Best-checkpoint metric: val_loss (sum of all weighted loss
            # terms over the in-distribution VAL set — same 10% cell-level
            # holdout from the training sections that early stopping
            # watches; see `train_val_cell_split`). Single source of truth
            # for "best epoch" — the saved ckpt is the one early stopping
            # would have stopped at. Robust to adversarial late-stage
            # instability: when the encoder oscillates, val_loss climbs
            # before train_loss does, so the kept snapshot is from before
            # the drift.
            "monitor": "val_loss",
            "checkpoint_params": {"mode": "min", "save_top_k": 1, "save_last": True},
            # max_epochs bumped 80 -> 200 when batch_size went 256 -> 1024.
            # Each epoch now contains ~4x fewer gradient steps (~97 vs
            # ~390 for a 100k-cell dataset), so 200 epochs at batch=1024
            # is roughly equivalent in total gradient-step budget to
            # 50 epochs at batch=256 — still less than the original 80,
            # but early stopping below will cap things in practice if
            # val_loss plateaus earlier.
            "max_epochs": 200,
            "enable_checkpointing": True,
            "ckpt_path": "best",
            # Early stopping: watch the val_loss (= sum of all weighted
            # loss terms over the val set). Switching from train_loss to
            # val_loss makes early stopping respond to in-distribution
            # generalisation rather than the optimisation objective on
            # the training set, which matters in particular for the
            # adversarial variants where train_loss can keep declining
            # while val performance silently degrades. patience=20 epochs
            # ≈ 8k gradient steps at batch_size=256 / ~100k cells —
            # enough to distinguish a true plateau from a codebook
            # reshuffle, and enough to ride through the adversarial
            # alpha ramp-up that comes after `adv-warmup10` (otherwise
            # patience=15 can fire during the 10-epoch warm-up + ~5
            # post-warm-up epochs, before the adv-driven val_loss drift
            # has had time to stabilise). min_delta=0.1: small absolute,
            # but val_loss in this codebase ranges ~140-170 so 0.1 ≈
            # 0.07% relative.
            "early_stopping_params": {
                "enabled":    True,
                "monitor":    "val_loss",
                "mode":       "min",
                "patience":   20,
                "min_delta":  0.1,
            },
        },
    }


# ---------------------------------------------------------------------------
# 2b. Training config — full SQUINT (FiLM + masking + adjacency)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 2b. Ablation patch helpers
# ---------------------------------------------------------------------------
# Each patch is a small function that mutates a DEEP COPY of a base config and
# returns it. A patch changes ONE design component so its effect can be
# isolated. Compose patches to build compound variants.
#
# The full materialised config is always saved to disk per run, so
# reproducibility does not depend on patch helpers being preserved.
# ---------------------------------------------------------------------------

def _copy(cfg: dict) -> dict:
    import copy
    return copy.deepcopy(cfg)


# ---- Reconstruction mode ---------------------------------------------------

def _patch_recon_cell(cfg: dict) -> dict:
    """Per-cell NB reconstruction only (baseline). Explicitly sets recon_mode."""
    lk = cfg["model"]["loss_params"]["loss_kwargs"]
    lk["recon_mode"] = "cell"
    losses = cfg["model"]["loss_params"]["loss_names"]
    # Ensure only the cell loss is present, remove nbr branch if any.
    if "nb_attribute_reconstruction_loss_nbr" in losses:
        losses.remove("nb_attribute_reconstruction_loss_nbr")
    if "nb_attribute_reconstruction_loss" not in losses:
        losses.insert(0, "nb_attribute_reconstruction_loss")
    cfg["trainer"]["monitor"] = "val_loss"
    return cfg


def _patch_recon_nbr(cfg: dict) -> dict:
    """
    Neighbourhood-only NB reconstruction.
    Per-cell NB is deactivated: recon_mode='nbr' makes training_step route
    the 1-hop-aggregated pair through the legacy pred_attr/target_attr keys
    so `nb_attribute_reconstruction_loss` computes NB on neighbourhood means.
    Per-cell counts are not reconstructed in this variant.
    """
    lk = cfg["model"]["loss_params"]["loss_kwargs"]
    lk["recon_mode"] = "nbr"
    losses = cfg["model"]["loss_params"]["loss_names"]
    if "nb_attribute_reconstruction_loss_nbr" in losses:
        losses.remove("nb_attribute_reconstruction_loss_nbr")
    if "nb_attribute_reconstruction_loss" not in losses:
        losses.insert(0, "nb_attribute_reconstruction_loss")
    cfg["trainer"]["monitor"] = "val_loss"
    return cfg


def _patch_recon_both(cfg: dict) -> dict:
    """
    Per-cell + 1-hop neighbourhood NB reconstruction simultaneously.
    Uses TWO separate loss dispatches:
      nb_attribute_reconstruction_loss     -> cell pair (pred_attr / target_attr)
      nb_attribute_reconstruction_loss_nbr -> nbr pair  (pred_attr_nbr / target_attr_nbr)
    Weighted equally by default (wt_attr_reconstr = wt_attr_reconstr_nbr = 1.0).
    """
    lk = cfg["model"]["loss_params"]["loss_kwargs"]
    lk["recon_mode"] = "both"
    losses = cfg["model"]["loss_params"]["loss_names"]
    if "nb_attribute_reconstruction_loss" not in losses:
        losses.insert(0, "nb_attribute_reconstruction_loss")
    if "nb_attribute_reconstruction_loss_nbr" not in losses:
        idx = losses.index("nb_attribute_reconstruction_loss") + 1
        losses.insert(idx, "nb_attribute_reconstruction_loss_nbr")
    cfg["trainer"]["monitor"] = "val_loss"
    return cfg


# ---- Adjacency reconstruction ----------------------------------------------

def _patch_adj(cfg: dict, weight: float = 1.0) -> dict:
    """
    Add BCE adjacency reconstruction loss.
    The adjacency decoder predicts which pairs of cells in the seed-node
    induced subgraph are connected in the 8-NN spatial graph. Pushes the
    H_adj latent toward encoding local spatial topology.
    Does NOT change recon_mode — stack with a recon patch for combinations.
    """
    losses = cfg["model"]["loss_params"]["loss_names"]
    if "bce_adjacency_reconstruction_loss" not in losses:
        losses.append("bce_adjacency_reconstruction_loss")
    cfg["model"]["loss_params"]["loss_kwargs"]["wt_adj_reconstr"] = float(weight)
    return cfg


# ---- FiLM conditioning -----------------------------------------------------

def _patch_film(cfg: dict) -> dict:
    """
    Add FiLM (feature-wise linear modulation) conditioning on
    (rbf_distances, cell_batch_id) in both encoder and attribute decoder.
    Enables the model to distinguish cells from different batches / spatial
    scales without leaking batch identity into the VQ codebook.
    """
    film_params = {
        "condition_list": ["rbf_distances", "cell_batch_id"],
        "use_bias": True,
        "use_residual": False,
        "residual_weight": 0.2,
        "init_mode": "identity",
    }
    cfg["model"]["encoder_params"]["conditioning_params"] = film_params.copy()
    cfg["model"]["attribute_decoder_params"]["apply_conditioning"] = "in-MLP"
    cfg["model"]["attribute_decoder_params"]["conditioning_params"] = film_params.copy()
    cfg["model"]["optimizer_params"]["mask_lr_scale"] = 1.0  # unchanged here
    return cfg


# ---- MAE-style masking -----------------------------------------------------

def _patch_masking(cfg: dict,
                   base_ratio: float = 0.2,
                   final_ratio: float = 0.6,
                   warmup: int = 5) -> dict:
    """
    Enable MAE-style gene masking. A learnable mask token replaces a random
    fraction of input genes; the model must reconstruct all genes from the
    partial signal. Forces the encoder to propagate neighbourhood context
    rather than copying the input. mask_lr_scale=2.0 gives the mask token
    a higher LR so it moves fast relative to the rest of the network.
    """
    cfg["model"]["imputation_params"] = {
        "mask_strategy": "learnable_parameter",
        "base_mask_ratio": base_ratio,
        "final_mask_ratio": final_ratio,
        "warmup_epochs": warmup,
        "deterministic_masking": False,
        "compute_mask_input_diversity": False,
        "mask_token_eps": 0.0001,
    }
    cfg["model"]["optimizer_params"]["mask_lr_scale"] = 2.0
    losses = cfg["model"]["loss_params"]["loss_names"]
    if "mask_token_regularization" not in losses:
        losses.append("mask_token_regularization")
    cfg["model"]["loss_params"]["loss_kwargs"]["wt_mask_token_regularization"] = 0.001
    return cfg


# ---- Multi-head VQ ---------------------------------------------------------

def _patch_multihead_vq(cfg: dict, heads: int = 10, codebook_size: int = 5000) -> dict:
    """
    Switch from a single 30-code codebook to a multi-head VQ with `heads`
    parallel codebooks of size `codebook_size` each. Total number of
    representable niche combinations = codebook_size^heads (exponential), but
    the model learns to use a much smaller effective set. Used in full SQUINT.
    """
    cfg["model"]["encoder_params"]["vq_params"].update({
        "codebook_size": codebook_size,
        "heads": heads,
        "separate_codebook_per_head": False,
    })
    return cfg


# ---- Residual VQ (RQ-VAE) --------------------------------------------------

def _patch_rvq(cfg: dict, codebook_sizes=(30, 200)) -> dict:
    """
    Switch the VQ to ResidualVQ_Squint: an L-level residual quantizer where
    each cell is encoded as a SUM of L codes drawn from L independent
    codebooks of (potentially different) sizes.

    Default: levels=[30, 200] -> 30 * 200 = 6000 distinct quantized
    representations, with a clean coarse->fine interpretation.
    Level 1 (30) acts as the macro niche; level 2 (200) as a refinement
    code applied uniformly across the whole space (no per-bucket codebook).

    The lucidrains-internal commit_loss is left at 0 (commitment_weight=0.0)
    so that the external `mse_commit_loss` remains the single source of
    commit signal — same convention as the rest of the variants.
    """
    sizes = list(codebook_sizes)
    cfg["model"]["encoder_params"]["vq_params"] = {
        "vq_name": "ResidualVQ_Squint",
        "num_quantizers": len(sizes),
        "codebook_size": sizes,             # accepted as list by our wrapper
        "use_cosine_sim": True,
        "ema_update": True,
        "decay": 0.8,
        "eps": 0.00001,
        "threshold_ema_dead_code": 2,       # only applied to level 1 internally
        "kmeans_init": True,                # only applied to level 1 internally
        "kmeans_iters": 10,
        "sync_kmeans": True,
        "commitment_weight": 0.0,
        "sample_codebook_temp": 0.0,
    }
    return cfg


# ---- Conditional / Tree VQ -------------------------------------------------

def _patch_cvq(cfg: dict, k1: int = 30, k2: int = 10) -> dict:
    """
    Switch the VQ to ConditionalVQ: a true 2-level tree where each level-1
    code (K1 of them) selects a separate level-2 codebook (K1 codebooks of
    K2 codes each).

    Default: K1=30, K2=10 -> 300 distinct quantized representations,
    organised hierarchically. With ~100k cells / 30 buckets ≈ 3300 cells per
    bucket / 10 codes ≈ 330 cells per (l1, l2) pair, level-2 codebooks have
    enough data to train cleanly.

    Level-2 codebooks have kmeans_init disabled and dead-code resampling
    disabled (per-bucket data is too sparse to cluster reliably and we don't
    want to spuriously revive level-2 codes in small buckets).
    """
    cfg["model"]["encoder_params"]["vq_params"] = {
        "vq_name": "ConditionalVQ",
        "codebook_size_l1": int(k1),
        "codebook_size_l2": int(k2),
        "use_cosine_sim": True,
        "ema_update": True,
        "decay": 0.8,
        "eps": 0.00001,
        "threshold_ema_dead_code": 2,
        "kmeans_init": True,
        "kmeans_iters": 10,
        "sync_kmeans": True,
        "commitment_weight": 0.0,
        "sample_codebook_temp": 0.0,
    }
    return cfg


# ---- NicheCompass-style adjacency reconstruction ---------------------------

def _patch_nichecompass_adj(cfg: dict, weight: float = 1.0) -> dict:
    """
    Switch the BCE adjacency reconstruction to the NicheCompass / standard
    graph-VAE formulation, and weight it on par with the gene-expression NB
    loss (default `weight=1.0`).

    Two changes vs. `_patch_adj`:
      (a) `bypass_adj_decoder=True` makes the BCE compute its edge logits
          directly on the quantized latent z_q (raw inner product
          z_q_i^T z_q_j), bypassing the SQUINT adjacency_decoder MLP. This
          mirrors NicheCompass's `A_hat = sigmoid(Z @ Z^T)` formulation —
          there is no extra non-linearity between the latent and the
          edge prediction. The codebook embeddings themselves are forced
          to encode pairwise spatial proximity.
      (b) `wt_adj_reconstr=1.0` (vs. the legacy default 0.1) gives the
          edge reconstruction roughly the same loss-magnitude footprint as
          NB attribute reconstruction. The NicheCompass user guide
          explicitly notes that "finding a balance between gene expression
          and edge reconstruction is a key element for good niche
          identification" — the default 0.1 weight in SQUINT was 10x too
          weak for a niche-focused signal.

    Adds `bce_adjacency_reconstruction_loss` to the loss list if not
    already present.
    """
    losses = cfg["model"]["loss_params"]["loss_names"]
    if "bce_adjacency_reconstruction_loss" not in losses:
        losses.append("bce_adjacency_reconstruction_loss")
    lk = cfg["model"]["loss_params"]["loss_kwargs"]
    lk["wt_adj_reconstr"]    = float(weight)
    lk["bypass_adj_decoder"] = True
    return cfg


# ---- GATv2 encoder (NicheCompass-style GNN) --------------------------------

def _patch_gatv2_encoder(cfg: dict) -> dict:
    """
    Switch the encoder GNN from GraphSAGE to GATv2 (dynamic attention).
    NicheCompass recommends GATv2 over GCN/GraphSAGE for niche-focused
    representations — attention weights are spatially adaptive, so cells
    in dense parts of a niche aggregate from more informative neighbours
    than cells at niche boundaries.

    The MLP / hidden-channel / depth structure is left unchanged — this
    patch only flips `gnn_name`. Use it stacked with a recon patch and
    `_patch_nichecompass_adj` to get a NicheCompass-shaped pipeline.
    """
    cfg["model"]["encoder_params"]["gnn_name"] = "GATv2Conv"
    return cfg


# ---------------------------------------------------------------------------
# 2c. Variant registry
# ---------------------------------------------------------------------------
# Naming convention: <recon-mode>[+<component>][+<component>...]
#   recon-cell   per-cell NB only
#   recon-nbr    neighbourhood NB only
#   recon-both   per-cell + neighbourhood NB
#   +adj         + adjacency reconstruction
#   +film        + FiLM conditioning
#   +mask        + MAE-style masking
#   +mhvq        + multi-head VQ
#   full         every component active
# ---------------------------------------------------------------------------

def _B():
    """Shorthand: deep-copy of the base POC config."""
    return _copy(make_train_config_poc())


# ---------------------------------------------------------------------------
# 2c. VQNiche_Dual: dual-codebook config builder
# ---------------------------------------------------------------------------
# The dual model (`VQNiche_Dual`) has a separate cell and niche codebook + a
# separate decoder for each branch. The config below is built by starting
# from the single-codebook POC config and then:
#   - replacing model_name/encoder_name with the dual classes
#   - replacing `vq_params` with `vq_cell_params` + `vq_niche_params`
#   - replacing `attribute_decoder_params` with `attribute_decoder_cell_params`
#     + `attribute_decoder_niche_params`
#   - replacing `loss_names` with the dual loss list
#   - dropping `adjacency_decoder_params` (the dual model uses a direct
#     cosine-similarity adjacency BCE on z_q_niche, so there is no MLP
#     adjacency decoder)
# ---------------------------------------------------------------------------

def _default_vq_params(codebook_size: int = 30) -> dict:
    """
    Single-codebook (non-hierarchical) VectorQuantize defaults, identical
    settings as the POC config so cell- and niche-branches start from a
    matched VQ baseline. Swap in `_default_rvq_params(...)` for hierarchical.
    """
    return {
        "vq_name": "VectorQuantize",
        "freeze_codebook": False,
        "use_cosine_sim": True,
        "ema_update": True,
        "manual_ema_update": False,
        "threshold_ema_dead_code": 2,
        "manual_in_place_optimizer_update": False,
        "learnable_codebook": False,
        "codebook_size": int(codebook_size),
        "heads": 1,
        "separate_codebook_per_head": False,
        "decay": 0.8,
        "eps": 0.00001,
        "kmeans_init": True,
        "kmeans_iters": 10,
        "sync_kmeans": True,
        "sample_codebook_temp": 0.0,
        "commitment_weight": 0.0,
        "commitment_use_cross_entropy_loss": False,
        "codebook_diversity_loss_weight": 0.0,
        "codebook_diversity_temperature": 100.0,
        "orthogonal_reg_weight": 0.0,
        "orthogonal_reg_max_codes": None,
        "orthogonal_reg_active_codes_only": False,
    }


def _default_rvq_params(codebook_sizes=(30, 200)) -> dict:
    """Residual-VQ defaults (hierarchical). Plug into vq_cell_params or vq_niche_params."""
    return {
        "vq_name": "ResidualVQ_Squint",
        "num_quantizers": len(codebook_sizes),
        "codebook_size": list(codebook_sizes),
        "use_cosine_sim": True,
        "ema_update": True,
        "decay": 0.8,
        "eps": 0.00001,
        "threshold_ema_dead_code": 2,
        "kmeans_init": True,
        "kmeans_iters": 10,
        "sync_kmeans": True,
        "commitment_weight": 0.0,
        "sample_codebook_temp": 0.0,
    }


def _default_cvq_params(
        k1: int = 30,
        k2: int = 10,
        k3: Optional[int] = None,
    ) -> dict:
    """Conditional / tree VQ defaults. Plug into vq_cell_params or vq_niche_params.

    `k3=None` (default) gives the legacy 2-level CVQ (K1 macro buckets,
    K2 sub-children each — K1 × K2 distinct codes). Passing an integer
    `k3` activates 3-level CVQ: K1 × K2 × K3 distinct codes, with one
    leaf VQ of size K3 per (l1, l2) bucket. The 3-level pass is wired
    inside `ConditionalVQ.forward` and exposes `num_quantizers=3` so
    downstream codebook-utilisation metrics stratify all three levels.
    """
    out = {
        "vq_name": "ConditionalVQ",
        "codebook_size_l1": int(k1),
        "codebook_size_l2": int(k2),
        "use_cosine_sim": True,
        "ema_update": True,
        "decay": 0.8,
        "eps": 0.00001,
        "threshold_ema_dead_code": 2,
        "kmeans_init": True,
        "kmeans_iters": 10,
        "sync_kmeans": True,
        "commitment_weight": 0.0,
        "sample_codebook_temp": 0.0,
    }
    if k3 is not None:
        # ConditionalVQ.__init__ accepts `codebook_size_l3` as an
        # optional kwarg (None -> legacy 2-level). Setting it here
        # activates the 3-level pass.
        out["codebook_size_l3"] = int(k3)
    return out


def make_train_config_dualvq() -> dict:
    """
    Build the base config for VQNiche_Dual.

    Defaults (per design D1–D6):
      - encoder: shared MLP [400, 256], GraphSAGE GNN with hidden=256,
        num_layers=1 (matches POC config)
      - vq_cell  = VectorQuantize(k=30) on the post-MLP latent
      - vq_niche = VectorQuantize(k=30) on the post-GNN latent
      - cell decoder + niche decoder are independent MLPSoftmax with
        hidden_channels=[400, 400] each
      - losses (all weights = 1.0):
          nb_attribute_reconstruction_loss          (cell branch NB)
          nb_attribute_reconstruction_loss_nbr      (niche branch NB,
                                                     aggregated post-decoder)
          mse_commit_loss_cell
          mse_commit_loss_niche
          bce_cosine_adjacency_reconstruction_loss  (on z_q_niche, cosine sim,
                                                     NicheCompass-style)
      - adjacency BCE uses cosine_temperature=0.1, weight=250 (heavily
        upweighted to balance against gene-NB ~150 nats — empirical sweet
        spot on the MERFISH/STARmap MB panel)
      - cosine-similarity adjacency operates on z_gnn (continuous niche
        embedding, NicheCompass-faithful) by default; switch via
        `loss_kwargs['adj_loss_input'] = 'z_q_niche'` to use the
        quantized version
      - the BCE input tensor is the FULL batch (including sampled
        neighbours) so negative-edge sampling has enough nodes
    """
    cfg = make_train_config_poc()

    # --- model class names ------------------------------------------------
    cfg["model"]["model_name"]   = "VQNiche_Dual"
    cfg["model"]["encoder_name"] = "VQNiche_Dual_Encoder"
    cfg["model"]["adjacency_decoder_name"] = None   # not used

    # --- encoder: replace single vq_params with two slots ----------------
    enc = cfg["model"]["encoder_params"]
    enc["vq_cell_params"]  = _default_vq_params(codebook_size=30)
    enc["vq_niche_params"] = _default_vq_params(codebook_size=30)
    enc.pop("vq_params", None)   # legacy single-codebook key — removed

    # --- two attribute decoders ------------------------------------------
    cfg["model"]["attribute_decoder_cell_params"]  = _copy(cfg["model"]["attribute_decoder_params"])
    cfg["model"]["attribute_decoder_niche_params"] = _copy(cfg["model"]["attribute_decoder_params"])
    cfg["model"].pop("attribute_decoder_params", None)
    cfg["model"].pop("adjacency_decoder_params",  None)

    # --- losses & weights -------------------------------------------------
    # Niche NB uses the *_dual variant so it reads `dispersion_niche` instead
    # of `dispersion` — i.e. the niche branch has its own per-gene NB
    # dispersion, decoupled from the cell branch's dispersion. This is
    # essential because the per-cell counts (cell branch) and the
    # neighbourhood-mean counts (niche branch) have very different variance
    # structure; sharing the dispersion forces a compromise that makes the
    # cell-branch NB loss drift up over training.
    cfg["model"]["loss_params"]["loss_names"] = [
        "nb_attribute_reconstruction_loss",            # cell branch
        "nb_attribute_reconstruction_loss_nbr_dual",   # niche branch (separate theta)
        "mse_commit_loss_cell",
        "mse_commit_loss_niche",
        "bce_cosine_adjacency_reconstruction_loss",    # on z_q_niche
    ]
    lk = cfg["model"]["loss_params"]["loss_kwargs"]
    # Drop legacy single-codebook flags that don't apply.
    for k in ("recon_mode", "k_hop_nb_loss", "bypass_adj_decoder",
              "estimate_adj_kwargs"):
        lk.pop(k, None)
    lk.update({
        # Two primary reconstruction objectives at parity. Both are
        # NB log-likelihood summed over 431 genes -> magnitudes ~150 nats.
        "wt_attr_reconstr":     1.0,    # NB cell
        "wt_attr_reconstr_nbr": 1.0,    # NB nbr
        # VQ-VAE commit losses are SOFT regularisers, not reconstruction
        # objectives. Original VQ-VAE paper (van den Oord 2017) uses
        # beta=0.25 and notes "results are robust to beta". We default to
        # 1.0 (slightly tighter than the paper's 0.25) but importantly NOT
        # match-magnitude with NB — that would over-constrain the encoder.
        # See response notes: equal-magnitude weighting is NOT recommended
        # in VQ-VAE literature; commit terms are intentionally smaller.
        "wt_commit_cell":       1.0,    # commit_cell
        "wt_commit_niche":      1.0,    # commit_niche
        # Adjacency BCE has small absolute magnitude (~0.5-1 nat at init,
        # bounded by log(2)) — it must be UPWEIGHTED to balance against
        # the gene-NB losses (~150 nats). Empirically `wt_adj_reconstr=1000`
        # gives the strongest spatial-niche coherence on the MERFISH/STARmap
        # MB panel without measurably hurting NB Pearson once the wide
        # codebook setup is used (k=50 macro + RVQ refinement). The
        # earlier "Pearson drop after epoch 1" symptom turned out to be
        # the codebook-size bottleneck (NB→niche-mean predictor pathology),
        # NOT the adj weight — once that's fixed via wide+RVQ, the model
        # tolerates much higher adjacency pressure.
        "wt_adj_reconstr":      1000.0,  # cosine adj BCE (1.0 -> 100 -> 25 -> 250 -> 1000)
        "edge_sampling_ratio":  2.0,
        "use_pos_weight":       True,
        "cosine_temperature":   0.1,
        # Which embedding the adjacency BCE operates on:
        #   "z_gnn"     -> continuous, NicheCompass-faithful (default)
        #   "z_q_niche" -> quantized (opt-in via _patch_dual_adj_on_zqniche)
        # See `bce_cosine_adjacency_reconstruction_loss` for the rationale.
        "adj_loss_input":       "z_gnn",
        # K-hop aggregation radius for the niche reconstruction target.
        #   1 -> mean over the cell's 1-hop spatial neighbourhood (default,
        #        legacy behaviour).
        #   2 -> mean over the 2-hop neighbourhood. Smoother targets ->
        #        the niche codebook is forced to capture larger-scale
        #        spatial structure rather than 1-hop variation. Requires
        #        the datamodule sampler to sample at least 2-hop neighbours
        #        (`num_neighbors=[8, 8]` or deeper) to be exact.
        "nbr_aggregation_hops": 1,
    })

    # The dual model doesn't yet support recon_mode-style modes (it always
    # reconstructs both branches), but downstream code reads `recon_mode`
    # from loss_kwargs in some places — set it to a no-op marker.
    lk["recon_mode"]  = "dual"

    # Monitor metric: training-loss based (as set up by the early-stopping
    # patch). Pearson-based monitors still work; default to log1p-cell since
    # the dual model produces a meaningful per-cell X_hat.
    cfg["trainer"]["monitor"] = "val_loss"

    return cfg


def _BD():
    """Shorthand: deep-copy of the dual-VQ base config."""
    return _copy(make_train_config_dualvq())


# ---------------------------------------------------------------------------
# 2d. VQNiche_Dual ablation patches
# ---------------------------------------------------------------------------

def _patch_dual_no_adj(cfg: dict) -> dict:
    """Drop the adjacency BCE loss term (test if it's load-bearing)."""
    losses = cfg["model"]["loss_params"]["loss_names"]
    if "bce_cosine_adjacency_reconstruction_loss" in losses:
        losses.remove("bce_cosine_adjacency_reconstruction_loss")
    return cfg


def _patch_dual_no_cell_recon(cfg: dict) -> dict:
    """
    Drop the cell-branch NB loss + cell commit loss (test the niche-only
    setup; the cell decoder still runs but receives no gradient signal).
    Useful as a ceiling test for whether the cell branch helps the niche
    branch via the shared MLP trunk.
    """
    losses = cfg["model"]["loss_params"]["loss_names"]
    for ln in ("nb_attribute_reconstruction_loss", "mse_commit_loss_cell"):
        if ln in losses:
            losses.remove(ln)
    return cfg


def _patch_dual_no_spatial(cfg: dict) -> dict:
    """
    Drop EVERY spatial-supervision loss term so the model trains as a
    pure cell-branch VQ-VAE + (whatever batch-integration loss is set).
    Removes:
      - nb_attribute_reconstruction_loss_nbr_dual  (niche-branch NB recon)
      - mse_commit_loss_niche                       (niche VQ commit)
      - bce_cosine_adjacency_reconstruction_loss    (adjacency BCE)

    The niche-side modules (GNN, vq_niche, niche decoder, adjacency
    decoder) are still instantiated by the model and run forward, but
    receive zero gradient signal — they're effectively idle. This is
    the DIAGNOSTIC variant for the hypothesis "spatial losses are
    pulling the cell encoder away from cell-type-discriminative features".
    Under this patch, training conditions match scVI / Harmony shape:
    pure cell-NB + cell-commit + adversarial/MMD batch integration.

    Sibling of `_patch_dual_no_adj` (drops only adjacency) and
    `_patch_dual_no_cell_recon` (drops only cell-side). When all three
    are needed together for ablations, apply this LAST so the niche
    losses get removed regardless of order.
    """
    losses = cfg["model"]["loss_params"]["loss_names"]
    for ln in (
        "nb_attribute_reconstruction_loss_nbr_dual",
        "nb_attribute_reconstruction_loss_nbr",   # legacy single-codebook name
        "mse_commit_loss_niche",
        "bce_cosine_adjacency_reconstruction_loss",
        # The vanilla (non-dual) adjacency-BCE name, in case earlier patches
        # left it in the loss list.
        "bce_adjacency_reconstruction_loss",
    ):
        if ln in losses:
            losses.remove(ln)
    return cfg


def _patch_dual_rvq(cfg: dict,
                    branch: str = "both",
                    codebook_sizes=(30, 200)) -> dict:
    """
    Swap one or both VQ slots for ResidualVQ_Squint. `branch` may be
    'cell', 'niche', or 'both'.
    """
    if branch in ("cell", "both"):
        cfg["model"]["encoder_params"]["vq_cell_params"]  = _default_rvq_params(codebook_sizes)
    if branch in ("niche", "both"):
        cfg["model"]["encoder_params"]["vq_niche_params"] = _default_rvq_params(codebook_sizes)
    return cfg


def _patch_dual_vq(cfg: dict,
                   branch: str = "cell",
                   codebook_size: int = 30) -> dict:
    """
    Swap one or both VQ slots for a single-level (non-hierarchical)
    `VectorQuantize` with the specified codebook_size. Mirrors
    `_patch_dual_rvq` but for the single-level case — useful for v7
    ablations that hold one branch at a non-hierarchical VQ while the
    other uses RVQ. `branch` may be 'cell', 'niche', or 'both'.
    """
    if branch in ("cell", "both"):
        cfg["model"]["encoder_params"]["vq_cell_params"]  = _default_vq_params(codebook_size=codebook_size)
    if branch in ("niche", "both"):
        cfg["model"]["encoder_params"]["vq_niche_params"] = _default_vq_params(codebook_size=codebook_size)
    return cfg


def _patch_dual_cvq(cfg: dict,
                    branch: str = "both",
                    k1: int = 30,
                    k2: int = 10,
                    k3: Optional[int] = None) -> dict:
    """
    Swap one or both VQ slots for ConditionalVQ (tree-structured VQ).

    `k3=None` -> legacy 2-level CVQ (K1 macro × K2 sub-children).
    `k3=<int>` -> 3-level CVQ (K1 × K2 × K3 distinct codes), with one
    leaf VQ of size K3 per (l1, l2) bucket. The leaf forward pass and
    extra residual STE are wired inside `ConditionalVQ.forward`.
    """
    if branch in ("cell", "both"):
        cfg["model"]["encoder_params"]["vq_cell_params"]  = _default_cvq_params(k1, k2, k3)
    if branch in ("niche", "both"):
        cfg["model"]["encoder_params"]["vq_niche_params"] = _default_cvq_params(k1, k2, k3)
    return cfg


def _patch_dual_gatv2(cfg: dict) -> dict:
    """Switch the GNN encoder from SAGE to GATv2 (still 1 layer)."""
    cfg["model"]["encoder_params"]["gnn_name"] = "GATv2Conv"
    return cfg


def _patch_dual_adj_weight(cfg: dict, weight: float) -> dict:
    """Override the adjacency BCE weight (default in the dual base is 250)."""
    cfg["model"]["loss_params"]["loss_kwargs"]["wt_adj_reconstr"] = float(weight)
    return cfg


def _patch_dual_wide(
        cfg: dict,
        cell_codebook_size:  int = 50,
        niche_codebook_size: int = 50,
        gnn_layers:          int = 2,
        sampler_neighbors:   List[int] = [8, 8],
        nbr_hops:            int = 2,
    ) -> dict:
    """
    "Wide" variant of the dual model: bigger codebooks + deeper GNN +
    deeper sampler + multi-hop nbr aggregation. These knobs are coupled —
    if you bump the GNN depth without bumping the sampler depth, the GNN
    will see partial neighbourhoods and learn nonsense; if you bump the
    nbr-target aggregation without the sampler, the K-th-hop aggregate is
    approximate (computed on incomplete neighbour info). This patch sets
    them all together.

    Defaults (calibrated for ~100k cells / 50 niches):
      - cell VQ k=50          (was 30)
      - niche VQ k=50         (was 30)
      - GNN num_layers=2      (was 1) — niche encoder sees 2-hop context
      - sampler [8, 8]        (was [8]) — needed for the deeper GNN
      - nbr_aggregation_hops=2 — niche target is the 2-hop neighbourhood
                                 mean, encouraging codes to capture
                                 larger-scale spatial structure
    """
    enc = cfg["model"]["encoder_params"]
    enc["vq_cell_params"]["codebook_size"]  = int(cell_codebook_size)
    enc["vq_niche_params"]["codebook_size"] = int(niche_codebook_size)
    enc["gnn_params"]["num_layers"]         = int(gnn_layers)

    cfg["datamodule"]["sampler_params"]["num_neighbors"] = list(sampler_neighbors)

    cfg["model"]["loss_params"]["loss_kwargs"]["nbr_aggregation_hops"] = int(nbr_hops)
    return cfg


# ---------------------------------------------------------------------------
# Architecture ablations (mouse data)
# ---------------------------------------------------------------------------

def _patch_dual_small(cfg: dict, hidden: int = 128) -> dict:
    """
    scvi-style compact architecture: a single hidden layer of `hidden`
    dim (default 128) for the encoder MLP, the GNN, and both decoders.

    Configures:
      - encoder MLP:   hidden_channels = [hidden]   (was [400, 256])
      - GNN:           hidden_channels = hidden, num_layers = 1
      - cell decoder:  hidden_channels = [hidden]   (was [400, 400])
      - niche decoder: hidden_channels = [hidden]   (was [400, 400])

    Latent / codebook embedding dim becomes `hidden` (cell_dim and
    niche_dim both = `hidden`), so the codebook itself is also more
    compact. Codebook size is left untouched (still 30 by default; bump
    via `_patch_dual_rvq` etc. on top of this).

    Tests whether the smaller-is-better intuition that drives scvi
    transfers to spatial VQ-VAE. Parameter count drops ~5x vs the
    SQUINT default.
    """
    enc = cfg["model"]["encoder_params"]
    enc["mlp_params"]["hidden_channels"]  = [int(hidden)]
    enc["gnn_params"]["hidden_channels"]  = int(hidden)
    enc["gnn_params"]["num_layers"]       = 1
    for dec_key in ("attribute_decoder_cell_params",
                    "attribute_decoder_niche_params"):
        cfg["model"][dec_key]["mlp_params"]["hidden_channels"] = [int(hidden)]
    return cfg


def _patch_dual_gnn2(
        cfg: dict,
        sampler_neighbors: List[int] = [8, 8],
        nbr_aggregation_hops: int = 2,
    ) -> dict:
    """
    Bump GNN depth from 1 to 2 layers WITHOUT changing the codebook
    size (`_patch_dual_wide` couples both). Sampler depth and niche
    target aggregation hops move with it (otherwise the GNN samples
    nodes it won't aggregate over, or the nbr-target aggregation runs
    on incomplete neighbours — see `_patch_dual_wide` docstring).

    Use this as a clean isolation of "does deeper spatial context
    help?" — codebook stays at 30 (or whatever was set upstream),
    everything else as default.
    """
    cfg["model"]["encoder_params"]["gnn_params"]["num_layers"] = 2
    cfg["datamodule"]["sampler_params"]["num_neighbors"] = list(sampler_neighbors)
    cfg["model"]["loss_params"]["loss_kwargs"]["nbr_aggregation_hops"] = int(
        nbr_aggregation_hops
    )
    return cfg


def _patch_dual_small_latent(cfg: dict, latent_dim: int = 128) -> dict:
    """
    Shrink the latent / codebook-embedding dimension. Both `cell_dim`
    (= last layer of encoder MLP) and `niche_dim` (= GNN hidden) move
    to `latent_dim` (default 128, down from 256).

    Tighter bottleneck → fewer dimensions to spread information across,
    so each code's embedding has to be more semantically concentrated.
    NicheCompass / scvi typically use 10–32; 128 still leaves headroom
    while halving the SQUINT default.

    Decoder hidden widths and codebook size are unchanged.
    """
    enc = cfg["model"]["encoder_params"]
    h = list(enc["mlp_params"].get("hidden_channels", []))
    if len(h) == 0:
        # No MLP trunk — just set the GNN dim. The cell branch then
        # quantises raw input directly which is unusual; warn but
        # respect.
        print("WARN: _patch_dual_small_latent: no MLP trunk configured "
              "(mlp_params.hidden_channels is empty). Setting GNN hidden "
              "only — cell branch latent dim will equal in_channels.")
    else:
        h[-1] = int(latent_dim)
        enc["mlp_params"]["hidden_channels"] = h
    enc["gnn_params"]["hidden_channels"] = int(latent_dim)
    return cfg


def _patch_dual_dropout(cfg: dict, p: float = 0.1) -> dict:
    """
    Add dropout `p` (default 0.1) to encoder MLP, GNN, and both
    decoders. Cheap regularization — useful when val Pearson plateaus
    while train Pearson keeps climbing (a sign of overfitting). With
    val_loss-based early stopping, dropout's effect is monitorable
    directly in wandb.
    """
    enc = cfg["model"]["encoder_params"]
    enc["mlp_params"]["dropout"] = float(p)
    enc["gnn_params"]["dropout"] = float(p)
    for dec_key in ("attribute_decoder_cell_params",
                    "attribute_decoder_niche_params"):
        cfg["model"][dec_key]["mlp_params"]["dropout"] = float(p)
    return cfg


def _patch_dual_mlp_width(
        cfg: dict,
        encoder_hidden: List[int],
        decoder_hidden: List[int],
    ) -> dict:
    """
    Override the MLP hidden widths for BOTH the encoder trunk and the
    cell + niche attribute decoders. Useful for capacity ablations
    that want a coordinated change (encoder + decoder grow / shrink
    together) rather than a separate sweep on each side.

    The LAST element of `encoder_hidden` is the encoder MLP's output
    dim — i.e. the latent / codebook embedding dim. Pick it equal to
    the existing latent (typically 256) to isolate "hidden width"
    from "latent dim"; pick something else to ablate latent dim
    along with width.

    Decoders take only `decoder_hidden` (their output dim is fixed at
    n_genes by the model wiring).
    """
    enc = cfg["model"]["encoder_params"]
    enc["mlp_params"]["hidden_channels"] = list(encoder_hidden)
    for dec_key in ("attribute_decoder_cell_params",
                    "attribute_decoder_niche_params"):
        if dec_key in cfg["model"]:
            cfg["model"][dec_key]["mlp_params"]["hidden_channels"] = list(decoder_hidden)
    return cfg


def _patch_dual_encoder_deeper(
        cfg: dict,
        hidden_channels: Optional[List[int]] = None,
    ) -> dict:
    """
    Deepen the encoder MLP trunk: from `[400, 256]` (2 hidden layers)
    to `[400, 400, 256]` (3 hidden layers) by default. Latent dim
    (= last layer = 256) is preserved so codebook + GNN dims don't
    shift; only the depth grows. The extra layer adds ~160k params
    and a bit of FLOPs but lets the encoder learn richer non-linear
    feature mixes before quantisation.
    """
    enc = cfg["model"]["encoder_params"]
    target = list(hidden_channels) if hidden_channels else [400, 400, 256]
    enc["mlp_params"]["hidden_channels"] = target
    return cfg


def _patch_dual_gnn_hidden(
        cfg: dict,
        hidden: int = 384,
    ) -> dict:
    """
    Bump the niche-branch GNN hidden width (default 256 -> 384).
    Note: this changes the GNN's *internal* dim AND the niche latent
    dim (= GNN output) — codebook embedding dim follows. Larger spatial
    capacity at modest cost (~1.5x params on the GNN).
    """
    cfg["model"]["encoder_params"]["gnn_params"]["hidden_channels"] = int(hidden)
    return cfg


def _patch_dual_sampler_neighbors(
        cfg: dict,
        num_neighbors: Optional[List[int]] = None,
    ) -> dict:
    """
    Override the `NeighborLoader.num_neighbors` per layer (default
    `[8]` for 1-hop). Increasing to `[16]` doubles the niche
    aggregation neighborhood, often improving spatial-domain coherence
    at the cost of larger per-step memory.
    """
    target = list(num_neighbors) if num_neighbors else [16]
    cfg["datamodule"]["sampler_params"]["num_neighbors"] = target
    return cfg


def _patch_dual_adversarial(
        cfg: dict,
        alpha: float = 1.0,
        wt_adv_batch: float = 150.0,
        hidden_channels: Optional[List[int]] = None,
        warmup_epochs: int = 0,
    ) -> dict:
    """
    Enable a domain-adversarial batch-invariance head on `z_mlp`.

    What it does: a small MLP classifier predicts the per-cell batch
    label from `z_mlp`. A Gradient Reversal Layer (GRL) is placed between
    the encoder and the classifier, so:
      - The classifier learns to predict batch (gets the true gradient).
      - The encoder is pushed AWAY from being batch-predictive (gets the
        negated gradient).

    The model classifies the FULL `z_mlp` (seeds + sampled GNN neighbours),
    not only the seed-node prefix — see VQNiche_Dual._step for the rationale.
    This makes the niche branch's GNN aggregate over batch-invariant inputs.

    Effect: provides the missing batch-invariance pressure on the latent
    that NicheCompass gets from its KL prior. Pair with
    `_patch_dual_decoder_covariate` for the full NicheCompass-shaped
    setup (decoder absorbs per-batch gene patterns, encoder is pushed
    to be batch-invariant).

    Parameters
    ----------
    alpha: float
        Strength of the gradient reversal (encoder pressure). Default 1.0
        — the standard Ganin-Lempitsky setting; multiplies the negated
        gradient that flows back to the encoder. Earlier we tried alpha
        =2.0 (with wt_adv_batch=100) and the codes started to collapse
        (encoder pressure too aggressive). Reverted to alpha=1.0 and
        bumped `wt_adv_batch` instead — the encoder gradient magnitude
        is `alpha * wt_adv_batch * d_CE/d_z`, so the two knobs are
        interchangeable for encoder pressure; the difference is that
        `wt_adv_batch` also accelerates the classifier (sharper
        adversary signal at the cost of faster CE→0 saturation).
    wt_adv_batch: float
        Weight of the CE loss in the total loss. Default 150.0. With 2
        batches the binary CE is bounded by log(2) ≈ 0.69 nats while
        the cell + niche NB losses sit at ~150 nats each; at
        `wt_adv_batch=1.0` the adversarial gradient is ~150x weaker
        than reconstruction and the encoder ignores it. 150.0 brings
        the encoder pressure to ~1.5x the "almost good" alpha=1, wt=100
        baseline without the codebook collapse seen at alpha=2. Sweep
        log-ish (100, 150, 200, 300) — the right value depends on
        dataset, codebook size, and how aggressive the decoder's batch-
        absorption pathway (concat / FiLM) already is.
    hidden_channels: list of int, optional
        Hidden widths for the classifier MLP. Default [128] (one
        hidden layer of 128 units).
    """
    cfg["model"]["adversarial_batch_dim_request"] = True
    cfg["model"]["adversarial_alpha"] = float(alpha)
    if hidden_channels is not None:
        cfg["model"]["adversarial_hidden_channels"] = list(hidden_channels)
    # `warmup_epochs` is wired through to VQNiche_Dual.__init__ and
    # consumed inside `_step`: during the first `warmup_epochs`
    # training epochs, the GRL is run with alpha=0 so the encoder
    # gets zero adversarial gradient (codes settle on biology first).
    # The classifier still trains and the CE term still appears in
    # total_loss for logging. Default 0 = legacy behaviour.
    cfg["model"]["adversarial_warmup_epochs"] = int(warmup_epochs)

    losses = cfg["model"]["loss_params"]["loss_names"]
    if "adversarial_batch_loss" not in losses:
        losses.append("adversarial_batch_loss")
    cfg["model"]["loss_params"]["loss_kwargs"]["wt_adv_batch"] = float(wt_adv_batch)
    return cfg


def _patch_dual_adversarial_cosine(
        cfg: dict,
        total_epochs: int = 100,
    ) -> dict:
    """
    Switch the adversarial alpha to a cosine schedule (vs the legacy
    constant). After warmup, alpha follows a half-sine envelope: rises
    from 0 at `warmup_epochs`, peaks at the midpoint of the post-warmup
    phase, decays back to ~0 at `total_epochs`. Past `total_epochs` the
    alpha stays at 0.

    Why use this: the constant schedule keeps adversarial pressure on
    the encoder until the very last epoch. Combined with `adv-warmup`,
    the cell token doesn't get a "consolidation phase" to refine
    cell-type-discriminative features without the gradient reversal
    pulling them apart. The cosine decay creates that consolidation
    phase in the final ~25% of training, while still giving strong
    integration pressure mid-training.

    Pair with `_patch_dual_adversarial(warmup_epochs=10, ...)` and an
    `adv-warmup10` variant tag for the intended use case.

    NOTE: requires `_patch_dual_adversarial` to have been called first
    on the cfg (this patch only flips the schedule flag — the
    adversarial head itself is built by the prior patch).
    """
    if "adversarial_batch_dim_request" not in cfg["model"]:
        raise RuntimeError(
            "_patch_dual_adversarial_cosine: call "
            "_patch_dual_adversarial() first to enable the adversarial "
            "head, then chain this patch on top."
        )
    cfg["model"]["adversarial_schedule"] = "cosine"
    cfg["model"]["adversarial_total_epochs"] = int(total_epochs)
    return cfg


def _patch_dual_adversarial_cell_only(cfg: dict) -> dict:
    """
    Restrict the adversarial classifier to the SEED prefix of z_mlp
    (i.e. the per-cell embedding before VQ), instead of classifying
    the full tensor (seeds + sampled neighbours).

    Why: the niche-branch GNN aggregates over neighbour z_mlp rows to
    produce z_gnn. With the legacy `apply_to='full'` setting, the
    adversarial gradient reaches BOTH the cell pathway (z_mlp[:B] -> VQ
    cell) and the niche pathway (z_mlp[B:] -> GNN -> VQ niche). Both
    branches lose batch-correlated info. With `apply_to='cell'`, only
    the seed rows feed the adversary's CE — the niche branch is left
    un-pressured by the adversary, keeping its spatial-aggregation
    capacity intact.

    Hypothesis: cell-type identification benefits from concentrating
    the adversarial "budget" on the cell branch where it matters most;
    the niche branch's batch-mixing is achieved instead by the
    decoder covariate + adjacency reconstruction (which already
    absorb batch-specific patterns elsewhere).

    NOTE: requires `_patch_dual_adversarial` to have been called first.
    """
    if "adversarial_batch_dim_request" not in cfg["model"]:
        raise RuntimeError(
            "_patch_dual_adversarial_cell_only: call "
            "_patch_dual_adversarial() first to enable the adversarial "
            "head, then chain this patch on top."
        )
    cfg["model"]["adversarial_apply_to"] = "cell"
    return cfg


def _patch_dual_mmd_batch(
        cfg: dict,
        wt_mmd_batch: float = 50.0,
        n_sub: int = 512,
    ) -> dict:
    """
    Add a non-adversarial MMD batch-invariance loss on the cell-token
    input (z_mlp[:batch_size]).

    Computes a kernel-based MMD between per-batch distributions on
    each mini-batch and adds it directly to the total loss. Unlike
    the adversarial CE — which involves a min-max game between the
    classifier and the encoder, and can fail to remove
    batch-correlated features that were learned during a warmup phase —
    MMD applies a deterministic distribution-matching pressure on
    every step.

    Recipe (mirrors `mmd_comparable` in compute_inference_metrics.py
    so train-time pressure and test-time evaluation use the same
    kernel):
      - z-score per-dim normalisation
      - RBF kernel with sigma chosen by median pairwise distance
      - Pairwise MMD^2 across all batches present, summed

    Use cases:
      1. Pair with `_patch_dual_adversarial(warmup_epochs=10)` to give
         a non-adversarial integration safety-net during the warmup
         period (during which the GRL alpha is 0).
      2. Replace the adversarial CE entirely (drop
         `_patch_dual_adversarial`) to test whether MMD alone is
         sufficient for batch alignment — simpler architecture, no
         classifier to tune.

    Parameters
    ----------
    wt_mmd_batch
        Loss weight for the MMD term. Default 50.0. The unweighted
        MMD^2 is in [0, 2] for a normalised kernel; values around
        50-200 put it on a comparable scale to the adversarial CE
        (which sits at log(2) * wt_adv_batch ≈ 100 for wt=150).
    n_sub
        Sub-sample size per batch. Default 512 — keeps the pairwise
        kernel matrix at ~1 MB.
    """
    losses = cfg["model"]["loss_params"]["loss_names"]
    if "mmd_batch_loss" not in losses:
        losses.append("mmd_batch_loss")
    cfg["model"]["loss_params"]["loss_kwargs"]["wt_mmd_batch"] = float(wt_mmd_batch)
    cfg["model"]["loss_params"]["loss_kwargs"]["mmd_n_sub"] = int(n_sub)
    # The model's `_step` populates `mmd_target` and `mmd_target_labels`
    # from `z_mlp[:batch_size]` and `adata_batch_ids[:batch_size]`
    # unconditionally; the dispatcher only consumes them when
    # `mmd_batch_loss` is in `loss_names`.
    return cfg


def _patch_dual_encoder_shallow(
        cfg: dict,
        hidden_channels: Optional[List[int]] = None,
    ) -> dict:
    """
    Shrink the encoder MLP trunk: from `[400, 256]` (2 hidden layers,
    the default) to `[256]` (1 hidden layer) by default. Latent dim
    (= last layer output = 256) is preserved so codebook + GNN dims
    don't shift; only the depth shrinks.

    Hypothesis: a smaller-capacity encoder has less room to memorise
    batch-specific features, which (combined with a brief warmup
    phase) may push the cell-type discriminative signal up while
    keeping the integration cost manageable. Sibling of
    `_patch_dual_encoder_deeper`; both target the same MLP trunk
    inside the encoder.

    NOTE: this is the trunk only — the GNN depth and decoder MLP
    width are controlled separately by `_patch_dual_wide` / the
    sampler patches / `_patch_dual_small`. For an even more
    aggressive capacity bottleneck, pair with `_patch_dual_small`.
    """
    enc = cfg["model"]["encoder_params"]
    target = list(hidden_channels) if hidden_channels else [256]
    enc["mlp_params"]["hidden_channels"] = target
    return cfg


def _patch_dual_commit_cell_weight(
        cfg: dict,
        wt_commit_cell: float = 2.0,
    ) -> dict:
    """
    Increase the cell-branch commit-loss weight from the default 1.0
    (set per-loss inside `_make_default_dual_config`) to `wt_commit_cell`.

    The commit loss pulls `z_mlp[:batch_size]` (the cell-branch input
    pre-VQ) toward `z_q_cell` (the post-VQ embedding). Higher weight
    means z_mlp commits more aggressively to discrete codes — codes
    are sharper / less smeared, and within-class variance in the
    cell-token shrinks.

    Hypothesis: with the cell-token discrete codes "sharpened", cells
    of the same cell-type cluster around fewer codes, raising the
    NMI(cell-token, cell_type). Trade-off: higher cell-commit can
    starve the niche-branch GNN, which consumes the SAME z_mlp tensor
    (sampled neighbours) — if z_mlp is dominated by sharp cell-cluster
    geometry, the GNN's spatial aggregation may degrade. Pair with
    a weaker niche-branch lever (e.g. lower wt_commit_niche) if you
    see niche-NMI regress.
    """
    cfg["model"]["loss_params"]["loss_kwargs"]["wt_commit_cell"] = float(wt_commit_cell)
    return cfg


def _patch_dual_cross_entropy(
        cfg: dict,
        wt_cross_entropy: float = 10.0,
    ) -> dict:
    """
    Turn ON the supervised cell-type cross-entropy loss on the
    Linear predictor head (z_mlp[:batch_size] -> n_cell_types) with
    weight `wt_cross_entropy`.

    NOTE on methodological framing: this makes the variant
    SEMI-SUPERVISED — it consumes ground-truth `cell_type` labels
    at training time. The other SQUINT variants are unsupervised
    (reconstruction + commit + adjacency + adversarial only). Reporting
    a semi-supervised variant alongside the unsupervised ones is fair
    iff clearly labeled in the methods — it gives an empirical upper
    bound for "how much can cell-NMI improve if we directly optimise
    for it?", which scaffolds the unsupervised story.

    Mechanism: the Linear predictor is already built by `BaseModel`
    (256 -> n_cell_types) but its output is unused unless
    `cross_entropy_loss` is in `loss_names`. This patch flips it on
    and sets the weight. The loss data dispatcher in
    `BaseModel.criterion` already reads `logits` and `labels` from
    loss_data, both of which `VQNiche_Dual._step` populates from
    `predictor(z_mlp[:batch_size])` and `batch.y` respectively.

    Weight calibration: typical NB attribute-reconstruction loss is
    ~150 nats, CE on a 49-way classifier is bounded by log(49) ≈ 3.9.
    A weight of 10 puts the CE gradient at ~10 × 3.9 / 150 ≈ 0.26x
    the NB gradient — comparable to wt_adv_batch's effective scale —
    so the predictor's signal is significant but doesn't dominate.
    Lower (5) for more conservative; higher (50, 100) if cell-NMI
    isn't the bottleneck.
    """
    losses = cfg["model"]["loss_params"]["loss_names"]
    if "cross_entropy_loss" not in losses:
        losses.append("cross_entropy_loss")
    cfg["model"]["loss_params"]["loss_kwargs"]["wt_cross_entropy"] = float(wt_cross_entropy)
    return cfg


def _patch_dual_decoder_film(
        cfg: dict,
        condition_list: List[str] = ["cell_batch_id"],
        init_mode: str = "small_random",
        small_random_std: float = 0.1,
    ) -> dict:
    """
    Enable FiLM batch-correction INSIDE the dual-model attribute decoders
    (cell + niche). Mutually exclusive with `_patch_dual_decoder_covariate`
    (which concatenates the one-hot onto z_q instead).

    What it does: sets `apply_conditioning='in-MLP'` on both decoder
    parameter dicts and configures their FiLM `condition_list` (default
    ['cell_batch_id'], i.e. per-cell batch one-hot). The data loader's
    `attr_decoder_conditions` tensor is then threaded through to each
    decoder by VQNiche_Dual.forward, and FiLM modulates each decoder
    layer with γ/β derived from the batch one-hot.

    Why init_mode defaults to 'small_random', not 'identity'
    --------------------------------------------------------
    Empirically, with `init_mode='identity'` the niche branch in this
    setup did NOT integrate well across platforms while `+decoder-cov`
    (concat) DID. The reason is the gradient signal at init:
      - 'identity' starts with γ=1, β=0 *constants* regardless of the
        batch one-hot. The decoder defaults to "ignore the batch info"
        and the path of least resistance is to encode batch in z_q
        instead of via FiLM. Gradient pressure to deviate from identity
        only kicks in if the codes ALREADY don't carry batch — which
        they do, because of the per-batch spatial graph.
      - 'small_random' starts with weights ~ N(0, std²) and identity
        bias, so γ ≈ 1 ± std and β ≈ 0 ± std *per channel per batch*
        from epoch 0. Different batches see different (small) γ/β
        perturbations → there is condition-dependent gradient signal
        from the very first step → the decoder readily learns to
        absorb per-batch shifts via FiLM, freeing the codes (same
        mechanism that makes concat work).
    Concat works for the same reason at the architectural level: the
    decoder's first linear layer initialises W_b with kaiming_uniform,
    so the batch one-hot already injects condition-dependent noise that
    the decoder is incentivised to replace with a meaningful per-batch
    bias. 'small_random' is the FiLM-side analog of that.

    `small_random_std=0.1` makes the per-channel perturbation magnitude
    on the order of the activation scale at init (10%) — comparable to
    a Kaiming-init linear layer's contribution from a one-hot input.
    """
    film_params = {
        "condition_list":    list(condition_list),
        "use_bias":          True,
        "use_residual":      False,
        "residual_weight":   0.2,
        "init_mode":         init_mode,
        "small_random_std":  float(small_random_std),
    }
    for dec_key in ("attribute_decoder_cell_params",
                    "attribute_decoder_niche_params"):
        cfg["model"][dec_key]["apply_conditioning"] = "in-MLP"
        cfg["model"][dec_key]["conditioning_params"] = film_params.copy()
    return cfg


def _patch_dual_decoder_covariate(
        cfg: dict,
        embed_dim: int = 16,
    ) -> dict:
    """
    Enable NicheCompass-style decoder covariate via a LEARNED batch
    embedding (concat-based batch correction).

    What it does: at runtime, `train()` sets `decoder_covariate_dim` to
    the number of distinct samples in the dataset (= n train batches).
    The dual model then learns an `nn.Embedding(n_batches, embed_dim)`
    and concatenates `embedding[adata_batch_ids]` (an `embed_dim`-dim
    learned vector per cell) onto z_q_cell and z_q_niche before each
    decoder. The decoders are constructed with extended `in_channels`
    by `embed_dim` (NOT by `n_batches` — the old one-hot scheme).

    Why a learned embedding instead of one-hot:
      - Generalises to NOVEL batches at inference. The mean of trained
        embeddings is a meaningful "neutral" covariate for held-out
        sections; one-hot has no valid index for unseen batches and
        forces an arbitrary "treat as if it were train batch 0"
        fallback. The dual model's forward path swaps in the mean
        embedding for cells flagged via `adata_batch_ids_unseen_mask`
        (set by `build_batch_one_hot_from_obs` when the cell's label
        wasn't in the train-time densification map).
      - Lower-dim conditional (default embed_dim=16 vs n_batches which
        can be 8-20+).
      - Encoder still doesn't see batch info → codes (= VQ output) are
        batch-invariant by construction; only the DECODER reconstruction
        depends on the embedding lookup.

    Effect:
      - Decoders use the learned batch embedding to fit per-batch
        gene patterns (different gene-capture rates, different noise
        structure across MERFISH/STARmap/CosMx).
      - Encoder is "freed" from encoding batch info in z_q because the
        decoder can absorb per-batch differences via the embedding.
      - Codebook (shared across batches) is biased toward batch-
        invariant biological identity.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the learned batch embedding. 16 is a good
        default — large enough to capture meaningful cross-batch
        differences, small enough to keep the decoder input compact.
    """
    cfg["model"]["decoder_covariate_dim_request"] = True
    cfg["model"]["decoder_covariate_embed_dim"]   = int(embed_dim)
    return cfg


def _patch_dual_chl59_lung5(
        cfg: dict,
        train_batch_idx: Optional[List[int]] = None,
        test_batch_idx: Optional[List[int]] = None,
        batch_size: int = 128,
        edge_sampling_ratio: float = 1.0,
    ) -> dict:
    """
    Swap the dataset to /nfs/team361/sb75/DATASETS/silver/chl59-8b_1p (8
    CosMx Lung samples) and configure whole-section holdout via
    SpatialBatchSplit.

    Parameters
    ----------
    train_batch_idx, test_batch_idx:
        `adata_batch_id` values (= the integer parsed out of
        `uns['batch']`, e.g. `uns['batch']='batch6'` -> id=6), NOT
        alphabetical file positions in the silver dir. SpatialBatchSplit
        matches `data.adata_batch_id` against these lists, so they MUST
        be the actual ids.

        Translation to `cfg['dataset']['adata_batch_idx']` (which IS
        position-based, used by `initialize_databatch` to slice the
        dataset blob) is done at runtime in `train()` after the blob
        is loaded, by reading each section's actual `adata_batch_id`
        and inverting the id -> position mapping.

        Two buckets:
          - `train_batch_idx`: sections used for gradient updates.
                               OPTIONAL — when omitted (default), train()
                               picks every blob section that isn't in
                               test/val. The base config's 10% cell-level
                               split runs on these and provides the val
                               signal for early stopping +
                               best-checkpoint.
          - `test_batch_idx`:  whole sections held out ENTIRELY from
                               training (NOT loaded into the training
                               data graph). Evaluated post-hoc by
                               `predict()` + the inference metrics
                               pipeline. Default `None` = no test set.

    Memory: lowers `batch_size` to 128 (from the 256 default inherited
    from the mmb-smb config) and `edge_sampling_ratio` to 1.0 (from 2.0).
    The Lung sections are denser than the mouse-brain ones AND have a
    larger gene panel (~946 vs. 431), so per-step autograd memory is
    roughly 2x the mmb-smb cost at the same batch_size. Override via the
    kwargs if your GPU has more memory.
    """
    test_batch_idx = list(test_batch_idx) if test_batch_idx else []
    # train_batch_idx is OPTIONAL — leave the train_batches list empty
    # when not specified so train() defaults to "every blob section not
    # in test/val".
    train_batch_idx = list(train_batch_idx) if train_batch_idx else []

    cfg["dataset"]["dataset_name"]   = "chl59-8b_1p"
    cfg["dataset"]["dataset_tag"]    = "chl59-8b_1p"
    cfg["dataset"]["root_data_dir"]  = "/nfs/team361/sb75/DATASETS"
    # `adata_batch_idx` (positions) is filled in at runtime in `train()`
    # by translating `train_batches` (ids) via the loaded blob's
    # id-to-position mapping. We leave it as a placeholder here so the
    # variant config archives the user-intended ids rather than positions
    # baked under a particular id-derivation assumption.
    cfg["dataset"]["adata_batch_idx"] = []

    cfg["dataset"]["train_transform_params"] = {
        "region":         None,
        "train_batches":  list(train_batch_idx),
        "val_batches":    [],   # no whole-section val; cell-level split provides val
        # `test_batches` here is INFORMATIONAL — these sections aren't
        # in `adata_batch_idx` so the data graph never contains them and
        # `SpatialBatchSplit.forward` is never called on them at training
        # time. We stamp the ids anyway so `predict()` reads them from
        # the saved config and tags inference cells with
        # `data_split = "test"` for stratified post-hoc metrics.
        "test_batches":   test_batch_idx,
        "xy_key":         "xy_coordinates",
        # Cell-level 10% split provides the val signal during training.
        "train_val_cell_split":
            cfg["dataset"]["train_transform_params"].get("train_val_cell_split", 0.10),
        "cell_split_seed":
            cfg["dataset"]["train_transform_params"].get("cell_split_seed", 0),
    }

    # The Lung AnnDatas have `obs['batch']` populated with per-sample
    # labels; the batch-key flag tells the blob builder to read it.
    cfg["dataset"]["graph_params"]["batch_key"] = "batch"

    # Memory tuning for the larger Lung gene panel + denser sampling
    # (see docstring). Defaults are conservative for a 44 GiB GPU; bump
    # them on bigger cards.
    cfg["datamodule"]["loader_params"]["batch_size"] = int(batch_size)
    cfg["model"]["loss_params"]["loss_kwargs"]["edge_sampling_ratio"] = float(
        edge_sampling_ratio
    )

    # Validation runs on the held-out replicate(s); name the monitor
    # accordingly so checkpoints are saved on test-set Pearson.
    cfg["trainer"]["monitor"] = "val_loss"
    return cfg


def _patch_dual_mmb20(
        cfg: dict,
        train_adata_ids: Optional[List[int]] = None,
        test_adata_ids: Optional[List[int]] = None,
        batch_size: int = 64,
        edge_sampling_ratio: float = 1.0,
        gnn_layers: int = 1,
        sampler_neighbors: List[int] = [8],
        nbr_aggregation_hops: int = 1,
    ) -> dict:
    """
    Swap the dataset to /lustre/.../silver/mmb0-1b_smb1-20b_1p_shared_genes
    (1 MERFISH + 20 STARmap, all reindexed to MERFISH's gene panel by
    examples/harmonize_mmb_smb20_panels.py) and configure whole-section
    holdout via SpatialBatchSplit.

    Parameters
    ----------
    train_adata_ids, test_adata_ids:
        `adata_batch_id` values (= the integer parsed out of `uns['batch']`
        that the harmonize script stamps from the filename), NOT
        positions in the sorted blob. STARmap sections are 1..20; MERFISH
        is 82. `SpatialBatchSplit` matches `data.adata_batch_id` against
        these lists, so they MUST be the actual ids — passing positions
        would silently produce no masks for any section and trigger a
        `KeyError: 'train_mask'` at the next collate step.

        Two buckets:
          - `train_adata_ids`: sections used for gradient updates. The
                               base config's 10% cell-level split runs
                               on these and provides the val signal for
                               early stopping + best-checkpoint
                               selection.
          - `test_adata_ids`:  sections held out ENTIRELY from training
                               (NOT loaded into the training data graph
                               at all — excluded from `adata_batch_idx`).
                               Evaluated post-hoc by `predict()` + the
                               inference metrics pipeline. Default
                               `None` = no test set.

        `cfg['dataset']['adata_batch_idx']` is position-based (used by
        `initialize_databatch` to slice the dataset blob); the
        translation from ids to positions happens at runtime in
        `train()` once the blob is loaded.

    batch_size, edge_sampling_ratio:
        Memory tuning. mmb20 stays on the 431-gene MERFISH panel so
        per-step activations are similar to mmb-smb, but it keeps
        ~19 sections worth of static `x` + edge_index on GPU (~2.5 GiB
        more than mmb-smb's 2 sections) before sampling starts. The
        edge sampling ratio caps the negative-sampling memory in the
        cosine adjacency loss.

    gnn_layers, sampler_neighbors, nbr_aggregation_hops:
        Niche-branch depth knobs that move TOGETHER. The mmb20 default
        narrows back to single-hop (gnn=1, sampler=[8], nbr_hops=1)
        instead of the `+wide` 2-hop stack. The big memory win is the
        sampler depth: `[8, 8]` worst-case expands 64 seeds → 64·(1+8+64)
        = 4,672 sampled nodes; `[8]` expands to 64·(1+8) = 576 — almost
        an order of magnitude smaller everywhere downstream (input `x`,
        MLP/GNN activations, decoder activations, NB loss intermediates,
        cosine-adjacency pairs). The cost is less spatial context per
        cell (1-hop instead of 2-hop niche), but that's NicheCompass's
        default and the integration question (cross-platform retrieval)
        doesn't fundamentally need 2-hop context.

        These three are coupled — narrowing one without the others is
        wasteful: a 2-hop sampler feeding a 1-hop GNN samples neighbors
        the GNN won't use; a 2-hop nbr-target aggregation on a 1-hop
        sampler is approximate (computed on incomplete neighbour info).

    Reads `obs['batch']` (also stamped by the harmonize script) for the
    per-cell batch one-hots used by the encoder/decoder batch correction
    (FiLM, decoder covariate, adversarial head).
    """
    test_adata_ids = list(test_adata_ids) if test_adata_ids else []
    # train_adata_ids is OPTIONAL — when omitted train() defaults to
    # every blob section not in test/val.
    train_adata_ids = list(train_adata_ids) if train_adata_ids else []

    cfg["dataset"]["dataset_name"]   = "mmb0-1b_smb1-20b_1p_shared_genes"
    cfg["dataset"]["dataset_tag"]    = "mmb0-1b_smb1-20b_1p"
    cfg["dataset"]["root_data_dir"]  = "/lustre/scratch126/cellgen/lotfollahi/DATASETS"
    # `adata_batch_idx` (positions) is resolved at runtime in `train()`
    # from `train_batches` (ids) via the loaded blob's id-to-position
    # mapping. Test sections are deliberately excluded — they never
    # enter the training data graph; `predict()` builds its own dataset
    # from the silver dir and CAN run inference on them post-hoc.
    cfg["dataset"]["adata_batch_idx"] = []

    cfg["dataset"]["train_transform_params"] = {
        "region":         None,
        "train_batches":  list(train_adata_ids),
        "val_batches":    [],   # no whole-section val; cell-level split provides val
        # `test_batches` here is INFORMATIONAL — these sections aren't
        # in `adata_batch_idx` so the data graph never contains them and
        # `SpatialBatchSplit.forward` is never called on them at training
        # time. We stamp the ids anyway so `predict()` reads them from
        # the saved config and tags inference cells with
        # `data_split = "test"` for stratified post-hoc metrics.
        "test_batches":   test_adata_ids,
        "xy_key":         "xy_coordinates",
        # Cell-level 10% split provides the val signal during training.
        "train_val_cell_split":
            cfg["dataset"]["train_transform_params"].get("train_val_cell_split", 0.10),
        "cell_split_seed":
            cfg["dataset"]["train_transform_params"].get("cell_split_seed", 0),
    }

    cfg["dataset"]["graph_params"]["batch_key"] = "batch"

    # Memory tuning (see docstring). Defaults are conservative for a
    # 44 GiB GPU; bump them on bigger cards.
    cfg["datamodule"]["loader_params"]["batch_size"] = int(batch_size)
    cfg["model"]["loss_params"]["loss_kwargs"]["edge_sampling_ratio"] = float(
        edge_sampling_ratio
    )

    # Niche-branch depth (GNN / sampler / nbr-target aggregation move
    # together — see docstring). These OVERRIDE whatever `_patch_dual_wide`
    # set upstream, since this patch is applied last in the variant chain.
    cfg["model"]["encoder_params"]["gnn_params"]["num_layers"] = int(gnn_layers)
    cfg["datamodule"]["sampler_params"]["num_neighbors"] = list(sampler_neighbors)
    cfg["model"]["loss_params"]["loss_kwargs"]["nbr_aggregation_hops"] = int(
        nbr_aggregation_hops
    )

    # Validation runs on the held-out section(s); name the monitor
    # accordingly so checkpoints are saved on test-set Pearson.
    cfg["trainer"]["monitor"] = "val_loss"
    return cfg


def _patch_dual_mmb20_holdout(cfg: dict) -> dict:
    """
    Convenience holdout for the mmb20 cross-platform retrieval experiment.

    Train: STARmap+ batches 1..14 + 16..20 (19 sections used for gradient
           updates).
    Val  : (empty) — no whole-section val. The base config's cell-level
           10% in-section sample of the 19 training sections is the
           in-distribution early-stopping / best-ckpt-on-val signal.
    Test : STARmap+ batch 15 (within-platform baseline; matches the
           STARmap section held out in the smaller mmb-smb experiment)
           AND MERFISH batch 82 (cross-platform OOD). Both are TRUE
           held-outs — not loaded into the training data graph at all.
           They are evaluated post-hoc by the predict +
           compute_inference_metrics pipeline (which loads every file
           in the silver dir, runs inference, and computes Pearson /
           NMI / ARI / batch-integration metrics on every section). The
           training pipeline itself never sees them.

    These are `adata_batch_id` values (not positions); see
    `_patch_dual_mmb20` for the rationale.

    Only `test_adata_ids` is specified — `train()` defaults to "every
    blob section that isn't in test/val", so STARmap+ batches 1..14 +
    16..20 (19 sections) get used for training automatically.
    """
    return _patch_dual_mmb20(
        cfg,
        test_adata_ids=[15, 82],
    )


def _patch_holdout_regions(
        cfg: dict,
        regions: Optional[dict] = None,
    ) -> dict:
    """
    Hold out spatially-contiguous patches WITHIN sections for the
    gene-reconstruction downstream task (NicheCompass / MLGenX-style).

    Both source sections stay in the train graph (so the model still
    sees most of each tissue), but the cells inside the per-batch
    `regions` rectangle become `test_mask=True` and are masked OUT of
    the training loss. At inference, predict() runs over every cell and
    `compute_inference_metrics.py` reports stratified Pearson on the
    `data_split == "test"` subset only.

    Parameters
    ----------
    regions : dict[int, dict]
        Mapping `adata_batch_id -> region_spec`. Each region spec can mix
        absolute keys (`x_min`, `x_max`, `y_min`, `y_max`) and percentile
        keys (`x_min_pct`, `x_max_pct`, `y_min_pct`, `y_max_pct`,
        fractions of the section's xy range). Percentile keys are
        recommended for cross-platform datasets where MERFISH and
        STARmap+ live in very different coordinate frames — the same
        spec then resolves to a comparable patch in both.

        Default: a central 25% × 25% patch in batch 15 (STARmap+ MB) and
        a different central-but-shifted 25% × 25% patch in batch 82
        (MERFISH MB), so the two held-outs cover different anatomical
        zones across platforms. Override to pick specific anatomical
        landmarks (e.g. cortex, hippocampus).

    Notes
    -----
    - Independent from `test_batches` (which holds out whole sections).
    - The base mmb-smb config has `train_batches=[15, 82]` and empty
      `test_batches`, which is exactly what this patch needs — both
      batches stay in training, just with cell-level masks applied.
    - Predict() reads `test_regions` from the saved config and tags
      held-out cells with `obs["data_split"] = "test"` so post-inference
      Pearson stratifies correctly.
    """
    # Default regions: rectangular patches near the centre of each
    # section (STARmap+ batch 15 and MERFISH batch 82 live in different
    # coordinate frames, so percentile-based regions resolve cleanly in
    # both).
    if regions is None:
        regions = {
            # STARmap+ batch 15: a patch in the upper-left quadrant.
            15: {
                "x_min_pct": 0.10, "x_max_pct": 0.35,
                "y_min_pct": 0.55, "y_max_pct": 0.80,
            },
            # MERFISH batch 82: a patch in the lower-right quadrant
            # (different anatomical region than batch 15's hold-out).
            82: {
                "x_min_pct": 0.55, "x_max_pct": 0.80,
                "y_min_pct": 0.20, "y_max_pct": 0.45,
            },
        }

    cfg["dataset"]["train_transform_params"]["test_regions"] = {
        int(k): v for k, v in regions.items()
    }
    return cfg


def _patch_dual_adj_on_zqniche(cfg: dict) -> dict:
    """
    Switch the adjacency BCE input from continuous z_gnn (default) to the
    quantized z_q_niche. Lets you A/B continuous vs. quantized adjacency
    while keeping every other knob fixed.
    """
    cfg["model"]["loss_params"]["loss_kwargs"]["adj_loss_input"] = "z_q_niche"
    return cfg


def _patch_quick(
        cfg: dict,
        max_epochs: int = 1,
        batch_size: int = 32,
    ) -> dict:
    """
    Smoke-test patch: cap training at `max_epochs` (default 1), disable
    early stopping so the run definitely finishes, force a single-hop GNN
    config, and set a small batch_size. The narrow GNN + small batch
    keeps memory low so the smoke test can run on the largest dataset
    (chl59 with 6 dense Lung sections, mmb20 with 19 STARmap sections)
    without OOMing.

    The narrow GNN settings (num_layers=1, sampler=[8], nbr_aggregation_hops=1)
    are EXPLICITLY re-asserted here even though `_BD()` already defaults
    to them — that way later patches in the smoke-test build chain
    (e.g. `_patch_dual_decoder_covariate`, `_patch_dual_adversarial`)
    can't accidentally drift the smoke test into a heavier config.
    Dataset patches applied AFTER `_patch_quick` (e.g. `_patch_dual_mmb20`)
    can still override batch_size if they need to.
    """
    cfg["trainer"]["max_epochs"] = int(max_epochs)
    cfg["trainer"]["early_stopping_params"]["enabled"] = False
    # Narrow GNN — single-hop everywhere.
    cfg["model"]["encoder_params"]["gnn_params"]["num_layers"] = 1
    cfg["datamodule"]["sampler_params"]["num_neighbors"] = [8]
    cfg["model"]["loss_params"]["loss_kwargs"]["nbr_aggregation_hops"] = 1
    # Smaller batch for memory headroom.
    cfg["datamodule"]["loader_params"]["batch_size"] = int(batch_size)
    return cfg


VARIANTS: dict = {
    # ---- Smoke test (1 epoch end-to-end) ----------------------------------
    "smoke-test+mmb0-1b_smb1-1b_1p": {
        "description": (
            "1-epoch end-to-end pipeline test on the mmb-smb mouse data. "
            "Dual-VQ + NicheCompass-style decoder covariate + adversarial "
            "GRL on the FULL z_mlp (alpha=1.0, wt_adv_batch=150) — same "
            "batch-correction recipe used by the overnight sweep, capped "
            "at max_epochs=1, early stopping disabled, narrow GNN "
            "(num_layers=1, sampler=[8], nbr_hops=1), batch_size=32. Use "
            "before launching the sweep to verify train -> predict -> "
            "plots -> metrics works end-to-end on the current blob in a "
            "few minutes."
        ),
        "patches": [
            "+quick(max_epochs=1, batch_size=32, narrow GNN)",
            "+decoder_covariate",
            "+adversarial_batch(alpha=1.0, wt=150.0)",
        ],
        "build": lambda: _patch_dual_adversarial(
            _patch_dual_decoder_covariate(
                _patch_quick(_BD(), max_epochs=1, batch_size=32)
            ),
            alpha=1.0, wt_adv_batch=150.0,
        ),
    },
    "smoke-test+chl59-8b_1p": {
        "description": (
            "1-epoch end-to-end pipeline test on the chl59-8b_1p CosMx Lung "
            "data (test ids = [2, 3], train = the other 6 sections, val = "
            "10% in-section cell sample). Dual-VQ + decoder covariate + "
            "adversarial GRL — same batch-correction recipe as the "
            "overnight sweep, capped at 1 epoch, early stopping disabled, "
            "narrow GNN (num_layers=1, sampler=[8], nbr_hops=1), "
            "batch_size lowered to 16 (the Lung sections have a ~2x "
            "larger gene panel than mmb-smb, so per-step autograd memory "
            "needs an extra cushion)."
        ),
        "patches": [
            "+quick(max_epochs=1, batch_size=16, narrow GNN)",
            "+decoder_covariate",
            "+adversarial_batch(alpha=1.0, wt=150.0)",
            "+chl59-8b_1p(test_ids=[2,3])",
        ],
        "build": lambda: _patch_dual_chl59_lung5(
            _patch_dual_adversarial(
                _patch_dual_decoder_covariate(
                    _patch_quick(_BD(), max_epochs=1, batch_size=16)
                ),
                alpha=1.0, wt_adv_batch=150.0,
            ),
            test_batch_idx=[2, 3],
            batch_size=16,
        ),
    },
    "smoke-test+mmb0-1b_smb1-20b_1p": {
        "description": (
            "1-epoch end-to-end pipeline test on the mmb20 MERFISH+STARmap+ "
            "21-section data (test ids = [15, 82], train = the other 19 "
            "STARmap+ sections, val = 10% in-section cell sample). "
            "Dual-VQ + decoder covariate + adversarial GRL — same "
            "batch-correction recipe as the overnight sweep, capped at 1 "
            "epoch, early stopping disabled, narrow GNN (num_layers=1, "
            "sampler=[8], nbr_hops=1), batch_size lowered to 16 (memory "
            "is tight at 19 training sections worth of static `x` + "
            "edge_index on GPU)."
        ),
        "patches": [
            "+quick(max_epochs=1, batch_size=16, narrow GNN)",
            "+decoder_covariate",
            "+adversarial_batch(alpha=1.0, wt=150.0)",
            "+mmb0-1b_smb1-20b_1p(test_ids=[15,82])",
        ],
        "build": lambda: _patch_dual_mmb20(
            _patch_dual_adversarial(
                _patch_dual_decoder_covariate(
                    _patch_quick(_BD(), max_epochs=1, batch_size=16)
                ),
                alpha=1.0, wt_adv_batch=150.0,
            ),
            test_adata_ids=[15, 82],
            batch_size=16,
        ),
    },
    # ========================================================================
    # mmb0-1b_smb1-1b_1p 1-layer-GNN focused ablation matrix.
    # ========================================================================
    # Spine baseline = RVQ on both branches (k1=30, k2=90) + NicheCompass-
    # style decoder covariate + adversarial GRL on FULL z_mlp (alpha=1.0,
    # wt_adv_batch=150) + 1-layer SAGEConv GNN. Each variant ablates ONE
    # axis from this spine. Group via `--all-dataset mmb0-1b_smb1-1b_1p-ablations`.
    # The 1-layer spine (vs. the +wide 2-layer one used by the broader
    # sweep) is the user's preferred starting point on this dataset.
    #
    # Existing variants reused for the matrix:
    #   - dualvq+rvq-both+decoder-cov+adv+mmb0-1b_smb1-1b_1p     (the spine itself)
    #   - dualvq+wide+rvq-both+decoder-cov+adv+mmb0-1b_smb1-1b_1p (2-layer GNN)
    #   - dualvq+small+rvq-both+decoder-cov+adv+mmb0-1b_smb1-1b_1p (scvi-style encoder)
    "dualvq+rvq-both+decoder-cov+adv+gatv2+mmb0-1b_smb1-1b_1p": {
        "description": (
            "Spine ablation — SAGEConv -> GATv2Conv on the 1-layer GNN. "
            "Tests per-edge attention vs. uniform mean aggregation at "
            "1-hop on mmb-smb. Direct A/B against the +wide+gatv2 variant "
            "to see whether GAT helps at narrow GNN depth."
        ),
        "patches": [
            "+rvq(branch=both, levels=[30, 90])",
            "+decoder_covariate",
            "+adversarial_batch(alpha=1.0, wt=150.0)",
            "+gatv2(SAGEConv -> GATv2Conv, num_layers=1)",
        ],
        "build": lambda: _patch_dual_gatv2(
            _patch_dual_adversarial(
                _patch_dual_decoder_covariate(
                    _patch_dual_rvq(
                        _patch_dual_rvq(_BD(),
                            branch="niche", codebook_sizes=(30, 90)),
                        branch="cell", codebook_sizes=(30, 90),
                    ),
                ),
                alpha=1.0, wt_adv_batch=150.0,
            ),
        ),
    },
    "dualvq+rvq-both+decoder-cov+adv-w300+mmb0-1b_smb1-1b_1p": {
        "description": (
            "Spine ablation — adversarial weight bumped 150 -> 300 (2x). "
            "Pushes the encoder harder toward batch-invariant z_mlp; "
            "useful diagnostic for whether the default 150 is leaving "
            "headroom or whether stronger pressure starts collapsing "
            "codes."
        ),
        "patches": [
            "+rvq(branch=both, levels=[30, 90])",
            "+decoder_covariate",
            "+adversarial_batch(alpha=1.0, wt=300.0)",
        ],
        "build": lambda: _patch_dual_adversarial(
            _patch_dual_decoder_covariate(
                _patch_dual_rvq(
                    _patch_dual_rvq(_BD(),
                        branch="niche", codebook_sizes=(30, 90)),
                    branch="cell", codebook_sizes=(30, 90),
                ),
            ),
            alpha=1.0, wt_adv_batch=300.0,
        ),
    },
    "dualvq+rvq-both+decoder-cov+adv-w50+mmb0-1b_smb1-1b_1p": {
        "description": (
            "Spine ablation — adversarial weight cut 150 -> 50 (~3x "
            "lower). Tests whether the default GRL is over-regularising "
            "z_mlp at the cost of biological signal."
        ),
        "patches": [
            "+rvq(branch=both, levels=[30, 90])",
            "+decoder_covariate",
            "+adversarial_batch(alpha=1.0, wt=50.0)",
        ],
        "build": lambda: _patch_dual_adversarial(
            _patch_dual_decoder_covariate(
                _patch_dual_rvq(
                    _patch_dual_rvq(_BD(),
                        branch="niche", codebook_sizes=(30, 90)),
                    branch="cell", codebook_sizes=(30, 90),
                ),
            ),
            alpha=1.0, wt_adv_batch=50.0,
        ),
    },
    "dualvq+rvq-both+decoder-cov+adv+adj-w3000+mmb0-1b_smb1-1b_1p": {
        "description": (
            "Spine ablation — cosine adjacency BCE weight bumped 1000 -> "
            "3000 (3x). Stronger spatial-niche pressure on z_gnn; tests "
            "whether the codebooks tolerate even more spatial coherence "
            "than the default."
        ),
        "patches": [
            "+rvq(branch=both, levels=[30, 90])",
            "+decoder_covariate",
            "+adversarial_batch(alpha=1.0, wt=150.0)",
            "+adj(weight=3000)",
        ],
        "build": lambda: _patch_dual_adj_weight(
            _patch_dual_adversarial(
                _patch_dual_decoder_covariate(
                    _patch_dual_rvq(
                        _patch_dual_rvq(_BD(),
                            branch="niche", codebook_sizes=(30, 90)),
                        branch="cell", codebook_sizes=(30, 90),
                    ),
                ),
                alpha=1.0, wt_adv_batch=150.0,
            ),
            weight=3000.0,
        ),
    },
    "dualvq+rvq-both+decoder-cov+adv+adj-w250+mmb0-1b_smb1-1b_1p": {
        "description": (
            "Spine ablation — cosine adjacency BCE weight cut 1000 -> 250 "
            "(4x lower; matches an earlier default). Tests whether the "
            "default 1000 is over-pressuring spatial coherence at the "
            "cost of NB Pearson."
        ),
        "patches": [
            "+rvq(branch=both, levels=[30, 90])",
            "+decoder_covariate",
            "+adversarial_batch(alpha=1.0, wt=150.0)",
            "+adj(weight=250)",
        ],
        "build": lambda: _patch_dual_adj_weight(
            _patch_dual_adversarial(
                _patch_dual_decoder_covariate(
                    _patch_dual_rvq(
                        _patch_dual_rvq(_BD(),
                            branch="niche", codebook_sizes=(30, 90)),
                        branch="cell", codebook_sizes=(30, 90),
                    ),
                ),
                alpha=1.0, wt_adv_batch=150.0,
            ),
            weight=250.0,
        ),
    },
    # ========================================================================
    # mmb0-1b_smb1-1b_1p 1-layer-spine ablation matrix v2.
    # ========================================================================
    # Eight more ablations on top of the spine
    # (dualvq+rvq-both+decoder-cov+adv, 1-layer GNN), picked for
    # plausible upside on a small dataset (~85k cells, 431 genes).
    # Each varies ONE knob; group via
    # `--all-dataset mmb0-1b_smb1-1b_1p-ablations-v2`.
    "dualvq+rvq-both-large+decoder-cov+adv+mmb0-1b_smb1-1b_1p": {
        "description": (
            "Spine ablation — RVQ levels (30, 90) -> (50, 150). Bigger "
            "primary + residual codebook capacity; tests whether the "
            "default 30/90 is the bottleneck on Pearson."
        ),
        "patches": [
            "+rvq(branch=both, levels=[50, 150])",
            "+decoder_covariate", "+adversarial_batch(alpha=1.0, wt=150.0)",
        ],
        "build": lambda: _patch_dual_adversarial(
            _patch_dual_decoder_covariate(
                _patch_dual_rvq(
                    _patch_dual_rvq(_BD(),
                        branch="niche", codebook_sizes=(50, 150)),
                    branch="cell", codebook_sizes=(50, 150),
                ),
            ),
            alpha=1.0, wt_adv_batch=150.0,
        ),
    },
    "dualvq+rvq-both-3level+decoder-cov+adv+mmb0-1b_smb1-1b_1p": {
        "description": (
            "Spine ablation — 2-level RVQ (30, 90) -> 3-level RVQ "
            "(30, 60, 120). Finer hierarchical decomposition; each "
            "successive residual layer captures finer-grained variation. "
            "Worth trying when a 2-level RVQ saturates on Pearson."
        ),
        "patches": [
            "+rvq(branch=both, levels=[30, 60, 120])",
            "+decoder_covariate", "+adversarial_batch(alpha=1.0, wt=150.0)",
        ],
        "build": lambda: _patch_dual_adversarial(
            _patch_dual_decoder_covariate(
                _patch_dual_rvq(
                    _patch_dual_rvq(_BD(),
                        branch="niche", codebook_sizes=(30, 60, 120)),
                    branch="cell", codebook_sizes=(30, 60, 120),
                ),
            ),
            alpha=1.0, wt_adv_batch=150.0,
        ),
    },
    "dualvq+rvq-both+decoder-cov+adv+latent-128+mmb0-1b_smb1-1b_1p": {
        "description": (
            "Spine ablation — latent / codebook embedding dim 256 -> 128 "
            "(only the encoder MLP last hidden + GNN hidden shrink; "
            "decoder hidden widths and codebook sizes are unchanged). "
            "Tighter bottleneck à la NicheCompass / scvi; forces each "
            "code's embedding to be more semantically concentrated."
        ),
        "patches": [
            "+rvq(branch=both, levels=[30, 90])",
            "+decoder_covariate", "+adversarial_batch(alpha=1.0, wt=150.0)",
            "+small-latent(latent_dim=128)",
        ],
        "build": lambda: _patch_dual_small_latent(
            _patch_dual_adversarial(
                _patch_dual_decoder_covariate(
                    _patch_dual_rvq(
                        _patch_dual_rvq(_BD(),
                            branch="niche", codebook_sizes=(30, 90)),
                        branch="cell", codebook_sizes=(30, 90),
                    ),
                ),
                alpha=1.0, wt_adv_batch=150.0,
            ),
            latent_dim=128,
        ),
    },
    "dualvq+rvq-both+decoder-cov+adv+dropout+mmb0-1b_smb1-1b_1p": {
        "description": (
            "Spine ablation — dropout p=0.1 on encoder MLP, GNN, and "
            "both decoders. Cheap regularization. Particularly relevant "
            "on a small dataset (~85k cells) where the model can over-"
            "fit; pair with val_loss-based early stopping for clear "
            "wandb diagnostics."
        ),
        "patches": [
            "+rvq(branch=both, levels=[30, 90])",
            "+decoder_covariate", "+adversarial_batch(alpha=1.0, wt=150.0)",
            "+dropout(p=0.1)",
        ],
        "build": lambda: _patch_dual_dropout(
            _patch_dual_adversarial(
                _patch_dual_decoder_covariate(
                    _patch_dual_rvq(
                        _patch_dual_rvq(_BD(),
                            branch="niche", codebook_sizes=(30, 90)),
                        branch="cell", codebook_sizes=(30, 90),
                    ),
                ),
                alpha=1.0, wt_adv_batch=150.0,
            ),
            p=0.1,
        ),
    },
    "dualvq+rvq-both+decoder-cov+adv+batch-emb32+mmb0-1b_smb1-1b_1p": {
        "description": (
            "Spine ablation — learned batch embedding dim 16 -> 32. "
            "More capacity for the decoder's per-batch covariate to "
            "absorb cross-platform / cross-replicate gene-pattern "
            "differences. Encoder is unchanged (codes still batch-"
            "invariant by construction); only the decoder side gets "
            "a richer batch context."
        ),
        "patches": [
            "+rvq(branch=both, levels=[30, 90])",
            "+decoder_covariate(embed_dim=32)",
            "+adversarial_batch(alpha=1.0, wt=150.0)",
        ],
        "build": lambda: _patch_dual_adversarial(
            _patch_dual_decoder_covariate(
                _patch_dual_rvq(
                    _patch_dual_rvq(_BD(),
                        branch="niche", codebook_sizes=(30, 90)),
                    branch="cell", codebook_sizes=(30, 90),
                ),
                embed_dim=32,
            ),
            alpha=1.0, wt_adv_batch=150.0,
        ),
    },
    "dualvq+rvq-both+decoder-cov+adv+sampler16+mmb0-1b_smb1-1b_1p": {
        "description": (
            "Spine ablation — NeighborLoader.num_neighbors [8] -> [16]. "
            "Doubles the per-step niche-aggregation neighborhood; the "
            "GNN sees richer spatial context per cell, at the cost of "
            "~2x larger sampled subgraph (memory + step time). Often "
            "improves spatial-domain coherence on dense tissues."
        ),
        "patches": [
            "+rvq(branch=both, levels=[30, 90])",
            "+decoder_covariate", "+adversarial_batch(alpha=1.0, wt=150.0)",
            "+sampler(num_neighbors=[16])",
        ],
        "build": lambda: _patch_dual_sampler_neighbors(
            _patch_dual_adversarial(
                _patch_dual_decoder_covariate(
                    _patch_dual_rvq(
                        _patch_dual_rvq(_BD(),
                            branch="niche", codebook_sizes=(30, 90)),
                        branch="cell", codebook_sizes=(30, 90),
                    ),
                ),
                alpha=1.0, wt_adv_batch=150.0,
            ),
            num_neighbors=[16],
        ),
    },
    "dualvq+rvq-both+decoder-cov+adv+enc-deeper+mmb0-1b_smb1-1b_1p": {
        "description": (
            "Spine ablation — encoder MLP [400, 256] -> [400, 400, 256] "
            "(2 hidden layers -> 3 hidden layers; latent dim preserved "
            "at 256). Lets the encoder learn richer non-linear feature "
            "mixes before the VQ bottleneck."
        ),
        "patches": [
            "+rvq(branch=both, levels=[30, 90])",
            "+decoder_covariate", "+adversarial_batch(alpha=1.0, wt=150.0)",
            "+enc-deeper(mlp=[400, 400, 256])",
        ],
        "build": lambda: _patch_dual_encoder_deeper(
            _patch_dual_adversarial(
                _patch_dual_decoder_covariate(
                    _patch_dual_rvq(
                        _patch_dual_rvq(_BD(),
                            branch="niche", codebook_sizes=(30, 90)),
                        branch="cell", codebook_sizes=(30, 90),
                    ),
                ),
                alpha=1.0, wt_adv_batch=150.0,
            ),
        ),
    },
    "dualvq+rvq-both+decoder-cov+adv+gnn-h384+mmb0-1b_smb1-1b_1p": {
        "description": (
            "Spine ablation — GNN hidden 256 -> 384. Wider niche-branch "
            "GNN; larger spatial-aggregation capacity. Note: this also "
            "widens the niche latent dim (= GNN output) so the niche "
            "codebook embedding is wider too. Modest param + memory "
            "increase (~1.5x on the GNN); likely upside on datasets "
            "with rich spatial domain structure."
        ),
        "patches": [
            "+rvq(branch=both, levels=[30, 90])",
            "+decoder_covariate", "+adversarial_batch(alpha=1.0, wt=150.0)",
            "+gnn-h(384)",
        ],
        "build": lambda: _patch_dual_gnn_hidden(
            _patch_dual_adversarial(
                _patch_dual_decoder_covariate(
                    _patch_dual_rvq(
                        _patch_dual_rvq(_BD(),
                            branch="niche", codebook_sizes=(30, 90)),
                        branch="cell", codebook_sizes=(30, 90),
                    ),
                ),
                alpha=1.0, wt_adv_batch=150.0,
            ),
            hidden=384,
        ),
    },
    # ========================================================================
    # mmb0-1b_smb1-1b_1p 2-layer-spine ablation matrix v3 (codebook scan).
    # ========================================================================
    # Spine combines the v1 winners: +wide (best on niche ID) + adj-w250
    # (best on batch integration), plus the standard RVQ-both +
    # decoder-cov + adv. Each variant ablates one knob from this spine,
    # focused on CODEBOOK CAPACITY + STRUCTURE — the lever most likely
    # to push niche ID further now that the spatial-context (wide) and
    # batch-correction (w250) axes are pinned to their winners. Group
    # via `--all-dataset mmb0-1b_smb1-1b_1p-ablations-v3`.
    "dualvq+wide+rvq-both+decoder-cov+adv+adj-w250+mmb0-1b_smb1-1b_1p": {
        "description": (
            "v3 spine — combine the v1 winners. +wide (2-layer GNN, "
            "sampler [8,8], 2-hop nbr aggregation) was best on niche "
            "identification; +adj-w250 (cosine adjacency BCE weight "
            "1000 -> 250) was best on batch integration. Stack both; "
            "everything else is the standard recipe (RVQ-both 30/90, "
            "decoder covariate, adversarial GRL alpha=1.0 wt=150)."
        ),
        "patches": [
            "+wide", "+rvq(branch=both, levels=[30, 90])",
            "+decoder_covariate", "+adversarial_batch(alpha=1.0, wt=150.0)",
            "+adj(weight=250)",
        ],
        "build": lambda: _patch_dual_adj_weight(
            _patch_dual_adversarial(
                _patch_dual_decoder_covariate(
                    _patch_dual_rvq(
                        _patch_dual_rvq(
                            _patch_dual_wide(_BD()),
                            branch="niche", codebook_sizes=(30, 90),
                        ),
                        branch="cell", codebook_sizes=(30, 90),
                    ),
                ),
                alpha=1.0, wt_adv_batch=150.0,
            ),
            weight=250.0,
        ),
    },
    "dualvq+wide+rvq-both-50-90+decoder-cov+adv+adj-w250+mmb0-1b_smb1-1b_1p": {
        "description": (
            "v3 spine + RVQ k1: 30 -> 50. Same residual layer (k2=90); "
            "macro codebook gets ~1.7x more entries. Targets niche ID "
            "headroom: with `+wide` already capturing more spatial "
            "context, a finer macro partition is the natural next step."
        ),
        "patches": [
            "+wide", "+rvq(branch=both, levels=[50, 90])",
            "+decoder_covariate", "+adversarial_batch(alpha=1.0, wt=150.0)",
            "+adj(weight=250)",
        ],
        "build": lambda: _patch_dual_adj_weight(
            _patch_dual_adversarial(
                _patch_dual_decoder_covariate(
                    _patch_dual_rvq(
                        _patch_dual_rvq(
                            _patch_dual_wide(_BD()),
                            branch="niche", codebook_sizes=(50, 90),
                        ),
                        branch="cell", codebook_sizes=(50, 90),
                    ),
                ),
                alpha=1.0, wt_adv_batch=150.0,
            ),
            weight=250.0,
        ),
    },
    "dualvq+wide+rvq-both-80-90+decoder-cov+adv+adj-w250+mmb0-1b_smb1-1b_1p": {
        "description": (
            "v3 spine + RVQ k1: 30 -> 80. Aggressive scan on the macro "
            "codebook; k1=80 is roughly the empirical n_unique cell "
            "subtypes in mouse-brain MERFISH/STARmap atlases. If 50 "
            "isn't enough, this should saturate the metric."
        ),
        "patches": [
            "+wide", "+rvq(branch=both, levels=[80, 90])",
            "+decoder_covariate", "+adversarial_batch(alpha=1.0, wt=150.0)",
            "+adj(weight=250)",
        ],
        "build": lambda: _patch_dual_adj_weight(
            _patch_dual_adversarial(
                _patch_dual_decoder_covariate(
                    _patch_dual_rvq(
                        _patch_dual_rvq(
                            _patch_dual_wide(_BD()),
                            branch="niche", codebook_sizes=(80, 90),
                        ),
                        branch="cell", codebook_sizes=(80, 90),
                    ),
                ),
                alpha=1.0, wt_adv_batch=150.0,
            ),
            weight=250.0,
        ),
    },
    "dualvq+wide+rvq-both-50-150+decoder-cov+adv+adj-w250+mmb0-1b_smb1-1b_1p": {
        "description": (
            "v3 spine + RVQ levels 30/90 -> 50/150. Bumps BOTH layers "
            "(macro + residual). Tests whether the extra residual "
            "headroom helps or just adds redundant codes."
        ),
        "patches": [
            "+wide", "+rvq(branch=both, levels=[50, 150])",
            "+decoder_covariate", "+adversarial_batch(alpha=1.0, wt=150.0)",
            "+adj(weight=250)",
        ],
        "build": lambda: _patch_dual_adj_weight(
            _patch_dual_adversarial(
                _patch_dual_decoder_covariate(
                    _patch_dual_rvq(
                        _patch_dual_rvq(
                            _patch_dual_wide(_BD()),
                            branch="niche", codebook_sizes=(50, 150),
                        ),
                        branch="cell", codebook_sizes=(50, 150),
                    ),
                ),
                alpha=1.0, wt_adv_batch=150.0,
            ),
            weight=250.0,
        ),
    },
    "dualvq+wide+rvq-both-80-200+decoder-cov+adv+adj-w250+mmb0-1b_smb1-1b_1p": {
        "description": (
            "v3 spine + RVQ levels 30/90 -> 80/200. Most aggressive "
            "codebook scan; most likely to overfit on a 2-section "
            "dataset (~85k cells), but worth knowing where the cliff "
            "is for the wide+w250 spine."
        ),
        "patches": [
            "+wide", "+rvq(branch=both, levels=[80, 200])",
            "+decoder_covariate", "+adversarial_batch(alpha=1.0, wt=150.0)",
            "+adj(weight=250)",
        ],
        "build": lambda: _patch_dual_adj_weight(
            _patch_dual_adversarial(
                _patch_dual_decoder_covariate(
                    _patch_dual_rvq(
                        _patch_dual_rvq(
                            _patch_dual_wide(_BD()),
                            branch="niche", codebook_sizes=(80, 200),
                        ),
                        branch="cell", codebook_sizes=(80, 200),
                    ),
                ),
                alpha=1.0, wt_adv_batch=150.0,
            ),
            weight=250.0,
        ),
    },
    "dualvq+wide+rvq-both-3level+decoder-cov+adv+adj-w250+mmb0-1b_smb1-1b_1p": {
        "description": (
            "v3 spine + 3-level RVQ (30, 60, 120) instead of 2-level "
            "(30, 90). Same total nominal capacity (3*60 vs 1*90 "
            "residual entries) but split across two refinement layers, "
            "encouraging hierarchical decomposition."
        ),
        "patches": [
            "+wide", "+rvq(branch=both, levels=[30, 60, 120])",
            "+decoder_covariate", "+adversarial_batch(alpha=1.0, wt=150.0)",
            "+adj(weight=250)",
        ],
        "build": lambda: _patch_dual_adj_weight(
            _patch_dual_adversarial(
                _patch_dual_decoder_covariate(
                    _patch_dual_rvq(
                        _patch_dual_rvq(
                            _patch_dual_wide(_BD()),
                            branch="niche", codebook_sizes=(30, 60, 120),
                        ),
                        branch="cell", codebook_sizes=(30, 60, 120),
                    ),
                ),
                alpha=1.0, wt_adv_batch=150.0,
            ),
            weight=250.0,
        ),
    },
    "dualvq+wide+cvq-both-30-10+decoder-cov+adv+adj-w250+mmb0-1b_smb1-1b_1p": {
        "description": (
            "v3 spine + ConditionalVQ (tree-structured) on both branches "
            "in place of RVQ. K1=30 macro buckets, K2=10 sub-children "
            "per bucket = 300 effective codes per branch. Different "
            "inductive bias from RVQ: each level-2 entry is "
            "CONDITIONAL on the level-1 macro bucket (strict tree), "
            "while RVQ refines additively. Fits when the data is "
            "naturally hierarchical (cell type -> cell state)."
        ),
        "patches": [
            "+wide", "+cvq(branch=both, k1=30, k2=10)",
            "+decoder_covariate", "+adversarial_batch(alpha=1.0, wt=150.0)",
            "+adj(weight=250)",
        ],
        "build": lambda: _patch_dual_adj_weight(
            _patch_dual_adversarial(
                _patch_dual_decoder_covariate(
                    _patch_dual_cvq(
                        _patch_dual_wide(_BD()),
                        branch="both", k1=30, k2=10,
                    ),
                ),
                alpha=1.0, wt_adv_batch=150.0,
            ),
            weight=250.0,
        ),
    },
    "dualvq+wide+cvq-both-50-10+decoder-cov+adv+adj-w250+mmb0-1b_smb1-1b_1p": {
        "description": (
            "v3 spine + larger ConditionalVQ. K1=50 macro buckets, "
            "K2=10 sub-children = 500 effective codes per branch. "
            "Direct A/B against `+cvq-both-30-10` to scan macro "
            "capacity within the tree-VQ family."
        ),
        "patches": [
            "+wide", "+cvq(branch=both, k1=50, k2=10)",
            "+decoder_covariate", "+adversarial_batch(alpha=1.0, wt=150.0)",
            "+adj(weight=250)",
        ],
        "build": lambda: _patch_dual_adj_weight(
            _patch_dual_adversarial(
                _patch_dual_decoder_covariate(
                    _patch_dual_cvq(
                        _patch_dual_wide(_BD()),
                        branch="both", k1=50, k2=10,
                    ),
                ),
                alpha=1.0, wt_adv_batch=150.0,
            ),
            weight=250.0,
        ),
    },
    # ========================================================================
    # v4 ablations (final round): hybridise the two strongest non-wide
    # signals (`+enc-deeper`, `+adj-w3000`) with the niche-NMI winner
    # `+wide+rvq-both+decoder-cov+adv`, and explore a NEW axis —
    # adversarial warmup — to fix the cell-pearson + cell-NMI regressions
    # that `+wide` introduced. Spine for all 8: the v3 best
    # `dualvq+wide+rvq-both+decoder-cov+adv+mmb0-1b_smb1-1b_1p`.
    #
    # Justification (from v1+v2+v3 metrics on summary_long.csv):
    #   - `+enc-deeper` is the empirical winner on every batch-integration
    #     metric (cell iLISI 0.51 vs 0.23 baseline; nbr iLISI 0.075 vs
    #     0.007; cell MMD 0.0033 vs 0.0063), at only 4% niche-NMI cost.
    #     Untried with `+wide`. Highest expected ROI.
    #   - `+adj-w3000` is the second-strongest signal (niche NMI 0.6618
    #     + cell iLISI 0.40 simultaneously). High adjacency pressure
    #     pushes the GNN to encode shared spatial structure across
    #     batches. Untried with `+wide` (we've done w250 / w1000 only).
    #   - `+adv-warmup10` is a NEW axis: zero out the GRL alpha for the
    #     first 10 epochs so codes settle on biology before
    #     batch-invariance pressure kicks in. Targets the cell-pearson +
    #     cell-NMI regression.
    # ========================================================================
    "dualvq+wide+rvq-both+decoder-cov+adv+enc-deeper+mmb0-1b_smb1-1b_1p": {
        "description": (
            "v3 spine + deeper encoder MLP ([400, 400, 256] vs [400, "
            "256]). `+enc-deeper` was the standalone empirical winner "
            "on every batch-integration metric; this variant tests "
            "whether stacking on top of `+wide` recovers the iLISI "
            "gap without sacrificing niche NMI."
        ),
        "patches": [
            "+wide", "+rvq(branch=both, levels=[30, 90])",
            "+decoder_covariate", "+adversarial_batch(alpha=1.0, wt=150.0)",
            "+enc-deeper(mlp=[400, 400, 256])",
        ],
        "build": lambda: _patch_dual_encoder_deeper(
            _patch_dual_adversarial(
                _patch_dual_decoder_covariate(
                    _patch_dual_rvq(
                        _patch_dual_rvq(
                            _patch_dual_wide(_BD()),
                            branch="niche", codebook_sizes=(30, 90),
                        ),
                        branch="cell", codebook_sizes=(30, 90),
                    ),
                ),
                alpha=1.0, wt_adv_batch=150.0,
            ),
        ),
    },
    "dualvq+wide+rvq-both+decoder-cov+adv+adj-w3000+mmb0-1b_smb1-1b_1p": {
        "description": (
            "v3 spine + much stronger adjacency BCE (3000 vs default "
            "1000). On the non-wide spine, `+adj-w3000` was the best "
            "co-optimiser of niche NMI (0.66) and cell iLISI (0.40); "
            "untried with `+wide`. Tests whether wide+higher-adj is "
            "Pareto-optimal on niche identification + batch integration."
        ),
        "patches": [
            "+wide", "+rvq(branch=both, levels=[30, 90])",
            "+decoder_covariate", "+adversarial_batch(alpha=1.0, wt=150.0)",
            "+adj(weight=3000)",
        ],
        "build": lambda: _patch_dual_adj_weight(
            _patch_dual_adversarial(
                _patch_dual_decoder_covariate(
                    _patch_dual_rvq(
                        _patch_dual_rvq(
                            _patch_dual_wide(_BD()),
                            branch="niche", codebook_sizes=(30, 90),
                        ),
                        branch="cell", codebook_sizes=(30, 90),
                    ),
                ),
                alpha=1.0, wt_adv_batch=150.0,
            ),
            weight=3000.0,
        ),
    },
    "dualvq+wide+rvq-both+decoder-cov+adv+enc-deeper+adj-w3000+mmb0-1b_smb1-1b_1p": {
        "description": (
            "v3 spine + the two strongest empirical knobs stacked: "
            "deeper encoder MLP + adj weight 3000. Best-case Pareto "
            "winner; if the two signals interact additively this is "
            "the variant most likely to top all four priority metrics "
            "simultaneously."
        ),
        "patches": [
            "+wide", "+rvq(branch=both, levels=[30, 90])",
            "+decoder_covariate", "+adversarial_batch(alpha=1.0, wt=150.0)",
            "+enc-deeper", "+adj(weight=3000)",
        ],
        "build": lambda: _patch_dual_adj_weight(
            _patch_dual_encoder_deeper(
                _patch_dual_adversarial(
                    _patch_dual_decoder_covariate(
                        _patch_dual_rvq(
                            _patch_dual_rvq(
                                _patch_dual_wide(_BD()),
                                branch="niche", codebook_sizes=(30, 90),
                            ),
                            branch="cell", codebook_sizes=(30, 90),
                        ),
                    ),
                    alpha=1.0, wt_adv_batch=150.0,
                ),
            ),
            weight=3000.0,
        ),
    },
    "dualvq+wide+rvq-both+decoder-cov+adv-warmup10+mmb0-1b_smb1-1b_1p": {
        "description": (
            "v3 spine + 10-epoch adversarial warmup. During the first "
            "10 train epochs the GRL is run with alpha=0 (encoder gets "
            "zero adversarial gradient) so cell + niche codes settle "
            "on biology before batch-invariance pressure kicks in. "
            "Targets the cell-NMI / cell-pearson regression that "
            "`+wide+adv` introduced. NEW axis (untried)."
        ),
        "patches": [
            "+wide", "+rvq(branch=both, levels=[30, 90])",
            "+decoder_covariate",
            "+adversarial_batch(alpha=1.0, wt=150.0, warmup=10)",
        ],
        "build": lambda: _patch_dual_adversarial(
            _patch_dual_decoder_covariate(
                _patch_dual_rvq(
                    _patch_dual_rvq(
                        _patch_dual_wide(_BD()),
                        branch="niche", codebook_sizes=(30, 90),
                    ),
                    branch="cell", codebook_sizes=(30, 90),
                ),
            ),
            alpha=1.0, wt_adv_batch=150.0, warmup_epochs=10,
        ),
    },
    "dualvq+wide+rvq-both+decoder-cov+adv-w50+mmb0-1b_smb1-1b_1p": {
        "description": (
            "v3 spine + lower adversary weight (50 vs default 150). "
            "On the non-wide spine, `+adv-w50` was the best niche-NMI "
            "variant. Tests whether `+wide` over-corrected on adversary "
            "weight; if so this should improve niche NMI further."
        ),
        "patches": [
            "+wide", "+rvq(branch=both, levels=[30, 90])",
            "+decoder_covariate", "+adversarial_batch(alpha=1.0, wt=50.0)",
        ],
        "build": lambda: _patch_dual_adversarial(
            _patch_dual_decoder_covariate(
                _patch_dual_rvq(
                    _patch_dual_rvq(
                        _patch_dual_wide(_BD()),
                        branch="niche", codebook_sizes=(30, 90),
                    ),
                    branch="cell", codebook_sizes=(30, 90),
                ),
            ),
            alpha=1.0, wt_adv_batch=50.0,
        ),
    },
    "dualvq+wide+rvq-both+decoder-cov+adv+enc-deeper+adv-warmup10+mmb0-1b_smb1-1b_1p": {
        "description": (
            "v3 spine + deeper encoder + 10-epoch adversarial warmup. "
            "Combines the strongest empirical iLISI signal with the "
            "new warmup axis: the encoder gets to learn rich features "
            "(deeper MLP) AND has 10 epochs to commit them to the "
            "codebook before adversarial pressure starts."
        ),
        "patches": [
            "+wide", "+rvq(branch=both, levels=[30, 90])",
            "+decoder_covariate",
            "+adversarial_batch(alpha=1.0, wt=150.0, warmup=10)",
            "+enc-deeper",
        ],
        "build": lambda: _patch_dual_encoder_deeper(
            _patch_dual_adversarial(
                _patch_dual_decoder_covariate(
                    _patch_dual_rvq(
                        _patch_dual_rvq(
                            _patch_dual_wide(_BD()),
                            branch="niche", codebook_sizes=(30, 90),
                        ),
                        branch="cell", codebook_sizes=(30, 90),
                    ),
                ),
                alpha=1.0, wt_adv_batch=150.0, warmup_epochs=10,
            ),
        ),
    },
    # ========================================================================
    # mmb0-1b_smb1-1b_1p ablations v6 — push cell-token NMI on cell_type
    # WITHOUT regressing batch integration. Diagnostic from summary_long.csv:
    #   - `adv-warmup10` family: best cell-NMI (~0.50) but iLISI ≈ 0
    #     (warm-up lets the cell token learn batch-correlated features
    #     early; the late-arriving adversary can't remove them).
    #   - `enc-deeper` family: best iLISI (~0.50) but cell-NMI ~0.40
    #     (strong adv from epoch 0 keeps the cell token batch-clean
    #     but never lets it commit batch-discriminating features).
    # The 8 variants below test orthogonal mechanisms for restoring
    # integration AFTER a warmup phase, plus capacity-bottleneck
    # variants with built-in protection. Group via
    # `--all-dataset mmb0-1b_smb1-1b_1p-ablations-v6`.
    # ========================================================================
    # Group A — smaller encoder + warmup
    "dualvq+wide+rvq-both+decoder-cov+adv+enc-shallow+adv-warmup5+mmb0-1b_smb1-1b_1p": {
        "description": (
            "v6 — encoder MLP shrunk to 1 hidden layer ([256] vs the "
            "default [400, 256]) + 5-epoch warmup. Hypothesis: capacity "
            "bottleneck reduces room to memorise batch-specific features; "
            "short warmup gives cell token a head start but limits how "
            "much batch-correlated info it can encode before the "
            "adversary kicks in. Expected: cell-NMI 0.42-0.46; "
            "iLISI 0.15-0.25."
        ),
        "patches": [
            "+wide", "+rvq(branch=both, levels=[30, 90])",
            "+decoder_covariate",
            "+adversarial_batch(alpha=1.0, wt=150.0, warmup=5)",
            "+enc-shallow(mlp=[256])",
        ],
        "build": lambda: _patch_dual_encoder_shallow(
            _patch_dual_adversarial(
                _patch_dual_decoder_covariate(
                    _patch_dual_rvq(
                        _patch_dual_rvq(
                            _patch_dual_wide(_BD()),
                            branch="niche", codebook_sizes=(30, 90),
                        ),
                        branch="cell", codebook_sizes=(30, 90),
                    ),
                ),
                alpha=1.0, wt_adv_batch=150.0, warmup_epochs=5,
            ),
        ),
    },
    "dualvq+wide+rvq-both+decoder-cov+adv+enc-shallow+adv-warmup10+mmb0-1b_smb1-1b_1p": {
        "description": (
            "v6 — same as above but warmup extended to 10 epochs. "
            "Tests whether the capacity bottleneck is enough to keep "
            "iLISI alive at the longer warmup that previously killed "
            "integration entirely. Expected: cell-NMI 0.46-0.50; "
            "iLISI 0.10-0.20."
        ),
        "patches": [
            "+wide", "+rvq(branch=both, levels=[30, 90])",
            "+decoder_covariate",
            "+adversarial_batch(alpha=1.0, wt=150.0, warmup=10)",
            "+enc-shallow(mlp=[256])",
        ],
        "build": lambda: _patch_dual_encoder_shallow(
            _patch_dual_adversarial(
                _patch_dual_decoder_covariate(
                    _patch_dual_rvq(
                        _patch_dual_rvq(
                            _patch_dual_wide(_BD()),
                            branch="niche", codebook_sizes=(30, 90),
                        ),
                        branch="cell", codebook_sizes=(30, 90),
                    ),
                ),
                alpha=1.0, wt_adv_batch=150.0, warmup_epochs=10,
            ),
        ),
    },
    "dualvq+small+rvq-both+decoder-cov+adv+adv-warmup5+mmb0-1b_smb1-1b_1p": {
        "description": (
            "v6 — `+small` (scvi-style 1-layer encoder + narrow GNN + "
            "narrow decoders) + warmup 5. The most aggressive capacity "
            "limit; tests whether a much smaller model finds a sweet "
            "spot. Existing `+small+rvq+adv` (no warmup) already gave "
            "cell-NMI 0.465 / iLISI 0.071; warmup should push cell-NMI "
            "higher. Expected: cell-NMI 0.46-0.50; iLISI 0.05-0.15."
        ),
        "patches": [
            "+small(hidden=128)", "+rvq(branch=both, levels=[30, 90])",
            "+decoder_covariate",
            "+adversarial_batch(alpha=1.0, wt=150.0, warmup=5)",
        ],
        "build": lambda: _patch_dual_adversarial(
            _patch_dual_decoder_covariate(
                _patch_dual_rvq(
                    _patch_dual_rvq(
                        _patch_dual_small(_BD()),
                        branch="niche", codebook_sizes=(30, 90),
                    ),
                    branch="cell", codebook_sizes=(30, 90),
                ),
            ),
            alpha=1.0, wt_adv_batch=150.0, warmup_epochs=5,
        ),
    },
    # Group B — warmup paired with non-adversarial integration mechanism (MMD)
    "dualvq+wide+rvq-both+decoder-cov+adv+adv-warmup10+mmd-w50+mmb0-1b_smb1-1b_1p": {
        "description": (
            "v6 — best-performing warmup10 spine + MMD-batch loss "
            "(λ=50) on the cell-token input z_mlp[:B]. MMD has no "
            "min-max game so it doesn't suffer the warmup pathology: "
            "it pushes per-batch distributions together by ordinary "
            "backprop on every step, including during the adversarial "
            "warmup window. Expected: the breakthrough variant — "
            "cell-NMI 0.48-0.51; iLISI 0.15-0.30."
        ),
        "patches": [
            "+wide", "+rvq(branch=both, levels=[30, 90])",
            "+decoder_covariate",
            "+adversarial_batch(alpha=1.0, wt=150.0, warmup=10)",
            "+mmd_batch(wt=50, n_sub=512)",
        ],
        "build": lambda: _patch_dual_mmd_batch(
            _patch_dual_adversarial(
                _patch_dual_decoder_covariate(
                    _patch_dual_rvq(
                        _patch_dual_rvq(
                            _patch_dual_wide(_BD()),
                            branch="niche", codebook_sizes=(30, 90),
                        ),
                        branch="cell", codebook_sizes=(30, 90),
                    ),
                ),
                alpha=1.0, wt_adv_batch=150.0, warmup_epochs=10,
            ),
            wt_mmd_batch=50.0, n_sub=512,
        ),
    },
    "dualvq+wide+rvq-both+decoder-cov+adv+adv-warmup10+mmd-w200+mmb0-1b_smb1-1b_1p": {
        "description": (
            "v6 — same as above but MMD λ=200 (4x stronger). Tests "
            "whether aggressive MMD makes integration a hard "
            "constraint. Risk: kills the cell-type signal if too "
            "strong. Expected: cell-NMI 0.45-0.50; iLISI 0.25-0.40."
        ),
        "patches": [
            "+wide", "+rvq(branch=both, levels=[30, 90])",
            "+decoder_covariate",
            "+adversarial_batch(alpha=1.0, wt=150.0, warmup=10)",
            "+mmd_batch(wt=200, n_sub=512)",
        ],
        "build": lambda: _patch_dual_mmd_batch(
            _patch_dual_adversarial(
                _patch_dual_decoder_covariate(
                    _patch_dual_rvq(
                        _patch_dual_rvq(
                            _patch_dual_wide(_BD()),
                            branch="niche", codebook_sizes=(30, 90),
                        ),
                        branch="cell", codebook_sizes=(30, 90),
                    ),
                ),
                alpha=1.0, wt_adv_batch=150.0, warmup_epochs=10,
            ),
            wt_mmd_batch=200.0, n_sub=512,
        ),
    },
    "dualvq+wide+rvq-both+decoder-cov+mmd-w100+mmb0-1b_smb1-1b_1p": {
        "description": (
            "v6 — REPLACE adversary with MMD (λ=100), no warmup needed. "
            "Critical comparison: tests whether the adversary is even "
            "necessary, or whether MMD alone gives both integration "
            "and a clean cell-token. Simpler architecture (no "
            "classifier to tune). Expected: cell-NMI 0.42-0.48; "
            "iLISI 0.20-0.35."
        ),
        "patches": [
            "+wide", "+rvq(branch=both, levels=[30, 90])",
            "+decoder_covariate",
            "+mmd_batch(wt=100, n_sub=512)",
        ],
        "build": lambda: _patch_dual_mmd_batch(
            _patch_dual_decoder_covariate(
                _patch_dual_rvq(
                    _patch_dual_rvq(
                        _patch_dual_wide(_BD()),
                        branch="niche", codebook_sizes=(30, 90),
                    ),
                    branch="cell", codebook_sizes=(30, 90),
                ),
            ),
            wt_mmd_batch=100.0, n_sub=512,
        ),
    },
    # Group C — schedule + asymmetric adversary
    "dualvq+wide+rvq-both+decoder-cov+adv-cell-only+adv-w300+mmb0-1b_smb1-1b_1p": {
        "description": (
            "v6 — adversarial classifier sees ONLY z_mlp[:B] (the "
            "cell-branch input pre-VQ) instead of the full tensor. "
            "Niche-branch GNN consumes un-classified neighbour rows "
            "of z_mlp, leaving the niche pathway un-pressured by "
            "GRL. Adv weight bumped to 300 to compensate for the "
            "smaller per-step adversarial signal. Hypothesis: "
            "concentrating the adversary on the cell branch where "
            "batch-correction matters most, while letting niche keep "
            "spatial info. Expected: cell-NMI 0.43-0.48; "
            "iLISI 0.30-0.45 on cell branch."
        ),
        "patches": [
            "+wide", "+rvq(branch=both, levels=[30, 90])",
            "+decoder_covariate",
            "+adversarial_batch(alpha=1.0, wt=300.0, apply_to=cell)",
        ],
        "build": lambda: _patch_dual_adversarial_cell_only(
            _patch_dual_adversarial(
                _patch_dual_decoder_covariate(
                    _patch_dual_rvq(
                        _patch_dual_rvq(
                            _patch_dual_wide(_BD()),
                            branch="niche", codebook_sizes=(30, 90),
                        ),
                        branch="cell", codebook_sizes=(30, 90),
                    ),
                ),
                alpha=1.0, wt_adv_batch=300.0,
            ),
        ),
    },
    "dualvq+wide+rvq-both+decoder-cov+adv-cosine+adv-warmup10+mmb0-1b_smb1-1b_1p": {
        "description": (
            "v6 — cosine schedule for adversarial alpha + warmup 10. "
            "Alpha = 0 during warmup, then ramps up via half-sine "
            "envelope to peak at the midpoint of the post-warmup "
            "phase, decays to ~0 at total_epochs=100. The decay "
            "phase lets the cell token consolidate cell-type-"
            "discriminative features WITHOUT active gradient reversal "
            "in the final epochs. Expected: cell-NMI 0.47-0.51; "
            "iLISI 0.10-0.25."
        ),
        "patches": [
            "+wide", "+rvq(branch=both, levels=[30, 90])",
            "+decoder_covariate",
            "+adversarial_batch(alpha=1.0, wt=150.0, warmup=10, schedule=cosine)",
        ],
        "build": lambda: _patch_dual_adversarial_cosine(
            _patch_dual_adversarial(
                _patch_dual_decoder_covariate(
                    _patch_dual_rvq(
                        _patch_dual_rvq(
                            _patch_dual_wide(_BD()),
                            branch="niche", codebook_sizes=(30, 90),
                        ),
                        branch="cell", codebook_sizes=(30, 90),
                    ),
                ),
                alpha=1.0, wt_adv_batch=150.0, warmup_epochs=10,
            ),
            total_epochs=100,
        ),
    },
    # ========================================================================
    # mmb0-1b_smb1-1b_1p ablations v6b — second wave targeting cell-token NMI.
    # The first v6 wave (3 MMD variants + enc-shallow + small + adv-cell-only +
    # adv-cosine) had the MMD variants error out on a kwarg name mismatch
    # (now fixed) and the rest underperform. v6b re-runs the 3 MMD ones
    # and adds 5 new mechanisms that ablate different parts of the
    # warmup-NMI / integration trade-off:
    #   - Post-warmup adv-weight scan (w50, w300) — does the ceiling come
    #     from the warmup phase or the post-warmup adversarial pressure?
    #   - Sharper cell VQ commit (commit-cell-w2) — discrete codes more
    #     concentrated, less smearing of cell-type clusters across codes.
    #   - Semi-supervised cell-type CE (ce-w10) — empirical upper bound
    #     for "what if we directly optimise cell-NMI?". Trains the
    #     existing Linear predictor head; CLEARLY LABELED as
    #     semi-supervised in methods.
    #   - Combo safety stack (enc-shallow + warmup10 + mmd-w50) — pulls
    #     the three best individual mechanisms together.
    # Group via `--all-dataset mmb0-1b_smb1-1b_1p-ablations-v6b`.
    # ========================================================================
    "dualvq+wide+rvq-both+decoder-cov+adv-w50+adv-warmup10+mmb0-1b_smb1-1b_1p": {
        "description": (
            "v6b — adv weight 50 (vs default 150) + 10-epoch warmup. "
            "Tests whether the warmup-NMI ceiling at ~0.50 comes from "
            "the post-warmup adversarial pressure being too strong "
            "(scrubbing cell-type signal) vs too weak (failing to "
            "remove batch info). Light adv after warmup = cell branch "
            "barely loses cell-type-discriminative signal. Expected: "
            "cell-NMI 0.50-0.53; iLISI 0.05-0.15 (low integration is "
            "the cost; report alongside MMD variants)."
        ),
        "patches": [
            "+wide", "+rvq(branch=both, levels=[30, 90])",
            "+decoder_covariate",
            "+adversarial_batch(alpha=1.0, wt=50.0, warmup=10)",
        ],
        "build": lambda: _patch_dual_adversarial(
            _patch_dual_decoder_covariate(
                _patch_dual_rvq(
                    _patch_dual_rvq(
                        _patch_dual_wide(_BD()),
                        branch="niche", codebook_sizes=(30, 90),
                    ),
                    branch="cell", codebook_sizes=(30, 90),
                ),
            ),
            alpha=1.0, wt_adv_batch=50.0, warmup_epochs=10,
        ),
    },
    "dualvq+wide+rvq-both+decoder-cov+adv-w300+adv-warmup10+mmb0-1b_smb1-1b_1p": {
        "description": (
            "v6b — adv weight 300 (2x default) + 10-epoch warmup. "
            "Other half of the post-warmup-adv scan: tests whether "
            "STRONGER adversary after warmup recovers integration "
            "without fully scrubbing the cell-type signal that warmup "
            "established. Expected: cell-NMI 0.46-0.50; iLISI 0.10-"
            "0.25 (better integration than w50, lower NMI than w50)."
        ),
        "patches": [
            "+wide", "+rvq(branch=both, levels=[30, 90])",
            "+decoder_covariate",
            "+adversarial_batch(alpha=1.0, wt=300.0, warmup=10)",
        ],
        "build": lambda: _patch_dual_adversarial(
            _patch_dual_decoder_covariate(
                _patch_dual_rvq(
                    _patch_dual_rvq(
                        _patch_dual_wide(_BD()),
                        branch="niche", codebook_sizes=(30, 90),
                    ),
                    branch="cell", codebook_sizes=(30, 90),
                ),
            ),
            alpha=1.0, wt_adv_batch=300.0, warmup_epochs=10,
        ),
    },
    "dualvq+wide+rvq-both+decoder-cov+commit-cell-w2+adv+adv-warmup10+mmb0-1b_smb1-1b_1p": {
        "description": (
            "v6b — cell-branch commit weight 2 (vs default 1) + 10-"
            "epoch warmup. Sharpens the VQ assignment: z_mlp pulled "
            "harder toward z_q_cell, codes become more concentrated "
            "(less smearing of within-cell-type variance across codes). "
            "Hypothesis: cells of the same cell-type collapse to fewer "
            "codes -> NMI(cell-token, cell_type) up. Risk: niche "
            "branch's GNN consumes the SAME z_mlp tensor (sampled "
            "neighbours), so a sharper z_mlp may degrade niche "
            "aggregation. Expected: cell-NMI 0.50-0.55; niche-NMI "
            "potentially down 0.02-0.05; iLISI similar to baseline."
        ),
        "patches": [
            "+wide", "+rvq(branch=both, levels=[30, 90])",
            "+decoder_covariate",
            "+commit_cell(weight=2.0)",
            "+adversarial_batch(alpha=1.0, wt=150.0, warmup=10)",
        ],
        "build": lambda: _patch_dual_commit_cell_weight(
            _patch_dual_adversarial(
                _patch_dual_decoder_covariate(
                    _patch_dual_rvq(
                        _patch_dual_rvq(
                            _patch_dual_wide(_BD()),
                            branch="niche", codebook_sizes=(30, 90),
                        ),
                        branch="cell", codebook_sizes=(30, 90),
                    ),
                ),
                alpha=1.0, wt_adv_batch=150.0, warmup_epochs=10,
            ),
            wt_commit_cell=2.0,
        ),
    },
    "dualvq+wide+rvq-both+decoder-cov+ce-w10+adv+adv-warmup10+mmb0-1b_smb1-1b_1p": {
        "description": (
            "v6b — auxiliary cross-entropy on the Linear predictor head "
            "(z_mlp[:B] -> n_cell_types) with weight 10. Adds direct "
            "supervised cell-type signal — the Linear predictor was "
            "always built but its loss was never enabled. SEMI-"
            "SUPERVISED: uses ground-truth cell_type at training time. "
            "Empirical upper bound for cell-NMI; CLEARLY LABEL as "
            "semi-supervised in paper methods (the rest of SQUINT, "
            "scVI, NicheCompass, the foundation-model baselines, etc. "
            "are unsupervised or zero-shot). Expected: cell-NMI 0.55-"
            "0.65 (large bump); iLISI varies based on adv-warmup "
            "interaction."
        ),
        "patches": [
            "+wide", "+rvq(branch=both, levels=[30, 90])",
            "+decoder_covariate",
            "+cross_entropy_loss(weight=10.0)",
            "+adversarial_batch(alpha=1.0, wt=150.0, warmup=10)",
        ],
        "build": lambda: _patch_dual_cross_entropy(
            _patch_dual_adversarial(
                _patch_dual_decoder_covariate(
                    _patch_dual_rvq(
                        _patch_dual_rvq(
                            _patch_dual_wide(_BD()),
                            branch="niche", codebook_sizes=(30, 90),
                        ),
                        branch="cell", codebook_sizes=(30, 90),
                    ),
                ),
                alpha=1.0, wt_adv_batch=150.0, warmup_epochs=10,
            ),
            wt_cross_entropy=10.0,
        ),
    },
    "dualvq+wide+rvq-both+decoder-cov+adv+enc-shallow+adv-warmup10+mmd-w50+mmb0-1b_smb1-1b_1p": {
        "description": (
            "v6b — combination safety stack: shallow encoder + 10-epoch "
            "warmup + MMD λ=50 for integration. Pulls together the "
            "three mechanisms expected to cooperate:\n"
            "  1. enc-shallow: capacity bottleneck reduces batch "
            "memorisation\n"
            "  2. adv-warmup10: cell token learns rich features early\n"
            "  3. mmd-w50: non-adversarial integration during warmup, "
            "stays active throughout\n"
            "Expected: cell-NMI 0.47-0.51; iLISI 0.20-0.35. The "
            "headline target — both axes simultaneously above the "
            "current Pareto frontier."
        ),
        "patches": [
            "+wide", "+rvq(branch=both, levels=[30, 90])",
            "+decoder_covariate",
            "+adversarial_batch(alpha=1.0, wt=150.0, warmup=10)",
            "+enc-shallow(mlp=[256])",
            "+mmd_batch(wt=50, n_sub=512)",
        ],
        "build": lambda: _patch_dual_mmd_batch(
            _patch_dual_encoder_shallow(
                _patch_dual_adversarial(
                    _patch_dual_decoder_covariate(
                        _patch_dual_rvq(
                            _patch_dual_rvq(
                                _patch_dual_wide(_BD()),
                                branch="niche", codebook_sizes=(30, 90),
                            ),
                            branch="cell", codebook_sizes=(30, 90),
                        ),
                    ),
                    alpha=1.0, wt_adv_batch=150.0, warmup_epochs=10,
                ),
            ),
            wt_mmd_batch=50.0, n_sub=512,
        ),
    },
    "dualvq+wide+rvq-both+decoder-cov+adv+nbr-hops-3+mmb0-1b_smb1-1b_1p": {
        "description": (
            "v3 spine + 3-hop neighbourhood mean as the niche target "
            "(was 2-hop). Sampler bumped to [8, 8, 8] so the GNN sees "
            "a complete 3-hop neighbourhood. Pushes niche codes to "
            "encode tissue-region-scale structure rather than 1-2-hop "
            "local context. Risk: niche becomes too coarse, niche NMI "
            "drops."
        ),
        "patches": [
            "+wide", "+rvq(branch=both, levels=[30, 90])",
            "+decoder_covariate", "+adversarial_batch(alpha=1.0, wt=150.0)",
            "+nbr-hops=3", "+sampler=[8, 8, 8]",
        ],
        "build": lambda: _patch_dual_sampler_neighbors(
            _patch_dual_adversarial(
                _patch_dual_decoder_covariate(
                    _patch_dual_rvq(
                        _patch_dual_rvq(
                            # Override nbr_aggregation_hops AFTER
                            # _patch_dual_wide (which sets it to 2).
                            (lambda c: (c["model"]["loss_params"]["loss_kwargs"].update(
                                {"nbr_aggregation_hops": 3}) or c))(
                                _patch_dual_wide(_BD())
                            ),
                            branch="niche", codebook_sizes=(30, 90),
                        ),
                        branch="cell", codebook_sizes=(30, 90),
                    ),
                ),
                alpha=1.0, wt_adv_batch=150.0,
            ),
            num_neighbors=[8, 8, 8],
        ),
    },
    "dualvq+wide+rvq-both+decoder-cov+adv+enc-deeper+adj-w3000+adv-warmup10+mmb0-1b_smb1-1b_1p": {
        "description": (
            "v3 spine + the kitchen sink of v4's three independently-"
            "promising signals: deeper encoder + adj weight 3000 + "
            "10-epoch adversarial warmup. If pairwise interactions are "
            "additive, this is the variant most likely to be Pareto-"
            "best across all four priority metrics simultaneously."
        ),
        "patches": [
            "+wide", "+rvq(branch=both, levels=[30, 90])",
            "+decoder_covariate",
            "+adversarial_batch(alpha=1.0, wt=150.0, warmup=10)",
            "+enc-deeper", "+adj(weight=3000)",
        ],
        "build": lambda: _patch_dual_adj_weight(
            _patch_dual_encoder_deeper(
                _patch_dual_adversarial(
                    _patch_dual_decoder_covariate(
                        _patch_dual_rvq(
                            _patch_dual_rvq(
                                _patch_dual_wide(_BD()),
                                branch="niche", codebook_sizes=(30, 90),
                            ),
                            branch="cell", codebook_sizes=(30, 90),
                        ),
                    ),
                    alpha=1.0, wt_adv_batch=150.0, warmup_epochs=10,
                ),
            ),
            weight=3000.0,
        ),
    },
    # ========================================================================
    # v5 ablations (final-final round): codebook-capacity scan on the new
    # champion spine `+wide+rvq-both-3level+decoder-cov+adv+enc-deeper`.
    #
    # Insight from v1+v3+v4: `+enc-deeper` drives batch integration,
    # `+wide` drives niche NMI, `+rvq-both-3level` drives Pearson. v5
    # combines all three and varies ONLY the codebook structure.
    #
    # 6 RVQ-3-level variants (capacity scan on residual hierarchy) +
    # 2 CVQ-3-level variants (architectural alternative: tree
    # partitioning instead of residual decomposition, at matched depth).
    # CVQ depth was extended from 2 to 3 levels via a ~80-line addition
    # to `ConditionalVQ.forward` (`vq_l3 = K1*K2 leaf codebooks of
    # size K3`); the residual STE now flows through three levels and
    # `num_quantizers=3` so codebook-utilisation metrics stratify all
    # three.
    #
    # Two of the symmetric scaling slots were swapped for MLP-width
    # ablations (per user request): one variant shrinks the encoder +
    # decoder MLPs in lockstep, one widens them.
    # ========================================================================
    "dualvq+wide+rvq-both-3level+decoder-cov+adv+enc-deeper+mmb0-1b_smb1-1b_1p": {
        "description": (
            "v5 anchor + new project default. Combines the three "
            "v1-v4 winners on independent axes: `+wide` (niche NMI), "
            "`+enc-deeper` (batch integration), `+rvq-both-3level` "
            "(Pearson). Codebook structure: 3-level residual VQ with "
            "(30, 60, 120) sizes per branch -- 216 k effective codes."
        ),
        "patches": [
            "+wide", "+rvq(branch=both, levels=[30, 60, 120])",
            "+decoder_covariate", "+adversarial_batch(alpha=1.0, wt=150.0)",
            "+enc-deeper(mlp=[400, 400, 256])",
        ],
        "build": lambda: _patch_dual_encoder_deeper(
            _patch_dual_adversarial(
                _patch_dual_decoder_covariate(
                    _patch_dual_rvq(
                        _patch_dual_rvq(
                            _patch_dual_wide(_BD()),
                            branch="niche", codebook_sizes=(30, 60, 120),
                        ),
                        branch="cell", codebook_sizes=(30, 60, 120),
                    ),
                ),
                alpha=1.0, wt_adv_batch=150.0,
            ),
        ),
    },
    "dualvq+wide+rvq-both-3level-50-100-200+decoder-cov+adv+enc-deeper+mmb0-1b_smb1-1b_1p": {
        "description": (
            "v5 spine + RVQ-3-level scaled up symmetrically: "
            "(30, 60, 120) -> (50, 100, 200). 5x effective capacity "
            "(216 k -> 1 M). Tests whether the 3-level structure is "
            "capacity-limited at the v5 anchor."
        ),
        "patches": [
            "+wide", "+rvq(branch=both, levels=[50, 100, 200])",
            "+decoder_covariate", "+adversarial_batch(alpha=1.0, wt=150.0)",
            "+enc-deeper",
        ],
        "build": lambda: _patch_dual_encoder_deeper(
            _patch_dual_adversarial(
                _patch_dual_decoder_covariate(
                    _patch_dual_rvq(
                        _patch_dual_rvq(
                            _patch_dual_wide(_BD()),
                            branch="niche", codebook_sizes=(50, 100, 200),
                        ),
                        branch="cell", codebook_sizes=(50, 100, 200),
                    ),
                ),
                alpha=1.0, wt_adv_batch=150.0,
            ),
        ),
    },
    "dualvq+wide+rvq-both-3level-80-160-320+decoder-cov+adv+enc-deeper+mmb0-1b_smb1-1b_1p": {
        "description": (
            "v5 spine + RVQ-3-level scaled up aggressively: "
            "(30, 60, 120) -> (80, 160, 320). ~19x effective capacity "
            "(216 k -> 4.1 M). Tests the ceiling. Risk: dead codes if "
            "per-leaf residual gets too sparse."
        ),
        "patches": [
            "+wide", "+rvq(branch=both, levels=[80, 160, 320])",
            "+decoder_covariate", "+adversarial_batch(alpha=1.0, wt=150.0)",
            "+enc-deeper",
        ],
        "build": lambda: _patch_dual_encoder_deeper(
            _patch_dual_adversarial(
                _patch_dual_decoder_covariate(
                    _patch_dual_rvq(
                        _patch_dual_rvq(
                            _patch_dual_wide(_BD()),
                            branch="niche", codebook_sizes=(80, 160, 320),
                        ),
                        branch="cell", codebook_sizes=(80, 160, 320),
                    ),
                ),
                alpha=1.0, wt_adv_batch=150.0,
            ),
        ),
    },
    "dualvq+wide+rvq-both-3level-20-40-80+decoder-cov+adv+enc-deeper+mmb0-1b_smb1-1b_1p": {
        "description": (
            "v5 spine + RVQ-3-level scaled DOWN symmetrically: "
            "(30, 60, 120) -> (20, 40, 80). 0.3x effective capacity "
            "(216 k -> 64 k). Tests whether the model actually needs "
            "the 216 k codes or if a tighter bottleneck does just as "
            "well -- a meaningful ablation for the paper since smaller "
            "codebooks have lower memory + faster inference."
        ),
        "patches": [
            "+wide", "+rvq(branch=both, levels=[20, 40, 80])",
            "+decoder_covariate", "+adversarial_batch(alpha=1.0, wt=150.0)",
            "+enc-deeper",
        ],
        "build": lambda: _patch_dual_encoder_deeper(
            _patch_dual_adversarial(
                _patch_dual_decoder_covariate(
                    _patch_dual_rvq(
                        _patch_dual_rvq(
                            _patch_dual_wide(_BD()),
                            branch="niche", codebook_sizes=(20, 40, 80),
                        ),
                        branch="cell", codebook_sizes=(20, 40, 80),
                    ),
                ),
                alpha=1.0, wt_adv_batch=150.0,
            ),
        ),
    },
    "dualvq+wide+rvq-both-3level+decoder-cov+adv+enc-deeper+mlp-h256+mmb0-1b_smb1-1b_1p": {
        "description": (
            "v5 spine + NARROWER MLPs. Encoder hidden widths shrink "
            "[400, 400, 256] -> [256, 256, 256]; cell + niche decoders "
            "[400, 400] -> [256, 256]. Latent / codebook embedding dim "
            "stays at 256. Tests whether `+enc-deeper`'s benefit comes "
            "from depth alone or also from the wider intermediate "
            "layers."
        ),
        "patches": [
            "+wide", "+rvq(branch=both, levels=[30, 60, 120])",
            "+decoder_covariate", "+adversarial_batch(alpha=1.0, wt=150.0)",
            "+enc-deeper", "+mlp-h256(enc=[256,256,256], dec=[256,256])",
        ],
        "build": lambda: _patch_dual_mlp_width(
            _patch_dual_encoder_deeper(
                _patch_dual_adversarial(
                    _patch_dual_decoder_covariate(
                        _patch_dual_rvq(
                            _patch_dual_rvq(
                                _patch_dual_wide(_BD()),
                                branch="niche", codebook_sizes=(30, 60, 120),
                            ),
                            branch="cell", codebook_sizes=(30, 60, 120),
                        ),
                    ),
                    alpha=1.0, wt_adv_batch=150.0,
                ),
            ),
            encoder_hidden=[256, 256, 256],
            decoder_hidden=[256, 256],
        ),
    },
    "dualvq+wide+rvq-both-3level+decoder-cov+adv+enc-deeper+mlp-h512+mmb0-1b_smb1-1b_1p": {
        "description": (
            "v5 spine + WIDER MLPs. Encoder hidden widths grow "
            "[400, 400, 256] -> [512, 512, 256]; cell + niche decoders "
            "[400, 400] -> [512, 512]. Latent / codebook embedding dim "
            "stays at 256. Tests whether the v4 `+enc-deeper` win "
            "saturates at hidden=400 or scales further with width."
        ),
        "patches": [
            "+wide", "+rvq(branch=both, levels=[30, 60, 120])",
            "+decoder_covariate", "+adversarial_batch(alpha=1.0, wt=150.0)",
            "+enc-deeper", "+mlp-h512(enc=[512,512,256], dec=[512,512])",
        ],
        "build": lambda: _patch_dual_mlp_width(
            _patch_dual_encoder_deeper(
                _patch_dual_adversarial(
                    _patch_dual_decoder_covariate(
                        _patch_dual_rvq(
                            _patch_dual_rvq(
                                _patch_dual_wide(_BD()),
                                branch="niche", codebook_sizes=(30, 60, 120),
                            ),
                            branch="cell", codebook_sizes=(30, 60, 120),
                        ),
                    ),
                    alpha=1.0, wt_adv_batch=150.0,
                ),
            ),
            encoder_hidden=[512, 512, 256],
            decoder_hidden=[512, 512],
        ),
    },
    "dualvq+wide+cvq-both-3level-30-10-5+decoder-cov+adv+enc-deeper+mmb0-1b_smb1-1b_1p": {
        "description": (
            "v5 architectural alternative: 3-level Conditional (tree) "
            "VQ instead of Residual VQ, at matched depth. K1=30 macro "
            "buckets, K2=10 sub-children per bucket, K3=5 leaves per "
            "(l1, l2) pair = 1500 distinct codes. Tree partitioning "
            "produces a DISJOINT bucket structure (each cell hits "
            "exactly one of K1*K2*K3 leaves) versus RVQ's residual "
            "decomposition (additive contributions from each level). "
            "Useful for downstream interpretability where leaf bucket "
            "membership is what matters."
        ),
        "patches": [
            "+wide", "+cvq(branch=both, k1=30, k2=10, k3=5)",
            "+decoder_covariate", "+adversarial_batch(alpha=1.0, wt=150.0)",
            "+enc-deeper",
        ],
        "build": lambda: _patch_dual_encoder_deeper(
            _patch_dual_adversarial(
                _patch_dual_decoder_covariate(
                    _patch_dual_cvq(
                        _patch_dual_wide(_BD()),
                        branch="both", k1=30, k2=10, k3=5,
                    ),
                ),
                alpha=1.0, wt_adv_batch=150.0,
            ),
        ),
    },
    "dualvq+wide+cvq-both-3level-50-15-5+decoder-cov+adv+enc-deeper+mmb0-1b_smb1-1b_1p": {
        "description": (
            "v5 architectural alternative: 3-level Conditional VQ "
            "with larger fan-out. K1=50 macro x K2=15 sub-children x "
            "K3=5 leaves = 3 750 distinct codes. A/B against the "
            "smaller `+cvq-both-3level-30-10-5` variant to scan the "
            "tree-VQ capacity-vs-structure trade-off."
        ),
        "patches": [
            "+wide", "+cvq(branch=both, k1=50, k2=15, k3=5)",
            "+decoder_covariate", "+adversarial_batch(alpha=1.0, wt=150.0)",
            "+enc-deeper",
        ],
        "build": lambda: _patch_dual_encoder_deeper(
            _patch_dual_adversarial(
                _patch_dual_decoder_covariate(
                    _patch_dual_cvq(
                        _patch_dual_wide(_BD()),
                        branch="both", k1=50, k2=15, k3=5,
                    ),
                ),
                alpha=1.0, wt_adv_batch=150.0,
            ),
        ),
    },
    # ========================================================================
    # mmb0-1b_smb1-1b_1p ablations v7 — push cell-token NMI on cell_type
    # by varying the CELL codebook structure while NICHE is held fixed at
    # 3-level RVQ (30, 60, 120). All variants use a fixed level-1 size of
    # 30 (matches niche level 1), and explicitly DROP the 10-epoch
    # adversarial warmup that the v6/v6b families relied on — empirically
    # the post-warmup loss spiked dramatically as the late-arriving
    # adversary tried to undo features the encoder committed during
    # warmup. v7 sticks with adv-from-epoch-0 (alpha=1.0, wt=150.0) and
    # asks: with a fixed niche codebook, what cell-side codebook
    # configuration maximises cell→cell_type NMI?
    #
    # Spine (all 8 share):
    #   +wide (2-layer GNN + sampler [8,8] + nbr_aggregation_hops=2)
    #   +rvq(niche, levels=[30, 60, 120])     # 3-level RVQ, fixed
    #   +decoder-cov                           # NicheCompass-style
    #   +adv(alpha=1.0, wt=150.0)              # NO warmup
    #   +enc-deeper                            # mlp=[400,400,256]
    #
    # Cell codebook varies across:
    #   1-level VQ:   k=30 / k=60 / k=200       (3 variants)
    #   2-level RVQ:  (30,60) / (30,90) / (30,200)  (3 variants)
    #   3-level RVQ:  (30,60,120) / (30,90,270)     (2 variants)
    #
    # Group via `--all-dataset mmb0-1b_smb1-1b_1p-ablations-v7`.
    # ========================================================================
    "dualvq+wide+rvq-niche-30-60-120+vq-cell-30+decoder-cov+adv+enc-deeper+mmb0-1b_smb1-1b_1p": {
        "description": (
            "v7 — niche 3-level RVQ (30, 60, 120) + cell SINGLE-level "
            "VQ k=30. Tightest cell bottleneck (30 effective codes). "
            "Tests whether cell-type signal can be fully captured at "
            "level-1 granularity alone, matching the niche level 1. "
            "Empirical question: is the 30-code level-1 codebook "
            "sufficient for cell-type discrimination, or does extra "
            "residual capacity help?"
        ),
        "patches": [
            "+wide",
            "+rvq(branch=niche, levels=[30, 60, 120])",
            "+vq(branch=cell, k=30)",
            "+decoder_covariate",
            "+adversarial_batch(alpha=1.0, wt=150.0, no warmup)",
            "+enc-deeper",
        ],
        "build": lambda: _patch_dual_encoder_deeper(
            _patch_dual_adversarial(
                _patch_dual_decoder_covariate(
                    _patch_dual_vq(
                        _patch_dual_rvq(
                            _patch_dual_wide(_BD()),
                            branch="niche", codebook_sizes=(30, 60, 120),
                        ),
                        branch="cell", codebook_size=30,
                    ),
                ),
                alpha=1.0, wt_adv_batch=150.0,
            ),
        ),
    },
    "dualvq+wide+rvq-niche-30-60-120+vq-cell-60+decoder-cov+adv+enc-deeper+mmb0-1b_smb1-1b_1p": {
        "description": (
            "v7 — niche 3-level RVQ (30, 60, 120) + cell SINGLE-level "
            "VQ k=60. 2x the level-1 capacity. Tests whether widening "
            "the single-level cell codebook past 30 helps disentangle "
            "cell types that share level-1 niche structure."
        ),
        "patches": [
            "+wide",
            "+rvq(branch=niche, levels=[30, 60, 120])",
            "+vq(branch=cell, k=60)",
            "+decoder_covariate",
            "+adversarial_batch(alpha=1.0, wt=150.0, no warmup)",
            "+enc-deeper",
        ],
        "build": lambda: _patch_dual_encoder_deeper(
            _patch_dual_adversarial(
                _patch_dual_decoder_covariate(
                    _patch_dual_vq(
                        _patch_dual_rvq(
                            _patch_dual_wide(_BD()),
                            branch="niche", codebook_sizes=(30, 60, 120),
                        ),
                        branch="cell", codebook_size=60,
                    ),
                ),
                alpha=1.0, wt_adv_batch=150.0,
            ),
        ),
    },
    "dualvq+wide+rvq-niche-30-60-120+vq-cell-200+decoder-cov+adv+enc-deeper+mmb0-1b_smb1-1b_1p": {
        "description": (
            "v7 — niche 3-level RVQ (30, 60, 120) + cell SINGLE-level "
            "VQ k=200. Wide single-level cell codebook at the same "
            "scale as the legacy 2-level RVQ second tier (30, 200). "
            "A/B test against `+rvq-cell-30-200` to disentangle "
            "'capacity' from 'residual hierarchy' on the cell side."
        ),
        "patches": [
            "+wide",
            "+rvq(branch=niche, levels=[30, 60, 120])",
            "+vq(branch=cell, k=200)",
            "+decoder_covariate",
            "+adversarial_batch(alpha=1.0, wt=150.0, no warmup)",
            "+enc-deeper",
        ],
        "build": lambda: _patch_dual_encoder_deeper(
            _patch_dual_adversarial(
                _patch_dual_decoder_covariate(
                    _patch_dual_vq(
                        _patch_dual_rvq(
                            _patch_dual_wide(_BD()),
                            branch="niche", codebook_sizes=(30, 60, 120),
                        ),
                        branch="cell", codebook_size=200,
                    ),
                ),
                alpha=1.0, wt_adv_batch=150.0,
            ),
        ),
    },
    "dualvq+wide+rvq-niche-30-60-120+rvq-cell-30-60+decoder-cov+adv+enc-deeper+mmb0-1b_smb1-1b_1p": {
        "description": (
            "v7 — niche 3-level RVQ (30, 60, 120) + cell 2-level RVQ "
            "(30, 60) — i.e. cell shares niche's first two levels. "
            "Effective cell capacity 30 × 60 = 1 800 codes. Tightest "
            "2-level cell variant; tests minimum residual depth needed "
            "for cell-NMI."
        ),
        "patches": [
            "+wide",
            "+rvq(branch=niche, levels=[30, 60, 120])",
            "+rvq(branch=cell, levels=[30, 60])",
            "+decoder_covariate",
            "+adversarial_batch(alpha=1.0, wt=150.0, no warmup)",
            "+enc-deeper",
        ],
        "build": lambda: _patch_dual_encoder_deeper(
            _patch_dual_adversarial(
                _patch_dual_decoder_covariate(
                    _patch_dual_rvq(
                        _patch_dual_rvq(
                            _patch_dual_wide(_BD()),
                            branch="niche", codebook_sizes=(30, 60, 120),
                        ),
                        branch="cell", codebook_sizes=(30, 60),
                    ),
                ),
                alpha=1.0, wt_adv_batch=150.0,
            ),
        ),
    },
    "dualvq+wide+rvq-niche-30-60-120+rvq-cell-30-90+decoder-cov+adv+enc-deeper+mmb0-1b_smb1-1b_1p": {
        "description": (
            "v7 — niche 3-level RVQ (30, 60, 120) + cell 2-level RVQ "
            "(30, 90). Replicates the legacy v3 spine cell codebook "
            "while niche moves to the v5 3-level structure. Effective "
            "cell capacity 30 × 90 = 2 700 codes. Closest to the "
            "well-characterised v3 cell side; useful as the v7 "
            "reference point for the asymmetric (3-level niche, "
            "2-level cell) design."
        ),
        "patches": [
            "+wide",
            "+rvq(branch=niche, levels=[30, 60, 120])",
            "+rvq(branch=cell, levels=[30, 90])",
            "+decoder_covariate",
            "+adversarial_batch(alpha=1.0, wt=150.0, no warmup)",
            "+enc-deeper",
        ],
        "build": lambda: _patch_dual_encoder_deeper(
            _patch_dual_adversarial(
                _patch_dual_decoder_covariate(
                    _patch_dual_rvq(
                        _patch_dual_rvq(
                            _patch_dual_wide(_BD()),
                            branch="niche", codebook_sizes=(30, 60, 120),
                        ),
                        branch="cell", codebook_sizes=(30, 90),
                    ),
                ),
                alpha=1.0, wt_adv_batch=150.0,
            ),
        ),
    },
    "dualvq+wide+rvq-niche-30-60-120+rvq-cell-30-200+decoder-cov+adv+enc-deeper+mmb0-1b_smb1-1b_1p": {
        "description": (
            "v7 — niche 3-level RVQ (30, 60, 120) + cell 2-level RVQ "
            "(30, 200). Matches `_default_rvq_params` defaults for the "
            "cell side; widest 2-level cell variant. Effective cell "
            "capacity 30 × 200 = 6 000 codes. Tests whether a wider "
            "level-2 (vs (30, 90) / (30, 60)) helps cell-NMI when the "
            "level-1 is bottlenecked at 30."
        ),
        "patches": [
            "+wide",
            "+rvq(branch=niche, levels=[30, 60, 120])",
            "+rvq(branch=cell, levels=[30, 200])",
            "+decoder_covariate",
            "+adversarial_batch(alpha=1.0, wt=150.0, no warmup)",
            "+enc-deeper",
        ],
        "build": lambda: _patch_dual_encoder_deeper(
            _patch_dual_adversarial(
                _patch_dual_decoder_covariate(
                    _patch_dual_rvq(
                        _patch_dual_rvq(
                            _patch_dual_wide(_BD()),
                            branch="niche", codebook_sizes=(30, 60, 120),
                        ),
                        branch="cell", codebook_sizes=(30, 200),
                    ),
                ),
                alpha=1.0, wt_adv_batch=150.0,
            ),
        ),
    },
    "dualvq+wide+rvq-niche-30-60-120+rvq-cell-30-60-120+decoder-cov+adv+enc-deeper+mmb0-1b_smb1-1b_1p": {
        "description": (
            "v7 — niche AND cell at MATCHED 3-level RVQ (30, 60, 120). "
            "Same shape as the v5 anchor; included here as the v7 "
            "reference point so the asymmetric variants can be "
            "compared against a symmetric baseline run with the v7 "
            "harness (no warmup). Effective cell capacity 30 × 60 × "
            "120 = 216 k codes."
        ),
        "patches": [
            "+wide",
            "+rvq(branch=niche, levels=[30, 60, 120])",
            "+rvq(branch=cell, levels=[30, 60, 120])",
            "+decoder_covariate",
            "+adversarial_batch(alpha=1.0, wt=150.0, no warmup)",
            "+enc-deeper",
        ],
        "build": lambda: _patch_dual_encoder_deeper(
            _patch_dual_adversarial(
                _patch_dual_decoder_covariate(
                    _patch_dual_rvq(
                        _patch_dual_rvq(
                            _patch_dual_wide(_BD()),
                            branch="niche", codebook_sizes=(30, 60, 120),
                        ),
                        branch="cell", codebook_sizes=(30, 60, 120),
                    ),
                ),
                alpha=1.0, wt_adv_batch=150.0,
            ),
        ),
    },
    "dualvq+wide+rvq-niche-30-60-120+rvq-cell-30-90-270+decoder-cov+adv+enc-deeper+mmb0-1b_smb1-1b_1p": {
        "description": (
            "v7 — niche 3-level RVQ (30, 60, 120) + cell 3-level RVQ "
            "(30, 90, 270). Cell is widened relative to niche at every "
            "post-level-1 tier — niche stays at the v5 anchor for fair "
            "comparison. Effective cell capacity 30 × 90 × 270 = 729 k "
            "codes (~3.4x the v5 anchor cell). Tests whether the cell "
            "side benefits from MORE residual capacity than niche."
        ),
        "patches": [
            "+wide",
            "+rvq(branch=niche, levels=[30, 60, 120])",
            "+rvq(branch=cell, levels=[30, 90, 270])",
            "+decoder_covariate",
            "+adversarial_batch(alpha=1.0, wt=150.0, no warmup)",
            "+enc-deeper",
        ],
        "build": lambda: _patch_dual_encoder_deeper(
            _patch_dual_adversarial(
                _patch_dual_decoder_covariate(
                    _patch_dual_rvq(
                        _patch_dual_rvq(
                            _patch_dual_wide(_BD()),
                            branch="niche", codebook_sizes=(30, 60, 120),
                        ),
                        branch="cell", codebook_sizes=(30, 90, 270),
                    ),
                ),
                alpha=1.0, wt_adv_batch=150.0,
            ),
        ),
    },
    # ========================================================================
    # mmb0-1b_smb1-1b_1p ablations v8 — 8 variants split into two groups of 4:
    #
    # Spine (all 8 share — v5-anchor minus +enc-deeper to leave a clean
    # baseline for the encoder/decoder size sweep):
    #   +wide (2-layer GNN + sampler [8,8] + nbr_aggregation_hops=2)
    #   +rvq(branch=both, levels=[30, 60, 120])     # 3-level RVQ both sides
    #   +decoder-cov                                 # NicheCompass-style
    #   +adv(alpha=1.0, wt=150.0, no warmup)         # only Group B varies
    #
    # Group A — smaller encoder/decoder MLPs. The v5 default is enc=[400, 256]
    # / dec=[400, 400]. v8-A asks: how aggressively can we shrink the trunks
    # before cell/niche-NMI regress? Four shrink axes, picked to be
    # mechanistically distinct (depth vs width vs latent-dim vs full compact):
    #   A1. enc-shallow:   encoder 2-layer [400,256] -> 1-layer [256]
    #   A2. mlp-h256:      encoder [400,256] -> [256,256]; decoder [400,400] -> [256,256]
    #   A3. small-latent:  encoder [400,256] -> [400,128]; latent + GNN hidden -> 128
    #   A4. small:         scvi-style compact (hidden=128 in encoder + GNN + decoders;
    #                       drops +wide because +small re-sets GNN num_layers=1)
    #
    # Group B — MMD + adversarial batch-integration sweep. Replaces or
    # supplements the default adversary with MMD on the cell-token input
    # (z_mlp[:B]). Designed as a 2x2 grid so each variant can be read as
    # one axis of {MMD-only vs adv+MMD} x {weak (50) vs strong (200)}:
    #   B1. mmd-w50, NO adv         (MMD-only, weak)
    #   B2. mmd-w200, NO adv        (MMD-only, strong)
    #   B3. adv + mmd-w50           (combo, weak MMD)
    #   B4. adv + mmd-w200          (combo, strong MMD)
    #
    # Group via `--all-dataset mmb0-1b_smb1-1b_1p-ablations-v8`.
    # ========================================================================
    # ------ Group A: encoder/decoder size sweep (4 variants) ----------------
    "dualvq+wide+rvq-both-3level+decoder-cov+adv+enc-shallow+mmb0-1b_smb1-1b_1p": {
        "description": (
            "v8-A1 — v5 spine without +enc-deeper; instead the encoder "
            "trunk is SHRUNK from 2 layers [400, 256] to 1 layer [256]. "
            "Latent dim (= last layer = 256), GNN, decoders, and codebook "
            "all unchanged. Tests whether reducing encoder capacity "
            "(fewer hidden layers) helps cell-NMI by preventing the "
            "encoder from memorising batch-correlated features, without "
            "the adversarial-warmup pathology that bit v6/v6b."
        ),
        "patches": [
            "+wide",
            "+rvq(branch=both, levels=[30, 60, 120])",
            "+decoder_covariate",
            "+adversarial_batch(alpha=1.0, wt=150.0, no warmup)",
            "+enc-shallow(mlp=[256])",
        ],
        "build": lambda: _patch_dual_encoder_shallow(
            _patch_dual_adversarial(
                _patch_dual_decoder_covariate(
                    _patch_dual_rvq(
                        _patch_dual_rvq(
                            _patch_dual_wide(_BD()),
                            branch="niche", codebook_sizes=(30, 60, 120),
                        ),
                        branch="cell", codebook_sizes=(30, 60, 120),
                    ),
                ),
                alpha=1.0, wt_adv_batch=150.0,
            ),
        ),
    },
    "dualvq+wide+rvq-both-3level+decoder-cov+adv+mlp-h256+mmb0-1b_smb1-1b_1p": {
        "description": (
            "v8-A2 — v5 spine without +enc-deeper; encoder + decoder MLPs "
            "are NARROWED in lockstep. Encoder [400, 256] -> [256, 256] "
            "(latent dim preserved, intermediate width shrunk); both cell "
            "and niche decoders [400, 400] -> [256, 256]. Same depth as "
            "the v5 default, just narrower hidden widths. Tests the "
            "width axis of MLP capacity, holding depth fixed at 2."
        ),
        "patches": [
            "+wide",
            "+rvq(branch=both, levels=[30, 60, 120])",
            "+decoder_covariate",
            "+adversarial_batch(alpha=1.0, wt=150.0, no warmup)",
            "+mlp-h256(enc=[256, 256], dec=[256, 256])",
        ],
        "build": lambda: _patch_dual_mlp_width(
            _patch_dual_adversarial(
                _patch_dual_decoder_covariate(
                    _patch_dual_rvq(
                        _patch_dual_rvq(
                            _patch_dual_wide(_BD()),
                            branch="niche", codebook_sizes=(30, 60, 120),
                        ),
                        branch="cell", codebook_sizes=(30, 60, 120),
                    ),
                ),
                alpha=1.0, wt_adv_batch=150.0,
            ),
            encoder_hidden=[256, 256],
            decoder_hidden=[256, 256],
        ),
    },
    "dualvq+wide+rvq-both-3level+decoder-cov+adv+small-latent+mmb0-1b_smb1-1b_1p": {
        "description": (
            "v8-A3 — v5 spine + half-sized latent / codebook-embedding "
            "dim. Encoder last layer 256 -> 128, GNN hidden 256 -> 128, "
            "codebook embedding dim 256 -> 128. Encoder depth (2 layers) "
            "and decoder widths ([400, 400]) are unchanged. Tests the "
            "BOTTLENECK axis: fewer dims for codes to spread across "
            "forces each code's embedding to be more semantically "
            "concentrated. NicheCompass / scvi typically use 10-32; 128 "
            "still leaves headroom but halves the SQUINT default."
        ),
        "patches": [
            "+wide",
            "+rvq(branch=both, levels=[30, 60, 120])",
            "+decoder_covariate",
            "+adversarial_batch(alpha=1.0, wt=150.0, no warmup)",
            "+small-latent(latent_dim=128)",
        ],
        "build": lambda: _patch_dual_small_latent(
            _patch_dual_adversarial(
                _patch_dual_decoder_covariate(
                    _patch_dual_rvq(
                        _patch_dual_rvq(
                            _patch_dual_wide(_BD()),
                            branch="niche", codebook_sizes=(30, 60, 120),
                        ),
                        branch="cell", codebook_sizes=(30, 60, 120),
                    ),
                ),
                alpha=1.0, wt_adv_batch=150.0,
            ),
            latent_dim=128,
        ),
    },
    "dualvq+small+rvq-both-3level+decoder-cov+adv+mmb0-1b_smb1-1b_1p": {
        "description": (
            "v8-A4 — fully compact scvi-style architecture. Replaces "
            "+wide with +small(hidden=128): encoder MLP [128], GNN 1 "
            "layer hidden=128, both decoders [128]. Latent / codebook "
            "embedding dim become 128. Drops the deeper spatial context "
            "from +wide (sampler=[8], nbr_hops=1) — this is the most "
            "aggressive capacity ablation on the v5 spine and tests "
            "whether SQUINT's gains over scvi require the wider "
            "architecture or come from RVQ + decoder-cov + adv alone."
        ),
        "patches": [
            "+small(hidden=128)",
            "+rvq(branch=both, levels=[30, 60, 120])",
            "+decoder_covariate",
            "+adversarial_batch(alpha=1.0, wt=150.0, no warmup)",
        ],
        "build": lambda: _patch_dual_adversarial(
            _patch_dual_decoder_covariate(
                _patch_dual_rvq(
                    _patch_dual_rvq(
                        _patch_dual_small(_BD()),
                        branch="niche", codebook_sizes=(30, 60, 120),
                    ),
                    branch="cell", codebook_sizes=(30, 60, 120),
                ),
            ),
            alpha=1.0, wt_adv_batch=150.0,
        ),
    },
    # ------ Group B: MMD + adversarial batch-integration sweep (4 variants) -
    "dualvq+wide+rvq-both-3level+decoder-cov+mmd-w50+mmb0-1b_smb1-1b_1p": {
        "description": (
            "v8-B1 — REPLACE adversary with MMD (λ=50, n_sub=512), no "
            "warmup. MMD-only weak corner of the 2x2 mmd-x-adv grid. "
            "MMD applies a deterministic distribution-matching pressure "
            "on the cell-token input every step (no min-max instability "
            "from the GRL). Tests whether weak MMD alone is enough to "
            "integrate batches at this codebook structure."
        ),
        "patches": [
            "+wide",
            "+rvq(branch=both, levels=[30, 60, 120])",
            "+decoder_covariate",
            "+mmd_batch(wt=50, n_sub=512)",
        ],
        "build": lambda: _patch_dual_mmd_batch(
            _patch_dual_decoder_covariate(
                _patch_dual_rvq(
                    _patch_dual_rvq(
                        _patch_dual_wide(_BD()),
                        branch="niche", codebook_sizes=(30, 60, 120),
                    ),
                    branch="cell", codebook_sizes=(30, 60, 120),
                ),
            ),
            wt_mmd_batch=50.0, n_sub=512,
        ),
    },
    "dualvq+wide+rvq-both-3level+decoder-cov+mmd-w200+mmb0-1b_smb1-1b_1p": {
        "description": (
            "v8-B2 — REPLACE adversary with MMD (λ=200, n_sub=512), no "
            "warmup. MMD-only strong corner of the 2x2 grid. 4x B1; "
            "tests whether aggressive MMD makes integration a hard "
            "constraint without an adversary. Risk: kills cell-type "
            "signal if too strong, just as a heavy adv does."
        ),
        "patches": [
            "+wide",
            "+rvq(branch=both, levels=[30, 60, 120])",
            "+decoder_covariate",
            "+mmd_batch(wt=200, n_sub=512)",
        ],
        "build": lambda: _patch_dual_mmd_batch(
            _patch_dual_decoder_covariate(
                _patch_dual_rvq(
                    _patch_dual_rvq(
                        _patch_dual_wide(_BD()),
                        branch="niche", codebook_sizes=(30, 60, 120),
                    ),
                    branch="cell", codebook_sizes=(30, 60, 120),
                ),
            ),
            wt_mmd_batch=200.0, n_sub=512,
        ),
    },
    "dualvq+wide+rvq-both-3level+decoder-cov+adv+mmd-w50+mmb0-1b_smb1-1b_1p": {
        "description": (
            "v8-B3 — v5 spine adv (α=1.0, wt=150.0, no warmup) PLUS MMD "
            "(λ=50, n_sub=512) on the cell-token input. Weak combo "
            "corner of the 2x2 grid. Tests whether a light non-adv "
            "regulariser, working alongside the adv, gives smoother "
            "integration than adv alone — without the warmup pathology "
            "from v6/v6b. Both losses target z_mlp[:B] but via "
            "different mechanisms (CE-based min-max vs kernel MMD)."
        ),
        "patches": [
            "+wide",
            "+rvq(branch=both, levels=[30, 60, 120])",
            "+decoder_covariate",
            "+adversarial_batch(alpha=1.0, wt=150.0, no warmup)",
            "+mmd_batch(wt=50, n_sub=512)",
        ],
        "build": lambda: _patch_dual_mmd_batch(
            _patch_dual_adversarial(
                _patch_dual_decoder_covariate(
                    _patch_dual_rvq(
                        _patch_dual_rvq(
                            _patch_dual_wide(_BD()),
                            branch="niche", codebook_sizes=(30, 60, 120),
                        ),
                        branch="cell", codebook_sizes=(30, 60, 120),
                    ),
                ),
                alpha=1.0, wt_adv_batch=150.0,
            ),
            wt_mmd_batch=50.0, n_sub=512,
        ),
    },
    "dualvq+wide+rvq-both-3level+decoder-cov+adv+mmd-w200+mmb0-1b_smb1-1b_1p": {
        "description": (
            "v8-B4 — same as v8-B3 but MMD λ=200 (4x stronger). Strong "
            "combo corner of the 2x2 grid. Adv + strong MMD: tests "
            "whether stacking both batch-integration pressures gives "
            "the BEST iLISI without collapsing cell-NMI, or whether "
            "the encoder loses too much cell-type signal under "
            "combined pressure."
        ),
        "patches": [
            "+wide",
            "+rvq(branch=both, levels=[30, 60, 120])",
            "+decoder_covariate",
            "+adversarial_batch(alpha=1.0, wt=150.0, no warmup)",
            "+mmd_batch(wt=200, n_sub=512)",
        ],
        "build": lambda: _patch_dual_mmd_batch(
            _patch_dual_adversarial(
                _patch_dual_decoder_covariate(
                    _patch_dual_rvq(
                        _patch_dual_rvq(
                            _patch_dual_wide(_BD()),
                            branch="niche", codebook_sizes=(30, 60, 120),
                        ),
                        branch="cell", codebook_sizes=(30, 60, 120),
                    ),
                ),
                alpha=1.0, wt_adv_batch=150.0,
            ),
            wt_mmd_batch=200.0, n_sub=512,
        ),
    },
    # ========================================================================
    # mmb0-1b_smb1-1b_1p ablations v9 — 8 variants on a SMALLER + FASTER
    # architecture, organised as a 2 x 4 grid:
    #
    #   architectural size   ∈ {"compact"  (h32 + codebook (30, 10, 10)),
    #                           "medium"   (h64 + codebook (30, 20, 10))}
    #
    #   batch integration    ∈ {adv (wt=150),         # default
    #                           adv-w300,             # boosted adv
    #                           adv + mmd-w50,        # combo (adv + MMD)
    #                           mmd-w100 (no adv)}    # MMD only
    #
    # = 8 variants. The architectural size axis pairs encoder dim with
    # codebook width (compact uses both smaller; medium uses both larger)
    # — this is intentional because we want to read each "size scale" as
    # a single point rather than disentangling encoder dim from codebook
    # capacity within this sweep. The batch-integration axis is the
    # main thing being varied at each size scale.
    #
    # Shared spine (NO +wide so 1-hop GNN/sampler=[8]/nbr_hops=1):
    #   +small(hidden=N)                  # encoder MLP, GNN, decoders all -> N
    #   +rvq(branch=both, levels=[30, L1, L2])
    #   +decoder_covariate                # NicheCompass-shape batch slot
    #   (per-variant batch-integration loss combination)
    #
    # Effective codebook sizes:
    #   (30, 10, 10) -> 3,000 codes  (vs v5 anchor 216k)
    #   (30, 20, 10) -> 6,000 codes
    #
    # Motivation: smaller / shallower encoders helped niche-NMI in past
    # ablations but hurt iLISI. The four batch-integration strategies
    # span the space from default adv pressure to MMD-only — pairing
    # each strategy at TWO size scales lets us read "does compact
    # architecture work AND how much batch-integration pressure does
    # it need?" off the resulting 4x2 metric grid.
    #
    # Submit all 8 via:
    #   bash examples/submit_mmb_smb_ablations_v9.sh
    # ========================================================================
    # ------ compact size: hidden=32, codebook (30, 10, 10) ------------------
    "dualvq+small-h32+rvq-both-3level-30-10-10+decoder-cov+adv+mmb0-1b_smb1-1b_1p": {
        "description": (
            "v9 (compact size, default adv) — scvi-style compact "
            "architecture (hidden=32 in encoder MLP, GNN, decoders) + "
            "smallest 3-level RVQ (30, 10, 10) + default adv (wt=150). "
            "Floor of the sweep on every capacity axis; tests whether "
            "the very tightest config still produces useful niche / "
            "cell-type signal under default batch pressure."
        ),
        "patches": [
            "+small(hidden=32)",
            "+rvq(branch=both, levels=[30, 10, 10])",
            "+decoder_covariate",
            "+adversarial_batch(alpha=1.0, wt=150.0)",
        ],
        "build": lambda: _patch_dual_adversarial(
            _patch_dual_decoder_covariate(
                _patch_dual_rvq(
                    _patch_dual_rvq(
                        _patch_dual_small(_BD(), hidden=32),
                        branch="niche", codebook_sizes=(30, 10, 10),
                    ),
                    branch="cell", codebook_sizes=(30, 10, 10),
                ),
            ),
            alpha=1.0, wt_adv_batch=150.0,
        ),
    },
    "dualvq+small-h32+rvq-both-3level-30-10-10+decoder-cov+adv-w300+mmb0-1b_smb1-1b_1p": {
        "description": (
            "v9 (compact size, boosted adv) — hidden=32 + (30, 10, 10) "
            "with adversarial weight 2x the default (wt=300). Tests "
            "whether the compact encoder needs extra batch-integration "
            "pressure to recover iLISI."
        ),
        "patches": [
            "+small(hidden=32)",
            "+rvq(branch=both, levels=[30, 10, 10])",
            "+decoder_covariate",
            "+adversarial_batch(alpha=1.0, wt=300.0)",
        ],
        "build": lambda: _patch_dual_adversarial(
            _patch_dual_decoder_covariate(
                _patch_dual_rvq(
                    _patch_dual_rvq(
                        _patch_dual_small(_BD(), hidden=32),
                        branch="niche", codebook_sizes=(30, 10, 10),
                    ),
                    branch="cell", codebook_sizes=(30, 10, 10),
                ),
            ),
            alpha=1.0, wt_adv_batch=300.0,
        ),
    },
    "dualvq+small-h32+rvq-both-3level-30-10-10+decoder-cov+adv+mmd-w50+mmb0-1b_smb1-1b_1p": {
        "description": (
            "v9 (compact size, adv + MMD combo) — hidden=32 + "
            "(30, 10, 10) with default adv PLUS a light MMD on the "
            "cell-token input (lambda=50). MMD applies deterministic "
            "distribution-matching every step (no min-max instability), "
            "so the combo gives the encoder two complementary "
            "batch-mixing signals."
        ),
        "patches": [
            "+small(hidden=32)",
            "+rvq(branch=both, levels=[30, 10, 10])",
            "+decoder_covariate",
            "+adversarial_batch(alpha=1.0, wt=150.0)",
            "+mmd_batch(wt=50, n_sub=512)",
        ],
        "build": lambda: _patch_dual_mmd_batch(
            _patch_dual_adversarial(
                _patch_dual_decoder_covariate(
                    _patch_dual_rvq(
                        _patch_dual_rvq(
                            _patch_dual_small(_BD(), hidden=32),
                            branch="niche", codebook_sizes=(30, 10, 10),
                        ),
                        branch="cell", codebook_sizes=(30, 10, 10),
                    ),
                ),
                alpha=1.0, wt_adv_batch=150.0,
            ),
            wt_mmd_batch=50.0, n_sub=512,
        ),
    },
    "dualvq+small-h32+rvq-both-3level-30-10-10+decoder-cov+mmd-w100+mmb0-1b_smb1-1b_1p": {
        "description": (
            "v9 (compact size, MMD only) — hidden=32 + (30, 10, 10) with "
            "REPLACED adv: kernel MMD (lambda=100, n_sub=512) on the "
            "cell-token input and NO adversarial classifier. Tests "
            "whether the compact encoder can integrate batches via "
            "MMD alone — simpler architecture (no classifier to tune), "
            "no min-max game."
        ),
        "patches": [
            "+small(hidden=32)",
            "+rvq(branch=both, levels=[30, 10, 10])",
            "+decoder_covariate",
            "+mmd_batch(wt=100, n_sub=512)",
        ],
        "build": lambda: _patch_dual_mmd_batch(
            _patch_dual_decoder_covariate(
                _patch_dual_rvq(
                    _patch_dual_rvq(
                        _patch_dual_small(_BD(), hidden=32),
                        branch="niche", codebook_sizes=(30, 10, 10),
                    ),
                    branch="cell", codebook_sizes=(30, 10, 10),
                ),
            ),
            wt_mmd_batch=100.0, n_sub=512,
        ),
    },
    # ------ medium size: hidden=64, codebook (30, 20, 10) -------------------
    "dualvq+small-h64+rvq-both-3level-30-20-10+decoder-cov+adv+mmb0-1b_smb1-1b_1p": {
        "description": (
            "v9 (medium size, default adv) — hidden=64 (twice the compact "
            "encoder) + (30, 20, 10) (twice the effective codes) + "
            "default adv. Mid-size baseline; tests whether doubling "
            "both capacity axes recovers any NMI lost at the compact "
            "size under default batch-integration pressure."
        ),
        "patches": [
            "+small(hidden=64)",
            "+rvq(branch=both, levels=[30, 20, 10])",
            "+decoder_covariate",
            "+adversarial_batch(alpha=1.0, wt=150.0)",
        ],
        "build": lambda: _patch_dual_adversarial(
            _patch_dual_decoder_covariate(
                _patch_dual_rvq(
                    _patch_dual_rvq(
                        _patch_dual_small(_BD(), hidden=64),
                        branch="niche", codebook_sizes=(30, 20, 10),
                    ),
                    branch="cell", codebook_sizes=(30, 20, 10),
                ),
            ),
            alpha=1.0, wt_adv_batch=150.0,
        ),
    },
    "dualvq+small-h64+rvq-both-3level-30-20-10+decoder-cov+adv-w300+mmb0-1b_smb1-1b_1p": {
        "description": (
            "v9 (medium size, boosted adv) — hidden=64 + (30, 20, 10) + "
            "adv wt=300. Same boosted-pressure question as the compact "
            "h32 sibling but at higher capacity."
        ),
        "patches": [
            "+small(hidden=64)",
            "+rvq(branch=both, levels=[30, 20, 10])",
            "+decoder_covariate",
            "+adversarial_batch(alpha=1.0, wt=300.0)",
        ],
        "build": lambda: _patch_dual_adversarial(
            _patch_dual_decoder_covariate(
                _patch_dual_rvq(
                    _patch_dual_rvq(
                        _patch_dual_small(_BD(), hidden=64),
                        branch="niche", codebook_sizes=(30, 20, 10),
                    ),
                    branch="cell", codebook_sizes=(30, 20, 10),
                ),
            ),
            alpha=1.0, wt_adv_batch=300.0,
        ),
    },
    "dualvq+small-h64+rvq-both-3level-30-20-10+decoder-cov+adv+mmd-w50+mmb0-1b_smb1-1b_1p": {
        "description": (
            "v9 (medium size, adv + MMD combo) — hidden=64 + "
            "(30, 20, 10) with default adv plus MMD lambda=50. "
            "Combo-pressure question at the higher capacity."
        ),
        "patches": [
            "+small(hidden=64)",
            "+rvq(branch=both, levels=[30, 20, 10])",
            "+decoder_covariate",
            "+adversarial_batch(alpha=1.0, wt=150.0)",
            "+mmd_batch(wt=50, n_sub=512)",
        ],
        "build": lambda: _patch_dual_mmd_batch(
            _patch_dual_adversarial(
                _patch_dual_decoder_covariate(
                    _patch_dual_rvq(
                        _patch_dual_rvq(
                            _patch_dual_small(_BD(), hidden=64),
                            branch="niche", codebook_sizes=(30, 20, 10),
                        ),
                        branch="cell", codebook_sizes=(30, 20, 10),
                    ),
                ),
                alpha=1.0, wt_adv_batch=150.0,
            ),
            wt_mmd_batch=50.0, n_sub=512,
        ),
    },
    "dualvq+small-h64+rvq-both-3level-30-20-10+decoder-cov+mmd-w100+mmb0-1b_smb1-1b_1p": {
        "description": (
            "v9 (medium size, MMD only) — hidden=64 + (30, 20, 10) with "
            "MMD lambda=100 replacing the adversary. Simpler architecture "
            "(no classifier), no min-max game; same MMD-only question "
            "as the compact h32 sibling but at higher capacity."
        ),
        "patches": [
            "+small(hidden=64)",
            "+rvq(branch=both, levels=[30, 20, 10])",
            "+decoder_covariate",
            "+mmd_batch(wt=100, n_sub=512)",
        ],
        "build": lambda: _patch_dual_mmd_batch(
            _patch_dual_decoder_covariate(
                _patch_dual_rvq(
                    _patch_dual_rvq(
                        _patch_dual_small(_BD(), hidden=64),
                        branch="niche", codebook_sizes=(30, 20, 10),
                    ),
                    branch="cell", codebook_sizes=(30, 20, 10),
                ),
            ),
            wt_mmd_batch=100.0, n_sub=512,
        ),
    },
    # ========================================================================
    # mmb0-1b_smb1-1b_1p ablations v10 — "is spatial supervision hurting
    # cell-type NMI?" diagnostic. 4 variants on a fixed medium-size
    # architecture (h64 + codebook (30, 20, 10)), crossed over:
    #
    #   batch integration       ∈ { adv (wt=150),  mmd-w100 (no adv) }
    #   spatial-loss treatment  ∈ { +no-spatial   (drops nbr-NB + niche-commit + adj-BCE),
    #                                +no-adj        (keeps nbr-NB + niche-commit;
    #                                                drops ONLY the adjacency BCE) }
    #
    # = 2 x 2 = 4 variants. Architectural size is held constant so the
    # diagnostic isolates the effect of LOSSES (not capacity) on
    # cell-type NMI. h64 + (30,20,10) was chosen as a balance: small
    # enough to be fast, large enough not to be the bottleneck.
    #
    # Read of the 4 outcomes:
    #   - +no-adj alone recovers cell-NMI -> adjacency BCE is the
    #     specific culprit; niche NB recon is fine.
    #   - Only +no-spatial recovers -> all spatial supervision contributes;
    #     the full two-stage train (cell first, then frozen-cell +
    #     spatial) is the right next step.
    #   - Neither recovers -> bottleneck is elsewhere (codebook structure,
    #     encoder capacity, adversary calibration). Don't build two-stage.
    #
    # Submit all 4 via:
    #   bash examples/submit_mmb_smb_ablations_v10.sh
    # ========================================================================
    "dualvq+small-h64+rvq-both-3level-30-20-10+decoder-cov+adv+no-spatial+mmb0-1b_smb1-1b_1p": {
        "description": (
            "v10 — medium (h64 + (30,20,10)) + default adv + NO spatial "
            "losses. Trains as a pure cell-branch VQ-VAE + adversarial "
            "batch integration, matching scVI / Harmony training shape "
            "on the cell branch. Tests whether spatial supervision is "
            "pulling cell-NMI down."
        ),
        "patches": [
            "+small(hidden=64)",
            "+rvq(branch=both, levels=[30, 20, 10])",
            "+decoder_covariate",
            "+adversarial_batch(alpha=1.0, wt=150.0)",
            "+no-spatial",
        ],
        "build": lambda: _patch_dual_no_spatial(
            _patch_dual_adversarial(
                _patch_dual_decoder_covariate(
                    _patch_dual_rvq(
                        _patch_dual_rvq(
                            _patch_dual_small(_BD(), hidden=64),
                            branch="niche", codebook_sizes=(30, 20, 10),
                        ),
                        branch="cell", codebook_sizes=(30, 20, 10),
                    ),
                ),
                alpha=1.0, wt_adv_batch=150.0,
            ),
        ),
    },
    "dualvq+small-h64+rvq-both-3level-30-20-10+decoder-cov+mmd-w100+no-spatial+mmb0-1b_smb1-1b_1p": {
        "description": (
            "v10 — medium (h64 + (30,20,10)) + MMD-only (wt=100) + NO "
            "spatial losses. Same diagnostic as the +adv sibling but with "
            "MMD instead (no adversarial min-max game)."
        ),
        "patches": [
            "+small(hidden=64)",
            "+rvq(branch=both, levels=[30, 20, 10])",
            "+decoder_covariate",
            "+mmd_batch(wt=100, n_sub=512)",
            "+no-spatial",
        ],
        "build": lambda: _patch_dual_no_spatial(
            _patch_dual_mmd_batch(
                _patch_dual_decoder_covariate(
                    _patch_dual_rvq(
                        _patch_dual_rvq(
                            _patch_dual_small(_BD(), hidden=64),
                            branch="niche", codebook_sizes=(30, 20, 10),
                        ),
                        branch="cell", codebook_sizes=(30, 20, 10),
                    ),
                ),
                wt_mmd_batch=100.0, n_sub=512,
            ),
        ),
    },
    # ------ v10 "no-adj only" variants: keeps niche NB recon + niche
    # commit, drops ONLY the cosine adjacency BCE. Tests whether the
    # adjacency loss specifically is the cell-NMI drag, vs `+no-spatial`
    # which drops every spatial supervision term at once. -------------------
    "dualvq+small-h64+rvq-both-3level-30-20-10+decoder-cov+adv+no-adj+mmb0-1b_smb1-1b_1p": {
        "description": (
            "v10 — medium (h64 + (30,20,10)) + default adv with the "
            "ADJACENCY BCE dropped. Niche NB recon + niche VQ commit "
            "are kept, so the niche branch still receives gradient — "
            "but the encoder no longer has the cosine-adjacency loss "
            "pulling z_q_niche toward local spatial topology. Partners "
            "with the `+no-spatial` sibling: if `+no-adj` alone recovers "
            "cell-NMI, adjacency is the specific culprit; if only "
            "`+no-spatial` recovers, all spatial supervision contributes."
        ),
        "patches": [
            "+small(hidden=64)",
            "+rvq(branch=both, levels=[30, 20, 10])",
            "+decoder_covariate",
            "+adversarial_batch(alpha=1.0, wt=150.0)",
            "+no-adj",
        ],
        "build": lambda: _patch_dual_no_adj(
            _patch_dual_adversarial(
                _patch_dual_decoder_covariate(
                    _patch_dual_rvq(
                        _patch_dual_rvq(
                            _patch_dual_small(_BD(), hidden=64),
                            branch="niche", codebook_sizes=(30, 20, 10),
                        ),
                        branch="cell", codebook_sizes=(30, 20, 10),
                    ),
                ),
                alpha=1.0, wt_adv_batch=150.0,
            ),
        ),
    },
    "dualvq+small-h64+rvq-both-3level-30-20-10+decoder-cov+mmd-w100+no-adj+mmb0-1b_smb1-1b_1p": {
        "description": (
            "v10 — medium (h64 + (30,20,10)) + MMD-only (wt=100) with the "
            "adjacency BCE dropped. Niche NB recon + niche commit kept."
        ),
        "patches": [
            "+small(hidden=64)",
            "+rvq(branch=both, levels=[30, 20, 10])",
            "+decoder_covariate",
            "+mmd_batch(wt=100, n_sub=512)",
            "+no-adj",
        ],
        "build": lambda: _patch_dual_no_adj(
            _patch_dual_mmd_batch(
                _patch_dual_decoder_covariate(
                    _patch_dual_rvq(
                        _patch_dual_rvq(
                            _patch_dual_small(_BD(), hidden=64),
                            branch="niche", codebook_sizes=(30, 20, 10),
                        ),
                        branch="cell", codebook_sizes=(30, 20, 10),
                    ),
                ),
                wt_mmd_batch=100.0, n_sub=512,
            ),
        ),
    },
    # ========================================================================
    # mmb0-1b_smb1-1b_1p held-out region (downstream gene-reconstruction
    # task, NicheCompass / MLGenX-style). Both batches stay in training
    # but a contiguous patch in each is masked out cell-wise; Pearson is
    # then reported only on the held-out cells via
    # `compute_inference_metrics.py`'s `data_split == "test"` rows.
    # See `_patch_holdout_regions` for the default region geometry; use
    # `examples/plot_holdout_regions.py` to visualise / iterate on it.
    # ========================================================================
    "dualvq+rvq-both+decoder-cov+adv+region-holdout+mmb0-1b_smb1-1b_1p": {
        "description": (
            "1-layer-spine baseline (RVQ-both + decoder-cov + adv) WITH "
            "per-batch held-out spatial patches: a 25% x 25% box near "
            "the upper-left of STARmap+ batch15 and a 25% x 25% box near "
            "the lower-right of MERFISH batch82. Both sections stay in "
            "training; the held-out cells are masked out cell-wise and "
            "are tagged `data_split=\"test\"` so post-inference Pearson "
            "isolates the gene-reconstruction task from the trained-on "
            "cells."
        ),
        "patches": [
            "+rvq(branch=both, levels=[30, 90])",
            "+decoder_covariate",
            "+adversarial_batch(alpha=1.0, wt=150.0)",
            "+region_holdout(test_regions={15: 25%x25% upper-left, "
            "82: 25%x25% lower-right})",
        ],
        "build": lambda: _patch_holdout_regions(
            _patch_dual_adversarial(
                _patch_dual_decoder_covariate(
                    _patch_dual_rvq(
                        _patch_dual_rvq(_BD(),
                            branch="niche", codebook_sizes=(30, 90)),
                        branch="cell", codebook_sizes=(30, 90),
                    ),
                ),
                alpha=1.0, wt_adv_batch=150.0,
            ),
        ),
    },
    # ---- Reconstruction-mode ablations (all other components fixed) --------
    "recon-cell": {
        "description": (
            "Baseline: per-cell NB reconstruction only. No neighbourhood "
            "loss, no FiLM, no masking, no adjacency. recon_mode='cell'."
        ),
        "patches": ["recon_mode=cell"],
        "build": lambda: _patch_recon_cell(_B()),
    },
    "recon-nbr": {
        "description": (
            "Neighbourhood-only NB reconstruction. Per-cell counts are NOT "
            "reconstructed; the NB target is the 1-hop neighbourhood mean "
            "(including self-loop). recon_mode='nbr'. Tests whether a purely "
            "niche-oriented signal produces spatially coherent codes."
        ),
        "patches": ["recon_mode=nbr"],
        "build": lambda: _patch_recon_nbr(_B()),
    },
    "recon-both": {
        "description": (
            "Per-cell + neighbourhood NB reconstruction simultaneously. Two "
            "separate loss dispatches: nb_attribute_reconstruction_loss (cell) "
            "and nb_attribute_reconstruction_loss_nbr (nbr). recon_mode='both'. "
            "Tests the additive contribution of the neighbourhood signal."
        ),
        "patches": ["recon_mode=both", "+nb_attribute_reconstruction_loss_nbr"],
        "build": lambda: _patch_recon_both(_B()),
    },
    # ---- Adjacency ablations (cell recon base) ----------------------------
    "recon-cell+adj": {
        "description": (
            "Per-cell NB + BCE adjacency reconstruction. Tests whether "
            "predicting the 8-NN spatial graph connectivity improves "
            "niche-aware code structure."
        ),
        "patches": ["recon_mode=cell", "+bce_adjacency_reconstruction_loss"],
        "build": lambda: _patch_adj(_patch_recon_cell(_B())),
    },
    "recon-nbr+adj": {
        "description": (
            "Neighbourhood NB + BCE adjacency reconstruction. "
            "Both losses drive the latent toward spatial structure."
        ),
        "patches": ["recon_mode=nbr", "+bce_adjacency_reconstruction_loss"],
        "build": lambda: _patch_adj(_patch_recon_nbr(_B())),
    },
    "recon-both+adj": {
        "description": (
            "Per-cell + neighbourhood NB + BCE adjacency reconstruction. "
            "Maximum spatial supervision without FiLM or masking."
        ),
        "patches": ["recon_mode=both", "+nb_attribute_reconstruction_loss_nbr",
                    "+bce_adjacency_reconstruction_loss"],
        "build": lambda: _patch_adj(_patch_recon_both(_B())),
    },
    # ---- FiLM ablation (cell recon base) ----------------------------------
    "recon-cell+film": {
        "description": (
            "Per-cell NB + FiLM conditioning on (rbf_distances, cell_batch_id). "
            "Tests batch-correction and spatial-scale conditioning without "
            "neighbourhood or adjacency losses."
        ),
        "patches": ["recon_mode=cell", "+film(rbf_distances, cell_batch_id)"],
        "build": lambda: _patch_film(_patch_recon_cell(_B())),
    },
    # ---- Masking ablation (cell recon base) -------------------------------
    "recon-cell+mask": {
        "description": (
            "Per-cell NB + MAE-style gene masking (learnable_parameter, "
            "0.2->0.6 over 5 warmup epochs). Forces the encoder to use "
            "neighbourhood context rather than copying the input."
        ),
        "patches": ["recon_mode=cell", "+masking(base=0.2, final=0.6, warmup=5)"],
        "build": lambda: _patch_masking(_patch_recon_cell(_B())),
    },
    # ---- Multi-head VQ ablation (cell recon base) -------------------------
    "recon-cell+mhvq": {
        "description": (
            "Per-cell NB + multi-head VQ (heads=10, codebook_size=5000). "
            "Tests the capacity gain of a larger, multi-head codebook while "
            "holding all other components fixed."
        ),
        "patches": ["recon_mode=cell", "+mhvq(heads=10, codebook_size=5000)"],
        "build": lambda: _patch_multihead_vq(_patch_recon_cell(_B()),
                                             heads=10, codebook_size=5000),
    },
    # ========================================================================
    # RVQ ablations — same axes as the standard-VQ variants above, with the
    # codebook swapped for ResidualVQ_Squint (levels=[30, 200]).
    # ========================================================================
    "recon-cell+rvq": {
        "description": (
            "Per-cell NB + Residual VQ (levels=[30, 200]). Tests whether the "
            "RVQ capacity bump alone improves per-cell reconstruction without "
            "any neighbourhood / adjacency / FiLM / masking signal."
        ),
        "patches": ["recon_mode=cell", "+rvq(levels=[30, 200])"],
        "build": lambda: _patch_rvq(_patch_recon_cell(_B())),
    },
    "recon-nbr+rvq": {
        "description": (
            "Neighbourhood-only NB + RVQ (levels=[30, 200]). Tests RVQ on the "
            "purely niche-oriented objective."
        ),
        "patches": ["recon_mode=nbr", "+rvq(levels=[30, 200])"],
        "build": lambda: _patch_rvq(_patch_recon_nbr(_B())),
    },
    "recon-cell+rvq+adj": {
        "description": (
            "Per-cell NB + RVQ + BCE adjacency reconstruction. RVQ + spatial "
            "graph supervision."
        ),
        "patches": ["recon_mode=cell", "+rvq(levels=[30, 200])",
                    "+bce_adjacency_reconstruction_loss"],
        "build": lambda: _patch_adj(_patch_rvq(_patch_recon_cell(_B()))),
    },
    "recon-nbr+rvq+adj": {
        "description": (
            "Neighbourhood NB + RVQ + BCE adjacency reconstruction."
        ),
        "patches": ["recon_mode=nbr", "+rvq(levels=[30, 200])",
                    "+bce_adjacency_reconstruction_loss"],
        "build": lambda: _patch_adj(_patch_rvq(_patch_recon_nbr(_B()))),
    },
    "recon-both+rvq+adj": {
        "description": (
            "Cell + neighbourhood NB + RVQ + BCE adjacency reconstruction. "
            "Maximum spatial supervision with RVQ capacity."
        ),
        "patches": ["recon_mode=both", "+nb_attribute_reconstruction_loss_nbr",
                    "+rvq(levels=[30, 200])", "+bce_adjacency_reconstruction_loss"],
        "build": lambda: _patch_adj(_patch_rvq(_patch_recon_both(_B()))),
    },
    "recon-cell+rvq+film": {
        "description": (
            "Per-cell NB + RVQ + FiLM conditioning on (rbf_distances, "
            "cell_batch_id). Tests RVQ + batch/spatial conditioning."
        ),
        "patches": ["recon_mode=cell", "+rvq(levels=[30, 200])",
                    "+film(rbf_distances, cell_batch_id)"],
        "build": lambda: _patch_film(_patch_rvq(_patch_recon_cell(_B()))),
    },
    "recon-cell+rvq+mask": {
        "description": (
            "Per-cell NB + RVQ + MAE-style gene masking (0.2->0.6 over 5 "
            "warmup epochs). Tests RVQ + forced neighbourhood-aware encoding."
        ),
        "patches": ["recon_mode=cell", "+rvq(levels=[30, 200])",
                    "+masking(base=0.2, final=0.6, warmup=5)"],
        "build": lambda: _patch_masking(_patch_rvq(_patch_recon_cell(_B()))),
    },
    # ========================================================================
    # CVQ ablations — same axes again, with the codebook swapped for the
    # tree-structured ConditionalVQ (K1=30, K2=10).
    # ========================================================================
    "recon-cell+cvq": {
        "description": (
            "Per-cell NB + Conditional / Tree VQ (K1=30, K2=10). Tests the "
            "tree codebook structure on the per-cell objective."
        ),
        "patches": ["recon_mode=cell", "+cvq(k1=30, k2=10)"],
        "build": lambda: _patch_cvq(_patch_recon_cell(_B())),
    },
    "recon-nbr+cvq": {
        "description": (
            "Neighbourhood-only NB + CVQ (K1=30, K2=10). Tests CVQ on the "
            "niche-only objective; expect strong level-1 spatial structure."
        ),
        "patches": ["recon_mode=nbr", "+cvq(k1=30, k2=10)"],
        "build": lambda: _patch_cvq(_patch_recon_nbr(_B())),
    },
    "recon-cell+cvq+adj": {
        "description": (
            "Per-cell NB + CVQ + BCE adjacency reconstruction."
        ),
        "patches": ["recon_mode=cell", "+cvq(k1=30, k2=10)",
                    "+bce_adjacency_reconstruction_loss"],
        "build": lambda: _patch_adj(_patch_cvq(_patch_recon_cell(_B()))),
    },
    "recon-nbr+cvq+adj": {
        "description": (
            "Neighbourhood NB + CVQ + BCE adjacency reconstruction."
        ),
        "patches": ["recon_mode=nbr", "+cvq(k1=30, k2=10)",
                    "+bce_adjacency_reconstruction_loss"],
        "build": lambda: _patch_adj(_patch_cvq(_patch_recon_nbr(_B()))),
    },
    "recon-both+cvq+adj": {
        "description": (
            "Cell + neighbourhood NB + CVQ + BCE adjacency reconstruction. "
            "Maximum spatial supervision with hierarchical tree codebook."
        ),
        "patches": ["recon_mode=both", "+nb_attribute_reconstruction_loss_nbr",
                    "+cvq(k1=30, k2=10)", "+bce_adjacency_reconstruction_loss"],
        "build": lambda: _patch_adj(_patch_cvq(_patch_recon_both(_B()))),
    },
    "recon-cell+cvq+film": {
        "description": (
            "Per-cell NB + CVQ + FiLM conditioning on (rbf_distances, "
            "cell_batch_id). Tests tree codebook + batch/spatial conditioning."
        ),
        "patches": ["recon_mode=cell", "+cvq(k1=30, k2=10)",
                    "+film(rbf_distances, cell_batch_id)"],
        "build": lambda: _patch_film(_patch_cvq(_patch_recon_cell(_B()))),
    },
    "recon-cell+cvq+mask": {
        "description": (
            "Per-cell NB + CVQ + MAE-style gene masking (0.2->0.6 over 5 "
            "warmup epochs). Tests tree codebook + masked-pretraining."
        ),
        "patches": ["recon_mode=cell", "+cvq(k1=30, k2=10)",
                    "+masking(base=0.2, final=0.6, warmup=5)"],
        "build": lambda: _patch_masking(_patch_cvq(_patch_recon_cell(_B()))),
    },
    # ---- Combined-loss capacity ablations (recon-both base) ---------------
    # These three share recon_mode='both' so the cell + neighbourhood NB
    # objectives are both active, isolating the effect of the VQ architecture.
    "recon-both+mhvq": {
        "description": (
            "Cell + neighbourhood NB + multi-head VQ (heads=10, codebook_size=5000). "
            "Tests whether a larger product-quantization codebook benefits the "
            "joint cell/niche objective."
        ),
        "patches": ["recon_mode=both", "+nb_attribute_reconstruction_loss_nbr",
                    "+mhvq(heads=10, codebook_size=5000)"],
        "build": lambda: _patch_multihead_vq(_patch_recon_both(_B()),
                                             heads=10, codebook_size=5000),
    },
    "recon-both+rvq": {
        "description": (
            "Cell + neighbourhood NB + Residual VQ (RQ-VAE; levels=[30, 200]). "
            "Each cell is encoded as a SUM of two codes from independent "
            "codebooks. Total expressivity 30*200=6000, with a coarse->fine "
            "interpretation (level 1 = macro niche, level 2 = global refinement)."
        ),
        "patches": ["recon_mode=both", "+nb_attribute_reconstruction_loss_nbr",
                    "+rvq(levels=[30, 200])"],
        "build": lambda: _patch_rvq(_patch_recon_both(_B()),
                                    codebook_sizes=(30, 200)),
    },
    "recon-both+cvq": {
        "description": (
            "Cell + neighbourhood NB + Conditional / Tree VQ (K1=30, K2=10). "
            "Each cell is routed to a coarse niche by level 1; conditional on "
            "that niche it is refined by a niche-specific level-2 codebook of "
            "10 codes. Total expressivity 30*10=300 with a strict tree "
            "interpretation (level-2 codes are NOT shared across buckets)."
        ),
        "patches": ["recon_mode=both", "+nb_attribute_reconstruction_loss_nbr",
                    "+cvq(k1=30, k2=10)"],
        "build": lambda: _patch_cvq(_patch_recon_both(_B()), k1=30, k2=10),
    },
    # ========================================================================
    # NicheCompass-style ablation matrix on the recon-nbr base.
    #
    # The full `recon-nbr+ncc` variant bundled three changes vs. plain
    # `recon-nbr`:
    #     (i)   adjacency BCE active           = "+adj" or "+adj-bypass"
    #     (ii)  adjacency input = z_q directly = "-bypass" suffix
    #     (iii) encoder GNN = GATv2            = "+gatv2"
    # The variants below isolate each change so we can attribute any
    # improvement (or absence of improvement) to the specific component.
    #
    # Note:  `recon-nbr+adj` already exists above (standard +adj with MLP
    # adjacency decoder, weight=1.0, GraphSAGE encoder). Together with the
    # three ablations below + the full `recon-nbr+ncc`, this gives the full
    # 2x3 matrix {SAGE, GATv2} x {no adj, adj-MLP, adj-bypass}.
    # ========================================================================
    "recon-nbr+adj-bypass": {
        "description": (
            "recon-nbr + NicheCompass-style adjacency BCE on z_q (raw "
            "inner-product formulation: edge_logit_ij = z_q_i^T z_q_j, "
            "bypassing the SQUINT adjacency_decoder MLP) at weight 1.0. "
            "Encoder is unchanged (GraphSAGE). This isolates the bypass-MLP "
            "change vs. `recon-nbr+adj` (which uses the same weight but "
            "keeps the adjacency_decoder MLP)."
        ),
        "patches": [
            "recon_mode=nbr",
            "+bce_adjacency_reconstruction_loss",
            "+nichecompass_adj(weight=1.0, bypass_adj_decoder=True)",
        ],
        "build": lambda: _patch_nichecompass_adj(
                            _patch_recon_nbr(_B()),
                            weight=1.0),
    },
    "recon-nbr+gatv2": {
        "description": (
            "recon-nbr with the encoder GNN switched from GraphSAGE to "
            "GATv2 (dynamic attention). No adjacency reconstruction loss. "
            "Isolates the GATv2 change vs. plain `recon-nbr`."
        ),
        "patches": ["recon_mode=nbr", "+gatv2_encoder"],
        "build": lambda: _patch_gatv2_encoder(_patch_recon_nbr(_B())),
    },
    "recon-nbr+gatv2+adj": {
        "description": (
            "recon-nbr + GATv2 encoder + standard SQUINT adjacency BCE "
            "(weight=1.0, MLP adjacency_decoder still on the gradient path). "
            "Tests GATv2 + standard adjacency — between `recon-nbr+gatv2` "
            "(GATv2 alone) and `recon-nbr+ncc` (GATv2 + bypass adj)."
        ),
        "patches": ["recon_mode=nbr", "+gatv2_encoder",
                    "+bce_adjacency_reconstruction_loss(weight=1.0)"],
        "build": lambda: _patch_adj(
                            _patch_gatv2_encoder(_patch_recon_nbr(_B())),
                            weight=1.0),
    },
    "recon-nbr+ncc": {
        "description": (
            "NicheCompass-shaped recon-nbr: GATv2 encoder + neighbourhood-only "
            "NB reconstruction + NicheCompass-style adjacency BCE on z_q "
            "(bypassing the SQUINT adjacency_decoder MLP) at weight 1.0. "
            "Tests the full bundle of NicheCompass-style changes."
        ),
        "patches": [
            "recon_mode=nbr",
            "+gatv2_encoder",
            "+bce_adjacency_reconstruction_loss",
            "+nichecompass_adj(weight=1.0, bypass_adj_decoder=True)",
        ],
        "build": lambda: _patch_nichecompass_adj(
                            _patch_gatv2_encoder(
                                _patch_recon_nbr(_B())),
                            weight=1.0),
    },
    # ========================================================================
    # VQNiche_Dual variants
    # ========================================================================
    # Two parallel codebooks: cell (post-MLP, pre-aggregation) and niche
    # (post-GNN). Cell decoder reconstructs per-cell counts; niche decoder
    # reconstructs neighbourhood-mean counts; cosine-similarity BCE on
    # z_q_niche enforces spatial coherence on the niche codebook.
    # ========================================================================
    "dualvq": {
        "description": (
            "VQNiche_Dual baseline: two single-codebook VQs (k=30 each), "
            "shared MLP trunk, GraphSAGE GNN feeding the niche branch. "
            "Joint losses: NB(cell) + NB(nbr) + commit_cell + commit_niche "
            "+ cosine-sim adjacency BCE on z_q_niche. All weights = 1.0."
        ),
        "patches": ["dualvq_baseline"],
        "build": lambda: _BD(),
    },
    "dualvq+rvq-niche": {
        "description": (
            "Dual VQ with RVQ on the niche branch only "
            "(levels=[30, 200]). Cell branch is plain VectorQuantize(k=30). "
            "Tests whether scaling capacity on the niche side helps niche "
            "discrimination without affecting the cell branch."
        ),
        "patches": ["+rvq(branch=niche, levels=[30, 200])"],
        "build": lambda: _patch_dual_rvq(_BD(), branch="niche", codebook_sizes=(30, 200)),
    },
    "dualvq+rvq-cell": {
        "description": (
            "Dual VQ with RVQ on the cell branch only "
            "(levels=[30, 200]). Niche branch is plain VectorQuantize(k=30). "
            "Tests whether the cell codebook benefits from extra capacity."
        ),
        "patches": ["+rvq(branch=cell, levels=[30, 200])"],
        "build": lambda: _patch_dual_rvq(_BD(), branch="cell", codebook_sizes=(30, 200)),
    },
    "dualvq+rvq-both": {
        "description": (
            "Dual VQ with RVQ on BOTH branches (levels=[30, 200] each). "
            "Maximum capacity per branch."
        ),
        "patches": ["+rvq(branch=both, levels=[30, 200])"],
        "build": lambda: _patch_dual_rvq(_BD(), branch="both", codebook_sizes=(30, 200)),
    },
    "dualvq+cvq-niche": {
        "description": (
            "Dual VQ with ConditionalVQ on the niche branch (K1=30, K2=10). "
            "Cell branch unchanged. Tree-structured niche codebook."
        ),
        "patches": ["+cvq(branch=niche, k1=30, k2=10)"],
        "build": lambda: _patch_dual_cvq(_BD(), branch="niche", k1=30, k2=10),
    },
    "dualvq+gatv2": {
        "description": (
            "Dual VQ with GATv2 GNN (instead of GraphSAGE). Tests whether "
            "attention-weighted aggregation makes the niche branch sharper."
        ),
        "patches": ["+gatv2_encoder"],
        "build": lambda: _patch_dual_gatv2(_BD()),
    },
    "dualvq+adj-mild": {
        "description": (
            "Dual VQ with adjacency BCE DOWN-weighted to 100 (vs. default 1000). "
            "Tests how much spatial coherence is sacrificed at 1/10 the "
            "default weight. Useful sanity check that the high default is "
            "actually doing useful work."
        ),
        "patches": ["+cosine_adj(weight=100)"],
        "build": lambda: _patch_dual_adj_weight(_BD(), weight=100.0),
    },
    "dualvq+adj-strong": {
        "description": (
            "Dual VQ with adjacency BCE UP-weighted to 2500 (vs. default 1000). "
            "Tests whether 2.5x default weight further improves niche-code "
            "spatial coherence. May sacrifice some NB Pearson."
        ),
        "patches": ["+cosine_adj(weight=2500)"],
        "build": lambda: _patch_dual_adj_weight(_BD(), weight=2500.0),
    },
    "dualvq+adj-extreme": {
        "description": (
            "Dual VQ with adjacency BCE UP-weighted to 5000 (vs. default 1000). "
            "Maximum spatial-coherence pressure. Expect noticeable NB "
            "Pearson degradation — useful only as an upper-bound run for "
            "what spatial coherence looks like when adjacency dominates."
        ),
        "patches": ["+cosine_adj(weight=5000)"],
        "build": lambda: _patch_dual_adj_weight(_BD(), weight=5000.0),
    },
    # ========================================================================
    # "Wide" dualvq variants — deeper GNN + deeper sampler + bigger codebooks
    # + 2-hop nbr aggregation. Recommended starting point for niche-focused
    # runs once the basic dualvq architecture is validated.
    # ========================================================================
    "dualvq+wide": {
        "description": (
            "Wide dual VQ: GNN num_layers=2 (sees 2-hop context), sampler "
            "[8, 8] (provides 2-hop neighbours), VQ k=50 per branch, niche "
            "target aggregated over 2-hop neighbourhood. The GNN/sampler/"
            "aggregation depths are all coupled so the receptive field is "
            "consistent end-to-end."
        ),
        "patches": [
            "+gnn_layers=2", "+sampler=[8,8]",
            "+codebook_size=50 (cell+niche)", "+nbr_aggregation_hops=2",
        ],
        "build": lambda: _patch_dual_wide(_BD()),
    },
    "dualvq+wide+rvq-niche-3level": {
        "description": (
            "Wide dual VQ + RVQ on the niche branch with THREE levels "
            "(K=[30, 90, 30], 81000 effective codes). Macro -> sub-niche "
            "-> fine-grained refinement. Cell branch is plain VQ k=50. "
            "Run only after `+rvq-niche` (2-level) results are confirmed "
            "good — adding a level adds ~K3=30 codebook vectors of params "
            "and a small per-step overhead, so worth a comparison."
        ),
        "patches": ["+wide", "+rvq(branch=niche, levels=[30, 90, 30])"],
        "build": lambda: _patch_dual_rvq(
            _patch_dual_wide(_BD()),
            branch="niche", codebook_sizes=(30, 90, 30),
        ),
    },
    "dualvq+wide+rvq-niche-larger": {
        "description": (
            "Wide dual VQ + RVQ on the niche branch with larger level-2 "
            "(K=[30, 200], 6000 effective codes). Cell branch is plain "
            "VQ k=50. Use if the level-2 utilization in `+rvq-niche` "
            "([30, 90]) saturates near 90/90."
        ),
        "patches": ["+wide", "+rvq(branch=niche, levels=[30, 200])"],
        "build": lambda: _patch_dual_rvq(
            _patch_dual_wide(_BD()),
            branch="niche", codebook_sizes=(30, 200),
        ),
    },
    "dualvq+wide+rvq-niche-finer-l1": {
        "description": (
            "Wide dual VQ + RVQ on the niche branch with finer level-1 "
            "(K=[50, 90], 4500 effective codes). Cell branch is plain "
            "VQ k=50. Use if the level-1 utilization in `+rvq-niche` "
            "([30, 90]) saturates near 30/30 — gives more macro-niche "
            "resolution at the same level-2 capacity."
        ),
        "patches": ["+wide", "+rvq(branch=niche, levels=[50, 90])"],
        "build": lambda: _patch_dual_rvq(
            _patch_dual_wide(_BD()),
            branch="niche", codebook_sizes=(50, 90),
        ),
    },
    "dualvq+wide+rvq-niche": {
        "description": (
            "Wide dual VQ + RVQ on the niche branch (levels=[30, 90], "
            "2700 effective niche codes). Cell branch is plain VQ k=50. "
            "Smaller-than-max capacity for first-pass testing — converges "
            "faster, less risk of underused level-2 codes, still 54x more "
            "effective codes than the K=50 baseline."
        ),
        "patches": [
            "+wide", "+rvq(branch=niche, levels=[30, 90])",
        ],
        "build": lambda: _patch_dual_rvq(
            _patch_dual_wide(_BD()),
            branch="niche", codebook_sizes=(30, 90),
        ),
    },
    "dualvq+wide+rvq-both": {
        "description": (
            "Wide dual VQ + RVQ on BOTH branches (levels=[30, 90] each, "
            "2700 effective codes per branch). Symmetric capacity for "
            "easy A/B against `+rvq-niche`. Use when cell-state granularity "
            "matters too. Bump to [50, 200] / [50, 100] later if the "
            "smaller sizes saturate."
        ),
        "patches": [
            "+wide", "+rvq(branch=niche, levels=[30, 90])",
            "+rvq(branch=cell, levels=[30, 90])",
        ],
        "build": lambda: _patch_dual_rvq(
            _patch_dual_rvq(
                _patch_dual_wide(_BD()),
                branch="niche", codebook_sizes=(30, 90),
            ),
            branch="cell", codebook_sizes=(30, 90),
        ),
    },
    "dualvq+wide+cvq-niche": {
        "description": (
            "Wide dual VQ + ConditionalVQ (tree) on the niche branch "
            "(K1=30 macro niches, K2=5 sub-niches per macro = 150 "
            "effective). Conservative tree size: ~100k / 30 ≈ 3300 cells "
            "per macro niche, then / 5 ≈ 660 cells per (macro, sub) pair "
            "— well above the threshold where level-2 EMA becomes noisy."
        ),
        "patches": [
            "+wide", "+cvq(branch=niche, k1=30, k2=5)",
        ],
        "build": lambda: _patch_dual_cvq(
            _patch_dual_wide(_BD()),
            branch="niche", k1=30, k2=5,
        ),
    },
    # ========================================================================
    # CosMx Lung holdout-replicate variants (dataset: chl59-8b_1p)
    # ========================================================================
    # Whole-replicate holdout: train batches go entirely to the train loader,
    # test batches go entirely to the val loader -> wandb logs `train_*`
    # metrics on the training replicates and `val_*` metrics on the held-out
    # replicate(s).
    "dualvq+wide+rvq-both+decoder-cov+chl59-8b_1p": {
        "description": (
            "chl59-8b_1p split + decoder-covariate batch correction. "
            "Train: adata_batch_id 0 + 1 + 4 + 5 + 6 + 7; Test: "
            "adata_batch_id 2 + 3. Decoder covariate concatenates "
            "per-cell batch one-hot to z_q before each decoder, freeing "
            "the codebook to be batch-invariant across replicates."
        ),
        "patches": [
            "+wide", "+rvq(branch=both, levels=[30, 90])",
            "+decoder_covariate", "+chl59-8b_1p(test_ids=[2,3])",
        ],
        "build": lambda: _patch_dual_chl59_lung5(
            _patch_dual_decoder_covariate(
                _patch_dual_rvq(
                    _patch_dual_rvq(
                        _patch_dual_wide(_BD()),
                        branch="niche", codebook_sizes=(30, 90),
                    ),
                    branch="cell", codebook_sizes=(30, 90),
                ),
            ),
            test_batch_idx=[2, 3],
        ),
    },
    "dualvq+wide+rvq-both+dec-film+chl59-8b_1p": {
        "description": (
            "chl59-8b_1p split + FiLM batch-correction INSIDE the "
            "cell + niche decoders (`apply_conditioning='in-MLP'`, "
            "condition=cell_batch_id). Train: adata_batch_id "
            "0 + 1 + 4 + 5 + 6 + 7; Test: adata_batch_id 2 + 3. Direct "
            "A/B against `+decoder-cov+chl59-8b_1p` (concat covariate) "
            "and `+decoder-cov+adv+chl59-8b_1p` (concat + adversary). "
            "FiLM is recommended over concat for the deeper non-linear "
            "MLPSoftmax decoders used here."
        ),
        "patches": [
            "+wide", "+rvq(branch=both, levels=[30, 90])",
            "+dec-film(in-MLP, cell_batch_id)",
            "+chl59-8b_1p(test_ids=[2,3])",
        ],
        "build": lambda: _patch_dual_chl59_lung5(
            _patch_dual_decoder_film(
                _patch_dual_rvq(
                    _patch_dual_rvq(
                        _patch_dual_wide(_BD()),
                        branch="niche", codebook_sizes=(30, 90),
                    ),
                    branch="cell", codebook_sizes=(30, 90),
                ),
            ),
            test_batch_idx=[2, 3],
        ),
    },
    "dualvq+wide+rvq-both+decoder-cov+adv+chl59-8b_1p": {
        "description": (
            "chl59-8b_1p split + decoder covariate + domain-"
            "adversarial batch-invariance head. Train: adata_batch_id "
            "0 + 1 + 4 + 5 + 6 + 7; Test: adata_batch_id 2 + 3. "
            "NicheCompass-shape recipe: covariate concat lets the "
            "decoder absorb per-batch gene patterns; adversarial GRL "
            "(applied to FULL z_mlp incl. sampled neighbours) actively "
            "pushes the encoder toward batch-invariance. wt_adv_batch="
            "150 calibrated against the ~150-nat NB reconstruction "
            "losses."
        ),
        "patches": [
            "+wide", "+rvq(branch=both, levels=[30, 90])",
            "+decoder_covariate", "+adversarial_batch(alpha=1.0, wt=150.0)",
            "+chl59-8b_1p(test_ids=[2,3])",
        ],
        "build": lambda: _patch_dual_chl59_lung5(
            _patch_dual_adversarial(
                _patch_dual_decoder_covariate(
                    _patch_dual_rvq(
                        _patch_dual_rvq(
                            _patch_dual_wide(_BD()),
                            branch="niche", codebook_sizes=(30, 90),
                        ),
                        branch="cell", codebook_sizes=(30, 90),
                    ),
                ),
                alpha=1.0, wt_adv_batch=150.0,
            ),
            test_batch_idx=[2, 3],
        ),
    },
    "dualvq+wide+rvq-both+adv+chl59-8b_1p": {
        "description": (
            "chl59-8b_1p split + adversarial-only batch correction "
            "(no decoder covariate, no FiLM). Train: adata_batch_id "
            "0 + 1 + 4 + 5 + 6 + 7; Test: adata_batch_id 2 + 3. Tests "
            "whether the GRL alone is enough — should not work as "
            "well as the +decoder-cov+adv combo since the decoder has "
            "no way to fit per-batch gene patterns."
        ),
        "patches": [
            "+wide", "+rvq(branch=both, levels=[30, 90])",
            "+adversarial_batch(alpha=1.0, wt=150.0)",
            "+chl59-8b_1p(test_ids=[2,3])",
        ],
        "build": lambda: _patch_dual_chl59_lung5(
            _patch_dual_adversarial(
                _patch_dual_rvq(
                    _patch_dual_rvq(
                        _patch_dual_wide(_BD()),
                        branch="niche", codebook_sizes=(30, 90),
                    ),
                    branch="cell", codebook_sizes=(30, 90),
                ),
                alpha=1.0, wt_adv_batch=150.0,
            ),
            test_batch_idx=[2, 3],
        ),
    },
    # ------------------------------------------------------------------------
    # chl59 port of the mmb-smb winner variant.
    # Same spine as `dualvq+rvq-both+decoder-cov+adv+enc-deeper+mmb0-1b_smb1-1b_1p`
    # (NO +wide -> 1-hop GNN; rvq-both with codebook sizes (30, 90);
    # decoder-covariate; adversarial-batch (alpha=1.0, wt=150.0); enc-deeper
    # MLP [400, 400, 256]) — only the dataset patch differs. Standard
    # chl59-8b_1p train/test split: train on adata_batch_id 0+1+4+5+6+7,
    # hold out 2+3 (replicate + new donor) for downstream evaluation.
    # ------------------------------------------------------------------------
    "dualvq+rvq-both+decoder-cov+adv+enc-deeper+chl59-8b_1p": {
        "description": (
            "chl59-8b_1p port of the mmb-smb winner variant "
            "(`dualvq+rvq-both+decoder-cov+adv+enc-deeper+mmb0-1b_smb1-1b_1p`). "
            "1-hop GNN spine + RVQ-both (30, 90) + decoder-covariate + "
            "adversarial-batch (alpha=1.0, wt=150.0) + deeper encoder MLP "
            "[400, 400, 256]. Train: adata_batch_id 0+1+4+5+6+7. "
            "Test: 2+3 (held-out replicate + new donor)."
        ),
        "patches": [
            "+rvq(branch=both, levels=[30, 90])",
            "+decoder_covariate",
            "+adversarial_batch(alpha=1.0, wt=150.0)",
            "+enc-deeper(mlp=[400, 400, 256])",
            "+chl59-8b_1p(test_ids=[2,3])",
        ],
        "build": lambda: _patch_dual_chl59_lung5(
            _patch_dual_encoder_deeper(
                _patch_dual_adversarial(
                    _patch_dual_decoder_covariate(
                        _patch_dual_rvq(
                            _patch_dual_rvq(_BD(),
                                branch="niche", codebook_sizes=(30, 90)),
                            branch="cell", codebook_sizes=(30, 90),
                        ),
                    ),
                    alpha=1.0, wt_adv_batch=150.0,
                ),
            ),
            test_batch_idx=[2, 3],
        ),
    },
    # ========================================================================
    # mmb20 cross-platform retrieval (dataset: mmb0-1b_smb1-20b_1p_shared_genes)
    # ========================================================================
    # Train on 19 STARmap+ sections (mouse CNS, batches 1..14 + 16..20);
    # hold out STARmap batch 15 (within-platform baseline; matches the
    # STARmap section held out in the smaller mmb-smb experiment) and
    # MERFISH batch 82 (cross-platform OOD test). Both holdouts are TRUE
    # held-outs — they never enter the training data graph; post-hoc
    # evaluation runs through predict + compute_inference_metrics on the
    # full silver dir.
    "dualvq+narrow+rvq-both+dec-film+adv+mmb0-1b_smb1-20b_1p": {
        "description": (
            "mmb0-1b_smb1-20b_1p (1 MERFISH + 20 STARmap, shared MERFISH "
            "gene panel) with FiLM-in-decoder + domain-adversarial batch "
            "correction. Train: 19 STARmap+ sections (batches 1..14 + "
            "16..20). Val: 10% cell-level sample of the training "
            "sections (in-distribution early-stopping signal). Test: "
            "STARmap+ batch15 + MERFISH batch82 — BOTH held out from "
            "training entirely (not loaded into the training data graph) "
            "and evaluated post-hoc by the predict + analysis pipeline "
            "(within-platform vs cross-platform retrieval). "
            "Recommended starting recipe for the cross-platform retrieval "
            "experiment — FiLM gives the decoder a per-layer batch-"
            "modulation surface; the adversary on FULL z_mlp pushes the "
            "encoder toward batch-invariant codes (wt_adv_batch=150). "
            "'narrow' = single-hop GNN + sampler=[8] + nbr_hops=1, "
            "calibrated for memory on a 44 GiB GPU at this dataset scale."
        ),
        "patches": [
            "+narrow(gnn=1, sampler=[8], nbr_hops=1)",
            "+rvq(branch=niche, levels=[30, 90])",
            "+rvq(branch=cell, levels=[30, 90])",
            "+dec-film(in-MLP, cell_batch_id)",
            "+adversarial_batch(alpha=1.0, wt=150.0)",
            "+mmb0-1b_smb1-20b_1p(test_ids=[15,82])",
        ],
        "build": lambda: _patch_dual_mmb20_holdout(
            _patch_dual_adversarial(
                _patch_dual_decoder_film(
                    _patch_dual_rvq(
                        _patch_dual_rvq(
                            _BD(),
                            branch="niche", codebook_sizes=(30, 90),
                        ),
                        branch="cell", codebook_sizes=(30, 90),
                    ),
                ),
                alpha=1.0, wt_adv_batch=150.0,
            ),
        ),
    },
    "dualvq+narrow+rvq-both+decoder-cov+adv+mmb0-1b_smb1-20b_1p": {
        "description": (
            "mmb0-1b_smb1-20b_1p with concat decoder covariate + "
            "adversarial. Same holdout split as the dec-film+adv "
            "variant (train=batches 1..14 + 16..20, val=10% in-section "
            "cell sample, test=STARmap batch15 + MERFISH batch82 "
            "evaluated post-hoc); A/B against it to measure "
            "FiLM-in-decoder vs. concat-based covariate on the 21-"
            "section retrieval setup. Same 'narrow' GNN/sampler config."
        ),
        "patches": [
            "+narrow(gnn=1, sampler=[8], nbr_hops=1)",
            "+rvq(branch=niche, levels=[30, 90])",
            "+rvq(branch=cell, levels=[30, 90])",
            "+decoder_covariate",
            "+adversarial_batch(alpha=1.0, wt=150.0)",
            "+mmb0-1b_smb1-20b_1p(test_ids=[15,82])",
        ],
        "build": lambda: _patch_dual_mmb20_holdout(
            _patch_dual_adversarial(
                _patch_dual_decoder_covariate(
                    _patch_dual_rvq(
                        _patch_dual_rvq(
                            _BD(),
                            branch="niche", codebook_sizes=(30, 90),
                        ),
                        branch="cell", codebook_sizes=(30, 90),
                    ),
                ),
                alpha=1.0, wt_adv_batch=150.0,
            ),
        ),
    },
    # Alias for the mmb20 narrow baseline under the dataset-agnostic
    # recipe name (`dualvq+rvq-both+decoder-cov+adv+<dataset_tag>`,
    # same name pattern as the mmb-smb / chl59 1-layer baselines). The
    # build is identical to `dualvq+narrow+rvq-both+decoder-cov+adv+...`
    # — the mmb20 dataset patch enforces narrow (1-layer GNN) for
    # memory regardless of whether `+wide` is applied, so dropping the
    # `+narrow` tag from the variant name doesn't change the model.
    "dualvq+rvq-both+decoder-cov+adv+mmb0-1b_smb1-20b_1p": {
        "description": (
            "mmb0-1b_smb1-20b_1p baseline with RVQ on both branches + "
            "decoder covariate + adversarial GRL. Same recipe name as "
            "the mmb-smb / chl59 1-layer baselines; the underlying GNN "
            "is 1-layer (enforced by `_patch_dual_mmb20` regardless of "
            "+wide/+narrow tag, due to memory)."
        ),
        "patches": [
            "+rvq(branch=both, levels=[30, 90])",
            "+decoder_covariate",
            "+adversarial_batch(alpha=1.0, wt=150.0)",
            "+mmb0-1b_smb1-20b_1p(test_ids=[15,82])",
        ],
        "build": lambda: _patch_dual_mmb20_holdout(
            _patch_dual_adversarial(
                _patch_dual_decoder_covariate(
                    _patch_dual_rvq(
                        _patch_dual_rvq(
                            _BD(),
                            branch="niche", codebook_sizes=(30, 90),
                        ),
                        branch="cell", codebook_sizes=(30, 90),
                    ),
                ),
                alpha=1.0, wt_adv_batch=150.0,
            ),
        ),
    },
    "dualvq+narrow+rvq-both+adv+mmb0-1b_smb1-20b_1p": {
        "description": (
            "mmb0-1b_smb1-20b_1p with adversarial-only batch correction "
            "(no decoder covariate, no FiLM). Train on 19 STARmap+ "
            "sections (batches 1..14 + 16..20), hold out STARmap "
            "batch15 + MERFISH batch82. The strictest 'biology-only' "
            "setup — the decoder has no batch-specific pathway, so the "
            "codes have to be platform-invariant by themselves. Useful "
            "as a control for the cross-platform retrieval experiment. "
            "Same 'narrow' GNN/sampler config."
        ),
        "patches": [
            "+narrow(gnn=1, sampler=[8], nbr_hops=1)",
            "+rvq(branch=niche, levels=[30, 90])",
            "+rvq(branch=cell, levels=[30, 90])",
            "+adversarial_batch(alpha=1.0, wt=150.0)",
            "+mmb0-1b_smb1-20b_1p(test_ids=[15,82])",
        ],
        "build": lambda: _patch_dual_mmb20_holdout(
            _patch_dual_adversarial(
                _patch_dual_rvq(
                    _patch_dual_rvq(
                        _BD(),
                        branch="niche", codebook_sizes=(30, 90),
                    ),
                    branch="cell", codebook_sizes=(30, 90),
                ),
                alpha=1.0, wt_adv_batch=150.0,
            ),
        ),
    },
    # NOTE: the previous `lung5+lung9-multi-test` variants were removed
    # when chl59 switched to `adata_batch_id`-based splits — their
    # position-based indices `[0,1,2,3,5,6]` / `[4,7]` were derived from
    # alphabetical filename order at a time when chl59 files lacked
    # `uns['batch']`, and don't translate to a clean `uns['batch']`
    # split without re-inspecting the silver files. Re-add them by
    # passing the corresponding `adata_batch_id` lists once the mapping
    # is confirmed.
    # ========================================================================
    # NicheCompass-style decoder-covariate batch correction (concat).
    # Replaces the previous FiLM-at-encoder mechanism, which conceptually
    # AMPLIFIES batch effects rather than removing them (no
    # batch-invariance pressure on the encoder).
    # ========================================================================
    "dualvq+wide+decoder-cov": {
        "description": (
            "Wide dual VQ + NicheCompass-style decoder covariate "
            "(per-cell batch one-hot is concatenated to z_q before each "
            "decoder). The decoders absorb per-batch gene patterns, "
            "freeing the codebook to encode batch-invariant biological "
            "identity. Recommended starting point for cross-sample / "
            "cross-platform integration."
        ),
        "patches": ["+wide", "+decoder_covariate(concat per-cell batch one-hot)"],
        "build": lambda: _patch_dual_decoder_covariate(_patch_dual_wide(_BD())),
    },
    # ========================================================================
    # Architecture ablations on mmb0-1b_smb1-1b_1p (mouse-brain MERFISH+STARmap) — fixed batch correction
    # (decoder-cov + adversarial), varying encoder/decoder/GNN/latent only.
    # All assume the current best batch-correction recipe and isolate the
    # architecture effect.
    # ========================================================================
    "dualvq+decoder-cov+adv": {
        "description": (
            "Basic dual VQ baseline (no +wide, no +rvq) with decoder "
            "covariate + adversarial batch correction. Reference point "
            "for the architecture ablations below — same batch correction "
            "everywhere, only the encoder/decoder/GNN/latent shape "
            "differs."
        ),
        "patches": [
            "+decoder_covariate",
            "+adversarial_batch(alpha=1.0, wt=150.0)",
        ],
        "build": lambda: _patch_dual_adversarial(
            _patch_dual_decoder_covariate(_BD()),
            alpha=1.0, wt_adv_batch=150.0,
        ),
    },
    "dualvq+small+decoder-cov+adv": {
        "description": (
            "scvi-style compact: 1 hidden layer of 128 dim everywhere "
            "(encoder MLP, GNN, both decoders). Latent / codebook "
            "embedding dim = 128 (down from 256). Codebook size stays "
            "at 30. ~5x fewer parameters than the SQUINT default; tests "
            "whether scvi's smaller-is-better intuition transfers to "
            "spatial VQ-VAE."
        ),
        "patches": [
            "+small(hidden=128 — encoder MLP + GNN + both decoders)",
            "+decoder_covariate",
            "+adversarial_batch(alpha=1.0, wt=150.0)",
        ],
        "build": lambda: _patch_dual_adversarial(
            _patch_dual_decoder_covariate(_patch_dual_small(_BD())),
            alpha=1.0, wt_adv_batch=150.0,
        ),
    },
    "dualvq+gnn2+decoder-cov+adv": {
        "description": (
            "Like the basic baseline but with GNN depth=2 (sampler=[8,8], "
            "nbr_hops=2). Isolates the effect of deeper spatial context "
            "on the niche branch from `+wide`'s codebook-size bump."
        ),
        "patches": [
            "+gnn2(num_layers=2, sampler=[8,8], nbr_hops=2)",
            "+decoder_covariate",
            "+adversarial_batch(alpha=1.0, wt=150.0)",
        ],
        "build": lambda: _patch_dual_adversarial(
            _patch_dual_decoder_covariate(_patch_dual_gnn2(_BD())),
            alpha=1.0, wt_adv_batch=150.0,
        ),
    },
    "dualvq+gatv2+decoder-cov+adv": {
        "description": (
            "Like the basic baseline but swap the GNN from GraphSAGE "
            "(SAGEConv, default) to GATv2 (GATv2Conv). 1 layer either "
            "way. SAGEConv aggregates neighbours uniformly (mean / "
            "weighted-by-degree); GATv2 learns per-edge attention "
            "weights so the GNN can up- or down-weight individual "
            "neighbours based on feature similarity. For sparser, more "
            "heterogeneous spatial neighbourhoods this can sharpen "
            "niche signal; for dense, homogeneous ones (mouse brain) "
            "it usually doesn't change much. Direct A/B against "
            "`dualvq+decoder-cov+adv` (same config, just SAGEConv)."
        ),
        "patches": [
            "+gatv2(SAGEConv -> GATv2Conv, num_layers=1)",
            "+decoder_covariate",
            "+adversarial_batch(alpha=1.0, wt=150.0)",
        ],
        "build": lambda: _patch_dual_adversarial(
            _patch_dual_decoder_covariate(_patch_dual_gatv2(_BD())),
            alpha=1.0, wt_adv_batch=150.0,
        ),
    },
    "dualvq+small-latent+decoder-cov+adv": {
        "description": (
            "Tighter latent bottleneck: cell_dim and niche_dim = 128 "
            "(down from 256). Encoder MLP keeps its first layer at 400; "
            "only the last hidden layer + GNN hidden shrink. Decoder "
            "widths unchanged. Tests whether forcing more "
            "semantically-concentrated codebook embeddings helps."
        ),
        "patches": [
            "+small-latent(latent_dim=128)",
            "+decoder_covariate",
            "+adversarial_batch(alpha=1.0, wt=150.0)",
        ],
        "build": lambda: _patch_dual_adversarial(
            _patch_dual_decoder_covariate(_patch_dual_small_latent(_BD())),
            alpha=1.0, wt_adv_batch=150.0,
        ),
    },
    "dualvq+dropout+decoder-cov+adv": {
        "description": (
            "Add dropout p=0.1 to encoder MLP, GNN, and both decoders. "
            "Default architecture otherwise. Useful if the model is "
            "overfitting (val Pearson plateaus while train Pearson keeps "
            "climbing) — early-stopping on val_loss already mitigates "
            "this, but dropout reduces the gap to the unattainable "
            "ceiling."
        ),
        "patches": [
            "+dropout(p=0.1)",
            "+decoder_covariate",
            "+adversarial_batch(alpha=1.0, wt=150.0)",
        ],
        "build": lambda: _patch_dual_adversarial(
            _patch_dual_decoder_covariate(_patch_dual_dropout(_BD())),
            alpha=1.0, wt_adv_batch=150.0,
        ),
    },
    "dualvq+wide+rvq-both+decoder-cov+adv": {
        "description": (
            "Wide dual VQ + RVQ on both branches + decoder covariate + "
            "domain-adversarial batch-invariance head. The combination is "
            "the NicheCompass-shape recipe: covariate concat lets the "
            "decoder absorb per-batch gene patterns; adversarial GRL "
            "(applied to the FULL z_mlp incl. sampled neighbours) "
            "actively pushes the encoder to make z_mlp batch-invariant. "
            "Recommended cross-tissue / cross-platform integration variant."
        ),
        "patches": [
            "+wide", "+rvq(branch=both, levels=[30, 90])",
            "+decoder_covariate", "+adversarial_batch(alpha=1.0, wt=150.0)",
        ],
        "build": lambda: _patch_dual_adversarial(
            _patch_dual_decoder_covariate(
                _patch_dual_rvq(
                    _patch_dual_rvq(
                        _patch_dual_wide(_BD()),
                        branch="niche", codebook_sizes=(30, 90),
                    ),
                    branch="cell", codebook_sizes=(30, 90),
                ),
            ),
            alpha=1.0, wt_adv_batch=150.0,
        ),
    },
    "dualvq+wide+rvq-both+adv": {
        "description": (
            "Wide dual VQ + RVQ on both branches + adversarial-only batch "
            "correction (no decoder covariate). Tests whether the GRL "
            "alone is enough — should not work as well as +decoder-cov+adv "
            "since the decoder has no way to fit per-batch noise."
        ),
        "patches": [
            "+wide", "+rvq(branch=both, levels=[30, 90])",
            "+adversarial_batch(alpha=1.0, wt=150.0)",
        ],
        "build": lambda: _patch_dual_adversarial(
            _patch_dual_rvq(
                _patch_dual_rvq(
                    _patch_dual_wide(_BD()),
                    branch="niche", codebook_sizes=(30, 90),
                ),
                branch="cell", codebook_sizes=(30, 90),
            ),
            alpha=1.0, wt_adv_batch=150.0,
        ),
    },
    "dualvq+wide+rvq-both+decoder-cov": {
        "description": (
            "Wide dual VQ + RVQ on both branches (levels=[30, 90] each) + "
            "NicheCompass-style decoder covariate. Combines best "
            "single-sample setup with concat-based cross-sample batch "
            "correction. Recommended cross-tissue (MERFISH+STARmap) "
            "starting point."
        ),
        "patches": [
            "+wide",
            "+rvq(branch=niche, levels=[30, 90])",
            "+rvq(branch=cell, levels=[30, 90])",
            "+decoder_covariate(concat per-cell batch one-hot)",
        ],
        "build": lambda: _patch_dual_decoder_covariate(
            _patch_dual_rvq(
                _patch_dual_rvq(
                    _patch_dual_wide(_BD()),
                    branch="niche", codebook_sizes=(30, 90),
                ),
                branch="cell", codebook_sizes=(30, 90),
            ),
        ),
    },
    "dualvq+wide+rvq-both+dec-film": {
        "description": (
            "Wide dual VQ + RVQ on both branches (levels=[30, 90] each) + "
            "FiLM batch-correction INSIDE the cell + niche decoders "
            "(`apply_conditioning='in-MLP'`, condition=cell_batch_id). "
            "No decoder-input concat, no adversarial head. The encoder "
            "is pushed toward batch-invariant codes purely by the "
            "structural separation between encoder (sees only counts) "
            "and decoder (gets the per-cell batch one-hot via FiLM, "
            "modulating each layer's activations). A/B against "
            "`+decoder-cov` (concat-based) and `+decoder-cov+adv` "
            "(concat + adversarial)."
        ),
        "patches": [
            "+wide",
            "+rvq(branch=niche, levels=[30, 90])",
            "+rvq(branch=cell, levels=[30, 90])",
            "+dec-film(in-MLP, cell_batch_id)",
        ],
        "build": lambda: _patch_dual_decoder_film(
            _patch_dual_rvq(
                _patch_dual_rvq(
                    _patch_dual_wide(_BD()),
                    branch="niche", codebook_sizes=(30, 90),
                ),
                branch="cell", codebook_sizes=(30, 90),
            ),
        ),
    },
    "dualvq+wide+rvq-both+dec-film+adv": {
        "description": (
            "Wide dual VQ + RVQ on both branches + FiLM INSIDE the cell + "
            "niche decoders (`apply_conditioning='in-MLP'`, condition="
            "cell_batch_id) + domain-adversarial batch-invariance head. "
            "Completes the 2x2: {FiLM, concat} x {with adv, without adv}. "
            "FiLM gives the decoder a structurally cleaner per-layer "
            "modulation surface than concat (better fit for the deeper "
            "non-linear MLPSoftmax decoders); the adversary (applied to "
            "the FULL z_mlp incl. sampled neighbours, wt_adv_batch=150) "
            "acts as a regularizer against the decoder over-explaining "
            "genuine biology as batch."
        ),
        "patches": [
            "+wide",
            "+rvq(branch=niche, levels=[30, 90])",
            "+rvq(branch=cell, levels=[30, 90])",
            "+dec-film(in-MLP, cell_batch_id)",
            "+adversarial_batch(alpha=1.0, wt=150.0)",
        ],
        "build": lambda: _patch_dual_adversarial(
            _patch_dual_decoder_film(
                _patch_dual_rvq(
                    _patch_dual_rvq(
                        _patch_dual_wide(_BD()),
                        branch="niche", codebook_sizes=(30, 90),
                    ),
                    branch="cell", codebook_sizes=(30, 90),
                ),
            ),
            alpha=1.0, wt_adv_batch=150.0,
        ),
    },
    "dualvq+wide+cvq-both": {
        "description": (
            "Wide dual VQ + ConditionalVQ (tree) on BOTH branches "
            "(K1=30, K2=5 each = 150 effective codes per branch). "
            "Symmetric tree-structured codebooks for easy A/B against "
            "`+rvq-both`. Cell branch gets a coarse cell-type → "
            "cell-state hierarchy; niche branch gets a macro-niche → "
            "sub-niche hierarchy. Same per-(K1, K2)-bucket data as "
            "`+cvq-niche` (~660 cells), so level-2 EMA is well-conditioned."
        ),
        "patches": [
            "+wide", "+cvq(branch=both, k1=30, k2=5)",
        ],
        "build": lambda: _patch_dual_cvq(
            _patch_dual_wide(_BD()),
            branch="both", k1=30, k2=5,
        ),
    },
    "dualvq+adj-on-zqniche": {
        "description": (
            "Dual VQ with the adjacency BCE input switched from z_gnn "
            "(continuous, default) to z_q_niche (quantized). Useful for "
            "A/B testing whether the continuous-vs-quantized choice matters "
            "in practice. Adjacency weight kept at the default (100); pair "
            "with `_patch_dual_adj_weight` for higher weights."
        ),
        "patches": ["+cosine_adj(input=z_q_niche)"],
        "build": lambda: _patch_dual_adj_on_zqniche(_BD()),
    },
    "dualvq-no-adj": {
        "description": (
            "Dual VQ WITHOUT the cosine-sim adjacency BCE. Niche branch is "
            "supervised only by neighbourhood NB. Tests whether the "
            "adjacency term is load-bearing for spatial code coherence."
        ),
        "patches": ["-cosine_adj"],
        "build": lambda: _patch_dual_no_adj(_BD()),
    },
    "dualvq-niche-only": {
        "description": (
            "Dual VQ with the cell-branch NB and cell-commit losses dropped: "
            "only the niche branch is supervised. The cell decoder/codebook "
            "still run forward but receive no gradient. Tests whether the "
            "cell branch contributes useful signal to the shared MLP trunk."
        ),
        "patches": ["-nb_cell", "-commit_cell"],
        "build": lambda: _patch_dual_no_cell_recon(_BD()),
    },
    # ---- Full SQUINT (all components) -------------------------------------
    "full": {
        "description": (
            "Full SQUINT: recon-both + adjacency + FiLM + masking + "
            "multi-head VQ (heads=10, codebook=5000). All design components "
            "active. Run only after ablations confirm each component helps."
        ),
        "patches": [
            "recon_mode=both",
            "+nb_attribute_reconstruction_loss_nbr",
            "+bce_adjacency_reconstruction_loss",
            "+film(rbf_distances, cell_batch_id)",
            "+masking(base=0.2, final=0.6, warmup=5)",
            "+mhvq(heads=10, codebook_size=5000)",
        ],
        "build": lambda: _patch_multihead_vq(
            _patch_masking(
                _patch_film(
                    _patch_adj(
                        _patch_recon_both(_B())))),
            heads=10, codebook_size=5000),
    },
}


# ---------------------------------------------------------------------------
# Per-dataset ablation sweep matrix (10 variants per dataset).
# ---------------------------------------------------------------------------
# A consistent 10-axis ablation sweep registered for each of the 3 datasets.
# Run one dataset's sweep per HPC job (one per node) via:
#
#   python examples/run_squint.py --all-dataset mmb0-1b_smb1-1b_1p
#   python examples/run_squint.py --all-dataset chl59-8b_1p
#   python examples/run_squint.py --all-dataset mmb0-1b_smb1-20b_1p
#
# Each variant ablates ONE axis from a fixed spine baseline. The spine is:
#   wide (or narrow on memory-constrained datasets) + RVQ on both branches
#   (k1=30, k2=90) + NicheCompass-style decoder covariate (concat per-cell
#   batch one-hot) + adversarial GRL on FULL z_mlp (alpha=1.0, wt=150).
#
# For wide-spine datasets the 10 axes are:
#   1. baseline       — pure spine
#   2. gatv2          — SAGEConv -> GATv2Conv
#   3. small          — scvi-style hidden=128 everywhere (replaces wide)
#   4. small-latent   — latent / codebook-embedding dim 256 -> 128
#   5. dropout        — p=0.1 on encoder MLP, GNN, both decoders
#   6. dec-film       — FiLM-in-decoder INSTEAD of decoder covariate concat
#   7. no-adv         — drop the adversarial head
#   8. no-arch        — drop +wide (1-hop GNN, default codebook)
#   9. no-rvq         — single VQ k=30 on both branches (no RVQ)
#  10. no-adj         — drop the cosine adjacency BCE loss
#
# For the narrow (memory-constrained) mmb20 dataset, axis 8 changes from
# 'no-arch' to 'codebook-large' (RVQ levels 50/150 instead of 30/90),
# because a wider GNN OOMs at 19 sections on a 44 GiB GPU.

_SWEEP_AXES_WIDE = [
    "baseline", "gatv2", "small", "small-latent", "dropout",
    "dec-film", "no-adv", "no-arch", "no-rvq", "no-adj",
]
_SWEEP_AXES_NARROW = [
    "baseline", "gatv2", "small", "small-latent", "dropout",
    "dec-film", "no-adv", "codebook-large", "no-rvq", "no-adj",
]

_SWEEP_AXIS_DESC = {
    "baseline": (
        "Spine baseline (no ablation). Wide-or-narrow GNN + RVQ on both "
        "branches (k1=30, k2=90) + NicheCompass-style decoder covariate "
        "(concat per-cell batch one-hot) + adversarial GRL on FULL z_mlp "
        "(alpha=1.0, wt_adv_batch=150). Reference point for the 10-axis "
        "sweep."
    ),
    "gatv2": (
        "Spine + SAGEConv -> GATv2Conv on the niche-branch GNN. Tests "
        "per-edge attention vs. uniform mean aggregation. Helps more on "
        "sparse / heterogeneous neighbourhoods."
    ),
    "small": (
        "scvi-style compact architecture: 1 hidden layer of 128 dim "
        "everywhere (encoder MLP, GNN, both decoders). Latent / codebook "
        "embedding dim = 128. ~5x fewer parameters than the SQUINT default. "
        "Replaces +wide (which sets 2-layer GNN at 256 dim). RVQ + "
        "decoder-cov + adv unchanged. Tests whether smaller-is-better "
        "transfers to spatial VQ-VAE."
    ),
    "small-latent": (
        "Spine + latent / codebook-embedding dim 256 -> 128. Encoder MLP "
        "first layer stays at 400; only the last hidden layer + GNN hidden "
        "shrink. Tests whether forcing more semantically-concentrated "
        "codebook embeddings helps."
    ),
    "dropout": (
        "Spine + dropout p=0.1 on encoder MLP, GNN, and both decoders. "
        "Cheap regularization. Useful diagnostic when val Pearson plateaus "
        "while train Pearson keeps climbing."
    ),
    "dec-film": (
        "Spine but FiLM batch-correction INSIDE the cell + niche decoders "
        "(`apply_conditioning='in-MLP'`, condition=cell_batch_id) INSTEAD "
        "of concat-based decoder covariate. FiLM gives the decoder a "
        "per-layer modulation surface; recommended over concat for the "
        "deeper non-linear MLPSoftmax decoders."
    ),
    "no-adv": (
        "Spine WITHOUT the adversarial head. Decoder still gets the "
        "covariate. Tests whether the GRL is contributing batch invariance "
        "beyond what concat covariate alone achieves."
    ),
    "no-arch": (
        "Spine WITHOUT +wide (so 1-hop GNN, sampler=[8], nbr_hops=1, "
        "default codebook size). Tests whether the deeper spatial context "
        "from +wide is load-bearing once RVQ is in place."
    ),
    "no-rvq": (
        "Spine WITHOUT RVQ — single VQ at k=30 on both branches instead "
        "of residual (30, 90). Tests whether the hierarchical RVQ "
        "structure helps over a flat codebook of comparable budget."
    ),
    "no-adj": (
        "Spine WITHOUT the cosine adjacency BCE loss. Niche branch is "
        "supervised only by neighbourhood NB. Tests whether the "
        "adjacency term is load-bearing for spatial code coherence."
    ),
    "codebook-large": (
        "Spine but RVQ levels 30,90 -> 50,150 on both branches (larger "
        "primary + larger residual codebooks). Single coupled "
        "codebook-capacity scan; substituted for +wide on narrow "
        "(memory-constrained) datasets where a 2-hop GNN OOMs."
    ),
}


def _compose_sweep_spine(ablation: str, narrow: bool = False) -> dict:
    """
    Build the dual-VQ spine for the per-dataset ablation sweep, ablating
    exactly one axis. See the SWEEP_AXES tables for the axis catalogue.

    The dataset wrapper (e.g. _patch_dual_chl59_lung5,
    _patch_dual_mmb20_holdout) is applied AFTER this returns — those
    patches are not the responsibility of this function.
    """
    cfg = _BD()

    # Architecture (wide vs. narrow vs. small).
    if ablation == "small":
        # 'small' replaces both +wide AND the default-1-hop GNN (it sets
        # num_layers=1, hidden=128 directly). On narrow datasets the
        # dataset patch later re-asserts num_layers=1 (no conflict).
        cfg = _patch_dual_small(cfg)
    elif ablation == "no-arch":
        # No architectural patch: defaults to 1-hop GNN, default codebook.
        pass
    elif not narrow:
        cfg = _patch_dual_wide(cfg)
    # narrow + non-small: defer GNN config to the dataset wrapper (e.g.
    # _patch_dual_mmb20 sets num_layers=1, sampler=[8], nbr_hops=1).

    # VQ structure.
    if ablation == "no-rvq":
        # Single VQ at base k=30 on both branches — no RVQ wrap.
        pass
    elif ablation == "codebook-large":
        cfg = _patch_dual_rvq(cfg, branch="niche", codebook_sizes=(50, 150))
        cfg = _patch_dual_rvq(cfg, branch="cell",  codebook_sizes=(50, 150))
    else:
        cfg = _patch_dual_rvq(cfg, branch="niche", codebook_sizes=(30, 90))
        cfg = _patch_dual_rvq(cfg, branch="cell",  codebook_sizes=(30, 90))

    # Latent dim shrink.
    if ablation == "small-latent":
        cfg = _patch_dual_small_latent(cfg, latent_dim=128)

    # Dropout.
    if ablation == "dropout":
        cfg = _patch_dual_dropout(cfg, p=0.1)

    # GNN type.
    if ablation == "gatv2":
        cfg = _patch_dual_gatv2(cfg)

    # Decoder batch correction shape.
    if ablation == "dec-film":
        cfg = _patch_dual_decoder_film(cfg)
    else:
        cfg = _patch_dual_decoder_covariate(cfg)

    # Adversarial head.
    if ablation != "no-adv":
        cfg = _patch_dual_adversarial(cfg, alpha=1.0, wt_adv_batch=150.0)

    # Adjacency.
    if ablation == "no-adj":
        cfg = _patch_dual_no_adj(cfg)

    return cfg


def _sweep_variant_name(ablation: str, dataset_tag: str, narrow: bool) -> str:
    """Compose a descriptive variant name for the sweep matrix."""
    arch = "narrow" if narrow else "wide"
    if ablation == "baseline":
        core = f"{arch}+rvq-both+decoder-cov+adv"
    elif ablation == "gatv2":
        core = f"{arch}+rvq-both+decoder-cov+adv+gatv2"
    elif ablation == "small":
        # 'small' overrides architecture; prefix narrow-dataset names with
        # 'narrow+' so the dataset context stays visible.
        core = f"{'narrow+' if narrow else ''}small+rvq-both+decoder-cov+adv"
    elif ablation == "small-latent":
        core = f"{arch}+rvq-both+decoder-cov+adv+small-latent"
    elif ablation == "dropout":
        core = f"{arch}+rvq-both+decoder-cov+adv+dropout"
    elif ablation == "dec-film":
        core = f"{arch}+rvq-both+dec-film+adv"
    elif ablation == "no-adv":
        core = f"{arch}+rvq-both+decoder-cov"
    elif ablation == "no-arch":
        core = f"rvq-both+decoder-cov+adv"
    elif ablation == "no-rvq":
        core = f"{arch}+decoder-cov+adv"
    elif ablation == "no-adj":
        core = f"{arch}+rvq-both+decoder-cov+adv+no-adj"
    elif ablation == "codebook-large":
        core = f"{arch}+rvq-both-large+decoder-cov+adv"
    else:
        raise ValueError(f"unknown sweep ablation: {ablation!r}")
    return f"dualvq+{core}+{dataset_tag}"


def _sweep_patch_tags(ablation: str, narrow: bool) -> list[str]:
    """Patch-tag list for the variant's `patches` field (display only)."""
    arch_tag = "+narrow" if narrow else "+wide"
    rvq_tag = "+rvq(branch=both, levels=[30, 90])"
    base = [arch_tag, rvq_tag, "+decoder_covariate",
            "+adversarial_batch(alpha=1.0, wt=150.0)"]
    if ablation == "baseline":              return list(base)
    if ablation == "gatv2":                 return list(base) + ["+gatv2"]
    if ablation == "small":
        return ["+small(hidden=128 — encoder MLP + GNN + both decoders)", rvq_tag, "+decoder_covariate",
                "+adversarial_batch(alpha=1.0, wt=150.0)"]
    if ablation == "small-latent":          return list(base) + ["+small-latent(latent_dim=128)"]
    if ablation == "dropout":               return list(base) + ["+dropout(p=0.1)"]
    if ablation == "dec-film":
        return [arch_tag, rvq_tag, "+dec-film(in-MLP, cell_batch_id)",
                "+adversarial_batch(alpha=1.0, wt=150.0)"]
    if ablation == "no-adv":                return [arch_tag, rvq_tag, "+decoder_covariate"]
    if ablation == "no-arch":               return [rvq_tag, "+decoder_covariate",
                                                   "+adversarial_batch(alpha=1.0, wt=150.0)"]
    if ablation == "no-rvq":                return [arch_tag, "+decoder_covariate",
                                                   "+adversarial_batch(alpha=1.0, wt=150.0)"]
    if ablation == "no-adj":                return list(base) + ["-cosine_adj"]
    if ablation == "codebook-large":
        return [arch_tag, "+rvq(branch=both, levels=[50, 150])",
                "+decoder_covariate", "+adversarial_batch(alpha=1.0, wt=150.0)"]
    raise ValueError(f"unknown sweep ablation: {ablation!r}")


def _register_dataset_sweep(
        dataset_tag: str,
        dataset_apply_fn,
        narrow: bool = False,
    ) -> list[str]:
    """
    Register the 10-axis sweep variants for a dataset under VARIANTS.

    Idempotent: if a variant with the composed name already exists in
    VARIANTS (e.g. user-curated entries higher up in this file), the
    existing entry is kept and only the name is appended to the returned
    list. This lets the sweep matrix coexist with the hand-written
    variants without clobbering them.

    Returns the ordered list of variant names — used by run_dataset_pipeline.
    """
    axes = _SWEEP_AXES_NARROW if narrow else _SWEEP_AXES_WIDE
    names: list[str] = []
    for ablation in axes:
        name = _sweep_variant_name(ablation, dataset_tag, narrow)
        if name not in VARIANTS:
            # Bind ablation+narrow into closures so each variant builds its own.
            def _build(_a=ablation, _n=narrow, _apply=dataset_apply_fn):
                return _apply(_compose_sweep_spine(_a, narrow=_n))
            VARIANTS[name] = {
                "description": (
                    _SWEEP_AXIS_DESC[ablation]
                    + f" Dataset: {dataset_tag}."
                ),
                "patches": _sweep_patch_tags(ablation, narrow)
                           + [f"+{dataset_tag}"],
                "build": _build,
            }
        names.append(name)
    return names


# Map short dataset tag -> ordered list of sweep variant names. The CLI's
# --all-dataset flag accepts these short tags. The full dataset_name (used
# as the cfg key + artifact dir) is set by each variant's build() via the
# corresponding _patch_dual_{chl59_lung5,mmb20_holdout} (or the base config
# for mmb-smb).
DATASET_VARIANTS: dict = {}

DATASET_VARIANTS["mmb0-1b_smb1-1b_1p"] = _register_dataset_sweep(
    dataset_tag="mmb0-1b_smb1-1b_1p",
    dataset_apply_fn=lambda c: c,   # base config IS the mmb-smb dataset
    narrow=False,
)

DATASET_VARIANTS["chl59-8b_1p"] = _register_dataset_sweep(
    dataset_tag="chl59-8b_1p",
    dataset_apply_fn=lambda c: _patch_dual_chl59_lung5(
        c, test_batch_idx=[2, 3]),
    narrow=False,
)

DATASET_VARIANTS["mmb0-1b_smb1-20b_1p"] = _register_dataset_sweep(
    dataset_tag="mmb0-1b_smb1-20b_1p",
    dataset_apply_fn=lambda c: _patch_dual_mmb20_holdout(c),
    narrow=True,
)

# Targeted mmb0-1b_smb1-1b_1p 1-layer-GNN ablation matrix (separate from the
# 10-axis sweep above). The spine is RVQ-both + decoder-cov + adv +
# 1-layer GNN; each variant ablates ONE axis. Run via:
#   python examples/run_squint.py --all-dataset mmb0-1b_smb1-1b_1p-ablations
# Note: the key is NOT a dataset tag — DATASET_VARIANTS doubles here as
# a registry of named variant groups. The runner only cares about the
# value (a list of variant names); it does not require the key to match
# any `dataset_name` / `dataset_tag` on the cfg.
DATASET_VARIANTS["mmb0-1b_smb1-1b_1p-ablations"] = [
    # baseline (the spine itself; registered earlier as a sweep variant)
    "dualvq+rvq-both+decoder-cov+adv+mmb0-1b_smb1-1b_1p",
    # GNN type axis: SAGE -> GATv2
    "dualvq+rvq-both+decoder-cov+adv+gatv2+mmb0-1b_smb1-1b_1p",
    # GNN depth axis: 1 -> 2 layers (registered as the +wide sweep variant)
    "dualvq+wide+rvq-both+decoder-cov+adv+mmb0-1b_smb1-1b_1p",
    # Encoder size axis: default -> scvi-style small (hidden=128)
    "dualvq+small+rvq-both+decoder-cov+adv+mmb0-1b_smb1-1b_1p",
    # Adversarial weight axis: 150 -> {300, 50}
    "dualvq+rvq-both+decoder-cov+adv-w300+mmb0-1b_smb1-1b_1p",
    "dualvq+rvq-both+decoder-cov+adv-w50+mmb0-1b_smb1-1b_1p",
    # Adjacency BCE weight axis: 1000 -> {3000, 250}
    "dualvq+rvq-both+decoder-cov+adv+adj-w3000+mmb0-1b_smb1-1b_1p",
    "dualvq+rvq-both+decoder-cov+adv+adj-w250+mmb0-1b_smb1-1b_1p",
]

# mmb-smb 1-layer ablation matrix v2: capacity / regularization /
# batch-correction / spatial-context axes. Each variant ablates ONE
# knob from the same RVQ-both + decoder-cov + adv + 1-layer-GNN spine
# the v1 ablations use. Submit all 8 via:
#   bash examples/submit_dataset_sweep.sh mmb0-1b_smb1-1b_1p-ablations-v2
DATASET_VARIANTS["mmb0-1b_smb1-1b_1p-ablations-v2"] = [
    # codebook capacity
    "dualvq+rvq-both-large+decoder-cov+adv+mmb0-1b_smb1-1b_1p",
    "dualvq+rvq-both-3level+decoder-cov+adv+mmb0-1b_smb1-1b_1p",
    # latent / regularization
    "dualvq+rvq-both+decoder-cov+adv+latent-128+mmb0-1b_smb1-1b_1p",
    "dualvq+rvq-both+decoder-cov+adv+dropout+mmb0-1b_smb1-1b_1p",
    # batch correction
    "dualvq+rvq-both+decoder-cov+adv+batch-emb32+mmb0-1b_smb1-1b_1p",
    # spatial context / encoder capacity
    "dualvq+rvq-both+decoder-cov+adv+sampler16+mmb0-1b_smb1-1b_1p",
    "dualvq+rvq-both+decoder-cov+adv+enc-deeper+mmb0-1b_smb1-1b_1p",
    "dualvq+rvq-both+decoder-cov+adv+gnn-h384+mmb0-1b_smb1-1b_1p",
]

# mmb-smb 2-layer-spine ablation matrix v3: codebook capacity / structure
# scan on top of the v1 winners' combined spine
# (+wide [best on niche ID] + adj-w250 [best on batch integration]).
# Submit all 8 via:
#   bash examples/submit_dataset_sweep.sh mmb0-1b_smb1-1b_1p-ablations-v3
DATASET_VARIANTS["mmb0-1b_smb1-1b_1p-ablations-v3"] = [
    # spine baseline (winners combined)
    "dualvq+wide+rvq-both+decoder-cov+adv+adj-w250+mmb0-1b_smb1-1b_1p",
    # RVQ macro-codebook scan (k1)
    "dualvq+wide+rvq-both-50-90+decoder-cov+adv+adj-w250+mmb0-1b_smb1-1b_1p",
    "dualvq+wide+rvq-both-80-90+decoder-cov+adv+adj-w250+mmb0-1b_smb1-1b_1p",
    # RVQ both-layer scan
    "dualvq+wide+rvq-both-50-150+decoder-cov+adv+adj-w250+mmb0-1b_smb1-1b_1p",
    "dualvq+wide+rvq-both-80-200+decoder-cov+adv+adj-w250+mmb0-1b_smb1-1b_1p",
    # 3-level RVQ
    "dualvq+wide+rvq-both-3level+decoder-cov+adv+adj-w250+mmb0-1b_smb1-1b_1p",
    # CVQ (tree-structured) family
    "dualvq+wide+cvq-both-30-10+decoder-cov+adv+adj-w250+mmb0-1b_smb1-1b_1p",
    "dualvq+wide+cvq-both-50-10+decoder-cov+adv+adj-w250+mmb0-1b_smb1-1b_1p",
]

# mmb-smb v4 ablations (final round): hybridise the strongest non-wide
# signals (`+enc-deeper`, `+adj-w3000`) with the v3 winner
# (`+wide+rvq-both+decoder-cov+adv`), and explore a NEW axis —
# adversarial warmup — to recover cell-NMI / cell-pearson regressions.
# See the variant comments above + the v4 selection writeup for the
# per-variant rationale. Submit all 8 via:
#   bash examples/submit_mmb_smb_ablations_v4.sh
#   # (or: bash examples/submit_dataset_sweep.sh mmb0-1b_smb1-1b_1p-ablations-v4)
DATASET_VARIANTS["mmb0-1b_smb1-1b_1p-ablations-v4"] = [
    # Single-knob ports of empirical winners onto the v3 spine.
    "dualvq+wide+rvq-both+decoder-cov+adv+enc-deeper+mmb0-1b_smb1-1b_1p",
    "dualvq+wide+rvq-both+decoder-cov+adv+adj-w3000+mmb0-1b_smb1-1b_1p",
    "dualvq+wide+rvq-both+decoder-cov+adv-w50+mmb0-1b_smb1-1b_1p",
    # Stack of the two top empirical signals.
    "dualvq+wide+rvq-both+decoder-cov+adv+enc-deeper+adj-w3000+mmb0-1b_smb1-1b_1p",
    # NEW axis: adversarial warmup (alpha=0 for first 10 epochs).
    "dualvq+wide+rvq-both+decoder-cov+adv-warmup10+mmb0-1b_smb1-1b_1p",
    "dualvq+wide+rvq-both+decoder-cov+adv+enc-deeper+adv-warmup10+mmb0-1b_smb1-1b_1p",
    # Independent exploration: longer-range niche target.
    "dualvq+wide+rvq-both+decoder-cov+adv+nbr-hops-3+mmb0-1b_smb1-1b_1p",
    # Kitchen sink of the three top signals.
    "dualvq+wide+rvq-both+decoder-cov+adv+enc-deeper+adj-w3000+adv-warmup10+mmb0-1b_smb1-1b_1p",
]

# mmb-smb v5 ablations (final-final round): codebook-capacity scan on
# the new champion spine `+wide+rvq-both-3level+decoder-cov+adv+enc-deeper`.
# 6 RVQ-3-level variants (capacity scan + MLP-width pair) +
# 2 CVQ-3-level variants (architectural alternative; CVQ extended from
# 2 to 3 levels in `hierarchical_vq.py` for this round).
# Submit all 8 via:
#   bash examples/submit_mmb_smb_ablations_v5.sh
DATASET_VARIANTS["mmb0-1b_smb1-1b_1p-ablations-v5"] = [
    # Spine baseline (the new project default).
    "dualvq+wide+rvq-both-3level+decoder-cov+adv+enc-deeper+mmb0-1b_smb1-1b_1p",
    # RVQ-3-level capacity scan: small / large / xlarge.
    "dualvq+wide+rvq-both-3level-50-100-200+decoder-cov+adv+enc-deeper+mmb0-1b_smb1-1b_1p",
    "dualvq+wide+rvq-both-3level-80-160-320+decoder-cov+adv+enc-deeper+mmb0-1b_smb1-1b_1p",
    "dualvq+wide+rvq-both-3level-20-40-80+decoder-cov+adv+enc-deeper+mmb0-1b_smb1-1b_1p",
    # MLP-width ablations (encoder + decoder shrink / grow together).
    "dualvq+wide+rvq-both-3level+decoder-cov+adv+enc-deeper+mlp-h256+mmb0-1b_smb1-1b_1p",
    "dualvq+wide+rvq-both-3level+decoder-cov+adv+enc-deeper+mlp-h512+mmb0-1b_smb1-1b_1p",
    # 3-level Conditional VQ (tree partitioning) -- architectural A/B.
    "dualvq+wide+cvq-both-3level-30-10-5+decoder-cov+adv+enc-deeper+mmb0-1b_smb1-1b_1p",
    "dualvq+wide+cvq-both-3level-50-15-5+decoder-cov+adv+enc-deeper+mmb0-1b_smb1-1b_1p",
]

# v6 — 8 ablations targeted at improving cell-token NMI on cell_type
# WITHOUT regressing batch integration. See descriptions on each
# variant for the per-experiment hypothesis. Three orthogonal
# mechanisms are tested:
#   - Capacity bottleneck (smaller encoder) + warmup    (3 variants)
#   - Warmup paired with a non-adversarial integration   (3 variants)
#     mechanism (MMD-batch loss): two strengths + an
#     MMD-only ablation (no adversary at all).
#   - Schedule + asymmetric adversary                    (2 variants)
# Submit all 8 via:
#   bash examples/submit_mmb_smb_ablations_v6.sh
DATASET_VARIANTS["mmb0-1b_smb1-1b_1p-ablations-v6"] = [
    # Group A — smaller encoder + warmup (user hypothesis).
    "dualvq+wide+rvq-both+decoder-cov+adv+enc-shallow+adv-warmup5+mmb0-1b_smb1-1b_1p",
    "dualvq+wide+rvq-both+decoder-cov+adv+enc-shallow+adv-warmup10+mmb0-1b_smb1-1b_1p",
    "dualvq+small+rvq-both+decoder-cov+adv+adv-warmup5+mmb0-1b_smb1-1b_1p",
    # Group B — warmup paired with MMD batch-invariance (3 variants).
    "dualvq+wide+rvq-both+decoder-cov+adv+adv-warmup10+mmd-w50+mmb0-1b_smb1-1b_1p",
    "dualvq+wide+rvq-both+decoder-cov+adv+adv-warmup10+mmd-w200+mmb0-1b_smb1-1b_1p",
    "dualvq+wide+rvq-both+decoder-cov+mmd-w100+mmb0-1b_smb1-1b_1p",
    # Group C — schedule + asymmetric adversary.
    "dualvq+wide+rvq-both+decoder-cov+adv-cell-only+adv-w300+mmb0-1b_smb1-1b_1p",
    "dualvq+wide+rvq-both+decoder-cov+adv-cosine+adv-warmup10+mmb0-1b_smb1-1b_1p",
]

# v6b — second wave targeting cell-token NMI on cell_type. The 3 MMD
# variants from v6 errored on a kwarg name mismatch (now fixed); we
# re-run them here. The other 5 v6 variants underperformed; we replace
# them with new mechanisms ablating different parts of the
# warmup-NMI / integration trade-off:
#   - Post-warmup adv-weight scan (adv-w50, adv-w300 with warmup10)
#   - Sharper cell VQ commit (commit-cell-w2 + warmup10)
#   - Semi-supervised cell-type CE (ce-w10 + warmup10) — empirical
#     upper bound; CLEARLY LABEL as semi-supervised in paper
#   - Combo safety stack (enc-shallow + warmup10 + mmd-w50)
# Submit all 8 via:
#   bash examples/submit_mmb_smb_ablations_v6b.sh
DATASET_VARIANTS["mmb0-1b_smb1-1b_1p-ablations-v6b"] = [
    # Re-runs of the 3 MMD variants (now that mmd_target arg-name is fixed).
    "dualvq+wide+rvq-both+decoder-cov+adv+adv-warmup10+mmd-w50+mmb0-1b_smb1-1b_1p",
    "dualvq+wide+rvq-both+decoder-cov+adv+adv-warmup10+mmd-w200+mmb0-1b_smb1-1b_1p",
    "dualvq+wide+rvq-both+decoder-cov+mmd-w100+mmb0-1b_smb1-1b_1p",
    # 5 NEW mechanisms.
    "dualvq+wide+rvq-both+decoder-cov+adv-w50+adv-warmup10+mmb0-1b_smb1-1b_1p",
    "dualvq+wide+rvq-both+decoder-cov+adv-w300+adv-warmup10+mmb0-1b_smb1-1b_1p",
    "dualvq+wide+rvq-both+decoder-cov+commit-cell-w2+adv+adv-warmup10+mmb0-1b_smb1-1b_1p",
    "dualvq+wide+rvq-both+decoder-cov+ce-w10+adv+adv-warmup10+mmb0-1b_smb1-1b_1p",
    "dualvq+wide+rvq-both+decoder-cov+adv+enc-shallow+adv-warmup10+mmd-w50+mmb0-1b_smb1-1b_1p",
]

# v7 — 8 ablations targeted at improving cell-token NMI on cell_type by
# varying ONLY the cell codebook structure. Niche is held fixed at
# 3-level RVQ (30, 60, 120). Level-1 size is 30 across all variants
# (matches niche level 1). The 10-epoch adv-warmup family from v6/v6b
# is dropped — the post-warmup loss spiked dramatically in those runs
# as the late-arriving adversary tried to undo features the encoder
# had committed during warmup. v7 uses adv-from-epoch-0 (alpha=1.0,
# wt=150.0) throughout.
#
# Cell codebook sweep (level 1 = 30 always):
#   - 1-level VQ:   k ∈ {30, 60, 200}
#   - 2-level RVQ:  (30, 60) / (30, 90) / (30, 200)
#   - 3-level RVQ:  (30, 60, 120) [matches niche] / (30, 90, 270)
# Submit all 8 via:
#   bash examples/submit_mmb_smb_ablations_v7.sh
DATASET_VARIANTS["mmb0-1b_smb1-1b_1p-ablations-v7"] = [
    # 1-level VQ on the cell side (single codebook, no residual depth).
    "dualvq+wide+rvq-niche-30-60-120+vq-cell-30+decoder-cov+adv+enc-deeper+mmb0-1b_smb1-1b_1p",
    "dualvq+wide+rvq-niche-30-60-120+vq-cell-60+decoder-cov+adv+enc-deeper+mmb0-1b_smb1-1b_1p",
    "dualvq+wide+rvq-niche-30-60-120+vq-cell-200+decoder-cov+adv+enc-deeper+mmb0-1b_smb1-1b_1p",
    # 2-level RVQ on the cell side (level 1 = 30; level 2 varies).
    "dualvq+wide+rvq-niche-30-60-120+rvq-cell-30-60+decoder-cov+adv+enc-deeper+mmb0-1b_smb1-1b_1p",
    "dualvq+wide+rvq-niche-30-60-120+rvq-cell-30-90+decoder-cov+adv+enc-deeper+mmb0-1b_smb1-1b_1p",
    "dualvq+wide+rvq-niche-30-60-120+rvq-cell-30-200+decoder-cov+adv+enc-deeper+mmb0-1b_smb1-1b_1p",
    # 3-level RVQ on the cell side: matches-niche, then wider.
    "dualvq+wide+rvq-niche-30-60-120+rvq-cell-30-60-120+decoder-cov+adv+enc-deeper+mmb0-1b_smb1-1b_1p",
    "dualvq+wide+rvq-niche-30-60-120+rvq-cell-30-90-270+decoder-cov+adv+enc-deeper+mmb0-1b_smb1-1b_1p",
]

# v8 — 8 ablations split into two groups of 4 on the v5-anchor spine
# (+wide +rvq-both-3level +decoder-cov +adv, no warmup, no +enc-deeper).
# Group A varies encoder/decoder MLP size (depth, width, latent-dim,
# fully compact); Group B varies batch-integration mechanism on a 2x2
# {MMD-only vs adv+MMD} x {weak 50 vs strong 200} grid.
# Submit all 8 via:
#   bash examples/submit_mmb_smb_ablations_v8.sh
DATASET_VARIANTS["mmb0-1b_smb1-1b_1p-ablations-v8"] = [
    # Group A — encoder/decoder size sweep (4 variants).
    "dualvq+wide+rvq-both-3level+decoder-cov+adv+enc-shallow+mmb0-1b_smb1-1b_1p",
    "dualvq+wide+rvq-both-3level+decoder-cov+adv+mlp-h256+mmb0-1b_smb1-1b_1p",
    "dualvq+wide+rvq-both-3level+decoder-cov+adv+small-latent+mmb0-1b_smb1-1b_1p",
    "dualvq+small+rvq-both-3level+decoder-cov+adv+mmb0-1b_smb1-1b_1p",
    # Group B — MMD + adversarial batch-integration sweep (4 variants, 2x2).
    "dualvq+wide+rvq-both-3level+decoder-cov+mmd-w50+mmb0-1b_smb1-1b_1p",
    "dualvq+wide+rvq-both-3level+decoder-cov+mmd-w200+mmb0-1b_smb1-1b_1p",
    "dualvq+wide+rvq-both-3level+decoder-cov+adv+mmd-w50+mmb0-1b_smb1-1b_1p",
    "dualvq+wide+rvq-both-3level+decoder-cov+adv+mmd-w200+mmb0-1b_smb1-1b_1p",
]

# v9 — 8 ablations on a SMALLER + FASTER architecture, organised as
# a 2 x 4 grid: { compact (h32 + (30,10,10)), medium (h64 + (30,20,10)) }
# crossed with 4 batch-integration strategies { default adv, boosted adv,
# adv+MMD combo, MMD-only }. Shared spine: NO +wide (1-hop GNN),
# +small(hidden=N), +rvq-both-3level (L0=30, symmetric), +decoder-cov.
# Submit all 8 via:
#   bash examples/submit_mmb_smb_ablations_v9.sh
DATASET_VARIANTS["mmb0-1b_smb1-1b_1p-ablations-v9"] = [
    # Compact size: hidden=32 + (30, 10, 10).
    "dualvq+small-h32+rvq-both-3level-30-10-10+decoder-cov+adv+mmb0-1b_smb1-1b_1p",
    "dualvq+small-h32+rvq-both-3level-30-10-10+decoder-cov+adv-w300+mmb0-1b_smb1-1b_1p",
    "dualvq+small-h32+rvq-both-3level-30-10-10+decoder-cov+adv+mmd-w50+mmb0-1b_smb1-1b_1p",
    "dualvq+small-h32+rvq-both-3level-30-10-10+decoder-cov+mmd-w100+mmb0-1b_smb1-1b_1p",
    # Medium size: hidden=64 + (30, 20, 10).
    "dualvq+small-h64+rvq-both-3level-30-20-10+decoder-cov+adv+mmb0-1b_smb1-1b_1p",
    "dualvq+small-h64+rvq-both-3level-30-20-10+decoder-cov+adv-w300+mmb0-1b_smb1-1b_1p",
    "dualvq+small-h64+rvq-both-3level-30-20-10+decoder-cov+adv+mmd-w50+mmb0-1b_smb1-1b_1p",
    "dualvq+small-h64+rvq-both-3level-30-20-10+decoder-cov+mmd-w100+mmb0-1b_smb1-1b_1p",
]

# v11 — single-variant smoke-test sweep used to verify the updated
# DataLoader config (num_workers=8 / persistent_workers=True /
# pin_memory=True) and the 10-CPU LSF allocation actually take effect.
# Picks one representative variant from v9 (medium architecture +
# default adversarial); the exact choice is incidental — any variant
# would do. Useful as the first job to submit after the DataLoader
# change lands, before re-running the full sweeps.
# Submit via:
#   bash examples/submit_mmb_smb_ablations_v11.sh
DATASET_VARIANTS["mmb0-1b_smb1-1b_1p-ablations-v11"] = [
    "dualvq+small-h64+rvq-both-3level-30-20-10+decoder-cov+adv+mmb0-1b_smb1-1b_1p",
]

# v10 — "is spatial supervision hurting cell-type NMI?" diagnostic.
# 4 variants on a fixed medium architecture (h64 + codebook (30, 20, 10))
# crossed over { adv vs mmd-w100 } x { +no-spatial vs +no-adj }.
# Architectural size is held constant so the diagnostic isolates the
# effect of LOSSES (not capacity) on cell-type NMI.
# Submit all 4 via:
#   bash examples/submit_mmb_smb_ablations_v10.sh
DATASET_VARIANTS["mmb0-1b_smb1-1b_1p-ablations-v10"] = [
    # +no-spatial: drops nbr-NB + niche-commit + adj-BCE (all spatial losses).
    "dualvq+small-h64+rvq-both-3level-30-20-10+decoder-cov+adv+no-spatial+mmb0-1b_smb1-1b_1p",
    "dualvq+small-h64+rvq-both-3level-30-20-10+decoder-cov+mmd-w100+no-spatial+mmb0-1b_smb1-1b_1p",
    # +no-adj: keeps nbr-NB + niche-commit, drops ONLY adjacency BCE.
    "dualvq+small-h64+rvq-both-3level-30-20-10+decoder-cov+adv+no-adj+mmb0-1b_smb1-1b_1p",
    "dualvq+small-h64+rvq-both-3level-30-20-10+decoder-cov+mmd-w100+no-adj+mmb0-1b_smb1-1b_1p",
]

# Held-out-region downstream task. Currently a single variant — the
# 1-layer-spine baseline (RVQ-both + decoder-cov + adv) with per-batch
# spatial holdout patches. Add more entries here later (e.g. larger
# patches, different anatomical zones) without touching the submitter.
DATASET_VARIANTS["mmb0-1b_smb1-1b_1p-region-holdout"] = [
    "dualvq+rvq-both+decoder-cov+adv+region-holdout+mmb0-1b_smb1-1b_1p",
]


def _ablation_summary(variant: str) -> dict:
    """A small, human-readable record of what this run is. Saved next to the
    full materialised config so you can grep / cat the summary without
    diffing the whole config across runs."""
    spec = VARIANTS[variant]
    return {
        "variant": variant,
        "description": spec["description"],
        "patches_applied": spec["patches"],
    }


# ---------------------------------------------------------------------------
# Optional helper: patch .uns metadata if your harmonised files don't have it
# ---------------------------------------------------------------------------

def patch_anndata_uns():
    """
    Run this ONCE if your two .h5ad files are missing the .uns keys SQUINT
    needs. Skip it if they're already set.
    """
    import anndata as ad

    silver = DATA_ROOT / "silver" / DATASET_NAME

    files = {
        silver / "harmonised_merfish_mouse_brain_239-batch_batch82_shared_genes.h5ad": {
            "batch": "batch82",
            "dataset_id": "mmb0",
            "tissue": "mouse_brain",
            "species": "mouse",
        },
        silver / "harmonised_starmap_plus_mouse_cns_batch15_shared_genes.h5ad": {
            "batch": "batch15",
            "dataset_id": "smb1",
            "tissue": "mouse_brain",
            "species": "mouse",
        },
    }

    for path, uns in files.items():
        adata = ad.read_h5ad(path)
        for k, v in uns.items():
            adata.uns[k] = v
        # SQUINT expects obs['cell_id'] containing "batchN"; if missing, fabricate.
        if "cell_id" not in adata.obs.columns:
            adata.obs["cell_id"] = [
                f"{uns['dataset_id']}_{uns['batch']}_{i}"
                for i in range(adata.n_obs)
            ]
        adata.write_h5ad(path)
        print(f"patched {path.name}: uns={uns}, n_obs={adata.n_obs}")


def harmonize_anndata_var():
    """
    Make `.var` identical across all AnnData files in the silver folder.

    InMemoryDatasetBlob.process() requires `adata.var` to be *strictly* equal
    across batches (`pandas.DataFrame.equals`) — not just the gene set, but
    the row order, the column set and the column dtypes.  MERFISH vs.
    STARmap+ files almost always disagree on extra `.var` metadata
    (probe ids, QC columns, ensembl ids only on one platform, etc.) and
    that's enough to trigger:

        ValueError: All batches must have the same gene panel

    This helper:
      1. computes the intersection of gene names across files,
      2. subsets each AnnData to those genes in a *common sorted order*,
      3. replaces `.var` with a fresh DataFrame whose only content is that
         shared, sorted gene index (no extra columns), and
      4. writes the harmonised file back in place.

    Run this ONCE before --build-blob if you hit the panel-mismatch error.
    """
    import anndata as ad
    import pandas as pd

    silver = DATA_ROOT / "silver" / DATASET_NAME
    files = sorted(silver.glob("*.h5ad"))
    if len(files) < 2:
        raise SystemExit(f"Expected >=2 .h5ad files under {silver}, found {len(files)}")

    # 1. Read once just to compute the shared gene set.
    var_indices = []
    for p in files:
        a = ad.read_h5ad(p, backed="r") if hasattr(ad, "read_h5ad") else ad.read(p)
        var_indices.append(set(a.var_names))
        a.file.close() if a.isbacked else None

    shared = sorted(set.intersection(*var_indices))
    if not shared:
        raise SystemExit("No genes are shared across all .h5ad files; cannot harmonise.")
    print(f"Shared genes: {len(shared)} (out of {[len(v) for v in var_indices]})")

    # 2. + 3. + 4. Subset each file to the shared gene list, drop .var columns.
    for p in files:
        a = ad.read_h5ad(p)
        before_n_vars = a.n_vars
        a = a[:, shared].copy()                  # subset + force same row order
        a.var = pd.DataFrame(index=a.var_names)  # drop all .var columns
        # Sanity: also clear any varm / varp that would still depend on old order.
        for k in list(a.varm.keys()):
            del a.varm[k]
        for k in list(a.varp.keys()):
            del a.varp[k]
        a.write_h5ad(p)
        print(f"harmonised {p.name}: n_vars {before_n_vars} -> {a.n_vars} "
              f"(.var has {len(a.var.columns)} columns)")


# ---------------------------------------------------------------------------
# Build dataset blob (one-time preprocessing)
# ---------------------------------------------------------------------------

def build_blob(dataset: str = "mmb0-1b_smb1-1b_1p"):
    """
    Build the in-memory PyG DatasetBlob in-process.

    `dataset` selects which builder config to use:
      - 'mmb0-1b_smb1-1b_1p' (default): MERFISH MB + STARmap MB
                               (`mmb0-1b_smb1-1b_1p_coord_aligned`).
      - 'chl59-8b_1p'        : CosMx Lung 8-sample dataset (`chl59-8b_1p`),
                               built from /nfs/team361/sb75/DATASETS/silver/chl59-8b_1p
      - 'mmb0-1b_smb1-20b_1p': MERFISH MB + 20-section STARmap+ CNS dataset
                               (`mmb0-1b_smb1-20b_1p_shared_genes`).

    Implementation note: previously this shelled out to
        squint-reproducibility/analysis/create_in_memory_dataset_blob.py
    via subprocess.  That works only if the analysis script on the cluster
    is in sync with the squint package — and it isn't always.  To avoid
    cross-repo file-sync issues we instead instantiate ``InMemoryDatasetBlob``
    here directly with the same arguments.
    """
    CONFIG_OUT_DIR.mkdir(parents=True, exist_ok=True)
    if dataset == "chl59-8b_1p":
        cfg = make_dataset_blob_config_chl59()
        cfg_path = CONFIG_OUT_DIR / "build_blob_chl59-8b_1p.yaml"
    elif dataset == "mmb0-1b_smb1-20b_1p":
        cfg = make_dataset_blob_config_mmb20()
        cfg_path = CONFIG_OUT_DIR / "build_blob_mmb0-1b_smb1-20b_1p.yaml"
    elif dataset == "mmb0-1b_smb1-1b_1p":
        cfg = make_dataset_blob_config()
        cfg_path = CONFIG_OUT_DIR / "build_blob_mmb0-1b_smb1-1b_1p.yaml"
    else:
        raise ValueError(
            f"Unknown --build-blob-dataset {dataset!r}. Choices: "
            f"'mmb0-1b_smb1-1b_1p', 'chl59-8b_1p', 'mmb0-1b_smb1-20b_1p'."
        )
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"Wrote dataset-blob config to {cfg_path}")

    # Make squint's vqniche package importable (in case the active venv hasn't
    # been pip-installed editable).
    if str(SQUINT_PKG / "src") not in sys.path:
        sys.path.insert(0, str(SQUINT_PKG / "src"))

    from vqniche.dataset.in_memory_dataset_blob import InMemoryDatasetBlob

    ds_cfg = cfg["dataset"]
    print(f"Building dataset blob into "
          f"{Path(ds_cfg['data_directory_path']) / 'gold' / 'in-memory-PyG-dataset-blob' / ds_cfg['name']}")

    dataset_blob = InMemoryDatasetBlob(
        name=ds_cfg["name"],
        feature_names=ds_cfg["feature_names"],
        label_names=ds_cfg["label_names"],
        graph_kwargs=ds_cfg["graph_kwargs"],
        data_directory_path=ds_cfg["data_directory_path"],
        pre_transform=ds_cfg["pre_transform"],
        pre_filter=ds_cfg["pre_filter"],
        overwrite=ds_cfg["overwrite"],
        software_paths=cfg["software_paths"],
    )

    for data_batch in dataset_blob:
        print(f"Batch: {data_batch.adata_batch_id}")
        print(f"Data: {data_batch}")

    print(f"Processed data saved at {dataset_blob.processed_dir}")


# ---------------------------------------------------------------------------
# Train (in-process so we control the wandb project AND the artifact layout)
# ---------------------------------------------------------------------------

def train(variant: str):
    """
    Train SQUINT.

    Artifact layout (flat, NOT controlled by wandb; ablation-aware):
        <ARTIFACTS_DIR>/<DATASET_NAME>/<VARIANT>/<YYYYMMDD_HHMMSS>/
            user_specified_config.yaml       <- canonical config for predict()
            ablation_summary.yaml            <- variant + description + patches
            checkpoints/<best>.ckpt          <- pl.callbacks.ModelCheckpoint
            wandb/run-<id>/...               <- wandb local cache (logger)

    Cloud logging to the "squint" wandb project still works via WandbLogger.
    The directory printed at the end is what you pass to --run-dir for
    inference; predict() expects the flat layout above (no `files/` subdir).
    """
    # Make the squint package importable in case the active venv hasn't been
    # pip-installed editable. We no longer import anything from
    # squint-reproducibility/analysis here — train() is fully in-process.
    if str(SQUINT_PKG / "src") not in sys.path:
        sys.path.insert(0, str(SQUINT_PKG / "src"))

    from datetime import datetime

    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import WandbLogger

    from vqniche.initializers.initialize import (
        initialize_dataset_blob,
        initialize_databatch,
        initialize_datamodule,
        initialize_model,
    )

    # Determinism / numerical settings (mirror train_model.__main__).
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("medium")

    # ---- Resolve variant from registry ------------------------------------
    if variant not in VARIANTS:
        raise SystemExit(
            f"Unknown variant '{variant}'. Known variants: "
            f"{sorted(VARIANTS.keys())}. (Add new variants to the VARIANTS "
            f"registry near the top of this file.)"
        )
    cfg = VARIANTS[variant]["build"]()
    summary = _ablation_summary(variant)

    # ---- Resolve our own RUN_DIR ------------------------------------------
    # Each variant gets its own subdir under the dataset, so different
    # ablations don't collide and we can grep run dirs by variant.
    # Filesystem-safe slug: 'poc+nbr' -> 'poc+nbr' is fine on POSIX, but we
    # play it safe and keep '+' (no spaces, no slashes).
    variant_slug = variant.replace("/", "_").replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Route artifacts under the variant's CHOSEN dataset, using the
    # short `dataset_tag` (e.g. `mmb0-1b_smb1-1b_1p`) rather than the
    # full `dataset_name` (e.g. `mmb0-1b_smb1-1b_1p_coord_aligned`). The
    # tag drops technical suffixes (`_coord_aligned`, `_shared_genes`)
    # that belong on the silver / blob paths but make on-disk artifact
    # dirs unwieldy. Falls back to `dataset_name` if a variant doesn't
    # set a tag.
    run_dataset_tag = cfg["dataset"].get(
        "dataset_tag",
        cfg["dataset"].get("dataset_name", DATASET_NAME),
    )
    run_dir = ARTIFACTS_DIR / run_dataset_tag / variant_slug / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Persist the full materialised config in TWO places:
    #   (a) the canonical location predict() expects (flat, inside run_dir),
    #   (b) an at-a-glance copy under <ARTIFACTS_DIR>/configs/ for diffing
    #       across runs.
    config_path_in_run = run_dir / "user_specified_config.yaml"
    with open(config_path_in_run, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    # Ablation summary lives next to the config so a quick `cat` tells you
    # what kind of run this was without diffing the whole config.
    summary_path = run_dir / "ablation_summary.yaml"
    with open(summary_path, "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)

    CONFIG_OUT_DIR.mkdir(parents=True, exist_ok=True)
    config_path_archive = CONFIG_OUT_DIR / f"train_{variant_slug}_{timestamp}.yaml"
    with open(config_path_archive, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print(f"Variant: {variant}")
    print(f"  description: {summary['description']}")
    print(f"  patches:     {summary['patches_applied']}")
    print(f"Run directory: {run_dir}")
    print(f"  config:      {config_path_in_run}")
    print(f"  summary:     {summary_path}")
    print(f"  checkpoints: {ckpt_dir}")
    print(f"  archive:     {config_path_archive}")

    # ---- Resolve HVG transform based on apply_hvg flag --------------------
    # The config stores apply_hvg and n_hvg as first-class keys so variants
    # can toggle them via a simple patch, without duplicating the full
    # gene_count_transform_names list.  We materialise the transform list here
    # just before handing the config to the initializers.
    if cfg["dataset"].get("apply_hvg", False):
        cfg["dataset"]["gene_count_transform_names"] = ["SubsetHVG"]
        cfg["dataset"]["gene_count_transform_params"] = {
            "n_genes": cfg["dataset"].get("n_hvg", 2000)
        }
    else:
        cfg["dataset"]["gene_count_transform_names"] = []
        cfg["dataset"]["gene_count_transform_params"] = {}

    # ---- Determinism (mirror train_model.train) ---------------------------
    pl.seed_everything(cfg["experiment"]["seed"])

    # ---- Dataset / Databatch / Datamodule ---------------------------------
    dataset_blob = initialize_dataset_blob(cfg)

    # Translate `train_batches` / `val_batches` (which are stored as
    # `adata_batch_id` values — what `SpatialBatchSplit` matches against
    # `data.adata_batch_id`) into the position indices that
    # `cfg["dataset"]["adata_batch_idx"]` actually expects. The mapping
    # is dataset-dependent: it depends on the value of
    # `data.adata_batch_id` for each section, which is only known after
    # the blob is loaded. Doing this at runtime avoids hardcoded
    # id-to-position helpers per dataset (which were fragile — e.g.
    # mmb20 uses parsed `uns['batch']` ids while chl59 falls back to
    # file alphabetical position).
    #
    # `test_batches` are NOT loaded — they're held out from training
    # entirely and only consumed by predict().
    #
    # `train_batches` is OPTIONAL: when omitted (or empty) the training
    # set defaults to every blob section that isn't in `test_batches` /
    # `val_batches`. This lets variants only specify what's held out for
    # evaluation; the train side is auto-derived from the blob. If neither
    # train nor test is set, every section is used for training.
    _ts_params = cfg["dataset"].get("train_transform_params", {})
    _train_ids = list(_ts_params.get("train_batches", []) or [])
    _val_ids   = list(_ts_params.get("val_batches",   []) or [])
    _test_ids  = list(_ts_params.get("test_batches",  []) or [])

    _id_to_pos: dict = {}
    for _pos in range(len(dataset_blob)):
        _d = dataset_blob[_pos]
        _id_to_pos[int(_d.adata_batch_id)] = _pos
    _all_blob_ids = sorted(_id_to_pos.keys())

    if not _train_ids:
        _excluded = set(int(i) for i in _test_ids) | set(int(i) for i in _val_ids)
        _train_ids = [int(i) for i in _all_blob_ids if int(i) not in _excluded]
        cfg["dataset"]["train_transform_params"]["train_batches"] = list(_train_ids)
        print(
            f"train_batches not specified -> defaulting to all blob "
            f"sections except test+val: {_train_ids}"
        )

    _needed_ids = list(dict.fromkeys(_train_ids + _val_ids))
    if _needed_ids:
        _missing = [i for i in _needed_ids if int(i) not in _id_to_pos]
        if _missing:
            raise RuntimeError(
                f"Requested adata_batch_ids {_missing} are not present "
                f"in the dataset blob. Available ids (in the blob): "
                f"{_all_blob_ids}. Either rebuild the blob or "
                f"adjust the train/val/test ids in the variant patch."
            )
        cfg["dataset"]["adata_batch_idx"] = sorted(set(
            _id_to_pos[int(i)] for i in _needed_ids
        ))
        print(
            f"Resolved adata_batch_idx from train+val ids "
            f"{_needed_ids} -> positions "
            f"{cfg['dataset']['adata_batch_idx']} via blob's "
            f"adata_batch_id mapping."
        )

    # Re-save the (now fully-resolved) config. The earlier dump at
    # `config_path_in_run` was made BEFORE this block ran — at that
    # point `adata_batch_idx=[]` (placeholder set by the dataset patch
    # since positions can only be derived after the blob is loaded) and
    # `train_batches` may have been empty (auto-default fills it from
    # blob ids minus test/val). `predict()` reads
    # `user_specified_config.yaml` to know which sections to run
    # inference on; if we leave the placeholder there, predict() loads
    # zero sections and crashes with
    # `batch_size=0 (BatchSampler)` deep inside torch_geometric.
    # The archive copy gets the same refresh so on-disk diffs across
    # runs remain comparable.
    with open(config_path_in_run, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    with open(config_path_archive, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    # Re-init the blob now that `train_transform_params['train_batches']`
    # is fully resolved. `initialize_dataset_blob` builds the
    # `SpatialBatchSplit` transform from the cfg AT CONSTRUCTION TIME
    # and stores it on the blob; mutating the cfg after the fact does
    # NOT update the live transform, so the auto-defaulted train_batches
    # would be ignored — every section's `SpatialBatchSplit.forward()`
    # check would miss, no `train_mask` would be set, and the next step
    # would AttributeError on `data.train_mask`.
    # Re-initialising is cheap: the on-disk processed blob is reused, only
    # the transform pipeline is rebuilt.
    dataset_blob = initialize_dataset_blob(cfg)

    data_batch = initialize_databatch(config=cfg, dataset_blob=dataset_blob)
    datamodule_batch = initialize_datamodule(
        config=cfg,
        data=data_batch,
        obs_per_batch_id=getattr(dataset_blob, 'obs_per_batch_id', None),
    )

    # ---- Model: bind condition dims if FiLM is enabled --------------------
    if "conditioning_params" in cfg["model"]["encoder_params"]:
        cfg["model"]["encoder_params"]["conditioning_params"]["condition_dim"] = (
            data_batch.encoder_condition_dim
        )
    if "spatial_prior_params" in cfg["model"]["encoder_params"]:
        cfg["model"]["encoder_params"]["spatial_prior_params"][
            "spatial_prior_feature_dim"
        ] = data_batch.spatial_prior_feature_dim
    # Helper: derive the number of distinct batches in this run from the
    # densified per-cell batch IDs. `adata_batch_ids` is ALWAYS populated
    # by `initialize_databatch` (regardless of whether encoder FiLM is on),
    # so this is a reliable source. `encoder_condition_dim` was the wrong
    # source — it only gets set when FiLM is enabled, which is not the
    # case for adversarial-only or covariate-only variants.
    def _n_distinct_batches(db) -> int:
        return int(db.adata_batch_ids.max().item()) + 1

    # NicheCompass-style decoder covariate (concat-based batch correction).
    # When the user enables it via _patch_dual_decoder_covariate, the dual
    # model expects `decoder_covariate_dim` to equal the dimensionality of
    # the per-cell batch one-hot. Set it from the data loader here.
    if (
        cfg["model"].get("model_name") == "VQNiche_Dual"
        and cfg["model"].get("decoder_covariate_dim_request") is True
    ):
        cfg["model"]["decoder_covariate_dim"] = _n_distinct_batches(data_batch)

    # Domain-adversarial batch-invariance head. When requested, sized to
    # the number of distinct batches.
    if (
        cfg["model"].get("model_name") == "VQNiche_Dual"
        and cfg["model"].get("adversarial_batch_dim_request") is True
    ):
        cfg["model"]["adversarial_batch_dim"] = _n_distinct_batches(data_batch)

    # Bind FiLM condition dims onto whichever decoder param dicts the
    # current config carries. Single-codebook configs (VQNiche etc.) use
    # `attribute_decoder_params` + `adjacency_decoder_params`. The
    # dual-codebook config (VQNiche_Dual) uses
    # `attribute_decoder_cell_params` + `attribute_decoder_niche_params`
    # and has no adjacency decoder.
    for dec_key in ("attribute_decoder_params",
                    "attribute_decoder_cell_params",
                    "attribute_decoder_niche_params"):
        dec_params = cfg["model"].get(dec_key)
        if dec_params is None:
            continue
        if "conditioning_params" in dec_params:
            dec_params["conditioning_params"]["condition_dim"] = (
                data_batch.attr_decoder_condition_dim
            )
    if "adjacency_decoder_params" in cfg["model"]:
        adj_params = cfg["model"]["adjacency_decoder_params"]
        if "conditioning_params" in adj_params:
            adj_params["conditioning_params"]["condition_dim"] = (
                data_batch.adj_decoder_condition_dim
            )

    model = initialize_model(
        config=cfg,
        in_channels=data_batch.num_features,
        out_channels=data_batch.num_classes,
    )

    # ---- Logger: wandb cloud logging, local cache lives inside run_dir ----
    logging_enabled = cfg["logging"].get("enabled", True)
    if logging_enabled:
        # Make timestamp prominent in the wandb run identity:
        #   - `name`: shows up as the run title in the wandb UI
        #   - `tags`: filterable tag (variant + timestamp)
        #   - `config["timestamp"]` and `config["run_dir"]`: top-level keys in
        #     the run's "Config" panel so the timestamp is the first thing
        #     visible alongside the variant.
        run_name = f"{variant}-{timestamp}"
        # Wandb caps tag length at 64 chars; truncate the `variant:` tag if
        # the variant slug itself is long enough that the prefixed tag
        # would overflow. The full variant name is still in `name=run_name`
        # (no length cap), in `config["variant"]`, and in the run_dir path,
        # so nothing is lost — this only affects the filterable tag in the
        # wandb UI.
        def _wandb_tag(prefix: str, value: str, max_len: int = 64) -> str:
            tag = f"{prefix}{value}"
            if len(tag) <= max_len:
                return tag
            keep = max_len - len(prefix) - 1  # leave 1 char for an ellipsis marker
            return f"{prefix}{value[:keep]}~"
        logger = WandbLogger(
            save_dir=str(run_dir),                    # local cache -> <run_dir>/wandb/
            project=WANDB_PROJECT,
            name=run_name,
            # Group by dataset+batches so all ablations of one dataset cluster
            # together. The variant goes in `tags` so we can filter the wandb
            # UI to e.g. all "poc+nbr" runs across timestamps.
            group=f"{cfg['dataset']['dataset_name']}:batch={cfg['dataset']['adata_batch_idx']}",
            job_type="train",
            tags=[
                _wandb_tag("variant:",   variant),
                _wandb_tag("timestamp:", timestamp),
            ],
            mode="offline" if cfg["logging"]["offline"] else "online",
            log_model=cfg["logging"]["log_model"],
            config={
                # Hoist run identity to the top of the wandb Config panel.
                "timestamp":         timestamp,
                "variant":           variant,
                "run_dir":           str(run_dir),
                "ablation_summary":  summary,
                **cfg,
            },
        )
    else:
        logger = False

    # ---- Callbacks: checkpoint + early stopping ----------------------------
    callbacks: list = []

    enable_checkpointing = cfg["trainer"]["enable_checkpointing"]
    if enable_checkpointing:
        monitor = cfg["trainer"]["monitor"]
        # filename templates known to ModelCheckpoint -> map monitor metric to
        # a filename pattern PL can format. Fall back to val_loss.
        _filename_by_monitor = {
            "val_loss":                                 "{epoch}-{val_loss:.3f}",
            "train_loss":                               "{epoch}-{train_loss:.3f}",
            "val_pearson_cell_wise":                    "{epoch}-{val_pearson_cell_wise:.2f}",
            "val_pearson_1hop_nbr":                     "{epoch}-{val_pearson_1hop_nbr:.2f}",
            "val_pearson_gene_wise":                    "{epoch}-{val_pearson_gene_wise:.2f}",
            "val_pearson_gene_wise_1hop_nbr":           "{epoch}-{val_pearson_gene_wise_1hop_nbr:.2f}",
            "train_pearson_cell_wise":                  "{epoch}-{train_pearson_cell_wise:.2f}",
            "train_pearson_1hop_nbr":                   "{epoch}-{train_pearson_1hop_nbr:.2f}",
            "train_pearson_gene_wise":                  "{epoch}-{train_pearson_gene_wise:.2f}",
            "train_pearson_gene_wise_1hop_nbr":         "{epoch}-{train_pearson_gene_wise_1hop_nbr:.2f}",
            # log1p variants — cell branch
            "train_pearson_gene_wise_log1p":                    "{epoch}-{train_pearson_gene_wise_log1p:.3f}",
            "train_pearson_gene_wise_log1p_median":             "{epoch}-{train_pearson_gene_wise_log1p_median:.3f}",
            "train_pearson_gene_wise_hvg50_log1p":              "{epoch}-{train_pearson_gene_wise_hvg50_log1p:.3f}",
            "train_pearson_gene_wise_hvg50_log1p_median":       "{epoch}-{train_pearson_gene_wise_hvg50_log1p_median:.3f}",
            "train_pearson_cell_wise_log1p":                    "{epoch}-{train_pearson_cell_wise_log1p:.3f}",
            "train_pearson_cell_wise_log1p_median":             "{epoch}-{train_pearson_cell_wise_log1p_median:.3f}",
            # log1p variants — neighbourhood branch
            "train_pearson_gene_wise_1hop_nbr_log1p":           "{epoch}-{train_pearson_gene_wise_1hop_nbr_log1p:.3f}",
            "train_pearson_gene_wise_1hop_nbr_log1p_median":    "{epoch}-{train_pearson_gene_wise_1hop_nbr_log1p_median:.3f}",
            "train_pearson_gene_wise_hvg50_1hop_nbr_log1p":     "{epoch}-{train_pearson_gene_wise_hvg50_1hop_nbr_log1p:.3f}",
            "train_pearson_gene_wise_hvg50_1hop_nbr_log1p_median": "{epoch}-{train_pearson_gene_wise_hvg50_1hop_nbr_log1p_median:.3f}",
            "train_pearson_cell_wise_1hop_nbr_log1p":           "{epoch}-{train_pearson_cell_wise_1hop_nbr_log1p:.3f}",
            "train_pearson_cell_wise_1hop_nbr_log1p_median":    "{epoch}-{train_pearson_cell_wise_1hop_nbr_log1p_median:.3f}",
            # val equivalents
            "val_pearson_gene_wise_log1p":                      "{epoch}-{val_pearson_gene_wise_log1p:.3f}",
            "val_pearson_gene_wise_log1p_median":               "{epoch}-{val_pearson_gene_wise_log1p_median:.3f}",
            "val_pearson_gene_wise_hvg50_log1p":                "{epoch}-{val_pearson_gene_wise_hvg50_log1p:.3f}",
            "val_pearson_gene_wise_hvg50_log1p_median":         "{epoch}-{val_pearson_gene_wise_hvg50_log1p_median:.3f}",
            "val_pearson_cell_wise_log1p":                      "{epoch}-{val_pearson_cell_wise_log1p:.3f}",
            "val_pearson_cell_wise_log1p_median":               "{epoch}-{val_pearson_cell_wise_log1p_median:.3f}",
            "val_pearson_gene_wise_1hop_nbr_log1p":             "{epoch}-{val_pearson_gene_wise_1hop_nbr_log1p:.3f}",
            "val_pearson_gene_wise_1hop_nbr_log1p_median":      "{epoch}-{val_pearson_gene_wise_1hop_nbr_log1p_median:.3f}",
            "val_pearson_gene_wise_hvg50_1hop_nbr_log1p":       "{epoch}-{val_pearson_gene_wise_hvg50_1hop_nbr_log1p:.3f}",
            "val_pearson_gene_wise_hvg50_1hop_nbr_log1p_median": "{epoch}-{val_pearson_gene_wise_hvg50_1hop_nbr_log1p_median:.3f}",
            "val_pearson_cell_wise_1hop_nbr_log1p":             "{epoch}-{val_pearson_cell_wise_1hop_nbr_log1p:.3f}",
            "val_pearson_cell_wise_1hop_nbr_log1p_median":      "{epoch}-{val_pearson_cell_wise_1hop_nbr_log1p_median:.3f}",
        }
        filename = _filename_by_monitor.get(
            monitor, "{epoch}-{val_loss:.3f}"
        )
        if monitor not in _filename_by_monitor:
            monitor = "val_loss"

        checkpoint_params = cfg["trainer"]["checkpoint_params"]
        callbacks.append(
            pl.callbacks.ModelCheckpoint(
                dirpath=str(ckpt_dir),
                monitor=monitor,
                filename=filename,
                **checkpoint_params,
            )
        )

    # ---- Early-stopping callback -------------------------------------------
    es_cfg = cfg["trainer"].get("early_stopping_params", {})
    if es_cfg.get("enabled", True):
        callbacks.append(
            pl.callbacks.EarlyStopping(
                monitor   = es_cfg.get("monitor", "train_nb_attribute_reconstruction_loss"),
                mode      = es_cfg.get("mode", "min"),
                patience  = es_cfg.get("patience", 20),
                min_delta = es_cfg.get("min_delta", 0.1),
                verbose   = True,
            )
        )

    callbacks = callbacks or None   # PL wants None, not [], when empty

    # ---- Trainer ----------------------------------------------------------
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        deterministic=True,
        logger=logger,
        callbacks=callbacks,
        strategy="ddp_find_unused_parameters_true",
        max_epochs=cfg["trainer"]["max_epochs"],
        enable_checkpointing=enable_checkpointing,
        num_sanity_val_steps=0,
        enable_progress_bar=False,
        enable_model_summary=True,
    )

    # ---- Fit --------------------------------------------------------------
    print("Training Model...")
    try:
        trainer.fit(model=model, datamodule=datamodule_batch)
    finally:
        # WandbLogger.experiment is the `wandb.Run`; closing it flushes uploads.
        if logging_enabled and logger is not False:
            try:
                logger.experiment.finish()
            except Exception:
                pass

    print()
    print("=" * 78)
    print("Training finished. Run directory (pass to --run-dir when predicting):")
    print(f"  {run_dir}")
    print("=" * 78)
    return run_dir


# ---------------------------------------------------------------------------
# Predict / inference: folder of .h5ad -> single AnnData with SQUINT outputs
# ---------------------------------------------------------------------------

def _infer_adata_files_in_dir(silver_dir: Path) -> list[Path]:
    files = sorted(p for p in silver_dir.glob("*.h5ad"))
    if not files:
        raise FileNotFoundError(f"No .h5ad files found in {silver_dir}")
    return files


def _build_clean_adata_from_inference(
    inference_data: dict,
    source_adatas: dict,
    id_to_path: dict,
    gene_names: list | None = None,
    test_batch_ids: list | None = None,
    test_regions: dict | None = None,
    # Legacy kwargs kept for signature back-compat — ignored.
    label_categories_dict: dict | None = None,
    source_paths: list | None = None,
    obs_per_batch_id: dict | None = None,
    obs_per_filename: dict | None = None,
) -> "ad.AnnData":
    """
    Build the predicted AnnData by starting from the source AnnDatas
    themselves and bolting model outputs on top.

    Strategy
    --------
    1. For each cell in the inference output (ordered by
       `inference_data["adata_batch_ids"]` + `obs_row_index`), slice
       the corresponding row out of its source AnnData. Sub-AnnDatas
       are concatenated and reordered to match the inference cell order.
    2. If `gene_names` is supplied (typically the canonical / HVG gene
       panel saved at blob-build time), reindex `var` to that set so
       `adata.X` matches the model's input dimensionality (and therefore
       so do the model output layers).
    3. Stamp model outputs into `obsm` / `obs` / `layers` / `uns`. The
       `data_split` column is derived from `test_batch_ids` (which the
       saved training config carries forward via `train_transform_params`'
       `test_batches`).

    Why this is preferable to synthesising an AnnData from torch tensors:
    every original `obs` column (e.g. `cell_type`, `Sub_molecular_*`,
    `ccf_region_name`, `donor_id`, …) and `var` metadata flows through
    automatically. `ad.concat(..., join="outer")` handles partially-
    present columns by NaN-filling cells from sources that don't carry
    them. No manual obs propagation, no h5py dtype-coercion gymnastics,
    no failure modes from mismatched id-derivations between
    `obs_per_batch_id.pkl` and `dataset_blob.pt`.

    Parameters
    ----------
    inference_data:
        The collated dict produced by `collate_predict_outputs`. Must
        contain `adata_batch_ids` and `obs_row_index` per cell.
    source_adatas:
        Mapping `adata_batch_id -> ad.AnnData` for every section the
        cells in `inference_data` came from. The id derivation must
        match `InMemoryDatasetBlob._derive_adata_batch_id`; use
        `predict()`'s helper to build it.
    id_to_path:
        Mapping `adata_batch_id -> Path` of the source files. Used to
        stamp `adata.obs["source_file"]` per cell.
    gene_names:
        Optional canonical gene panel (typically HVG-subset or full
        panel) loaded from `dataset_blob.processed_dir/gene_panel.pkl`.
        When set, `var` is reindexed to this list so model outputs in
        `layers` (which are in this gene-space) match `var` length.
    test_batch_ids:
        `adata_batch_id` values that were held out from training; cells
        from these sections get `obs["data_split"] = "test"`,
        everything else is "train".
    test_regions:
        Optional mapping `adata_batch_id -> region_spec` for the held-
        out-region downstream task (`SpatialBatchSplit.test_regions`).
        Cells whose `obsm["spatial"]` falls inside the region for their
        batch ALSO get `data_split = "test"` (in addition to whatever
        `test_batch_ids` already marked). Region keys: `x_min`, `x_max`,
        `y_min`, `y_max` (absolute) and/or their `_pct` percentile
        equivalents (resolved against this batch's per-section coordinate
        range — same convention as the training-time
        `_is_in_single_region`).
    """
    import anndata as ad
    import numpy as np
    import pandas as pd

    def to_np(x):
        # Avoid an unnecessary GPU->CPU copy when the tensor is already
        # on CPU. The predict-collator path already moves cached
        # tensors to CPU; an extra .cpu() call is a no-op semantically
        # but allocates a fresh tensor under the hood (multi-GB on
        # mmb20 across several layers + obsm slots).
        if hasattr(x, "cpu"):
            return x.numpy() if not x.is_cuda else x.cpu().numpy()
        return np.asarray(x)

    # ---- sanity checks on inputs ----
    if "adata_batch_ids" not in inference_data:
        raise RuntimeError(
            "_build_clean_adata_from_inference: inference_data is missing "
            "'adata_batch_ids' — cannot reconstruct cell -> source mapping."
        )
    if "obs_row_index" not in inference_data:
        raise RuntimeError(
            "_build_clean_adata_from_inference: inference_data is missing "
            "'obs_row_index'. Rebuild the dataset blob with the latest "
            "InMemoryDatasetBlob.process_anndata_batch (which stamps "
            "obs_row_index per cell)."
        )

    batch_ids_np = to_np(inference_data["adata_batch_ids"]).astype(int)
    row_idx_np   = to_np(inference_data["obs_row_index"]).astype(int)
    n_cells      = len(batch_ids_np)
    if len(row_idx_np) != n_cells:
        raise RuntimeError(
            f"adata_batch_ids has length {n_cells} but obs_row_index has "
            f"length {len(row_idx_np)} — they must be aligned."
        )

    # ---- group cells by source section ----
    by_bid: dict = {}
    for k in range(n_cells):
        by_bid.setdefault(int(batch_ids_np[k]), []).append(
            (k, int(row_idx_np[k]))
        )

    missing = [b for b in by_bid if b not in source_adatas]
    if missing:
        raise RuntimeError(
            f"Cells reference adata_batch_ids {sorted(missing)} but no "
            f"source AnnData was provided for those ids. Available source "
            f"ids: {sorted(source_adatas)}. Check that the silver dir "
            f"contains the files used at blob-build time."
        )

    # ---- slice + var-reindex each source ----
    sub_adatas = []
    for bid in sorted(by_bid):
        cells       = by_bid[bid]
        inf_pos     = np.array([c[0] for c in cells], dtype=int)
        src_rows    = [c[1] for c in cells]
        source      = source_adatas[bid]

        # Bounds check — better here than at concat time when the error
        # would be opaque.
        if any(r < 0 or r >= source.n_obs for r in src_rows):
            bad = [r for r in src_rows if r < 0 or r >= source.n_obs]
            raise RuntimeError(
                f"obs_row_index out of bounds for adata_batch_id={bid} "
                f"(n_obs in source = {source.n_obs}); offending rows: "
                f"{bad[:5]}{'...' if len(bad) > 5 else ''}. Likely the "
                f"silver file changed since the blob was built — rebuild."
            )

        sub = source[src_rows].copy()
        if gene_names is not None:
            if list(sub.var_names) != list(gene_names):
                try:
                    sub = sub[:, list(gene_names)].copy()
                except KeyError as exc:
                    raise RuntimeError(
                        f"Source for adata_batch_id={bid} (file "
                        f"{id_to_path[bid].name}) is missing genes from the "
                        f"canonical panel: {exc}"
                    )
            # After reindex, var_names MUST match the canonical panel
            # exactly (same order, same set). Mismatch here means the
            # earlier reindex failed silently — better to crash than to
            # write a misaligned predicted_adata.h5ad whose `.X` and
            # `.layers['X_hat']` live in different gene orders.
            if list(sub.var_names) != list(gene_names):
                raise RuntimeError(
                    f"Gene-order mismatch after reindex for "
                    f"adata_batch_id={bid} (file {id_to_path[bid].name}): "
                    f"sub.var_names[:5]={list(sub.var_names)[:5]}, "
                    f"gene_names[:5]={list(gene_names)[:5]}. The source "
                    f"AnnData's `.var` is incompatible with the canonical "
                    f"gene panel; rebuild the dataset blob or restore the "
                    f"silver file's gene order."
                )

        # Track inference position so we can reorder after concat. Stored
        # in obs (so it survives concat); deleted just before return.
        sub.obs["_inf_pos"] = inf_pos
        sub_adatas.append(sub)

    # ---- concatenate + reorder to inference cell order ----
    if len(sub_adatas) == 1:
        adata = sub_adatas[0]
    else:
        # `join="outer"` so source-specific obs/var columns NaN-fill where
        # absent. `index_unique="-"` makes obs_names globally unique even
        # when sources share cell barcodes.
        adata = ad.concat(
            sub_adatas, axis=0, join="outer", index_unique="-",
        )

    perm = np.argsort(adata.obs["_inf_pos"].values)
    adata = adata[perm].copy()
    del adata.obs["_inf_pos"]

    if adata.n_obs != n_cells:
        raise RuntimeError(
            f"Reconstructed adata.n_obs={adata.n_obs} != inference n_cells="
            f"{n_cells}. Cell ordering broke during concat — this is a bug."
        )

    # ---- per-cell SQUINT outputs ---------------------------------------------

    # Spatial coordinates (overwrite source's `spatial` if present, since
    # the model emits its own copy of the same data).
    if "XY_coordinates" in inference_data:
        adata.obsm["spatial"] = to_np(inference_data["XY_coordinates"])

    # Per-cell scalars: which source section + filename + train/test.
    adata.obs["adata_batch_id"] = batch_ids_np.astype(int)
    name_for_id = {bid: id_to_path[bid].name for bid in id_to_path}
    adata.obs["source_file"] = pd.Categorical(
        [name_for_id[bid] for bid in batch_ids_np]
    )
    test_set = set(int(b) for b in (test_batch_ids or []))
    is_test = np.array(
        [int(b) in test_set for b in batch_ids_np], dtype=bool
    )

    # Held-out-region downstream task: mark cells whose spatial xy falls
    # inside per-batch test_regions. Resolve percentile bounds against
    # each section's actual coord range (mirrors SpatialBatchSplit's
    # `_is_in_single_region` semantics — keeping the predict-time data
    # split consistent with the train-time one).
    if test_regions and "spatial" in adata.obsm:
        regions_normed: dict = {int(k): v for k, v in test_regions.items()}
        spatial = np.asarray(adata.obsm["spatial"], dtype=float)
        if spatial.shape[1] >= 2:
            for bid, region_spec in regions_normed.items():
                cells_in_bid = (batch_ids_np.astype(int) == int(bid))
                if not cells_in_bid.any():
                    continue
                # Resolve per-batch xy range from THIS batch's cells (not
                # the global range) so percentile bounds line up with
                # what SpatialBatchSplit computed at training time.
                xy_b = spatial[cells_in_bid, :2]
                x_lo, y_lo = xy_b.min(axis=0)
                x_hi, y_hi = xy_b.max(axis=0)
                regions_list = (
                    region_spec if isinstance(region_spec, list)
                    else [region_spec]
                )
                in_any = np.zeros(spatial.shape[0], dtype=bool)
                for r in regions_list:
                    rx_min = float(r["x_min"]) if "x_min" in r and r["x_min"] is not None else (
                        x_lo + float(r["x_min_pct"]) * (x_hi - x_lo)
                        if "x_min_pct" in r and r["x_min_pct"] is not None else float("-inf")
                    )
                    rx_max = float(r["x_max"]) if "x_max" in r and r["x_max"] is not None else (
                        x_lo + float(r["x_max_pct"]) * (x_hi - x_lo)
                        if "x_max_pct" in r and r["x_max_pct"] is not None else float("inf")
                    )
                    ry_min = float(r["y_min"]) if "y_min" in r and r["y_min"] is not None else (
                        y_lo + float(r["y_min_pct"]) * (y_hi - y_lo)
                        if "y_min_pct" in r and r["y_min_pct"] is not None else float("-inf")
                    )
                    ry_max = float(r["y_max"]) if "y_max" in r and r["y_max"] is not None else (
                        y_lo + float(r["y_max_pct"]) * (y_hi - y_lo)
                        if "y_max_pct" in r and r["y_max_pct"] is not None else float("inf")
                    )
                    in_box = (
                        (spatial[:, 0] >= rx_min)
                        & (spatial[:, 0] <= rx_max)
                        & (spatial[:, 1] >= ry_min)
                        & (spatial[:, 1] <= ry_max)
                    )
                    in_any |= in_box
                # Region only applies to its own batch's cells.
                is_test |= (cells_in_bid & in_any)

    adata.obs["data_split"] = pd.Categorical(
        ["test" if t else "train" for t in is_test],
        categories=["train", "test"],
    )

    # Embeddings -> obsm.
    if "H_latent" in inference_data:
        adata.obsm["X_squint"]            = to_np(inference_data["H_latent"])
    if "H_quantized" in inference_data:
        adata.obsm["X_squint_quantized"]  = to_np(inference_data["H_quantized"])
    if "H_adj" in inference_data:
        adata.obsm["X_squint_adj"]        = to_np(inference_data["H_adj"])
    if "H_quantized_cell" in inference_data:
        adata.obsm["cell_emb"]            = to_np(inference_data["H_quantized_cell"])
    if "H_quantized_niche" in inference_data:
        adata.obsm["neighborhood_emb"]    = to_np(inference_data["H_quantized_niche"])
    if "H_latent_cell" in inference_data:
        adata.obsm["cell_latent"]         = to_np(inference_data["H_latent_cell"])
    if "H_latent_niche" in inference_data:
        adata.obsm["neighborhood_latent"] = to_np(inference_data["H_latent_niche"])

    # Codebook indices: 1D -> obs, multi-level -> obsm.
    if "Indices" in inference_data:
        idx = to_np(inference_data["Indices"])
        if idx.ndim == 1 or (idx.ndim == 2 and idx.shape[1] == 1):
            adata.obs["code_index"] = idx.reshape(-1).astype(int)
        else:
            adata.obsm["code_indices"] = idx.astype(int)
    for src_key, prefix in [("Indices_cell", "cell"),
                            ("Indices_niche", "neighborhood")]:
        if src_key not in inference_data:
            continue
        idx = to_np(inference_data[src_key])
        if idx.ndim == 1 or (idx.ndim == 2 and idx.shape[1] == 1):
            adata.obs[f"{prefix}_code_index"] = idx.reshape(-1).astype(int)
        else:
            adata.obsm[f"{prefix}_code_indices"] = idx.astype(int)

    # Reconstructions + 1-hop neighbourhood ground truth -> layers.
    if "X_hat" in inference_data:
        adata.layers["X_hat"]     = to_np(inference_data["X_hat"])
    if "X_hat_nbr" in inference_data:
        adata.layers["X_hat_nbr"] = to_np(inference_data["X_hat_nbr"])
    if "X_nbr" in inference_data:
        adata.layers["X_nbr"]     = to_np(inference_data["X_nbr"])

    # Global metadata.
    num_quantizers = int(inference_data.get("num_quantizers", 1))
    cb_sizes       = inference_data.get("codebook_sizes", None)
    squint_meta = {
        "codebook_size":     int(inference_data.get("codebook_size", 0)),
        "num_heads":         int(inference_data.get("num_heads", 1)),
        "num_quantizers":    num_quantizers,
        "codebook_sizes":    list(cb_sizes) if cb_sizes is not None else None,
        "separate_codebook": bool(inference_data.get("separate", False)),
    }
    is_dual = (
        "H_quantized_cell"  in inference_data
        or "H_quantized_niche" in inference_data
    )
    if is_dual:
        squint_meta["dual"] = True
        for branch in ("cell", "niche"):
            squint_meta[f"codebook_size_{branch}"] = int(
                inference_data.get(f"codebook_size_{branch}", 0)
            )
            squint_meta[f"num_quantizers_{branch}"] = int(
                inference_data.get(f"num_quantizers_{branch}", 1)
            )
            cb_sz = inference_data.get(f"codebook_sizes_{branch}", None)
            if cb_sz is not None:
                squint_meta[f"codebook_sizes_{branch}"] = list(cb_sz)
    adata.uns["squint"] = squint_meta
    if "edge_index" in inference_data:
        # Graph-level object — keep in uns.
        adata.uns["edge_index"] = to_np(inference_data["edge_index"])

    return adata


def predict(
    run_dir: str,
    silver_dir: str | None = None,
    model_ckpt_fname: str | None = None,
    output_dir: str | None = None,
):
    """
    Run inference on a folder of .h5ad files using a trained SQUINT
    checkpoint.  Returns a single AnnData with SQUINT outputs packaged
    into .obsm / .obs / .layers, and writes it to disk.

    Parameters
    ----------
    run_dir
        Path to the run directory of the trained model. Two layouts are
        supported (resolved automatically by the helpers in
        ``vqniche.utils.parse_test_configs``):
          (a) flat (current ``train()``): ``<run_dir>/user_specified_config.yaml``
              and ``<run_dir>/checkpoints/*.ckpt`` directly under run_dir.
          (b) legacy wandb-controlled: same files under ``<run_dir>/files/``.
    silver_dir
        Folder containing the .h5ad files to run inference on. Defaults to
        the silver folder this script was configured with.
    model_ckpt_fname
        Optional explicit checkpoint path. If omitted, the best checkpoint
        is auto-selected from ``<run_dir>/checkpoints/`` (or the legacy
        ``<run_dir>/files/checkpoints/``).
    output_dir
        Where to write predicted_adata.h5ad. Defaults to ``run_dir`` itself
        — predictions land alongside checkpoints / config /
        ablation_summary.yaml so everything for one training run lives in
        one folder.
    """
    # --- Imports done lazily so the script's CLI / config helpers don't
    #     pull torch on every invocation. ---
    if str(REPRO_REPO / "analysis") not in sys.path:
        sys.path.insert(0, str(REPRO_REPO / "analysis"))
    if str(SQUINT_PKG / "src") not in sys.path:
        sys.path.insert(0, str(SQUINT_PKG / "src"))

    import pickle
    import anndata as ad
    import torch
    import pytorch_lightning as pl
    from predict_model import collate_predict_outputs
    from vqniche.utils.parse_test_configs import collect_test_configs
    from vqniche.initializers.initialize import (
        initialize_dataset_blob,
        initialize_databatch,
        initialize_datamodule,
        set_model_class,
    )

    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("medium")

    run_dir = str(run_dir)

    # Load the user-saved training config FIRST so we can derive the
    # silver_dir from the variant's chosen dataset (chl59-8b_1p,
    # mmb0-1b_smb1-20b_1p_shared_genes, ...). Without this, predict() would
    # always look under the module-level DATASET_NAME and miss non-default
    # datasets.
    # NOTE: collect_test_configs takes a parameter named `wandb_run_dir` for
    # historical reasons; it now supports both the flat and legacy layouts.
    config = collect_test_configs(
        wandb_run_dir=run_dir,
        model_ckpt_fname=model_ckpt_fname,
    )
    print(f"Using checkpoint: {config['model']['model_ckpt_fname']}")

    if silver_dir:
        silver_dir = Path(silver_dir)
    else:
        cfg_root = config["dataset"].get("root_data_dir") or str(DATA_ROOT)
        cfg_name = config["dataset"].get("dataset_name") or DATASET_NAME
        silver_dir = Path(cfg_root) / "silver" / cfg_name

    # Sanity: list of .h5ad files (also used to label cells with their source).
    source_paths = _infer_adata_files_in_dir(silver_dir)
    print(f"Running inference on {len(source_paths)} file(s) under {silver_dir}:")
    for p in source_paths:
        print(f"  - {p.name}")

    # Determinism (mirrors predict_model.test).
    pl.seed_everything(config["experiment"]["seed"])

    # Build dataset + datamodule + model from the saved config.
    dataset_blob = initialize_dataset_blob(config)
    with open(Path(dataset_blob.processed_dir) / "label_categories.pkl", "rb") as f:
        label_categories = pickle.load(f)

    # Expand `adata_batch_idx` to EVERY section in the blob so predict()
    # also encodes whole-section holdouts (e.g. chl59 batches 2 + 3,
    # mmb20 batches 15 + 82). The saved training config carries
    # `adata_batch_idx = [train+val positions]` only — without this
    # override held-out sections never enter the dataloader, never get
    # encoded, and never appear in `predicted_adata.h5ad` (so the
    # downstream `plot_code_indices_spatial` etc. silently skip them).
    # Existing behaviour is preserved when no sections are held out:
    # `range(len(blob))` equals the original `adata_batch_idx` whenever
    # train+val already covers the whole blob.
    _orig_idx = list(config["dataset"].get("adata_batch_idx", []))
    config["dataset"]["adata_batch_idx"] = list(range(len(dataset_blob)))
    if config["dataset"]["adata_batch_idx"] != _orig_idx:
        print(
            f"Predict: expanding adata_batch_idx {_orig_idx} -> "
            f"all {len(dataset_blob)} sections "
            f"{config['dataset']['adata_batch_idx']} so held-out sections "
            f"are encoded and visible in downstream plots."
        )

    # Try to load the canonical gene panel saved at blob-build time so we can
    # propagate gene names through HVG selection into the predicted AnnData.
    gene_panel = None
    gene_panel_path = Path(dataset_blob.processed_dir) / "gene_panel.pkl"
    if gene_panel_path.exists():
        with open(gene_panel_path, "rb") as f:
            gene_panel = pickle.load(f)

    # Reconstruct the TRAIN-TIME `label_to_dense` mapping for batch one-
    # hot. The trained decoder-covariate / FiLM / adversary head all
    # have a fixed input/output dim N = number of train batches. With
    # the predict-time `adata_batch_idx` expansion above (so held-out
    # sections also get encoded), `obs_batch` now contains labels the
    # train pipeline never saw — re-densifying observed labels would
    # produce a one-hot dim > N and trigger CUDA index OOB inside the
    # adversary's softmax / decoder concat.
    # Mirror `build_batch_one_hot_from_obs`'s sorted-unique densification
    # but ONLY over train sections; held-out cells fall back to dense
    # ID 0 (= the first/reference train batch). Codes for held-out
    # cells are then "what would this cell look like if it were in the
    # reference batch" — usable for code-index visualisation and for
    # post-hoc Pearson, but the batch-correction effect is treated as
    # if from the reference batch.
    _train_batch_ids = (
        config.get("dataset", {})
              .get("train_transform_params", {})
              .get("train_batches", []) or []
    )
    _train_batch_ids_set = {int(b) for b in _train_batch_ids}
    _train_labels: set = set()
    for _pos in range(len(dataset_blob)):
        _d = dataset_blob[_pos]
        if int(_d.adata_batch_id) in _train_batch_ids_set:
            grp = getattr(_d, "obs_batch", None)
            if grp:
                _train_labels.add(str(grp[0]))
    _train_label_to_dense: dict = {
        lbl: i for i, lbl in enumerate(sorted(_train_labels))
    }
    if _train_label_to_dense:
        print(
            f"Predict: train-time label->dense map (used for batch "
            f"covariate at predict): {_train_label_to_dense}"
        )

    data_batch = initialize_databatch(
        config=config,
        dataset_blob=dataset_blob,
        batch_label_to_dense=_train_label_to_dense or None,
        unknown_batch_label_dense_id=0,
    )
    datamodule = initialize_datamodule(
        config=config,
        data=data_batch,
        obs_per_batch_id=getattr(dataset_blob, 'obs_per_batch_id', None),
    )

    # Recover the gene names the MODEL was trained on. The model's X_hat
    # output (and `inference_data['X']` cache) is in `gene_panel.index`
    # column order — every section is reindexed to that canonical panel
    # at blob-build time (`process_anndata_batch` line 367-371). The
    # source AnnDatas on disk may have a *different* native `var_names`
    # order (different platforms / different upstream tools); without a
    # reindex, `adata.X` from the source slice and `adata.layers['X_hat']`
    # from the model output land in different gene orders → cell-wise
    # pearson tanks because columns are paired wrong.
    #
    # Two cases:
    #   - SubsetHVG transform was applied: `hvg_indices` is stored per
    #     section; gene_names = canonical panel sliced to those indices.
    #   - No HVG transform: gene_names = the FULL canonical panel.
    #
    # `_build_clean_adata_from_inference` then reindexes every source
    # slice to `gene_names`, guaranteeing X_hat and X share gene order.
    gene_names = None
    if gene_panel is not None:
        try:
            sec0 = dataset_blob[0]
            if hasattr(sec0, "hvg_indices") and sec0.hvg_indices is not None:
                idxs = sec0.hvg_indices.cpu().numpy().tolist()
                # Sanity: HVG selection runs per-section; verify all sections
                # converged on the same gene set (otherwise the model would
                # have crashed on dim mismatch — but warn just in case).
                for i in range(1, len(dataset_blob)):
                    other = dataset_blob[i].hvg_indices.cpu().numpy().tolist()
                    if other != idxs:
                        print(
                            f"WARNING: section {i} HVG indices differ from "
                            f"section 0 — using section 0 names; downstream "
                            f"per-gene comparisons may be off."
                        )
                        break
                gene_names = [str(gene_panel.index[j]) for j in idxs]
                print(f"Resolved gene names for {len(gene_names)} HVGs from gene_panel.pkl.")
            else:
                # No HVG transform: pin to the full canonical gene panel so
                # the source slice is reindexed to match X_hat's column order.
                gene_names = [str(g) for g in gene_panel.index]
                print(
                    f"Resolved gene names for {len(gene_names)} genes from "
                    f"gene_panel.pkl (no HVG; using full canonical panel)."
                )
        except Exception as exc:  # noqa: BLE001
            print(f"NOTE: could not recover gene names ({exc}); falling back to integer-string names.")

    Model = set_model_class(config["model"]["model_name"])
    model = Model.load_from_checkpoint(config["model"]["model_ckpt_fname"])
    model.eval()

    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        deterministic=True,
        logger=False,
        strategy="ddp_find_unused_parameters_true",
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    print("Predicting model...")
    predict_data_cache = trainer.predict(model=model, datamodule=datamodule)
    predict_data_dict = collate_predict_outputs(
        data_cache=predict_data_cache,
        model=model,
        predict_dataloader=datamodule.predict_dataloader(),
    )

    # ----------------------------------------------------------------------
    # Two post-inference fixups in one pass over `predict_data_dict`:
    #
    # (1) Replace per-cell `adata_batch_ids` (DENSE, what the model used
    #     for one-hot indexing) with per-cell RAW ids (`uns['batch']`-
    #     derived ints, e.g. 15, 82). `source_adatas` below is keyed by
    #     raw blob ids; without this swap the source-AnnData lookup
    #     fails ("Cells reference adata_batch_ids [0, 1] but available
    #     source ids are [15, 82]"). The raw ids come straight from
    #     `data_batch.adata_batch_ids_raw` — set in `initialize_databatch`
    #     via `data_batch.adata_batch_id[data_batch.batch]` and cached
    #     by `_cache_inference_data`. We do NOT invert the train-time
    #     densification: that's ambiguous when held-out batches all
    #     collapse to dense=0 via the unknown-label fallback.
    #
    # (2) Undo PyG's auto-offset on `obs_row_index`. Any tensor whose key
    #     contains the substring 'index' triggers PyG's
    #     `Data.__inc__()` magic during `Batch.from_data_list` — it adds
    #     `num_nodes` to make `edge_index` etc. contiguous across the
    #     batched graph. `obs_row_index` is NOT an edge index but gets
    #     caught by the same heuristic, so per-section indices in
    #     [0, n_obs_section) get shifted by the cumulative node count of
    #     all preceding sections. We recover the original per-section
    #     index by subtracting `cum_offset_for_that_section[raw_id]`.
    if hasattr(data_batch, "obs_batch") and data_batch.obs_batch is not None:
        # Build the per-raw-id PyG offset (cumulative n_cells of all
        # sections preceding this raw id in concat order). Drives the
        # obs_row_index unshift below.
        _raw_to_pyg_offset: dict = {}
        _cum_offset = 0
        for _sec_idx in range(len(data_batch.obs_batch)):
            _group = data_batch.obs_batch[_sec_idx]
            if not _group:
                continue
            _raw_section_id = data_batch.adata_batch_id[_sec_idx]
            _raw_id = int(
                _raw_section_id.item() if hasattr(_raw_section_id, "item")
                else _raw_section_id
            )
            _raw_to_pyg_offset[_raw_id] = _cum_offset
            _cum_offset += len(_group)

        if "adata_batch_ids_raw" in predict_data_dict:
            import numpy as _np
            _raw_ids = predict_data_dict["adata_batch_ids_raw"]
            _raw_ids_np = (
                _raw_ids.cpu().numpy() if hasattr(_raw_ids, "cpu")
                else _np.asarray(_raw_ids)
            ).astype(int)
            # (1) point downstream code at the raw ids — overwrites the
            #     dense ids with raw ids; downstream consumers
            #     (`_build_clean_adata_from_inference`, the data_split
            #     tagger) all expect raw ids.
            predict_data_dict["adata_batch_ids"] = torch.tensor(
                _raw_ids_np, dtype=torch.long
            )
            print(
                f"Set per-cell adata_batch_ids from cached raw ids "
                f"(unique: {sorted(set(_raw_ids_np.tolist()))})."
            )

            # (2) Undo PyG's auto-offset on obs_row_index using the
            #     per-raw-id cumulative offset.
            if "obs_row_index" in predict_data_dict and any(
                _v != 0 for _v in _raw_to_pyg_offset.values()
            ):
                _row = predict_data_dict["obs_row_index"]
                _row_np = (_row.cpu().numpy() if hasattr(_row, "cpu")
                           else _np.asarray(_row)).astype(int)
                _missing_raw = sorted(set(_raw_ids_np.tolist()) - _raw_to_pyg_offset.keys())
                if _missing_raw:
                    raise RuntimeError(
                        f"Cells reference raw adata_batch_ids "
                        f"{_missing_raw} that have no offset in "
                        f"data_batch (known raw ids: "
                        f"{sorted(_raw_to_pyg_offset)})."
                    )
                _offsets = _np.array(
                    [_raw_to_pyg_offset[int(r)] for r in _raw_ids_np],
                    dtype=int,
                )
                predict_data_dict["obs_row_index"] = torch.tensor(
                    _row_np - _offsets, dtype=torch.long
                )
                print(
                    f"Undid PyG auto-offset on obs_row_index using "
                    f"per-raw-id offsets {_raw_to_pyg_offset}."
                )
        else:
            print(
                "WARN: predict_data_dict has no `adata_batch_ids_raw`. "
                "The dataset blob predates the per-cell-raw-id field; "
                "dense ids will be passed through unchanged. If you "
                "have whole-section holdouts, source-AnnData lookup may "
                "fail — rebuild the blob to populate the field."
            )

    # Build the clean AnnData.
    #
    # We start from the source AnnDatas themselves (concatenated in
    # inference cell order) and bolt model outputs on top. This is far
    # simpler than synthesising a fresh AnnData from torch tensors and
    # then trying to graft labels back: every original `obs` / `var`
    # column flows through automatically, including `cell_type`,
    # `Sub_molecular_*`, `ccf_region_name`, `donor_id`, etc. Cells from
    # a source that doesn't carry a given column get NaN via
    # `ad.concat(..., join="outer")`.
    #
    # Build `id_to_path` (and `source_adatas`) by matching each silver
    # file to a blob section via cell_id intersection — NOT by
    # re-deriving the id from `uns['batch']` on the silver file. The
    # latter is fragile: if the blob was built when `uns['batch']`
    # wasn't parseable (e.g. the user added it later, or the format
    # differs), the freshly-derived id won't match the id stamped on
    # the cells in `dataset_blob.pt`. cell_id matching is robust because
    # cell ids are stable across silver-file edits and survive the blob
    # build verbatim.
    #
    # `test_batch_ids` comes from the saved training config — sections
    # in this list were held out from training. The builder stamps
    # `adata.obs["data_split"] = "test"` for cells whose source section
    # is in this list (everything else is "train", with val cells folded
    # in per the user spec). compute_inference_metrics.py reads it for
    # stratified NMI / ARI / Pearson.
    print(f"Loading {len(source_paths)} source AnnDatas + matching them "
          f"to blob sections via cell_id ...")

    # Step 1: read each blob section's first cell_id and adata_batch_id.
    # The blob has the canonical id assignments (whatever they are).
    blob_id_to_first_cell: dict = {}
    for _pos in range(len(dataset_blob)):
        _d = dataset_blob[_pos]
        if hasattr(_d, "cell_id") and len(_d.cell_id) > 0:
            blob_id_to_first_cell[int(_d.adata_batch_id)] = str(_d.cell_id[0])
        else:
            print(f"  WARN: blob position {_pos} has no cell_id list — "
                  f"section can't be matched to a silver file by cell_id.")

    # Steps 2 + 3 (fused): each silver file is read in FULL only once.
    # The previous version opened every file twice — first in
    # `backed='r'` mode to lift `obs['cell_id'].iloc[0]`, then again in
    # full-read mode for row-slicing in `_build_clean_adata_from_inference`.
    # On mmb20 (21 files, multi-GB each on NFS) the second read pass
    # alone added 30-60 s. Read once into a per-path cache, do the
    # cell_id matching against the in-memory copy, then point
    # `source_adatas` at the same objects.
    silver_first_cells: dict = {}
    silver_full_adata: dict = {}
    for p in source_paths:
        try:
            a = ad.read_h5ad(p)  # FULL read; needed by _build_clean_adata_from_inference anyway
        except Exception as exc:
            print(f"  WARN: could not read {p.name}: {exc}")
            continue
        silver_full_adata[p] = a
        silver_first_cells[p] = str(a.obs["cell_id"].iloc[0])

    # Step 3: invert. id_to_path[bid] = the silver file whose first
    # cell_id matches the blob's id-bid section.
    id_to_path: dict = {}
    unmatched_blob_ids: list = []
    used_paths: set = set()
    for bid, blob_first in blob_id_to_first_cell.items():
        match_p = None
        for p, silver_first in silver_first_cells.items():
            if p in used_paths:
                continue
            if silver_first == blob_first:
                match_p = p
                break
        if match_p is None:
            unmatched_blob_ids.append(bid)
        else:
            id_to_path[bid] = match_p
            used_paths.add(match_p)

    if unmatched_blob_ids:
        raise RuntimeError(
            f"Could not match blob adata_batch_ids {unmatched_blob_ids} "
            f"to any silver file via cell_id intersection. Available "
            f"silver files: {[p.name for p in source_paths]}. "
            f"Available blob ids: {sorted(blob_id_to_first_cell)}. "
            f"Likely cause: silver dir contents differ from what was "
            f"used to build the dataset blob."
        )

    # Step 4: point source_adatas at the cached, already-loaded AnnDatas.
    source_adatas: dict = {}
    for bid, p in id_to_path.items():
        source_adatas[bid] = silver_full_adata[p]
        print(f"  adata_batch_id={bid:<3d}  {p.name}  "
              f"(n_obs={source_adatas[bid].n_obs}, "
              f"n_vars={source_adatas[bid].n_vars})")

    test_batch_ids = (
        config.get("dataset", {})
              .get("train_transform_params", {})
              .get("test_batches", [])
        or []
    )
    # Per-batch held-out regions (downstream gene-reconstruction task).
    # Read from the saved training config so predict()'s `data_split`
    # tagging matches what `SpatialBatchSplit` masked out at train time.
    test_regions = (
        config.get("dataset", {})
              .get("train_transform_params", {})
              .get("test_regions", None)
    )
    adata = _build_clean_adata_from_inference(
        inference_data=predict_data_dict,
        source_adatas=source_adatas,
        id_to_path=id_to_path,
        gene_names=gene_names,
        test_batch_ids=list(test_batch_ids),
        test_regions=test_regions,
    )
    adata.uns["squint"]["run_dir"] = run_dir
    adata.uns["squint"]["ckpt"] = str(config["model"]["model_ckpt_fname"])

    # If the run was produced by the variant-aware layout, surface the variant
    # name + ablation description on the predicted AnnData so plotting scripts
    # can label themselves without re-parsing the path.
    summary_path = Path(run_dir) / "ablation_summary.yaml"
    if summary_path.exists():
        try:
            with open(summary_path, "r") as f:
                _summary = yaml.safe_load(f) or {}
            adata.uns["squint"]["variant"] = str(_summary.get("variant", ""))
            adata.uns["squint"]["variant_description"] = str(_summary.get("description", ""))
        except Exception as exc:  # noqa: BLE001
            print(f"NOTE: could not read {summary_path} ({exc}); skipping variant tagging.")

    # Resolve output path.
    # Default: write inference outputs (predicted_adata.h5ad + downstream
    # plots / metrics) directly into the run_dir alongside checkpoints
    # and the user-saved config — so everything for one training run
    # lives in one folder
    # (`<ARTIFACTS_DIR>/<dataset>/<variant>/<timestamp>/`). `--output-dir`
    # still overrides if you want a separate location.
    if output_dir is None:
        output_dir = Path(run_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "predicted_adata.h5ad"

    # AnnData can't write nested-dict .uns -> flatten the squint metadata.
    # We keep the dict but cast values to writeable types.
    adata.uns["squint"] = {k: (str(v) if not isinstance(v, (int, float, bool)) else v)
                          for k, v in adata.uns["squint"].items()}
    # gzip compression shrinks the predicted_adata.h5ad 3-10x at
    # negligible CPU cost — matters most when the file lives on shared
    # NFS (network transfer dominates). compression_opts=4 is a sane
    # default (level 4: balanced compression / write speed). h5py
    # accepts integers 0-9; raise for higher ratio at write-time cost.
    adata.write_h5ad(out_path, compression="gzip", compression_opts=4)

    print()
    print("=" * 78)
    print(f"Wrote inference AnnData to: {out_path}")
    print(f"  n_cells   = {adata.n_obs}")
    print(f"  n_genes   = {adata.n_vars}")
    print(f"  obsm keys = {list(adata.obsm.keys())}")
    print(f"  obs cols  = {list(adata.obs.columns)}")
    print(f"  layers    = {list(adata.layers.keys())}")
    print("=" * 78)
    return adata


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run_inference_and_analysis(
        run_dir: str,
        silver_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        model_ckpt_fname: Optional[str] = None,
        label_keys: str = "cell_type,niche,Sub_molecular_tissue_region,ccf_region_name",
        batch_rename: Optional[str] = None,
        cell_label_keys: Optional[str] = None,
        niche_label_keys: Optional[str] = None,
        skip_predict: bool = False,
        skip_code_index_plots: bool = False,
        skip_svg_plots: bool = False,
        skip_umap: bool = False,
        skip_metrics: bool = False,
    ) -> Path:
    """
    Steps 2-6 of the end-to-end pipeline (everything AFTER `train()`):

      2. predict(run_dir)          -> writes predicted_adata.h5ad into the
                                       run_dir itself
      3. plot_code_indices_spatial -> per-batch spatial plots of code indices
      4. plot_svg_reconstruction   -> SVG vs. reconstruction sanity plots
      5. plot_latent_umap          -> UMAP of each latent slot (rapids env)
      6. compute_inference_metrics -> NMI/ARI vs. ground-truth labels +
                                       batch integration (iLISI, ASW, MMD)
                                       + avg cosine similarity

    Plot + metrics scripts run via subprocess so each gets a clean
    process / cuda state and one's output doesn't pollute the others.
    Output files (subdirs of the predict folder) are written next to
    `predicted_adata.h5ad` in the run_dir.

    Returns the resolved `predict_dir` (= output_dir if set, else run_dir).
    """
    import os
    import subprocess

    # ------------------------------------------------------------------
    # 2. Predict
    # ------------------------------------------------------------------
    # Resolve predict_dir up-front so the skip-predict path can still
    # locate an existing predicted_adata.h5ad for downstream steps.
    if output_dir is None:
        predict_dir = Path(run_dir)
    else:
        predict_dir = Path(output_dir)
    predicted_adata_path = predict_dir / "predicted_adata.h5ad"

    if skip_predict:
        print()
        print("=" * 78)
        print(f"[2/6] SKIPPED predict — using existing {predicted_adata_path}")
        print("=" * 78)
        if not predicted_adata_path.exists():
            raise FileNotFoundError(
                f"--skip-predict requested but {predicted_adata_path} is missing."
            )
    else:
        print()
        print("=" * 78)
        print(f"[2/6] Predict on run_dir={run_dir}")
        print("=" * 78)
        predict(
            run_dir=run_dir,
            silver_dir=silver_dir,
            model_ckpt_fname=model_ckpt_fname,
            output_dir=output_dir,
        )
        if not predicted_adata_path.exists():
            raise FileNotFoundError(
                f"Expected predicted AnnData at {predicted_adata_path} but it's missing."
            )
        print(f"[2/6] Done. predicted_adata={predicted_adata_path}")

    # ------------------------------------------------------------------
    # 3. plot_code_indices_spatial
    # ------------------------------------------------------------------
    examples_dir = Path(__file__).parent
    common_args = ["--predicted-adata", str(predicted_adata_path)]

    if skip_code_index_plots:
        print()
        print("=" * 78)
        print("[3/6] SKIPPED plot_code_indices_spatial")
        print("=" * 78)
    else:
        print()
        print("=" * 78)
        print("[3/6] plot_code_indices_spatial")
        print("=" * 78)
        subprocess.run(
            [sys.executable, str(examples_dir / "plot_code_indices_spatial.py")] + common_args,
            check=True,
        )

    # ------------------------------------------------------------------
    # 4. plot_svg_reconstruction
    # ------------------------------------------------------------------
    if skip_svg_plots:
        print()
        print("=" * 78)
        print("[4/6] SKIPPED plot_svg_reconstruction")
        print("=" * 78)
    else:
        print()
        print("=" * 78)
        print("[4/6] plot_svg_reconstruction")
        print("=" * 78)
        subprocess.run(
            [sys.executable, str(examples_dir / "plot_svg_reconstruction.py")] + common_args,
            check=True,
        )

    # ------------------------------------------------------------------
    # 5. plot_latent_umap
    # ------------------------------------------------------------------
    # The UMAP step is the only one that benefits from rapids-singlecell
    # (GPU UMAP via cuML). Rapids is fragile to install alongside torch +
    # pyg in a single venv, so it lives in a separate conda env. Invoke
    # the UMAP script via a wrapper that does `module load cellgen/conda`
    # + `conda activate $RAPIDS_ENV` before running. Override the env
    # path at submit time:
    #   RAPIDS_ENV=/path/to/conda/env bash examples/submit_dataset_sweep.sh ...
    # If $RAPIDS_ENV is unset OR the env is missing, fall back to running
    # the script in the current (squint uv) venv — plot_latent_umap.py's
    # `_detect_gpu_umap_backend()` then auto-falls-back to scanpy CPU.
    if skip_umap:
        print()
        print("=" * 78)
        print("[5/6] SKIPPED plot_latent_umap")
        print("=" * 78)
    else:
        # Free torch's cached GPU memory FIRST. The rapids subprocess uses
        # the same GPU (LSF gives the job one exclusive device), and torch's
        # caching allocator can hold gigabytes of "freed" blocks after
        # predict() completes. Without this release, rapids' first cupy
        # allocation fails with `MemoryError: failed to allocate <small>
        # bytes` even though no work is actually running on the GPU. Wrap in
        # try/except so the path still works in CPU-only configurations.
        try:
            import gc as _gc
            import torch as _torch
            _gc.collect()
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
                _torch.cuda.synchronize()
                print(f"[5/6] released parent torch CUDA cache before rapids subprocess")
        except Exception as _e:  # noqa: BLE001
            print(f"[5/6] could not release torch CUDA cache (ok): {_e}")
        print()
        print("=" * 78)
        print("[5/6] plot_latent_umap")
        print("=" * 78)
        umap_args = list(common_args) + ["--label-keys", label_keys]
        if batch_rename:
            umap_args += ["--batch-rename", batch_rename]

        rapids_env = os.environ.get(
            "RAPIDS_ENV", "/nfs/team361/sb75/ENVS/rapids-singlecell"
        )
        rapids_wrapper = examples_dir / "_run_umap_rapids.sh"
        if Path(rapids_env).exists() and rapids_wrapper.exists():
            print(f"[5/6] using rapids conda env: {rapids_env}")
            subprocess.run(
                ["bash", str(rapids_wrapper), *umap_args],
                check=True,
                env={**os.environ, "RAPIDS_ENV": rapids_env},
            )
        else:
            print(
                f"[5/6] rapids env not found at {rapids_env} (or wrapper "
                f"missing); running plot_latent_umap.py in the current venv "
                f"(scanpy CPU fallback)."
            )
            subprocess.run(
                [sys.executable, str(examples_dir / "plot_latent_umap.py")] + umap_args,
                check=True,
            )

    # ------------------------------------------------------------------
    # 6. compute_inference_metrics
    # ------------------------------------------------------------------
    # NMI/ARI between cell + niche codes and ground-truth labels, plus
    # batch integration metrics (iLISI / ASW / MMD) and avg cosine
    # similarity on every embedding obsm slot.
    #
    # Defaults to the user's preferred label sets:
    #   cell  codes   vs. {cell_type, cell_types}
    #   niche codes   vs. {niche, Sub_molecular_tissue_region, ccf_region_name}
    # — labels not present on the AnnData are skipped silently, so the
    # same script works for both the mouse-brain (cell_type +
    # Sub_molecular_tissue_region + ccf_region_name) and chl59 lung
    # (cell_type + niche) datasets.
    if skip_metrics:
        print()
        print("=" * 78)
        print("[6/6] SKIPPED compute_inference_metrics")
        print("=" * 78)
    else:
        print()
        print("=" * 78)
        print("[6/6] compute_inference_metrics")
        print("=" * 78)
        metrics_args = list(common_args)
        if cell_label_keys is not None:
            metrics_args += ["--cell-label-keys", cell_label_keys]
        if niche_label_keys is not None:
            metrics_args += ["--niche-label-keys", niche_label_keys]
        subprocess.run(
            [sys.executable, str(examples_dir / "compute_inference_metrics.py")] + metrics_args,
            check=True,
        )

    print()
    print("=" * 78)
    print("Pipeline complete.")
    print(f"  Run dir:           {run_dir}")
    print(f"  Predicted AnnData: {predicted_adata_path}")
    print(f"  Plots:             {predict_dir}/{{code_index_plots,svg_plots,umap_plots}}/")
    print(f"  Metrics:           {predict_dir}/metrics/")
    print("=" * 78)
    return predict_dir


def run_all_pipeline(
        variant: str,
        silver_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        model_ckpt_fname: Optional[str] = None,
        label_keys: str = "cell_type,niche,Sub_molecular_tissue_region,ccf_region_name",
        batch_rename: Optional[str] = None,
        cell_label_keys: Optional[str] = None,
        niche_label_keys: Optional[str] = None,
    ) -> None:
    """
    End-to-end pipeline:

      1. train(variant) -> writes <ARTIFACTS_DIR>/<dataset>/<variant>/<TS>/
      2-6. inference + plots + metrics (see run_inference_and_analysis)

    To re-run only steps 2-6 against an already-trained checkpoint, use
    `examples/run_inference.py --variant <V> --timestamp <TS>` (it calls
    `run_inference_and_analysis` directly).
    """
    # ------------------------------------------------------------------
    # 1. Train
    # ------------------------------------------------------------------
    print("=" * 78)
    print(f"[1/6] Train  variant={variant!r}")
    print("=" * 78)
    run_dir = train(variant)
    if run_dir is None:
        # Older train() signatures didn't return; fall back to looking up
        # the most recent run dir for this variant. Resolve the variant's
        # dataset tag from its build so the lookup lands in the same dir
        # train() would have created (chl59 / mmb20 don't live under the
        # mmb-smb default).
        variant_slug = variant.replace("/", "_")
        _variant_cfg = VARIANTS[variant]["build"]()
        _variant_tag = _variant_cfg["dataset"].get(
            "dataset_tag",
            _variant_cfg["dataset"].get("dataset_name", DATASET_NAME),
        )
        candidates = sorted((ARTIFACTS_DIR / _variant_tag / variant_slug).glob("*/"))
        if not candidates:
            raise RuntimeError(f"Could not auto-locate run_dir for variant {variant}")
        run_dir = str(candidates[-1])
    print(f"[1/6] Done. run_dir={run_dir}")

    run_inference_and_analysis(
        run_dir=run_dir,
        silver_dir=silver_dir,
        output_dir=output_dir,
        model_ckpt_fname=model_ckpt_fname,
        label_keys=label_keys,
        batch_rename=batch_rename,
        cell_label_keys=cell_label_keys,
        niche_label_keys=niche_label_keys,
    )


def resolve_run_dir(variant: str, timestamp: str) -> Path:
    """
    Look up the run_dir for a given (variant, timestamp).

    Maps to `<ARTIFACTS_DIR>/<dataset_tag>/<variant_slug>/<timestamp>`,
    where `dataset_tag` is read from the variant's build config (the
    short identifier — `mmb0-1b_smb1-1b_1p`, `chl59-8b_1p`,
    `mmb0-1b_smb1-20b_1p`). Falls back to `dataset_name` if a variant
    doesn't set a tag.

    Raises if the variant doesn't exist or the resolved directory is
    missing on disk (probably the timestamp is wrong, or the run was
    deleted).
    """
    if variant not in VARIANTS:
        raise KeyError(
            f"Unknown variant {variant!r}. Use --list-variants on "
            f"run_squint.py to see registered variants."
        )
    variant_slug = variant.replace("/", "_").replace(" ", "_")
    cfg = VARIANTS[variant]["build"]()
    dataset_tag = cfg["dataset"].get(
        "dataset_tag",
        cfg["dataset"].get("dataset_name", DATASET_NAME),
    )
    run_dir = ARTIFACTS_DIR / dataset_tag / variant_slug / timestamp
    if not run_dir.exists():
        # Surface neighbouring timestamps so the user sees what's
        # available without having to ls the directory themselves.
        parent = ARTIFACTS_DIR / dataset_tag / variant_slug
        siblings = (
            sorted(p.name for p in parent.glob("*") if p.is_dir())
            if parent.exists() else []
        )
        raise FileNotFoundError(
            f"Run dir not found: {run_dir}. "
            f"Available timestamps under {parent}: "
            f"{siblings if siblings else '<none>'}"
        )
    return run_dir


def run_dataset_pipeline(
        dataset: str,
        variants: Optional[List[str]] = None,
        continue_on_error: bool = True,
        **pipeline_kwargs,
    ) -> dict:
    """
    Run `run_all_pipeline` for every variant in DATASET_VARIANTS[dataset].

    Use this for overnight per-HPC-node sweeps (one node per dataset):
      python examples/run_squint.py --all-dataset mmb0-1b_smb1-1b_1p
      python examples/run_squint.py --all-dataset chl59-8b_1p
      python examples/run_squint.py --all-dataset mmb0-1b_smb1-20b_1p

    Parameters
    ----------
    dataset : short dataset tag (key of DATASET_VARIANTS).
    variants : optional explicit subset of variant names to run; defaults
        to the full DATASET_VARIANTS[dataset] list.
    continue_on_error : if True (default), a failure in one variant prints
        the traceback and proceeds to the next. Set False to abort on the
        first failure.
    pipeline_kwargs : forwarded to run_all_pipeline (silver_dir,
        output_dir, model_ckpt_fname, label_keys, batch_rename,
        cell_label_keys, niche_label_keys).

    Returns
    -------
    A dict {variant_name: "ok" | "<error message>"} summarising the run,
    also printed to stdout at the end.
    """
    import traceback as _tb

    if dataset not in DATASET_VARIANTS:
        raise ValueError(
            f"Unknown dataset {dataset!r}. Choose from "
            f"{sorted(DATASET_VARIANTS.keys())}."
        )
    todo = list(variants) if variants else list(DATASET_VARIANTS[dataset])

    print("=" * 78)
    print(f"Dataset sweep: {dataset}  ({len(todo)} variants)")
    for v in todo:
        print(f"  - {v}")
    print("=" * 78)

    results: dict = {}
    for i, variant in enumerate(todo, 1):
        print()
        print("#" * 78)
        print(f"# [{i}/{len(todo)}] {variant}")
        print("#" * 78)
        try:
            run_all_pipeline(variant=variant, **pipeline_kwargs)
            results[variant] = "ok"
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
            results[variant] = err
            print(f"\n!!! Variant {variant!r} FAILED: {err}")
            _tb.print_exc()
            if not continue_on_error:
                print("Aborting sweep (continue_on_error=False).")
                break

    print()
    print("=" * 78)
    print(f"Dataset sweep complete: {dataset}")
    for v in todo:
        status = results.get(v, "skipped")
        print(f"  {status:30s}  {v}")
    print("=" * 78)
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--patch-uns", action="store_true",
                   help="Patch .uns/.obs of the two .h5ad files (one-time, only if missing).")
    p.add_argument("--harmonize-var", action="store_true",
                   help="Subset and align .var across the two .h5ad files (one-time, "
                        "run if --build-blob fails with 'All batches must have the same gene panel').")
    p.add_argument("--build-blob", action="store_true",
                   help="Build the in-memory PyG DatasetBlob (one-time). "
                        "Use --build-blob-dataset to pick which dataset.")
    p.add_argument("--build-blob-dataset", type=str,
                   default="mmb0-1b_smb1-1b_1p",
                   choices=[
                       "mmb0-1b_smb1-1b_1p",
                       "chl59-8b_1p",
                       "mmb0-1b_smb1-20b_1p",
                   ],
                   help="Which dataset to build (default: "
                        "mmb0-1b_smb1-1b_1p — MERFISH + STARmap mouse "
                        "brain, 1+1 sections). "
                        "'chl59-8b_1p' = /nfs/team361/sb75/DATASETS/"
                        "silver/chl59-8b_1p (CosMx Lung, 8 samples). "
                        "'mmb0-1b_smb1-20b_1p' = /lustre/.../silver/"
                        "mmb0-1b_smb1-20b_1p_shared_genes (MERFISH + 20 "
                        "STARmap, harmonised by "
                        "examples/harmonize_mmb_smb20_panels.py first).")
    p.add_argument("--train", action="store_true",
                   help="Train SQUINT on the WHOLE dataset blob (logs to wandb project 'squint').")
    p.add_argument(
        "--variant", type=str,
        default="dualvq+wide+rvq-both-3level+decoder-cov+adv+enc-deeper+mmb0-1b_smb1-1b_1p",
        help="Which training/ablation config to use. The default is the "
             "v5 anchor — the project's current best variant on the "
             "mmb0-1b_smb1-1b_1p dataset (combines `+wide` for niche "
             "NMI, `+enc-deeper` for batch integration, and "
             "`+rvq-both-3level` for Pearson reconstruction). "
             "Use --list-variants to see all registered variants.",
    )
    p.add_argument("--list-variants", action="store_true",
                   help="Print all registered ablation variants and exit.")
    p.add_argument("--list-dataset-sweeps", action="store_true",
                   help="Print the per-dataset sweep matrix (DATASET_VARIANTS) "
                        "and exit. The 10 variant names per dataset are what "
                        "--all-dataset iterates over.")
    p.add_argument("--variants-for-dataset", type=str, default=None,
                   help="Print just the variant names for one DATASET_VARIANTS "
                        "key (one per line, no decoration) and exit. Intended "
                        "for shell-script consumption (e.g. LSF submitters that "
                        "bsub one job per variant). Use --list-dataset-sweeps "
                        "for the human-readable form.")
    p.add_argument("--predict", action="store_true",
                   help="Run inference on a folder of .h5ad files using a trained checkpoint.")
    p.add_argument("--all", action="store_true",
                   help="End-to-end: train -> predict -> 3 plot scripts "
                        "(plot_code_indices_spatial, plot_svg_reconstruction, "
                        "plot_latent_umap) -> compute_inference_metrics "
                        "(NMI/ARI vs ground-truth labels + iLISI/ASW/MMD batch "
                        "integration + avg cosine similarity). Convenience "
                        "wrapper for `--train --variant <V>` followed by "
                        "`--predict --run-dir <auto>`.")
    p.add_argument("--all-dataset", type=str, default=None,
                   choices=sorted(DATASET_VARIANTS.keys()),
                   help="Run --all (full pipeline) for EVERY variant in the "
                        "given dataset's 10-axis sweep, sequentially. "
                        "Submit one HPC job per dataset (one node each) for "
                        "an overnight sweep. Use --list-dataset-sweeps to "
                        "see the variant names. Failures within the sweep "
                        "are logged but the loop continues (set "
                        "--abort-on-error to stop instead).")
    p.add_argument("--abort-on-error", action="store_true",
                   help="With --all-dataset, abort the sweep on the first "
                        "variant that errors out instead of continuing to the "
                        "next one.")
    p.add_argument("--batch-rename", type=str, default=None,
                   help="Optional comma-separated rename for adata_batch_id "
                        "categories in UMAP plots (e.g. 'MERFISH,STARmap PLUS').")
    p.add_argument(
        "--label-keys", type=str,
        default="cell_type,niche,Sub_molecular_tissue_region,ccf_region_name",
        help="Comma-separated obs columns to color UMAPs and code-index "
             "plots by. Default covers both the mouse-brain (cell_type, "
             "Sub_molecular_tissue_region, ccf_region_name) and CosMx Lung "
             "(cell_type, niche) niche label conventions; "
             "missing columns are silently skipped per UMAP. One UMAP "
             "panel is emitted per label per latent slot.",
    )
    p.add_argument("--cell-label-keys", type=str, default=None,
                   help="Comma-separated obs columns to compare CELL codes against "
                        "in the post-inference NMI/ARI metrics (default: lets the "
                        "metrics script use its own defaults of cell_type,cell_types). "
                        "Forwarded to compute_inference_metrics.py via --all.")
    p.add_argument("--niche-label-keys", type=str, default=None,
                   help="Comma-separated obs columns to compare NICHE codes against "
                        "in the post-inference NMI/ARI metrics (default: lets the "
                        "metrics script use its own defaults of niche,"
                        "Sub_molecular_tissue_region,ccf_region_name). Forwarded to "
                        "compute_inference_metrics.py via --all.")
    p.add_argument("--run-dir", type=str, default=None,
                   help="Path to the run directory of the trained model "
                        "(<ARTIFACTS_DIR>/<DATASET_NAME>/<TIMESTAMP>). Required with --predict.")
    p.add_argument("--wandb-run-dir", type=str, default=None,
                   help="DEPRECATED alias for --run-dir (kept for backward compatibility).")
    p.add_argument("--silver-dir", type=str, default=None,
                   help="Folder of .h5ad files for inference (default: same silver "
                        "folder used for training).")
    p.add_argument("--model-ckpt-fname", type=str, default=None,
                   help="Optional path to a specific .ckpt; otherwise the best "
                        "checkpoint is auto-selected.")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Where to write predicted_adata.h5ad and downstream "
                        "plot/metrics output. Defaults to the run_dir itself "
                        "so all artifacts for one training run live in one "
                        "folder.")
    # Opt-in switch for the heavy per-epoch Pearson metrics. Off by
    # default since `compute_inference_metrics.py` produces the
    # benchmark CSVs at the end of training anyway. See
    # `_resolve_metrics_list()` for what gets enabled.
    p.add_argument(
        "--with-pearson", action="store_true",
        help="Compute the full Pearson metrics list every val epoch "
             "during training (turns the in-flight wandb Pearson curves "
             "back on). OFF by default — expect ~10-20x slower epochs "
             "when enabled, because each metric reconstructs the full "
             "val AnnData and runs a cells x genes correlation. The "
             "final benchmark CSVs are produced by "
             "`compute_inference_metrics.py` at the end of the "
             "pipeline regardless of this flag. Equivalent to setting "
             "the env var SQUINT_WITH_PEARSON=1.",
    )
    args = p.parse_args()

    # Propagate the CLI flag to the env-var that `_resolve_metrics_list()`
    # checks at config-build time. Setting the env var here (before any
    # `train()` call below materialises a cfg) ensures the new
    # `train_metrics_list` / `test_metrics_list` reflect the chosen
    # behaviour for this invocation. Setting it directly via env var on
    # the command line (`SQUINT_WITH_PEARSON=1 python ...`) also works
    # — this just bridges from CLI flag to env var.
    if args.with_pearson:
        os.environ["SQUINT_WITH_PEARSON"] = "1"

    if args.list_variants:
        print("Registered ablation variants:")
        for name, spec in VARIANTS.items():
            print(f"  {name}")
            print(f"    description: {spec['description']}")
            print(f"    patches:     {spec['patches']}")
        return

    if args.list_dataset_sweeps:
        print("Per-dataset sweep matrix (DATASET_VARIANTS):")
        print("Run all variants of one dataset sequentially via:")
        print("  python examples/run_squint.py --all-dataset <DATASET>")
        print()
        for ds in sorted(DATASET_VARIANTS.keys()):
            names = DATASET_VARIANTS[ds]
            print(f"  {ds}  ({len(names)} variants)")
            for n in names:
                print(f"    - {n}")
            print()
        return

    if args.variants_for_dataset is not None:
        if args.variants_for_dataset not in DATASET_VARIANTS:
            raise SystemExit(
                f"Unknown DATASET key {args.variants_for_dataset!r}. "
                f"Available: {sorted(DATASET_VARIANTS.keys())}."
            )
        for n in DATASET_VARIANTS[args.variants_for_dataset]:
            print(n)
        return

    if not (args.patch_uns or args.harmonize_var or args.build_blob
            or args.train or args.predict or args.all
            or args.all_dataset):
        p.print_help()
        return

    if args.patch_uns:
        patch_anndata_uns()
    if args.harmonize_var:
        harmonize_anndata_var()
    if args.build_blob:
        build_blob(dataset=args.build_blob_dataset)
    if args.train:
        train(args.variant)
    if args.predict:
        run_dir = args.run_dir or args.wandb_run_dir
        if run_dir is None:
            raise SystemExit("--predict requires --run-dir")
        if args.wandb_run_dir is not None and args.run_dir is None:
            print("Note: --wandb-run-dir is deprecated; use --run-dir instead.")
        predict(
            run_dir=run_dir,
            silver_dir=args.silver_dir,
            model_ckpt_fname=args.model_ckpt_fname,
            output_dir=args.output_dir,
        )

    # --all: end-to-end train -> predict -> 3 plot scripts -> metrics.
    if args.all:
        run_all_pipeline(
            variant=args.variant,
            silver_dir=args.silver_dir,
            output_dir=args.output_dir,
            model_ckpt_fname=args.model_ckpt_fname,
            label_keys=args.label_keys,
            batch_rename=args.batch_rename,
            cell_label_keys=args.cell_label_keys,
            niche_label_keys=args.niche_label_keys,
        )

    # --all-dataset: loop --all over every variant in the dataset's sweep.
    if args.all_dataset:
        run_dataset_pipeline(
            dataset=args.all_dataset,
            continue_on_error=not args.abort_on_error,
            silver_dir=args.silver_dir,
            output_dir=args.output_dir,
            model_ckpt_fname=args.model_ckpt_fname,
            label_keys=args.label_keys,
            batch_rename=args.batch_rename,
            cell_label_keys=args.cell_label_keys,
            niche_label_keys=args.niche_label_keys,
        )


if __name__ == "__main__":
    main()
