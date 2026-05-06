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
  python examples/run_squint_mmb_smb.py --patch-uns

  # 1) one-time only — preprocess the two .h5ad files into a cached PyG
  #    DatasetBlob.  Output:
  #        <silver-root>/gold/in-memory-PyG-dataset-blob/<DATASET_NAME>/
  python examples/run_squint_mmb_smb.py --build-blob

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
  python examples/run_squint_mmb_smb.py --train --variant recon-cell
  python examples/run_squint_mmb_smb.py --train --variant recon-both+adj

  # 3) inference on a folder of .h5ad files. Returns a single AnnData with
  #    the SQUINT outputs in .obsm / .obs / .layers, saved as .h5ad under
  #    <ARTIFACTS_DIR>/inference/<variant>/<timestamp>/predicted_adata.h5ad.
  #    --run-dir is the directory printed at the end of step 2.
  python examples/run_squint_mmb_smb.py --predict \
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
import sys
from pathlib import Path
from typing import List, Optional, Sequence

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

# Where inference results go.
INFERENCE_DIR = ARTIFACTS_DIR / "inference"

# Project name in wandb.
WANDB_PROJECT = "squint"


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
            "log_model": True,
            "offline": False,
            "enabled": True,
        },
        "dataset": {
            "dataset_name": DATASET_NAME,
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
            },
        },
        "datamodule": {
            "loader_name": "NeighborLoader",
            "loader_params": {"batch_size": 256},
            "sampler_name": "NeighborSampler",
            "sampler_params": {"num_neighbors": [8]},  # 1 hop = num_layers
            "inference_params": {"sample_neighbors_for_inference": False},
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
            # Primary monitor: pearson_gene_wise_log1p (literature standard).
            "train_metrics_list": [
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
                # legacy raw-count metrics
                "pearson_gene_wise",
            ],
            "test_metrics_list": [
                "codebook_utilization",
                "pearson_gene_wise_log1p",
                "pearson_gene_wise_log1p_median",
                "pearson_gene_wise_hvg50_log1p",
                "pearson_gene_wise_hvg50_log1p_median",
                "pearson_cell_wise_log1p",
                "pearson_cell_wise_log1p_median",
                "pearson_gene_wise_1hop_nbr_log1p",
                "pearson_gene_wise_1hop_nbr_log1p_median",
                "pearson_gene_wise_hvg50_1hop_nbr_log1p",
                "pearson_gene_wise_hvg50_1hop_nbr_log1p_median",
                "pearson_cell_wise_1hop_nbr_log1p",
                "pearson_cell_wise_1hop_nbr_log1p_median",
                "pearson_gene_wise",
                "pearson_cell_wise",
            ],
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
            "monitor": "train_pearson_gene_wise_log1p",
            "checkpoint_params": {"mode": "max", "save_top_k": 1, "save_last": True},
            "max_epochs": 80,
            "enable_checkpointing": True,
            "ckpt_path": "best",
            # Early stopping: watch the GLOBAL training loss (the sum of
            # all weighted loss terms — NB cell, NB nbr if active, commit,
            # BCE adjacency if active, mask-token reg if active, etc.).
            # This is the actual optimisation objective and the right thing
            # to stop on. base_model logs it as `train_loss`.
            # patience=15 epochs: at batch_size=256 / ~100k cells that is
            # roughly 6k gradient steps, enough to distinguish a true plateau
            # from temporary stagnation after a codebook reshuffle.
            # min_delta=0.1: small in absolute terms, but the train loss for
            # variants in this codebase ranges ~140-170, so 0.1 corresponds
            # to roughly 0.07% relative improvement required per window.
            "early_stopping_params": {
                "enabled":    True,
                "monitor":    "train_loss",
                "mode":       "min",
                "patience":   15,
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
    cfg["trainer"]["monitor"] = "train_pearson_gene_wise_log1p"
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
    cfg["trainer"]["monitor"] = "train_pearson_gene_wise_1hop_nbr_log1p"
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
    cfg["trainer"]["monitor"] = "train_pearson_gene_wise_log1p"
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


def _default_cvq_params(k1: int = 30, k2: int = 10) -> dict:
    """Conditional / tree VQ defaults. Plug into vq_cell_params or vq_niche_params."""
    return {
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
    cfg["trainer"]["monitor"] = "train_pearson_gene_wise_log1p"

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


def _patch_dual_cvq(cfg: dict,
                    branch: str = "both",
                    k1: int = 30, k2: int = 10) -> dict:
    """Swap one or both VQ slots for ConditionalVQ (tree)."""
    if branch in ("cell", "both"):
        cfg["model"]["encoder_params"]["vq_cell_params"]  = _default_cvq_params(k1, k2)
    if branch in ("niche", "both"):
        cfg["model"]["encoder_params"]["vq_niche_params"] = _default_cvq_params(k1, k2)
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


def _patch_dual_film(
        cfg: dict,
        condition_list: List[str] = ["cell_batch_id"],
    ) -> dict:
    """
    Enable FiLM batch-correction on the dual encoder.

    The FiLM module is attached to the MLP output (post-MLP, pre-VQ-cell,
    pre-GNN), so BOTH the cell codebook and the niche codebook see a
    batch-corrected representation. This means cells from different
    samples (e.g. MERFISH brain vs. STARmap brain) with the same
    biological identity should map to the same code.

    `condition_list` controls what the FiLM module is conditioned on.
    Default ['cell_batch_id'] uses the per-cell sample identity (one-hot
    encoded by the data loader from `adata.obs['batch']` if present, or
    parsed from `cell_id` strings as a fallback). You can also include
    'rbf_distances', 'timepoint_id', etc. — anything the loader's
    encoder-conditioning pipeline knows how to populate.
    """
    enc = cfg["model"]["encoder_params"]
    enc["conditioning_params"] = {
        "condition_list":   list(condition_list),
        "use_bias":         True,
        "use_residual":     False,
        "residual_weight":  0.2,
        "init_mode":        "identity",
    }
    return cfg


def _patch_dual_adversarial(
        cfg: dict,
        alpha: float = 1.0,
        wt_adv_batch: float = 100.0,
        hidden_channels: Optional[List[int]] = None,
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
        Strength of the gradient reversal (encoder pressure). 1.0 is
        the standard Ganin-Lempitsky setting; multiplies the negated
        gradient that flows back to the encoder.
    wt_adv_batch: float
        Weight of the CE loss in the total loss. Default 100.0 (was 1.0
        in the original paper recipe). With 2 batches the binary CE is
        bounded by log(2) ≈ 0.69 nats while the cell + niche NB losses
        sit at ~150 nats each; at `wt_adv_batch=1.0` the adversarial
        gradient is ~150x weaker than the reconstruction signal and the
        encoder essentially ignores it. 100.0 brings the adversarial
        gradient into the same order of magnitude as the reconstruction
        gradients. Sweep on a log scale (50, 100, 250, 500) if 100 isn't
        enough — the right value depends on dataset and codebook size.
    hidden_channels: list of int, optional
        Hidden widths for the classifier MLP. Default [128] (one
        hidden layer of 128 units).
    """
    cfg["model"]["adversarial_batch_dim_request"] = True
    cfg["model"]["adversarial_alpha"] = float(alpha)
    if hidden_channels is not None:
        cfg["model"]["adversarial_hidden_channels"] = list(hidden_channels)

    losses = cfg["model"]["loss_params"]["loss_names"]
    if "adversarial_batch_loss" not in losses:
        losses.append("adversarial_batch_loss")
    cfg["model"]["loss_params"]["loss_kwargs"]["wt_adv_batch"] = float(wt_adv_batch)
    return cfg


def _patch_dual_decoder_film(
        cfg: dict,
        condition_list: List[str] = ["cell_batch_id"],
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

    Compared to `_patch_dual_decoder_covariate` (NicheCompass-style
    concat-on-z_q):
      - Concat works cleanly when the decoder is a *linear* scoring head
        over gene programs (NicheCompass): the one-hot has a clean per-
        batch additive effect on each gene rate.
      - With our deeper non-linear MLPSoftmax decoder, the concat path
        diffuses through ReLU/GELU layers and the network has no
        architectural pressure to keep batch effects siloed in the
        covariate's pathway. FiLM gives the covariate a structurally-
        enforced, multiplicative role on every layer's activations,
        which is a better fit for non-linear decoders.

    Encoder is "freed" to encode batch-invariant biological identity
    because the decoder can absorb per-batch gene patterns via FiLM.
    """
    film_params = {
        "condition_list":   list(condition_list),
        "use_bias":         True,
        "use_residual":     False,
        "residual_weight":  0.2,
        "init_mode":        "identity",
    }
    for dec_key in ("attribute_decoder_cell_params",
                    "attribute_decoder_niche_params"):
        cfg["model"][dec_key]["apply_conditioning"] = "in-MLP"
        cfg["model"][dec_key]["conditioning_params"] = film_params.copy()
    return cfg


def _patch_dual_decoder_covariate(cfg: dict) -> dict:
    """
    Enable NicheCompass-style decoder covariate (concat-based batch
    correction).

    What it does: at runtime, `train()` sets `decoder_covariate_dim` to
    the number of distinct samples in the dataset (= n_unique batch one-
    hot dim). The dual model then concatenates the per-cell batch one-hot
    onto z_q_cell and z_q_niche before each decoder. The decoders are
    constructed with extended `in_channels` to consume the concat tensor.

    Effect:
      - Decoders use the batch covariate to fit per-batch / per-platform
        gene patterns (different gene-capture rates, different noise
        structure across MERFISH/STARmap/CosMx).
      - Encoder is "freed" from encoding batch info in z_q because the
        decoder can absorb per-batch differences via the covariate.
      - Codebook (shared across batches) is biased toward batch-invariant
        biological identity.

    Compared to the previous `+film` (FiLM at encoder MLP output): FiLM
    has no positive signal for batch-INVARIANCE; with reconstruction-only
    training it learns to AMPLIFY batch-specific signal because that's
    easier for the decoder to fit. Decoder covariate flips the
    incentive: the decoder gets free access to batch info, so the
    encoder benefits from being batch-invariant rather than batch-
    specific.
    """
    cfg["model"]["decoder_covariate_dim_request"] = True
    return cfg


def _patch_dual_chl59_lung5(
        cfg: dict,
        train_batch_idx: List[int],
        test_batch_idx: List[int],
    ) -> dict:
    """
    Swap the dataset to /nfs/team361/sb75/DATASETS/silver/chl59-8b_1p (8
    CosMx Lung samples) and configure whole-replicate holdout via
    SpatialBatchSplit.

    The 8 source AnnDatas in alphabetical order (sorted as the blob
    builder sees them):
       0  Lung12+SMI+Flat+data.tar.h5ad
       1  Lung13+SMI+Flat+data.tar.h5ad
       2  Lung5_Rep1+SMI+Flat+data.tar.h5ad
       3  Lung5_Rep2+SMI+Flat+data.tar.h5ad
       4  Lung5_Rep3+SMI+Flat+data.tar.h5ad
       5  Lung6+SMI+Flat+data.tar.h5ad
       6  Lung9_Rep1+SMI+Flat+data.tar.h5ad
       7  Lung9_Rep2+SMI+Flat+data.tar.h5ad

    Cells from `train_batch_idx` go entirely to train; cells from
    `test_batch_idx` go entirely to val (-> show up as `val_pearson_*` in
    wandb, computed on the held-out replicate(s)). With the modified
    `SpatialBatchSplit` (region=None semantics), val_batches with no
    region means "all cells are val" — perfect for whole-replicate
    holdout.

    All 8 batches are loaded into the blob; only the train+test subset
    is used (their adata_batch_id values are the positional indices
    above). FiLM batch-correction is configured to read from
    `obs[batch_key]` (default 'batch') so the per-sample / per-replicate
    label is used for batch one-hots.
    """
    cfg["dataset"]["dataset_name"]   = "chl59-8b_1p"
    cfg["dataset"]["root_data_dir"]  = "/nfs/team361/sb75/DATASETS"
    cfg["dataset"]["adata_batch_idx"] = sorted(set(list(train_batch_idx) + list(test_batch_idx)))

    # Whole-batch holdout: train_batches keeps Rep1/Rep2 cells in train,
    # val_batches puts entire Rep3 (and any other holdout replicates) into
    # val. region=None means "all cells are val" (see SpatialBatchSplit).
    cfg["dataset"]["train_transform_params"] = {
        "region":         None,
        "train_batches":  list(train_batch_idx),
        "val_batches":    list(test_batch_idx),
        "test_batches":   [],
        "xy_key":         "xy_coordinates",
    }

    # The Lung AnnDatas have `obs['batch']` populated with per-sample
    # labels; the batch-key flag tells the blob builder to read it.
    cfg["dataset"]["graph_params"]["batch_key"] = "batch"

    # Validation runs on the held-out replicate(s); name the monitor
    # accordingly so checkpoints are saved on test-set Pearson.
    cfg["trainer"]["monitor"] = "val_pearson_gene_wise_log1p"
    return cfg


def _patch_dual_adj_on_zqniche(cfg: dict) -> dict:
    """
    Switch the adjacency BCE input from continuous z_gnn (default) to the
    quantized z_q_niche. Lets you A/B continuous vs. quantized adjacency
    while keeping every other knob fixed.
    """
    cfg["model"]["loss_params"]["loss_kwargs"]["adj_loss_input"] = "z_q_niche"
    return cfg


VARIANTS: dict = {
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
    # FiLM batch-correction variants (MLP-output FiLM, both branches affected)
    # ========================================================================
    "dualvq+wide+film": {
        "description": (
            "Wide dual VQ + FiLM batch-correction at the MLP output. Both "
            "the cell and niche codebooks see a batch-corrected representation, "
            "so cells from different samples (MERFISH/STARmap, replicates, "
            "platforms) with the same biological identity should map to the "
            "same code. Conditioning on `cell_batch_id`. Use this as the "
            "starting point for cross-sample integration."
        ),
        "patches": ["+wide", "+film(MLP-output, cell_batch_id)"],
        "build": lambda: _patch_dual_film(_patch_dual_wide(_BD())),
    },
    # ========================================================================
    # CosMx Lung holdout-replicate variants (dataset: chl59-8b_1p)
    # ========================================================================
    # Whole-replicate holdout: train batches go entirely to the train loader,
    # test batches go entirely to the val loader -> wandb logs `train_*`
    # metrics on the training replicates and `val_*` metrics on the held-out
    # replicate(s). Uses the wide+rvq-both+film recipe (best result on the
    # mouse-brain MERFISH+STARmap data) as the model backbone.
    "dualvq+wide+rvq-both+film+lung5-rep3-test": {
        "description": (
            "Lung5 holdout-replicate setup: train on Lung5_Rep1 + Lung5_Rep2 "
            "(adata_batch_idx=[2, 3]), validate on Lung5_Rep3 (idx=[4]). "
            "Wide+RVQ-both backbone + FiLM batch-correction conditioned on "
            "obs['batch']. `val_pearson_*` metrics in wandb are the "
            "test-set (Rep3) metrics, computed simultaneously each epoch."
        ),
        "patches": [
            "+wide", "+rvq(branch=both, levels=[30, 90])",
            "+film(MLP-output, cell_batch_id)",
            "+chl59_lung5(train=[2,3], test=[4])",
        ],
        "build": lambda: _patch_dual_chl59_lung5(
            _patch_dual_film(
                _patch_dual_rvq(
                    _patch_dual_rvq(
                        _patch_dual_wide(_BD()),
                        branch="niche", codebook_sizes=(30, 90),
                    ),
                    branch="cell", codebook_sizes=(30, 90),
                ),
            ),
            train_batch_idx=[2, 3],
            test_batch_idx=[4],
        ),
    },
    "dualvq+wide+rvq-both+decoder-cov+lung5-rep3-test": {
        "description": (
            "Lung5 holdout-replicate + decoder-covariate batch correction. "
            "Train: Lung5_Rep1 + Lung5_Rep2 (idx=[2, 3]); Val (test): "
            "Lung5_Rep3 (idx=[4]). Decoder covariate concatenates per-cell "
            "batch one-hot to z_q before each decoder, freeing the "
            "codebook to be batch-invariant across replicates."
        ),
        "patches": [
            "+wide", "+rvq(branch=both, levels=[30, 90])",
            "+decoder_covariate", "+chl59_lung5(train=[2,3], test=[4])",
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
            train_batch_idx=[2, 3],
            test_batch_idx=[4],
        ),
    },
    "dualvq+wide+rvq-both+dec-film+lung5-rep3-test": {
        "description": (
            "Lung5 holdout-replicate + FiLM batch-correction INSIDE the "
            "cell + niche decoders (`apply_conditioning='in-MLP'`, "
            "condition=cell_batch_id). Train: Lung5_Rep1 + Lung5_Rep2 "
            "(idx=[2, 3]); Val (test): Lung5_Rep3 (idx=[4]). Direct A/B "
            "against `+decoder-cov+lung5-rep3-test` (concat covariate) "
            "and `+decoder-cov+adv+lung5-rep3-test` (concat + adversary). "
            "FiLM is recommended over concat for the deeper non-linear "
            "MLPSoftmax decoders used here."
        ),
        "patches": [
            "+wide", "+rvq(branch=both, levels=[30, 90])",
            "+dec-film(in-MLP, cell_batch_id)",
            "+chl59_lung5(train=[2,3], test=[4])",
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
            train_batch_idx=[2, 3],
            test_batch_idx=[4],
        ),
    },
    "dualvq+wide+rvq-both+decoder-cov+adv+lung5-rep3-test": {
        "description": (
            "Lung5 holdout-replicate + decoder covariate + domain-"
            "adversarial batch-invariance head. Train: Lung5_Rep1 + "
            "Lung5_Rep2 (idx=[2, 3]); Val (test): Lung5_Rep3 (idx=[4]). "
            "NicheCompass-shape recipe: covariate concat lets the "
            "decoder absorb per-batch gene patterns; adversarial GRL "
            "(applied to FULL z_mlp incl. sampled neighbours) actively "
            "pushes the encoder toward batch-invariance. wt_adv_batch=100 "
            "calibrated against the ~150-nat NB reconstruction losses."
        ),
        "patches": [
            "+wide", "+rvq(branch=both, levels=[30, 90])",
            "+decoder_covariate", "+adversarial_batch(alpha=1.0, wt=100.0)",
            "+chl59_lung5(train=[2,3], test=[4])",
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
                alpha=1.0, wt_adv_batch=100.0,
            ),
            train_batch_idx=[2, 3],
            test_batch_idx=[4],
        ),
    },
    "dualvq+wide+rvq-both+adv+lung5-rep3-test": {
        "description": (
            "Lung5 holdout-replicate + adversarial-only batch correction "
            "(no decoder covariate, no FiLM). Train: Lung5_Rep1 + "
            "Lung5_Rep2 (idx=[2, 3]); Val (test): Lung5_Rep3 (idx=[4]). "
            "Tests whether the GRL alone is enough — should not work as "
            "well as the +decoder-cov+adv combo since the decoder has no "
            "way to fit per-batch gene patterns."
        ),
        "patches": [
            "+wide", "+rvq(branch=both, levels=[30, 90])",
            "+adversarial_batch(alpha=1.0, wt=100.0)",
            "+chl59_lung5(train=[2,3], test=[4])",
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
                alpha=1.0, wt_adv_batch=100.0,
            ),
            train_batch_idx=[2, 3],
            test_batch_idx=[4],
        ),
    },
    "dualvq+wide+rvq-both+decoder-cov+lung5+lung9-multi-test": {
        "description": (
            "Lung5+Lung9 holdout + decoder-covariate batch correction. "
            "Train: 6 samples (idx=[0,1,2,3,5,6]); Val (test): Lung5_Rep3 "
            "+ Lung9_Rep2 (idx=[4, 7]). The decoder covariate is the key "
            "ingredient that makes codes consistent across the 6 training "
            "samples."
        ),
        "patches": [
            "+wide", "+rvq(branch=both, levels=[30, 90])",
            "+decoder_covariate",
            "+chl59_lung5(train=[0,1,2,3,5,6], test=[4,7])",
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
            train_batch_idx=[0, 1, 2, 3, 5, 6],
            test_batch_idx=[4, 7],
        ),
    },
    "dualvq+wide+rvq-both+film+lung5+lung9-multi-test": {
        "description": (
            "Lung5+Lung9 holdout-replicate setup: train on all 8 samples "
            "EXCEPT Lung5_Rep3 and Lung9_Rep2 (adata_batch_idx="
            "[0, 1, 2, 3, 5, 6]); validate on the two held-out replicates "
            "(idx=[4, 7]). Wide+RVQ-both backbone + FiLM batch-correction "
            "(via obs['batch']). `val_pearson_*` metrics are the average "
            "test-set Pearson across both held-out replicates."
        ),
        "patches": [
            "+wide", "+rvq(branch=both, levels=[30, 90])",
            "+film(MLP-output, cell_batch_id)",
            "+chl59_lung5(train=[0,1,2,3,5,6], test=[4,7])",
        ],
        "build": lambda: _patch_dual_chl59_lung5(
            _patch_dual_film(
                _patch_dual_rvq(
                    _patch_dual_rvq(
                        _patch_dual_wide(_BD()),
                        branch="niche", codebook_sizes=(30, 90),
                    ),
                    branch="cell", codebook_sizes=(30, 90),
                ),
            ),
            train_batch_idx=[0, 1, 2, 3, 5, 6],
            test_batch_idx=[4, 7],
        ),
    },
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
            "+decoder_covariate", "+adversarial_batch(alpha=1.0, wt=100.0)",
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
            alpha=1.0, wt_adv_batch=100.0,
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
            "+adversarial_batch(alpha=1.0, wt=100.0)",
        ],
        "build": lambda: _patch_dual_adversarial(
            _patch_dual_rvq(
                _patch_dual_rvq(
                    _patch_dual_wide(_BD()),
                    branch="niche", codebook_sizes=(30, 90),
                ),
                branch="cell", codebook_sizes=(30, 90),
            ),
            alpha=1.0, wt_adv_batch=100.0,
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
    "dualvq+wide+rvq-both+film": {
        "description": (
            "Wide dual VQ + RVQ on both branches (levels=[30, 90] each) + "
            "FiLM batch-correction at the MLP output. Combines the best "
            "single-sample setup with cross-sample batch correction — "
            "recommended starting point for cross-tissue (MERFISH+STARmap) "
            "or cross-replicate atlases."
        ),
        "patches": [
            "+wide",
            "+rvq(branch=niche, levels=[30, 90])",
            "+rvq(branch=cell, levels=[30, 90])",
            "+film(MLP-output, cell_batch_id)",
        ],
        "build": lambda: _patch_dual_film(
            _patch_dual_rvq(
                _patch_dual_rvq(
                    _patch_dual_wide(_BD()),
                    branch="niche", codebook_sizes=(30, 90),
                ),
                branch="cell", codebook_sizes=(30, 90),
            ),
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

def build_blob(dataset: str = "mmb-smb"):
    """
    Build the in-memory PyG DatasetBlob in-process.

    `dataset` selects which builder config to use:
      - 'mmb-smb' (default): MERFISH MB + STARmap MB (`mmb0-1b_smb1-1b_1p_coord_aligned`)
      - 'chl59'              : CosMx Lung 8-sample dataset (`chl59-8b_1p`),
                               built from /nfs/team361/sb75/DATASETS/silver/chl59-8b_1p

    Implementation note: previously this shelled out to
        squint-reproducibility/analysis/create_in_memory_dataset_blob.py
    via subprocess.  That works only if the analysis script on the cluster
    is in sync with the squint package — and it isn't always.  To avoid
    cross-repo file-sync issues we instead instantiate ``InMemoryDatasetBlob``
    here directly with the same arguments.
    """
    CONFIG_OUT_DIR.mkdir(parents=True, exist_ok=True)
    if dataset == "chl59":
        cfg = make_dataset_blob_config_chl59()
        cfg_path = CONFIG_OUT_DIR / "build_blob_chl59.yaml"
    else:
        cfg = make_dataset_blob_config()
        cfg_path = CONFIG_OUT_DIR / "build_blob.yaml"
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
    run_dir = ARTIFACTS_DIR / DATASET_NAME / variant_slug / timestamp
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
        logger = WandbLogger(
            save_dir=str(run_dir),                    # local cache -> <run_dir>/wandb/
            project=WANDB_PROJECT,
            name=run_name,
            # Group by dataset+batches so all ablations of one dataset cluster
            # together. The variant goes in `tags` so we can filter the wandb
            # UI to e.g. all "poc+nbr" runs across timestamps.
            group=f"{cfg['dataset']['dataset_name']}:batch={cfg['dataset']['adata_batch_idx']}",
            job_type="train",
            tags=[f"variant:{variant}", f"timestamp:{timestamp}"],
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
        # a filename pattern PL can format. Fall back to val_pearson_cell_wise.
        _filename_by_monitor = {
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
            monitor, "{epoch}-{val_pearson_cell_wise:.2f}"
        )
        if monitor not in _filename_by_monitor:
            monitor = "val_pearson_cell_wise"

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
                patience  = es_cfg.get("patience", 15),
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
    label_categories_dict: dict,
    source_paths: list[Path],
    gene_names: list[str] | None = None,
) -> "ad.AnnData":
    """
    Package the dict produced by `collate_predict_outputs` into a clean
    AnnData where SQUINT outputs live in the conventional slots:
        .obsm  for embeddings / per-cell vectors
        .obs   for per-cell scalars
        .layers for per-cell, per-gene matrices
        .uns   for global metadata only

    `gene_names`, if provided, is assigned to ``adata.var.index``. If omitted,
    AnnData defaults to integer-string names ('0', '1', ...). The list length
    must equal X.shape[1].
    """
    import anndata as ad
    import numpy as np
    import pandas as pd

    def to_np(x):
        return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)

    # ---- backbone ----
    X = to_np(inference_data["X"])
    adata = ad.AnnData(X=X)
    if gene_names is not None:
        if len(gene_names) != X.shape[1]:
            raise ValueError(
                f"gene_names has length {len(gene_names)} but X has "
                f"{X.shape[1]} columns."
            )
        adata.var = pd.DataFrame(index=pd.Index(gene_names, name="gene"))

    # ---- spatial ----
    if "XY_coordinates" in inference_data:
        adata.obsm["spatial"] = to_np(inference_data["XY_coordinates"])

    # ---- one-hot labels -> string categoricals ----
    for key in inference_data:
        if not key.startswith("y_"):
            continue
        label_name = key[2:]
        cats = (label_categories_dict or {}).get(label_name)
        oh = to_np(inference_data[key])
        idx = oh.argmax(axis=1)
        if cats is not None:
            adata.obs[label_name] = [cats[i] for i in idx]
        else:
            adata.obs[label_name] = idx

    # ---- per-cell scalars ----
    if "adata_batch_ids" in inference_data:
        adata.obs["adata_batch_id"] = to_np(inference_data["adata_batch_ids"]).astype(int)

    # ---- attach source-file column (which .h5ad each cell came from) ----
    # The blob is sorted by adata_batch_id; we mirror that ordering here.
    if "adata_batch_id" in adata.obs.columns and source_paths:
        order = sorted(range(len(source_paths)),
                       key=lambda i: source_paths[i].name)
        ordered_names = [source_paths[i].name for i in order]
        # Map dataset-position index -> filename
        adata.obs["source_file"] = [
            ordered_names[i] if 0 <= i < len(ordered_names) else "unknown"
            for i in adata.obs["adata_batch_id"].astype(int).tolist()
        ]

    # ---- embeddings -> .obsm ----
    if "H_latent" in inference_data:
        adata.obsm["X_squint"] = to_np(inference_data["H_latent"])
    if "H_quantized" in inference_data:
        adata.obsm["X_squint_quantized"] = to_np(inference_data["H_quantized"])
    if "H_adj" in inference_data:
        adata.obsm["X_squint_adj"] = to_np(inference_data["H_adj"])

    # ---- VQNiche_Dual: cell- and niche-branch quantized embeddings ----
    # Per design choice D6: keep them under the user-facing names
    # 'cell_emb' and 'neighborhood_emb' so analysis tools can address each
    # branch directly.
    if "H_quantized_cell" in inference_data:
        adata.obsm["cell_emb"] = to_np(inference_data["H_quantized_cell"])
    if "H_quantized_niche" in inference_data:
        adata.obsm["neighborhood_emb"] = to_np(inference_data["H_quantized_niche"])
    # Also expose the pre-quantization continuous latents for diagnostics.
    if "H_latent_cell" in inference_data:
        adata.obsm["cell_latent"] = to_np(inference_data["H_latent_cell"])
    if "H_latent_niche" in inference_data:
        adata.obsm["neighborhood_latent"] = to_np(inference_data["H_latent_niche"])

    # ---- codebook indices -> .obs (single head) or .obsm (multi-head) ----
    num_heads = int(inference_data.get("num_heads", 1))
    if "Indices" in inference_data:
        idx = to_np(inference_data["Indices"])
        if idx.ndim == 1 or (idx.ndim == 2 and idx.shape[1] == 1):
            adata.obs["code_index"] = idx.reshape(-1).astype(int)
        else:
            adata.obsm["code_indices"] = idx.astype(int)

    # ---- VQNiche_Dual: per-branch codebook indices ----
    # Same single/multi-dim convention as for legacy 'Indices' — 1D goes to
    # `obs[<prefix>_code_index]`, multi-level/multi-head to
    # `obsm[<prefix>_code_indices]`. Prefixes match D6: 'cell' / 'neighborhood'.
    for src_key, prefix in [("Indices_cell", "cell"),
                            ("Indices_niche", "neighborhood")]:
        if src_key not in inference_data:
            continue
        idx = to_np(inference_data[src_key])
        if idx.ndim == 1 or (idx.ndim == 2 and idx.shape[1] == 1):
            adata.obs[f"{prefix}_code_index"] = idx.reshape(-1).astype(int)
        else:
            adata.obsm[f"{prefix}_code_indices"] = idx.astype(int)

    # ---- reconstructions + neighbourhood ground truth -> .layers ----
    if "X_hat" in inference_data:
        adata.layers["X_hat"] = to_np(inference_data["X_hat"])
    if "X_hat_nbr" in inference_data:
        adata.layers["X_hat_nbr"] = to_np(inference_data["X_hat_nbr"])
    # Cache the 1-hop neighbourhood mean of the raw counts as well — same
    # shape (n_cells, n_genes), and required by analysis tools that compare
    # X_hat_nbr against its ground truth without rebuilding the spatial
    # graph at analysis time.
    if "X_nbr" in inference_data:
        adata.layers["X_nbr"] = to_np(inference_data["X_nbr"])

    # ---- global metadata ----
    num_quantizers = int(inference_data.get("num_quantizers", 1))
    cb_sizes       = inference_data.get("codebook_sizes", None)
    squint_meta = {
        "codebook_size":     int(inference_data.get("codebook_size", 0)),
        "num_heads":         num_heads,
        "num_quantizers":    num_quantizers,
        "codebook_sizes":    list(cb_sizes) if cb_sizes is not None else None,
        "separate_codebook": bool(inference_data.get("separate", False)),
    }
    # If this is a dual-model run, also expose per-branch codebook metadata
    # under explicit `cell` / `niche` keys so analysis code can branch on
    # them without having to inspect the obs/obsm slots.
    is_dual = ("H_quantized_cell" in inference_data) or ("H_quantized_niche" in inference_data)
    if is_dual:
        squint_meta["dual"] = True
        for branch in ("cell", "niche"):
            squint_meta[f"codebook_size_{branch}"]  = int(inference_data.get(f"codebook_size_{branch}", 0))
            squint_meta[f"num_quantizers_{branch}"] = int(inference_data.get(f"num_quantizers_{branch}", 1))
            cb_sz = inference_data.get(f"codebook_sizes_{branch}", None)
            if cb_sz is not None:
                squint_meta[f"codebook_sizes_{branch}"] = list(cb_sz)
    adata.uns["squint"] = squint_meta
    if "edge_index" in inference_data:
        # Keep edges in .uns — they're a graph-level object, not per-cell.
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
        Where to write predicted_adata.h5ad. Defaults to
        ``<ARTIFACTS_DIR>/inference/<run_dir-basename>/``.
    """
    # --- Imports done lazily so the script's CLI / config helpers don't
    #     pull torch on every invocation. ---
    if str(REPRO_REPO / "analysis") not in sys.path:
        sys.path.insert(0, str(REPRO_REPO / "analysis"))
    if str(SQUINT_PKG / "src") not in sys.path:
        sys.path.insert(0, str(SQUINT_PKG / "src"))

    import pickle
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
    silver_dir = Path(silver_dir) if silver_dir else (DATA_ROOT / "silver" / DATASET_NAME)

    # Sanity: list of .h5ad files (also used to label cells with their source).
    source_paths = _infer_adata_files_in_dir(silver_dir)
    print(f"Running inference on {len(source_paths)} file(s) under {silver_dir}:")
    for p in source_paths:
        print(f"  - {p.name}")

    # Load the user-saved training config and resolve checkpoint.
    # NOTE: collect_test_configs takes a parameter named `wandb_run_dir` for
    # historical reasons; it now supports both the flat and legacy layouts.
    config = collect_test_configs(
        wandb_run_dir=run_dir,
        model_ckpt_fname=model_ckpt_fname,
    )
    print(f"Using checkpoint: {config['model']['model_ckpt_fname']}")

    # Determinism (mirrors predict_model.test).
    pl.seed_everything(config["experiment"]["seed"])

    # Build dataset + datamodule + model from the saved config.
    dataset_blob = initialize_dataset_blob(config)
    with open(Path(dataset_blob.processed_dir) / "label_categories.pkl", "rb") as f:
        label_categories = pickle.load(f)

    # Try to load the canonical gene panel saved at blob-build time so we can
    # propagate gene names through HVG selection into the predicted AnnData.
    gene_panel = None
    gene_panel_path = Path(dataset_blob.processed_dir) / "gene_panel.pkl"
    if gene_panel_path.exists():
        with open(gene_panel_path, "rb") as f:
            gene_panel = pickle.load(f)

    data_batch = initialize_databatch(config=config, dataset_blob=dataset_blob)
    datamodule = initialize_datamodule(
        config=config,
        data=data_batch,
        obs_per_batch_id=getattr(dataset_blob, 'obs_per_batch_id', None),
    )

    # Recover the post-HVG gene names (if available). SubsetHVG stores the
    # surviving column indices on each Data section as `hvg_indices`. Indexing
    # the dataset_blob applies the transform pipeline; the resulting Data has
    # `.hvg_indices` (LongTensor) we can use to look up names in gene_panel.
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

    # Build the clean AnnData.
    adata = _build_clean_adata_from_inference(
        inference_data=predict_data_dict,
        label_categories_dict=label_categories,
        source_paths=source_paths,
        gene_names=gene_names,
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
    if output_dir is None:
        # We support three run-dir layouts and try to mirror the structure
        # under <INFERENCE_DIR>/ so inference outputs are co-located with
        # their training runs by variant:
        #
        #   (1) New (variant-aware):
        #         <ARTIFACTS_DIR>/<DATASET>/<VARIANT>/<TIMESTAMP>/
        #       -> ablation_summary.yaml is present at run_dir.
        #       -> output: <INFERENCE_DIR>/<VARIANT>/<TIMESTAMP>/
        #
        #   (2) Old flat (pre-ablation):
        #         <ARTIFACTS_DIR>/<DATASET>/<TIMESTAMP>/
        #       -> output: <INFERENCE_DIR>/<TIMESTAMP>/
        #
        #   (3) Legacy wandb-controlled:
        #         <ARTIFACTS_DIR>/.../<RUN_NAME>/files/
        #       -> output: <INFERENCE_DIR>/<RUN_NAME>/
        run_dir_path = Path(run_dir)
        if (run_dir_path / "ablation_summary.yaml").exists():
            # New layout: variant is the parent dir name.
            variant_slug = run_dir_path.parent.name
            run_name = run_dir_path.name
            output_dir = INFERENCE_DIR / variant_slug / run_name
        elif run_dir_path.name == "files":
            output_dir = INFERENCE_DIR / run_dir_path.parent.name
        else:
            output_dir = INFERENCE_DIR / run_dir_path.name
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "predicted_adata.h5ad"

    # AnnData can't write nested-dict .uns -> flatten the squint metadata.
    # We keep the dict but cast values to writeable types.
    adata.uns["squint"] = {k: (str(v) if not isinstance(v, (int, float, bool)) else v)
                          for k, v in adata.uns["squint"].items()}
    adata.write_h5ad(out_path)

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

def run_all_pipeline(
        variant: str,
        silver_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        model_ckpt_fname: Optional[str] = None,
        label_keys: str = "cell_type",
        batch_rename: Optional[str] = None,
    ) -> None:
    """
    End-to-end pipeline:

      1. train(variant)            -> writes <ARTIFACTS_DIR>/<dataset>/<variant>/<TS>/
      2. predict(run_dir=auto)     -> writes <ARTIFACTS_DIR>/inference/<run-name>/
                                       predicted_adata.h5ad
      3. plot_code_indices_spatial -> per-batch spatial plots of code indices
      4. plot_svg_reconstruction   -> SVG vs. reconstruction sanity plots
      5. plot_latent_umap          -> UMAP of each latent slot

    Plot scripts run via subprocess so each gets a clean process / cuda
    state and the output of one doesn't pollute the others. They write
    SVGs/PNGs into subdirectories of the predict output folder.
    """
    import subprocess

    # ------------------------------------------------------------------
    # 1. Train
    # ------------------------------------------------------------------
    print("=" * 78)
    print(f"[1/5] Train  variant={variant!r}")
    print("=" * 78)
    run_dir = train(variant)
    if run_dir is None:
        # Older train() signatures didn't return; fall back to looking up
        # the most recent run dir for this variant.
        variant_slug = variant.replace("/", "_")
        candidates = sorted((ARTIFACTS_DIR / DATASET_NAME / variant_slug).glob("*/"))
        if not candidates:
            raise RuntimeError(f"Could not auto-locate run_dir for variant {variant}")
        run_dir = str(candidates[-1])
    print(f"[1/5] Done. run_dir={run_dir}")

    # ------------------------------------------------------------------
    # 2. Predict
    # ------------------------------------------------------------------
    print()
    print("=" * 78)
    print(f"[2/5] Predict on run_dir={run_dir}")
    print("=" * 78)
    predict(
        run_dir=run_dir,
        silver_dir=silver_dir,
        model_ckpt_fname=model_ckpt_fname,
        output_dir=output_dir,
    )
    # Resolve where the predicted_adata.h5ad landed.
    if output_dir is None:
        run_name = Path(run_dir).name
        # Match the naming convention used inside `predict()`:
        # variant slug parent of run_dir + run-name child.
        variant_slug = Path(run_dir).parent.name
        predict_dir = INFERENCE_DIR / variant_slug / run_name
    else:
        predict_dir = Path(output_dir)
    predicted_adata_path = predict_dir / "predicted_adata.h5ad"
    if not predicted_adata_path.exists():
        raise FileNotFoundError(
            f"Expected predicted AnnData at {predicted_adata_path} but it's missing."
        )
    print(f"[2/5] Done. predicted_adata={predicted_adata_path}")

    # ------------------------------------------------------------------
    # 3. plot_code_indices_spatial
    # ------------------------------------------------------------------
    examples_dir = Path(__file__).parent
    common_args = ["--predicted-adata", str(predicted_adata_path)]

    print()
    print("=" * 78)
    print("[3/5] plot_code_indices_spatial")
    print("=" * 78)
    subprocess.run(
        [sys.executable, str(examples_dir / "plot_code_indices_spatial.py")] + common_args,
        check=True,
    )

    # ------------------------------------------------------------------
    # 4. plot_svg_reconstruction
    # ------------------------------------------------------------------
    print()
    print("=" * 78)
    print("[4/5] plot_svg_reconstruction")
    print("=" * 78)
    subprocess.run(
        [sys.executable, str(examples_dir / "plot_svg_reconstruction.py")] + common_args,
        check=True,
    )

    # ------------------------------------------------------------------
    # 5. plot_latent_umap
    # ------------------------------------------------------------------
    print()
    print("=" * 78)
    print("[5/5] plot_latent_umap")
    print("=" * 78)
    umap_args = list(common_args) + ["--label-keys", label_keys]
    if batch_rename:
        umap_args += ["--batch-rename", batch_rename]
    subprocess.run(
        [sys.executable, str(examples_dir / "plot_latent_umap.py")] + umap_args,
        check=True,
    )

    print()
    print("=" * 78)
    print("Pipeline complete.")
    print(f"  Run dir:           {run_dir}")
    print(f"  Predicted AnnData: {predicted_adata_path}")
    print(f"  Plots:             {predict_dir}/{{code_index_plots,svg_plots,umap_plots}}/")
    print("=" * 78)


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
    p.add_argument("--build-blob-dataset", type=str, default="mmb-smb",
                   choices=["mmb-smb", "chl59"],
                   help="Which dataset to build (default: mmb-smb). "
                        "'chl59' = /nfs/team361/sb75/DATASETS/silver/chl59-8b_1p "
                        "(CosMx Lung, 8 samples).")
    p.add_argument("--train", action="store_true",
                   help="Train SQUINT on the WHOLE dataset blob (logs to wandb project 'squint').")
    p.add_argument("--variant", type=str, default="poc",
                   help="Which training/ablation config to use (default: poc). "
                        "Use --list-variants to see all registered variants.")
    p.add_argument("--list-variants", action="store_true",
                   help="Print all registered ablation variants and exit.")
    p.add_argument("--predict", action="store_true",
                   help="Run inference on a folder of .h5ad files using a trained checkpoint.")
    p.add_argument("--all", action="store_true",
                   help="End-to-end: train -> predict -> run all 3 plot scripts "
                        "(plot_code_indices_spatial, plot_svg_reconstruction, "
                        "plot_latent_umap) on the predicted AnnData. Convenience "
                        "wrapper for `--train --variant <V>` followed by "
                        "`--predict --run-dir <auto>`.")
    p.add_argument("--batch-rename", type=str, default=None,
                   help="Optional comma-separated rename for adata_batch_id "
                        "categories in UMAP plots (e.g. 'MERFISH,STARmap PLUS').")
    p.add_argument("--label-keys", type=str, default="cell_type",
                   help="Comma-separated obs columns to color UMAPs and code-index "
                        "plots by (default: cell_type).")
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
                   help="Where to write predicted_adata.h5ad (default: "
                        "<ARTIFACTS_DIR>/inference/<run-name>/).")
    args = p.parse_args()

    if args.list_variants:
        print("Registered ablation variants:")
        for name, spec in VARIANTS.items():
            print(f"  {name}")
            print(f"    description: {spec['description']}")
            print(f"    patches:     {spec['patches']}")
        return

    if not (args.patch_uns or args.harmonize_var or args.build_blob
            or args.train or args.predict or args.all):
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

    # --all: end-to-end train -> predict -> 3 plot scripts.
    if args.all:
        run_all_pipeline(
            variant=args.variant,
            silver_dir=args.silver_dir,
            output_dir=args.output_dir,
            model_ckpt_fname=args.model_ckpt_fname,
            label_keys=args.label_keys,
            batch_rename=args.batch_rename,
        )


if __name__ == "__main__":
    main()
