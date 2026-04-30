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

def build_blob():
    """
    Build the in-memory PyG DatasetBlob in-process.

    Implementation note: previously this shelled out to
        squint-reproducibility/analysis/create_in_memory_dataset_blob.py
    via subprocess.  That works only if the analysis script on the cluster
    is in sync with the squint package — and it isn't always.  To avoid
    cross-repo file-sync issues we instead instantiate ``InMemoryDatasetBlob``
    here directly with the same arguments.
    """
    CONFIG_OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg_path = CONFIG_OUT_DIR / "build_blob.yaml"
    cfg = make_dataset_blob_config()
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
    datamodule_batch = initialize_datamodule(config=cfg, data=data_batch)

    # ---- Model: bind condition dims if FiLM is enabled --------------------
    if "conditioning_params" in cfg["model"]["encoder_params"]:
        cfg["model"]["encoder_params"]["conditioning_params"]["condition_dim"] = (
            data_batch.encoder_condition_dim
        )
    if "spatial_prior_params" in cfg["model"]["encoder_params"]:
        cfg["model"]["encoder_params"]["spatial_prior_params"][
            "spatial_prior_feature_dim"
        ] = data_batch.spatial_prior_feature_dim
    if "conditioning_params" in cfg["model"]["attribute_decoder_params"]:
        cfg["model"]["attribute_decoder_params"]["conditioning_params"][
            "condition_dim"
        ] = data_batch.attr_decoder_condition_dim
    if "conditioning_params" in cfg["model"]["adjacency_decoder_params"]:
        cfg["model"]["adjacency_decoder_params"]["conditioning_params"][
            "condition_dim"
        ] = data_batch.adj_decoder_condition_dim

    model = initialize_model(
        config=cfg,
        in_channels=data_batch.num_features,
        out_channels=data_batch.num_classes,
    )

    # ---- Logger: wandb cloud logging, local cache lives inside run_dir ----
    logging_enabled = cfg["logging"].get("enabled", True)
    if logging_enabled:
        logger = WandbLogger(
            save_dir=str(run_dir),                    # local cache -> <run_dir>/wandb/
            project=WANDB_PROJECT,
            # Group by dataset+batches so all ablations of one dataset cluster
            # together. The variant goes in `tags` so we can filter the wandb
            # UI to e.g. all "poc+nbr" runs across timestamps.
            group=f"{cfg['dataset']['dataset_name']}:batch={cfg['dataset']['adata_batch_idx']}",
            job_type="train",
            tags=[f"variant:{variant}"],
            mode="offline" if cfg["logging"]["offline"] else "online",
            log_model=cfg["logging"]["log_model"],
            config={"variant": variant, "ablation_summary": summary, **cfg},
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

    # ---- codebook indices -> .obs (single head) or .obsm (multi-head) ----
    num_heads = int(inference_data.get("num_heads", 1))
    if "Indices" in inference_data:
        idx = to_np(inference_data["Indices"])
        if idx.ndim == 1 or (idx.ndim == 2 and idx.shape[1] == 1):
            adata.obs["code_index"] = idx.reshape(-1).astype(int)
        else:
            adata.obsm["code_indices"] = idx.astype(int)

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
    adata.uns["squint"] = {
        "codebook_size":     int(inference_data.get("codebook_size", 0)),
        "num_heads":         num_heads,
        "num_quantizers":    num_quantizers,
        "codebook_sizes":    list(cb_sizes) if cb_sizes is not None else None,
        "separate_codebook": bool(inference_data.get("separate", False)),
    }
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
    datamodule = initialize_datamodule(config=config, data=data_batch)

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

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--patch-uns", action="store_true",
                   help="Patch .uns/.obs of the two .h5ad files (one-time, only if missing).")
    p.add_argument("--harmonize-var", action="store_true",
                   help="Subset and align .var across the two .h5ad files (one-time, "
                        "run if --build-blob fails with 'All batches must have the same gene panel').")
    p.add_argument("--build-blob", action="store_true",
                   help="Build the in-memory PyG DatasetBlob (one-time).")
    p.add_argument("--train", action="store_true",
                   help="Train SQUINT on the WHOLE dataset blob (logs to wandb project 'squint').")
    p.add_argument("--variant", type=str, default="poc",
                   help="Which training/ablation config to use (default: poc). "
                        "Use --list-variants to see all registered variants.")
    p.add_argument("--list-variants", action="store_true",
                   help="Print all registered ablation variants and exit.")
    p.add_argument("--predict", action="store_true",
                   help="Run inference on a folder of .h5ad files using a trained checkpoint.")
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

    if not (args.patch_uns or args.harmonize_var or args.build_blob or args.train or args.predict):
        p.print_help()
        return

    if args.patch_uns:
        patch_anndata_uns()
    if args.harmonize_var:
        harmonize_anndata_var()
    if args.build_blob:
        build_blob()
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


if __name__ == "__main__":
    main()
