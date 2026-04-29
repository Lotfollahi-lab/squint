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
  #      poc      = minimal Graph VQ-VAE baseline   (start here)
  #      poc+nbr  = POC + 1-hop neighborhood NB reconstruction (first ablation)
  #      full     = full SQUINT (FiLM + masking + adjacency loss + ...)
  #    Output (flat run dir, ablation-aware):
  #      <ARTIFACTS_DIR>/<DATASET_NAME>/<VARIANT>/<YYYYMMDD_HHMMSS>/
  #          user_specified_config.yaml      <- full materialised config
  #          ablation_summary.yaml           <- variant + description + patches
  #          checkpoints/<best>.ckpt
  #          wandb/run-<id>/...              (wandb local cache; cloud logging
  #                                           still goes to project "squint")
  python examples/run_squint_mmb_smb.py --train --variant poc
  python examples/run_squint_mmb_smb.py --train --variant poc+nbr

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
    Minimal Graph VQ-VAE: GraphSAGE encoder + per-cell NB reconstruction +
    EMA codebook. No FiLM, no masking, no adjacency loss. Use this first to
    verify the core mechanism trains on this data.

    For now, train on the WHOLE dataset (both batches, every cell).
    No held-out test/val set — evaluation/inference is done after training
    via `predict()`, which loops over all nodes regardless of mask.
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
            # adata_batch_idx are *positions* in the sorted DatasetBlob, not
            # the raw batch numbers. After sorting by adata_batch_id:
            #   idx 0 -> batch15 (STARmap+)
            #   idx 1 -> batch82 (MERFISH)
            "adata_batch_idx": [0, 1],
            "root_data_dir": str(DATA_ROOT),
            "gene_count_transform_names": ["SubsetHVG"],
            "gene_count_transform_params": {"n_genes": 1000},
            "graph_params": {
                "spatial_key": "spatial",
                "delaunay": False,
                "n_neighs": 8,
                "radius": None,
            },
            "feature_names": ["X"],
            "label_name": "cell_types",
            # SpatialBatchSplit assigns train/val/test masks per cell. With
            # both batches (15 and 82) listed in `train_batches` and nothing
            # in val/test, every cell gets train_mask=True. The model trains
            # on the WHOLE dataset. (`train_batches`, `val_batches`, and
            # `test_batches` use actual batch IDs, not dataset indices.)
            #
            # Inference / evaluation is done after training via the predict
            # entry point below, whose `predict_dataloader` loops over all
            # nodes (input_nodes=None) regardless of mask — so we still get
            # embeddings + reconstructions on every cell.
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
            "loader_params": {"batch_size": 128},
            "sampler_name": "NeighborSampler",
            "sampler_params": {"num_neighbors": [8, 8]},
            "inference_params": {"sample_neighbors_for_inference": False},
        },
        "model": {
            "model_name": "VQNiche",
            "encoder_name": "VQNiche_Encoder",
            "attribute_decoder_name": "MLPSoftmax",
            "adjacency_decoder_name": "MLP_AdjacencyDecoder",
            "predictor_name": "Linear",
            "imputation_params": {
                # POC: NO masking
                "mask_strategy": "original",
                "base_mask_ratio": 0.0,
                "final_mask_ratio": 0.0,
                "warmup_epochs": 0,
                "deterministic_masking": False,
                "compute_mask_input_diversity": False,
                "mask_token_eps": 0.0001,
            },
            "train_metrics_list": ["codebook_utilization", "pearson_cell_wise"],
            "test_metrics_list": [
                "codebook_utilization", "pearson_cell_wise", "pearson_gene_wise",
            ],
            "encoder_params": {
                "gnn_name": "SAGEConv",
                "mlp_params": {
                    "hidden_channels": [600, 400],
                    "dropout": 0.1,
                    "act": "relu",
                    "norm": None,
                },
                "gnn_params": {
                    "hidden_channels": 400,
                    "num_layers": 1,
                    "act_first": True,
                    "activation": "relu",
                    "norm": None,
                    "dropout": 0.1,
                    "init_method": "kaiming_uniform",
                },
                # POC: no FiLM conditioning -> conditioning_params block omitted
                "vq_params": {
                    "vq_name": "VectorQuantize",
                    "freeze_codebook": False,
                    "use_cosine_sim": True,
                    "ema_update": True,
                    "manual_ema_update": False,
                    "threshold_ema_dead_code": 2,
                    "manual_in_place_optimizer_update": False,
                    "learnable_codebook": False,
                    # 30 codes, single head -> each cell gets one index in [0, 30).
                    # This is intentionally tight so the codebook acts like a
                    # clustering and can be compared against an scVI/Leiden
                    # partition of similar cardinality.
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
                    "hidden_channels": [600],
                    "dropout": 0.1,
                    "act": "gelu",
                    "norm": "layer_norm",
                },
            },
            "adjacency_decoder_params": {
                "out_channels": 600,
                "mlp_params": {
                    "hidden_channels": [400],
                    "dropout": 0.2,
                    "act": "relu",
                    "norm": "layer_norm",
                },
            },
            "optimizer_params": {
                "optimizer_name": "adam",
                "lr": 0.001,
                "weight_decay": 0.001,
                "mask_lr_scale": 1.0,
            },
            "loss_params": {
                "loss_names": [
                    "nb_attribute_reconstruction_loss",
                    "mse_commit_loss",
                ],
                "loss_kwargs": {
                    "k_hop_nb_loss": 0,
                    "only_masked": False,
                    "edge_sampling_ratio": 2,
                    "use_pos_weight": True,
                    "estimate_adj_kwargs": {"nonlinearity": "sigmoid", "k": 8},
                    "wt_attr_reconstr": 1.0,
                    "wt_commit": 1.0,
                },
            },
        },
        "trainer": {
            "monitor": "train_pearson_cell_wise",
            "checkpoint_params": {"mode": "max", "save_top_k": 1, "save_last": True},
            "max_epochs": 30,
            "enable_checkpointing": True,
            "ckpt_path": "best",
        },
    }


# ---------------------------------------------------------------------------
# 2b. Training config — full SQUINT (FiLM + masking + adjacency)
# ---------------------------------------------------------------------------

def make_train_config_full() -> dict:
    """
    The full SQUINT recipe with FiLM conditioning on (rbf_distances,
    cell_batch_id), MAE-style masking, BCE adjacency reconstruction, and
    1-hop NB reconstruction. Only switch to this once the POC trains stably.
    """
    cfg = make_train_config_poc()

    # ---- masking back on ----
    cfg["model"]["imputation_params"] = {
        "mask_strategy": "learnable_parameter",
        "base_mask_ratio": 0.2,
        "final_mask_ratio": 0.6,
        "warmup_epochs": 5,
        "deterministic_masking": False,
        "compute_mask_input_diversity": False,
        "mask_token_eps": 0.0001,
    }
    cfg["model"]["train_metrics_list"] = [
        "codebook_utilization", "pearson_1hop_nbr",
    ]
    cfg["model"]["test_metrics_list"] = [
        "codebook_utilization",
        "pearson_cell_wise",
        "pearson_1hop_nbr",
        "mmd_1hop_nbr",
        "mmd_pca_1hop_nbr",
        "pearson_gene_wise",
        "pearson_gene_wise_1hop_nbr",
    ]

    # ---- FiLM in encoder ----
    cfg["model"]["encoder_params"]["conditioning_params"] = {
        "condition_list": ["rbf_distances", "cell_batch_id"],
        "use_bias": True,
        "use_residual": False,
        "residual_weight": 0.2,
        "init_mode": "identity",
    }

    # ---- VQ scaled back up to SQUINT defaults ----
    cfg["model"]["encoder_params"]["vq_params"].update({
        "codebook_size": 5000,
        "heads": 10,
        "separate_codebook_per_head": False,
        # threshold_ema_dead_code stays 2 — see literature_review.md §1.2.
    })

    # ---- FiLM in attribute decoder ----
    cfg["model"]["attribute_decoder_params"]["apply_conditioning"] = "in-MLP"
    cfg["model"]["attribute_decoder_params"]["conditioning_params"] = {
        "condition_list": ["rbf_distances", "cell_batch_id"],
        "use_bias": True,
        "use_residual": False,
        "residual_weight": 0.2,
        "init_mode": "identity",
    }

    # ---- mask_lr_scale matters now that masking is on ----
    cfg["model"]["optimizer_params"]["mask_lr_scale"] = 2.0

    # ---- losses: full set ----
    cfg["model"]["loss_params"]["loss_names"] = [
        "nb_attribute_reconstruction_loss",
        "bce_adjacency_reconstruction_loss",
        "mse_commit_loss",
        "mse_code_loss",
        "mask_token_regularization",
    ]
    cfg["model"]["loss_params"]["loss_kwargs"].update({
        "k_hop_nb_loss": 1,
        "wt_cross_entropy": 1.0,
        "wt_adj_reconstr": 1.0,
        "wt_joint_code_commit": 1.0,
        # Under EMA codebook, mse_code_loss is redundant with commit — keep 0.
        "wt_code": 0.0,
        # Squared-L2 mask-token reg has init magnitude ~5000 (1000-dim mask
        # token ~ N(2,1)); scale weight down so it doesn't dominate.
        "wt_mask_token_regularization": 0.001,
    })

    # ---- monitor uses 1-hop neighborhood Pearson now ----
    cfg["trainer"]["monitor"] = "train_pearson_1hop_nbr"
    cfg["trainer"]["max_epochs"] = 30
    return cfg


# ---------------------------------------------------------------------------
# 2c. Ablation patches and variant registry
# ---------------------------------------------------------------------------
# Each ablation is expressed as a small "patch" function that mutates a copy
# of the POC config in place and returns it. A patch should change ONE
# component (one loss term, one architectural choice, one hyperparameter
# group) so its effect can be isolated. New variants are added by composing
# patches and registering them in VARIANTS below.
#
# The full resolved config is always saved to disk per run so reproducibility
# does not depend on the patch helpers being preserved.
# ---------------------------------------------------------------------------

def _copy(cfg: dict) -> dict:
    import copy
    return copy.deepcopy(cfg)


def _patch_khop_nb_loss(cfg: dict, k_hop: int = 1) -> dict:
    """
    Turn on k-hop neighborhood NB reconstruction. The model is now asked to
    predict, for each cell, the mean expression of its k-hop neighborhood
    (in addition to its own counts). Drives the latent toward niche-aware
    structure rather than purely cell-intrinsic identity.
    """
    cfg["model"]["loss_params"]["loss_kwargs"]["k_hop_nb_loss"] = k_hop
    metrics = cfg["model"]["train_metrics_list"]
    if "pearson_1hop_nbr" not in metrics:
        cfg["model"]["train_metrics_list"] = metrics + ["pearson_1hop_nbr"]
    test_metrics = cfg["model"]["test_metrics_list"]
    if "pearson_1hop_nbr" not in test_metrics:
        cfg["model"]["test_metrics_list"] = test_metrics + ["pearson_1hop_nbr"]
    cfg["trainer"]["monitor"] = "train_pearson_1hop_nbr"
    return cfg


# Future patches to add as new ablations are introduced. Each one should be a
# small standalone function so it can be composed independently.
#
# def _patch_adj_loss(cfg, weight=0.3):  ... # add bce_adjacency_reconstruction_loss
# def _patch_film(cfg):                 ... # add encoder/decoder FiLM conditioning
# def _patch_masking(cfg):              ... # MAE-style masking
# def _patch_deeper_gnn(cfg, layers=2): ... # GraphSAGE -> 2-3 layers + matching neighbor sample sizes
# def _patch_codebook_size(cfg, k):     ... # change vq.codebook_size


# ---------------------------------------------------------------------------
# Variant registry
# ---------------------------------------------------------------------------

VARIANTS: dict = {
    "poc": {
        "description": (
            "Graph VQ-VAE baseline. Per-cell NB reconstruction only. "
            "No neighborhood loss, no FiLM, no masking, no adjacency loss. "
            "Codebook=30, heads=1."
        ),
        "patches": [],
        "build": lambda: make_train_config_poc(),
    },
    "poc+nbr": {
        "description": (
            "POC + 1-hop neighborhood NB reconstruction. First ablation: "
            "tests whether forcing the model to predict the neighborhood "
            "mean counts pushes codes toward CellCharter/BANKSY-style niche "
            "clusters. Everything else identical to POC."
        ),
        "patches": ["k_hop_nb_loss=1", "monitor=train_pearson_1hop_nbr"],
        "build": lambda: _patch_khop_nb_loss(_copy(make_train_config_poc()), k_hop=1),
    },
    "full": {
        "description": (
            "Full SQUINT recipe: FiLM (rbf_distances + cell_batch_id) + "
            "MAE-style masking + adjacency reconstruction + 1-hop NB "
            "reconstruction + multi-head VQ (heads=10, codebook=5000)."
        ),
        "patches": ["see make_train_config_full()"],
        "build": lambda: make_train_config_full(),
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

    # ---- Checkpoint callback: write directly to <run_dir>/checkpoints/ ----
    enable_checkpointing = cfg["trainer"]["enable_checkpointing"]
    if enable_checkpointing:
        monitor = cfg["trainer"]["monitor"]
        # filename templates known to ModelCheckpoint -> map monitor metric to
        # a filename pattern PL can format. Fall back to val_pearson_cell_wise.
        _filename_by_monitor = {
            "val_pearson_cell_wise":           "{epoch}-{val_pearson_cell_wise:.2f}",
            "val_pearson_1hop_nbr":            "{epoch}-{val_pearson_1hop_nbr:.2f}",
            "val_pearson_gene_wise":           "{epoch}-{val_pearson_gene_wise:.2f}",
            "val_pearson_gene_wise_1hop_nbr":  "{epoch}-{val_pearson_gene_wise_1hop_nbr:.2f}",
            "train_pearson_cell_wise":         "{epoch}-{train_pearson_cell_wise:.2f}",
            "train_pearson_1hop_nbr":          "{epoch}-{train_pearson_1hop_nbr:.2f}",
            "train_pearson_gene_wise":         "{epoch}-{train_pearson_gene_wise:.2f}",
            "train_pearson_gene_wise_1hop_nbr": "{epoch}-{train_pearson_gene_wise_1hop_nbr:.2f}",
        }
        filename = _filename_by_monitor.get(
            monitor, "{epoch}-{val_pearson_cell_wise:.2f}"
        )
        if monitor not in _filename_by_monitor:
            monitor = "val_pearson_cell_wise"

        checkpoint_params = cfg["trainer"]["checkpoint_params"]
        callbacks = [
            pl.callbacks.ModelCheckpoint(
                dirpath=str(ckpt_dir),
                monitor=monitor,
                filename=filename,
                **checkpoint_params,
            )
        ]
    else:
        callbacks = None

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

    # ---- reconstructions -> .layers ----
    if "X_hat" in inference_data:
        adata.layers["X_hat"] = to_np(inference_data["X_hat"])
    if "X_hat_nbr" in inference_data:
        adata.layers["X_hat_nbr"] = to_np(inference_data["X_hat_nbr"])

    # ---- global metadata ----
    adata.uns["squint"] = {
        "codebook_size": int(inference_data.get("codebook_size", 0)),
        "num_heads": num_heads,
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
