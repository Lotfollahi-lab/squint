# SQUINT

SQUINT learns two discrete codes per cell from spatial transcriptomics data: a
cell code that mainly reflects cell-intrinsic identity and a niche code that
mainly reflects the cell's spatial context. It is a graph vector-quantized
variational autoencoder. An MLP encodes each cell's counts into a cell latent,
which a cell codebook quantises directly; a graph neural network over the
spatial k-nearest-neighbour graph of each section aggregates these latents into
a neighbourhood latent, which a niche codebook quantises. Both codebooks are
two-level residual stacks (30 and 90 codes by default), so every cell receives a
coarse-to-fine code tuple per branch. Two negative-binomial decoders
reconstruct per-cell and 1-hop-neighbourhood counts; training adds a
within-section cosine adjacency loss, a cross-section mutual-nearest-neighbour
contrastive loss on the cell latents and a decoder section covariate. A
separate second stage, a MaskGIT-style masked-code transformer over the frozen
codes, in-paints the codes of held-out regions for gene-expression imputation.

This repository holds the model, training and inference code for the MLCB 2026
paper *Learning Discrete Cell and Niche Codes from Spatial Transcriptomics Using
Dual Residual Vector Quantization* (see [Citation](#citation)). The code that
reproduces the paper's experiments is in
[squint-reproducibility](https://github.com/Lotfollahi-lab/squint-reproducibility).

SQUINT extends a single-codebook precursor of the same name, presented at the
ICLR 2026 MLGenX workshop (Merchant et al., 2026; see [Citation](#citation)). The residual
quantiser (`src/squint/modules/hierarchical_vq.py`) builds on
[vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch),
and `src/squint/vqgraph/` is an adapted copy of the
[VQGraph](https://github.com/YangLing0818/VQGraph) code, of which the dual model
uses only `vq.py`.

## Installation

You need Python 3.10 to 3.12; the commands below use 3.10. `pyproject.toml`
pins torch 2.2.0 to the **CUDA 12.1 wheel index** and pulls the matching
PyTorch Geometric companion wheels (pyg-lib, torch-scatter, torch-sparse,
torch-cluster, torch-spline-conv) from `data.pyg.org`. The recommended workflow
uses [uv](https://github.com/astral-sh/uv), which reads both settings from
`pyproject.toml`.

### Recommended: uv

```bash
# 1. Install uv (one-time)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone, then create .venv and install everything from the repo root
git clone https://github.com/Lotfollahi-lab/squint.git
cd squint
uv venv .venv --python 3.10
uv sync                  # no lock file is committed; uv resolves the pins in pyproject.toml
source .venv/bin/activate
```

For a CPU-only or a different CUDA build, the comment block at the top of
`pyproject.toml` lists the three places to change (the `find-links` URL, the
index in `[tool.uv.sources]` and the PyG companion pins).

### Fallback: pure pip

```bash
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121
# Then, from the repo root:
pip install -e . -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
```

Install from this repository only. The name `squint` on PyPI belongs to an
unrelated package, so do not run `pip install squint` or `pip install -U
squint`: that replaces this package and `import squint` then loads the wrong
library.

`environment.yaml` records the conda environment of early development. It
predates the current dependencies (it lacks `vector-quantize-pytorch` and
`scib-metrics`, for example) and is not maintained; use `pyproject.toml`.

### Upgrading an environment from `vqniche`

This package used to be called `vqniche`. In an environment that has the old
package installed, and in an existing checkout:

```bash
pip uninstall vqniche        # or: uv pip uninstall vqniche
rm -rf src/vqniche           # leftover, git-ignored __pycache__ directories
pip install -e .             # or: uv pip install -e .
python -c "import squint; print(squint.__file__, squint.__version__)"
```

Configurations and checkpoints that still name `VQNiche` or `VQNiche_Dual`
(or their encoders) keep working: the old class names are mapped to `Squint`
and `Squint_Dual`.

## Input data

Each dataset is a folder of per-section `.h5ad` files under a data root that
also receives the processed dataset:

```text
<DATA_ROOT>/
├── silver/<dataset>/                              # input: one .h5ad per section
│   ├── <section_1>.h5ad
│   └── <section_2>.h5ad
└── gold/in-memory-PyG-dataset-blob/<dataset>/     # written by --build-blob
```

Each file holds one section, with

- `.X`: raw counts (cells × genes; sparse or dense). All sections must have the
  same genes and the same per-gene `.var` columns (the order may differ; files
  are reindexed to the first one);
- `.obsm["spatial"]`: 2-D coordinates. The k-NN graph is built per section, so
  coordinates of different sections need not be aligned;
- `.obs["cell_id"]`: a cell identifier;
- `.uns["batch"]`: the section identifier, an int, an int-string or `"batchN"`
  (required; it is also the section covariate of the decoders);
- `.uns["dataset_id"]`, `.uns["tissue"]`, `.uns["species"]` (optional
  metadata);
- label columns in `.obs` (optional): cell types and niches, registered in the
  dataset's blob configuration (`label_names`, e.g. `"cell_types=cell_type"`).
  Training does not use them; they are only scored after inference.

## Usage

The paper's models were trained on a single NVIDIA H200 GPU; a CUDA GPU is
strongly recommended. One Mouse Brain run (86,822 cells) takes about 16 minutes
there, training and inference included.

`examples/run_squint.py` is the entry point. It was written for the authors'
cluster and sets its paths as module constants, so edit them first:

- `DATA_ROOT` (l. 173): the folder that holds `silver/` and `gold/`;
- `REPRO_REPO` (l. 179): a clone of
  [squint-reproducibility](https://github.com/Lotfollahi-lab/squint-reproducibility).
  Inference (`--predict`, `--all`) imports `collate_predict_outputs` from its
  `analysis/predict_model.py`, so clone both repositories;
- `SQUINT_PKG` (l. 180): this clone;
- `ARTIFACTS_DIR` (l. 184): the root of all run outputs.

Training logs to Weights & Biases (project `squint`) and there is no
command-line switch for it; to log locally only, set `"offline": True` in the
`logging` block of `make_train_config_poc()` (l. 907).

### Prepare a dataset (once)

```bash
python examples/run_squint.py --build-blob --build-blob-dataset mmb0-1b_smb1-1b_1p
```

This builds the dataset blob with spatial k-NN graphs for several k. The
paper's datasets are `mmb0-1b_smb1-1b_1p` (Mouse Brain; read from
`silver/mmb0-1b_smb1-1b_1p_coord_aligned/`), `chl59-2b_1p` (CosMx NSCLC,
Table 1), `chl59-8b_1p` (CosMx NSCLC cohort, section-level query-to-reference
mapping) and `xhs1000-3b_1p` (Xenium Eczema). The `_coord_aligned` suffix is
the name of the prepared Mouse Brain files, whose two sections share one
coordinate frame; SQUINT does not depend on it, because its k-NN graph is built
within each section and never joins two sections. The CosMx and Xenium entries
carry their own data roots, which also need editing: `data_directory_path`
(ll. 314, 362 and 620) and `root_data_dir` (ll. 4359, 4437 and 4645).

### Train and evaluate in one command

```bash
python examples/run_squint.py --all \
    --variant "s57_v19_reference-filmscale+mmb0-1b_smb1-1b_1p"
```

This trains on all sections, predicts every cell, renders spatial code maps,
reconstructions of spatially variable genes and UMAPs, and computes the
metrics (`--metrics-only` skips the plots). Everything for one run lands in

```text
<ARTIFACTS_DIR>/<dataset>/<variant>/<YYYYMMDD_HHMMSS>_seed<N>/
```

### Train and run inference separately

```bash
python examples/run_squint.py --train \
    --variant "s57_v19_reference-filmscale+mmb0-1b_smb1-1b_1p"

RUN=<ARTIFACTS_DIR>/mmb0-1b_smb1-1b_1p/s57_v19_reference-filmscale+mmb0-1b_smb1-1b_1p/<YYYYMMDD_HHMMSS>_seed0
python examples/run_squint.py --predict --run-dir $RUN
python examples/compute_inference_metrics.py --predicted-adata $RUN/predicted_adata.h5ad
```

`--predict` uses the best checkpoint unless `--model-ckpt-fname` is given,
reads the training data unless `--silver-dir` is given and writes into the run
directory unless `--output-dir` is given. `compute_inference_metrics.py` takes
the label columns with `--cell-label-keys` and `--niche-label-keys`; the
defaults cover the paper's datasets.

### Several seeds

`bash examples/submit_multi_seed.sh <variant> 0,1,2,3,4` submits one LSF job
per seed and a dependent aggregator. It defaults to the authors' queue, group,
environment and folders, so pass `--queue` and `--group` and set these
variables first (the seed jobs inherit them):

```bash
export VENV_PATH=/path/to/venv                  # environment with squint installed
export LOG_ROOT=<ARTIFACTS_DIR>/logs            # LSF logs of the sweep
export WANDB_BASE=<ARTIFACTS_DIR>/wandb_cache   # W&B run and cache folders of the seed jobs
```

`submit_multi_seed.sh` creates `LOG_ROOT`, and each seed job
(`_run_one_seed.sh`) creates `WANDB_BASE`; both stop if they cannot, so the
defaults fail outside the authors' cluster. Without LSF:

```bash
V="s57_v19_reference-filmscale+mmb0-1b_smb1-1b_1p"
OUT=<ARTIFACTS_DIR>/mmb0-1b_smb1-1b_1p/${V}__multiseed/$(date +%Y%m%d_%H%M%S)
for s in 0 1 2 3 4; do
  python examples/_run_one_seed.py "$V" $s "$OUT" --skip-umap --skip-code-index-plots --skip-svg-plots
done
python examples/run_squint_multi_seed.py --out-dir "$OUT"
```

`_run_one_seed.py` sets the seed, times training plus inference (the runtime
reported in the paper) and then computes the metrics;
`run_squint_multi_seed.py` collects them into `$OUT/metrics/per_seed_*.csv`,
the format that the benchmark scripts of the reproducibility repository read.

### Variants and changing the configuration

The configuration of every variant is defined in code rather than in separate
YAML files: `examples/run_squint.py` holds a registry, `VARIANTS` (l. 5101),
that maps a variant name to a function returning the full configuration
dictionary (852 variants; `--list-variants` prints them). Each run writes the
complete configuration it used to `user_specified_config.yaml` in its run
directory. The paper's models:

```bash
H="dualvq+rvq-both+decoder-cov+no-batch-int+enc-deeper+dec-w32+knn16+sampler16+cell-w1+bs512+lr7e-4+within-sec+decoupled-enc+diversity-w10+filmscale+crossmnn-wt10-k1"
#   s57_v19_reference-filmscale+mmb0-1b_smb1-1b_1p   Mouse Brain (Table 1; default of Table 3 and Table S1)
#   $H+chl59-2b_1p                                   CosMx NSCLC (Table 1)
#   $H+xhs1000-3b_1p                                 Xenium Eczema (Table 1)
#   $H+region-holdout+mmb0-1b_smb1-1b_1p             Mouse Brain with two held-out regions (Table 2)
#   s57_v1_* ... s57_v33_*                           ablations (Table 3, Table S1)
```

To change a setting, register a new variant; its builder can start from any
existing one, and the `_patch_dual_*` helpers in the same file implement the
paper's ablations. For example (this reproduces the registered
`s65_v3_emadecay-0p9+...` variant):

```python
# in examples/run_squint.py, e.g. directly above `def main()`
VARIANTS["my-variant+mmb0-1b_smb1-1b_1p"] = {
    "description": "Paper default with codebook EMA decay 0.9",
    "patches": [],
    "build": lambda: _patch_dual_ema_decay(
        VARIANTS["s57_v19_reference-filmscale+mmb0-1b_smb1-1b_1p"]["build"](), 0.9),
}
```

`--precision`, `--strategy`, `--compile`/`--no-compile` and `--deterministic`
override the trainer for one run. The paper's variants train in fp32
(`32-true`) with `ddp_find_unused_parameters_true`, no compilation and
deterministic algorithms; the `--help` texts of these flags still describe
older defaults.

### Training on a new dataset

The command line is organised around registered datasets. To add one:

1. put one `.h5ad` per section in `<DATA_ROOT>/silver/<name>/` (see
   [Input data](#input-data));
2. add a blob configuration, e.g. a copy of `make_dataset_blob_config_xhs_3b()`
   (l. 710) with your name and label columns (its helper
   `_make_spatch_blob_config()` sets the data root at l. 620), and register it
   in `build_blob()` (l. 36945) and in the `--build-blob-dataset` choices
   (l. 39209);
3. register a variant that points the paper's configuration at it:

```python
def _build_my_dataset():
    cfg = VARIANTS["dualvq+rvq-both+decoder-cov+no-batch-int+enc-deeper+dec-w32+knn16+sampler16"
                   "+cell-w1+bs512+lr7e-4+within-sec+decoupled-enc+diversity-w10+filmscale"
                   "+crossmnn-wt10-k1+chl59-2b_1p"]["build"]()   # the CosMx NSCLC model
    cfg = _patch_dataset_name(cfg, "my_dataset")              # silver/ and gold/ folder name
    cfg["dataset"]["root_data_dir"] = str(DATA_ROOT)
    return cfg

VARIANTS["squint+my_dataset"] = {
    "description": "SQUINT on my_dataset", "patches": [], "build": _build_my_dataset,
}
```

This trains on all sections and holds out 10% of the cells for early stopping.
Then run `--build-blob --build-blob-dataset my_dataset` and `--all --variant
"squint+my_dataset"`, adding `--cell-label-keys` and `--niche-label-keys` if
your label columns are not among the defaults.

### Outputs

Run directory:

- `user_specified_config.yaml`, `ablation_summary.yaml`: the configuration and
  the variant's description;
- `checkpoints/`: the best (validation loss) and the last checkpoint;
  `wandb/`: local W&B files;
- `predicted_adata.h5ad`: every cell of every section, with the source `obs`
  columns and
  - `obs["adata_batch_id"]` (section), `obs["source_file"]`,
    `obs["data_split"]` (`train`/`test`);
  - `obsm["cell_code_indices"]`, `obsm["neighborhood_code_indices"]`: the codes,
    one (N, 2) integer array per branch (see below);
  - `obsm["cell_emb"]`, `obsm["neighborhood_emb"]`: the quantized embeddings,
    on which the paper's integration metrics are computed;
  - `obsm["cell_latent"]`, `obsm["neighborhood_latent"]`: the latents before
    quantization (MLP and GNN outputs);
  - `layers["X_hat"]`, `layers["X_hat_nbr"]`: reconstructed per-cell and
    neighbourhood-mean counts; `layers["X_nbr"]`: the observed neighbourhood
    mean;
  - `uns["squint"]` (codebook sizes) and `uns["edge_index"]` (spatial graph);
- `metrics/`: `niche_identification_metrics.csv` (NMI and ARI of each code
  level against each label column), `batch_integration_metrics.csv` (iLISI,
  ASW and MMD per embedding), `pearson_reconstruction_metrics.csv`,
  `avg_cosine_similarity.csv` and `analysis_summary.json`;
- `code_index_plots/`, `svg_plots/`, `umap_plots/` (with `--all`).

### Imputation with the stage-2 prior

The masked-code prior (`src/squint/stage2/`, documented in its own `README.md`)
trains on the codes in `predicted_adata.h5ad` of a frozen run and never reads
expression. For the paper it was trained on the region-holdout model, whose
held-out cells carry `obs["data_split"] == "test"`:

```bash
RUN=<ARTIFACTS_DIR>/mmb0-1b_smb1-1b_1p/$H+region-holdout+mmb0-1b_smb1-1b_1p/<YYYYMMDD_HHMMSS>_seed0
S2=$RUN/stage2/imputation
python examples/run_stage2.py --predicted-adata $RUN/predicted_adata.h5ad \
    --holdout-split test --out-dir $S2 \
    --decode-steps 20 --decode-temperature 0.5 --decode-noise-anneal 0.0
python examples/stage2_decode_pearson.py --predicted-adata $RUN/predicted_adata.h5ad \
    --stage2-codes $S2/stage2_predicted_codes.npz --out-metrics-dir $S2/metrics \
    --read-depth-mode neighbor --decode-samples 1000 --niche-from-cell
```

`run_stage2.py` writes `stage2_model.pt`, `stage2_config.json`,
`stage2_eval.csv`/`.json` and the in-painted codes with their posteriors
(`stage2_predicted_codes.npz`). `stage2_decode_pearson.py` decodes them through
the frozen stage-1 decoders and scores the held-out cells
(`metrics/per_seed_pearson_reconstruction.csv`). `--decode-samples 1000`
averages 1,000 code configurations drawn from the posteriors ("SQUINT (MC)" in
Table 2; omit it for the single decode, "SQUINT (No MC)"), and
`--niche-from-cell` builds the neighbourhood-level prediction by averaging the
cell-level one over the spatial graph, as for the baselines.

**Read depth.** `--read-depth-mode neighbor` scales each held-out cell's
prediction by the mean library size of its 16 nearest observed cells, the
leak-free rule used for all imputation results in the paper. The default,
`true`, uses the held-out cell's own library size and so reads part of the
target; it is only meant for reconstruction checks.

Table 1 to 3 values, their statistics and the figure panels are computed in
the reproducibility repository.

## Key model options

Keys of the configuration dictionary built by `examples/run_squint.py`, with
the values of the paper's models:

| Key | Paper value | Meaning |
|---|---|---|
| `model.encoder_params.vq_{cell,niche}_params.num_quantizers` | `2` | Residual levels per codebook |
| `model.encoder_params.vq_{cell,niche}_params.codebook_size` | `[30, 90]` | Codes per level, (K(1), K(2)) |
| `model.encoder_params.vq_{cell,niche}_params.decay` | `0.8` | EMA decay of the codebook updates |
| `model.encoder_params.vq_{cell,niche}_params.use_cosine_sim`, `kmeans_init`, `threshold_ema_dead_code` | `true`, `true`, `2` | Cosine lookup; k-means initialisation and dead-code resampling (level 1 only) |
| `model.encoder_params.vq_cell_params.codebook_diversity_loss_weight` | `10` (`0` for the niche codebook) | Codebook-diversity regulariser |
| `model.encoder_params.cell_conditioned_niche.mode` | `film_scale` | Cell-to-niche coupling: the level-1 cell code scales the niche embedding |
| `model.encoder_params.mlp_params.hidden_channels` | `[400, 400, 256]` | Cell MLP; the GNN (`gnn_name`, `gnn_params`) is one `SAGEConv` layer of width 256 |
| `dataset.graph_params.n_neighs` | `16` | k of the spatial k-NN graph |
| `model.loss_params.loss_kwargs.wt_adj_reconstr` | `1000` | Weight of the within-section cosine adjacency loss |
| `model.loss_params.loss_kwargs.edge_sampling_ratio`, `use_pos_weight` | `2.0` (Mouse Brain), `1.0` (CosMx, Xenium); `true` | Non-edges sampled per spatial edge in that loss; edges are weighted by the number of non-edges per edge, so edges and non-edges contribute equally |
| `model.loss_params.loss_kwargs.wt_contrastive_cell` / `wt_cross` | `10` / `10` | Within-section NT-Xent (`k_pos` 5) and cross-section MNN attraction (`k_cross` 1) |
| `model.decoder_covariate_dim_request` | `true` | Section covariate in both decoders |
| `model.attribute_decoder_{cell,niche}_params.mlp_params.hidden_channels` | `[32]` | Decoder hidden layer |
| `model.optimizer_params.lr` / `weight_decay` | `7e-4` / `1e-3` | Adam |
| `datamodule.loader_params.batch_size` | `512` | Seed cells per step |
| `trainer.max_epochs` | `80` (`400` on Xenium Eczema) | With early stopping on the validation loss |

### Codes, levels and iLISI

Column 0 of `obsm["cell_code_indices"]` and `obsm["neighborhood_code_indices"]`
is the coarse level (the paper's Level 1, K(1) = 30) and column 1 the residual
level (Level 2, K(2) = 90). The paper uses column 0 directly as the cluster
assignment for NMI and ARI; `compute_inference_metrics.py` reports it as
`cell_code_indices[level_0]` and `neighborhood_code_indices[level_0]`, next to
`[composite]`, the pair of both levels.

The quantized embeddings contain large groups of identical rows (all cells
that share a code). Since commit
[`89d9c82`](https://github.com/Lotfollahi-lab/squint/commit/89d9c82c649ab9550135d392d765c01bbaa470b0),
`compute_ilisi` in `compute_inference_metrics.py` permutes the cells before the
neighbour search, so ties are broken at random rather than by storage order
(section by section), which understated SQUINT's iLISI in the reviewed version
of the paper. It also requires `scib-metrics` and skips iLISI when it cannot
be imported, instead of falling back to a different, unscaled statistic.

## Reproducing the paper

The tag **`mlcb2026-camera-ready`** marks the state of this repository and of
the reproducibility repository for the paper. At the tag, the stage-1 model
code (`src/squint/` outside `stage2/`) is unchanged since commit
[`4d561b0`](https://github.com/Lotfollahi-lab/squint/tree/4d561b0d8cb53e62ab9647fb6501a4fe61ae59c2)
(28 June 2026) and `src/squint/stage2/` since
[`9802adf`](https://github.com/Lotfollahi-lab/squint/tree/9802adfe684605e852480f5eee0d11954e569d34)
(30 June 2026), apart from the package rename. The iLISI values of the
reported runs were recomputed with the corrected estimator (see
[Codes, levels and iLISI](#codes-levels-and-ilisi)).

Data: the sources and the expected silver folders are listed in the
reproducibility repository (Section 2). The Table 1 models, five seeds each:

```bash
for d in mmb0-1b_smb1-1b_1p chl59-2b_1p xhs1000-3b_1p; do
  python examples/run_squint.py --build-blob --build-blob-dataset $d
done

# one LSF job per seed plus an aggregator; set VENV_PATH, LOG_ROOT and WANDB_BASE
# first, and see "Several seeds" to run without LSF
bash examples/submit_multi_seed.sh "s57_v19_reference-filmscale+mmb0-1b_smb1-1b_1p" 0,1,2,3,4   # Mouse Brain
bash examples/submit_multi_seed.sh "$H+chl59-2b_1p" 0,1,2,3,4                                    # CosMx NSCLC
bash examples/submit_multi_seed.sh "$H+xhs1000-3b_1p" 0,1,2,3,4                                  # Xenium Eczema
```

The three share one configuration ([Key model options](#key-model-options));
the human-data variants change only the dataset, the non-edge sampling ratio
and, on Xenium Eczema, the epoch limit. Paper values for SQUINT (mean over
seeds 0–4; MMD in 10⁻³, runtime RT in minutes):

| Dataset | Branch | NMI | ARI | MMD | iLISI | RT |
|---|---|---|---|---|---|---|
| Mouse Brain | Niche | 0.702 | 0.410 | 1.05 | 0.897 | 16 |
| | Cell | 0.494 | 0.204 | 2.27 | 0.810 | 16 |
| CosMx NSCLC | Niche | 0.397 | 0.172 | 0.52 | 0.948 | 33 |
| | Cell | 0.604 | 0.339 | 0.61 | 0.967 | 33 |
| Xenium Eczema | Niche | 0.514 | 0.207 | 3.76 | 0.542 | 25 |
| | Cell | 0.593 | 0.322 | 8.93 | 0.702 | 25 |

The [reproducibility repository](https://github.com/Lotfollahi-lab/squint-reproducibility)
has the baselines, the region-holdout and stage-2 runs (Table 2), the ablations
(Table 3, Table S1), the codebook analyses (Tables S2 and S3, Fig. S4), the
section-level query-to-reference analysis (Fig. 3), the corrected-iLISI pipeline, the
statistics and the figure panels. The LSF launchers of both repositories
default to the authors' cluster paths and queues.

Trained checkpoints are not distributed, and the Xenium human-skin data are not
publicly available.

## Tests

```bash
uv sync --extra test         # or: pip install -e ".[test]"
pytest tests/ src/squint/stage2/tests/
```

A bare `pytest` runs only `tests/` (`testpaths` in `pyproject.toml`); the
stage-2 tests live next to the module.

## Repository layout

| Path | Contents |
|---|---|
| `src/squint/models/squint_dual.py`, `src/squint/encoders/squint_dual_encoder.py` | The dual-codebook model (`Squint_Dual`, `Squint_Dual_Encoder`) |
| `src/squint/modules/` | Residual quantiser (`hierarchical_vq.py`), FiLM and the other coupling modules, GNN and MLP blocks |
| `src/squint/loss/` | Negative-binomial reconstruction, cosine adjacency, contrastive, commitment and codebook losses |
| `src/squint/dataset/`, `src/squint/dataloaders/`, `src/squint/initializers/` | Dataset blob, `NeighborLoader` data module, builders for data, model and trainer |
| `src/squint/stage2/` | Masked-code prior (stage 2), with its own README and tests |
| `src/squint/models/squint.py`, `src/squint/encoders/squint_encoder.py`, `vanilla_*` | Single-codebook model and simple baselines from earlier development |
| `src/squint/vqgraph/` | Adapted VQGraph code; only `vq.py` is used (the rest needs DGL and OGB, which `pyproject.toml` does not install) |
| `examples/` | Command-line entry points, LSF launchers, metrics and plotting scripts |
| `tests/` | Unit tests |
| `docs/`, `DEVELOPER_GUIDE.md`, `environment.yaml` | Project-template documentation scaffold, early-development notes and environment; not maintained |

## Citation

If you use SQUINT, please cite:

```bibtex
@inproceedings{birk2026squint,
  title     = {Learning Discrete Cell and Niche Codes from Spatial Transcriptomics Using Dual Residual Vector Quantization},
  author    = {Birk, Sebastian and Merchant, Arpit and Vahidi, Amirhossein and Theis, Fabian J. and Lotfollahi, Mo},
  booktitle = {Proceedings of the Machine Learning in Computational Biology (MLCB) 2026},
  series    = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
  year      = {2026},
  note      = {Volume and pages to be assigned}
}
```

The single-codebook precursor of SQUINT was presented at the ICLR 2026 MLGenX
workshop:

```bibtex
@inproceedings{merchant2026squint,
  title     = {{SQUINT}: Spatial Quantization for Understanding and In-painting Tissues},
  author    = {Merchant, Arpit and Birk, Sebastian and Vahidi, Amirhossein and Jafree, Daniyal and Steele, Lloyd and Foster, April Rose and Vijayabaskar, M. S. and Haniffa, Muzlifah and Lotfollahi, Mohammad},
  booktitle = {ICLR 2026 Workshop on Machine Learning for Genomics Explorations (MLGenX)},
  year      = {2026},
  url       = {https://openreview.net/forum?id=Cpg9yYyEPi}
}
```

## Contact

For questions and bug reports, please use the
[issue tracker](https://github.com/Lotfollahi-lab/squint/issues).

## Licence

MIT; see [LICENSE](LICENSE).
