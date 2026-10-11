# `squint.stage2`: masked-code prior (stage 2)

A **graph-native, MaskGIT-style masked-code transformer** that predicts the
discrete code stacks of held-out cells from the codes and positions of the
observed cells around them. It is a separate module: it imports nothing from
SQUINT's training path and reads only what SQUINT's inference writes to
`predicted_adata.h5ad`. It never reads expression.

```text
Stage 1: SQUINT (frozen)   counts -> dual residual code stacks per cell
Stage 2: this module       observed codes + positions -> codes of held-out cells
Read-out                   predicted codes -> expression through the frozen stage-1
                           decoders (examples/stage2_decode_pearson.py)
```

Stage 2 predicts **codes from spatial context**; it never encodes the
expression of held-out cells. Its vocabulary is SQUINT's frozen codebook, so
every prediction is a code that the stage-1 decoders can read out.

## Design

The model combines two published recipes:

- **MaskGIT** (Chang et al., CVPR 2022): a bidirectional masked-token
  transformer with iterative, confidence-scheduled parallel decoding.
  In-painting is native: observed neighbours are unmasked tokens and held-out
  cells are `[MASK]`.
- **RQ-Transformer and Contextual RQ-Transformer** (Lee et al., 2022): a frozen
  residual-quantization tokeniser (here SQUINT) with a second-stage transformer
  that predicts the code stack coarse to fine, decoded by masked in-filling.

Three adaptations fit it to the irregular spatial graph of single cells:

1. **Distance-biased attention** instead of grid attention: full
   self-attention within a patch plus a learnable, positive per-head bias
   `-γ_h · ‖x_i - x_j‖`, so each head chooses its own locality.
2. **2-D Fourier positional features** (random Fourier features of the
   continuous coordinates) instead of learned grid position embeddings.
3. **Spatially contiguous block masking** during training instead of random
   scatter, matching the geometry of the held-out regions.

SQUINT's two residual code stacks (cell and niche, each `[30, 90]`) are
predicted as one ordered stack per cell, `(cell level 1, cell level 2, niche
level 1, niche level 2)` (columns 0 and 1 of each branch's code array), with two
kinds of conditioning in the output heads:

- **residual depth**: each level conditions on the coarser levels of its branch;
- **hierarchy**: the niche levels condition on the predicted cell stack,
  mirroring SQUINT's cell-to-niche FiLM coupling (`hierarchical=True` by
  default; `--no-hierarchical` switches it off).

## Module layout

| File | Role | Needs torch |
|---|---|:-:|
| `config.py` | Dataclass configurations (`Stage2Config`, ...); prediction-target order | no |
| `masking.py` | Contiguous-block and random masking | no |
| `data.py` | `AnnDataCodeSource`, `PatchSampler`, `inpainting_patch`, k-NN, coordinate normalisation | no |
| `positional.py` | `FourierPositionalEncoding` | yes |
| `transformer.py` | Distance-biased self-attention and transformer block | yes |
| `backbones/` | The transformer backbone (default) and alternative backbones selected with `--arch` (`gnn`, `labelprop`, `graphmae`, `gps`, `diffusion`); only the transformer is reported in the paper | yes |
| `model.py` | `SpatialCodeTransformer` (`encode`, `head_logits`, `forward`, `loss`) | yes |
| `decode.py` | `decode_patch`, `inpaint` (MaskGIT iterative decoding) | yes |
| `evaluate.py` | `holdout_split_eval` (the cells held out of stage 1) and `region_holdout_eval` (synthetic holes) | yes |
| `checkpoint.py` | `save_stage2`, `load_stage2` | yes |
| `datamodule.py` | torch `Dataset`, `collate_patches`, `Stage2DataModule` | yes |
| `lightning.py` | `Stage2LightningModule` (training and decoded-accuracy validation) | yes |
| `tests/` | `test_numpy_core.py` (no torch) and `test_torch_model.py` (skipped without torch) | |

The numpy core does not import torch and is unit-tested without it. The package
`__init__` loads the torch symbols lazily, so
`from squint.stage2 import AnnDataCodeSource` works with numpy alone.

## Frozen stage-1 interface (read from `predicted_adata.h5ad`)

| Quantity | Location | Shape |
|---|---|---|
| Cell codes | `obsm['cell_code_indices']` (or `uns['Indices_cell']` if present) | `(N, 2)` |
| Niche codes | `obsm['neighborhood_code_indices']` (or `uns['Indices_niche']` if present) | `(N, 2)` |
| Codebook sizes | `uns['codebook_sizes_cell']`, `uns['codebook_sizes_niche']` if present; otherwise the largest index + 1 per level | lists |
| Positions | `obsm['spatial']` | `(N, 2)` |
| Section | `obs['adata_batch_id']` | `(N,)` |
| Held-out cells (optional) | `obs['data_split'] == 'test'` (`--holdout-split test`) | `(N,)` |

Expression is never read.

## Running it

`examples/run_stage2.py` trains the prior on a frozen `predicted_adata.h5ad`
and evaluates it. With `--holdout-split test`, the cells that stage 1 held out
(`obs['data_split'] == 'test'`) are excluded from the training patches and
in-painted from the observed cells around them; this is the protocol of
Table 2. The paper's settings, on the region-holdout model:

```bash
RUN=<run directory of the region-holdout model>
python examples/run_stage2.py --predicted-adata $RUN/predicted_adata.h5ad \
    --holdout-split test --out-dir $RUN/stage2/imputation \
    --decode-steps 20 --decode-temperature 0.5 --decode-noise-anneal 0.0
```

Without `--out-dir`, outputs go to `<run directory>/stage2/<timestamp>/`:
`stage2_model.pt` and `stage2_config.json` (the model at the end of training;
reload with `squint.stage2.load_stage2(out_dir)`), `stage2_eval.csv` and
`stage2_eval.json` (decoded top-1 code accuracy per branch and level, against a
majority-code baseline), `stage2_predicted_codes.npz` (with `--holdout-split`:
the in-painted codes and their per-level posteriors), Lightning checkpoints and
`logs/` (CSV metrics, including teacher-forced and decoded validation
accuracy). `--smoke` runs a tiny model for a quick end-to-end check.

On an LSF cluster, `examples/submit_stage2.sh` submits one GPU job; everything
after `--` is passed to `run_stage2.py`, and `DRY_RUN=1` prints the `bsub`
command:

```bash
bash examples/submit_stage2.sh $RUN/predicted_adata.h5ad -- --holdout-split test \
    --decode-steps 20 --decode-temperature 0.5 --decode-noise-anneal 0.0
```

Like the other launchers, it defaults to the authors' queue, group and
environment: set `LSF_QUEUE`, `LSF_GROUP`, `VENV_PATH` and `LOG_ROOT`. The
paper's five-seed runs used `examples/submit_stage2_ablation.sh` (arm
`decode-temp05-nonoise`); see Section 3.2 of the
[reproducibility repository](https://github.com/Lotfollahi-lab/squint-reproducibility).

## From codes to expression

`examples/stage2_decode_pearson.py` reads `stage2_predicted_codes.npz`, looks
the codes up in the frozen stage-1 codebooks, sums the residual levels,
re-applies the cell-to-niche coupling and decodes them with the frozen
negative-binomial decoders. It first checks that decoding a cell's own codes
reproduces the stored reconstruction, then scores the held-out cells. The
settings of Table 2:

```bash
python examples/stage2_decode_pearson.py --predicted-adata $RUN/predicted_adata.h5ad \
    --stage2-codes $RUN/stage2/imputation/stage2_predicted_codes.npz \
    --out-metrics-dir $RUN/stage2/imputation/metrics \
    --read-depth-mode neighbor --decode-samples 1000 --niche-from-cell
```

- `--read-depth-mode neighbor` scales each held-out cell's prediction by the
  mean library size of its 16 nearest observed cells. The default, `true`, uses
  the held-out cell's own library size and so reads part of the target; use it
  only for reconstruction checks.
- `--decode-samples 1000` averages 1,000 code configurations drawn from the
  posteriors ("SQUINT (MC)"); the default, 1, decodes the in-painted codes once
  ("SQUINT (No MC)").
- `--niche-from-cell` builds the neighbourhood-level prediction by averaging
  the cell-level prediction over the spatial graph, as for the baselines.

## Python API

```python
from squint.stage2 import Stage2Config, AnnDataCodeSource, Stage2DataModule
from squint.stage2 import Stage2LightningModule, inpaint
import pytorch_lightning as pl

src = AnnDataCodeSource("<run>/predicted_adata.h5ad")      # frozen SQUINT codes
cfg = Stage2Config.squint_default(                          # RVQ(30, 90), both branches
    codebook_sizes_cell=src.branch_sizes()["cell"],
    codebook_sizes_niche=src.branch_sizes()["niche"],
)

dm = Stage2DataModule(src, cfg, num_workers=4)
lit = Stage2LightningModule(cfg)
pl.Trainer(max_steps=cfg.optim.max_steps).fit(
    lit, dm.train_dataloader(), dm.val_dataloader())

# in-paint a held-out region (global cell indices within one section)
res = inpaint(lit.model, src, holdout_idx, cfg)
res["codes"]["cell"]    # (n_holdout, 2) predicted cell code stacks
res["codes"]["niche"]   # (n_holdout, 2) predicted niche code stacks
```

`run_stage2.py` shows how to set up the held-out split (`holdout_key`,
`holdout_value` of `AnnDataCodeSource`) and the evaluation.

## Tests

```bash
pytest src/squint/stage2/tests/
```

`test_numpy_core.py` covers the contiguous masks, single-section connected
patches, coordinate normalisation, the k-NN and the in-painting patch
assembly. `test_torch_model.py` covers the forward and backward pass, gradient
flow, a short overfit, iterative decoding without teacher forcing (observed
cells unchanged) and `inpaint()` end to end.
