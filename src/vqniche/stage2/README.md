# `vqniche.stage2` — Spatial code prior (stage 2)

A **graph-native, MaskGIT-style masked-code transformer** that predicts held-out
cells' discrete code stacks from the spatial context of observed cells. It is a
**generative spatial prior over discrete tissue tokens**, and it is a *separate
module*: it imports nothing from SQUINT's training path and never touches the
stage-1 code. It consumes only what SQUINT's predict pipeline exports.

```
 Stage 1: SQUINT VQ-VAE  (FROZEN)   raw counts ─▶ dual residual code stacks per cell
 Stage 2: this module               observed codes + positions ─▶ held-out codes
 (optional) decode predicted codes back to expression with SQUINT's frozen NB decoder
```

The defining property: stage 2 predicts **codes from spatial context**, it never
encodes held-out expression. Because the vocabulary is SQUINT's frozen codebook,
every prediction is in-distribution by construction.

## Why this design

The research converged on a **graph-native Contextual RQ-Transformer**, a synthesis of:

- **MaskGIT** (Chang et al., CVPR 2022) — bidirectional masked-token transformer
  with iterative confidence-scheduled parallel decoding. In-painting is native:
  observed neighbours are unmasked tokens, held-out cells are `[MASK]`.
- **RQ-Transformer / Contextual RQ-Transformer** (Lee et al., 2022) — the
  published recipe for *residual* codes: a frozen RQ-VAE tokenizer (= SQUINT)
  plus a stage-2 transformer that predicts the code stack coarse-to-fine, with a
  masked-infilling ("Draft-and-Revise") decoder.

Three adaptations make it fit SQUINT's irregular single-cell spatial graph:

1. **Graph attention** instead of grid attention — full self-attention within a
   patch + an additive, learnable per-head spatial bias `−γ_h · ‖xᵢ − xⱼ‖`
   (GPS/SpaGT "structure-reinforced attention"). Heads choose their own locality.
2. **2D Fourier positional features** (Random Fourier Features over continuous
   coordinates) instead of learned grid position embeddings.
3. **Spatially-contiguous block masking** instead of random scatter, matching the
   region-holdout (hole-in-the-tissue) geometry the model is evaluated on.

SQUINT's **dual residual stacks** (cell-type + niche, each `[30, 90]`) are
predicted as one ordered stack per cell — `(cell L0, cell L1, niche L0, niche L1)` —
with two kinds of conditioning baked into the heads:

- **residual depth**: level `l` conditions on the coarser levels `< l` (RQ axis);
- **hierarchy**: the niche levels condition on the predicted cell stack, mirroring
  SQUINT's existing FiLM cell→niche coupling (`hierarchical=True`, default).

## Module layout

| file | role | needs torch |
|------|------|:-----------:|
| `config.py`     | dataclass configs (`Stage2Config`, …); prediction-target order | no |
| `masking.py`    | contiguous-block & random masking | no |
| `data.py`       | `AnnDataCodeSource`, `PatchSampler`, `inpainting_patch`, kNN, coord-norm | no |
| `positional.py` | `FourierPositionalEncoding` | yes |
| `transformer.py`| distance-biased self-attention + transformer block | yes |
| `model.py`      | `SpatialCodeTransformer` (`encode` / `head_logits` / `forward` / `loss`) | yes |
| `decode.py`     | `decode_patch`, `inpaint` (MaskGIT iterative decoding) | yes |
| `datamodule.py` | torch `Dataset`, `collate_patches`, `Stage2DataModule` | yes |
| `lightning.py`  | `Stage2LightningModule` (training + decoded-accuracy validation) | yes |
| `tests/`        | `test_numpy_core.py` (no torch) + `test_torch_model.py` (torch-gated) | — |

The numpy core is intentionally torch-free and unit-tested without the ML stack.
The package `__init__` lazy-loads the torch symbols, so
`from vqniche.stage2 import AnnDataCodeSource` works with numpy alone.

## Frozen stage-1 interface (read from `predicted_adata.h5ad`)

Matches SQUINT's predict pipeline / `report_codebook_usage.py`:

| quantity | location | shape |
|----------|----------|-------|
| cell codes  | `uns['Indices_cell']`  (→ `obsm['cell_code_indices']` → `obs['cell_code_index']`)   | `(N, Lc)` |
| niche codes | `uns['Indices_niche']` (→ `obsm['neighborhood_code_indices']` → …) | `(N, Ln)` |
| codebook sizes | `uns['codebook_sizes_cell' / 'codebook_sizes_niche']` | lists |
| positions | `obsm['spatial']` | `(N, 2)` |
| section / batch | `obs['adata_batch_id']` | `(N,)` |

Expression is never read.

## Usage

```python
from vqniche.stage2 import Stage2Config, AnnDataCodeSource, Stage2DataModule
from vqniche.stage2 import Stage2LightningModule, inpaint
import pytorch_lightning as pl

src = AnnDataCodeSource("…/predicted_adata.h5ad")          # frozen SQUINT codes
cfg = Stage2Config.squint_default(                          # RVQ(30,90) dual
    codebook_sizes_cell=src.branch_sizes()["cell"],
    codebook_sizes_niche=src.branch_sizes()["niche"],
)

dm = Stage2DataModule(src, cfg, num_workers=4)
lit = Stage2LightningModule(cfg)
pl.Trainer(max_steps=cfg.optim.max_steps).fit(
    lit, dm.train_dataloader(), dm.val_dataloader())

# in-paint a held-out region (list of global cell indices in one section)
res = inpaint(lit.model, src, holdout_idx, cfg)
res["codes"]["cell"]    # (n_holdout, Lc) predicted cell-type code stacks
res["codes"]["niche"]   # (n_holdout, Ln) predicted niche code stacks
```

To turn predicted codes back into expression, look the codes up in SQUINT's
frozen codebook embeddings (`encoder.vq_*.layers[l]._codebook.embed`), sum the
residual levels, and run SQUINT's frozen NB decoder. That step is *optional* and
lives outside this module (it needs the trained stage-1 model, not just the
exported codes).

## Running it (farm)

Train + region-holdout evaluate from a frozen `predicted_adata.h5ad` via the
entry-point `squint/examples/run_stage2.py`, submitted to LSF with
`squint/examples/submit_stage2.sh` (one exclusive GPU, `gpu-lotfollahi`,
group `team361`, venv `/nfs/team361/sb75/.venvs/squint` — same conventions as
`submit_multi_seed.sh`):

```bash
# quick end-to-end sanity run (tiny model, ~80 steps)
bash examples/submit_stage2.sh /path/to/predicted_adata.h5ad -- --smoke

# real run (everything after `--` is forwarded to run_stage2.py)
bash examples/submit_stage2.sh /path/to/predicted_adata.h5ad -- \
    --max-steps 20000 --d-model 256 --n-layers 6 --patch-size 1024 \
    --batch-size 8 --lr 3e-4

# render the bsub without submitting
DRY_RUN=1 bash examples/submit_stage2.sh /path/to/predicted_adata.h5ad -- --smoke
```

Outputs land in `<run_dir>/stage2/<timestamp>/`:
`stage2_model.pt` + `stage2_config.json` (reload with
`vqniche.stage2.load_stage2(out_dir)`), `stage2_eval.csv` / `.json`
(decoded top-1 code accuracy per branch/level vs. a majority-code baseline),
`last.ckpt`, and `logs/` (CSV metrics incl. teacher-forced **and** decoded
validation accuracy).

## Evaluation (recommended)

No ST benchmark exists for discrete-code generative imputation. Report:

1. **code accuracy** — top-1 / NLL per level per branch on the held-out patch
   (the `val/acc_dec_*` decoded metric is the faithful one; `val/acc_tf_*` is
   teacher-forced);
2. **decoded expression quality** — Pearson of decoded NB expression on held-out
   cells (same as the region-holdout task);
3. **biological validity** — cell-type composition & niche-label recovery on the
   in-painted region;
4. **distributional realism** — MMD between predicted and true code distributions;
5. **calibration** — decoder confidence vs. correctness.

## Status & validation

- numpy core: unit-tested locally (`test_numpy_core.py`) — contiguous-mask
  purity, single-section connected patches, coord normalisation, kNN, in-paint
  patch assembly.
- torch model: validated on CPU (`test_torch_model.py`) — forward/backward,
  gradient flow, a 60-step overfit drives loss 15.9→3.2 and cell-L0 accuracy to
  ~0.83, the iterative decoder reproduces that accuracy without teacher forcing
  and leaves observed cells unchanged, and `inpaint()` runs end-to-end.
- entry-point: `run_stage2.py` validated end-to-end on a synthetic
  `predicted_adata.h5ad` (train → checkpoint → region-holdout eval); a short real
  training run drives decoded in-painting accuracy from ~chance to **~0.75** on
  spatially-structured toy codes (majority baseline 0.07), confirming the
  train→decode→eval loop learns. Checkpoints round-trip via
  `save_stage2`/`load_stage2`.
- **Not yet done** (future work): wire the optional frozen-NB-decoder expression
  read-out; outpainting / unconditional generation; the full evaluation harness
  in (1)–(5) (decoded Pearson, cell-type/niche-label recovery, code-dist MMD,
  calibration) — `region_holdout_eval` currently covers code top-1 accuracy.
