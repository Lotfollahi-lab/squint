"""
Torch integration test for the stage-2 model (skipped if torch is absent).

Exercises forward / loss / backward, a tiny overfit, the iterative decoder
(no teacher forcing), end-to-end ``inpaint`` from a frozen source, and the
non-hierarchical / untied variant. Mirrors what runs on the farm.

Run:  python -m vqniche.stage2.tests.test_torch_model
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    _HAVE_TORCH = True
except Exception:                       # pragma: no cover
    _HAVE_TORCH = False


def _toy_adata(n_per_section=300, n_sections=2, seed=0):
    import anndata
    import pandas as pd

    rng = np.random.default_rng(seed)
    coords, secs = [], []
    side = int(np.sqrt(n_per_section))
    for s in range(n_sections):
        gx, gy = np.meshgrid(np.arange(side), np.arange(side))
        c = np.stack([gx.ravel(), gy.ravel()], 1).astype(float)
        c += rng.normal(scale=0.05, size=c.shape)
        c[:, 0] += s * (side + 10)
        coords.append(c)
        secs.append(np.full(c.shape[0], f"sec{s}"))
    coords = np.concatenate(coords)
    secs = np.concatenate(secs)
    n = coords.shape[0]
    sc, sn = [30, 90], [30, 90]
    band = (coords[:, 1] // 3).astype(int)           # spatially structured codes
    idx_cell = np.stack([(band % k) for k in sc], 1).astype(np.int64)
    idx_niche = np.stack([(band % k) for k in sn], 1).astype(np.int64)
    ad = anndata.AnnData(X=np.zeros((n, 5), np.float32))
    ad.obsm["spatial"] = coords
    ad.obs["adata_batch_id"] = pd.Categorical(secs)
    ad.uns["Indices_cell"] = idx_cell
    ad.uns["Indices_niche"] = idx_niche
    ad.uns["codebook_sizes_cell"] = sc
    ad.uns["codebook_sizes_niche"] = sn
    return ad


def test_torch_end_to_end():
    if not _HAVE_TORCH:
        print("SKIP test_torch_end_to_end (torch not installed)")
        return

    from vqniche.stage2.config import (
        Stage2Config, DataConfig, ModelConfig, DecodeConfig, OptimConfig,
    )
    from vqniche.stage2.data import AnnDataCodeSource
    from vqniche.stage2.datamodule import SpatialCodeDataset, collate_patches
    from vqniche.stage2.model import SpatialCodeTransformer
    from vqniche.stage2.decode import decode_patch, inpaint
    from vqniche.stage2.masking import block_mask

    torch.manual_seed(0)
    src = AnnDataCodeSource(_toy_adata())
    cfg = Stage2Config.squint_default(
        [30, 90], [30, 90],
        data=DataConfig(patch_size=96, knn=8, mask_kind="block",
                        mask_frac_min=0.2, mask_frac_max=0.4),
        model=ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128,
                          pos_num_freqs=16),
        decode=DecodeConfig(steps=6, temperature=0.0),
        optim=OptimConfig(batch_size=4),
    )
    ds = SpatialCodeDataset(src, cfg, length=8, deterministic=True, seed=0)
    batch = collate_patches([ds[i] for i in range(cfg.optim.batch_size)])
    assert batch["codes"].shape[2] == 4

    model = SpatialCodeTransformer(cfg)
    out = model.loss(batch["codes"], batch["coords"], batch["mask"],
                     batch["key_padding_mask"])
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    gnorm = sum(p.grad.norm().item() ** 2
                for p in model.parameters() if p.grad is not None) ** 0.5
    assert gnorm > 0

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    first = last = None
    for step in range(60):
        opt.zero_grad()
        o = model.loss(batch["codes"], batch["coords"], batch["mask"],
                       batch["key_padding_mask"])
        o["loss"].backward()
        opt.step()
        if first is None:
            first = float(o["loss"])
        last = float(o["loss"])
    assert last < first, (first, last)

    filled = decode_patch(model, batch["codes"], batch["coords"], batch["mask"],
                          batch["key_padding_mask"], cfg.decode)
    assert filled.shape == batch["codes"].shape
    obs = (~batch["mask"]) & (~batch["key_padding_mask"])
    assert (filled[obs] == batch["codes"][obs]).all()        # observed unchanged
    Ks = [model.target_K[f"{b}__{l}"] for (b, l) in model.targets]
    for ti, K in enumerate(Ks):
        assert int(filled[..., ti].max()) < K                # within vocabulary

    rows0 = src.section_of(0)
    hold = rows0[block_mask(src.coords[rows0], 0.1, np.random.default_rng(1))]
    res = inpaint(model, src, hold, cfg)
    assert res["codes"]["cell"].shape == (len(hold), 2)
    assert res["codes"]["niche"].shape == (len(hold), 2)
    assert set(res["global_idx"].tolist()) == set(hold.tolist())
    print(f"ok  test_torch_end_to_end (loss {first:.2f}->{last:.2f}, "
          f"params {sum(p.numel() for p in model.parameters())/1e3:.0f}k)")


def test_non_hierarchical_untied_builds():
    if not _HAVE_TORCH:
        print("SKIP test_non_hierarchical_untied_builds (torch not installed)")
        return
    from vqniche.stage2.config import Stage2Config, ModelConfig
    from vqniche.stage2.model import SpatialCodeTransformer

    cfg = Stage2Config.squint_default(
        [30, 90], [30, 90],
        model=ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128,
                          pos_num_freqs=16, hierarchical=False,
                          tie_code_embeddings=False),
    )
    m = SpatialCodeTransformer(cfg)
    assert m.cond_embed is not m.code_embed
    print("ok  test_non_hierarchical_untied_builds")


def main():
    test_torch_end_to_end()
    test_non_hierarchical_untied_builds()
    print("\nTORCH TESTS DONE")


if __name__ == "__main__":
    main()
