"""
Numpy-only tests for the stage-2 data/masking/config core.

Runnable without torch:  python -m vqniche.stage2.tests.test_numpy_core
(or via pytest). Validates patch geometry, contiguous masking, coordinate
normalisation, kNN, and the frozen-adata reader against a synthetic AnnData.
"""

from __future__ import annotations

import numpy as np


def _toy_adata(n_per_section=400, n_sections=2, seed=0):
    """Synthetic predicted_adata: 2 sections on a grid, RVQ(30,90) dual codes."""
    import anndata
    import pandas as pd

    rng = np.random.default_rng(seed)
    coords, secs = [], []
    side = int(np.sqrt(n_per_section))
    for s in range(n_sections):
        gx, gy = np.meshgrid(np.arange(side), np.arange(side))
        c = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(float)
        c = c + rng.normal(scale=0.05, size=c.shape)        # jitter
        c[:, 0] += s * (side + 10)                          # offset sections
        coords.append(c)
        secs.append(np.full(c.shape[0], f"sec{s}"))
    coords = np.concatenate(coords, 0)
    secs = np.concatenate(secs, 0)
    n = coords.shape[0]

    sizes_cell, sizes_niche = [30, 90], [30, 90]
    idx_cell = np.stack([rng.integers(k, size=n) for k in sizes_cell], 1)
    idx_niche = np.stack([rng.integers(k, size=n) for k in sizes_niche], 1)

    ad = anndata.AnnData(X=np.zeros((n, 5), dtype=np.float32))
    ad.obsm["spatial"] = coords
    ad.obs["adata_batch_id"] = pd.Categorical(secs)
    ad.uns["Indices_cell"] = idx_cell
    ad.uns["Indices_niche"] = idx_niche
    ad.uns["codebook_sizes_cell"] = sizes_cell
    ad.uns["codebook_sizes_niche"] = sizes_niche
    return ad, sizes_cell, sizes_niche


def test_source_reads_codes():
    from vqniche.stage2.data import AnnDataCodeSource

    ad, sc, sn = _toy_adata()
    src = AnnDataCodeSource(ad)
    assert src.n_cells == ad.n_obs
    assert src.branch_sizes() == {"cell": sc, "niche": sn}
    assert src.codes["cell"].shape == (ad.n_obs, 2)
    assert src.codes["niche"].shape == (ad.n_obs, 2)
    assert src.section_ids.size == 2
    # codes within vocabulary
    for b, sizes in (("cell", sc), ("niche", sn)):
        for q, k in enumerate(sizes):
            assert src.codes[b][:, q].max() < k
    print("ok  test_source_reads_codes")


def test_patch_is_connected_and_single_section():
    from vqniche.stage2.config import DataConfig
    from vqniche.stage2.data import AnnDataCodeSource, PatchSampler

    ad, _, _ = _toy_adata()
    src = AnnDataCodeSource(ad)
    cfg = DataConfig(patch_size=128, mask_kind="block")
    sampler = PatchSampler(src, cfg)
    rng = np.random.default_rng(1)
    for _ in range(20):
        p = sampler.sample(rng)
        assert p.size == 128
        # single section
        assert np.unique(src.section[p.global_idx]).size == 1
        # patch diameter << full section diameter (it's a local disk)
        full = src.coords[src.section_of(p.section)]
        full_diam = np.ptp(full, axis=0).max()
        patch_diam = np.ptp(p.coords_raw, axis=0).max()
        assert patch_diam < full_diam
        # codes carried for both branches, right shape
        assert p.codes["cell"].shape == (128, 2)
        assert p.codes["niche"].shape == (128, 2)
    print("ok  test_patch_is_connected_and_single_section")


def test_block_mask_is_contiguous():
    """The masked cells should form a spatial disk: the masked centroid's
    nearest cells are (mostly) masked, far cells are observed."""
    from vqniche.stage2.masking import block_mask

    rng = np.random.default_rng(2)
    side = 30
    gx, gy = np.meshgrid(np.arange(side), np.arange(side))
    coords = np.stack([gx.ravel(), gy.ravel()], 1).astype(float)
    mask = block_mask(coords, frac=0.2, rng=rng)
    n_mask = mask.sum()
    assert n_mask == round(0.2 * coords.shape[0])

    centre = coords[mask].mean(0)
    d = np.sqrt(((coords - centre) ** 2).sum(1))
    order = np.argsort(d)
    # the n_mask cells closest to the masked centroid: vast majority masked
    closest = order[:n_mask]
    purity = mask[closest].mean()
    assert purity > 0.9, purity
    # the farthest cells are all observed
    assert not mask[order[-n_mask:]].any()
    print(f"ok  test_block_mask_is_contiguous (purity={purity:.2f})")


def test_random_mask_keeps_one_observed():
    from vqniche.stage2.masking import random_mask

    rng = np.random.default_rng(3)
    for frac in (0.1, 0.5, 0.99):
        m = random_mask(50, frac, rng)
        assert m.sum() >= 1
        assert (~m).sum() >= 1
    print("ok  test_random_mask_keeps_one_observed")


def test_coord_normalisation():
    from vqniche.stage2.data import normalise_coords

    rng = np.random.default_rng(4)
    coords = rng.normal(scale=37.0, size=(200, 2)) + np.array([1000.0, -500.0])
    cn = normalise_coords(coords, "std")
    assert np.allclose(cn.mean(0), 0, atol=1e-9)           # centred
    rms = np.sqrt((cn ** 2).sum(1).mean())
    assert abs(rms - 1.0) < 1e-6                            # unit scale
    cn2 = normalise_coords(coords, "knn")
    assert np.allclose(cn2.mean(0), 0, atol=1e-9)
    print("ok  test_coord_normalisation")


def test_knn_indices():
    from vqniche.stage2.data import knn_indices

    side = 12
    gx, gy = np.meshgrid(np.arange(side), np.arange(side))
    coords = np.stack([gx.ravel(), gy.ravel()], 1).astype(float)
    idx = knn_indices(coords, k=4)
    assert idx.shape == (side * side, 4)
    # an interior point's 4-NN on a grid are its 4 axis neighbours (dist 1)
    interior = side * 5 + 5
    d = np.sqrt(((coords[idx[interior]] - coords[interior]) ** 2).sum(1))
    assert np.allclose(np.sort(d), [1, 1, 1, 1])
    assert interior not in idx[interior]                   # excludes self
    print("ok  test_knn_indices")


def test_inpainting_patch_includes_holdout_and_context():
    from vqniche.stage2.config import DataConfig
    from vqniche.stage2.data import AnnDataCodeSource, inpainting_patch
    from vqniche.stage2.masking import block_mask

    ad, _, _ = _toy_adata()
    src = AnnDataCodeSource(ad)
    # define a held-out disk in section 0
    rows0 = src.section_of(0)

    rng = np.random.default_rng(5)
    sub_mask = block_mask(src.coords[rows0], frac=0.1, rng=rng)
    holdout = rows0[sub_mask]
    cfg = DataConfig(patch_size=400, mask_kind="block")
    p = inpainting_patch(src, holdout, cfg, context_radius_mult=2.0)
    # every held-out cell present and masked
    present = set(p.global_idx.tolist())
    assert set(holdout.tolist()).issubset(present)
    held_positions = {int(g): m for g, m in zip(p.global_idx, p.mask)}
    assert all(held_positions[int(h)] for h in holdout)
    # context cells (observed) included
    assert (~p.mask).sum() > 0
    # all one section
    assert np.unique(src.section[p.global_idx]).size == 1
    print(f"ok  test_inpainting_patch (patch={p.size}, holdout={len(holdout)}, "
          f"context={(~p.mask).sum()})")


def test_config_prediction_targets():
    from vqniche.stage2.config import Stage2Config

    cfg = Stage2Config.squint_default([30, 90], [30, 90])
    assert cfg.prediction_targets == [
        ("cell", 0), ("cell", 1), ("niche", 0), ("niche", 1)
    ]
    assert cfg.branch("cell").codebook_sizes == [30, 90]
    print("ok  test_config_prediction_targets")


def main():
    test_config_prediction_targets()
    test_source_reads_codes()
    test_patch_is_connected_and_single_section()
    test_block_mask_is_contiguous()
    test_random_mask_keeps_one_observed()
    test_coord_normalisation()
    test_knn_indices()
    test_inpainting_patch_includes_holdout_and_context()
    print("\nALL NUMPY-CORE TESTS PASSED")


if __name__ == "__main__":
    main()
