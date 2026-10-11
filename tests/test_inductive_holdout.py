"""
Tests for the opt-in inductive held-out-region protocol:

- squint.dataset.transforms.make_inductive_edge_index (graph rewrite),
- SpatialBatchSplit / init_train_transforms holdout_graph knobs,
- initialize_dataset_blob's holdout_knn_k injection,
- the InMemoryDataModule static guard (assert_inductive_holdout),
- the Squint_Dual runtime batch guards,
- PyG NeighborLoader never putting a held-out node into an observed
  seed's batch once the graph is rewritten.

Synthetic data: a uniform 2-D point cloud, an axis-aligned held-out box and
a symmetrised kNN graph with self-loops built in numpy/scipy with the blob's
conventions (k nearest OTHER points, self-loops, A.maximum(A.T)). The graph
builder is injected via `knn_builder`, so only the one test that checks the
default (blob) builder exercises squidpy's kNN. Importing `squint.dataset`
still pulls in scanpy and squidpy (in_memory_dataset_blob ->
preprocessors.graph_constructors), so both are needed to collect the module.

Meant for the farm venv (torch + PyG). Skips cleanly when torch,
torch_geometric, scanpy or squidpy are not installed.
"""
import types

import pytest

np = pytest.importorskip("numpy")
sp = pytest.importorskip("scipy.sparse")
torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
# `import squint.dataset` imports scanpy and squidpy at module level.
pytest.importorskip("scanpy")
pytest.importorskip("squidpy")

from torch_geometric.data import Data  # noqa: E402

from squint.dataset.transforms import (  # noqa: E402
    SpatialBatchSplit,
    init_train_transforms,
    make_inductive_edge_index,
)


N = 1500
K = 8
# Absolute box on the unit square: x_min, x_max, y_min, y_max.
BOX = (0.35, 0.60, 0.40, 0.65)
BOX_REGION = {"x_min": BOX[0], "x_max": BOX[1], "y_min": BOX[2], "y_max": BOX[3]}
BATCH_ID = 15

# The run_squint base train_transform_params plus the region-holdout patch,
# i.e. a config saved before the holdout_graph keys existed.
LEGACY_TRAIN_TRANSFORM_KWARGS = {
    "train_transform_names": ["SpatialBatchSplit"],
    "region": None,
    "train_batches": [15, 82],
    "val_batches": [],
    "test_batches": [],
    "xy_key": "xy_coordinates",
    "train_val_cell_split": 0.10,
    "cell_split_seed": 0,
    "test_regions": {
        15: {"x_min_pct": 0.10, "x_max_pct": 0.35, "y_min_pct": 0.55, "y_max_pct": 0.80},
        82: {"x_min_pct": 0.55, "x_max_pct": 0.80, "y_min_pct": 0.20, "y_max_pct": 0.45},
    },
}


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _cloud(n=N, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((n, 2)).astype(np.float32)


def _box_mask(xy, box=BOX):
    x0, x1, y0, y1 = box
    return (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)


def _sq_dists(xy):
    xy = np.asarray(xy, dtype=np.float64)
    d2 = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    return d2


def _numpy_knn_adjacency(xy, k):
    """
    Blob-convention kNN adjacency: the k nearest OTHER points of every point
    (sklearn kneighbors() semantics, as used by squidpy), plus self-loops,
    symmetrised with A.maximum(A.T).
    """
    n = int(np.asarray(xy).shape[0])
    nn = np.argsort(_sq_dists(xy), axis=1, kind="stable")[:, :k]
    rows = np.concatenate([np.repeat(np.arange(n), k), np.arange(n)])
    cols = np.concatenate([nn.reshape(-1), np.arange(n)])
    A = sp.csr_matrix(
        (np.ones(rows.size, dtype=np.float32), (rows, cols)), shape=(n, n)
    )
    return A.maximum(A.T).tocsr()


def _kth_nn_dist(xy, k):
    return np.sqrt(np.sort(_sq_dists(xy), axis=1)[:, k - 1])


def _edge_index_from_adj(A):
    A = A.tocoo()
    return torch.stack([
        torch.from_numpy(A.row).to(torch.long),
        torch.from_numpy(A.col).to(torch.long),
    ], dim=0)


def _edge_set(ei):
    return set(map(tuple, ei.t().tolist()))


def _in_nbrs(ei, node):
    return set(ei[0, ei[1] == node].tolist())


def _out_nbrs(ei, node):
    return set(ei[1, ei[0] == node].tolist())


def _check_core_guarantees(out, t, num_nodes, ref_dtype=torch.long):
    """C1 guarantees (i), (iv), (v) plus shape / dtype."""
    assert out.dim() == 2 and out.size(0) == 2
    assert out.dtype == ref_dtype
    src, dst = out[0], out[1]
    # (i) no held-out -> observed edge
    assert not bool((t[src] & ~t[dst]).any())
    # (iv) exactly one self-loop per node
    loops = src[src == dst]
    assert torch.equal(torch.sort(loops).values, torch.arange(num_nodes))
    # (v) coalesced: sorted by (src, dst), no duplicates
    key = src * num_nodes + dst
    assert torch.equal(key, torch.unique(key, sorted=True))


def _section_data(xy, ei, batch_id=BATCH_ID):
    data = Data(
        x=torch.rand(xy.shape[0], 4),
        edge_index=ei.clone(),
        xy_coordinates=xy.clone(),
        adata_batch_id=batch_id,
    )
    # SetExperimentDataKeys stores num_edges explicitly.
    data.num_edges = int(ei.size(1))
    return data


@pytest.fixture(scope="module")
def section():
    xy = _cloud()
    mask = _box_mask(xy)
    ei = _edge_index_from_adj(_numpy_knn_adjacency(xy, K))
    return torch.from_numpy(xy), torch.from_numpy(mask), ei


# ----------------------------------------------------------------------------
# make_inductive_edge_index
# ----------------------------------------------------------------------------

def test_fixture_is_transductive(section):
    """Sanity: the input graph does connect held-out cells to observed ones."""
    xy, t, ei = section
    assert 50 < int(t.sum()) < N - 50
    assert int((t[ei[0]] & ~t[ei[1]]).sum()) > 0
    # symmetric with self-loops, like the blob graph
    assert _edge_set(ei) == _edge_set(ei.flip(0))
    assert int((ei[0] == ei[1]).sum()) == N


@pytest.mark.parametrize("test_inedges", ["full", "test_only", "none"])
@pytest.mark.parametrize("mode", ["drop", "rebuild"])
def test_inductive_edge_index_guarantees(section, mode, test_inedges):
    xy, t, ei = section
    out = make_inductive_edge_index(
        ei, t, xy, mode=mode, k=K, test_inedges=test_inedges,
        knn_builder=_numpy_knn_adjacency,
    )
    _check_core_guarantees(out, t, N, ref_dtype=ei.dtype)
    assert out.device == ei.device
    src, dst = out[0], out[1]

    # (ii) observed -> observed component
    obs_out = _edge_set(out[:, ~t[src] & ~t[dst]])
    if mode == "drop":
        expected_obs = _edge_set(ei[:, ~t[ei[0]] & ~t[ei[1]]])
    else:
        obs_idx = torch.nonzero(~t, as_tuple=False).view(-1)
        local = _edge_index_from_adj(_numpy_knn_adjacency(xy[obs_idx].numpy(), K))
        expected_obs = _edge_set(obs_idx[local])
    assert obs_out == expected_obs

    # (iii) edges INTO held-out cells
    into_test = _edge_set(out[:, t[dst]])
    orig_into_test = ei[:, t[ei[1]]]
    if test_inedges == "full":
        expected_in = _edge_set(orig_into_test)
    elif test_inedges == "test_only":
        expected_in = _edge_set(orig_into_test[:, t[orig_into_test[0]]])
    else:
        expected_in = {(i, i) for i in torch.nonzero(t).view(-1).tolist()}
    assert into_test == expected_in

    # Nothing else: every edge is observed->observed or ends in a held-out cell.
    assert _edge_set(out) == obs_out | into_test


def test_inductive_edge_index_does_not_mutate_input(section):
    xy, t, ei = section
    ei_before = ei.clone()
    t_before = t.clone()
    make_inductive_edge_index(ei, t, xy, mode="rebuild", k=K, knn_builder=_numpy_knn_adjacency)
    make_inductive_edge_index(ei, t, xy, mode="drop")
    assert torch.equal(ei, ei_before)
    assert torch.equal(t, t_before)


def test_rebuild_restores_observed_degree(section):
    """'rebuild' gives boundary observed cells >= k observed neighbours, 'drop' does not."""
    xy, t, ei = section
    cross = t[ei[0]] & ~t[ei[1]]
    band = torch.unique(ei[1][cross]).tolist()
    assert len(band) > 0

    rebuilt = make_inductive_edge_index(ei, t, xy, mode="rebuild", k=K, knn_builder=_numpy_knn_adjacency)
    dropped = make_inductive_edge_index(ei, t, xy, mode="drop")
    for i in band:
        assert len(_in_nbrs(rebuilt, i) - {i}) >= K
    assert any(len(_in_nbrs(dropped, i) - {i}) < K for i in band)


@pytest.mark.parametrize("mode", ["drop", "rebuild"])
def test_all_false_mask_reproduces_input_graph(section, mode):
    xy, _, ei = section
    t0 = torch.zeros(N, dtype=torch.bool)
    out = make_inductive_edge_index(
        ei, t0, xy, mode=mode, k=K, knn_builder=_numpy_knn_adjacency,
    )
    assert _edge_set(out) == _edge_set(ei)
    assert out.size(1) == ei.size(1)


def test_rebuild_keeps_interior_neighbourhoods(section):
    """
    Observed cells farther than R + R' from the box keep identical in- and
    out-neighbourhoods under 'rebuild', where R is the largest k-th NN
    distance over all cells and R' the largest k-th NN distance among
    observed cells only (a sufficient bound for a symmetrised kNN).
    """
    xy, t, ei = section
    out = make_inductive_edge_index(ei, t, xy, mode="rebuild", k=K, knn_builder=_numpy_knn_adjacency)

    xy_np = xy.numpy()
    t_np = t.numpy()
    R = float(_kth_nn_dist(xy_np, K).max())
    R_obs = float(_kth_nn_dist(xy_np[~t_np], K).max())
    x0, x1, y0, y1 = BOX
    dx = np.maximum(np.maximum(x0 - xy_np[:, 0], 0.0), xy_np[:, 0] - x1)
    dy = np.maximum(np.maximum(y0 - xy_np[:, 1], 0.0), xy_np[:, 1] - y1)
    dist_to_box = np.hypot(dx, dy)
    interior = np.where((~t_np) & (dist_to_box > R + R_obs))[0]
    assert interior.size >= 50, (interior.size, R, R_obs)

    for i in interior.tolist():
        assert _in_nbrs(out, i) == _in_nbrs(ei, i)
        assert _out_nbrs(out, i) == _out_nbrs(ei, i)


def test_invalid_arguments(section):
    xy, t, ei = section
    with pytest.raises(ValueError):
        make_inductive_edge_index(ei, t, xy, mode="transductive")
    with pytest.raises(ValueError):
        make_inductive_edge_index(ei, t, xy, mode="rebuild", k=None)
    with pytest.raises(ValueError):
        make_inductive_edge_index(ei, t, xy, mode="drop", test_inedges="all")


# ----------------------------------------------------------------------------
# PyG NeighborLoader: no held-out node in any observed-seed batch
# ----------------------------------------------------------------------------

def _loader_batches(data, **kwargs):
    from torch_geometric.loader import NeighborLoader
    try:
        return list(NeighborLoader(data, **kwargs))
    except ImportError as e:  # neither pyg-lib nor torch-sparse installed
        pytest.skip(f"NeighborLoader sampling backend unavailable: {e}")


@pytest.mark.parametrize("num_neighbors", [[K], [K, K], [-1]])
def test_neighbor_loader_train_batches_never_contain_test_nodes(section, num_neighbors):
    xy, t, ei = section
    rng = torch.Generator().manual_seed(0)
    val_mask = (~t) & (torch.rand(N, generator=rng) < 0.1)
    train_mask = (~t) & ~val_mask

    out = make_inductive_edge_index(ei, t, xy, mode="rebuild", k=K, knn_builder=_numpy_knn_adjacency)
    data = Data(x=torch.rand(N, 4), edge_index=out, test_mask=t,
                train_mask=train_mask, val_mask=val_mask)

    for seeds in (train_mask, val_mask):
        batches = _loader_batches(data, num_neighbors=num_neighbors,
                                  input_nodes=seeds, batch_size=64, shuffle=True)
        assert batches
        n_seeds = 0
        for b in batches:
            assert not bool(b.test_mask.any())
            if len(num_neighbors) == 1:
                # directional 1-hop sampling: every edge points INTO a seed
                assert bool((b.edge_index[1] < b.batch_size).all())
            n_seeds += int(b.batch_size)
        assert n_seeds == int(seeds.sum())


def test_neighbor_loader_transductive_graph_does_leak(section):
    """Control: on the original graph boundary seeds DO pull held-out nodes."""
    xy, t, ei = section
    data = Data(x=torch.rand(N, 4), edge_index=ei, test_mask=t, train_mask=~t)
    batches = _loader_batches(data, num_neighbors=[-1], input_nodes=~t,
                              batch_size=64, shuffle=False)
    assert any(bool(b.test_mask.any()) for b in batches)


@pytest.mark.parametrize("mode", ["drop", "rebuild"])
def test_neighbor_loader_predict_batches(section, mode):
    """
    Predict-style pass (input_nodes=None, full fan-out): held-out nodes may
    appear, but no batch edge goes from a held-out node to an observed one,
    and held-out seeds keep their full original in-neighbourhood ('full').
    """
    xy, t, ei = section
    out = make_inductive_edge_index(ei, t, xy, mode=mode, k=K, knn_builder=_numpy_knn_adjacency)
    data = Data(x=torch.rand(N, 4), edge_index=out, test_mask=t,
                node_id=torch.arange(N))
    batches = _loader_batches(data, num_neighbors=[-1], input_nodes=None,
                              batch_size=256, shuffle=False)
    n_seeds = 0
    seen_obs_to_test = False
    for b in batches:
        tm = b.test_mask
        src, dst = b.edge_index
        assert not bool((tm[src] & ~tm[dst]).any())
        seen_obs_to_test |= bool((~tm[src] & tm[dst]).any())
        # every seed's in-neighbourhood in the batch equals its graph one
        gid = b.node_id
        for s in range(int(b.batch_size)):
            got = set(gid[src[dst == s]].tolist())
            assert got == _in_nbrs(out, int(gid[s]))
        n_seeds += int(b.batch_size)
    assert n_seeds == N
    assert seen_obs_to_test


# ----------------------------------------------------------------------------
# SpatialBatchSplit / init_train_transforms
# ----------------------------------------------------------------------------

def test_init_train_transforms_legacy_kwargs():
    transforms = init_train_transforms(**LEGACY_TRAIN_TRANSFORM_KWARGS)
    assert len(transforms) == 1
    split = transforms[0]
    assert isinstance(split, SpatialBatchSplit)
    assert split.holdout_graph == "transductive"
    assert split.holdout_knn_k is None
    assert split.holdout_test_inedges == "full"


def test_init_train_transforms_forwards_holdout_kwargs():
    transforms = init_train_transforms(
        **LEGACY_TRAIN_TRANSFORM_KWARGS,
        holdout_graph="rebuild", holdout_knn_k=16, holdout_test_inedges="test_only",
    )
    split = transforms[0]
    assert split.holdout_graph == "rebuild"
    assert split.holdout_knn_k == 16
    assert split.holdout_test_inedges == "test_only"


def test_spatial_batch_split_validation():
    with pytest.raises(ValueError):
        SpatialBatchSplit(holdout_graph="rebuild")  # needs holdout_knn_k
    with pytest.raises(ValueError):
        SpatialBatchSplit(holdout_graph="inductive")
    with pytest.raises(ValueError):
        SpatialBatchSplit(holdout_graph="drop", holdout_test_inedges="all")
    # YAML nulls fall back to the legacy defaults
    split = SpatialBatchSplit(holdout_graph=None, holdout_test_inedges=None)
    assert split.holdout_graph == "transductive"
    assert split.holdout_test_inedges == "full"


def test_spatial_batch_split_default_keeps_graph(section):
    xy, t, ei = section
    split = SpatialBatchSplit(train_batches=[], val_batches=[], test_batches=[],
                              test_regions={BATCH_ID: BOX_REGION})
    out = split(_section_data(xy, ei))
    assert torch.equal(out.edge_index, ei)
    assert out.num_edges == ei.size(1)
    assert torch.equal(out.test_mask, t)


@pytest.mark.parametrize("test_inedges", ["full", "none"])
def test_spatial_batch_split_drop(section, test_inedges):
    xy, t, ei = section
    kw = dict(train_batches=[], val_batches=[], test_batches=[],
              train_val_cell_split=0.10, cell_split_seed=0,
              test_regions={BATCH_ID: BOX_REGION})
    legacy = SpatialBatchSplit(**kw)(_section_data(xy, ei))
    out = SpatialBatchSplit(**kw, holdout_graph="drop",
                            holdout_test_inedges=test_inedges)(_section_data(xy, ei))

    # masks (and the val subsample) are unchanged, only edges move
    for key in ("test_mask", "train_mask", "val_mask"):
        assert torch.equal(out[key], legacy[key])
    _check_core_guarantees(out.edge_index, out.test_mask, N, ref_dtype=ei.dtype)
    assert out.num_edges == out.edge_index.size(1)
    expected = make_inductive_edge_index(ei, out.test_mask, xy, mode="drop",
                                         test_inedges=test_inedges)
    assert torch.equal(out.edge_index, expected)


def test_spatial_batch_split_untouched_section(section):
    """A section without a test_regions entry keeps its graph under 'drop'."""
    xy, _, ei = section
    split = SpatialBatchSplit(train_batches=[82], val_batches=[], test_batches=[],
                              test_regions={BATCH_ID: BOX_REGION},
                              holdout_graph="drop")
    out = split(_section_data(xy, ei, batch_id=82))
    assert torch.equal(out.edge_index, ei)
    assert not bool(out.test_mask.any())


def test_default_builder_matches_blob_graph(section):
    """
    The default 'rebuild' builder goes through the blob's own constructor
    (squint spatial_neighbors + A.maximum(A.T)): with an all-False mask it
    must reproduce a blob-built graph, and with a box it must equal the blob
    construction on the observed cells only.
    """
    import anndata as ad
    import pandas as pd
    from torch_geometric.utils.convert import from_scipy_sparse_matrix
    from squint.preprocessors.graph_constructors import spatial_neighbors

    def blob_graph(xy_np):
        adata = ad.AnnData(obs=pd.DataFrame(index=[str(i) for i in range(xy_np.shape[0])]),
                           obsm={"spatial": xy_np})
        adata = spatial_neighbors(adata, coord_type="generic", spatial_key="spatial",
                                  delaunay=False, n_neighs=K, radius=None,
                                  include_self_loop=True, key_added="ref")
        A = adata.obsp["ref_connectivities"]
        if (A != A.T).nnz > 0:
            A = A.maximum(A.T)
        return from_scipy_sparse_matrix(A)[0]

    xy, _, _ = section
    ei_blob = blob_graph(xy.numpy())
    t0 = torch.zeros(N, dtype=torch.bool)
    out0 = make_inductive_edge_index(ei_blob, t0, xy, mode="rebuild", k=K)
    assert _edge_set(out0) == _edge_set(ei_blob)

    split = SpatialBatchSplit(train_batches=[], val_batches=[], test_batches=[],
                              test_regions={BATCH_ID: BOX_REGION},
                              holdout_graph="rebuild", holdout_knn_k=K)
    out = split(_section_data(xy, ei_blob))
    t = out.test_mask
    assert bool(t.any())
    _check_core_guarantees(out.edge_index, t, N, ref_dtype=ei_blob.dtype)
    src, dst = out.edge_index
    obs_idx = torch.nonzero(~t, as_tuple=False).view(-1)
    expected_obs = _edge_set(obs_idx[blob_graph(xy[obs_idx].numpy())])
    assert _edge_set(out.edge_index[:, ~t[src] & ~t[dst]]) == expected_obs
    assert _edge_set(out.edge_index[:, t[dst]]) == _edge_set(ei_blob[:, t[ei_blob[1]]])


# ----------------------------------------------------------------------------
# initialize_dataset_blob: holdout_knn_k injection
# ----------------------------------------------------------------------------

def _minimal_config(train_transform_params, graph_params=None):
    return {
        "dataset": {
            "gene_count_transform_names": [],
            "gene_count_transform_params": {},
            "graph_params": graph_params or {
                "spatial_key": "spatial", "delaunay": False,
                "n_neighs": 16, "radius": None,
            },
            "feature_names": ["X"],
            "label_name": "cell_types",
            "train_transform_names": ["SpatialBatchSplit"],
            "train_transform_params": train_transform_params,
            "root_data_dir": "/nonexistent",
            "dataset_name": "synthetic",
        },
        "model": {"encoder_params": {}},
    }


def _init_blob_with_stub(monkeypatch, config):
    pytest.importorskip("pytorch_lightning")
    import squint.initializers.initialize as init_mod

    class _StubBlob:
        def __init__(self, name, data_directory_path, transform):
            self.transform = transform

    monkeypatch.setattr(init_mod, "InMemoryDatasetBlob", _StubBlob)
    blob = init_mod.initialize_dataset_blob(config)
    splits = [tr for tr in blob.transform.transforms if isinstance(tr, SpatialBatchSplit)]
    assert len(splits) == 1
    return splits[0]


def test_initialize_injects_holdout_knn_k(monkeypatch):
    tp = {k: v for k, v in LEGACY_TRAIN_TRANSFORM_KWARGS.items()
          if k != "train_transform_names"}
    tp["holdout_graph"] = "rebuild"
    config = _minimal_config(tp)
    split = _init_blob_with_stub(monkeypatch, config)
    assert split.holdout_graph == "rebuild"
    assert split.holdout_knn_k == 16
    # the config records the injected k (for the re-saved run config and
    # predict()'s provenance) in a new dict; the caller's dict is untouched
    assert config["dataset"]["train_transform_params"]["holdout_knn_k"] == 16
    assert config["dataset"]["train_transform_params"] is not tp
    assert "holdout_knn_k" not in tp


def test_initialize_legacy_config_untouched(monkeypatch):
    tp = {k: v for k, v in LEGACY_TRAIN_TRANSFORM_KWARGS.items()
          if k != "train_transform_names"}
    config = _minimal_config(tp)
    split = _init_blob_with_stub(monkeypatch, config)
    assert split.holdout_graph == "transductive"
    assert split.holdout_knn_k is None
    assert config["dataset"]["train_transform_params"] is tp
    assert "holdout_knn_k" not in tp


def test_initialize_rejects_rebuild_on_delaunay_graph(monkeypatch):
    tp = {k: v for k, v in LEGACY_TRAIN_TRANSFORM_KWARGS.items()
          if k != "train_transform_names"}
    tp["holdout_graph"] = "rebuild"
    config = _minimal_config(tp, graph_params={
        "spatial_key": "spatial", "delaunay": True, "n_neighs": None, "radius": None,
    })
    with pytest.raises(ValueError):
        _init_blob_with_stub(monkeypatch, config)


# ----------------------------------------------------------------------------
# InMemoryDataModule static guard
# ----------------------------------------------------------------------------

def _loader_data(xy, t, edge_index):
    n = xy.shape[0]
    return Data(
        x=torch.rand(n, 4),
        y=torch.zeros(n, 0),
        edge_index=edge_index,
        xy_coordinates=xy,
        train_mask=~t,
        val_mask=torch.zeros(n, dtype=torch.bool),
        test_mask=t,
        adata_batch_ids=torch.zeros(n, dtype=torch.long),
    )


def _datamodule_kwargs(flag):
    return dict(
        loader_name="NeighborLoader",
        loader_params={"batch_size": 64, "num_workers": 0},
        sampler_name="NeighborSampler",
        sampler_params={"num_neighbors": [K]},
        assert_inductive_holdout=flag,
    )


def test_datamodule_guard(section, capsys):
    pytest.importorskip("pytorch_lightning")
    from squint.dataloaders.in_memory_datamodule import InMemoryDataModule

    xy, t, ei = section
    with pytest.raises(RuntimeError, match="held-out->observed"):
        InMemoryDataModule(data=_loader_data(xy, t, ei), **_datamodule_kwargs(True))

    out = make_inductive_edge_index(ei, t, xy, mode="rebuild", k=K, knn_builder=_numpy_knn_adjacency)
    capsys.readouterr()
    InMemoryDataModule(data=_loader_data(xy, t, out), **_datamodule_kwargs(True))
    printed = capsys.readouterr().out
    assert "held-out->observed edges: 0" in printed
    assert f"n_test={int(t.sum())}" in printed
    assert f"n_edges={int(out.size(1))}" in printed


def test_datamodule_guard_off_by_default(section, capsys):
    pytest.importorskip("pytorch_lightning")
    from squint.dataloaders.in_memory_datamodule import InMemoryDataModule

    xy, t, ei = section
    kwargs = _datamodule_kwargs(False)
    kwargs.pop("assert_inductive_holdout")
    InMemoryDataModule(data=_loader_data(xy, t, ei), **kwargs)
    assert "held-out->observed" not in capsys.readouterr().out


# ----------------------------------------------------------------------------
# Squint_Dual runtime guards (called on a stand-in `self`; the guard runs
# before the forward pass, which the stand-in replaces with a sentinel)
# ----------------------------------------------------------------------------

class _ReachedForward(Exception):
    pass


class _FakeModel:
    def __init__(self, flag):
        self.loss_kwargs = {"assert_inductive_holdout": flag}

    def __call__(self, *args, **kwargs):
        raise _ReachedForward()


def _fake_batch(test_mask, edge_index):
    n = int(test_mask.numel())
    return types.SimpleNamespace(
        x=torch.rand(n, 3),
        edge_index=edge_index,
        test_mask=test_mask,
        batch_size=1,
    )


def test_squint_dual_step_guard():
    from squint.models.squint_dual import Squint_Dual

    ei = torch.tensor([[0, 1, 1], [0, 1, 0]])
    with_test = _fake_batch(torch.tensor([False, True]), ei)
    clean = _fake_batch(torch.tensor([False, False]), ei)

    for mode in ("train", "val"):
        with pytest.raises(RuntimeError, match="held-out"):
            Squint_Dual._step(_FakeModel(True), with_test, mode)
        with pytest.raises(_ReachedForward):
            Squint_Dual._step(_FakeModel(True), clean, mode)
        # legacy behaviour: no check without the flag
        with pytest.raises(_ReachedForward):
            Squint_Dual._step(_FakeModel(False), with_test, mode)


def test_squint_dual_predict_guard():
    from squint.models.squint_dual import Squint_Dual

    tm = torch.tensor([False, True])
    leaky = _fake_batch(tm, torch.tensor([[0, 1, 1], [0, 1, 0]]))   # 1 -> 0: held-out -> observed
    inductive = _fake_batch(tm, torch.tensor([[0, 0, 1], [0, 1, 1]]))  # 0 -> 1 only

    with pytest.raises(RuntimeError, match="held-out->observed"):
        Squint_Dual.predict_step(_FakeModel(True), leaky)
    with pytest.raises(_ReachedForward):
        Squint_Dual.predict_step(_FakeModel(True), inductive)
    with pytest.raises(_ReachedForward):
        Squint_Dual.predict_step(_FakeModel(False), leaky)
