"""
Reference: https://github.com/Lotfollahi-lab/nichecompass/blob/main/src/nichecompass/benchmarking/utils.py
"""
import scanpy as sc
from anndata import AnnData


def compute_knn_graph_connectivities_and_distances(
        adata: AnnData,
        feature_key: str="H_adj",
        knng_key: str="H_adj_15knng",
        fully_connected: bool=False,
        n_neighbors: int=15,
        random_state: int=0,
    ) -> None:
    """
    Compute connectivity graph using PyNNDescent.
    If `fully_connected` is True, uses a Gaussian kernel to assign low weights to nodes more distant than the k-nearest neighbors.
    If `fully_connected` is False, uses a hard threshold to restrict the number of neighbors to `n_neighbors` using a UMAP connectivities.

    Parameters
    ----------
    adata:
        AnnData object with the features for connectivity graph computation stored in
        `adata.obsm[feature_key]`.
    feature_key:
        Key in `adata.obsm` that will be used to compute the connectivity graph.
    fully_connected:
        If True, uses a Gaussian kernel to assign low weights to nodes more distant than the k-nearest neighbors.
        If False, uses a hard threshold to restrict the number of neighbors to `n_neighbors` using a UMAP connectivities.
    knng_key:
        Key under which the connectivity graph will be stored in `adata.obsp`
        with the suffix '_connectivities'.
    n_neighbors:
        Number of neighbors of the knn graph.
    random_state:
        Random state for reproducibility.   
    """
    if fully_connected:
        knn = False
        method='gauss'
    else:
        knn = True
        method='umap'

    sc.pp.neighbors(
            adata=adata,
            n_neighbors=n_neighbors, # default is 15
            knn=knn, # default is True, 
            method=method, # default is 'umap', 'gauss' assigns low weights to distant neighbors
            transformer='pynndescent',
            random_state=random_state,
            use_rep=feature_key, # which embedding to use for knn graph computation
            key_added=knng_key, # key under which the knn graph connectivities  will be stored
        )