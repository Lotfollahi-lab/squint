import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp
import networkx as nx
import anndata as ad
from typing import List, Optional

from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import InMemoryDataset


def sparse_mx_to_float_tensor(
        sparse_mx: sp.csr_matrix
    ) -> torch.Tensor:
    """
    Convert a scipy sparse matrix to a PyTorch float tensor.

    Parameters:
    ----------
    sparse_mx : scipy.sparse.csr_matrix
        A sparse matrix in Compressed Sparse Row format.

    Returns:
    --------
    torch.Tensor
        A dense PyTorch float tensor with dtype float32.
    """
    # Ensure the data is in float32 format directly
    sparse_mx = sparse_mx.astype(np.float32)

    # Convert directly to dense format as a PyTorch tensor
    dense_tensor = torch.tensor(sparse_mx.toarray(), dtype=torch.float32)
    return dense_tensor


def pandas_to_torch_one_hot(
        pandas_series: pd.Series,
        categories: list[str] | None = None
    ) -> torch.FloatTensor:
    """
    Convert a pandas Series of categorical labels to a PyTorch float tensor of one-hot encoded labels.

    Parameters:
    ----------
    pandas_series : pd.Series
        A pandas Series of categorical labels.
    categories : list[str]
        A list of categories for the categorical labels.

    Returns:
    --------
    torch.FloatTensor
        A PyTorch float tensor of one-hot encoded labels.
    """
    if categories is None:
        categories = sorted(pandas_series.unique())

    # Create a categorical series with all possible categories
    cat_series = pd.Categorical(pandas_series, categories=categories)

    # Convert to one-hot using all categories
    one_hot = pd.get_dummies(cat_series, prefix='', prefix_sep='')

    # Ensure all categories are present (even if they weren't in this batch)
    for cat in categories:
        if cat not in one_hot.columns:
            one_hot[cat] = 0

    # Sort columns to ensure consistent order
    one_hot = one_hot[sorted(one_hot.columns)]

    return torch.tensor(one_hot.values, dtype=torch.float32)


def one_dim_to_one_hot(
        vector: np.ndarray,
        n_classes: Optional[int] = None
    ) -> np.array:
    """
    Converts an input 1-D vector of integer labels into a 2-D array of one-hot
    vectors, where for an i'th input value of j, a '1' will be inserted in the
    i'th row and j'th column of the output one-hot vector.
    
    Implementation is adapted from
    https://github.com/theislab/scib/blob/29f79d0135f33426481f9ff05dd1ae55c8787142/scib/metrics/lisi.py#L498
    (05.12.22).

    Parameters
    ----------
    vector:
        Vector to be one-hot-encoded.
    n_classes:
        Number of classes to be considered for one-hot-encoding. If `None`, the
        number of classes will be inferred from `vector`.

    Returns
    ----------
    one_hot:
        2-D NumPy array of one-hot-encoded vectors.

    Example:
    ```
    vector = np.array((1, 0, 4))
    one_hot = one_dim_to_one_hot(vector)
    print(one_hot)
    [[0 1 0 0 0]
     [1 0 0 0 0]
     [0 0 0 0 1]]
    ```
    """
    if n_classes is None:
        n_classes = np.max(vector) + 1

    one_hot = np.zeros(shape=(len(vector), n_classes))
    one_hot[np.arange(len(vector)), vector] = 1
    return one_hot.astype(int)


def edge_index_to_adjacency_tensor(
        edge_index: torch.Tensor,
        max_num_nodes: int | None = None
    ) -> torch.Tensor:
    """
    Convert an edge index to an adjacency tensor.

    Parameters:
    ----------
    edge_index: torch.Tensor
        The edge index to convert to an adjacency tensor.
        Dimensions: (2, num_edges)
    max_num_nodes: int | None
        The maximum number of nodes in the graph.
        If None, the maximum node ID in edge_index will be used.

    Returns:
    --------
    adjacency_matrix: torch.Tensor
        The adjacency tensor.
        Dimensions: (edge_index.max() + 1, edge_index.max() + 1) if max_num_nodes is None else (max_num_nodes, max_num_nodes)
    """
    # to_undirected ensures that the adjacency matrix is symmetric
    # to_dense_adj returns a tensor of shape (1, num_nodes, num_nodes)
    # here, num_nodes = edge_index.max() + 1 if max_num_nodes is None else max_num_nodes
    adjacency_matrix = to_dense_adj(
                        edge_index.to(torch.long),
                        max_num_nodes=max_num_nodes
                        )[0]
    return adjacency_matrix


def adjacency_tensor_to_networkx(
        adjacency_matrix: torch.Tensor
    ) -> nx.Graph:
    """
    Convert an adjacency tensor to a networkx graph.

    Parameters
    ----------
    - adjacency_matrix: torch.Tensor
        The adjacency matrix.
        Dimensions: (num_nodes, num_nodes)

    Returns
    -------
    - graph: nx.Graph
        The networkx graph.
    """
    return nx.from_numpy_array(adjacency_matrix.numpy())


def torch_one_hot_to_label_name(
        one_hot: torch.Tensor,
        label_categories: list[str]
    ) -> pd.Series:
    """
    Convert a one-hot encoded label tensor to a pandas Series of label names.

    Parameters:
    ----------
    one_hot: torch.Tensor
        The one-hot encoded label tensor.
    label_categories: list[str]
        The list of label categories where the index of the one-hot encoded label tensor matches the index of the label name in the list.

    Returns:
    --------
    pd.Series
        The pandas Series of label names.
    """
    indices = one_hot.argmax(dim=1).cpu().numpy()
    return pd.Series([label_categories[i] for i in indices])


def data_batch_to_adata_list(
        dataset_blob: InMemoryDataset,
        data_batch: Batch
    ) -> List[ad.AnnData]:
    """
    Convert a PyTorch Geometric Batch object to a list of AnnData objects.
    Each AnnData object corresponds to a tissue section.

    Parameters:
    ----------
    dataset_blob: InMemoryDatasetBlob
        The dataset blob.
    data_batch: Batch
        Combined data batch comprising of multiple tissue sections.

    Returns:
    --------
    List[ad.AnnData]
        A list of AnnData objects, one per tissue section.
    """
    adata_list = []
    for data in data_batch:
        # ------------------------------------------------------------------------
        # set features
        adata = ad.AnnData(X=data.x.cpu().numpy())

        # ------------------------------------------------------------------------
        # set gene panel
        adata.var = dataset_blob.gene_panel

        # ------------------------------------------------------------------------
        # set labels
        adata.obs['cell_types'] = torch_one_hot_to_label_name(
                                    data.y_cell_types.cpu(),
                                    dataset_blob.label_categories['cell_types']
                                    )
        adata.obs['niche_types'] = torch_one_hot_to_label_name(
                                    data.y_niche_types.cpu(),
                                    dataset_blob.label_categories['niche_types']
                                    )
        # ------------------------------------------------------------------------
        # Convert edge index to dense adjacency matrix
        adj_matrix: torch.Tensor = to_dense_adj(data.edge_index)[0]
        # Convert to scipy sparse matrix
        sparse_adj: sp.csr_matrix = sp.csr_matrix(adj_matrix.cpu().numpy())
        # Add to adata.obsp
        adata.obsp['spatial_connectivities'] = sparse_adj
        # Add metadata
        adata.uns['spatial_neighbors'] = {
            'connectivities_key': 'spatial_connectivities',
            'params': {
                'n_neighbors': None,
                'method': 'edge_index',
                'key_added': 'spatial'
            }
        }

        # ------------------------------------------------------------------------
        # set spatial coordinates
        adata.obsm['spatial'] = data.xy_coordinates.cpu().numpy()

        # ------------------------------------------------------------------------
        # set metadata
        adata.obs['cell_id'] = data.cell_id
        adata.uns['dataset_id'] = data.dataset_id
        adata.uns['tissue'] = data.tissue
        adata.uns['species'] = data.species
        adata.uns['batch'] = f"batch_{data.batch_idx}"

        # ------------------------------------------------------------------------
        adata_list.append(adata)

    return adata_list