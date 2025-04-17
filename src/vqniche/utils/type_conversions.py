import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp
import networkx as nx
import anndata as ad
from typing import List

from torch_geometric.data import Batch
from torch_geometric.utils import to_undirected, to_dense_adj

from ..dataset.in_memory_dataset_blob import InMemoryDatasetBlob


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


def edge_index_to_adjacency_tensor(
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
    """
    Convert an edge index to an adjacency tensor.

    Parameters:
    ----------
    edge_index: torch.Tensor
        The edge index to convert to an adjacency tensor.
        Dimensions: (2, num_edges)

    Returns:
    --------
    adjacency_matrix: torch.Tensor
        The adjacency tensor.
        Dimensions: (edge_index.max() + 1, edge_index.max() + 1)
    """
    # to_undirected ensures that the adjacency matrix is symmetric
    # to_dense_adj returns a tensor of shape (1, num_nodes, num_nodes)
    # here, num_nodes = edge_index.max() + 1
    adjacency_matrix = to_dense_adj(
                        to_undirected(edge_index).to(torch.long)
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
    return pd.Series(label_categories[one_hot.argmax(dim=1)])


def data_batch_to_adata_list(
        dataset_blob: InMemoryDatasetBlob,
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
        adj_matrix = to_dense_adj(data.edge_index)[0]
        # Convert to scipy sparse matrix
        sparse_adj = sp.csr_matrix(adj_matrix.cpu().numpy())
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