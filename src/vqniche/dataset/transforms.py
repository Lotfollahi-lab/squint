from torch_geometric.data import Data
from typing import Optional, List

import scanpy as sc
import anndata as ad

import torch
import torch_geometric.transforms as T

from ..preprocessors.normalizers import normalize_by_read_depth, normalize_total_log1p


def init_data_transforms(
        data_transform_names: List[str] = [
            'SubsetHVG',
            'NormalizeFeatures',
            'RandomNodeSplit',
        ],
        n_genes: int = 1000,
        norm_method: str = 'total_log1p',
        target_size: int = 10_000,
        apply_CPM: bool = True,
        val_ratio: float = 0.1,
        test_ratio: float = 0.2,
    ) -> List[T.BaseTransform]:
    """
    Initialize a list of PyG transforms to be applied to the Data object.

    Parameters:
    ----------
    - data_transform_names: List[str])
        The list of PyG transforms to initialize.
    - n_genes: int
        The number of highly variable genes to subset the data to.
    - norm_method: str
        The method to use for normalization.
    - target_size: int
        The target size for normalization.
    - apply_CPM: bool
        If True, apply CPM normalization.
    - val_ratio: float
        The ratio of validation data.
    - test_ratio: float
        The ratio of test data.

    Returns:
    -------
    - List[T.BaseTransform]
        A list of PyG transforms to be applied to data.

    """
    data_transforms = []

    for data_transform_name in data_transform_names:
        if data_transform_name == 'SubsetHVG':
            data_transforms.append(
                SubsetHVG(
                    n_genes=n_genes
                )
            )
        elif data_transform_name == 'NormalizeFeatures':
            data_transforms.append(
                NormalizeFeatures(
                    feature_key='x',
                    norm_method=norm_method,
                    target_size=target_size,
                    apply_CPM=apply_CPM
                )
            )

        elif data_transform_name == 'RandomNodeSplit':
            data_transforms.append(
                T.RandomNodeSplit(
                    split='train_rest',
                    num_val=val_ratio,
                    num_test=test_ratio
                )
            )

        else:
            raise ValueError(f"{data_transform_name} Transform not found.")

    return data_transforms


class SubsetHVG(T.BaseTransform):
    def __init__(
            self,
            n_genes: int = 1000,
        ):
        self.n_genes = n_genes


    def forward(
            self,
            data: Data
        ) -> Data:
        adata = ad.AnnData(data.x.numpy())
        sc.pp.highly_variable_genes(
            adata,
            flavor='seurat_v3',
            n_top_genes=self.n_genes,
            subset=True
        )
        data.x = torch.from_numpy(adata.X).to(data.x.device)
        data.num_features = data.x.shape[1]
        print(f"SubsetHVG: Subsetted data to {data.num_features} features.")
        return data


class NormalizeFeatures(T.BaseTransform):
    def __init__(
            self,
            norm_method: str = 'read_depth',
            feature_key: str = 'x',
            target_size: int = 10_000,
            apply_CPM: Optional[bool] = True
        ):
        """
        Normalize the features of the PyG Data object.

        Parameters:
        ----------
        - norm_method: str
            The method to use for normalization.
        - feature_key: str
            The key for the node features that are to be normalized.
        - target_size: int
            The target size for normalization.
        - apply_CPM: bool
            If True, apply CPM normalization.

        Notes:
        -----
        Ideally, the feature_key should be 'x' for the node features. The option exists to pass a custom key to allow for flexibility during training, if required.
        """
        self.norm_method = norm_method
        self.feature_key = feature_key
        self.target_size = target_size
        self.apply_CPM = apply_CPM

    def forward(
            self,
            data: Data
        ) -> Data:
        """
        Normalize the features of the PyG Data object.
        """
        feature_data = getattr(data, self.feature_key)
        if self.norm_method == 'read_depth':
            feature_data = normalize_by_read_depth(
                            x=feature_data,
                            target_size=self.target_size
                        )
        elif self.norm_method == 'total_log1p':
            feature_data = normalize_total_log1p(
                            x=feature_data,
                            target_size=self.target_size,
                            apply_CPM=self.apply_CPM
                        )
        else:
            raise ValueError(f"Normalization method {self.norm_method} not found.")
        setattr(data, self.feature_key, feature_data)
        return data


class SetExperimentDataKeys(T.BaseTransform):
    def __init__(
            self,
            feature_name: str = 'cell_gene_counts',
            label_name: str = 'cell_types',
            edge_index_name: str = 'spatial-delaunay'
        ):
        """
        Set data.x, data.y, and data.edge_index keys for the PyG Data object from Experiment keys.
        This is useful to control before starting training so we can choose on the fly which features, labels, and edge indices to use (e.g., cell gene counts for cell type classification, etc.)

        Parameters:
        ----------
        - feature_name: str
            The key for the node features to set.
        - label_name: str
            The key for the node labels to set.
        - edge_index_name: str
            The key for the edge index to set.
        """
        self.feature_key = f"x_{feature_name}"
        self.label_key = f"y_{label_name}"
        self.edge_index_key = f"edge_index_{edge_index_name}"

    def forward(
            self,
            data: Data
        ) -> Data:
        """
        Set Experiment data keys for the PyG Data object.
        """
        data.x = getattr(data, self.feature_key)
        data.y = getattr(data, self.label_key)
        data.edge_index = getattr(data, self.edge_index_key)
        data.num_features = data.x.shape[1]
        data.num_classes = data.y.shape[1]
        data.num_nodes = data.x.shape[0]
        data.num_edges = data.edge_index.shape[1]

        # delete extra features, labels, and edge indices from the data object to reduce memory footprint during training
        for key in list(data.keys()):
            if key.startswith('x_') or key.startswith('edge_index_'):
                delattr(data, key)

        return data