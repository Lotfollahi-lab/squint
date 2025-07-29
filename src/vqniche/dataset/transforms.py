from typing import Optional, List, Literal

import scanpy as sc
import anndata as ad

import torch
from torch_geometric.data import Data
import torch_geometric.transforms as T

from ..preprocessors.normalizers import normalize_by_read_depth, normalize_total_log1p


def init_gene_count_transforms(
        gene_count_transform_names: List[Literal['SubsetHVG', 'NormalizeFeatures']] = ['SubsetHVG'],
        n_genes: int = 1000,
        norm_method: str = 'total_log1p',
        target_size: int = 10_000,
        apply_CPM: bool = True,
    ) -> List[T.BaseTransform]:
    """
    Initialize a list of PyG transforms to be applied to the raw cell gene counts matrix.

    Parameters:
    ----------
    - gene_count_transform_names: List[Literal['SubsetHVG', 'NormalizeFeatures']]
        The list of PyG transforms to initialize. Currently, only 'SubsetHVG' and 'NormalizeFeatures' are supported.
    - n_genes: int
        The number of highly variable genes to subset the data to.
    - norm_method: str
        The method to use for normalization.
    - target_size: int
        The target size for normalization.
    - apply_CPM: bool
        If True, apply CPM normalization.

    Returns:
    -------
    - List[T.BaseTransform]
        A list of PyG transforms to be applied to data.

    Notes:
    -----
    - Transforms are not order-invariant.
    - If 'SubsetHVG' is listed, it is applied before 'NormalizeFeatures'.
    """
    gene_count_transforms = []

    if 'SubsetHVG' in gene_count_transform_names:
        gene_count_transforms.append(
            SubsetHVG(
                n_genes=n_genes
            )
        )

    if 'NormalizeFeatures' in gene_count_transform_names:
        gene_count_transforms.append(
            NormalizeFeatures(
                norm_method=norm_method,
                target_size=target_size,
                apply_CPM=apply_CPM
            )
        )

    return gene_count_transforms


def init_train_transforms(
        train_transform_names: List[Literal['RandomNodeSplit']] = ['RandomNodeSplit'],
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ) -> List[T.BaseTransform]:
    """
    Initialize a list of training-related PyG transforms to be applied to the Data object.

    Parameters:
    ----------
    - train_transform_names: List[Literal['RandomNodeSplit']]
        The list of PyG transforms to initialize. Currently, only 'RandomNodeSplit' is supported.
    - val_ratio: float
        Fraction of the data to set aside for validation.
    - test_ratio: float
        Fraction of the data to set aside for testing.
    """
    train_transforms = []
    if 'RandomNodeSplit' in train_transform_names:
        train_transforms.append(
            T.RandomNodeSplit(
                split='train_rest',
                num_val=val_ratio,
                num_test=test_ratio,
                key='y'
            )
        )
    return train_transforms


class SubsetHVG(T.BaseTransform):
    def __init__(
            self,
            n_genes: int = 1000,
        ):
        """
        Subset the raw gene counts matrix to the top n_genes highly variable genes.

        Parameters:
        ----------
        - n_genes: int
            The number of highly variable genes to subset the data to.
        """
        self.n_genes = n_genes

    def forward(
            self,
            data: Data
        ) -> Data:
        """
        This transform is applied to the raw gene counts matrix.
        """
        adata = ad.AnnData(data.x_cell_gene_counts.numpy())
        sc.pp.highly_variable_genes(
            adata,
            flavor='seurat_v3',
            n_top_genes=self.n_genes,
            subset=True
        )
        # we manually set the data.x to the subsetted adata.X
        data.x = torch.from_numpy(adata.X)
        print(f"SubsetHVG: Subsetted data to {data.num_features} features.")
        return data


class NormalizeFeatures(T.BaseTransform):
    def __init__(
            self,
            norm_method: str = 'read_depth',
            target_size: int = 10_000,
            apply_CPM: Optional[bool] = True
        ):
        """
        Normalize the features of the PyG Data object.

        Parameters:
        ----------
        - norm_method: str
            The method to use for normalization.
        - target_size: int
            The target size for normalization.
        - apply_CPM: bool
            If True, apply CPM normalization.

        Notes:
        -----
        Ideally, the feature_key should be 'x' for the node features. The option exists to pass a custom key to allow for flexibility during training, if required.
        """
        self.norm_method = norm_method
        self.target_size = target_size
        self.apply_CPM = apply_CPM

    def forward(
            self,
            data: Data
        ) -> Data:
        """
        Normalize the features of the PyG Data object.
        """
        # if SubsetHVG is applied, the feature data to normalize is data.x
        if 'x' in data.keys():
            feature_data = data.x
        # if SubsetHVG is not applied, the feature data to normalize is the raw cell gene counts in data.x_cell_gene_counts
        else:
            feature_data = data.x_cell_gene_counts

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

        # the normalized feature data is stored in data.x
        setattr(data, 'x', feature_data)

        return data


class SetExperimentDataKeys(T.BaseTransform):
    def __init__(
            self,
            feature_names: List[Literal['X', 'U_lm_eigvecs', 'U_deepwalk', 'U_gosh']] = ['X'],
            label_name: str = 'cell_types',
            edge_index_name: str = 'spatial-delaunay',
            conditioning_sources: List[Literal['absolute_xy']] = []
        ):
        """
        Set data.x, data.y, and data.edge_index keys for the PyG Data object from Experiment keys.
        This is useful to control before starting training so we can choose on the fly which features, labels, and edge indices to use (e.g., cell gene counts for cell type classification, etc.)

        Parameters:
        ----------
        - feature_names: List[Literal['X', 'U_lm_eigvecs', 'U_deepwalk', 'U_gosh']]
            The keys for the node features to set.
        - label_name: Literal['cell_types']
            The key for the node labels to set.
        - edge_index_name: Literal['spatial-delaunay']
            The key for the edge index to set.
        - conditioning_sources: List[Literal['absolute_xy']]
            The keys for the conditioning sources to set.

        Notes:
        -----
        - X refers to the raw cell gene counts.
        - U_lm_eigvecs, U_deepwalk, U_gosh refer to the unsupervised node embeddings generated by the corresponding methods.
        - If feature_names contains multiple names, the features are concatenated along the feature dimension.
        """
        self.feature_names = feature_names
        self.label_name = label_name
        self.edge_index_name = edge_index_name
        self.conditioning_sources = conditioning_sources


    def set_node_attributes(
            self,
            data: Data,
        ) -> torch.Tensor:
        """
        Set the node attributes in the data object given the feature_name.

        Parameters:
        ----------
        - data: Data
            The data object to set the node attributes in.

        Returns:
        -------
        - torch.Tensor
            The node attributes.
        """
        feature_keys = []
        for feature_name in self.feature_names:
            if feature_name == 'X':
                # if SubsetHVG and/or NormalizeFeatures is applied, the feature data is in data.x
                if 'x' in data.keys():
                    feature_keys.append('x')
                # if SubsetHVG and/or NormalizeFeatures is not applied, the feature data is in data.x_cell_gene_counts
                else:
                    feature_keys.append('x_cell_gene_counts')
            elif feature_name in ['U_lm_eigvecs', 'U_deepwalk', 'U_gosh']:
                feature_keys.append(f"{feature_name}_{self.edge_index_name}")
            else:
                raise ValueError(f"Feature key {feature_name} not found in data.")

        x = torch.cat(
                [getattr(data, key) for key in feature_keys],
                dim=1
            )
        return x


    def set_node_labels(
            self,
            data: Data,
        ) -> torch.Tensor:
        """
        Set the node labels in the data object given the label_name.
        """
        if f"y_{self.label_name}" in data.keys():
            return getattr(data, f"y_{self.label_name}")
        else:
            raise ValueError(f"Label key {self.label_name} not found in data.")


    def set_edge_index(
            self,
            data: Data,
        ) -> torch.Tensor:
        """
        Set the edge index in the data object given the edge_index_name.
        """
        if f"edge_index_{self.edge_index_name}" in data.keys():
            return getattr(data, f"edge_index_{self.edge_index_name}")
        else:
            raise ValueError(f"Edge index key {self.edge_index_name} not found in data.")


    def set_conditioning_features(
            self,
            data: Data,
        ) -> Optional[torch.Tensor]:
        """
        Set the conditioning features in the data object given the conditioning_sources.
        """
        if len(self.conditioning_sources) > 0:
            conditioning_features = []
            for source in self.conditioning_sources:
                if source == 'absolute_xy':
                    conditioning_features.append(data.xy_coordinates)

                elif source == 'fourier_xy':
                    conditioning_features.append(
                        fourier_encode(
                            data.xy_coordinates,
                        )
                    )

                elif source == 'relative_xy':
                    centroid = data.xy_coordinates.mean(dim=0, keepdim=True)
                    rel_coords = data.xy_coordinates - centroid
                    conditioning_features.append(rel_coords)

                elif source == 'rbf_distances':
                    # Compute distance from centroid
                    centroid = data.xy_coordinates.mean(dim=0, keepdim=True)
                    dists = torch.norm(data.xy_coordinates - centroid, dim=1)  # (N,)
                    centers = torch.linspace(dists.min(), dists.max(), steps=8).to(dists.device)
                    rbf_feats = rbf_encode(dists, centers, gamma=10.0)
                    conditioning_features.append(rbf_feats)
                    
                elif source == 'cell_types':
                    conditioning_features.append(data.y)

                elif source == 'U_lm_eigvecs':
                    conditioning_features.append(getattr(data, f"U_lm_eigvecs_{self.edge_index_name}"))

                elif source == 'U_deepwalk':
                    conditioning_features.append(getattr(data, f"U_deepwalk_{self.edge_index_name}"))

                elif source == 'U_gosh':
                    conditioning_features.append(getattr(data, f"U_gosh_{self.edge_index_name}"))

            conditioning_features = torch.cat(conditioning_features, dim=-1)

            return conditioning_features
        else:
            return None


    def forward(
            self,
            data: Data
        ) -> Data:
        """
        Set Experiment data keys for the PyG Data object.
        """
        data.x = self.set_node_attributes(data)
        data.y = self.set_node_labels(data)
        data.edge_index = self.set_edge_index(data)

        conditioning_features = self.set_conditioning_features(data)
        if conditioning_features is not None:
            data.conditioning_features = conditioning_features
        print(f"{hasattr(data, 'conditioning_features')=}")
        
        data.num_features = data.x.shape[1]
        data.num_classes = data.y.shape[1]
        data.num_nodes = data.x.shape[0]
        data.num_edges = data.edge_index.shape[1]

        # delete extra features, edge indices, and embeddings from the data object to reduce memory footprint during training
        for key in list(data.keys()):
            if key.startswith('x_') or key.startswith('edge_index_') or key.startswith('U_'):
                delattr(data, key)

        return data
    
    
def fourier_encode(coords, num_freqs=6):
    """
    Sinusoidal positional encoding of 2D coordinates.
    coords: Tensor of shape (N, 2)
    returns: (N, 4 * num_freqs)
    """
    freq_bands = 2 ** torch.arange(num_freqs, device=coords.device) * torch.pi
    coords = coords.unsqueeze(-1)  # (N, 2, 1)
    sin = torch.sin(freq_bands * coords)  # (N, 2, F)
    cos = torch.cos(freq_bands * coords)  # (N, 2, F)
    enc = torch.cat([sin, cos], dim=-1)   # (N, 2, 2F)
    return enc.view(coords.size(0), -1)   # (N, 4F)

def rbf_encode(distances, centers, gamma):
    """
    RBF encoding: Gaussian basis functions centered at given values.
    distances: (N, )
    centers: (R, )
    returns: (N, R)
    """
    diff = distances.unsqueeze(1) - centers.view(1, -1)
    return torch.exp(-gamma * (diff ** 2))