import torch_geometric.transforms as T
from typing import Any, List

from ..preprocessors.normalizers import normalize_by_read_depth


def prepare_transforms(transform_list: List[str] = ['SetCustomData',
                                                    'RandomNodeSplit',
                                                    'NormalizeFeatures'],
                        val_ratio: float = 0.1,
                        test_ratio: float = 0.2,
                        feature_key: str = 'x_cell_gene_counts',
                        label_key: str = 'y_cell_types',
                        edge_index_key: str = 'edge_index_spatial-delaunay',
                        norm_method: str = 'read_depth'
                        ) -> T.Compose:
    """
    Prepare a list of PyG transforms to be applied to the dataset.

    Parameters:
    ----------
    - transform_list (List[str]): The list of transforms to be applied.
    - val_ratio (float): The ratio of validation data.
    - test_ratio (float): The ratio of test data.
    - feature_key (str): The key for the node features to set for experiments.
    - label_key (str): The key for the node labels to set for experiments.
    - edge_index_key (str): The key for the edge index to set for experiments.
    - norm_method (str): The method to use for normalization.

    Returns:
    -------
    - T.Compose: The composed PyG transforms.

    Notes:
    -----
    - The order of transforms is important. SetCustomDataKeys must be the first transform if data.x, data.y, and data.edge_index are not set.
    """
    transforms = []
    for transform in transform_list:
        if transform == 'RandomNodeSplit':
            transforms.append(T.RandomNodeSplit(
                                        split='train_rest',
                                        num_val=val_ratio,
                                        num_test=test_ratio)
                            )
        elif transform == 'NormalizeFeatures':
            transforms.append(NormalizeFeatures(norm_method=norm_method))
        elif transform == 'SetCustomDataKeys':
            transforms.append(SetCustomDataKeys(
                                    feature_key=feature_key,
                                    label_key=label_key,
                                    edge_index_key=edge_index_key)
                                    )
        else:
            raise ValueError(f"Transform {transform} not found.")

    return T.Compose(transforms)


class NormalizeFeatures(T.BaseTransform):
    def __init__(self,
                 norm_method: str = 'read_depth',
                 feature_key: str = 'x'):
        """
        Normalize the features of the PyG Data object.

        Parameters:
        ----------
        - norm_method (str): The method to use for normalization.
        - feature_key (str): The key for the node features to normalize.

        Returns:
        -------
        - None

        Notes:
        -----
        Ideally, the feature_key should be 'x' for the node features. The option exists to allow for flexibility.
        """
        self.norm_method = norm_method
        self.feature_key = feature_key

    def forward(self,
                data: Any) -> Any:
        """
        Normalize the features of the PyG Data object.
        """
        feature_data = getattr(data, self.feature_key)
        if self.norm_method == 'read_depth':
            feature_data = normalize_by_read_depth(feature_data)
        else:
            raise ValueError(f"Normalization method {self.norm_method} not found.")
        setattr(data, self.feature_key, feature_data)
        return data


class SetCustomDataKeys(T.BaseTransform):
    def __init__(self,
                 feature_key: str = 'x_cell_gene_counts',
                 label_key: str = 'y_cell_types',
                 edge_index_key: str = 'edge_index_spatial-delaunay'):
        """
        Set custom data keys for the PyG Data object.

        Parameters:
        ----------
        - feature_key (str): The key for the node features to set.
        - label_key (str): The key for the node labels to set.
        - edge_index_key (str): The key for the edge index to set.

        Returns:
        -------
        - None
        """
        self.feature_key = feature_key
        self.label_key = label_key
        self.edge_index_key = edge_index_key

    def forward(self,
                data: Any):
        """
        Set custom data keys for the PyG Data object.
        """
        data.x = getattr(data, self.feature_key)
        data.y = getattr(data, self.label_key)
        data.edge_index = getattr(data, self.edge_index_key)
        data.num_features = data.x.shape[1]
        data.num_classes = data.y.shape[1]

        delattr(data, self.feature_key)
        delattr(data, self.label_key)
        delattr(data, self.edge_index_key)

        return data