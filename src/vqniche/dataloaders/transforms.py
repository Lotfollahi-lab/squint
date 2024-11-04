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
                        ) -> T.Compose:

    transforms = []
    for transform in transform_list:
        if transform == 'RandomNodeSplit':
            transforms.append(T.RandomNodeSplit(
                                        split='train_rest',
                                        num_val=val_ratio,
                                        num_test=test_ratio)
                            )
        elif transform == 'NormalizeByReadDepth':
            transforms.append(NormalizeByReadDepth())
        elif transform == 'SetCustomDataKeys':
            transforms.append(SetCustomDataKeys(
                                    feature_key=feature_key,
                                    label_key=label_key,
                                    edge_index_key=edge_index_key)
                                    )
        else:
            raise ValueError(f"Transform {transform} not found.")

    return T.Compose(transforms)


class NormalizeByReadDepth(T.BaseTransform):
    def __init__(self,
                 feature_key: str = 'x'):
        self.feature_key = feature_key

    def forward(self,
                data: Any):
        feature_data = getattr(data, self.feature_key)
        feature_data = normalize_by_read_depth(feature_data)
        setattr(data, self.feature_key, feature_data)
        return data


class SetCustomDataKeys(T.BaseTransform):
    def __init__(self,
                 feature_key: str = 'x_cell_gene_counts',
                 label_key: str = 'y_cell_types',
                 edge_index_key: str = 'edge_index_spatial-delaunay'):
        self.feature_key = feature_key
        self.label_key = label_key
        self.edge_index_key = edge_index_key

    def forward(self,
                data: Any):
        data.x = getattr(data, self.feature_key)
        data.y = getattr(data, self.label_key)
        data.edge_index = getattr(data, self.edge_index_key)
        data.num_features = data.x.shape[1]
        data.num_classes = data.y.shape[1]

        delattr(data, self.feature_key)
        delattr(data, self.label_key)
        delattr(data, self.edge_index_key)

        return data