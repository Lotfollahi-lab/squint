"""
Extends the LightningNodeData class for two reasons:
1. So that we can inherit all the nice properties of the parent classes including setup, prepare_data and infer train-val-test nodes
2. To provide a more flexible way to create PyTorch Lightning DataModules that are compatible with PyTorch Geometric for our use case we split the Pytorch Geometric GNN models into separate encoder and predictor submodules.
    - This is useful for using a custom Loader for validation and testing and a different Loader+Sampler combination for training.

The customization is controlled by the train_loader_type parameter.

- If train_loader_type is 'full', we pass loader='full' and node_sampler=None so that then this class implements the full dataloader from the parent class.

- If train_loader_type is 'neighbor', we pass loader='neighbor' and node_sampler=None, so that then this class defaults to using the NeighborSampler as implemented in the LightningNodeData class. And we override the loader from NodeLoader in the parent class to NeighborLoader.

- We use the third option 'custom' to allow for any other combination of Loader and Sampler. This is not implemented yet.
"""
from torch_geometric.loader import NeighborLoader
from torch_geometric.data.lightning import LightningNodeData

from typing import Optional, List
from torch_geometric.data import Data


class InMemoryDataModule(LightningNodeData):
    def __init__(
        self,
        num_cores: int = 1,
        data: Data = None,
        train_loader_name: Optional[str] = 'DefaultNodeLoader',
        batch_size: Optional[int] = 1024,
        num_workers: Optional[int] = 2,
        train_sampler_name: Optional[str] = 'NeighborSampler',
        num_neighbors: Optional[List[int]] = [25, 10],
        val_loader_name: str = 'NeighborLoader',
        test_loader_name: str = 'NeighborLoader',
        use_full_graph_for_inference: bool = True,
        **kwargs
    ) -> None:
        if train_loader_name == 'FullLoader':
            # set train_loader_type to 'full' to use the default DataLoader
            train_loader_type = 'full'
            # batch size is set to 1 to avoid memory issues
            batch_size = 1
            # num_workers is set to 0 to avoid DataLoader issues
            num_workers = 0

            assert train_sampler_name is None, f"train_sampler_name must be None for train_loader_name='FullLoader'."

            # sampler_kwargs must be set to None
            num_neighbors = None

            # set self.train_sampler Callable to None
            self.train_sampler = None

        elif train_loader_name == 'DefaultNodeLoader':
            # set train_loader_type to 'neighbor' to use the default NodeLoader with NeighborSampler
            train_loader_type = 'neighbor'
            # batch size and other kwargs remain unchanged
            # set num_workers to half of the available cores
            num_workers = max(num_cores // 2, 1)

            # set train_sampler_name to 'NeighborSampler'
            train_sampler_name = 'NeighborSampler'

            # train_sampler_kwargs remain unchanged

            # Parent Class in Lightning will reset node_sampler to torch_geometric.loader.NeighborSampler and initialize it with the given sampler kwargs.
            # This will be gettable via the self.graph_sampler attribute.
            self.train_sampler = None

        else:
            # for all other cases, set train_loader_type to 'custom'
            train_loader_type = 'custom'
            # train_loader will be set to a Callable of type torch_geometric.loader.DataLoader in self.train_dataloader() below
            # set loader_kwargs to be a dictionary of arguments to be passed to the train_loader

            assert train_sampler_name is not None, "train_sampler_name must be provided for train_loader_type='custom'."
            # set train_sampler_kwargs
            # set node_sampler to be a Callable of type torch_geometric.sampler.BaseSampler here before super.__init__() is called. Cannot be None.
            # if train_sampler_name == 'GraphSAINTSampler':
                # self.train_sampler = ...
            raise NotImplementedError("Custom train loader not implemented.")

        self.train_loader_type = train_loader_type
        self.train_loader_name = train_loader_name
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_sampler_name = train_sampler_name
        self.num_neighbors = num_neighbors

        self.val_loader_name = val_loader_name
        self.test_loader_name = test_loader_name
        self.use_full_graph_for_inference = use_full_graph_for_inference

        self.train_loader_kwargs = kwargs

        super().__init__(
            data=data,
            loader=self.train_loader_type,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            node_sampler=self.train_sampler,
            num_neighbors=self.num_neighbors,
            **kwargs
        )

        print(f"Train Loader Type: {self.train_loader_type}")
        print(f"Train Loader Name: {self.train_loader_name}")
        print(f"Train Sampler Name: {self.train_sampler_name}")


    @property
    def train_shuffle(self) -> bool:
        shuffle = self.train_loader_kwargs.get('shuffle', False)
        return shuffle


    def train_dataloader(self):
        if self.train_loader_type in ['full', 'neighbor']:
            return super().train_dataloader()

        elif self.train_loader_type == 'custom':

            if self.train_loader_name == 'NeighborLoader':
                return NeighborLoader(
                    data=self.data,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    input_nodes=self.input_train_nodes,
                    input_time=self.input_train_time,
                    input_id=self.input_train_id,
                    neighbor_sampler=getattr(self, 'graph_sampler', None),
                    num_neighbors=self.num_neighbors,
                    shuffle=self.train_shuffle,
                )

            else:
                raise NotImplementedError(f"Train {self.train_loader_name} not implemented.")


    def val_dataloader(self):
        input_nodes = None if self.use_full_graph_for_inference else self.input_val_nodes

        if self.val_loader_name == 'DefaultNodeLoader':
            return super().val_dataloader()

        elif self.val_loader_name == 'NeighborLoader':
            # NeighborLoader + NeighborSampler
            val_subgraph_loader = NeighborLoader(
                                    data=self.data,
                                    input_nodes=input_nodes,
                                    batch_size=self.batch_size,
                                    num_workers=self.num_workers,
                                    neighbor_sampler=getattr(self, 'graph_sampler', None),
                                    num_neighbors=self.num_neighbors,
                                )
            return val_subgraph_loader

        else:
            raise NotImplementedError(f"Val {self.val_loader_name} not implemented.")


    def test_dataloader(self):
        input_nodes = None if self.use_full_graph_for_inference else self.input_test_nodes

        if self.test_loader_name == 'DefaultNodeLoader':
            return super().test_dataloader()

        elif self.test_loader_name == 'NeighborLoader':
            # NeighborLoader + NeighborSampler
            test_subgraph_loader = NeighborLoader(
                                    data=self.data,
                                    input_nodes=input_nodes,
                                    batch_size=self.batch_size,
                                    num_workers=self.num_workers,
                                    neighbor_sampler=getattr(self, 'graph_sampler', None),
                                    num_neighbors=self.num_neighbors,
                                )
            return test_subgraph_loader

        else:
            raise NotImplementedError(f"Test {self.test_loader_name} not implemented.")