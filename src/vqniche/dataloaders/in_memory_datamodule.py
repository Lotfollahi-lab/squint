"""
Extends the LightningNodeData class for two reasons:
1. So that we can inherit all the nice properties of the parent classes including setup, prepare_data and infer train-val-test nodes
2. To provide a more flexible way to create PyTorch Lightning DataModules that are compatible with PyTorch Geometric for our use case wherein we split the Pytorch Geometric GNN models into separate encoder and predictor submodules.
    - This is useful for using a custom Loader for validation and testing and a different Loader+Sampler combination for training.

The customization is controlled by the train_loader_type parameter.

- If train_loader_type is 'full', we pass loader='full' and node_sampler=None so that then this class implements the full dataloader from the parent class.

- If train_loader_type is 'neighbor', we pass loader='neighbor' and node_sampler=None, so that then this class defaults to using the NeighborSampler as implemented in the LightningNodeData class. And we override the loader from NodeLoader in the parent class to NeighborLoader.

- We use the third option 'custom' to allow for any other combination of Loader and Sampler. This is not implemented yet.
"""
from torch_geometric.loader import NeighborLoader
from torch_geometric.sampler import NeighborSampler
from torch_geometric.data.lightning import LightningNodeData

from typing import Optional
from torch_geometric.data import Data


class InMemoryDataModule(LightningNodeData):
    def __init__(
        self,
        num_cores: int = 1,
        data: Data = None,
        train_loader_name: Optional[str] = 'DefaultNodeLoader',
        train_loader_params: Optional[dict] = {'batch_size': 1024},
        train_sampler_name: Optional[str] = 'NeighborSampler',
        train_sampler_params: Optional[dict] = {'num_neighbors': [25, 10]},
        val_loader_name: str = 'NeighborLoader',
        test_loader_name: str = 'NeighborLoader',
        use_full_graph_for_inference: bool = True,
        **kwargs
    ) -> None:
        # keep only the necessary data for the loaders
        # this is necessary because metadata such as cell-ids cannot be used because loaders require tensors that don't have dtype as strings
        data_for_loader = Data(
                            x=data.x,
                            edge_index=data.edge_index,
                            y=data.y,
                        )

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
            batch_size = train_loader_params.get('batch_size', 1024)

            # set num_workers to half of the available cores
            num_workers = max(num_cores // 2, 1)

            # set train_sampler_name to 'NeighborSampler'
            train_sampler_name = 'NeighborSampler'

            # train_sampler_kwargs remain unchanged
            num_neighbors = train_sampler_params.get('num_neighbors', [25, 10])

            # Parent Class in Lightning will reset node_sampler to torch_geometric.loader.NeighborSampler and initialize it with the given sampler kwargs.
            # This will be gettable via the self.graph_sampler attribute.
            self.train_sampler = None

        else:
            # for all other cases, set train_loader_type to 'custom'
            # train_loader will be set to a Callable of type torch_geometric.loader.DataLoader in self.train_dataloader()
            train_loader_type = 'custom'

            # set batch_size to the given value
            batch_size = train_loader_params.get('batch_size', 1024)

            # set loader_kwargs to be a dictionary of arguments to be passed to the train_loader
            num_workers = max(num_cores // 2, 1)

            assert train_sampler_name is not None, "train_sampler_name must be provided for train_loader_type='custom'."

            num_neighbors = train_sampler_params.get('num_neighbors', [25, 10])

            # set node_sampler to be a Callable of type torch_geometric.sampler.BaseSampler here before super.__init__() is called. Cannot be None.
            # This will be gettable via the self.graph_sampler attribute.
            # This is necessary because the parent class will reset node_sampler to None if train_loader_type is 'full' or 'neighbor'.
            if train_sampler_name == 'NeighborSampler':
                self.train_sampler = NeighborSampler
            else:
                raise NotImplementedError(f"Train Sampler {train_sampler_name} not implemented.")

        self.train_loader_type = train_loader_type
        self.train_loader_name = train_loader_name
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_sampler_name = train_sampler_name
        self.num_neighbors = num_neighbors

        self.val_loader_name = val_loader_name
        self.test_loader_name = test_loader_name
        self.use_full_graph_for_inference = use_full_graph_for_inference

        self.train_loader_params = train_loader_params
        self.train_sampler_params = train_sampler_params

        super().__init__(
            data=data_for_loader,
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
        shuffle = self.train_loader_params.get('shuffle', False)
        return shuffle


    def train_dataloader(self):
        if self.train_loader_type in ['full', 'neighbor']:
            return super().train_dataloader()

        elif self.train_loader_type == 'custom':
            if self.train_loader_name == 'NeighborLoader':
                # NeighborLoader + NeighborSampler
                # setting neighbor_sampler to None inherently sets the neighbor_sampler = NeighborSampler in the case of NeighborLoader
                train_subgraph_loader = NeighborLoader(
                                            data=self.data,
                                            input_nodes=self.input_train_nodes,
                                            batch_size=self.batch_size,
                                            num_workers=self.num_workers,
                                            neighbor_sampler=None,
                                            num_neighbors=self.num_neighbors,
                                        )
                return train_subgraph_loader

            else:
                raise NotImplementedError(f"{self.train_loader_name} for Training not implemented.")


    def val_dataloader(self):
        num_neighbors = None if self.use_full_graph_for_inference else self.num_neighbors

        if self.val_loader_name == 'DefaultNodeLoader':
            return super().val_dataloader()

        elif self.val_loader_name == 'NeighborLoader':
            # NeighborLoader + NeighborSampler
            # setting input_nodes to self.input_val_nodes ensures that the validation accuracy is computed only on the validation nodes
            # setting neighbor_sampler to None inherently sets the neighbor_sampler = NeighborSampler in the case of NeighborLoader
            # setting num_neighbors to None ensures that the full graph is used for inference during validation of the nodes in the validation set
            val_subgraph_loader = NeighborLoader(
                                        data=self.data,
                                        input_nodes=self.input_val_nodes,
                                        batch_size=self.batch_size,
                                        num_workers=self.num_workers,
                                        neighbor_sampler=None,
                                        num_neighbors=num_neighbors,
                                    )
            return val_subgraph_loader

        else:
            raise NotImplementedError(f"{self.val_loader_name} for Validation not implemented.")


    def test_dataloader(self):
        num_neighbors = None if self.use_full_graph_for_inference else self.num_neighbors

        if self.test_loader_name == 'DefaultNodeLoader':
            return super().test_dataloader()

        elif self.test_loader_name == 'NeighborLoader':
            # NeighborLoader + NeighborSampler
            test_subgraph_loader = NeighborLoader(
                                        data=self.data,
                                        input_nodes=self.input_test_nodes,
                                        batch_size=self.batch_size,
                                        num_workers=self.num_workers,
                                        neighbor_sampler=None,
                                        num_neighbors=num_neighbors,
                                    )
            return test_subgraph_loader

        else:
            raise NotImplementedError(f"{self.test_loader_name} for Testing not implemented.")