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
import torch
import pytorch_lightning as pl
from torch_geometric.loader import NeighborLoader
from torch_geometric.data.lightning import LightningNodeData

from typing import Union, Optional, Callable
from torch_geometric.data import Data


class InMemoryDataModule(LightningNodeData):
    def __init__(
        self,
        data_batch: Data,
        num_cores: int = 1,
        train_loader_type: str = 'neighbor',
        train_loader_name: Optional[str] = 'NeighborLoader',
        train_loader_kwargs: Optional[dict] = None,
        train_sampler_name: Optional[str] = 'NeighborSampler',
        train_sampler_kwargs: Optional[dict] = None,
        **kwargs
    ) -> None:
        if train_loader_type == 'full':
            assert train_loader_name == 'FullLoader', f"train_loader_name must be 'FullLoader' for train_loader_type='full'."

            # batch size is set to 1 to avoid memory issues
            train_loader_kwargs['batch_size'] = 1
            # num_workers is set to 0 to avoid DataLoader issues
            train_loader_kwargs['num_workers'] = 0

            assert train_sampler_name is None, f"train_sampler_name must be None for train_loader_type='full'."

            # sampler_kwargs must be set to None
            self.train_sampler_kwargs = {}

            # node_sampler must be set to None
            self.train_sampler = None

        elif train_loader_type == 'neighbor':
            assert train_loader_name == 'NeighborLoader', f"train_loader_name must be 'NeighborLoader' for train_loader_type='neighbor'."

            # batch size and other kwargs remain unchanged
            # set num_workers to half of the available cores
            num_workers = max(num_cores // 2, 1)
            train_loader_kwargs['num_workers'] = num_workers

            # NOTE: Only NeighborSampler is supported for now with train_loader_type='neighbor'
            assert train_sampler_name == 'NeighborSampler', f"train_sampler_name must be 'NeighborSampler' for train_loader_type='neighbor'."

            # train_sampler_kwargs remain unchanged
            self.train_sampler_kwargs = train_sampler_kwargs

            # Parent Class in Lightning will reset node_sampler to torch_geometric.loader.NeighborSampler and initialize it with the given sampler kwargs.
            # This will be gettable via the self.graph_sampler attribute.
            self.train_sampler = None

        elif train_loader_type == 'custom':
            # train_loader to be a callable function of type torch_geometric.loader.DataLoader in self.train_dataloader() below
            # set loader_kwargs to be a dictionary of arguments to be passed to the train_loader
            # set node_sampler to be a callable function of type torch_geometric.sampler.BaseSampler. Cannot be None.
            # if sampler_name == 'GraphSAINTSampler':
                # self.train_sampler = ...
            raise NotImplementedError("Custom train loader not implemented.")
        else:
            raise ValueError(f"Train Loader type {train_loader_type} not found.")

        self.train_loader_type = train_loader_type
        self.train_loader_kwargs = train_loader_kwargs
        self.train_loader_name = train_loader_name
        self.train_sampler_name = train_sampler_name

        super().__init__(
            data=data_batch,
            loader=self.train_loader_type,
            node_sampler=self.train_sampler,
            **train_loader_kwargs,
            **train_sampler_kwargs,
            **kwargs
        )

        print(f"Train Loader Type: {self.train_loader_type}")
        print(f"Train Loader Name: {self.train_loader_name}")
        print(f"Train Sampler Name: {self.train_sampler_name}")


    def train_dataloader(self):
        if self.train_loader_type == 'full':
            return super().train_dataloader()
        elif self.train_loader_type == 'neighbor':
            return NeighborLoader(
                data=self.data,
                input_nodes=self.input_train_nodes,
                input_time=self.input_train_time,
                input_id=self.input_train_id,
                neighbor_sampler=getattr(self, 'graph_sampler', None),
                shuffle=self.train_shuffle,
                **self.train_loader_kwargs,
                **self.train_sampler_kwargs
            )
        elif self.train_loader_type == 'custom':
            raise NotImplementedError("Custom train loader not implemented.")


    def val_dataloader(self):
        # NeighborLoader + NeighborSampler
        val_subgraph_loader = NeighborLoader(
                                data=self.data,
                                input_nodes=None,
                                input_time=self.input_val_time,
                                input_id=self.input_val_id,
                                shuffle=False,
                                **self.train_loader_kwargs,
                                **self.train_sampler_kwargs
                            )
        return val_subgraph_loader


    def test_dataloader(self):
        # NeighborLoader + NeighborSampler
        test_graph_loader = NeighborLoader(
                                data=self.data,
                                input_nodes=None,
                                input_time=self.input_test_time,
                                input_id=self.input_test_id,
                                shuffle=False,
                                **self.train_loader_kwargs,
                                **self.train_sampler_kwargs
                            )
        return test_graph_loader
