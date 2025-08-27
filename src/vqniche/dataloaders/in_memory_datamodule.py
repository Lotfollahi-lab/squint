"""
This class extends the Pytorch Lightning's LightningDataModule class to provide a more flexible way to create PyTorch Lightning DataModules that are compatible with PyTorch Geometric for our use case wherein we split the Pytorch Geometric GNN models into separate encoder and predictor submodules.
This has two main advantages:
1. We can inherit all the nice properties of the parent classes including `setup`, `prepare_data` and `infer train-val-test` nodes. These can also be overriden in the future if necessary.
2. We can define our custom train, validation, test, and inference dataloaders that are compatible with PyTorch Geometric's DataLoader and Sampler classes.

Initializing this class requires defining a Loader and a Sampler along with their corresponding parameters.

The `loader_name` argument is used to define the DataLoader class that will be used to load the data. Currently, we support the following:
- `FullLoader`: This option provides backwards compatibility with the default `LightningNodeData` class that uses the full graph for training.
- `DefaultNodeLoader`: This option provides backwards compatibility with the default `LightningNodeData` class that uses the default `NodeLoader` along with `NeighborSampler` from PyTorch Geometric.
- NeighborLoader: This option uses the `NeighborLoader` class from PyTorch Geometric. Ignores the loader and sampler related setup in the parent class.
The `loader_params` argument is a dictionary that can be used to define the arguments that will be passed to the DataLoader class. We mainly focus on `batch_size` for now.
Use `FullLoader` and `DefaultNodeLoader` only if the default `LightningNodeData` class is sufficient for your use case. Otherwise, define and use your own custom Loader and Sampler.

The `sampler_name` argument is used to define the Sampler class that will be used to sample the subgraphs. Currently, we support the following:
- `NeighborSampler`: This option uses the `NeighborSampler` class from PyTorch Geometric.
The `sampler_params` argument is a dictionary that can be used to define the arguments that will be passed to the Sampler class. We mainly focus on `num_neighbors` for now.

KEY INFO:
---> The Loader + Sampler combination is kept the same across training, validation and testing.
---> The input data to this class must be a single PyTorch Geometric Data object. If the experiment requires multiple tissue sections, the data must be concatenated before being passed to this class.
---> The `infer_dataloader` method is provided to allow for looping over all the nodes in the graph unlike the training, validation, and testing dataloaders which only loop over the nodes in the training, validation, and test sets respectively.
---> The `sample_neighbors_for_inference` parameter is used to control whether the neighbors are sampled during inference or not. This parameter is the same for validation and testing and is ignored for training.
"""
import os
from typing import Literal, Optional, Callable

from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.loader import NodeLoader, NeighborLoader
from torch_geometric.sampler import NeighborSampler
from torch_geometric.data.lightning import LightningNodeData

NUM_CORES = 1
NUM_WORKERS = 1
BATCH_SIZE = 1024
NUM_NEIGHBORS = [5, 5]


class InMemoryDataModule(LightningNodeData):
    def __init__(
            self,
            data: Data,
            loader_name: Literal['DefaultFullLoader', 'DefaultNodeLoader', 'NeighborLoader'] = 'NeighborLoader',
            loader_params: Optional[dict] = {},
            sampler_name: Optional[Literal['NeighborSampler']] = 'NeighborSampler',
            sampler_params: Optional[dict] = {},
            sample_neighbors_for_inference: bool = False,
        ) -> None:
        """
        This function initializes the InMemoryDataModule class with the given DataLoader and Sampler classes along with their corresponding parameters.

        Parameters:
        -----------
        - data: Data
            A single PyTorch Geometric Data object that contains the graph data.
        - loader_name: Literal['DefaultFullLoader', 'DefaultNodeLoader', 'NeighborLoader']
            Name of the DataLoader class that will be used to load the data.
        - loader_params: Optional[dict]
            Dictionary that contains the arguments that will be passed to the DataLoader class, e.g. `batch_size`.
        - sampler_name: Optional[Literal['NeighborSampler']]
            Name of the Sampler class that will be used to sample the subgraphs.
        - sampler_params: Optional[dict]
            Dictionary that contains the arguments that will be passed to the Sampler class, e.g. `num_neighbors`.
        - sample_neighbors_for_inference: bool
            Whether to sample neighbors for inference or not. This is implemented by setting the number of neighbors to the maximum number of neighbors for inference.
        """
        assert isinstance(data, Data), f"data must be of type torch_geometric.data.Data, but got {type(data)}."

        # get the number of available cores
        num_cores_available = int(os.environ.get(
                                "LSB_DJOB_NUMPROC",
                                NUM_CORES
                                ))

        # Loaders will be instantiated in self.train_dataloader(), self.val_dataloader(), and self.test_dataloader()
        print(f"Loader Name: {loader_name}")

        # setting parameters for backward compatibility with the default LightningNodeData implementation of full graph training
        if loader_name == 'DefaultFullLoader':
            # set num_workers to 0 for FullLoader
            num_workers = 0

            # set loader_type to 'full' for FullLoader
            loader_type = 'full'
            loader_class: Callable = DataLoader

            # batch_size = 1 is the only parameter for FullLoader
            loader_params = {'batch_size': 1}

            # sampler must be set to None for FullLoader
            sampler_name = None
            sampler_class: Callable = None

            # sampler_params must be set to an empty dictionary for FullLoader
            sampler_params = {}

        # setting parameters for backward compatibility with the default LightningNodeData implementation of NodeLoader + NeighborSampler
        elif loader_name == 'DefaultNodeLoader':
            # set num_workers to half of the available cores
            num_workers = max(num_cores_available // 2, NUM_WORKERS)

            # set loader_type to 'neighbor' for DefaultNodeLoader
            loader_type = 'neighbor'
            loader_class: Callable = NodeLoader

            # set batch_size to BATCH_SIZE if not provided in loader_params and reset loader_params to only include batch_size
            loader_params['batch_size'] = loader_params.get(
                                            'batch_size',
                                            BATCH_SIZE
                                            )
            loader_params = {'batch_size': loader_params['batch_size']}

            # LightningNodeData will reset node_sampler to torch_geometric.loader.NeighborSampler and initialize it with the given sampler kwargs.
            # this will be gettable via the self.graph_sampler attribute.
            sampler_name = None
            sampler_class: Callable = None

            # set num_neighbors to NUM_NEIGHBORS if not provided in sampler_params and reset sampler_params to only include num_neighbors
            sampler_params['num_neighbors'] = sampler_params.get(
                                                'num_neighbors',
                                                NUM_NEIGHBORS
                                                )
            sampler_params = {'num_neighbors': sampler_params['num_neighbors']}

        # setting parameters for custom Loader and Sampler
        else:
            # set num_workers to half of the available cores
            num_workers = max(num_cores_available // 2, NUM_WORKERS)

            # train_loader will be set to a custom Callable in self.train_dataloader()
            # this is necessary to indicate to LightningNodeData that the train_loader is a custom DataLoader
            loader_type = 'custom'
            loader_class: Callable = self.set_custom_loader_class(loader_name)

            # set batch_size to BATCH_SIZE if not provided in loader_params
            loader_params['batch_size'] = loader_params.get(
                                            'batch_size',
                                            BATCH_SIZE
                                            )

            # set node_sampler to be a Callable of type torch_geometric.sampler.BaseSampler here before super.__init__() is called.
            # cannot be None because LightningNodeData will throw an error when loader_type is `custom`.
            # this will be gettable via the self.graph_sampler attribute (but we don't use it)
            sampler_class: Callable = self.set_custom_sampler_class(sampler_name)

            # set num_neighbors to NUM_NEIGHBORS if not provided in sampler_params
            sampler_params['num_neighbors'] = sampler_params.get(
                                                'num_neighbors',
                                                NUM_NEIGHBORS
                                                )

        # set the backend class attributes
        self.num_workers = num_workers
        print(f"Num Workers: {num_workers}")

        # set the loader attributes
        self.loader_name = loader_name
        self.loader_class = loader_class
        self.loader_params = loader_params
        print(f"Loader Name: {self.loader_name}")
        print(f"Loader Class: {self.loader_class}")
        print(f"Loader Params: {self.loader_params}")

        # set the sampler attributes
        self.sampler_name = sampler_name
        self.sampler_class = sampler_class
        self.sampler_params = sampler_params
        print(f"Sampler Name: {self.sampler_name}")
        print(f"Sampler Class: {self.sampler_class}")
        print(f"Sampler Params: {self.sampler_params}")

        # set whether to sample neighbors for inference
        self.sample_neighbors_for_inference = sample_neighbors_for_inference
        print(f"Sample Neighbors for Inference: {self.sample_neighbors_for_inference}")

        # build the data dictionary for the loader after data transforms
        base_data = {
            'x': data.x,
            'y': data.y,
            'edge_index': data.edge_index,
            'xy_coordinates': data.xy_coordinates,
            'train_mask': data.train_mask,
            'val_mask': data.val_mask,
            'test_mask': data.test_mask,
            'adata_batch_ids': data.adata_batch_ids,
        }
        
        optional_data_keys = [
            'y_cell_types',
            'y_niche_types',
            'encoder_conditions',
            'attr_decoder_conditions',
            'adj_decoder_conditions',
        ]
        
        data_dict_for_loader = {
            **base_data,
            **{attr: getattr(data, attr) for attr in optional_data_keys if getattr(data, attr, None) is not None}
        }
        
        data_for_loader = Data(**data_dict_for_loader)

        # keep all other keys that start with 'y_'
        for key in list(data.keys()):
            if key.startswith('y_'):
                setattr(data_for_loader, key, data[key])

        # call the parent class constructor
        super().__init__(
            data=data_for_loader,
            num_workers=self.num_workers,
            loader=loader_type,
            node_sampler=self.sampler_class,
            **self.loader_params,
            **self.sampler_params
        )


    def set_custom_loader_class(
            self,
            custom_loader_name: str
        ) -> Callable:
        """
        This function sets the DataLoader class that will be used to load the data when loader_type is `custom`.

        Parameters:
        -----------
        - custom_loader_name: str
            Name of the DataLoader class that will be used to load the data.

        Returns:
        --------
        - Callable
            DataLoader class that will be used to load the data.
        """
        if custom_loader_name == 'NeighborLoader':
            return NeighborLoader
        else:
            raise NotImplementedError(f"{custom_loader_name} not implemented.")


    def set_custom_sampler_class(
            self,
            custom_sampler_name: str
        ) -> Callable:
        """
        This function sets the Sampler class that will be used to sample the subgraphs when loader_type is `custom`.

        Parameters:
        -----------
        - custom_sampler_name: str
            Name of the Sampler class that will be used to sample the subgraphs.

        Returns:
        --------
        - Callable
            Sampler class that will be used to sample the subgraphs.
        """
        if custom_sampler_name == 'NeighborSampler':
            return NeighborSampler
        else:
            raise NotImplementedError(f"{custom_sampler_name} not implemented.")


    def train_dataloader(self):
        """
        This function constructs the DataLoader object for training based on the settings defined in the constructor of the InMemoryDataModule class. If the loader_name is set to 'DefaultFullLoader' or 'DefaultNodeLoader', the function will call the parent class' train_dataloader() function. Otherwise, it will instantiate `self.loader_class` with `self.sampler_class`.

        Notes:
        ------
        - `sample_neighbors_for_inference` is ignored for training.
        """
        if self.loader_name in ['DefaultFullLoader', 'DefaultNodeLoader']:
            return super().train_dataloader()

        else:
            # instantiate the sampler class for training
            train_sampler = self.sampler_class(
                            data=self.data,
                            **self.sampler_params,
                        )

            # instantiate the loader class for training
            train_loader = self.loader_class(
                                        data=self.data,
                                        num_workers=self.num_workers,
                                        input_nodes=self.input_train_nodes,
                                        neighbor_sampler=train_sampler,
                                        shuffle=False,
                                        **self.loader_params,
                                        **self.sampler_params,
                                    )
            return train_loader


    def val_dataloader(self):
        """
        This function constructs the DataLoader object for validation based on the settings defined in the constructor of the InMemoryDataModule class. If the loader_name is set to 'DefaultFullLoader' or 'DefaultNodeLoader', the function will call the parent class' val_dataloader() function. Otherwise, it will instantiate `self.loader_class` with `self.sampler_class`.

        Notes:
        ------
        - If `sample_neighbors_for_inference` is set to True, the neighbors are sampled during inference.
        - If `sample_neighbors_for_inference` is set to False, the neighbors are not sampled during inference.
        - `val_dataloader` loops over nodes in the validation set.
        """
        if self.loader_name in ['DefaultFullLoader', 'DefaultNodeLoader']:
            return super().val_dataloader()

        else:
            # reuse user-defined sampler parameters
            sampler_params = self.sampler_params

            if not self.sample_neighbors_for_inference:
                # update num_neighbors to -1 so that the neighbors are not sampled.
                sampler_params['num_neighbors'] = [-1] * len(self.sampler_params['num_neighbors'])

            # instantiate the sampler class for validation
            val_sampler = self.sampler_class(
                                data=self.data,
                                **sampler_params,
                            )

            # instantiate the loader class for validation
            val_loader = self.loader_class(
                                data=self.data,
                                num_workers=self.num_workers,
                                input_nodes=self.input_val_nodes,
                                neighbor_sampler=val_sampler,
                                shuffle=False,
                                **self.loader_params,
                                **sampler_params,
                            )
            return val_loader


    def test_dataloader(self):
        """
        This function constructs the DataLoader object for testing based on the settings defined in the constructor of the InMemoryDataModule class. If the loader_name is set to 'DefaultFullLoader' or 'DefaultNodeLoader', the function will call the parent class' test_dataloader() function. Otherwise, it will instantiate `self.loader_class` with `self.sampler_class`.

        Notes:
        ------
        - If `sample_neighbors_for_inference` is set to True, the neighbors are sampled during inference.
        - If `sample_neighbors_for_inference` is set to False, the neighbors are not sampled during inference.
        - `test_dataloader` loops over nodes in the test set.
        """
        if self.loader_name in ['DefaultFullLoader', 'DefaultNodeLoader']:
            return super().test_dataloader()

        else:
            # reuse user-defined sampler parameters
            sampler_params = self.sampler_params

            if not self.sample_neighbors_for_inference:
                # update num_neighbors to -1 so that the neighbors are not sampled.
                sampler_params['num_neighbors'] = [-1] * len(self.sampler_params['num_neighbors'])

            # instantiate the sampler class for testing
            test_sampler = self.sampler_class(
                                data=self.data,
                                **sampler_params,
                            )

            # instantiate the loader class for testing
            test_loader = self.loader_class(
                                data=self.data,
                                num_workers=self.num_workers,
                                input_nodes=self.input_test_nodes,
                                neighbor_sampler=test_sampler,
                                shuffle=False,
                                **self.loader_params,
                                **sampler_params,
                            )
            return test_loader


    def predict_dataloader(self):
        """
        This method returns a DataLoader object for all the nodes in the graph.
        This is required for looping over all the nodes in the graph for inference.

        Notes:
        ------
        - If `sample_neighbors_for_inference` is set to True, the neighbors are sampled during inference.
        - If `sample_neighbors_for_inference` is set to False, the neighbors are not sampled during inference.
        - `predict_dataloader` loops over all the nodes in the graph.
        """
        # reuse user-defined sampler parameters
        sampler_params = self.sampler_params

        # update num_neighbors to -1 so that the neighbors are not sampled.
        if not self.sample_neighbors_for_inference:
            sampler_params['num_neighbors'] = [-1] * len(self.sampler_params['num_neighbors'])

        # instantiate the sampler class to control the behavior of the loader
        infer_sampler = self.sampler_class(
                            data=self.data,
                            **sampler_params,
                        )

        # instantiate the loader class for inference
        # input_nodes is set to None. That is, all nodes are used.
        infer_loader = self.loader_class(
                                    data=self.data,
                                    num_workers=self.num_workers,
                                    input_nodes=None,
                                    neighbor_sampler=infer_sampler,
                                    shuffle=False,
                                    **self.loader_params,
                                    **sampler_params,
                                )
        return infer_loader
