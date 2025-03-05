"""
Define `Data` to refer to a PyG Data object that is constructed from a single batch of AnnData where each batch is a collection of cells originating from the same sample. Each `Data` object will have one or more of each type of the following attribute:
- `x_<feature_name>`: Node features (e.g.,cell-gene counts or neighborhood-gene counts)
- `y_<label_name>`: Node labels (e.g., cell types or niche types)
- `edge_index_<edge_index_name>`: Edge index (e.g., spatial neighbors via Delaunay triangulation or radius-based neighbors or k-nearest neighbors)
- `metadata_<metadata_name>`: Metadata (e.g., cell ids, batch ids, tissue, etc.)
We construct these attributes from the AnnData object stored as an `.h5ad` file on disk.

Next, define `Dataset` to refer to a PyG InMemoryDataset object comprising of collection of AnnData batches, i.e. `Data` objects. From an implementation perspective, we consider the Dataset to be a "blob" because we collate (combine) the individual Data objects (built previously from AnnData batches) into a single file called `dataset_blob.pt` before saving to disk.

For example, consider creating a DatasetBlob for `xhs1000-39b_1p` (Xenium Human Skin | id = 1000 | 39 batches | 1 gene panel). When we first process this data, we create 39 Data objects, each an attributed graph representation of the corresponding batch of cells, and save them all into `dataset_blob.pt`. When we load the DatasetBlob, we load the `dataset_blob.pt` file and then access the individual Data objects as needed.

Usage:
>>> dataset_blob = InMemoryDatasetBlob(name='xhs1000-39b_1p',
                                        label_names=['cell_types'],
                                        graph_kwargs={'delaunay': True},
                                        data_directory_path='/path/to/data',
                                        transform=transform)
>>> data = dataset[<batch-index>] # get a single Data object corresponding to a batch-index (say, 7)

Tradeoffs:
-------
- One dataset_blob.pt across all batches of cells in the experiment folder vs one data.pt per batch of cells in the experiment folder.
- One dataset_blob.pt is easier to manage and load than multiple data.pt files. During training, we can load the entire dataset in one go and use one or more batches per model as needed. This is useful because the subsequent subgraph sampling and batching can be done on the fly.
- For each epoch, we subset the dataset_blob by batch(es) and then sample subgraphs from the batch(es) for training. This is more time efficient than sampling subgraphs from each batch separately, but requires more memory.
- Memory usage is a concern because the entire dataset is loaded into memory at once. This is not a problem for small datasets but can be a bottleneck for large datasets.
- For larger datasets, we have two options:
    1. Create one dataset_blob.pt per batch of cells in the experiment folder. This is more memory efficient but increases time complexity because, if we have (say) T training epochs, each AnnData batch is loaded and removed from memory T times.
    2. Create an OnDiskDatasetBlob that loads parallely in chunks.
"""
import os
import copy
from pathlib import Path
import concurrent.futures

import numpy as np
import scanpy as sc
from typing import Optional, Callable, List, Tuple

from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils.convert import from_scipy_sparse_matrix

from ..preprocessors.graph_constructors import set_edge_index_name, spatial_neighbors
from ..utils.type_conversions import sparse_mx_to_float_tensor, pandas_to_torch_one_hot


class InMemoryDatasetBlob(InMemoryDataset):

    def __init__(
            self,
            name: str = "sss2-1b_1p",
            feature_names: List['str'] = [],
            label_names: List['str'] = [],
            graph_kwargs: dict = {},
            data_directory_path: Optional[ str | Path ] = "/lustre/scratch126/cellgen/team361/DATASETS",
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            pre_filter: Optional[Callable] = None,
            overwrite: bool = False
        ) -> InMemoryDataset:
        """
        CustomInMemoryDataset class for loading PyG Data objects from AnnData files.

        Parameters:
        ----------
        - name: str
            Name of the dataset.
        - feature_names: List[str]
            List of feature names to be constructed for the dataset.
        - label_names: List[str]
            List of label names to be constructed for the dataset.
        - graph_kwargs: dict
            Dictionary containing the keyword arguments for the graph construction.
        - data_directory_path: str | Path
            Path to the data directory which contains silver and gold data.
            Silver data is the preprocessed, harmonized AnnData files.
            Gold data is the processed, in-memory data ready for training and evaluation.
        - transform: Optional[Callable]
            A function/transform that takes in an pre-processed PyG Data object and returns a transformed version ready for training.
        - pre_transform: Optional[Callable]
            A function/transform that takes in an original PyG Data object and returns a pre-processed version.
        - pre_filter: Optional[Callable]
            A function that takes in an PyG Data object and returns True if the data object should be included in the final dataset.
        - overwrite: bool
            If True, the existing processed data is overwritten and then loaded. If False, the processed data is loaded from the processed directory.

        Returns:
        -------
        - InMemoryDataset
            A PyG InMemoryDataset object containing the processed PyG Data objects.
        """
        self.name = name

        self.feature_names = feature_names
        self.label_names = label_names
        self.graph_kwargs = graph_kwargs

        # path to the data directory which contains silver and gold data
        # silver data is the preprocessed, harmonized AnnData files
        # gold data is the processed, in-memory data ready for training and evaluation
        if isinstance(data_directory_path, str):
            data_directory_path = Path(data_directory_path)
        self.data_directory_path = data_directory_path

        # root = Root directory where the processed data is saved.
        # rerun the self.process if the processed data is not found or if overwrite is set to True.
        super().__init__(
            root=self.processed_dir,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            force_reload=overwrite
        )

        # This index is set to 0 because there is only one processed data file.
        self.load(self.processed_paths[0])


    def process(self) -> None:
        """
        Process the raw (silver) data and save it to the processed (gold) directory as a PyG InMemoryDataset. This method is called either when specifically `data.pt` is not found or when overwrite (force_reload) is set to True. Otherwise, `dataset_blob.pt` is loaded directly without calling this function.
        """
        # ----------------- First Pass over AnnData Batches -----------------
        # This pass is used to collect the unique categories for each label name across all batches.
        # NOTE: So far, this only does this for labels. This needs to be extended to features in the future when we have multiple gene panels across dataset-ids.
        # QUESTION: Can we parallelize this across batches?
        self.label_categories = {
            label_name: set()
            for label_name in self.label_names
        }

        for adata_batch_file in self.raw_paths:
            adata_batch = sc.read(adata_batch_file)
            for label_name in self.label_names:
                self.label_categories[label_name].update(adata_batch.obs[label_name].unique())

        for label_name in self.label_names:
            # sorting the categories for consistency across batches so that the one-hot encoding is consistent. e.g. when the one-hot encoding is [0, 1, 0], the label name is the second name in the sorted list of that label.
            self.label_categories[label_name] = sorted(list(self.label_categories[label_name]))
            print(f"Label Name: {label_name} | {self.label_categories[label_name]=}")

        # ----------------- Second Pass over AnnData Batches -----------------
        # This pass is used to process each batch of AnnData into a PyG Data object.

        # Process one batch of AnnData into one PyG Data object
        data_batches = []

        # parallelize the processing of each batch of AnnData
        num_cores = int(os.environ.get("LSB_DJOB_NUMPROC", 1))
        num_files = len(self.raw_paths)
        max_workers = min(num_cores, num_files)
        print(f"Number of Cores: {num_cores} | Number of Batches: {num_files}")
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for adata_batch_file in self.raw_paths:
                print(f"Processing {adata_batch_file}...")
                future = executor.submit(self.process_anndata_batch, adata_batch_file)
                futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                data_batch = future.result()
                data_batches.append(copy.deepcopy(data_batch))
                print("")

                del data_batch

        # sort the data batches by batch_idx
        # batch_id = 0, 1, ..., N
        # batch_idx = 0, 1, ..., N
        # NOTE: If we have multiple dataset ids, this needs sorting by two keys simultaneously: dataset-id and batch-id.
        data_batches = sorted(data_batches, key=lambda data_batch: data_batch.adata_batch_id)
        for i, data_batch in enumerate(data_batches):
            print(f"i: {i}, Batch: {data_batch.adata_batch_id}, Type: {type(data_batch.adata_batch_id)}")

        # self.save internally collates all Data objects in data_list into a single blob.
        # The index is set to 0 so that the collated object is stored at
        # path "self.processed_dir/dataset_blob.pt".
        self.save(
            data_list=data_batches,
            path=self.processed_paths[0],
        )

        del data_batches


    def process_anndata_batch(
            self,
            adata_batch_file: Path,
        ) -> Tuple[Data, str]:
        """
        Process one single batch of AnnData into a PyG Data object.

        Parameters:
        ----------
        - adata_batch_file: Path
            Path to the batch of AnnData file.

        Returns:
        -------
        - Data
            A PyG Data object containing the processed data.
        """
        # Read the AnnData file from disk
        adata_batch = sc.read(adata_batch_file)

        # ----------------- Build Neighborhood Graphs -----------------
        self.edge_index_names = []

        coord_type = self.graph_kwargs['coord_type']
        spatial_key = self.graph_kwargs['spatial_key']
        include_self_loop = self.graph_kwargs['include_self_loop']

        for delaunay in [True, False]:
            for radius in self.graph_kwargs['radius_list'] + [None]:
                # n_neighs is used only if delaunay is False
                if delaunay:
                    n_neighs_list = [None]
                else:
                    n_neighs_list = self.graph_kwargs['n_neighs_list']

                for n_neighs in n_neighs_list:
                    # set key name for edge index based on delaunay, radius, n_neighs
                    edge_index_name = set_edge_index_name(
                                        spatial_key=spatial_key,
                                        delaunay=delaunay,
                                        n_neighs=n_neighs,
                                        radius=radius
                                    )
                    self.edge_index_names.append(edge_index_name)

                    print(f"Computing spatial neighbors with delaunay={delaunay}, radius={radius}, n_neighs={n_neighs} at key {edge_index_name}...")
                    adata_batch = spatial_neighbors(
                                            adata_batch,
                                            coord_type=coord_type,
                                            spatial_key=spatial_key,
                                            delaunay=delaunay,
                                            n_neighs=n_neighs,
                                            radius=radius,
                                            include_self_loop=include_self_loop,
                                            key_added=edge_index_name
                                )

        print("Adata:")
        print(adata_batch)

        # ----------------- Build Data Dict -----------------
        batch_dict = {}

        # build for node features (sparse csr matrix to float tensor)
        assert len(self.feature_names) > 0, "At least one feature name must be provided"
        for feature_name in self.feature_names:
            batch_dict[f'x_{feature_name}'] = sparse_mx_to_float_tensor(
                                                    adata_batch.X
                                                )

        # build for node labels (categorical pandas series to one hot tensor)
        for label_name in self.label_names:
            batch_dict[f"y_{label_name}"] = pandas_to_torch_one_hot(
                                                adata_batch.obs[label_name],
                                                categories=self.label_categories[label_name]
                                            )

        # build for edge index (sparse csr matrix to edge index style tensor)
        assert len(self.edge_index_names) > 0, "At least one edge index name must be provided"
        for edge_index_name in self.edge_index_names:
            batch_dict[f"edge_index_{edge_index_name}"] = from_scipy_sparse_matrix(
                                                        adata_batch.obsp[f"{edge_index_name}_connectivities"]
                                                        )[0]

        for key, value in batch_dict.items():
            print(f"{key}: {value.shape=}, {value.dtype=}, {type(value)=}")

        # ----------------- Add Metadata -----------------
        batch_dict['cell_id'] = adata_batch.obs['cell_id'].to_list()

        batch_dict['dataset_id'] = adata_batch.uns['dataset_id']
        batch_dict['tissue'] = adata_batch.uns['tissue']
        batch_dict['species'] = adata_batch.uns['species']
        batch_dict['adata_batch_id'] = int(adata_batch.uns['batch'][5:])
        print(f"{batch_dict['adata_batch_id']=}, {adata_batch.uns['batch']=}")

        # ----------------- Convert Data Dict to PyG Data Object -----------------
        data_batch = Data(**batch_dict)

        del adata_batch, batch_dict

        if self.pre_filter is not None:
            data_batch = self.pre_filter(data_batch)

        if self.pre_transform is not None:
            data_batch = self.pre_transform(data_batch)

        return data_batch


    @property
    def raw_dir(self) -> Path:
        """
        Return the path to the raw data directory.
        """
        silver_data_path = self.data_directory_path / 'silver'
        raw_dir = silver_data_path / self.name
        return raw_dir

    @property
    def raw_file_names(self) -> List[Path]:
        """
        Return the list of raw file names.
        """
        file_names = list(self.raw_dir.glob('**/*.h5ad'))
        if len(file_names) == 0:
            raise FileNotFoundError("No AnnData files found")
        return file_names

    @property
    def raw_paths(self) -> List[Path]:
        """
        The absolute filepaths that must be present.
        """
        return self.raw_file_names

    @property
    def processed_dir(self) -> str:
        """
        Return the path to the processed data directory.
        """
        gold_data_path = self.data_directory_path / 'gold'
        processed_dir = str(gold_data_path / 'in-memory-PyG-dataset-blob' / self.name)
        return processed_dir

    @property
    def processed_file_names(self) -> List[Data]:
        """
        Return the list of processed file names.
        """
        return ['dataset_blob.pt']