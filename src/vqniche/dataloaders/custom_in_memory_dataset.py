import scanpy as sc
import anndata as ad
from pathlib import Path
from typing import Optional, Callable, List

from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils.convert import from_scipy_sparse_matrix

from ..preprocessors.graph_constructors import spatial_neighbors
from ..utils.type_conversions import sparse_mx_to_float_tensor, pandas_to_torch_one_hot


class CustomInMemoryDataset(InMemoryDataset):

    def __init__(self,
                 name: str = "sss2-1b_1p",
                 label_names: List['str'] = ['cell_types'],
                 graph_kwargs: dict = {},
                 data_directory_path: Optional[ str | Path ] = "/lustre/scratch126/cellgen/team361/DATASETS",
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 overwrite: bool = False) -> InMemoryDataset:
        """
        CustomInMemoryDataset class for loading PyG Data objects from AnnData files.

        Parameters:
        ----------
        - name: str
            Name of the dataset.
        - label_names: List[str]
            List of label names to be used for the dataset.
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
            If True, the processed data is overwritten. If False, the processed data is loaded from the processed directory.

        Returns:
        -------
        - InMemoryDataset
            A PyG InMemoryDataset object containing the processed PyG Data objects.
        """
        self.name = name

        assert len(label_names) > 0, "At least one label name must be provided"
        self.label_names = label_names

        self.graph_kwargs = graph_kwargs

        # path to the data directory which contains silver and gold data
        # silver data is the preprocessed, harmonized AnnData files
        # gold data is the processed, in-memory data ready for training and evaluation
        if isinstance(data_directory_path, str):
            data_directory_path = Path(data_directory_path)
        self.data_directory_path = data_directory_path

        # root = Root directory where the processed data is saved.
        super().__init__(root=self.processed_dir,
                         transform=transform,
                         pre_transform=pre_transform,
                         pre_filter=pre_filter)
        if overwrite:
            print("Overwriting the processed data...")
            self.process()

        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> Path:
        """
        Return the path to the raw data directory.
        """
        silver_data_path = self.data_directory_path / 'silver'
        raw_dir = silver_data_path / self.name
        return raw_dir

    @property
    def processed_dir(self) -> str:
        """
        Return the path to the processed data directory.
        """
        gold_data_path = self.data_directory_path / 'gold'
        processed_dir = str(gold_data_path / 'in-memory-PyG-data' / self.name)
        return processed_dir

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
    def processed_file_names(self) -> List[Data]:
        """
        Return the list of processed file names.
        """
        return ['data.pt']

    def process(self) -> None:
        """
        Process the raw data and save it to the processed data directory as a PyG InMemoryDataset. This method is called when the processed data is not found in the processed directory. Otherwise, the processed data is loaded from the processed directory.
        """
        data_attributes = {}

        # Read all AnnData files and convert them to one single PyG Data object
        # TODO: Explore the possibility of having a collection of PyG Data objects

        # TODO: Parallelize the processing of the AnnData files
        adata_batch_list = []
        for adata_batch_file in self.raw_paths:
            # Read the AnnData file
            print(f"Reading {adata_batch_file}...")
            adata_batch = sc.read(adata_batch_file)

            adata_batch_list.append(adata_batch)

        print("Concatenating AnnData files...")
        adata = ad.concat(adata_batch_list,
                          join='inner',
                          axis=0)

        print("Computing spatial neighbors...")
        adata, key_name = spatial_neighbors(adata,
                                            **self.graph_kwargs
                                            )

        # build for node features (sparse csr matrix to float tensor)
        data_attributes['x_cell_gene_counts'] = sparse_mx_to_float_tensor(adata.X)

        # build for edge index (sparse csr matrix to edge index style tensor)
        data_attributes[f"edge_index_{key_name}"] = from_scipy_sparse_matrix(
                                            adata.obsp[key_name + '_connectivities']
                                            )[0]

        # build for node labels (categorical pandas series to one hot tensor)
        for label_name in self.label_names:
            data_attributes[f"y_{label_name}"] = pandas_to_torch_one_hot(
                                            adata.obs[label_name]
                                            )

        # add metadata
        data_attributes['metadata_cell_id'] = adata.obs['cell_id']

        for key, value in data_attributes.items():
            print(f"{key}: {value.shape=}, {value.dtype=}, {type(value)=}")

        data_list = [Data(**data_attributes)]

        del data_attributes, adata

        if self.pre_filter is not None:
            data_list = [self.pre_filter(data) for data in data_list]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list=data_list,
                  path=self.processed_paths[0])