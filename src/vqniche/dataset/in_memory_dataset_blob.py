"""
Define `Data` to refer to a PyG Data object that is constructed from a single batch of AnnData where each batch is a collection of cells originating from the same tissue section. Each `Data` object will have one or more of each type of the following attribute:
- `x_<feature_name>`: Node features (e.g.,cell-gene counts or neighborhood-gene counts)
- `y_<label_name>`: Node labels (e.g., cell types or niche types)
- `edge_index_<edge_index_name>`: Edge index (e.g., spatial neighbors via Delaunay triangulation or radius-based neighbors or k-nearest neighbors)
- `metadata_<metadata_name>`: Metadata (e.g., cell ids, batch ids, tissue, etc.)
We construct these attributes from the AnnData object stored as an `.h5ad` file on disk.

Next, define `Dataset` to refer to a PyG InMemoryDataset object comprising of collection of AnnData batches, i.e. `Data` objects. From an implementation perspective, we consider the Dataset to be a "blob" because we collate (combine) the individual Data objects (built previously from AnnData batches) into a single file called `dataset_blob.pt` before saving to disk. Currently, all Data objects in the Dataset have the same gene panel, meaning that the input feature dimensions are the same across all tissue sections.

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
- One dataset_blob.pt across all tissue sections in the experiment folder vs one data.pt per tissue section in the experiment folder.
- One dataset_blob.pt is easier to manage and load than multiple data.pt files. During training, we can load the entire dataset in one go and use one or more tissue sections per model as needed. This is useful because the subsequent subgraph sampling and batching can be done on the fly.
- For each epoch, we subset the dataset_blob by batch(es) and then sample subgraphs from the batch(es) for training. This is more time efficient than sampling subgraphs from each batch separately, but requires more memory.
- Memory usage is a concern because the entire dataset is loaded into memory at once. This is not a problem for small datasets but can be a bottleneck for large datasets.
- For larger datasets, we have two options:
    1. Create one dataset_blob.pt per batch of cells in the experiment folder. This is more memory efficient but increases time complexity because, if we have (say) T training epochs, each AnnData batch is loaded and removed from memory T times.
    2. Create an OnDiskDatasetBlob that loads parallely in chunks.
"""
import os
import copy
import subprocess
from pathlib import Path
import concurrent.futures
from typing import Optional, Callable, List, Tuple

import numpy as np
import scanpy as sc
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

import torch
from torch import Tensor
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import is_undirected
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
            overwrite: bool = False,
            software_paths: dict = {}
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
        - software_paths: dict
            Dictionary containing the paths to the third-party software used to build the dataset.

        Returns:
        -------
        - InMemoryDataset
            A PyG InMemoryDataset object containing the processed PyG Data objects.
        """
        self.name = name

        self.feature_names = feature_names
        self.label_names = [label_name.split('=')[0] for label_name in label_names]
        self.label_keys = [label_name.split('=')[1] for label_name in label_names]
        self.graph_kwargs = graph_kwargs

        # path to the data directory which contains silver and gold data
        # silver data is the preprocessed, harmonized AnnData files
        # gold data is the processed, in-memory data ready for training and evaluation
        if isinstance(data_directory_path, str):
            data_directory_path = Path(data_directory_path)
        self.data_directory_path = data_directory_path

        self.software_paths = software_paths

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
        # This pass is used to collect gene panels and the unique categories for each label name across all batches.
        # All batches must have the same gene panel. Multiple gene panels across tissue sections are not supported.

        self.gene_panel = None

        # Collect the unique categories for each label name across all batches.
        self.label_categories = {
            label_name: set()
            for label_name in self.label_names
        }

        for adata_batch_file in self.raw_paths:
            adata_batch = sc.read(adata_batch_file)

            if self.gene_panel is None:
                # adata_batch.var is a pandas dataframe
                # index is the gene name
                # columns are the ensemble ids
                self.gene_panel = adata_batch.var
            else:
                # check if the gene panel is the same across all batches
                if not self.gene_panel.equals(adata_batch.var):
                    raise ValueError("All batches must have the same gene panel")

            for label_name, label_key in zip(self.label_names, self.label_keys):
                self.label_categories[label_name].update(adata_batch.obs[label_key].unique())

        print("All batches have the same gene panel.")

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
        print(f"Number of Cores: {num_cores} | Number of Tissue Sections: {num_files}")
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

        for delaunay in [False]:
            radius_list = [None]
            if self.graph_kwargs['radius_list'] is not None:
                radius_list = self.graph_kwargs['radius_list'] + [None]
            for radius in radius_list:
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

            # ----------------- Build Feature Matrices -----------------
        # build for node features (sparse csr matrix to float tensor)
        # NOTE: adata_batch.X is assumed to a the raw cell-gene counts matrix.
        assert len(self.feature_names) > 0, "At least one feature name must be provided"
        for feature_name in self.feature_names:
            if feature_name == 'cell_gene_counts':
                batch_dict[f'x_{feature_name}'] = sparse_mx_to_float_tensor(
                                                        adata_batch.X
                                                    )
            else:
                raise ValueError(f"Feature name {feature_name} not supported")

            # ----------------- Build Torch One-Hot Labels -----------------
        # build for node labels (categorical pandas series to one hot tensor)
        for label_name, label_key in zip(self.label_names, self.label_keys):
            batch_dict[f"y_{label_name}"] = pandas_to_torch_one_hot(
                                                adata_batch.obs[label_key],
                                                categories=self.label_categories[label_name]
                                            )

            # ----------------- Build Edge Indices -----------------
        # build one edge index for each edge index name (as defined by above)
        assert len(self.edge_index_names) > 0, "At least one edge index name must be provided"
        for edge_index_name in self.edge_index_names:
            # adata_batch.obsp[f"{edge_index_name}_connectivities"] is a symmetric matrix for an undirected, unweighted graph
            # the second element of the tuple returned by from_scipy_sparse_matrix is the edge weights which we don't need
            # undirected graphs are represented by including both i->j and j->i in the edge_index tensor
            # Check if the connectivity matrix is symmetric
            connectivity_matrix = adata_batch.obsp[f"{edge_index_name}_connectivities"]
            count_nnz = (connectivity_matrix != connectivity_matrix.T).nnz
            if count_nnz > 0:
                print(f"{edge_index_name} is not symmetric. Making it symmetric...")
                adata_batch.obsp[f"{edge_index_name}_connectivities"] = (
                    adata_batch.obsp[f"{edge_index_name}_connectivities"].maximum(
                        adata_batch.obsp[f"{edge_index_name}_connectivities"].T
                    )
                )

            batch_dict[f"edge_index_{edge_index_name}"] = from_scipy_sparse_matrix(
                                                        adata_batch.obsp[f"{edge_index_name}_connectivities"]
                                                        )[0]
            # check if the edge index is undirected
            if not is_undirected(batch_dict[f"edge_index_{edge_index_name}"]):
                raise ValueError(f"Edge index {edge_index_name} is not undirected")

            # ----------------- Build Unsupervised Node Embeddings -----------------
            # for small graphs, build spectral embeddings
            num_nodes = adata_batch.obsp[f"{edge_index_name}_connectivities"].shape[0]
            if num_nodes < 20_000:
                print(f"Computing {self.graph_kwargs['k']['lm_eigvecs']} eigenvectors of {edge_index_name}...")
                batch_dict[f"U_lm_eigvecs_{edge_index_name}"] = self.build_lm_eigenvectors(
                    A=adata_batch.obsp[f"{edge_index_name}_connectivities"],
                )
                print(f"U_lm_eigvecs_{edge_index_name}.shape: {batch_dict[f'U_lm_eigvecs_{edge_index_name}'].shape}")

            # save graph to disk in edgelist format
            self.save_adjacency_matrix_to_edgelist(
                batch_id=adata_batch.uns['batch'],
                A=adata_batch.obsp[f"{edge_index_name}_connectivities"],
                edge_index_name=edge_index_name
            )

            # build deepwalk embeddings
            batch_dict[f"U_deepwalk_{edge_index_name}"] = self.build_deepwalk_embeddings(
                batch_id=adata_batch.uns['batch'],
                edge_index_name=edge_index_name
            )
            print(f"U_deepwalk_{edge_index_name}.shape: {batch_dict[f'U_deepwalk_{edge_index_name}'].shape}")

            # build gosh embeddings
            batch_dict[f"U_gosh_{edge_index_name}"] = self.build_gosh_embeddings(
                batch_id=adata_batch.uns['batch'],
                edge_index_name=edge_index_name
            )
            print(f"U_gosh_{edge_index_name}.shape: {batch_dict[f'U_gosh_{edge_index_name}'].shape}")

        for key, value in batch_dict.items():
            print(f"{key}: {value.shape=}, {value.dtype=}, {type(value)=}")

        # ----------------- Add Metadata -----------------
        batch_dict['xy_coordinates'] = Tensor(adata_batch.obsm['spatial'])
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


    def build_lm_eigenvectors(
            self,
            A: sp.csr_matrix,
        ) -> Tensor:
        """
        Build k eigenvectors corresponding to the k largest (in magnitude) eigenvalues of the adjacency matrix of the graph.

        Parameters:
        ----------
        - A: sp.csr_matrix
            The adjacency matrix of the graph.

        Returns:
        -------
        - U: Tensor
            The eigenvectors of the Laplacian matrix of the graph.
        """
        U = eigsh(
                    A=A,
                    which='LM',
                    k=self.graph_kwargs['k']['lm_eigvecs'],
                    tol=1e-4,
                    maxiter=1e6,
                )[1]
        U = torch.from_numpy(U).to(torch.float32)

        return U


    def build_deepwalk_embeddings(
            self,
            batch_id: str,
            edge_index_name: str,
        ) -> Tensor:
        """
        Build deepwalk embeddings for the given edge index name.

        Parameters:
        ----------
        - batch_id: str
            The batch id of the adjacency matrix.
        - edge_index_name: str
            The name of the edge index to build deepwalk embeddings for.

        Returns:
        -------
        - U: Tensor
            The deepwalk embeddings.

        Notes:
        ------
        - Setting p=1 and q=1 in node2vec is equivalent to computing deepwalk embeddings.
        """
        deepwalk_executable = self.software_paths['deepwalk']
        batch_dir = Path(self.processed_dir) / batch_id
        batch_dir.mkdir(parents=True, exist_ok=True)
        edgelist_fname = batch_dir / f"{edge_index_name}.edgelist"
        emb_fname = batch_dir / f"U_deepwalk_{edge_index_name}.emb"

        exec_str = f"{deepwalk_executable} -i:{edgelist_fname} -o:{emb_fname} -d:{self.graph_kwargs['k']['deepwalk']} -p:1 -q:1 -r:10 -l:40 -v"

        print(f"Running {exec_str}...")
        _ = subprocess.run(exec_str, shell=True)
        print(f"U_deepwalk_{edge_index_name}.emb saved to {emb_fname}")

        U = self._read_deepwalk_embeddings_from_disk(emb_fname)

        return U


    def build_gosh_embeddings(
            self,
            batch_id: str,
            edge_index_name: str,
        ) -> Tensor:
        """
        Build GOSH embeddings for the given edge index name.

        Parameters:
        ----------
        - batch_id: str
            The batch id of the adjacency matrix.
        - edge_index_name: str
            The name of the edge index to build GOSH embeddings for.

        Returns:
        -------
        - U: Tensor
            The GOSH embeddings.

        Notes:
        ------
        - The embedding is binary format for fast loading.
        - For scaling to larger graphs, hyperparameters will need to be adjusted.
        """
        if not torch.cuda.is_available():
            raise ValueError("GOSH embeddings are not supported on CPU")

        gosh_executable = self.software_paths['gosh']
        batch_dir = Path(self.processed_dir) / batch_id
        batch_dir.mkdir(parents=True, exist_ok=True)
        edgelist_fname = batch_dir / f"{edge_index_name}.edgelist"
        emb_fname = batch_dir / f"U_gosh_{edge_index_name}.emb"

        exec_str = f"{gosh_executable} --input-graph {edgelist_fname} --output-embedding {emb_fname} --directed 0 --epochs 200 -d {self.graph_kwargs['k']['gosh']} -s 3 --negative-weight 1 --binary-output --sampling-algorithm 0 -a 0 -l 0.025 --learning-rate-decay-strategy 0 --coarsening-stopping-threshold 1000 --coarsening-stopping-precision 0.9 --coarsening-matching-threshold-ratio 400 --coarsening-min-vertices-in-graph 1000 --epoch-strategy s-fast --smoothing-ratio 0.5"

        print(f"Running {exec_str}...")
        _ = subprocess.run(exec_str, shell=True)
        print(f"U_gosh_{edge_index_name}.emb saved to {emb_fname}")

        U = torch.from_numpy(np.load(emb_fname)).to(torch.float32)

        return U


    def _read_deepwalk_embeddings_from_disk(
            self,
            emb_fname: str,
        ) -> Tensor:
        """
        Read the DeepWalk embeddings from disk.

        Parameters:
        ----------
        - emb_fname: str
            The name of the file to read the DeepWalk embeddings from.

        Returns:
        -------
        - U: Tensor
            The DeepWalk embeddings.

        Notes:
        ------
        - The first line of the file contains the number of nodes and the number of dimensions.
        - The remaining lines contain the node id and the embedding vector.
        """
        is_first_line = True
        with open(emb_fname, 'r') as f:
            for line in f:
                line = line.strip().split()
                if is_first_line:
                    is_first_line = False
                    n_nodes, n_dims = int(line[0]), int(line[1])
                    U = torch.zeros(n_nodes, n_dims)
                    continue
                node_id = int(line[0])
                U[node_id] = torch.tensor(
                                [float(x) for x in line[1:]],
                                dtype=torch.float32
                            )

        return U


    def save_adjacency_matrix_to_edgelist(
            self,
            batch_id: str,
            A: sp.csr_matrix,
            edge_index_name: str,
        ) -> None:
        """
        Save the adjacency matrix to disk in edgelist format.

        Parameters:
        ----------
        - batch_id: str
            The batch id of the adjacency matrix.
        - A: sp.csr_matrix
            The adjacency matrix to save.
        - edge_index_name: str
            The name of the edge index to save.
        """
        assert isinstance(A, sp.csr_matrix), "A must be a scipy.csr_matrix"

        batch_dir = Path(self.processed_dir) / batch_id
        batch_dir.mkdir(parents=True, exist_ok=True)
        edgelist_fname = batch_dir / f"{edge_index_name}.edgelist"

        with open(edgelist_fname, 'w') as f:
            for row in range(A.shape[0]):
                start_idx = A.indptr[row]
                end_idx = A.indptr[row + 1]
                cols = A.indices[start_idx:end_idx]

                for col in cols:
                    f.write(f"{row} {col}\n")

        print(f"Saved {edge_index_name} to {edgelist_fname}...")
        return


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