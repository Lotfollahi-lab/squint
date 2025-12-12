"""
Define `Data` to refer to a PyG Data object that is constructed from a single batch of AnnData where each batch is a collection of cells originating from the same tissue section. Each `Data` object will have one or more of each type of the following attribute:
- `x_<feature_name>`: Node features (e.g.,cell-gene counts or neighborhood-gene counts)
- `y_<label_name>`: Node labels (e.g., cell types or niche types)
- `edge_index_<edge_index_name>`: Edge index (e.g., spatial neighbors via Delaunay triangulation or radius-based neighbors or k-nearest neighbors)
- `metadata_<metadata_name>`: Metadata (e.g., cell ids, batch ids, tissue, etc.)
We construct these attributes from the AnnData object stored as an `.h5ad` file on disk.

Next, define `Dataset` to refer to a PyG OnDiskDataset object comprising a collection of AnnData batches, i.e. `Data` objects. From an implementation perspective, we save each individual Data object (built from AnnData batches) as a separate .pt file on disk. Currently, all Data objects in the Dataset have the same gene panel, meaning that the input feature dimensions are the same across all tissue sections.

Usage:
>>> dataset = OnDiskDatasetBlob(name='sss2-1b_1p.yaml',
                                        label_names=['cell_types'],
                                        graph_kwargs={'delaunay': True},
                                        data_directory_path='/path/to/data',
                                        transform=transform)
>>> data = dataset[<batch-index>] # get a single Data object corresponding to a batch-index (say, 7)

Tradeoffs:
-------
- Multiple .pt files (one per batch) vs loading all data into memory at once (as in InMemoryDataset).
- On-disk storage allows for larger datasets without memory constraints, as graphs are loaded on demand. This is more memory efficient but may be slower for frequent access.
- During training, we can load specific batches as needed, enabling processing of large datasets that don't fit in memory.
- For smaller datasets, an in-memory approach might be faster, but on-disk is better for scalability.
"""
import os
import copy
import pickle
import subprocess
from pathlib import Path
import concurrent.futures
import tracemalloc
from typing import Optional, Callable, List, Tuple

import numpy as np
import scanpy as sc
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

import torch
from torch import Tensor
from torch_geometric.data import OnDiskDataset, Data
from torch_geometric.utils import is_undirected
from torch_geometric.utils.convert import from_scipy_sparse_matrix

import sqlite3

from ..preprocessors.graph_constructors import set_edge_index_name, spatial_neighbors
from ..utils.type_conversions import sparse_mx_to_float_tensor, pandas_to_torch_one_hot


class OnDiskDatasetBlob(OnDiskDataset):

    def __init__(
        self,
        name: str = "mmb0-4b_1p",
        feature_names: List['str'] = [],
        label_names: List['str'] = [],
        graph_kwargs: dict = {},
        data_directory_path: Optional[ str | Path ] = "/lustre/scratch126/cellgen/team361/DATASETS",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        overwrite: bool = False,
        software_paths: dict = {},
        num_graphs_to_load=0,
    ):

        self.name = name
        
        self.feature_names = feature_names
        self.label_names = [x.split("=")[0] for x in label_names]
        self.label_keys = [x.split("=")[1] for x in label_names]
        self.graph_kwargs = graph_kwargs
        self.num_graphs_to_load = num_graphs_to_load

        if isinstance(data_directory_path, str):
            data_directory_path = Path(data_directory_path)
        self.data_directory_path = data_directory_path

        # define root manually before calling the parent; do NOT assign
        # to `processed_dir` because it is exposed as a @property below
        # (assigning to it raises `AttributeError: can't set attribute`).
        self.root = os.path.join(self.data_directory_path, "gold", "on-disk-PyG-dataset-blob", self.name)

        # store transforms/filters on the instance (some OnDiskDataset
        # implementations don't accept pre_transform/pre_filter kwargs)
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter

        # call parent (do not forward pre_transform/pre_filter as kwargs)
        super().__init__(
            root=str(self.root),
            transform=transform,
        )

    # Required for safety:
    def serialize(self, data):
        return pickle.dumps(data)

    def deserialize(self, data):
        """Try to deserialize bytes stored in the DB.

        Some producer code in this repo used `torch.save(..., buffer)` to
        write examples into sqlite (which produces a pickle stream that may
        contain persistent ids). Other code used plain `pickle.dumps`.
        Try pickle.loads first and fall back to torch.load on a BytesIO.
        """
        # If `data` is already an object, return it directly
        if not isinstance(data, (bytes, bytearray)):
            return data

        try:
            return pickle.loads(data)
        except Exception:
            # Fall back to torch.load for tensors / storages written by torch.save
            try:
                import io as _io
                buf = _io.BytesIO(data)
                return torch.load(buf, map_location='cpu')
            except Exception as e:
                # Re-raise original error with context
                raise RuntimeError(f"Failed to deserialize blob: {e}")
    
    def __iter__(self):
        """
        Stream dataset in chunks defined by num_graphs_to_load.
        """

        n = len(self)
        print("Number of graphs on disk:", n)
        print("ROOT:", self.root)
        print("PROCESSED_DIR:", self.processed_dir)

        for i in range(0, n, self.num_graphs_to_load):
            end = min(i + self.num_graphs_to_load, n)
            print(f"Loading graphs {i} to {end} from disk...", flush=True)

            chunk = []
            for j in range(i, end):
                try:
                    data = self[j]  # <-- lazy load .pt file
                except Exception as e:
                    print(f"Warning: failed to load index {j}: {e}", flush=True)
                    continue
                chunk.append(data)

            for data in chunk:
                yield data

            del chunk


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

            # save graph to disk in edgelist format
            print(f"Saving {edge_index_name} to disk in edgelist format...")
            self.save_adjacency_matrix_to_edgelist(
                batch_id=adata_batch.uns['batch'],
                A=adata_batch.obsp[f"{edge_index_name}_connectivities"],
                edge_index_name=edge_index_name
            )

            # ----------------- Build Unsupervised Node Embeddings -----------------
            # if dimensions for lm_eigvecs are specified, build spectral embeddings
            # if 'lm_eigvecs' in self.graph_kwargs['k']:
            #     print(f"Computing {self.graph_kwargs['k']['lm_eigvecs']} eigenvectors of {edge_index_name}...")
            #     batch_dict[f"U_lm_eigvecs_{edge_index_name}"] = self.build_lm_eigenvectors(
            #         A=adata_batch.obsp[f"{edge_index_name}_connectivities"],
            #     )
            #     print(f"U_lm_eigvecs_{edge_index_name}.shape: {batch_dict[f'U_lm_eigvecs_{edge_index_name}'].shape}")

            # # if dimensions for deepwalk are specified, build deepwalk embeddings
            # if 'deepwalk' in self.graph_kwargs['k']:
            #     batch_dict[f"U_deepwalk_{edge_index_name}"] = self.build_deepwalk_embeddings(
            #         batch_id=adata_batch.uns['batch'],
            #         edge_index_name=edge_index_name
            #     )
            #     print(f"U_deepwalk_{edge_index_name}.shape: {batch_dict[f'U_deepwalk_{edge_index_name}'].shape}")

            # # if dimensions for gosh are specified, build gosh embeddings
            # if 'gosh' in self.graph_kwargs['k']:
            #     batch_dict[f"U_gosh_{edge_index_name}"] = self.build_gosh_embeddings(
            #         batch_id=adata_batch.uns['batch'],
            #         edge_index_name=edge_index_name
            #     )
            #     print(f"U_gosh_{edge_index_name}.shape: {batch_dict[f'U_gosh_{edge_index_name}'].shape}")

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

    def process(self):
        """
        Reads all .h5ad batches for this dataset, builds PyG Data objects,
        and saves each graph as a separate .pt file in the processed_dir.
        """
        print("=== Starting OnDiskDatasetBlob.process() ===")
        print("Writing dataset to:", self.processed_dir)

        # Ensure processed directory exists
        os.makedirs(self.processed_dir, exist_ok=True)

        # Initialize gene panel and label categories
        self.gene_panel = None
        self.label_categories = {label_name: set() for label_name in self.label_names}

        # Locate all silver .h5ad batches
        silver_dir = self.data_directory_path / "silver" / self.name
        if not silver_dir.exists():
            raise FileNotFoundError(f"Silver directory not found: {silver_dir}")

        batch_files = sorted(silver_dir.glob("*.h5ad"))
        if len(batch_files) == 0:
            print("⚠ No .h5ad files found. Nothing to process.")
            return

        print(f"Found {len(batch_files)} batches.")

        # ----------------- First Pass -----------------
        # Collect label categories and verify gene panel
        for batch_file in batch_files:
            adata_batch = sc.read(batch_file)

            if self.gene_panel is None:
                self.gene_panel = adata_batch.var
            else:
                if not self.gene_panel.equals(adata_batch.var):
                    raise ValueError("All batches must have the same gene panel")

            for label_name, label_key in zip(self.label_names, self.label_keys):
                self.label_categories[label_name].update(adata_batch.obs[label_key].unique())

        # Sort categories
        for label_name in self.label_names:
            self.label_categories[label_name] = sorted(list(self.label_categories[label_name]))
            print(f"Label Name: {label_name} | {self.label_categories[label_name]=}")

        # Save label categories and gene panel
        with open(Path(self.processed_dir) / "label_categories.pkl", "wb") as f:
            pickle.dump(self.label_categories, f)
        with open(Path(self.processed_dir) / "gene_panel.pkl", "wb") as f:
            pickle.dump(self.gene_panel, f)

        print("Saved label categories and gene panel to disk.")

        # ----------------- Second Pass -----------------
        # Process each batch and save each graph as a .pt file
        graph_counter = 0
        for i, batch_file in enumerate(batch_files):
            print(f"\n--- Processing batch {i+1}/{len(batch_files)} ---")
            print("File:", batch_file)

            data_batch = self.process_anndata_batch(batch_file)

            # Save each graph as a .pt file
            graph_file = Path(self.processed_dir) / f"{graph_counter}.pt"
            torch.save(data_batch, graph_file)
            print(f"Saved graph {graph_counter} to {graph_file}")

            graph_counter += 1

        print("Finished writing all graphs to disk.")
        print("Number of graphs on disk:", graph_counter)


    
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
        processed_dir = str(gold_data_path / 'on-disk-PyG-dataset-blob' / self.name)
        return processed_dir

    @property
    def processed_file_names(self) -> List[str]:
        """Return a list of .pt files in the processed_dir."""
        pt_files = sorted([f.name for f in Path(self.processed_dir).glob("*.pt")])
        if not pt_files:
            print(f"Warning: No processed .pt files found in {self.processed_dir}")
        return pt_files


    def __len__(self) -> int:
        """Return number of graphs on disk."""
        # Count .pt files in processed_dir (only numbered graph files)
        pt_files = [f for f in Path(self.processed_dir).glob("*.pt") if f.stem.isdigit()]
        return len(pt_files)

    def __getitem__(self, idx: int):
        """Load a graph by index."""
        # load from .pt file
        pt_files = sorted([f for f in Path(self.processed_dir).glob("*.pt") if f.stem.isdigit()])
        if idx < 0:
            idx = len(pt_files) + idx
        if idx >= len(pt_files) or idx < 0:
            raise IndexError(f"Index {idx} out of range")
        return torch.load(pt_files[idx])
