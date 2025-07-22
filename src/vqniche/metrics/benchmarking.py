"""
Source: https://github.com/Lotfollahi-lab/nichecompass/blob/main/src/nichecompass/benchmarking/metrics.py

This module contains the functionality to compute all benchmarking metrics based
on the spatial (physical) feature space and the learned latent feature space of
a deep generative model. The benchmark consists of metrics for gene expression imputation, spatial neighborhood reconstruction, spatial
conservation, biological conservation, niche identification, and batch
correction.
"""

import time
from typing import Optional, Literal

import numpy as np
import scanpy as sc
import scib_metrics
import networkx as nx
from anndata import AnnData
import torch

from .utils import compute_knn_graph_connectivities_and_distances
from ..utils.type_conversions import edge_index_to_adjacency_tensor
from ..utils.adjacency_reconstruction import reconstruct_adjacency_matrix as construct_binary_adjacency_matrix

from .cas import compute_cas
from .clisis import compute_clisis
from .gcs import compute_gcs
from .mlami import compute_mlami
from .nasw import compute_nasw
from .pearson_correlation import compute_pearson_correlation
from .mmd import compute_mmd_score, degree_histogram, eigenvalues_pmf


def compute_benchmarking_metrics(
        adata: AnnData,
        metrics: list=[
                        "mlami", # unsupervised global spatial conservation
                        "gcs", # unsupervised local spatial conservation
                        "nasw", # niche coherence
                    ],
        cell_type_key: str="cell_types",
        batch_key: Optional[str]=None, 
        spatial_key: str="spatial",
        latent_key: str="H_adj",
        pcr_X_pre: Optional[np.array]=None,
        fully_connected: bool=False,
        nonlinearity: Literal['min-max', 'sigmoid', 'softmax', 'relu-clamp'] = 'relu-clamp',
        k: int=8,
        seed: int=0
    ) -> dict:
    """
    Compute all specified benchmarking metrics.

    Parameters
    ----------
    adata:
        AnnData object to run the benchmarks for.
    metrics:
        List of metrics which will be computed.
    cell_type_key:
        Key under which the cell type annotations are stored in `adata.obs`.
    spatial_key:
        Key under which the spatial coordinates are stored in `adata.obsm`.
    latent_key:
        Key under which the latent representation from a model is stored in
        `adata.obsm`.
    pcr_X_pre:
        The unintegrated feature space for the computation of the pcr metric.
        If None, computes PCA on the raw counts stored in `adata.X`.
    fully_connected:
        If True, uses a Gaussian kernel to assign low weights to nodes more distant than the k-nearest neighbors.
        If False, uses a hard threshold to restrict the number of neighbors to `n_neighbors` using a UMAP connectivities.
    nonlinearity:
        The nonlinearity function to apply to the matrix product.
        - 'min-max': Min-max normalization to [0, 1] range
        - 'sigmoid': Sigmoid function for probability interpretation
        - 'softmax': Row-wise softmax for probability distributions
        - 'relu-clamp': ReLU followed by clamping to [0, 1]
        Default: 'relu-clamp'
    k:
        The number of top-k values to use to reconstruct a binary adjacency matrix.
        Default: 8
    seed:
        Random seed for reproducibility.

    Returns
    ----------
    benchmarking_dict:
        Dictionary containing the calculated benchmarking metrics.
    """
    start_time = time.time()

    benchmarking_dict = {}

    # ------------------------------------------------------------------------
    # Compute attribute imputation metrics
    # ------------------------------------------------------------------------
    if "pearson_cell_wise" in metrics:
        print("Computing Pearson correlation (cell-wise)...")
        benchmarking_dict["pearson_cell_wise"] = compute_pearson_correlation(
            adata=adata,
            X_key='X',
            X_hat_key='X_hat',
        )

    if "pearson_1hop_nbr" in metrics:
        print("Computing Pearson correlation (1-hop neighbor-wise)...")
        benchmarking_dict["pearson_1hop_nbr"] = compute_pearson_correlation(
            adata=adata,
            X_key='X',
            X_hat_key='X_hat',
            nbr_key='edge_index',
        )
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print("Pearson correlation computed. "
            f"Elapsed time: {minutes} minutes "
            f"{seconds} seconds.\n")
    
    # ------------------------------------------------------------------------
    # Compute graph imputation metrics
    # ------------------------------------------------------------------------
    if any(metric in ["mmd_degree", "mmd_eigenvalues", "num_edges", "max_degree"] for metric in metrics):
        # build a networkx graph from the edge index
        print("Computing graph from edge index...")
        G = nx.from_numpy_array(
                edge_index_to_adjacency_tensor(
                    adata.uns['edge_index']
                ).cpu().numpy()
            )

        # build a networkx graph from the estimated adjacency matrix
        print("Computing graph from estimated adjacency matrix...")
        G_hat = nx.from_numpy_array(
                construct_binary_adjacency_matrix(
                    h_index_nodes=torch.Tensor(adata.obsm['H_adj']),
                    nonlinearity=nonlinearity,
                    k=k,
                ).cpu().numpy()
            )
        
        if "num_edges" in metrics:
            print("Computing number of edges...")
            benchmarking_dict["G_num_edges"] = G.number_of_edges()
            benchmarking_dict["G_hat_num_edges"] = G_hat.number_of_edges()

        if "max_degree" in metrics:
            print("Computing maximum degree...")
            benchmarking_dict["G_max_degree"] = max(dict(G.degree()).values())
            benchmarking_dict["G_hat_max_degree"] = max(dict(G_hat.degree()).values())

        if "mmd_degree" in metrics:
            print("Computing MMD degree...")
            benchmarking_dict["mmd_degree"] = compute_mmd_score(
                [degree_histogram(G)],
                [degree_histogram(G_hat)],
            )
        
        if "mmd_eigenvalues" in metrics:
            print("Computing MMD eigenvalues...")
            benchmarking_dict["mmd_eigenvalues"] = compute_mmd_score(
                [eigenvalues_pmf(G)],
                [eigenvalues_pmf(G_hat)],
            )
        
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        print("Graph imputation metrics computed. "
              f"Elapsed time: {minutes} minutes "
              f"{seconds} seconds.\n")

    # ------------------------------------------------------------------------
    # Compute Connectivity Graphs for metrics that use them
    # ------------------------------------------------------------------------
    # Metrics use different k's for the knn graph
    # Based on specified metrics, determine which knn graphs to compute
    n_neighbors_list = []
    if any(metric in ["gcs", "mlami", "cas", "nasw"] for metric in
           metrics):
        n_neighbors_list.append(15) # default k for connectivity-based
                                    # metrics
    if any(metric in ["kbet"] for metric in metrics):
        n_neighbors_list.append(50) # kbet-specific k
    if any(metric in ["clisis", "clisi", "blisi"] for metric in metrics):
        n_neighbors_list.append(90) # lisi-specific k
    
    # Compute nearest neighbor graphs
    # Otherwise different metrics require different neighbor graphs and
    # this will be handled in the metric functions themselves
    if len(n_neighbors_list) > 0:
        # Compute spatial nearest neighbor graphs
        for n_neighbors in n_neighbors_list:
            if (f"{spatial_key}_{n_neighbors}knng_connectivities"
                not in adata.obsp):
                print("Computing spatial nearest neighbor graph with "
                      f"{n_neighbors} neighbors for entire dataset...")
                compute_knn_graph_connectivities_and_distances(
                        adata=adata,
                        feature_key=spatial_key,
                        knng_key=f"{spatial_key}_{n_neighbors}knng",
                        fully_connected=fully_connected,
                        n_neighbors=n_neighbors,
                        random_state=seed,
                    )
            else:
                print(f"Using precomputed spatial nearest neighbor graph "
                      f"with {n_neighbors} neighbors...")

        # Compute latent nearest neighbor graphs
        for n_neighbors in n_neighbors_list:
            if (f"{latent_key}_{n_neighbors}knng_connectivities"
                not in adata.obsp):
                print("Computing latent nearest neighbor graph with "
                      f"{n_neighbors} neighbors for entire dataset...")
                compute_knn_graph_connectivities_and_distances(
                        adata=adata,
                        feature_key=latent_key,
                        knng_key=f"{latent_key}_{n_neighbors}knng",
                        fully_connected=fully_connected,
                        n_neighbors=n_neighbors,
                        random_state=seed,
                    )
            else:
                print(f"Using precomputed latent nearest neighbor graph "
                      f"with {n_neighbors} neighbors...")

        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        print("Neighbor graphs computed. "
              f"Elapsed time: {minutes} minutes "
              f"{seconds} seconds.\n")

    # ------------------------------------------------------------------------
    # Global spatial conservation metrics
    # ------------------------------------------------------------------------
    # Supervised (cell-type label)
    if "cas" in metrics:
        print("Computing CAS metric...")
        benchmarking_dict["cas"] = compute_cas(
            adata=adata,
            cell_type_key=cell_type_key,
            batch_key=batch_key,
            spatial_knng_key=f"{spatial_key}_15knng",
            latent_knng_key=f"{latent_key}_15knng",
            spatial_key=spatial_key,
            latent_key=latent_key,
            seed=seed)
              
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        print("CAS metric computed. "
              f"Elapsed time: {minutes} minutes "
              f"{seconds} seconds.\n")

    # Unsupervised
    if "mlami" in metrics:
        print("Computing MLAMI Metric...")
        benchmarking_dict["mlami"] = compute_mlami(
            adata=adata,
            batch_key=batch_key,
            spatial_knng_key=f"{spatial_key}_15knng",
            latent_knng_key=f"{latent_key}_15knng",
            spatial_key=spatial_key,
            latent_key=latent_key,
            seed=seed)
        
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        print("MLAMI metric computed. "
              f"Elapsed time: {minutes} minutes "
              f"{seconds} seconds.\n")
    
    # ------------------------------------------------------------------------
    # Local spatial conservation metrics
    # ------------------------------------------------------------------------
    # Supervised (cell-type label)
    if "clisis" in metrics:
        try:
            print("Computing CLISIS metric...")
            benchmarking_dict["clisis"] = compute_clisis(
                adata=adata,
                cell_type_key=cell_type_key,
                batch_key=batch_key,
                spatial_knng_key=f"{spatial_key}_90knng",
                latent_knng_key=f"{latent_key}_90knng",
                spatial_key=spatial_key,
                latent_key=latent_key,
                seed=seed)

            elapsed_time = time.time() - start_time
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            print("CLISIS metric computed. "
                  f"Elapsed time: {minutes} minutes "
                  f"{seconds} seconds.\n")
        except:
            print("Could not compute CLISIS metric.")
            benchmarking_dict["clisis"] = 0.
    
    # Unsupervised
    if "gcs" in metrics:
        print("Computing GCS metric...")
        benchmarking_dict["gcs"] = compute_gcs(
            adata=adata,
            batch_key=batch_key,
            spatial_knng_key=f"{spatial_key}_15knng",
            latent_knng_key=f"{latent_key}_15knng",
            spatial_key=spatial_key,
            latent_key=latent_key,
            seed=seed)

        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        print("GCS metric computed. "
              f"Elapsed time: {minutes} minutes "
              f"{seconds} seconds.\n")            
    
    # ------------------------------------------------------------------------
    # Niche coherence metrics
    # ------------------------------------------------------------------------
    # Supervised (cell-type label)
    if "cnmi" in metrics or "cari" in metrics:
        print("Computing CNMI and CARI metrics...")
        cnmi_cari_dict = scib_metrics.nmi_ari_cluster_labels_kmeans(
            X=adata.obsm[latent_key],
            labels=adata.obs[cell_type_key])
              
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        print("CNMI and CARI metrics computed. "
              f"Elapsed time: {minutes} minutes "
              f"{seconds} seconds.\n")

        if "cnmi" in metrics:
              benchmarking_dict["cnmi"] = cnmi_cari_dict["nmi"]
        if "cari" in metrics:
              benchmarking_dict["cari"] = cnmi_cari_dict["ari"]
    
    # Supervised (cell-type label)
    if "casw" in metrics:
        print("Computing CASW Metric...")
        benchmarking_dict["casw"] = scib_metrics.silhouette_label(
            X=adata.obsm[latent_key],
            labels=adata.obs[cell_type_key])
              
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        print("CASW metric computed. "
              f"Elapsed time: {minutes} minutes "
              f"{seconds} seconds.\n")
    
    # Supervised (cell-type label)
    if "clisi" in metrics:
        try:
            print("Computing CLISI metric...")
            benchmarking_dict["clisi"] = scib_metrics.clisi_knn(
                X=adata.obsp[f"{latent_key}_90knng_distances"],
                labels=adata.obs[cell_type_key])

            elapsed_time = time.time() - start_time
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            print("CLISI metric computed. "
                  f"Elapsed time: {minutes} minutes "
                  f"{seconds} seconds.\n")
        except:
            print("Could not compute CLISI metric.")
            benchmarking_dict["clisi"] = 0.
        
    # Unsupervised
    if "nasw" in metrics:
        print("Computing NASW Metric...")
        benchmarking_dict["nasw"] = compute_nasw(
                adata=adata,
                latent_knng_key=f"{latent_key}_15knng",
                latent_key=latent_key,
                seed=seed)
        
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        print("NASW metric computed. "
              f"Elapsed time: {minutes} minutes "
              f"{seconds} seconds.\n")

    # ------------------------------------------------------------------------
    # Batch correction metrics
    # ------------------------------------------------------------------------
    # Supervised (cell-type label)
    if batch_key is not None:
        if "basw" in metrics:
            print("Computing BASW Metric...")
            benchmarking_dict["basw"] = scib_metrics.silhouette_batch(
                X=adata.obsm[latent_key],
                labels=adata.obs[cell_type_key],
                batch=adata.obs[batch_key])
              
            elapsed_time = time.time() - start_time
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            print("BASW metric computed. "
                  f"Elapsed time: {minutes} minutes "
                  f"{seconds} seconds.\n")
    
        # Supervised (cell-type label)
        if "kbet" in metrics:
            benchmarking_dict["kbet"] = scib_metrics.kbet_per_label(
                X=adata.obsp[f"{latent_key}_50knng_connectivities"],
                batches=adata.obs[batch_key],
                labels=adata.obs[cell_type_key])
              
            elapsed_time = time.time() - start_time
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            print("KBET metric computed. "
                  f"Elapsed time: {minutes} minutes "
                  f"{seconds} seconds.")

        # Unsupervised
        if "blisi" in metrics:
            try:
                print("Computing BLISI Metric...")
                benchmarking_dict["blisi"] = scib_metrics.ilisi_knn(
                    X=adata.obsp[f"{latent_key}_90knng_distances"],
                    batches=adata.obs[batch_key])

                elapsed_time = time.time() - start_time
                minutes = int(elapsed_time // 60)
                seconds = int(elapsed_time % 60)
                print("BLISI metric computed. "
                      f"Elapsed time: {minutes} minutes "
                      f"{seconds} seconds.\n")
            except:
                print("Could not compute BLISI metric.")
                benchmarking_dict["blisi"] = 0.

        # Unsupervised
        if "pcr" in metrics:
            # https://github.com/yoseflab/scib-metrics/blob/0.4.0/src/scib_metrics/benchmark/_core.py#L171
            if pcr_X_pre is None:
                if "X_pca" not in adata.obsm:
                    sc.tl.pca(adata, use_highly_variable=False)
                pcr_X_pre = adata.obsm["X_pca"]
            benchmarking_dict["pcr"] = scib_metrics.pcr_comparison(
                X_pre=pcr_X_pre,
                X_post=adata.obsm[latent_key],
                covariate=adata.obs[batch_key],
                categorical=True)
            
            print(benchmarking_dict["pcr"])
            
            elapsed_time = time.time() - start_time
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            print("PCR metric computed. "
                  f"Elapsed time: {minutes} minutes "
                  f"{seconds} seconds.")

    return benchmarking_dict