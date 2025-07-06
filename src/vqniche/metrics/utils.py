"""
Source: https://github.com/Lotfollahi-lab/nichecompass/blob/main/src/nichecompass/benchmarking/utils.py
Source: https://github.com/Lotfollahi-lab/nichecompass/blob/main/src/nichecompass/benchmarking/metrics.py

This module contains helper functions for the ´benchmarking´ subpackage.
"""
from typing import Optional
import time

import numpy as np
import scanpy as sc
from anndata import AnnData
import scib_metrics
from scib_metrics.nearest_neighbors import pynndescent

from .cas import compute_cas
from .clisis import compute_clisis
from .gcs import compute_gcs
from .mlami import compute_mlami
from .nasw import compute_nasw



def compute_benchmarking_metrics(
        adata: AnnData,
        metrics: list=["cas", # global spatial conservation (cell label supervised)
                       "mlami", # global spatial conservation (unsupervised)
                       "clisis", # local spatial conservation (cell label supervised)
                       "gcs", # local spatial conservation (unsupervised)
                       "cnmi", # niche coherence
                       "nasw", # niche coherence
                       "basw", # batch correction
                       "blisi", # batch correction
                       "kbet", # batch correction
                       "pcr"], # batch correction
        cell_type_key: str="cell_type",
        batch_key: Optional[str]=None, 
        spatial_key: str="spatial",
        latent_key: str="nichecompass_latent",
        pcr_X_pre: Optional[np.array]=None,
        n_jobs: int=1,
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
    n_jobs:
        Number of jobs to use for parallelization of neighbor search.
    seed:
        Random seed for reproducibility.

    Returns
    ----------
    benchmarking_dict:
        Dictionary containing the calculated benchmarking metrics.
    """
    start_time = time.time()

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
    
    benchmarking_dict = {}
    
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
                        n_neighbors=n_neighbors,
                        random_state=seed,
                        n_jobs=n_jobs)
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
                        n_neighbors=n_neighbors,
                        random_state=seed,
                        n_jobs=n_jobs) # pynndescent has to be version 0.5.8 
                                       # otherwise this can throw errors for some random seeds and big latents
            else:
                print(f"Using precomputed latent nearest neighbor graph "
                      f"with {n_neighbors} neighbors...")

        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        print("Neighbor graphs computed. "
              f"Elapsed time: {minutes} minutes "
              f"{seconds} seconds.\n")
            
    # Compute benchmarking metrics
    print("Computing benchmarking metrics...")
    
    # Global spatial conservation metric (cell label supervised)    
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

    # Global spatial conservation metric (unsupervised)
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
    
    # Local spatial conservation metric (cell label supervised)
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
    
    # Local spatial conservation metric (unsupervised) 
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
    
    # Niche coherence metrics
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

    # Batch correction metrics
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


def compute_knn_graph_connectivities_and_distances(
        adata: AnnData,
        feature_key: str="nichecompass_latent",
        knng_key: str="nichecompass_latent_15knng",
        n_neighbors: int=15,
        random_state: int=0,
        n_jobs: int=1) -> None:
    """
    Compute approximate k-nearest-neighbors graph.

    Parameters
    ----------
    adata:
        AnnData object with the features for knn graph computation stored in
        `adata.obsm[feature_key]`.
    feature_key:
        Key in `adata.obsm` that will be used to compute the knn graph.
    knng_key:
        Key under which the knn graph connectivities  will be stored
        in `adata.obsp` with the suffix '_connectivities', the knn graph
        distances will be stored in `adata.obsp` with the suffix '_distances',
        and the number of neighbors will be stored in `adata.uns` with the suffix
        '_n_neighbors' .      
    n_neighbors:
        Number of neighbors of the knn graph.
    random_state:
        Random state for reproducibility.   
    n_jobs:
        Number of jobs to use for parallelization of neighbor search.
    """
    neigh_output = pynndescent(
        adata.obsm[feature_key],
        n_neighbors=n_neighbors,
        random_state=random_state,
        n_jobs=n_jobs)
    indices, distances = neigh_output.indices, neigh_output.distances
    
    # This is a trick to get lisi metrics to work by adding the tiniest possible value
    # to 0 distance neighbors so that each cell has the same amount of neighbors 
    # (otherwise some cells lose neighbors with distance 0 due to sparse representation)
    row_idx = np.where(distances == 0)[0]
    col_idx = np.where(distances == 0)[1]
    new_row_idx = row_idx[np.where(row_idx != indices[row_idx, col_idx])[0]]
    new_col_idx = col_idx[np.where(row_idx != indices[row_idx, col_idx])[0]]
    distances[new_row_idx, new_col_idx] = (distances[new_row_idx, new_col_idx] +
                                           np.nextafter(0, 1, dtype=np.float32))

    sp_distances, sp_conns = sc.neighbors._compute_connectivities_umap(
            indices[:, :n_neighbors],
            distances[:, :n_neighbors],
            adata.n_obs,
            n_neighbors=n_neighbors)
    adata.obsp[f"{knng_key}_connectivities"] = sp_conns
    adata.obsp[f"{knng_key}_distances"] = sp_distances
    adata.uns[f"{knng_key}_n_neighbors"] = n_neighbors


def convert_to_one_hot(vector: np.ndarray,
                       n_classes: Optional[int]) -> np.array:
    """
    Converts an input 1-D vector of integer labels into a 2-D array of one-hot
    vectors, where for an i'th input value of j, a '1' will be inserted in the
    i'th row and j'th column of the output one-hot vector.
    
    Implementation is adapted from
    https://github.com/theislab/scib/blob/29f79d0135f33426481f9ff05dd1ae55c8787142/scib/metrics/lisi.py#L498
    (05.12.22).

    Parameters
    ----------
    vector:
        Vector to be one-hot-encoded.
    n_classes:
        Number of classes to be considered for one-hot-encoding. If `None`, the
        number of classes will be inferred from `vector`.

    Returns
    ----------
    one_hot:
        2-D NumPy array of one-hot-encoded vectors.

    Example:
    ```
    vector = np.array((1, 0, 4))
    one_hot = _convert_to_one_hot(vector)
    print(one_hot)
    [[0 1 0 0 0]
     [1 0 0 0 0]
     [0 0 0 0 1]]
    ```
    """
    if n_classes is None:
        n_classes = np.max(vector) + 1

    one_hot = np.zeros(shape=(len(vector), n_classes))
    one_hot[np.arange(len(vector)), vector] = 1
    return one_hot.astype(int)