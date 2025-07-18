from typing import List

import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import anndata as ad


def plot_umap_attribute_imputation(  
        adata: ad.AnnData,
        embedding_keys: list[str],
        label_key: str = 'cell_types',
    ):
    """
    This function plots the UMAP embedding of original and imputed attributes colored by the label.
    
    Parameters
    ----------
    adata : ad.AnnData
        AnnData object with the data to be plotted.
    embedding_keys : list[str]
        Keys in adata.obsm to use for the embedding.
    label_key : str
        Key in adata.obs to use for the label.
    
    Returns
    -------
    None
    """
    # Create subplots with extra space for legend
    fig, axes = plt.subplots(1, len(embedding_keys), figsize=(12, 4))
    if len(embedding_keys) == 1:
        axes = [axes]
    
    flag = 1
    
    for ax, embedding_key in zip(axes, embedding_keys):
        if flag:
            legend_loc = None
        else:
            legend_loc = 'right margin'
        sc.pl.umap(
            adata=adata,
            color=label_key,
            layer=f'{embedding_key}_umap',
            neighbors_key=f'{embedding_key}_neighbors',
            show=False,
            ax=ax,
            legend_loc=legend_loc,  # Show legend only on first plot
        )
        flag = 0
        
        ax.set_title(embedding_key)
        
        # Add title to the legend on the first plot
        if flag == 0:  # This is the first plot (flag was 1, now set to 0)
            legend = ax.get_legend()
            if legend:
                legend.set_title(label_key)
        
    plt.show()
    
    
def compute_umap(
        adata: ad.AnnData,
        embedding_keys: List[str] = ['X'],
    ) -> ad.AnnData:
    """
    This function computes the nearest neighbor distance matrix and neighborhood graph,
    and embeds the neighborhood into 2D using UMAP.
    
    Parameters
    ----------
    adata : ad.AnnData
        AnnData object with the data to be processed.
    embedding_keys : List[str]
        Key(s) in adata.obsm to compute UMAP embeddings of.
    
    Returns
    -------
    adata : ad.AnnData
        AnnData object with the UMAP embedding(s) added to adata.obsm.
    """
    for embedding_key in embedding_keys:
        print(f"Computing UMAP for {embedding_key}..")
        if embedding_key == 'X':
            print(f"{np.mean(adata.X)=}")
        elif embedding_key == 'X_hat':
            adata.obsm['X_hat'] = adata.uns['X_hat'].cpu().numpy()
            print(f"{np.mean(adata.obsm['X_hat'])=}")
        # compute nearest neighbor distance matrix and neighborhood graph
        sc.pp.neighbors(
            adata=adata,
            n_neighbors=15, # default is 15
            knn=True, # default is True
            random_state=42,
            use_rep=embedding_key,
            key_added=f'{embedding_key}_neighbors',
        )
        #  The neighbors data is added to .uns[key_added], distances are stored in .obsp[key_added+'_distances'] and connectivities in .obsp[key_added+'_connectivities'].

        # embed the neighborhood into 2D using UMAP
        sc.tl.umap(
            adata=adata,
            min_dist=0.5, # default is 0.5
            spread=1.0, # default is 1.0
            alpha=1.0, # default is 1.0
            gamma=1.0, # default is 1.0
            negative_sample_rate=5, # default is 5
            random_state=42,
            neighbors_key=f'{embedding_key}_neighbors',
            key_added=f'{embedding_key}_umap',
        )
        # The embedding is stored as obsm[key_added] and the the parameters in uns[key_added].

    return adata