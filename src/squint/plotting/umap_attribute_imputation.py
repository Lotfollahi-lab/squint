from typing import List, Optional, Literal

import scanpy as sc
import anndata as ad
import numpy as np

import matplotlib.pyplot as plt


def plot_umap_attribute_imputation(  
        adata: ad.AnnData,
        embedding_keys: List[Literal['X', 'X_hat', 'X_nbr', 'X_hat_nbr']] = ['X'],
        label_key: str = 'cell_types',
        save_fname: Optional[str] = None,
    ):
    """
    This function plots the UMAP embedding of original and imputed attributes colored by the label.
    
    Parameters
    ----------
    adata : ad.AnnData
        AnnData object with the data to be plotted.
    embedding_keys : List[Literal['X', 'X_hat', 'X_nbr', 'X_hat_nbr']]
        Keys in adata.obsm to use for the embedding.
    label_key : str
        Key in adata.obs to use for the label.
    save_fname : Optional[str]
        Path to save the plot. If None, the plot is not saved.
    
    Returns
    -------
    None
    
    Notes
    -----
    - We use sc.pl.embedding() instead of sc.pl.umap() because scanpy does not use the layer information correctly. umap() checks for basis='X' or basis='X_umap' to use the UMAP embedding without the layer information. So we bypass this and use embedding() directly by setting the basis to the embedding umap key.
    """
    # Create subplots with extra space for legend
    if len(embedding_keys) == 2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    elif len(embedding_keys) == 4:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()  # flatten 2D array to 1D for easier iteration
    else:
        raise ValueError("Number of embedding keys must be either 2 or 4")
    
    handles, labels = None, None
    
    for i, (ax, embedding_key) in enumerate(zip(axes, embedding_keys)):
        sc.pl.embedding(
            adata=adata,
            basis=f'{embedding_key}_attr_umap',
            color=label_key,
            neighbors_key=f'{embedding_key}_attr_neighbors',
            show=False,
            ax=ax,
            legend_loc='right margin' if i == 0 else None,  # enable only for first plot
        )
        ax.set_title(embedding_key)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks([])
        ax.set_yticks([])        

        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
            ncol = max(1, int(np.ceil(len(labels)/25)))
            ax.get_legend().remove()  # remove it from the first plot and draw it globally
        
    # Add single global legend
    fig.subplots_adjust(right=0.82)  # shrink main plot area to make space
    fig.legend(
        handles, labels,
        title=label_key,
        loc='center left',
        bbox_to_anchor=(0.85, 0.5),
        ncol=ncol,
        borderaxespad=0.1,
        frameon=False,
    )

    if save_fname:
        plt.savefig(
            fname=save_fname,
            dpi=300,
            bbox_inches='tight',
        )
    else:
        plt.show()
    
    
def compute_umap(
        adata: ad.AnnData,
        embedding_keys: List[Literal['X', 'X_hat', 'X_nbr', 'X_hat_nbr']] = ['X'],
    ) -> ad.AnnData:
    """
    This function computes the nearest neighbor distance matrix and neighborhood graph,
    and embeds the neighborhood into 2D using UMAP.
    
    Parameters
    ----------
    adata : ad.AnnData
        AnnData object with the data to be processed.
    embedding_keys : List[Literal['X', 'X_hat', 'X_nbr', 'X_hat_nbr']]
        Key(s) in adata.obsm to compute UMAP embeddings of.
    
    Returns
    -------
    adata : ad.AnnData
        AnnData object with the UMAP embedding(s) added to adata.obsm.
        
    Notes
    -----
    - The X and X_hat are already keys in the adata.uns containing the original and reconstructed gene counts as tensors. 
    - sc.pp.neighbors() adds the neighbors to the adata.uns[key_added] and the distances are stored in adata.obsp[key_added+'_distances'] and connectivities in adata.obsp[key_added+'_connectivities'].
    - This would overwrite the adata.uns['X'] and adata.uns['X_hat'] keys. Therefore, we set the key_added to f'{embedding_key}_attr_neighbors'.
    - sc.tl.umap() adds the embedding to the adata.obsm[key_added] and the parameters in adata.uns[key_added].
    """
    for embedding_key in embedding_keys:
        print(f"Computing UMAP for {embedding_key}..")
        
        # convert the tensor to numpy array
        if embedding_key in ['X_hat', 'X_nbr', 'X_hat_nbr']:
            adata.obsm[embedding_key] = adata.uns[embedding_key].cpu().numpy()

        # compute nearest neighbor distance matrix and neighborhood graph
        print(f"Computing nearest neighbor distance matrix and neighborhood graph for {embedding_key}..")
        sc.pp.neighbors(
            adata=adata,
            n_neighbors=15, # default is 15
            knn=True, # default is True
            random_state=42,
            use_rep=embedding_key,
            key_added=f'{embedding_key}_attr_neighbors',
        )

        # embed the neighborhood into 2D using UMAP
        print(f"Embedding the neighborhood into 2D using UMAP for {embedding_key}..")
        sc.tl.umap(
            adata=adata,
            min_dist=0.5, # default is 0.5
            spread=1.0, # default is 1.0
            alpha=1.0, # default is 1.0
            gamma=1.0, # default is 1.0
            negative_sample_rate=5, # default is 5
            random_state=42,
            neighbors_key=f'{embedding_key}_attr_neighbors',
            key_added=f'{embedding_key}_attr_umap',
        )

    return adata