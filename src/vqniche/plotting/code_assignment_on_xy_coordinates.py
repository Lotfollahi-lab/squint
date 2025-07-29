from typing import Optional

import anndata as ad

import seaborn as sns
import matplotlib.pyplot as plt


def plot_code_assignment_on_xy_coordinates(
        adata: ad.AnnData,
        spatial_key: str = "spatial",
        label_key: str = "Indices",
        save_fname: Optional[str] = None,
        title: Optional[str] = "X-Y coordinates colored by Code Indices",
    ) -> None:
    """
    Plots the X-Y coordinates of the cells colored by the code indices.

    Parameters
    ----------
    adata : ad.AnnData
        The AnnData object containing the data to plot.
    spatial_key : str, optional
        The key in adata.obsm to use for the spatial coordinates.
    label_key : str, optional
        The key in adata.obs to use for the label.
    save_fname : str, optional
        The file name to save the plot to.
    title : str, optional
        The title of the plot.

    Returns
    -------
    None
    """
    # plot the X-Y coordinates of the cells colored by the code indices
    fig, ax = plt.subplots(1,1, figsize=(4,3))
    
    sns.scatterplot(
        x=adata.obsm[spatial_key][:,0],
        y=adata.obsm[spatial_key][:,1],
        hue=adata.obs[label_key]
    )

    ax.set_title(title)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])

    handles, labels = ax.get_legend_handles_labels()
    ax.get_legend().remove()

    fig.legend(
            handles, labels,
            title=label_key,
            loc='center left',
            bbox_to_anchor=(0.95, 0.5),
            borderaxespad=0.1,
            frameon=False,
        )

    if save_fname:
        plt.savefig(
            save_fname,
            dpi=300,
            bbox_inches='tight',
        )
    else:
        plt.show()    