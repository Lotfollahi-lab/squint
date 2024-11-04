import anndata as ad
import squidpy as sq
from typing import Optional


def spatial_neighbors(adata: ad.AnnData,
                      coord_type: str = 'generic',
                      spatial_key: str = 'spatial',
                      delaunay: Optional[bool] = True,
                      radius: Optional[float] = None,
                      set_diag: bool = True,
                      ) -> ad.AnnData:
    """
    Construct a neighborhood graph using Delaunay Triangulation (default) or radius.

    Parameters
    ----------
    adata:
        AnnData object with spatial coordinates available in
        `adata.obsm['spatial']`.
    coord_type:
        The type of spatial coordinates. Default is 'generic'.
    spatial_key:
        Key in `adata.obsm` where spatial coordinates are stored.
    delaunay:
        If `True`, use Delaunay Triangulation to construct the neighborhood graph.
        Default is `True`.
    radius:
        If set, construct the neighborhood graph using a fixed radius.
        Default is `None`.
    set_diag:
        If `True`, set the diagonal of the connectivity matrix to `True`.
        Default is `True`.
    key_prefix:
        Key prefix in `adata.obsp` where the spatial connectivities are stored.
        Default is 'spatial-delaunay'. The final key will be `{key_prefix}_connectivities`.

    Returns
    -------
    ad.AnnData:
        AnnData object with spatial connectivities and aggregated neighborhood gene counts in counts available in `adata.obsp['spatial_connectivities']` and `adata.layers['X_neighborhood']`, respectively.
    key_prefix:
        The key prefix used to store the spatial connectivities in `adata.obsp`.
    """
    # Ensure that only one of `delaunay` and `radius` is set
    assert (delaunay & (radius is None)) | (not delaunay & (radius is not None))

    if delaunay:
        key_prefix = f"{spatial_key}-delaunay"
    else:
        key_prefix = f"{spatial_key}-radius-{radius}"

    sq.gr.spatial_neighbors(adata=adata,
                            coord_type=coord_type,
                            spatial_key=spatial_key,
                            radius=radius,
                            delaunay=delaunay,
                            set_diag=set_diag,
                            key_added=key_prefix)

    return adata, key_prefix
