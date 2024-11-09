import anndata as ad
import squidpy as sq
from typing import Optional


def spatial_neighbors(adata: ad.AnnData,
                      coord_type: str = 'generic',
                      spatial_key: str = 'spatial',
                      delaunay: Optional[bool] = True,
                      radius: float = 55.0,
                      set_diag: bool = True,
                      key_added: str = 'spatial_delaunay'
                      ) -> ad.AnnData:
    """
    Construct a neighborhood graph using Delaunay Triangulation (default) and/or radius.

    Parameters
    ----------
    - adata: AnnData
        AnnData object with spatial coordinates available in
        `adata.obsm['spatial']`.
    - coord_type: str, optional (default: 'generic')
        The type of spatial coordinates.
    - spatial_key: str, optional (default: 'spatial')
        Key in `adata.obsm` where spatial coordinates are stored.
    - delaunay: bool, optional (default: True)
        If `True`, use Delaunay Triangulation to construct the neighborhood graph.
    - radius: float, optional (default: 55.0)
        If not `None`, construct neighborhood graph using radius.
        If `delaunay` is `True`, the radius is used to filter the Delaunay Triangulation.
    - set_diag: bool, optional (default: True)
        If `True`, set the diagonal of the connectivity matrix to `True`.
    - key_added: str, optional (default: 'spatial_delaunay')
        Prefix in `adata.obsp` where the spatial connectivities are stored. The final key will be `{key_added}_connectivities`.

    Returns
    -------
    - ad.AnnData:
        AnnData object with spatial connectivities available in `adata.obsp[f'{key_added}_connectivities']`.
    """
    assert delaunay or radius is not None, "Either `delaunay` or `radii` must be provided."

    sq.gr.spatial_neighbors(adata=adata,
                            coord_type=coord_type,
                            spatial_key=spatial_key,
                            delaunay=delaunay,
                            radius=radius,
                            set_diag=set_diag,
                            key_added=key_added)

    return adata
