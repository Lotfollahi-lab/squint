import anndata as ad
import squidpy as sq
from typing import List, Optional


def spatial_neighbors(adata: ad.AnnData,
                      coord_type: str = 'generic',
                      spatial_key: str = 'spatial',
                      delaunay: Optional[bool] = True,
                      radii: Optional[List[float]] = [55.0],
                      set_diag: bool = True,
                      ) -> ad.AnnData:
    """
    Construct a neighborhood graph using Delaunay Triangulation (default) and/or radius.

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
    radii:
        If not `None`, construct one neighborhood graph per radius in `radii`.
        Default is one radius-based neighborhood graph with radius 55 microns.
    set_diag:
        If `True`, set the diagonal of the connectivity matrix to `True`.
        Default is `True`.
    key_prefix:
        Key prefix in `adata.obsp` where the spatial connectivities are stored.
        Default is 'spatial-delaunay'. The final key will be `{key_prefix}_connectivities`.

    Returns
    -------
    ad.AnnData:
        AnnData object with spatial connectivities available in `adata.obsp[f'{key_prefix}_spatial_connectivities']`.
    List[str]:
        The list of key prefixes used to store the spatial connectivities in `adata.obsp`.
    """
    assert delaunay or len(radii) >= 1, "Either `delaunay` or `radii` must be provided."

    key_prefixes = []

    if delaunay:
        key_prefix = f"{spatial_key}_delaunay"
        key_prefixes.append(key_prefix)
        print(f"Constructing neighborhood graph using Delaunay Triangulation...")
        sq.gr.spatial_neighbors(adata=adata,
                                coord_type=coord_type,
                                spatial_key=spatial_key,
                                radius=None,
                                delaunay=True,
                                set_diag=set_diag,
                                key_added=key_prefix)

    if len(radii) >= 1:
        for radius in radii:
            key_prefix = f"{spatial_key}_radius_{radius}"
            key_prefixes.append(key_prefix)
            print(f"Constructing neighborhood graph using radius {radius}...")
            sq.gr.spatial_neighbors(adata=adata,
                                    coord_type=coord_type,
                                    spatial_key=spatial_key,
                                    radius=radius,
                                    delaunay=False,
                                    set_diag=set_diag,
                                    key_added=key_prefix)

    return adata, key_prefixes
