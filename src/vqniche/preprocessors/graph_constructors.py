import anndata as ad
import squidpy as sq
from typing import Optional


def set_edge_index_name(
    spatial_key: str = 'spatial',
    delaunay: Optional[bool] = True,
    n_neighs: Optional[int] = None,
    radius: Optional[float] = None,
    ) -> str:
    """
    Set the name of the edge index based on the graph construction parameters.

    Parameters
    ----------
    - spatial_key: str, optional (default: 'spatial')
        Key in `adata.obsm` where spatial coordinates are stored.
    - delaunay: bool, optional (default: True)
        If `True`, append '_delaunay' to the edge index name.
    - n_neighs: float, optional (default: None)
        If not `None`, append '_n_neighs_{n_neighs}' to the edge index name.
    - radius: float, optional (default: None)
        If not `None`, append '_radius_{radius}' to the edge index name.

    Returns
    -------
    - str:
        Name of the edge index based on the graph construction parameters.
    """
    if delaunay:
        edge_index_name = f"{spatial_key}_delaunay"
    else:
        edge_index_name = f"{spatial_key}"

    if not delaunay and n_neighs is not None:
        edge_index_name += f"_n_neighs_{n_neighs}"

    if radius is not None:
        edge_index_name += f"_radius_{radius}"

    return edge_index_name


def spatial_neighbors(adata: ad.AnnData,
                      coord_type: str = 'generic',
                      spatial_key: str = 'spatial',
                      delaunay: Optional[bool] = True,
                      n_neighs: Optional[int] = None,
                      radius: Optional[float] = None,
                      include_self_loop: bool = True,
                      key_added: str = 'spatial_delaunay'
                      ) -> ad.AnnData:
    """
    Construct a neighborhood graph using Delaunay Triangulation (default) and/or radius and/or number of neighbors.

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
        If `True`, use Delaunay Triangulation to construct the neighborhood graph. If 'radius' is not `None`, the Delaunay Triangulation is filtered by the radius. If 'n_neighs' is not `None`, the Delaunay Triangulation is filtered by the number of neighbors.
    - n_neighs: float, optional (default: None)
        If not `None`, construct neighborhood graph using 'n_neighs' nearest neighbors.
    - radius: float, optional (default: None)
        If not `None`, construct neighborhood graph using radius.
    - include_self_loop: bool, optional (default: True)
        If `True`, set the diagonal of the connectivity matrix to `True`.
    - key_added: str, optional (default: 'spatial_delaunay')
        Prefix in `adata.obsp` where the spatial connectivities are stored. The final key will be `{key_added}_connectivities`.

    Returns
    -------
    - ad.AnnData:
        AnnData object with spatial connectivities available in `adata.obsp[f'{key_added}_connectivities']`.
    """
    # we separate the construction of the spatial neighbors into two cases:
    # 1. if delaunay is true, radius is used if provided. squidpy ignores n_neighs. so we leave it as default which is 6.
    if delaunay:
        sq.gr.spatial_neighbors(
            adata=adata,
            coord_type=coord_type,
            spatial_key=spatial_key,
            delaunay=True,
            radius=radius,
            set_diag=include_self_loop,
            key_added=key_added,
        )
    # 2. if delaunay is false, n_neighs is used if provided. graph is then pruned by radius if provided.
    else:
        sq.gr.spatial_neighbors(
            adata=adata,
            coord_type=coord_type,
            spatial_key=spatial_key,
            delaunay=False,
            n_neighs=n_neighs,
            radius=radius,
            set_diag=include_self_loop,
            key_added=key_added,
        )

    return adata
