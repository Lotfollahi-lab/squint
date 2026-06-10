"""
Plot ground-truth cell-type and niche labels on tissue for the
chl59-8b_1p (CosMx Lung) dataset.

For each silver `.h5ad` under `--silver-dir`, render a 1x2 figure:
  - left panel:  cells coloured by `cell_type` obs column
  - right panel: cells coloured by `niche` obs column
The figure suptitle is "Donor X" if the donor has a single section, or
"Donor X - Replicate Y" if multiple sections share a donor.

Donor / replicate identification (in priority order; the first source
that yields a non-empty value is used):
  1. Explicit `--donor-from`/`--replicate-from` (e.g.
     `--donor-from obs:donor_id` to read `adata.obs['donor_id']`).
  2. `adata.uns['donor_id']` / `adata.uns['donor']`.
  3. `adata.obs['donor_id'].iloc[0]` (assumed constant per AnnData).
  4. Filename regex (default: `lung(\\d+)` -> "Lung<N>" for the donor,
     `(?:rep|replicate|r)(\\d+)` -> "Replicate <N>" for the replicate).
Replicate IDs are auto-numbered (1, 2, ...) only when multiple
sections share a donor AND no explicit replicate id was found.

Output (per section): `<out_dir>/<file_stem>.{png,svg}`, plus an optional
combined `<out_dir>/chl59_ground_truth.{png,svg}` figure with all sections
in a grid (one row per section, two columns: cell_type / niche). SVGs
are written with `svg.fonttype='none'` so all text is editable in
Illustrator / Inkscape.

Usage:
    python examples/plot_chl59_ground_truth.py \\
        --silver-dir /nfs/team361/sb75/DATASETS/silver/chl59-8b_1p

    # custom obs columns (some CosMx exports use `niche_label`):
    python examples/plot_chl59_ground_truth.py \\
        --silver-dir /nfs/.../silver/chl59-8b_1p \\
        --cell-type-key cell_type --niche-key niche_label

    # only the combined figure:
    python examples/plot_chl59_ground_truth.py \\
        --silver-dir /nfs/.../silver/chl59-8b_1p \\
        --no-per-section
"""

import argparse
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Match warning suppression in run_squint.py
warnings.filterwarnings(
    "ignore",
    message=r".*Importing read_text from `anndata` is deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*legacy Dask DataFrame implementation is deprecated.*",
    category=FutureWarning,
)

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Editable text in SVG (don't outline glyphs to paths). Same convention
# as plot_holdout_regions.py.
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _save_dual(fig, out_path: Path, **savefig_kwargs) -> None:
    """Save figure as both `.png` and `.svg` siblings.

    For SVG: text stays as <text> nodes (svg.fonttype='none' set
    globally), but any artist with `rasterized=True` is embedded as a
    PNG-encoded `<image>` element at the requested `dpi`. This gives a
    "hybrid" SVG where the dense scatter renders as a single raster
    layer (small file, fast) while titles / legends / axes stay
    editable as vectors.

    `dpi` therefore matters for both saves:
      - PNG: full-figure raster resolution.
      - SVG: resolution of embedded rasterised artists ONLY (the
        rest of the figure remains vector).
    """
    out_path = Path(out_path)
    fig.savefig(out_path.with_suffix(".png"), **savefig_kwargs)
    fig.savefig(out_path.with_suffix(".svg"), **savefig_kwargs)


# ---------------------------------------------------------------------------
# Donor / replicate parsing
# ---------------------------------------------------------------------------

# Default regexes for filename-based fallback. Captures the integer.
_DEFAULT_DONOR_RE = re.compile(r"lung(\d+)", re.IGNORECASE)
_DEFAULT_REPLICATE_RE = re.compile(r"(?:rep|replicate|r)(\d+)", re.IGNORECASE)


def _resolve_from_spec(adata: ad.AnnData, spec: Optional[str]) -> Optional[str]:
    """`spec` is like "obs:colname" / "uns:keyname" / a literal string.
    Returns the first non-empty string found, or None."""
    if not spec:
        return None
    if ":" not in spec:
        return spec  # treat as literal
    src, key = spec.split(":", 1)
    src = src.lower()
    if src == "obs" and key in adata.obs.columns:
        vals = adata.obs[key].astype(str)
        # Constant-per-section assumption — take the first value.
        v = next((s for s in vals if s and s != "nan"), None)
        return v
    if src == "uns" and key in adata.uns:
        v = adata.uns[key]
        return str(v) if v is not None else None
    return None


def _parse_donor(adata: ad.AnnData, filename: str,
                 cli_spec: Optional[str]) -> Optional[str]:
    """Try CLI spec, then uns/obs conventions, then filename regex."""
    # 1) Explicit CLI override
    v = _resolve_from_spec(adata, cli_spec)
    if v:
        return v
    # 2) common uns keys
    for k in ("donor_id", "donor", "Donor", "DonorID"):
        if k in adata.uns and adata.uns[k] is not None:
            return str(adata.uns[k])
    # 3) common obs columns (constant per section)
    for k in ("donor_id", "donor", "Donor"):
        if k in adata.obs.columns:
            vals = adata.obs[k].astype(str)
            v = next((s for s in vals if s and s != "nan"), None)
            if v:
                return v
    # 4) filename regex
    m = _DEFAULT_DONOR_RE.search(filename)
    if m:
        return f"Lung{m.group(1)}"
    return None


def _parse_replicate(adata: ad.AnnData, filename: str,
                     cli_spec: Optional[str]) -> Optional[str]:
    """Try CLI spec, then uns/obs conventions, then filename regex.
    Returns a SHORT identifier (e.g. "1", "2", "Rep1") — the caller
    formats it into the final label."""
    v = _resolve_from_spec(adata, cli_spec)
    if v:
        return v
    for k in ("replicate", "rep", "Replicate", "ReplicateID"):
        if k in adata.uns and adata.uns[k] is not None:
            return str(adata.uns[k])
    for k in ("replicate", "rep", "Replicate"):
        if k in adata.obs.columns:
            vals = adata.obs[k].astype(str)
            v = next((s for s in vals if s and s != "nan"), None)
            if v:
                return v
    m = _DEFAULT_REPLICATE_RE.search(filename)
    if m:
        return m.group(1)
    return None


def _normalise_replicate_id(rep) -> str:
    """Strip + lowercase + drop common 'rep'/'replicate'/'r' prefixes.
    Used to compare a section's replicate id (parsed from filename /
    obs / uns) against the spec given on `--query-donors`."""
    s = str(rep).strip().lower()
    for prefix in ("replicate", "rep", "r"):
        if s.startswith(prefix):
            tail = s[len(prefix):].lstrip("_- ")
            if tail.isdigit() or tail.isalnum():
                return tail
    return s


def _parse_query_spec(
        spec: Optional[str],
    ) -> Dict[str, Optional[set]]:
    """Parse the `--query-donors` CLI value into a per-donor map.

    Syntax: comma-separated entries of either
      - `<donor>`              -> the WHOLE donor is query (every replicate)
      - `<donor>:<replicate>`  -> only that replicate of that donor is
                                  query; other replicates of the donor
                                  are reference.
    Multiple replicates of one donor can be specified by listing the
    donor multiple times: `Lung5:1,Lung5:3,Lung13`.

    Returns a dict mapping `donor_lc -> set_of_replicate_ids`, where
    `set_of_replicate_ids = None` means "whole-donor query" (all
    replicates). Empty / None spec returns {}.
    """
    out: Dict[str, Optional[set]] = {}
    if not spec:
        return out
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            donor, rep = token.split(":", 1)
            donor_lc = donor.strip().lower()
            rep_n = _normalise_replicate_id(rep)
            if donor_lc not in out or out[donor_lc] is None:
                # If we previously saw a whole-donor entry for this
                # donor, the replicate-restricted form is more
                # specific — but the simpler-to-reason-about behaviour
                # is "whole-donor wins". We keep `None` if already set.
                if out.get(donor_lc) is None and donor_lc in out:
                    pass
                else:
                    out.setdefault(donor_lc, set()).add(rep_n)  # type: ignore[union-attr]
            elif isinstance(out[donor_lc], set):
                out[donor_lc].add(rep_n)
        else:
            # Whole donor: overrides any prior replicate-restricted
            # entry for this donor.
            out[token.strip().lower()] = None
    return out


def _resolve_section_titles(
        meta: List[Dict],
        query_spec: Optional[Dict[str, Optional[set]]] = None,
    ) -> List[str]:
    """Given a list of {donor, replicate, filename} dicts, return one
    title per entry:
      - Donor with 1 section -> "Donor X"
      - Donor with >1 sections -> "Donor X - Replicate Y"
        (auto-numbering replicates 1..K within a donor if the
        per-section replicate id was missing).

    If `query_spec` is provided (built by `_parse_query_spec`), every
    title gets a parenthesised role suffix:
      - " (query)"     when this section matches the spec
      - " (reference)" otherwise
    A donor mapping to `None` matches every section of that donor; a
    donor mapping to a SET of replicate ids matches only sections whose
    replicate (after normalisation: drops 'rep'/'replicate'/'r' prefix
    and lowercases) is in that set.
    """
    # Bucket sections by donor
    by_donor: Dict[str, List[int]] = {}
    for i, m in enumerate(meta):
        donor = m["donor"] or m["filename"]
        by_donor.setdefault(donor, []).append(i)

    titles: List[Optional[str]] = [None] * len(meta)
    # Replicate IDs assigned per section (raw or auto-numbered) — kept
    # so the role-suffix step can match them against the query spec.
    section_rep_ids: List[Optional[str]] = [None] * len(meta)
    for donor, idxs in by_donor.items():
        if len(idxs) == 1:
            titles[idxs[0]] = (
                f"Donor {donor}"
                if not str(donor).startswith("Donor") else str(donor)
            )
            section_rep_ids[idxs[0]] = meta[idxs[0]]["replicate"]
            continue
        # Multiple replicates per donor: format with replicate label.
        # If at least one replicate id was found, use those; otherwise
        # auto-number 1..K in filename-sort order so the suffix is at
        # least deterministic.
        # Sort idxs by filename for stable numbering.
        idxs_sorted = sorted(idxs, key=lambda j: meta[j]["filename"].lower())
        for k, j in enumerate(idxs_sorted, start=1):
            rep = meta[j]["replicate"]
            if not rep:
                rep = str(k)
            section_rep_ids[j] = rep
            # If the replicate already says "Rep<N>"-ish, don't double-
            # prefix; otherwise add "Replicate ".
            rep_label = (
                str(rep) if str(rep).lower().startswith(("rep", "replicate"))
                else f"Replicate {rep}"
            )
            donor_label = (
                str(donor) if str(donor).startswith("Donor")
                else f"Donor {donor}"
            )
            titles[j] = f"{donor_label} - {rep_label}"

    # Append query/reference role suffix.
    if query_spec:
        for i, m in enumerate(meta):
            donor_lc = (m["donor"] or m["filename"]).strip().lower()
            role = "reference"
            if donor_lc in query_spec:
                allowed = query_spec[donor_lc]
                if allowed is None:
                    # Whole-donor query
                    role = "query"
                else:
                    # Only specified replicates are query
                    rep_n = _normalise_replicate_id(section_rep_ids[i] or "")
                    if rep_n in allowed:
                        role = "query"
            titles[i] = f"{titles[i]} ({role})"

    return [t or "" for t in titles]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _categorical_palette(values: pd.Series, cmap_name: str = "tab20") -> Dict[str, str]:
    """Build a {category -> hex} mapping. Stable ordering via sorted unique."""
    cats = sorted(set(values.astype(str)) - {"nan", ""})
    if not cats:
        return {}
    cmap = mpl.colormaps.get_cmap(cmap_name)
    n = len(cats)
    colors = [mpl.colors.to_hex(cmap(i % cmap.N)) for i in range(n)]
    return dict(zip(cats, colors))


def _scatter_labels(ax, xy: np.ndarray, labels: pd.Series,
                    palette: Dict[str, str], spot_size: float,
                    title: str, legend: bool = True,
                    legend_markersize: float = 9.0) -> None:
    """One categorical scatter on `ax`. Cells with NaN/empty labels are
    drawn in light grey first so they don't visually dominate.

    `legend_markersize` controls the dot size in the legend INDEPENDENT
    of `spot_size`. The scatter `s` value is tuned for tissue density
    (small dots so cells don't blob together); legend dots need to be
    visible at human-reading scale regardless. We use explicit Line2D
    handles with `markersize` instead of relying on the scatter's
    `label=` auto-handles, which inherit the scatter's tiny `s`.
    """
    from matplotlib.lines import Line2D

    # `rasterized=True` makes the scatter render as a single embedded
    # PNG inside the SVG output (resolution = savefig dpi). The rest
    # of the figure (titles, legends, axis ticks) stays vector. Two
    # benefits: (a) SVG file size is bounded regardless of n_cells,
    # and (b) Illustrator / Inkscape don't choke on 100k-cell scatter
    # path collections.
    s = max(spot_size, 0.4)
    mask_known = labels.astype(str).isin(palette.keys()).to_numpy()
    if (~mask_known).any():
        ax.scatter(
            xy[~mask_known, 0], xy[~mask_known, 1],
            c="#dddddd", s=s, linewidths=0, marker="o",
            rasterized=True,
        )
    # Draw each category in its own scatter call (no `label=` here —
    # the legend uses fixed-size Line2D handles below).
    for cat, color in palette.items():
        m = (labels.astype(str) == cat).to_numpy()
        if not m.any():
            continue
        ax.scatter(
            xy[m, 0], xy[m, 1],
            c=color, s=s, linewidths=0, marker="o",
            rasterized=True,
        )
    ax.set_aspect("equal")
    ax.invert_yaxis()  # match scanpy spatial convention
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=12)
    if legend and palette:
        # Build explicit handles so legend marker size is independent
        # of the (tiny) scatter `s`.
        handles = [
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=color, markeredgewidth=0,
                   markersize=legend_markersize, label=cat)
            for cat, color in palette.items()
            # Skip categories with no cells in this section
            if (labels.astype(str) == cat).any()
        ]
        if handles:
            ax.legend(
                handles=handles,
                loc="center left", bbox_to_anchor=(1.02, 0.5),
                frameon=False, fontsize=9,
                handletextpad=0.4, borderaxespad=0.0,
            )


def _spot_size_for(xy: np.ndarray, n_cells: int) -> float:
    """Default matplotlib scatter `s` (= marker area in points^2).

    `scatter(s=...)` is INDEPENDENT of data coordinates — it's a
    figure-units value. So coord extent doesn't tell us anything
    useful here; only the rendered figure size + cell count matter.

    For ~100k-cell CosMx Lung sections in our 13-inch-wide layout,
    `s=1` produces single-pixel dots that read as a continuous tissue
    while still letting clusters be distinguishable by colour. Sparse
    sections (~few-thousand cells) tolerate larger; we cap at `s=4`
    via a log-scale ramp so the auto-default stays in a sane range
    across the chl59-8b_1p sections.

    For finer control, pass `--spot-size <N>` on the CLI.
    """
    if n_cells <= 0:
        return 2.0
    if n_cells >= 50_000:
        return 0.5    # very dense: hairline dots, individual cells visible
    if n_cells >= 20_000:
        return 1.0
    if n_cells >= 5_000:
        return 2.0
    return 4.0        # sparse section: bigger markers so dots aren't 1px


def _render_section(
        adata: ad.AnnData,
        cell_type_key: str,
        niche_key: str,
        title: str,
        out_path: Path,
        dpi: int,
        spot_size: Optional[float] = None,
    ) -> None:
    """One figure per section: cell_type | niche.

    `spot_size` overrides the per-cell-density auto-pick — useful when
    auto-sizing produces dots that still overlap on a particularly
    dense section.
    """
    if "spatial" not in adata.obsm:
        raise SystemExit(f"AnnData missing obsm['spatial']: {out_path}")
    xy = np.asarray(adata.obsm["spatial"], dtype=float)[:, :2]
    spot = (
        float(spot_size) if spot_size is not None
        else _spot_size_for(xy, adata.n_obs)
    )

    cell_labels = adata.obs[cell_type_key] if cell_type_key in adata.obs.columns else pd.Series(["?"] * adata.n_obs)
    niche_labels = adata.obs[niche_key] if niche_key in adata.obs.columns else pd.Series(["?"] * adata.n_obs)

    cell_palette = _categorical_palette(cell_labels, "tab20")
    niche_palette = _categorical_palette(niche_labels, "tab10")

    fig, axes = plt.subplots(
        1, 2,
        figsize=(13, 6),
        gridspec_kw={"wspace": 0.4},
    )
    _scatter_labels(axes[0], xy, cell_labels, cell_palette, spot,
                    title=f"cell_type ({cell_type_key})")
    _scatter_labels(axes[1], xy, niche_labels, niche_palette, spot,
                    title=f"niche ({niche_key})")
    fig.suptitle(title, fontsize=15, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save_dual(fig, out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _render_combined(
        rendered: List[Dict],
        cell_type_key: str,
        niche_key: str,
        out_path: Path,
        dpi: int,
        spot_size: Optional[float] = None,
    ) -> None:
    """Combined figure: one row per section, two columns (cell_type, niche).
    Shares a single legend per column to avoid N copies of the same
    legend across rows."""
    n = len(rendered)
    if n == 0:
        return
    fig, axes = plt.subplots(
        n, 2,
        figsize=(13, 6 * n),
        gridspec_kw={"wspace": 0.35, "hspace": 0.25},
        squeeze=False,
    )

    # Build palettes from the UNION of categories across all sections so
    # colours are consistent across rows.
    all_cell = pd.concat(
        [r["adata"].obs[cell_type_key] for r in rendered if cell_type_key in r["adata"].obs.columns],
        ignore_index=True,
    ) if any(cell_type_key in r["adata"].obs.columns for r in rendered) else pd.Series([])
    all_niche = pd.concat(
        [r["adata"].obs[niche_key] for r in rendered if niche_key in r["adata"].obs.columns],
        ignore_index=True,
    ) if any(niche_key in r["adata"].obs.columns for r in rendered) else pd.Series([])
    cell_palette  = _categorical_palette(all_cell, "tab20")
    niche_palette = _categorical_palette(all_niche, "tab10")

    for i, r in enumerate(rendered):
        a = r["adata"]
        xy = np.asarray(a.obsm["spatial"], dtype=float)[:, :2]
        spot = (
            float(spot_size) if spot_size is not None
            else _spot_size_for(xy, a.n_obs)
        )
        cell_labels  = a.obs[cell_type_key] if cell_type_key in a.obs.columns else pd.Series(["?"] * a.n_obs)
        niche_labels = a.obs[niche_key]     if niche_key     in a.obs.columns else pd.Series(["?"] * a.n_obs)
        # Only show legend on the FIRST row (categories already drawn from
        # the global palette, so labels are consistent across rows).
        show_legend = (i == 0)
        _scatter_labels(axes[i, 0], xy, cell_labels,  cell_palette,  spot,
                        title=f"{r['title']} — cell_type", legend=show_legend)
        _scatter_labels(axes[i, 1], xy, niche_labels, niche_palette, spot,
                        title=f"{r['title']} — niche",     legend=show_legend)

    fig.suptitle("CosMx Lung — ground-truth cell types and niches", fontsize=16, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    _save_dual(fig, out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--silver-dir", type=str,
        default="/nfs/team361/sb75/DATASETS/silver/chl59-8b_1p",
        help="Folder containing the chl59-8b_1p .h5ad files. "
             "Default: /nfs/team361/sb75/DATASETS/silver/chl59-8b_1p",
    )
    p.add_argument(
        "--out-dir", type=str,
        default="/nfs/team361/sb75/squint-reproducibility/artifacts/"
                "dataset_preparation/chl59-8b_1p/ground_truth",
        help="Where to write the plots. "
             "Default: <ARTIFACTS>/dataset_preparation/chl59-8b_1p/ground_truth",
    )
    p.add_argument(
        "--cell-type-key", type=str, default="cell_type",
        help="obs column for cell-type labels. Default: cell_type",
    )
    p.add_argument(
        "--niche-key", type=str, default="niche",
        help="obs column for niche labels. Default: niche",
    )
    p.add_argument(
        "--donor-from", type=str, default=None,
        help="Override donor source: 'obs:<col>' / 'uns:<key>'. "
             "Default: try uns['donor_id']/uns['donor'], "
             "obs['donor_id']/obs['donor'], then filename regex 'lung\\d+'.",
    )
    p.add_argument(
        "--replicate-from", type=str, default=None,
        help="Override replicate source: 'obs:<col>' / 'uns:<key>'. "
             "Default: try uns['replicate']/uns['rep'], "
             "obs['replicate']/obs['rep'], then filename regex.",
    )
    p.add_argument(
        "--combined-out", type=str, default="chl59_ground_truth",
        help="Stem for the combined figure (1 row per section, 2 cols). "
             "Set '' to skip. Default: 'chl59_ground_truth'.",
    )
    p.add_argument(
        "--no-per-section", action="store_true",
        help="Skip the per-section .png/.svg files (only emit combined).",
    )
    p.add_argument(
        "--spot-size", type=float, default=None,
        help="Forwarded to matplotlib scatter `s` (= marker area in "
             "points^2). Default auto-picks from cell density: smaller "
             "dots on dense sections, larger on sparse ones. Override "
             "if individual cells still overlap (try 1, 2, or 4) or "
             "are too small (try 8, 12).",
    )
    p.add_argument(
        "--query-donors", type=str, default="Lung5:3,Lung13",
        help="Comma-separated query specifiers. Each entry is either "
             "'<donor>' (whole donor is query, every replicate) OR "
             "'<donor>:<replicate>' (only that replicate of that donor "
             "is query; other replicates of the donor are reference). "
             "Multiple replicates of one donor: list it twice, e.g. "
             "'Lung5:1,Lung5:3'. Replicate matching strips common "
             "'rep'/'replicate'/'r' prefixes and is case-insensitive. "
             "Donors / replicates not listed default to '(reference)'. "
             "Pass an empty string to skip the role suffix entirely. "
             "Default: 'Lung5:3,Lung13' (Lung5 replicate 3 + Lung13 "
             "are query; everything else is reference).",
    )
    p.add_argument(
        "--dpi", type=int, default=300,
        help="Resolution of the rasterised tissue scatter (in PNG, "
             "the whole figure; in SVG, only the embedded raster "
             "layer — text/legend/axes stay vector at any dpi). "
             "Default 300.",
    )
    args = p.parse_args()

    silver_dir = Path(args.silver_dir)
    if not silver_dir.is_dir():
        raise SystemExit(f"silver_dir does not exist: {silver_dir}")
    silver_files = sorted(silver_dir.glob("*.h5ad"))
    if not silver_files:
        raise SystemExit(f"No .h5ad files under {silver_dir}.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # First pass: load each AnnData + extract donor/replicate metadata.
    rendered: List[Dict] = []
    for f in silver_files:
        a = ad.read_h5ad(f)
        donor = _parse_donor(a, f.name, args.donor_from)
        rep   = _parse_replicate(a, f.name, args.replicate_from)
        if donor is None:
            print(f"  WARNING: could not parse donor for {f.name}; "
                  f"using filename stem as donor label.")
            donor = f.stem
        rendered.append({
            "filename": f.name, "stem": f.stem,
            "donor": donor, "replicate": rep,
            "adata": a,
        })

    # Parse --query-donors into a per-donor map ({donor_lc -> set|None}).
    # See `_parse_query_spec` for the syntax (whole-donor vs per-replicate).
    query_spec = _parse_query_spec(args.query_donors)

    # Resolve titles using donor-grouping (auto "Donor X" vs "Donor X - Replicate Y"),
    # plus the (query) / (reference) role suffix when --query-donors is set.
    titles = _resolve_section_titles(rendered, query_spec=query_spec)
    for r, t in zip(rendered, titles):
        r["title"] = t

    print(f"Resolved {len(rendered)} sections under {silver_dir}:")
    for r in rendered:
        print(f"  {r['filename']:60s}  donor={r['donor']!r:>15s}  "
              f"replicate={str(r['replicate'])!r:>10s}  -> {r['title']!r}")

    # Per-section files.
    if not args.no_per_section:
        for r in rendered:
            out_path = out_dir / r["stem"]
            _render_section(
                adata=r["adata"],
                cell_type_key=args.cell_type_key,
                niche_key=args.niche_key,
                title=r["title"],
                out_path=out_path,
                dpi=args.dpi,
                spot_size=args.spot_size,
            )
        print(f"Wrote {len(rendered)} per-section plot(s) to {out_dir}")

    # Combined figure.
    if args.combined_out:
        out_path = out_dir / args.combined_out
        _render_combined(
            rendered=rendered,
            cell_type_key=args.cell_type_key,
            niche_key=args.niche_key,
            out_path=out_path,
            dpi=args.dpi,
            spot_size=args.spot_size,
        )
        print(f"Wrote combined figure to {out_path}.{{png,svg}}")


if __name__ == "__main__":
    main()
