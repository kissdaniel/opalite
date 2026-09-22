import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from opalite import singlecell as opasc


def qc_plots(
        adata,
        min_umi_counts=None,
        max_umi_counts=None,
        min_genes=None,
        max_genes=None,
        max_mt_percent=None
):
    """
    Generates standard quality control (QC) violin and scatter plots for single-cell data.

    Visualizes total UMI counts, number of detected genes, and mitochondrial gene
    percentage per cell. Optionally draws threshold lines to assist in setting filters.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object containing the single-cell data.
    min_umi_counts : int, optional
        Threshold line for the minimum number of UMI counts.
    max_umi_counts : int, optional
        Threshold line for the maximum number of UMI counts.
    min_genes : int, optional
        Threshold line for the minimum number of genes.
    max_genes : int, optional
        Threshold line for the maximum number of genes.
    max_mt_percent : float, optional
        Threshold line for the maximum percentage of mitochondrial counts.

    Examples
    --------
    >>> qc_plots(
    ...     adata=adata,
    ...     min_genes=200,
    ...     max_genes=7500,
    ...     max_mt_percent=10.0
    ... )
    """
    fig = plt.figure(figsize=(10, 8))

    ax1 = fig.add_subplot(2, 2, 1)
    y = adata.obs["total_counts"]
    x = np.random.uniform(0.9, 1.1, size=len(y))
    s = 50/len(y)
    ax1.violinplot(y, showextrema=False)
    ax1.scatter(x, y, s=s, c="black", alpha=0.8)
    if min_umi_counts:
        ax1.axhline(min_umi_counts, color="red", linestyle="--")
    if max_umi_counts:
        ax1.axhline(max_umi_counts, color="red", linestyle="--")
    ax1.set_ylabel("UMIs per barcode (log)")
    ax1.set_yscale("log", base=10)
    ax1.set_xticks([])

    ax2 = fig.add_subplot(2, 2, 2)
    y = adata.obs["n_genes_by_counts"]
    x = np.random.uniform(0.9, 1.1, size=len(y))
    s = 50/len(y)
    ax2.violinplot(y, showextrema=False)
    ax2.scatter(x, y, s=s, c="black", alpha=0.8)
    if min_genes:
        ax2.axhline(min_genes, color="red", linestyle="--")
    if max_genes:
        ax2.axhline(max_genes, color="red", linestyle="--")
    ax2.set_ylabel("Genes per barcode (linear)")
    ax2.set_xticks([])

    ax3 = fig.add_subplot(2, 2, 3)
    y = adata.obs["pct_counts_mt"]
    x = np.random.uniform(0.9, 1.1, size=len(y))
    s = 50/len(y)
    ax3.violinplot(y, showextrema=False)
    ax3.scatter(x, y, s=s, c="black", alpha=0.8)
    if max_mt_percent:
        ax3.axhline(max_mt_percent, color="red", linestyle="--")
    ax3.set_ylabel("% Mitochondrial UMIs per barcode (linear)")
    ax3.set_xticks([])

    ax4 = fig.add_subplot(2, 2, 4)
    x = adata.obs["log1p_total_counts"]
    y = adata.obs["log1p_n_genes_by_counts"]
    ax4.scatter(x, y, s=5, alpha=0.25)
    ax4.set_ylabel("Log of num. genes per cell")
    ax4.set_xlabel("Log library size")
    corr_coef = np.corrcoef(x, y)[0, 1]
    ax4.text(x=0.1, y=0.9, s="Correlation = " + str(round(corr_coef, 3)), fontsize=12, transform=ax4.transAxes)

    plt.tight_layout()
    plt.show()


def barcode_rank_plot(adata):
    """
    Generates a barcode rank plot (knee plot) for quality control.

    Plots the total UMI count against the barcode rank on a log-log scale.
    This helps in distinguishing valid cells from empty droplets.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object containing the single-cell data.

    Examples
    --------
    >>> barcode_rank_plot(adata)
    """
    counts = adata.obs['total_counts'].sort_values(ascending=False).values
    ranks = np.arange(len(counts))
    plt.figure(figsize=(6, 5))
    plt.loglog(ranks, counts, label='BC Rank Plot', color='navy')
    plt.xlabel('Barcodes')
    plt.ylabel('Total UMI Count')
    plt.grid(True, which="both", ls="-", alpha=0.25)
    plt.show()


def mito_content_plots(adata):
    """
    Generates scatter and histogram plots to visualize mitochondrial content.

    Creates three subplots: a histogram of mitochondrial percentage, a scatter plot
    of mitochondrial percentage vs log library size, and a scatter plot of mitochondrial
    percentage vs log number of genes per cell.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object containing the single-cell data.

    Examples
    --------
    >>> mito_content_plots(adata)
    """
    fig = plt.figure(figsize=(6 * 3, 5 * 1))
    ax = fig.add_subplot(1, 3, 1)
    ax.hist(adata.obs["pct_counts_mt"], 100)
    ax.set_xlabel("% MT-content", fontsize=14)
    ax.set_ylabel("Frequency", fontsize=14)
    ax = fig.add_subplot(1, 3, 2)
    ax.scatter(adata.obs["log1p_total_counts"], adata.obs["mt_pct_content"], alpha=0.25)
    ax.set_xlabel("Log library size", fontsize=14)
    ax.set_ylabel("% MT-content", fontsize=14)
    ax = fig.add_subplot(1, 3, 3)
    ax.scatter(adata.obs["log1p_n_genes_by_counts"], adata.obs["mt_pct_content"], alpha=0.25)
    ax.set_xlabel("Log num. genes per cell", fontsize=14)
    ax.set_ylabel("% MT-content", fontsize=14)
    plt.tight_layout()
    plt.show()


def umap(
        adata,
        label="sample",
        keys=None,
        categories=None,
        filename=None,
        n_pcs=30,
        n_neighbors=15,
        min_dist=0.5,
        spread=1,
        **kwargs
):
    """
    Computes PCA, neighborhood graph, and UMAP embedding, then displays or saves the plot.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object to compute the UMAP embedding on.
    label : str, default "sample"
        Optional label identifier.
    keys : list, optional
        Optional keys parameter.
    categories : str or list of str, optional
        Column name(s) in `adata.obs` or gene names to color the UMAP plot by.
    filename : str, optional
        Filename (without extension) to save the plot as PNG. If None, displays the plot.
    n_pcs : int, default 30
        Number of principal components to compute in PCA.
    n_neighbors : int, default 15
        Size of the local neighborhood used for graph construction.
    min_dist : float, default 0.5
        Effective minimum distance between embedded points for UMAP.
    spread : float, default 1
        Effective scale of embedded points for UMAP.
    **kwargs
        Additional keyword arguments passed to `scanpy.pl.umap`.

    Examples
    --------
    >>> umap(
    ...     adata=adata,
    ...     categories=["celltype", "sample"],
    ...     filename="umap_celltypes"
    ... )
    """
    print("Generating UMAP...")
    sc.tl.pca(adata, n_comps=n_pcs, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    sc.tl.umap(adata, min_dist=min_dist, spread=spread)
    if categories:
        sc.pl.umap(adata, color=categories, show=False, **kwargs)
    else:
        sc.pl.umap(adata, show=True)
    plt.tight_layout()
    if filename:
        plt.savefig(f"{filename}.png")
        plt.close()
    else:
        plt.show()


def gene_detection_dotplot(
        adata,
        gene_list: list,
        celltype_factor: str = "celltype",
        sample_factor: str = "sample",
        gene_names: str = "gene_symbols",
        counts_layer: str = None,
        split_by_sample: bool = True,
        min_count: int = 1,
        target_sum: float = 1e4,
        mean_in_expressing_cells: bool = False,
        min_cells: int = 20,
        pct_threshold: float = 10.0,
        max_dot_size: float = 300,
        cmap: str = "Reds",
        filename: str = None
) -> pd.DataFrame:
    """
    Draws one dot plot summarizing whether given genes are detected in each cell type.

    Every cell type appears as a row and every requested gene as a column, so the
    detection pattern of the whole gene panel can be read from a single figure.
    Dot area encodes the percentage of cells in which the gene reaches `min_count`
    raw counts, and dot color encodes the mean log-normalized expression, which are
    the two quantities the question "is this gene present" actually decomposes into.
    Neither number is a test: with untreated animals only, the evidence lies in the
    agreement between samples, which is why `split_by_sample` is on by default and
    each cell type is shown once per sample rather than pooled.

    Detection is always evaluated on raw counts, because any normalization or
    imputation applied beforehand changes which entries are zero and therefore
    changes the detection rate itself. QC filtering of cells is expected to have
    happened already, since low-depth and dying cells depress detection rates.
    Cell type / sample combinations built from fewer than `min_cells` cells are
    drawn faded, as their percentages are too noisy to interpret.

    Parameters
    ----------
    adata : anndata.AnnData
        The combined single-cell AnnData object containing all samples, holding raw
        (unnormalized) counts in `.X` unless `counts_layer` is given.
    gene_list : list of str
        Genes to display, matching the values in `adata.var[gene_names]`. Genes that
        are absent are reported and skipped. Order is preserved.
    celltype_factor : str, default "celltype"
        Column name in `adata.obs` holding the cell type assignment.
    sample_factor : str, default "sample"
        Column name in `adata.obs` identifying the sample.
    gene_names : str, default "gene_symbols"
        Column name in `adata.var` containing gene names. Falls back to
        `adata.var_names` if the column is absent.
    counts_layer : str, optional
        Name of a layer holding the raw counts. If omitted, `.X` is used and is
        assumed to hold raw counts.
    split_by_sample : bool, default True
        If True, each cell type is shown once per sample, which is what makes the
        reproducibility between the two animals visible. If False, samples are
        pooled into one row per cell type.
    min_count : int, default 1
        Minimum raw count for a cell to count as expressing the gene. Raising this
        to 2 suppresses part of the ambient RNA signal at the cost of sensitivity.
    target_sum : float, default 1e4
        Target count depth per cell for the internal normalization used for the
        color scale. Library sizes are taken from the whole object, not from the
        selected genes, so the values match a normalization of the full matrix.
    mean_in_expressing_cells : bool, default False
        If False, the color is the mean over all cells of the group, which is the
        usual dot plot convention. If True, the mean is taken over the expressing
        cells only, which separates expression level from detection rate at the
        cost of comparability with published dot plots.
    min_cells : int, default 20
        Groups with fewer cells are drawn faded and marked in the returned table.
    pct_threshold : float, default 10.0
        Detection percentage at or above which a dot is outlined, marking the call
        of the gene as present. The value is arbitrary and should be reported; a
        cell type known not to express the gene is the better reference for it.
    max_dot_size : float, default 300
        Marker area assigned to 100% detection.
    cmap : str, default "Reds"
        Matplotlib colormap used for the mean expression.
    filename : str, optional
        Filename (without extension) to save the plot as PNG. If None, displays
        the plot.

    Returns
    -------
    pandas.DataFrame
        One row per group and gene, with the cell count, the number and percentage
        of expressing cells, the mean expression, the median library size of the
        group and the `present` call. The median library size is included because
        detection rates are only comparable between groups of similar depth.

    Examples
    --------
    >>> detection = gene_detection_dotplot(
    ...     adata=combined_adata,
    ...     gene_list=["Ms4a6d", "P2ry12", "Sdf2l1"],
    ...     pct_threshold=10.0,
    ...     filename="gene_detection_dotplot"
    ... )
    >>> detection[detection["present"]]
    """
    print("Summarizing gene detection...")
    if gene_names in adata.var.columns:
        available = pd.Index(adata.var[gene_names].astype(str))
    else:
        print(f"Column '{gene_names}' not found in .var, using .var_names instead.")
        available = pd.Index(adata.var_names.astype(str))

    # First occurrence wins, so that duplicated symbols do not silently turn into
    # two columns of the figure.
    positions = pd.Series(np.arange(adata.n_vars), index=available)
    positions = positions[~positions.index.duplicated(keep="first")]

    requested = list(dict.fromkeys(gene_list))
    genes = [gene for gene in requested if gene in positions.index]
    missing = [gene for gene in requested if gene not in positions.index]
    if missing:
        print(f"Not found among the gene names: {', '.join(missing)}. Check for "
              "Ensembl identifiers, version suffixes or symbol synonyms.")
    if not genes:
        raise ValueError("None of the requested genes is present in the object.")

    subset = adata[:, positions[genes].to_numpy()].copy()
    subset.var_names = pd.Index(genes)
    if counts_layer:
        if counts_layer not in subset.layers:
            raise KeyError(
                f"Layer '{counts_layer}' not found in the input object. Available "
                f"layers: {list(adata.layers.keys())}."
                )
        subset.X = subset.layers[counts_layer]

    counts = subset.X.toarray() if issparse(subset.X) else np.asarray(subset.X)
    counts = np.asarray(counts, dtype=float)

    # Library sizes must come from all genes, not from the selected panel, so the
    # color scale matches what normalizing the whole object would produce.
    if counts_layer:
        library_sizes = np.asarray(adata.layers[counts_layer].sum(axis=1)).ravel()
    elif "total_counts" in adata.obs.columns:
        library_sizes = adata.obs["total_counts"].to_numpy()
    else:
        library_sizes = np.asarray(adata.X.sum(axis=1)).ravel()
    library_sizes = np.asarray(library_sizes, dtype=float)
    safe_sizes = np.where(library_sizes > 0, library_sizes, 1.0).reshape(-1, 1)
    normalized = np.log1p(counts / safe_sizes * target_sum)

    groups = pd.DataFrame({
        "celltype": adata.obs[celltype_factor].astype(str).to_numpy()
        }, index=np.arange(adata.n_obs))
    use_samples = split_by_sample and sample_factor in adata.obs.columns
    if split_by_sample and not use_samples:
        print(f"Column '{sample_factor}' not found in .obs, pooling the samples.")
    if use_samples:
        groups["sample"] = adata.obs[sample_factor].astype(str).to_numpy()
        groups["group"] = groups["celltype"] + " | " + groups["sample"]
        group_order = (
            groups[["celltype", "sample", "group"]]
            .drop_duplicates()
            .sort_values(["celltype", "sample"])["group"]
            .tolist()
            )
    else:
        groups["sample"] = "all"
        groups["group"] = groups["celltype"]
        group_order = sorted(groups["group"].unique())

    records = []
    for group in group_order:
        cell_mask = (groups["group"] == group).to_numpy()
        n_cells = int(cell_mask.sum())
        fractions = _detected_fraction(subset, cell_mask, min_count)
        group_counts = counts[cell_mask]
        group_normalized = normalized[cell_mask]
        median_library_size = float(np.median(library_sizes[cell_mask]))
        for column, gene in enumerate(genes):
            expressing = group_counts[:, column] >= min_count
            values = group_normalized[:, column]
            if mean_in_expressing_cells:
                mean_expression = float(values[expressing].mean()) if expressing.any() else 0.0
            else:
                mean_expression = float(values.mean())
            pct_detected = float(fractions[gene] * 100)
            records.append({
                "group": group,
                "celltype": groups.loc[cell_mask, "celltype"].iloc[0],
                "sample": groups.loc[cell_mask, "sample"].iloc[0],
                "gene": gene,
                "n_cells": n_cells,
                "n_detected": int(expressing.sum()),
                "pct_detected": pct_detected,
                "mean_expression": mean_expression,
                "median_umi": median_library_size,
                "enough_cells": n_cells >= min_cells,
                "present": bool(pct_detected >= pct_threshold and n_cells >= min_cells)
                })

    summary = pd.DataFrame.from_records(records)

    small_groups = sorted(summary.loc[~summary["enough_cells"], "group"].unique())
    if small_groups:
        print(f"Fewer than {min_cells} cells, shown faded: {', '.join(small_groups)}.")
    if use_samples:
        concordance = summary.groupby(["celltype", "gene"])["present"].agg(["sum", "size"])
        n_all = int((concordance["sum"] == concordance["size"]).sum())
        n_some = int(((concordance["sum"] > 0) & (concordance["sum"] < concordance["size"])).sum())
        print(f"{n_all} cell type / gene combinations reached {pct_threshold}% in every "
              f"sample, {n_some} in some of them only.")

    _draw_detection_dotplot(
            summary=summary,
            genes=genes,
            group_order=group_order,
            min_count=min_count,
            mean_in_expressing_cells=mean_in_expressing_cells,
            pct_threshold=pct_threshold,
            max_dot_size=max_dot_size,
            cmap=cmap
            )
    if filename:
        plt.savefig(f"{filename}.png", dpi=200, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
    return summary


def _detected_fraction(adata, cell_mask, min_count):
    """
    Returns the fraction of the selected cells in which each gene reaches `min_count`.

    Delegates to `singlecell._expressed_fraction` for the default threshold of one
    count, so that the detection rates drawn by the dot plot are computed exactly
    like the `pct_control` and `pct_treatment` columns of the differential
    expression summaries. Stricter thresholds are evaluated here.
    """
    if min_count <= 1:
        return opasc.expressed_fraction(adata, cell_mask)
    matrix = adata[cell_mask].X
    n_cells = int(cell_mask.sum())
    if issparse(matrix):
        detected = np.asarray((matrix >= min_count).sum(axis=0)).ravel()
    else:
        detected = np.asarray(np.asarray(matrix) >= min_count).sum(axis=0).ravel()
    return pd.Series(detected / n_cells, index=adata.var_names)


def _draw_detection_dotplot(
        summary,
        genes: list,
        group_order: list,
        min_count: int,
        mean_in_expressing_cells: bool,
        pct_threshold: float,
        max_dot_size: float,
        cmap: str
):
    """
    Renders the detection summary as a single dot plot on a new figure.

    Groups that did not reach the cell count threshold are drawn faded, dots at or
    above the detection threshold are outlined, and groups in which the gene was
    not detected at all are drawn as crosses so that a zero stays distinguishable
    from a missing combination.
    """
    gene_positions = {gene: index for index, gene in enumerate(genes)}
    group_positions = {group: index for index, group in enumerate(group_order)}

    figure_width = max(6.0, 3.6 + 0.6 * len(genes))
    figure_height = max(3.0, 1.2 + 0.34 * len(group_order))
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))

    x = summary["gene"].map(gene_positions).to_numpy()
    y = summary["group"].map(group_positions).to_numpy()
    pct = summary["pct_detected"].to_numpy()
    expression = summary["mean_expression"].to_numpy()
    reliable = summary["enough_cells"].to_numpy()
    vmax = float(expression.max()) if expression.max() > 0 else 1.0

    zero = pct <= 0
    if zero.any():
        ax.scatter(x[zero], y[zero], s=14, c="0.75", marker="x", linewidths=0.8)

    dots = None
    for mask, alpha in ((reliable & ~zero, 1.0), (~reliable & ~zero, 0.3)):
        if not mask.any():
            continue
        handle = ax.scatter(
                x[mask], y[mask],
                s=pct[mask] / 100 * max_dot_size,
                c=expression[mask],
                cmap=cmap,
                vmin=0,
                vmax=vmax,
                alpha=alpha,
                linewidths=0.8,
                edgecolors=np.where(pct[mask] >= pct_threshold, "black", "none")
                )
        if alpha == 1.0:
            dots = handle
    if dots is None:
        dots = ax.scatter([], [], c=[], cmap=cmap, vmin=0, vmax=vmax)

    ax.set_xticks(range(len(genes)))
    ax.set_xticklabels(genes, rotation=45, ha="right", style="italic")
    ax.set_yticks(range(len(group_order)))
    ax.set_yticklabels(group_order)
    ax.set_xlim(-0.6, len(genes) - 0.4)
    ax.set_ylim(-0.6, len(group_order) - 0.4)
    ax.invert_yaxis()
    ax.set_axisbelow(True)
    ax.grid(True, linestyle=":", alpha=0.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    colorbar = fig.colorbar(dots, ax=ax, fraction=0.04, pad=0.04, shrink=0.75)
    scope = "expressing cells" if mean_in_expressing_cells else "all cells"
    colorbar.set_label(
            f"Mean log-normalized expression ({scope})",
            fontsize=9,
            labelpad=8
            )
    colorbar.ax.tick_params(labelsize=8)

    size_handles = [
        ax.scatter([], [], s=level / 100 * max_dot_size, c="0.45",
                   edgecolors="none", label=f"{level}%")
        for level in (25, 50, 75, 100)
        ]
    legend = ax.legend(
            handles=size_handles,
            title=f"% cells with\n>= {min_count} count(s)",
            loc="upper left",
            bbox_to_anchor=(1.32, 1.0),
            frameon=False,
            fontsize=9,
            labelspacing=1.3,
            borderpad=0.8
            )
    legend.get_title().set_fontsize(9)
    ax.set_title(f"Outlined dots reach {pct_threshold}% detection", fontsize=9, loc="left")
    plt.tight_layout()
