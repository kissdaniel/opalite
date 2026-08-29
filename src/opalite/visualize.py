import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc


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
