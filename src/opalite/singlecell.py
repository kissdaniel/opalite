import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def read_custom_mappings(excel_file, sheet_name):
    xls_data = pd.read_excel(excel_file, sheet_name=sheet_name)
    mappings = {
        col: xls_data[col].dropna().tolist()
        for col in xls_data.columns
    }
    return mappings


def create_anndata_object(
        data_file: str,
        mapmycells_csv_file: str,
        mappings_xlsx_file: str,
        sample_name: str,
        xlsx_sheet_name: str = "Cell types",
        drop_na_types: bool = True
):
    extension = data_file.split('.')[-1]
    if extension == 'h5ad':
        adata = sc.read_h5ad(data_file)
    elif extension == 'h5':
        adata = sc.read_10x_h5(data_file, gex_only=False)
    else:
        raise ValueError(f"Unsupported file extension: {extension}")
    results = pd.read_csv(mapmycells_csv_file, comment="#")
    celltype_mapping = read_custom_mappings(mappings_xlsx_file, xlsx_sheet_name)
    inverted_mapping = {
        subclass: celltype
        for celltype, subclasses in celltype_mapping.items()
        for subclass in subclasses
    }
    adata.obs["subclass_name"] = results["subclass_name"].values
    adata.obs["celltype"] = adata.obs["subclass_name"].map(inverted_mapping)
    adata.obs["sample"] = sample_name
    if drop_na_types:
        adata = adata[~adata.obs["celltype"].isna(), :].copy()
    return adata


def generate_qc_plots(adata):
    fig = plt.figure(figsize=(6 * 3, 5 * 1))
    ax = fig.add_subplot(1, 3, 1)
    x = adata.obs["total_counts"]
    y = np.random.uniform(0, 1, size=len(x))
    ax.scatter(x, y, alpha=0.1)
    ax_t = ax.twinx()
    ax_t.hist(adata.obs["total_counts"], bins=100, density=True, histtype="step", color="black")
    ax_t.set_yticks([])
    ax.set_xscale("log", base=2)
    ax.set_xlabel("UMIs per barcode (log)", fontsize=14)
    ax = fig.add_subplot(1, 3, 2)
    x = adata.obs["n_genes_by_counts"]
    y = np.random.uniform(0, 1, size=len(x))
    ax.scatter(x, y, alpha=0.1)
    ax_t = ax.twinx()
    ax_t.hist(adata.obs["n_genes_by_counts"], bins=100, density=True, histtype="step", color="black")
    ax_t.set_yticks([])
    ax.set_xlabel("Genes per barcode (linear)", fontsize=14)
    ax = fig.add_subplot(1, 3, 3)
    x = adata.obs["log1p_total_counts"]
    y = adata.obs["log1p_n_genes_by_counts"]
    ax.scatter(x, y, s=5, alpha=0.25)
    ax.set_ylabel("Log of num. genes per cell", fontsize=14)
    ax.set_xlabel("Log library size", fontsize=14)
    corr_coef = np.corrcoef(x, y)[0, 1]
    ax.set_title("Correlation = " + str(round(corr_coef, 3)), fontsize=14)
    plt.tight_layout()
    plt.show()


def generate_barcode_rank_plot(adata):
    counts = adata.obs['total_counts'].sort_values(ascending=False).values
    ranks = np.arange(len(counts))
    plt.figure(figsize=(6, 5))
    plt.loglog(ranks, counts, label='BC Rank Plot', color='navy')
    plt.xlabel('Barcodes')
    plt.ylabel('Total UMI Count')
    plt.grid(True, which="both", ls="-", alpha=0.25)
    plt.show()


def generate_mt_plots(adata):
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


def filter_cells(
        adata,
        min_umi_counts=None,
        max_umi_counts=None,
        min_genes=None,
        max_genes=None,
        max_mt_percent=None,
        remove_doublets=False,
        doublet_threshold=None
):
    subset = adata.copy()
    initial_count = subset.obs.shape[0]
    if min_umi_counts:
        sc.pp.filter_cells(subset, min_counts=min_umi_counts)
    if max_umi_counts:
        sc.pp.filter_cells(subset, max_counts=max_umi_counts)
    if min_genes:
        sc.pp.filter_cells(subset, min_genes=min_genes)
    if max_genes:
        sc.pp.filter_cells(subset, max_genes=max_genes)
    if max_mt_percent:
        subset = subset[subset.obs["pct_counts_mt"] <= max_mt_percent, :].copy()
    if remove_doublets:
        sc.pp.scrublet(subset, threshold=doublet_threshold)
        subset = subset[~subset.obs['predicted_doublet'], :].copy()
    final_count = subset.obs.shape[0]
    print(f"{final_count} / {initial_count} barcodes passed filtering.")
    return subset
