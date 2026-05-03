import anndata as ad
import decoupler as dc
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats


def read_custom_mappings(excel_file, sheet_name):
    xls_data = pd.read_excel(excel_file, sheet_name=sheet_name)
    mappings = {
        col: xls_data[col].dropna().tolist()
        for col in xls_data.columns
    }
    return mappings


def export_from_anndata_to_csv(
        adata,
        attributes: list[str],
        filename: str
):
    df_to_save = adata.obs[attributes].copy()
    df_to_save.reset_index(inplace=True)
    df_to_save.rename(columns={df_to_save.columns[0]: 'cell_id'}, inplace=True)
    df_to_save.to_csv(filename, index=False)
    print(f"Export completed: {filename}")


def create_anndata_object(
        data_file: str,
        mapmycells_csv_file: str,
        mappings_xlsx_file: str,
        sample_name: str,
        xlsx_sheet_name: str = "Cell types",
        drop_na_types: bool = True,
        calculate_qc_metrics: bool = True,
        mt_gene_prefix: str = "mt-"
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
    if calculate_qc_metrics:
        adata.var["mt"] = adata.var_names.str.startswith(mt_gene_prefix)
        sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True)
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
    print("Filtering cells...")
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
    print(f"{final_count} / {initial_count} cells passed filtering.")
    return subset


def log_transform_and_scale(adata, inplace=False):
    print("Log-transforming and scaling...")
    if not inplace:
        adata = adata.copy()
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata_hvf = adata[:, adata.var.highly_variable].copy()
    # sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])
    sc.pp.scale(adata_hvf, max_value=10)
    return adata_hvf


def generate_umap(adata, label="sample", keys=None, n_pcs=30, categories=None, **kwargs):
    print("Generating UMAP...")
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_pcs=n_pcs)
    sc.tl.umap(adata)
    if categories:
        sc.pl.umap(adata, color=categories, show=False, **kwargs)
    else:
        sc.pl.umap(adata, show=True)
    plt.tight_layout()
    plt.show()


def concatenate_anndata_objects(adata_list, label="sample", names=None):
    if not names:
        names = [i for i in range(len(adata_list))]
    adata = sc.concat(
        adata_list,
        label=label,
        keys=names,
        index_unique="_"
        )
    return adata


def differential_expression(
        adata,
        tested_name,
        reference_name,
        filter_celltype=None,
        design="sample",
        n_top_genes=10,
        n_replicates=3,
        min_cells=10,
        min_counts=1000,
        out_filename=None,
        n_cpus=8
):
    if filter_celltype:
        adata = adata[adata.obs["celltype"] == filter_celltype].copy()
    else:
        adata = adata.copy()
    np.random.seed(0)
    adata.obs['pseudo_rep'] = np.random.randint(0, n_replicates, size=adata.n_obs)
    adata.obs['pseudo_sample'] = adata.obs['sample'].astype(str) + "_" + adata.obs['pseudo_rep'].astype(str)
    pbdata = dc.pp.pseudobulk(adata, sample_col="pseudo_sample", groups_col=None)
    dc.pp.filter_samples(pbdata, min_cells=min_cells, min_counts=min_counts)

    dds = DeseqDataSet(
        adata=pbdata,
        design=design,
        refit_cooks=True,
        n_cpus=n_cpus
    )
    dds.deseq2()

    stat_res = DeseqStats(dds, contrast=[design, tested_name, reference_name], quiet=True)
    stat_res.summary()
    if out_filename:
        stat_res.results_df.to_csv(out_filename)
    return stat_res.results_df


def enrichment_analysis(
        de_data,
        omnipath_organism:str = "mouse",
        method:str = "ulm",
        p_threshold: float = None,
        out_filename: str = None
):
    de_data.dropna(inplace=True)
    data = de_data[["stat"]].T.rename(index={"stat": f"treatment.vs.control"})
    hallmark = dc.op.hallmark(organism=omnipath_organism)
    hm_acts, hm_padj = dc.mt.ulm(data=data, net=hallmark)
    if p_threshold:
        msk = (hm_padj.T < p_threshold).iloc[:, 0]
        hm_acts = hm_acts.loc[:, msk]
    if out_filename:
        df1 = hm_acts.T.copy()
        df2 = hm_padj.T.copy()
        df1.columns = ['score']
        df2.columns = ['padj']
        df_combined = pd.concat([df1, df2], axis=1)
        df_combined.to_csv(out_filename)
    return hm_acts
