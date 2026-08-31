import decoupler as dc
import scanpy as sc
import pandas as pd
import numpy as np
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats


def create_anndata_object(
        data_file: str,
        mapmycells_csv_file: str,
        mappings_xlsx_file: str,
        sample_name: str,
        xlsx_sheet_name: str = "Cell types",
        mapmycells_bstrap_conf_thresholds: dict = None,
        drop_na_types: bool = True,
        calculate_qc_metrics: bool = True,
        mt_gene_prefix: str = "mt-"
):
    """
    Creates an AnnData object from a data file, integrating MapMyCells cell type annotations.

    This function reads a single-cell data file (.h5ad or .h5), applies cell type annotations
    from a MapMyCells CSV output, and filters the annotations based on bootstrapping
    probability thresholds. It also supports calculating basic QC metrics.

    Parameters
    ----------
    data_file : str
        Path to the single-cell data file (.h5ad or 10x .h5 format).
    mapmycells_csv_file : str
        Path to the MapMyCells CSV output containing taxonomy assignments.
    mappings_xlsx_file : str
        Path to the Excel file containing custom cell type mapping definitions.
    sample_name : str
        Identifier for the sample, added to `adata.obs['sample']`.
    xlsx_sheet_name : str, default "Cell types"
        The sheet name in `mappings_xlsx_file` to read the mappings from.
    mapmycells_bstrap_conf_thresholds : dict, optional
        Custom bootstrapping probability thresholds for taxonomy levels
        (e.g., {"class_bootstrapping_probability": 0.85}). Defaults to 0.8.
    drop_na_types : bool, default True
        If True, filters out cells that do not have an assigned cell type.
    calculate_qc_metrics : bool, default True
        If True, calculates standard scanpy QC metrics (e.g., mitochondrial gene content).
    mt_gene_prefix : str, default "mt-"
        The prefix used to identify mitochondrial genes in `adata.var['gene_symbols']`.

    Returns
    -------
    anndata.AnnData
        The annotated AnnData object with computed QC metrics and cell type assignments.

    Examples
    --------
    >>> adata = create_anndata_object(
    ...     data_file="sample1.h5ad",
    ...     mapmycells_csv_file="sample1_MMC.csv",
    ...     mappings_xlsx_file="celltype_mappings.xlsx",
    ...     sample_name="sample_1"
    ... )
    """
    extension = data_file.split('.')[-1]
    if extension == 'h5ad':
        adata = sc.read_h5ad(data_file)
    elif extension == 'h5':
        adata = sc.read_10x_h5(data_file, gex_only=False)
    else:
        raise ValueError(f"Unsupported file extension: {extension}")
    results = pd.read_csv(mapmycells_csv_file, comment="#")

    bstrap_thresholds = {
            "class_bootstrapping_probability": 0.8,
            "subclass_bootstrapping_probability": 0.8,
            "supertype_bootstrapping_probability": 0.8,
            "cluster_bootstrapping_probability": 0.8
            }
    if mapmycells_bstrap_conf_thresholds:
        for key in mapmycells_bstrap_conf_thresholds:
            bstrap_thresholds[key] = mapmycells_bstrap_conf_thresholds[key]

    results.loc[results["class_bootstrapping_probability"] < bstrap_thresholds["class_bootstrapping_probability"], ["class_name", "subclass_name", "supertype_name", "cluster_name"]] = np.nan
    results.loc[results["subclass_bootstrapping_probability"] < bstrap_thresholds["subclass_bootstrapping_probability"], ["subclass_name", "supertype_name", "cluster_name"]] = np.nan
    results.loc[results["supertype_bootstrapping_probability"] < bstrap_thresholds["supertype_bootstrapping_probability"], ["supertype_name", "cluster_name"]] = np.nan
    results.loc[results["cluster_bootstrapping_probability"] < bstrap_thresholds["cluster_bootstrapping_probability"], "cluster_name"] = np.nan

    celltype_mapping = _read_custom_mappings(mappings_xlsx_file, xlsx_sheet_name)
    inverted_mapping = {
        subclass: celltype
        for celltype, subclasses in celltype_mapping.items()
        for subclass in subclasses
    }

    def get_celltype(row):
        if row["class_name"] in inverted_mapping:
            return inverted_mapping[row["class_name"]]
        if row["subclass_name"] in inverted_mapping:
            return inverted_mapping[row["subclass_name"]]
        if row["supertype_name"] in inverted_mapping:
            return inverted_mapping[row["supertype_name"]]
        if row["cluster_name"] in inverted_mapping:
            return inverted_mapping[row["cluster_name"]]
        return None

    results["celltype"] = results.apply(get_celltype, axis=1)
    adata.obs["celltype"] = adata.obs.index.map(results.set_index("cell_id")["celltype"])
    adata.obs["sample"] = sample_name
    if drop_na_types:
        adata = adata[~adata.obs["celltype"].isna(), :].copy()
    if calculate_qc_metrics:
        adata.var["mt"] = adata.var["gene_symbols"].str.startswith(mt_gene_prefix)
        sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True)
    return adata


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


def calculate_pseudobulk_deg(
        combined_adata,
        celltype_names: list,
        control_name: str,
        treatment_names: list,
        design_factor: str = "condition",
        gene_names: str = "gene_symbols",
        min_cells: int = 10,
        min_counts: int = 1000,
        write_output_file=False,
        n_cpus: int = None,
        quiet: bool = True
) -> dict:
    """
    Performs pseudobulk differential expression analysis using PyDESeq2.

    Filters for the specified cell type (optional), aggregates single-cell data
    into pseudobulk samples per sample, and runs DESeq2 comparing treatment conditions
    against a control group.

    Parameters
    ----------
    combined_adata : anndata.AnnData
        The input single-cell AnnData object containing all combined samples.
    control_name : str
        The reference/control condition name for comparison.
    treatment_names : list of str
        List of treatment condition names to compare against the control.
    design_factor : str, default "condition"
        Column name in `adata.obs` representing the experimental condition/factor.
    gene_names : str, default "gene_symbols"
        Column name in `adata.var` containing gene names to set as variable names.
    min_cells : int, default 10
        Minimum number of cells required per pseudobulk sample.
    min_counts : int, default 1000
        Minimum total counts required per pseudobulk sample.
    write_output_file : bool, default False
        If True, saves differential expression results as CSV files.
    n_cpus : int, optional
        Number of CPU threads to use for PyDESeq2 computations.
    quiet : bool, default True
        Suppress status updates during run.

    Returns
    -------
    dict of {str: pandas.DataFrame}
        A dictionary mapping each treatment name to its corresponding DESeq2
        results DataFrame.

    Examples
    --------
    >>> de_results = calculate_pseudobulk_deg(
    ...     adata=adata,
    ...     control_name="Control",
    ...     treatment_names=["LPS", "PolyIC"],
    ...     filter_celltype="Microglia",
    ...     design_factors="condition"
    ... )
    """
    combined_adata.var_names = combined_adata.var[gene_names].astype(str)
    combined_adata.var_names_make_unique()

    pbdata = dc.pp.pseudobulk(combined_adata, sample_col="sample", groups_col="celltype")
    dc.pp.filter_samples(pbdata, min_cells=min_cells, min_counts=min_counts)

    results_dict = {}

    for celltype in celltype_names:
        sub_pb = pbdata[pbdata.obs["celltype"] == celltype].copy()
        sub_pb.obs[design_factor] = pd.Categorical(
                sub_pb.obs[design_factor],
                categories=[control_name] + treatment_names
                )
        dds = DeseqDataSet(
                adata=sub_pb,
                design=f"~ {design_factor}",
                n_cpus=n_cpus,
                quiet=quiet
                )
        dds.deseq2()

        results_dict[celltype] = {}

        for treatment in treatment_names:
            res = DeseqStats(
                dds,
                contrast=[design_factor, treatment, control_name],
                quiet=quiet
                )
            res.summary()

            res.lfc_shrink(coeff=f"condition[T.{treatment}]")
            results_dict[celltype][treatment] = res.results_df
            if write_output_file:
                res.results_df.to_csv(f"{celltype}_{treatment}_all_DEG.csv")

    return results_dict


def calculate_geneset_activities(
        de_data,
        gene_set_name: str,
        control_name: str = None,
        treatment_name: str = None,
        omnipath_organism: str = "mouse",
        method: str = "ulm",
        p_threshold: float = None,
        out_filename: str = None
):
    """
    Performs gene set enrichment analysis using the decoupler package and OmniPath networks.

    Calculates pathway or transcription factor activities from differential expression
    statistics using methods like ULM, GSEA, or ORA. The results are optionally filtered
    by a p-value threshold and can be saved to a CSV file.

    Parameters
    ----------
    de_data : pandas.DataFrame
        Differential expression results containing a 'stat' column (e.g., from DESeq2).
    gene_set_name : str
        The name of the OmniPath network to use ('collectri', 'hallmark', or 'progeny').
    control_name : str, optional
        Name of the control condition, used for naming the contrast.
    treatment_name : str, optional
        Name of the treatment condition, used for naming the contrast.
    omnipath_organism : str, default "mouse"
        Organism name for the OmniPath query (e.g., "mouse", "human").
    method : str, default "ulm"
        The decoupler method to use for enrichment ("ulm", "gsea", or "ora").
    p_threshold : float, optional
        If provided, filters out pathways/TFs with an adjusted p-value above this threshold.

    Returns
    -------
    pandas.DataFrame
        A dataframe containing the enrichment 'score' and 'padj' (adjusted p-value)
        for each pathway or transcription factor.

    Examples
    --------
    >>> enrichment_df = calculate_geneset_activities(
    ...     de_data=deseq_results,
    ...     gene_set_name="hallmark",
    ...     treatment_name="LPS",
    ...     control_name="Control",
    ...     method="ulm"
    ... )
    """
    de_data.dropna(inplace=True)
    sample_name = "treatment.vs.control"
    if treatment_name and control_name:
        sample_name = f"{treatment_name}.vs.{control_name}"
    data = de_data[["stat"]].T.rename(index={"stat": sample_name})

    if gene_set_name == "collectri":
        net = dc.op.collectri(organism=omnipath_organism)
    if gene_set_name == "hallmark":
        net = dc.op.hallmark(organism=omnipath_organism)
    if gene_set_name == "progeny":
        net = dc.op.progeny(organism=omnipath_organism)
    # if gene_set_name == "kegg":
    #     kegg_df = dc.op.resource(name="KEGG")
    #     kegg_mouse = kegg_df[kegg_df["ncbi_tax_id"] == 10090].copy()
    #     net = kegg_mouse.rename(columns={"geneset": "source", "genesymbol": "target"})
    #     net = net.dropna(subset=["source", "target"])
    #     net["source"] = net["source"].astype(str)
    #     net["target"] = net["target"].astype(str)
    #     net = net.drop_duplicates(subset=["source", "target"])

    if method == "gsea":
        hm_acts, hm_padj = dc.mt.gsea(data=data, net=net)
    if method == "ora":
        hm_acts, hm_padj = dc.mt.ora(data=data, net=net)
    if method == "ulm":
        hm_acts, hm_padj = dc.mt.ulm(data=data, net=net)
    if p_threshold:
        msk = (hm_padj.T < p_threshold).iloc[:, 0]
        hm_acts = hm_acts.loc[:, msk]
        hm_padj = hm_padj.loc[:, msk]
    df1 = hm_acts.T.copy()
    df2 = hm_padj.T.copy()
    df1.columns = ['score']
    df2.columns = ['padj']
    df_combined = pd.concat([df1, df2], axis=1)
    return df_combined


def filter_significant_genes(
        de_data,
        lfc_threshold: float = 0.5,
        p_threshold: float = 0.05,
        top=None
) -> pd.DataFrame:
    """
    Filters differential expression results and returns the significant genes
    in a Pandas DataFrame object.

    Filters genes based on a log2-fold change threshold and an adjusted p-value threshold.
    The resulting significant genes are sorted by their absolute test statistic and
    exported to disk.

    Parameters
    ----------
    de_data : pandas.DataFrame
        The differential expression results dataframe (e.g., from DESeq2).
    lfc_threshold : float, default 0.5
        The minimum absolute log2FoldChange required for significance.
    p_threshold : float, default 0.05
        The maximum adjusted p-value (padj) allowed for significance.
    top : int, optional
        If provided, limits the export to the top `top` most significant genes.
        If None, all genes passing the thresholds are exported.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the filtered significant genes (meeting the defined thresholds)
        alongside their corresponding statistics.

    Examples
    --------
    >>> filter_significant_genes(
    ...     de_data=deseq_results,
    ...     lfc_threshold=1.0,
    ...     p_threshold=0.01,
    ...     top=50
    ... )
    """
    max_stat = np.inf
    max_sign = np.inf
    thr_sign = -np.log10(p_threshold)
    df = de_data.copy()
    df["abs_stat"] = np.abs(df["stat"])
    non_zero_min = df["padj"][df["padj"] != 0].min()
    df["pval"] = -np.log10(df["padj"].clip(lower=non_zero_min, upper=1))
    msk_stat = np.abs(df["log2FoldChange"]) < np.abs(max_stat)
    msk_sign = df["pval"] < np.abs(max_sign)
    df = df.loc[msk_stat & msk_sign]
    thr_msk = (np.abs(df["log2FoldChange"]) >= lfc_threshold) & (df["pval"] >= thr_sign)
    signs = df[thr_msk].sort_values("abs_stat", ascending=False)
    if top:
        signs = signs.iloc[:top]
    return signs


def export_from_anndata_to_csv(
        adata,
        attributes: list[str],
        filename: str
):
    """
    Exports selected cell metadata from an AnnData object to a CSV file.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object containing the data.
    attributes : list[str]
        List of column names from `adata.obs` to export.
    filename : str
        The path and name of the output CSV file.

    Examples
    --------
    >>> export_from_anndata_to_csv(
    ...     adata=my_adata,
    ...     attributes=["celltype", "sample"],
    ...     filename="cell_metadata.csv"
    ... )
    """
    df_to_save = adata.obs[attributes].copy()
    df_to_save.reset_index(inplace=True)
    df_to_save.rename(columns={df_to_save.columns[0]: 'cell_id'}, inplace=True)
    df_to_save.to_csv(filename, index=False)
    print(f"Export completed: {filename}")


def _read_custom_mappings(excel_file, sheet_name):
    xls_data = pd.read_excel(excel_file, sheet_name=sheet_name)
    mappings = {
        col: xls_data[col].dropna().tolist()
        for col in xls_data.columns
    }
    return mappings
