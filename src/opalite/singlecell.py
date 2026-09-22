import os
import decoupler as dc
import scanpy as sc
import pandas as pd
import numpy as np
from scipy.sparse import issparse
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
        min_cells_per_gene=None,
        remove_doublets=False,
        doublet_threshold=None
):
    """
    Applies per-cell quality control filters to a single sample.

    All thresholds are optional; a filter is only applied when its argument is set.
    Note that `min_cells_per_gene` removes genes that are detected in almost no
    cell of this sample. It is a convenience filter only and does NOT replace the
    expression-based gene filtering performed on the pseudobulk matrix inside
    `calculate_pseudobulk_deg`, which operates on a different aggregation level.

    Parameters
    ----------
    adata : anndata.AnnData
        The input AnnData object for a single sample.
    min_umi_counts, max_umi_counts : int, optional
        Minimum / maximum total UMI counts per cell.
    min_genes, max_genes : int, optional
        Minimum / maximum number of detected genes per cell.
    max_mt_percent : float, optional
        Maximum percentage of mitochondrial counts per cell. Requires
        `pct_counts_mt` in `adata.obs`.
    min_cells_per_gene : int, optional
        If set, drops genes detected in fewer than this many cells.
    remove_doublets : bool, default False
        If True, runs scrublet and removes predicted doublets.
    doublet_threshold : float, optional
        Custom doublet score threshold passed to scrublet.

    Returns
    -------
    anndata.AnnData
        A new AnnData object containing the cells that passed filtering.

    Examples
    --------
    >>> adata_qc = filter_cells(
    ...     adata=adata,
    ...     min_genes=200,
    ...     max_mt_percent=10,
    ...     remove_doublets=True
    ... )
    """
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
    if min_cells_per_gene:
        initial_genes = subset.var.shape[0]
        sc.pp.filter_genes(subset, min_cells=min_cells_per_gene)
        print(f"{subset.var.shape[0]} / {initial_genes} genes passed filtering.")
    if remove_doublets:
        sc.pp.scrublet(subset, threshold=doublet_threshold)
        subset = subset[~subset.obs['predicted_doublet'], :].copy()
    final_count = subset.obs.shape[0]
    print(f"{final_count} / {initial_count} cells passed filtering.")
    return subset


def summarize_celltype_composition(
        adata,
        sample_factor: str = "sample",
        celltype_factor: str = "celltype",
        design_factor: str = None,
        min_cells: int = 10,
        out_filename: str = None
) -> pd.DataFrame:
    """
    Reports how many cells each sample contributes to each cell type.

    Run this before `calculate_pseudobulk_deg` to see which sample / cell type
    combinations are too small to become usable pseudobulk samples. If
    `design_factor` is given, the function also reports, per cell type, which
    experimental conditions would lose all of their samples and therefore drop
    out of that cell type's design.

    The prediction only accounts for the cell count threshold, not for the total
    count threshold also applied by `calculate_pseudobulk_deg`, so the reported
    losses are a lower bound.

    Parameters
    ----------
    adata : anndata.AnnData
        The combined single-cell AnnData object containing all samples.
    sample_factor : str, default "sample"
        Column name in `adata.obs` identifying the sample.
    celltype_factor : str, default "celltype"
        Column name in `adata.obs` holding the cell type assignment.
    design_factor : str, optional
        Column name in `adata.obs` representing the experimental condition. If
        given, conditions predicted to drop out are reported per cell type.
    min_cells : int, default 10
        Cell count threshold used for the prediction. Should match the
        `min_cells` value passed to `calculate_pseudobulk_deg`.
    out_filename : str, optional
        If given, the cell count table is written to this CSV file.

    Returns
    -------
    pandas.DataFrame
        A sample x cell type table of cell counts.

    Examples
    --------
    >>> counts = summarize_celltype_composition(
    ...     adata=combined_adata,
    ...     design_factor="condition",
    ...     min_cells=10
    ... )
    """
    counts = pd.crosstab(adata.obs[sample_factor], adata.obs[celltype_factor])
    n_below = int((counts < min_cells).sum().sum())
    print(f"{counts.shape[0]} samples x {counts.shape[1]} cell types, "
          f"{n_below} combinations below {min_cells} cells.")

    if design_factor:
        conditions = (
            adata.obs[[sample_factor, design_factor]]
            .astype(str)
            .drop_duplicates()
            .set_index(sample_factor)[design_factor]
            )
        all_conditions = set(conditions.unique())
        passing = counts >= min_cells
        for celltype in counts.columns:
            kept_samples = passing.index[passing[celltype]]
            kept = set(conditions.reindex(kept_samples).dropna().unique())
            lost = sorted(all_conditions - kept)
            if lost:
                print(f"{celltype}: no sample left for {', '.join(lost)}.")

    if out_filename:
        counts.to_csv(out_filename)
        print(f"Export completed: {out_filename}")
    return counts


def log_transform_and_scale(
        adata,
        normalize_total=True,
        target_sum=1e4,
        n_top_genes=2000,
        inplace=False
):
    """
    Normalizes, log-transforms and scales an AnnData object for embedding and clustering.

    Parameters
    ----------
    adata : anndata.AnnData
        The input AnnData object holding raw counts.
    normalize_total : bool, default True
        If True, normalizes each cell to `target_sum` counts before the log
        transform. Set to False only to reproduce results produced before this
        step was introduced; log1p on raw counts is not comparable across cells
        with different library sizes.
    target_sum : float, default 1e4
        Target count depth per cell for the normalization.
    n_top_genes : int, default 2000
        Number of highly variable genes to retain.
    inplace : bool, default False
        If True, the normalization and log transform are applied to the input
        object as well. The scaled highly variable subset is returned either way.

    Returns
    -------
    anndata.AnnData
        A new AnnData object restricted to the highly variable genes and scaled.

    Examples
    --------
    >>> adata_hvf = log_transform_and_scale(adata=adata_qc)
    """
    print("Log-transforming and scaling...")
    if not inplace:
        adata = adata.copy()
    if normalize_total:
        sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
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
        celltype_factor: str = "celltype",
        sample_factor: str = "sample",
        gene_names: str = "gene_symbols",
        min_cells: int = 10,
        min_counts: int = 1000,
        filter_genes: bool = True,
        min_gene_count: int = 10,
        min_total_gene_count: int = 15,
        write_output_file=False,
        output_dir: str = ".",
        return_pseudobulk: bool = False,
        n_cpus: int = None,
        quiet: bool = True
):
    """
    Performs pseudobulk differential expression analysis using PyDESeq2.

    Aggregates single-cell data into one pseudobulk sample per sample and cell
    type, drops pseudobulk samples built from too few cells or counts, and fits
    a single `~ design_factor` model per cell type using all remaining samples,
    so that dispersion is estimated across the whole experiment. Each treatment
    is then extracted as a Wald contrast against the control and its effect size
    is shrunk.

    Conditions that lost all of their samples for a given cell type are removed
    from the design instead of being kept as empty factor levels, which would
    make the design matrix rank deficient. Cell types whose design is saturated
    after filtering are skipped, and conditions left without biological
    replication are reported as warnings.

    Parameters
    ----------
    combined_adata : anndata.AnnData
        The input single-cell AnnData object containing all combined samples,
        holding raw (unnormalized) counts in `.X`.
    celltype_names : list of str
        Cell types to analyze, matching the values in `adata.obs[celltype_factor]`.
    control_name : str
        The reference/control condition name for comparison.
    treatment_names : list of str
        List of treatment condition names to compare against the control.
    design_factor : str, default "condition"
        Column name in `adata.obs` representing the experimental condition/factor.
    celltype_factor : str, default "celltype"
        Column name in `adata.obs` holding the cell type assignment.
    sample_factor : str, default "sample"
        Column name in `adata.obs` identifying the sample to aggregate over.
    gene_names : str, default "gene_symbols"
        Column name in `adata.var` containing gene names to set as variable names.
    min_cells : int, default 10
        Minimum number of cells required per pseudobulk sample.
    min_counts : int, default 1000
        Minimum total counts required per pseudobulk sample.
    filter_genes : bool, default True
        If True, removes lowly expressed genes from each cell type's pseudobulk
        matrix before fitting. Filtering per cell type is deliberate: a gene
        expressed only in another cell type carries no signal in this contrast.
    min_gene_count : int, default 10
        Minimum count a gene must reach to be counted as expressed in a sample.
    min_total_gene_count : int, default 15
        Minimum total count a gene must reach across all samples of a cell type.
    write_output_file : bool, default False
        If True, saves differential expression results as CSV files.
    output_dir : str, default "."
        Directory for the CSV files written when `write_output_file` is True.
        Created if it does not exist.
    return_pseudobulk : bool, default False
        If True, also returns the filtered pseudobulk AnnData object, which is
        useful for sample-level diagnostics such as a PCA over the samples.
    n_cpus : int, optional
        Number of CPU threads to use for PyDESeq2 computations.
    quiet : bool, default True
        Suppress status updates during run.

    Returns
    -------
    dict of {str: dict of {str: pandas.DataFrame}}
        A nested dictionary mapping each cell type to its treatments, and each
        treatment to its DESeq2 results DataFrame. Cell types and treatments
        that could not be tested are absent, so iterate over the returned keys
        rather than over the input lists.
    anndata.AnnData
        The filtered pseudobulk object, returned only if `return_pseudobulk`
        is True.

    Examples
    --------
    >>> de_results = calculate_pseudobulk_deg(
    ...     combined_adata=combined_adata,
    ...     celltype_names=["Microglia", "Astrocyte"],
    ...     control_name="Control",
    ...     treatment_names=["LPS", "poly_ic"]
    ... )
    >>> de_results["Microglia"]["LPS"].head()
    """
    if gene_names not in combined_adata.var.columns:
        raise KeyError(
            f"Column '{gene_names}' not found in .var. Available columns: "
            f"{list(combined_adata.var.columns)}. Note that concatenating with "
            "merge='same' drops .var columns that are not identical across samples."
            )
    combined_adata.var_names = combined_adata.var[gene_names].astype(str)
    combined_adata.var_names_make_unique()

    pbdata = dc.pp.pseudobulk(
            combined_adata,
            sample_col=sample_factor,
            groups_col=celltype_factor
            )
    n_samples_before = pbdata.n_obs
    pbdata = _apply_filter(
            dc.pp.filter_samples,
            pbdata,
            min_cells=min_cells,
            min_counts=min_counts
            )
    print(f"{pbdata.n_obs} / {n_samples_before} pseudobulk samples passed filtering "
          f"(min_cells={min_cells}, min_counts={min_counts}).")

    if write_output_file:
        os.makedirs(output_dir, exist_ok=True)

    results_dict = {}

    for celltype in celltype_names:
        sub_pb = pbdata[pbdata.obs[celltype_factor] == celltype].copy()

        if sub_pb.n_obs == 0:
            print(f"Skipping {celltype}: no pseudobulk sample passed filtering.")
            continue

        if filter_genes:
            n_genes_before = sub_pb.n_vars
            # No 'group' argument on purpose: filter_by_expr derives its sample
            # threshold from the smallest group, so a condition with a single
            # replicate would leave the filter effectively inactive.
            sub_pb = _apply_filter(
                    dc.pp.filter_by_expr,
                    sub_pb,
                    min_count=min_gene_count,
                    min_total_count=min_total_gene_count
                    )
            print(f"{celltype}: {sub_pb.n_vars} / {n_genes_before} genes passed "
                  "expression filtering.")

        # Only conditions that still have samples may become design matrix
        # levels. An empty level produces an all-zero column and makes the
        # design matrix rank deficient.
        present_conditions = set(sub_pb.obs[design_factor].astype(str).unique())

        if control_name not in present_conditions:
            print(f"Skipping {celltype}: no '{control_name}' sample left.")
            continue

        usable_treatments = [t for t in treatment_names if t in present_conditions]
        dropped_treatments = [t for t in treatment_names if t not in present_conditions]

        if not usable_treatments:
            print(f"Skipping {celltype}: no treatment group left to compare.")
            continue
        if dropped_treatments:
            print(f"{celltype}: dropped treatments with no sample left: "
                  f"{', '.join(dropped_treatments)}.")

        sub_pb.obs[design_factor] = pd.Categorical(
                sub_pb.obs[design_factor].astype(str),
                categories=[control_name] + usable_treatments
                )

        n_levels = len(usable_treatments) + 1
        residual_df = sub_pb.n_obs - n_levels
        if residual_df < 1:
            print(f"Skipping {celltype}: saturated design ({sub_pb.n_obs} samples, "
                  f"{n_levels} conditions), dispersion is not estimable.")
            continue

        group_sizes = sub_pb.obs[design_factor].value_counts()
        unreplicated = [str(c) for c in group_sizes[group_sizes == 1].index]
        if unreplicated:
            print(f"Warning - {celltype}: no biological replication for "
                  f"{', '.join(unreplicated)} (n=1). Their contrasts cannot be "
                  "separated from sample-specific effects and should be "
                  "reported separately.")

        print(f"Fitting {celltype}: {sub_pb.n_obs} samples, {n_levels} conditions, "
              f"{residual_df} residual degrees of freedom.")

        dds = DeseqDataSet(
                adata=sub_pb,
                design=f"~ {design_factor}",
                n_cpus=n_cpus,
                quiet=quiet
                )
        dds.deseq2()

        results_dict[celltype] = {}

        for treatment in usable_treatments:
            res = DeseqStats(
                dds,
                contrast=[design_factor, treatment, control_name],
                quiet=quiet
                )
            res.summary()

            # The control is the reference level of the categorical above, so
            # this coefficient is the same contrast that was just tested.
            res.lfc_shrink(coeff=f"{design_factor}[T.{treatment}]")
            results_dict[celltype][treatment] = res.results_df
            if write_output_file:
                res.results_df.to_csv(
                    os.path.join(output_dir, f"{celltype}_{treatment}_all_DEG.csv")
                    )

    if return_pseudobulk:
        return results_dict, pbdata
    return results_dict


def calculate_cell_level_deg(
        combined_adata,
        celltype_names: list,
        control_name: str,
        treatment_names: list,
        design_factor: str = "condition",
        celltype_factor: str = "celltype",
        sample_factor: str = "sample",
        gene_names: str = "gene_symbols",
        method: str = "wilcoxon",
        layer: str = None,
        target_sum: float = 1e4,
        min_cells: int = 20,
        min_pct: float = 0.1,
        lfc_threshold: float = 0.25,
        p_threshold: float = 0.05,
        min_pairs_passed: int = 1,
        write_output_file=False,
        output_dir: str = ".",
        return_pairwise: bool = False,
        quiet: bool = True
):
    """
    Performs cell-level differential expression analysis between pairs of samples.

    This is the cell-level counterpart of `calculate_pseudobulk_deg` and takes the
    same input object. Instead of aggregating cells, every control sample is
    compared against every treated sample of a condition separately, one cell type
    at a time, using the same rank-sum test that Seurat's FindMarkers applies by
    default. The per-pair results are then summarized into how consistently each
    gene changes in the same direction across the pairs.

    Cells are not independent observations of a treatment, so the p-values of a
    single pair only describe those two animals and are anticonservative as a
    statement about the condition. The summary is therefore ranked by consistency
    across pairs and by effect size, not by p-value, and is meant as a descriptive
    layer beside the pseudobulk analysis rather than as a replacement for it. What
    this layer adds over pseudobulk is the fraction of expressing cells, reported
    as `delta_pct`, which captures responses that change how many cells express a
    gene rather than how much they express it.

    Parameters
    ----------
    combined_adata : anndata.AnnData
        The input single-cell AnnData object containing all combined samples,
        holding raw (unnormalized) counts in `.X` unless `layer` is given.
    celltype_names : list of str
        Cell types to analyze, matching the values in `adata.obs[celltype_factor]`.
    control_name : str
        The reference/control condition name for comparison.
    treatment_names : list of str
        List of treatment condition names to compare against the control.
    design_factor : str, default "condition"
        Column name in `adata.obs` representing the experimental condition/factor.
    celltype_factor : str, default "celltype"
        Column name in `adata.obs` holding the cell type assignment.
    sample_factor : str, default "sample"
        Column name in `adata.obs` identifying the sample.
    gene_names : str, default "gene_symbols"
        Column name in `adata.var` containing gene names to set as variable names.
    method : str, default "wilcoxon"
        Test passed to `scanpy.tl.rank_genes_groups`. The default matches the
        Seurat FindMarkers default; "t-test" and "logreg" are also accepted.
    layer : str, optional
        Name of a layer already holding log-normalized values. If omitted, each
        cell type is normalized to `target_sum` counts and log-transformed
        internally. Note that this differs from the pseudobulk analysis, which
        requires raw counts.
    target_sum : float, default 1e4
        Target count depth per cell for the internal normalization. An explicit
        value is required so that normalizing one cell type at a time gives the
        same result as normalizing the whole object would.
    min_cells : int, default 20
        Minimum number of cells a sample must contribute to a cell type for that
        sample pair to be tested.
    min_pct : float, default 0.1
        A gene is only considered testable in a pair if at least this fraction of
        cells express it in one of the two samples.
    lfc_threshold : float, default 0.25
        Minimum absolute log2 fold change for a gene to count as changed in a pair.
    p_threshold : float, default 0.05
        Maximum adjusted p-value for a gene to count as changed in a pair.
    min_pairs_passed : int, default 1
        Genes changed in fewer pairs than this are left out of the summary.
    write_output_file : bool, default False
        If True, saves the summary and per-pair tables as CSV files.
    output_dir : str, default "."
        Directory for the CSV files written when `write_output_file` is True.
        Created if it does not exist.
    return_pairwise : bool, default False
        If True, returns the per-pair tables as a second value alongside the
        summaries. The per-pair tables are written to disk whenever
        `write_output_file` is True, regardless of this setting.
    quiet : bool, default True
        Suppress per-pair status updates during run.

    Returns
    -------
    dict of {str: dict of {str: pandas.DataFrame}}
        A nested dictionary mapping each cell type to its treatments, and each
        treatment to its consistency summary. Cell types and treatments that
        could not be tested are absent, so iterate over the returned keys rather
        than over the input lists.
    dict of {str: dict of {str: pandas.DataFrame}}
        The per-pair results in the same nested shape, returned only if
        `return_pairwise` is True.

    Examples
    --------
    >>> summaries, pairwise = calculate_celllevel_deg(
    ...     combined_adata=combined_adata,
    ...     celltype_names=["Microglia", "Astrocyte"],
    ...     control_name="Control",
    ...     treatment_names=["LPS", "poly_ic"],
    ...     return_pairwise=True
    ... )
    >>> summaries["Microglia"]["LPS"].head()
    """
    if gene_names not in combined_adata.var.columns:
        raise KeyError(
            f"Column '{gene_names}' not found in .var. Available columns: "
            f"{list(combined_adata.var.columns)}. Note that concatenating with "
            "merge='same' drops .var columns that are not identical across samples."
            )
    combined_adata.var_names = combined_adata.var[gene_names].astype(str)
    combined_adata.var_names_make_unique()

    sample_conditions = (
        combined_adata.obs[[sample_factor, design_factor]]
        .astype(str)
        .drop_duplicates()
        .set_index(sample_factor)[design_factor]
        )
    control_samples = sorted(sample_conditions[sample_conditions == control_name].index)
    if not control_samples:
        raise ValueError(
            f"No sample is assigned to the control condition '{control_name}'."
            )

    # Every control sample is paired with every treated sample of a condition.
    comparison_plan = {}
    for treatment in treatment_names:
        treated_samples = sorted(sample_conditions[sample_conditions == treatment].index)
        if not treated_samples:
            print(f"Skipping {treatment}: no sample assigned to this condition.")
            continue
        comparison_plan[treatment] = [
            (control_sample, treatment_sample)
            for control_sample in control_samples
            for treatment_sample in treated_samples
            ]
        if len(treated_samples) == 1:
            print(f"Warning - {treatment}: only one treated sample (n=1), so all of "
                  "its pairs share the same animal and cannot separate the treatment "
                  "from sample-specific effects.")

    if not comparison_plan:
        raise ValueError("None of the requested treatments has any sample.")

    n_pairs_total = sum(len(pairs) for pairs in comparison_plan.values())
    print(f"{len(control_samples)} control samples, {n_pairs_total} sample pairs over "
          f"{len(comparison_plan)} treatments, {len(celltype_names)} cell types "
          f"({n_pairs_total * len(celltype_names)} tests at most).")

    if write_output_file:
        os.makedirs(output_dir, exist_ok=True)

    results_dict = {}
    pairwise_dict = {}

    for celltype in celltype_names:
        celltype_mask = (combined_adata.obs[celltype_factor].astype(str) == celltype).to_numpy()
        if celltype_mask.sum() == 0:
            print(f"Skipping {celltype}: no cell carries this annotation.")
            continue

        celltype_adata = combined_adata[celltype_mask].copy()

        # Normalizing per cell type is equivalent to normalizing the whole object
        # because the depth target is explicit and the transform is per cell.
        if layer is None:
            sc.pp.normalize_total(celltype_adata, target_sum=target_sum)
            sc.pp.log1p(celltype_adata)
        elif layer in celltype_adata.layers:
            celltype_adata.X = celltype_adata.layers[layer]
        else:
            raise KeyError(f"Layer '{layer}' not found in the input object.")

        celltype_results = {}
        celltype_pairwise = {}

        for treatment, pairs in comparison_plan.items():
            pair_frames = []
            for control_sample, treatment_sample in pairs:
                pair_frame = _test_sample_pair(
                        celltype_adata=celltype_adata,
                        control_sample=control_sample,
                        treatment_sample=treatment_sample,
                        sample_factor=sample_factor,
                        method=method,
                        min_cells=min_cells,
                        min_pct=min_pct,
                        lfc_threshold=lfc_threshold,
                        p_threshold=p_threshold,
                        label=f"{celltype} / {treatment}",
                        quiet=quiet
                        )
                if pair_frame is not None:
                    pair_frames.append(pair_frame)

            if not pair_frames:
                print(f"Skipping {celltype} / {treatment}: no testable sample pair.")
                continue

            pairwise_df = pd.concat(pair_frames, ignore_index=True)
            summary_df = _summarize_pairwise_deg(pairwise_df, min_pairs_passed)

            celltype_results[treatment] = summary_df
            celltype_pairwise[treatment] = pairwise_df

            print(f"{celltype} / {treatment}: {len(pair_frames)} / {len(pairs)} pairs "
                  f"tested, {summary_df.shape[0]} genes changed in at least "
                  f"{min_pairs_passed} pair(s), "
                  f"{int((summary_df['consistency'] == 1).sum())} in all of them.")

            if write_output_file:
                summary_df.to_csv(
                    os.path.join(output_dir, f"{celltype}_{treatment}_celllevel_summary.csv"),
                    index=False
                    )
                pairwise_df.to_csv(
                    os.path.join(output_dir, f"{celltype}_{treatment}_celllevel_pairwise.csv"),
                    index=False
                    )

        if celltype_results:
            results_dict[celltype] = celltype_results
            pairwise_dict[celltype] = celltype_pairwise

    if return_pairwise:
        return results_dict, pairwise_dict
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
    if "stat" not in de_data.columns:
        raise KeyError(
            "Column 'stat' not found in the differential expression results. "
            "The Wald statistic is required to rank genes for enrichment."
            )
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
    Genes that DESeq2 left without a result, through independent filtering or Cook's
    distance outlier removal, carry NaN values and are dropped before filtering.

    A 'neg_log10_padj' column is added for plotting. Note that the log2-fold change
    is the shrunken estimate while the adjusted p-value comes from the Wald test
    performed before shrinkage, as DESeq2 does not recompute p-values during
    shrinkage. To test the threshold itself, pass `lfc_null` and `alt_hypothesis`
    to DeseqStats instead of filtering afterwards.

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
        alongside their corresponding statistics, sorted by decreasing evidence.

    Examples
    --------
    >>> filter_significant_genes(
    ...     de_data=deseq_results,
    ...     lfc_threshold=1.0,
    ...     p_threshold=0.01,
    ...     top=50
    ... )
    """
    df = de_data.copy()
    df = df.dropna(subset=["log2FoldChange", "padj"])
    if df.empty:
        print("No gene has both a log2FoldChange and an adjusted p-value.")
        return df

    # Kept on the -log10 scale for plotting, and named explicitly so it is not
    # mistaken for a p-value in the exported tables.
    non_zero_min = df["padj"][df["padj"] > 0].min()
    if pd.isna(non_zero_min):
        non_zero_min = np.finfo(float).tiny
    df["neg_log10_padj"] = -np.log10(df["padj"].clip(lower=non_zero_min, upper=1))

    thr_msk = (np.abs(df["log2FoldChange"]) >= lfc_threshold) & (df["padj"] <= p_threshold)
    signs = df[thr_msk].copy()

    # lfc_shrink replaces the effect size but leaves the Wald statistic in place.
    # Fall back to the shrunken effect size if the column is not available.
    if "stat" in signs.columns:
        signs["abs_stat"] = np.abs(signs["stat"])
        signs = signs.sort_values("abs_stat", ascending=False)
    else:
        signs["abs_lfc"] = np.abs(signs["log2FoldChange"])
        signs = signs.sort_values("abs_lfc", ascending=False)

    print(f"{signs.shape[0]} / {df.shape[0]} genes passed the significance thresholds "
          f"(|log2FoldChange| >= {lfc_threshold}, padj <= {p_threshold}).")
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


def expressed_fraction(adata, cell_mask):
    """
    Returns the fraction of the selected cells in which each gene is detected.

    Computed directly from the matrix rather than read from the scanpy result,
    so that the column is available regardless of the scanpy version.
    """
    matrix = adata[cell_mask].X
    n_cells = int(cell_mask.sum())
    if issparse(matrix):
        detected = np.asarray((matrix > 0).sum(axis=0)).ravel()
    else:
        detected = np.asarray(np.asarray(matrix) > 0).sum(axis=0).ravel()
    return pd.Series(detected / n_cells, index=adata.var_names)


def _test_sample_pair(
        celltype_adata,
        control_sample: str,
        treatment_sample: str,
        sample_factor: str,
        method: str,
        min_cells: int,
        min_pct: float,
        lfc_threshold: float,
        p_threshold: float,
        label: str,
        quiet: bool
):
    """
    Runs one rank-sum comparison between a control and a treated sample.

    Returns a table of the genes testable in this pair, with a 'passed' flag
    marking those that met the effect size and p-value thresholds, or None if
    either sample contributed too few cells.
    """
    pair_samples = celltype_adata.obs[sample_factor].astype(str).to_numpy()
    control_mask = pair_samples == control_sample
    treatment_mask = pair_samples == treatment_sample
    n_control = int(control_mask.sum())
    n_treatment = int(treatment_mask.sum())

    if n_control < min_cells or n_treatment < min_cells:
        if not quiet:
            print(f"{label}: skipping {treatment_sample} vs {control_sample}, "
                  f"{n_treatment} and {n_control} cells (min_cells={min_cells}).")
        return None

    pair_adata = celltype_adata[control_mask | treatment_mask].copy()
    # Only the two samples of this pair may remain as groups; an empty category
    # would otherwise be carried over from the full sample list.
    pair_adata.obs[sample_factor] = pd.Categorical(
            pair_adata.obs[sample_factor].astype(str),
            categories=[control_sample, treatment_sample]
            )

    try:
        sc.tl.rank_genes_groups(
                pair_adata,
                groupby=sample_factor,
                groups=[treatment_sample],
                reference=control_sample,
                method=method
                )
    except Exception as error:
        print(f"{label}: failed on {treatment_sample} vs {control_sample} - {error}")
        return None

    pair_df = sc.get.rank_genes_groups_df(pair_adata, group=treatment_sample)
    pair_df = pair_df.rename(columns={
        "names": "gene",
        "logfoldchanges": "log2FoldChange",
        "pvals": "pvalue",
        "pvals_adj": "padj"
        })

    in_pair_samples = pair_adata.obs[sample_factor].astype(str).to_numpy()
    pct_control = expressed_fraction(pair_adata, in_pair_samples == control_sample)
    pct_treatment = expressed_fraction(pair_adata, in_pair_samples == treatment_sample)
    pair_df["pct_control"] = pair_df["gene"].map(pct_control)
    pair_df["pct_treatment"] = pair_df["gene"].map(pct_treatment)

    # Genes absent from one side produce infinite fold changes that would
    # distort the medians taken across pairs.
    pair_df["log2FoldChange"] = pair_df["log2FoldChange"].replace(
            [np.inf, -np.inf], np.nan
            )

    testable = (pair_df["pct_control"] >= min_pct) | (pair_df["pct_treatment"] >= min_pct)
    pair_df = pair_df[testable].copy()

    pair_df["passed"] = (
        (pair_df["log2FoldChange"].abs() >= lfc_threshold)
        & (pair_df["padj"] <= p_threshold)
        )
    pair_df["control_sample"] = control_sample
    pair_df["treatment_sample"] = treatment_sample
    pair_df["n_cells_control"] = n_control
    pair_df["n_cells_treatment"] = n_treatment

    if not quiet:
        print(f"{label}: {treatment_sample} vs {control_sample}, "
              f"{n_treatment} vs {n_control} cells, "
              f"{int(pair_df['passed'].sum())} / {pair_df.shape[0]} genes changed.")

    return pair_df[[
        "gene", "control_sample", "treatment_sample",
        "n_cells_control", "n_cells_treatment",
        "log2FoldChange", "pct_control", "pct_treatment",
        "pvalue", "padj", "passed"
        ]]


def _summarize_pairwise_deg(pairwise_df, min_pairs_passed: int):
    """
    Summarizes per-pair results into one row per gene.

    Counts in how many pairs a gene changed in each direction and reports the
    medians of the per-pair statistics. Genes are ranked by the share of pairs
    agreeing on a direction, then by effect size, because with two animals per
    condition the agreement between pairs carries the evidence rather than the
    per-pair p-values.
    """
    scored = pairwise_df.copy()
    scored["is_up"] = scored["passed"] & (scored["log2FoldChange"] > 0)
    scored["is_down"] = scored["passed"] & (scored["log2FoldChange"] < 0)

    summary = scored.groupby("gene", sort=False).agg(
            n_pairs=("gene", "size"),
            n_up=("is_up", "sum"),
            n_down=("is_down", "sum"),
            median_log2FC=("log2FoldChange", "median"),
            median_pct_control=("pct_control", "median"),
            median_pct_treatment=("pct_treatment", "median"),
            median_padj=("padj", "median"),
            median_n_cells_treatment=("n_cells_treatment", "median")
            ).reset_index()

    summary["n_up"] = summary["n_up"].astype(int)
    summary["n_down"] = summary["n_down"].astype(int)
    summary["consistency"] = (
        summary[["n_up", "n_down"]].max(axis=1) / summary["n_pairs"]
        )
    summary["direction"] = np.where(
            summary["n_up"] > summary["n_down"], "up",
            np.where(summary["n_down"] > summary["n_up"], "down", "inconsistent")
            )
    summary["delta_pct"] = (
        summary["median_pct_treatment"] - summary["median_pct_control"]
        )

    summary = summary[(summary["n_up"] + summary["n_down"]) >= min_pairs_passed].copy()
    summary["abs_median_log2FC"] = summary["median_log2FC"].abs()
    summary = summary.sort_values(
            ["consistency", "abs_median_log2FC"],
            ascending=False,
            na_position="last"
            )

    return summary[[
        "gene", "n_pairs", "n_up", "n_down", "consistency", "direction",
        "median_log2FC", "median_pct_control", "median_pct_treatment", "delta_pct",
        "median_padj", "median_n_cells_treatment"
        ]].reset_index(drop=True)


def _apply_filter(filter_function, adata, **kwargs):
    """
    Applies a decoupler filtering function and returns the resulting object.

    The decoupler preprocessing functions modify the object in place and return
    None, but this is not guaranteed across versions, so both conventions are
    handled here.
    """
    result = filter_function(adata, **kwargs)
    return adata if result is None else result


def _read_custom_mappings(excel_file, sheet_name):
    xls_data = pd.read_excel(excel_file, sheet_name=sheet_name)
    mappings = {
        col: xls_data[col].dropna().tolist()
        for col in xls_data.columns
    }
    return mappings