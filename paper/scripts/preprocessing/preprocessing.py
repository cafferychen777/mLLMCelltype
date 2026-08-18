#!/usr/bin/env python3
"""
Preprocessing Utilities for Single-Cell RNA-seq Data

This module provides utility functions for preprocessing single-cell data,
including gene name cleaning, duplicate handling, and QC annotation.

Reference: Adapted from HLCA reproducibility pipeline
https://github.com/LungCellAtlas/HLCA

Usage:
    from preprocessing import (
        get_gene_renamer_dict,
        add_up_duplicate_gene_name_columns,
        add_cell_annotations,
        generate_r2python_gene_mapping
    )
"""

import collections
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Try to import scanpy/anndata, but don't fail if not available
try:
    import anndata
    import scanpy as sc

    HAS_SCANPY = True
except ImportError:
    HAS_SCANPY = False


def get_gene_renamer_dict(gene_names):
    """
    Detect and create mapping for misnamed genes.

    This function detects genes that are misnamed and creates a mapping dictionary.
    It handles:
    1. Genes with double dashes ("--") -> double underscores ("__")
    2. Genes with "1" suffix (.1, -1, _1) that also exist without suffix
    3. Genes with corresponding but different suffixes (.[suffix] vs -[suffix])

    Note: Should only be run after removing duplicate gene names!

    Parameters
    ----------
    gene_names : list
        List of gene names from adata.var.index

    Returns
    -------
    dict
        Dictionary mapping original gene names to corrected versions
    """
    # Check for duplicate gene names
    if len(gene_names) > len(set(gene_names)):
        raise ValueError(
            "There are duplicate gene names in your gene list. "
            "This function requires unique names. Remove duplicates first."
        )

    # Rename genes with double dash to double underscore
    genes_with_double_dash = [gene for gene in gene_names if "--" in gene]
    genes_to_change_suffix_dict = {}

    # Find genes with suffixed and non-suffixed versions
    for gene in gene_names:
        if (
            len(gene) > 1
            and gene[-2:] in [".1", "-1", "_1"]
            and gene[:-2] in gene_names
        ):
            genes_to_change_suffix_dict[gene] = gene[:-2]

    # Find genes with both .[suffix] and -[suffix] versions
    a = [
        _rreplace(
            original_gene_name,
            old="-",
            new=".",
            occurrence=1,
            only_if_not_followed_by=".",
        )
        for original_gene_name in gene_names
    ]
    from_dash_to_dot = [
        item for item, count in collections.Counter(a).items() if count > 1
    ]

    for gene in from_dash_to_dot:
        gene_minus_suffix_list = gene.split(".")
        gene_minus_suffix = ".".join(gene_minus_suffix_list[:-1])

        # Map "-" version to new name
        dot_position = gene.rfind(".")
        gene_as_list = list(gene)
        gene_as_list_dash = gene_as_list
        gene_as_list_dash[dot_position] = "-"
        gene_dash_name = "".join(gene_as_list_dash)

        # If suffix is single digit, use version without suffix
        if gene.split(".")[-1] in [str(digit) for digit in range(10)]:
            genes_to_change_suffix_dict[gene] = gene_minus_suffix
            genes_to_change_suffix_dict[gene_dash_name] = gene_minus_suffix
        else:
            # Use dash version (usually ensembl)
            genes_to_change_suffix_dict[gene] = gene_dash_name

    # Create remapping dictionary
    genes_remapper = {}
    for gene in genes_with_double_dash:
        genes_remapper[gene] = gene.replace("--", "__")
    for gene_name_old, gene_name_new in genes_to_change_suffix_dict.items():
        genes_remapper[gene_name_old] = gene_name_new

    return genes_remapper


def subset_and_pad_adata(gene_set, adata):
    """
    Subset AnnData to specific genes and pad missing genes with zeros.

    Parameters
    ----------
    gene_set : pd.DataFrame
        DataFrame with 'gene_symbols' column and optional Ensembl IDs
    adata : anndata.AnnData
        AnnData object to subset

    Returns
    -------
    anndata.AnnData
        Subsetted and padded AnnData object
    """
    if not HAS_SCANPY:
        raise ImportError("scanpy and anndata are required for this function")

    # Prep objects
    if "gene_symbols" in gene_set.columns:
        gene_set.index = gene_set["gene_symbols"]
    else:
        raise ValueError('The input gene list must have a "gene_symbols" column')

    # Subset adata object
    common_genes = [
        gene for gene in gene_set["gene_symbols"].values if gene in adata.var_names
    ]

    if len(common_genes) == 0:
        print("WARNING: No genes recovered. Check if var index uses gene symbols.")
        return None

    adata_sub = adata[:, common_genes].copy()

    # Pad with zeros if needed
    if len(common_genes) < len(gene_set):
        diff = len(gene_set) - len(common_genes)
        print(f"Padding with 0 counts for {diff} missing genes...")

        genes_to_add = set(gene_set["gene_symbols"].values).difference(
            set(adata_sub.var_names)
        )
        new_var = gene_set.loc[genes_to_add]

        if "Unnamed: 0" in new_var.columns:
            new_var["ensembl"] = new_var["Unnamed: 0"]
            del new_var["Unnamed: 0"]

        df_padding = pd.DataFrame(
            data=np.zeros((adata_sub.shape[0], len(genes_to_add))),
            index=adata_sub.obs_names,
            columns=new_var.index,
        )
        adata_padding = sc.AnnData(df_padding, var=new_var)

        adata_sub = anndata.concat(
            [adata_sub, adata_padding],
            axis=1,
            join="outer",
            index_unique=None,
            merge="unique",
        )

    # Ensure ensembl IDs are available
    if "Unnamed: 0" in gene_set.columns:
        adata_sub.var["ensembl"] = gene_set["Unnamed: 0"]

    return adata_sub


def add_up_duplicate_gene_name_columns(adata, print_gene_names=True, verbose=False):
    """
    Merge duplicate gene columns by summing their counts.

    For each cell, adds up counts of columns with the same gene name,
    removes duplicate columns, and replaces counts with the sum.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object with potential duplicate gene names
    print_gene_names : bool
        Whether to print duplicate gene names (default: True)
    verbose : bool
        Whether to print progress (default: False)

    Returns
    -------
    anndata.AnnData
        AnnData object with merged duplicate genes
    """
    duplicate_boolean = adata.var.index.duplicated()
    duplicate_genes = adata.var.index[duplicate_boolean]
    print(f"Number of duplicate genes: {len(duplicate_genes)}")

    if print_gene_names:
        print(duplicate_genes.tolist())

    columns_to_replace = []
    columns_to_remove = []
    new_columns_array = np.empty((adata.shape[0], 0))

    for gene in duplicate_genes:
        if verbose:
            print(f"Processing gene: {gene}")

        gene_colnumbers = np.where(adata.var.index == gene)[0]
        gene_counts = adata.X[:, gene_colnumbers].copy()

        # Sum counts and add to array
        new_columns_array = np.hstack((new_columns_array, np.sum(gene_counts, axis=1)))

        columns_to_replace.append(gene_colnumbers[0])
        columns_to_remove.extend(gene_colnumbers[1:].tolist())

    # Replace first column with summed counts
    adata.X[:, columns_to_replace] = new_columns_array

    # Remove duplicate columns
    columns_to_keep = [
        i for i in np.arange(adata.shape[1]) if i not in columns_to_remove
    ]
    adata = adata[:, columns_to_keep].copy()

    if verbose:
        print("Done!")

    return adata


def add_cell_annotations(adata, var_index="gene_symbols"):
    """
    Add QC annotations to AnnData object.

    Adds cell-level annotations:
    - total_counts, log10_total_counts
    - n_genes_detected
    - mito_frac, ribo_frac
    - compl (complexity)

    Adds gene-level annotations:
    - n_cells

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object with raw (unnormalized) counts
    var_index : str
        Type of gene naming in adata.var.index ("gene_symbols" or "gene_ids")

    Returns
    -------
    anndata.AnnData
        AnnData object with QC annotations
    """
    # Total counts per cell
    adata.obs["total_counts"] = np.sum(adata.X, axis=1)
    adata.obs["log10_total_counts"] = np.log10(adata.obs["total_counts"])

    # Number of genes detected
    boolean_expression = adata.X > 0
    adata.obs["n_genes_detected"] = np.sum(boolean_expression, axis=1)

    # Mitochondrial fraction
    if var_index == "gene_symbols":
        mito_genes = [
            gene for gene in adata.var.index if gene.lower().startswith("mt-")
        ]
    elif var_index == "gene_ids":
        mito_genes = [
            gene_id
            for gene_id, gene_symbol in zip(adata.var.index, adata.var.gene_symbols)
            if gene_symbol.lower().startswith("mt-")
        ]
    else:
        raise ValueError("var_index should be 'gene_symbols' or 'gene_ids'")

    adata.obs["mito_frac"] = (
        np.array(np.sum(adata[:, mito_genes].X, axis=1)).flatten()
        / adata.obs["total_counts"]
    )

    # Ribosomal fraction
    if var_index == "gene_symbols":
        ribo_genes = [
            gene
            for gene in adata.var.index
            if (gene.lower().startswith("rpl") or gene.lower().startswith("rps"))
        ]
    elif var_index == "gene_ids":
        ribo_genes = [
            gene_id
            for gene_id, gene_symbol in zip(adata.var.index, adata.var.gene_symbols)
            if (
                gene_symbol.lower().startswith("rpl")
                or gene_symbol.lower().startswith("rps")
            )
        ]

    adata.obs["ribo_frac"] = (
        np.array(np.sum(adata[:, ribo_genes].X, axis=1)).flatten()
        / adata.obs["total_counts"]
    )

    # Cell complexity
    adata.obs["compl"] = adata.obs["n_genes_detected"] / adata.obs["total_counts"]

    # Gene-level: number of cells expressing
    adata.var["n_cells"] = np.sum(boolean_expression, axis=0).T

    return adata


def generate_r2python_gene_mapping(r_genes, python_genes):
    """
    Create mapping from R gene names to Python gene names.

    R replaces "-" with "." in gene names, this function creates a mapping
    to convert back.

    Parameters
    ----------
    r_genes : list
        Gene names from R (with "." instead of "-")
    python_genes : list
        Gene names from Python

    Returns
    -------
    dict
        Mapping from R gene names to Python gene names
    """
    r2python_gene_mapping = {}

    for gene in r_genes:
        if gene in python_genes:
            r2python_gene_mapping[gene] = gene
        elif "-".join(gene.rsplit(".", 1)) in python_genes:
            r2python_gene_mapping[gene] = "-".join(gene.rsplit(".", 1))
        elif "-".join(gene.split(".", 1)) in python_genes:
            r2python_gene_mapping[gene] = "-".join(gene.split(".", 1))
        elif "-".join(gene.split(".")) in python_genes:
            r2python_gene_mapping[gene] = "-".join(gene.split("."))
        else:
            print(f"No equivalent found for gene: {gene}")

    return r2python_gene_mapping


def update_adata(adata, cov1, cov2, mapping):
    """
    Update AnnData obs column based on mapping.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData to update
    cov1 : str
        Column name to map from
    cov2 : str
        Column name to map to
    mapping : dict
        Mapping from cov1 categories to cov2 values

    Returns
    -------
    anndata.AnnData
        Updated AnnData
    """
    cov1_in_data = adata.obs[cov1].unique()

    for cat in mapping:
        if cat not in cov1_in_data:
            raise ValueError(f"{cat} not a category of {cov1} in your data.")

    cov2_updates = adata.obs[cov1].map(mapping).dropna()

    # Set to list to allow new categories
    adata.obs[cov2] = adata.obs[cov2].tolist()
    adata.obs.loc[cov2_updates.index, cov2] = cov2_updates.values

    # Convert back to categorical
    adata.obs[cov2] = pd.Categorical(adata.obs[cov2])

    return adata


def plot_QC(anndata_dict, project_name, scale=1):
    """
    Create QC plots for multiple samples.

    Parameters
    ----------
    anndata_dict : dict
        Dictionary mapping sample IDs to AnnData objects
    project_name : str
        Name for the plot title
    scale : float
        Scale factor for figure size (default: 1)

    Returns
    -------
    matplotlib.figure.Figure
        Figure with QC plots
    """
    n_cols = 4
    n_rows = len(anndata_dict.keys())
    fig = plt.figure(figsize=(n_cols * 3 * scale, n_rows * 2.5 * scale))
    plt.suptitle(f"QC {project_name}", fontsize=16, y=1.02)
    title_fontsize = 8
    ax_dict = {}
    ax = 1

    for sample_ID in sorted(anndata_dict.keys()):
        adata = anndata_dict[sample_ID]

        # Total counts
        ax_dict[ax] = fig.add_subplot(n_rows, n_cols, ax)
        ax_dict[ax].hist(adata.obs["log10_total_counts"], bins=50, range=(2, 5))
        ax_dict[ax].set_title("log10 total counts per cell", fontsize=title_fontsize)
        ax_dict[ax].set_xlabel("log10 total counts")
        ax_dict[ax].set_ylabel(sample_ID, fontsize=16)
        ax += 1

        # Mitochondrial fraction
        ax_dict[ax] = fig.add_subplot(n_rows, n_cols, ax)
        ax_dict[ax].hist(adata.obs["mito_frac"], bins=50, range=(0, 1))
        ax_dict[ax].set_title(
            "fraction of mitochondrial transcripts/cell", fontsize=title_fontsize
        )
        ax_dict[ax].set_xlabel("fraction of mito transcripts")
        ax_dict[ax].set_ylabel("frequency")
        ax += 1

        # Number of genes detected
        ax_dict[ax] = fig.add_subplot(n_rows, n_cols, ax)
        ax_dict[ax].hist(np.log10(adata.obs["n_genes_detected"]), bins=50, range=(2, 5))
        ax_dict[ax].set_title("log10 number of genes detected", fontsize=title_fontsize)
        ax_dict[ax].set_xlabel("log10 number of genes")
        ax_dict[ax].set_ylabel("frequency")
        ax += 1

        # Cell complexity scatter
        ax_dict[ax] = fig.add_subplot(n_rows, n_cols, ax)
        mappable_ax = f"{ax}_m"
        ax_dict[mappable_ax] = ax_dict[ax].scatter(
            adata.obs["log10_total_counts"],
            np.log10(adata.obs["n_genes_detected"]),
            c=adata.obs["mito_frac"],
            s=1,
        )
        ax_dict[ax].set_title(
            "cell complexity\ncolored by mitochondrial fraction",
            fontsize=title_fontsize,
        )
        ax_dict[ax].set_xlabel("log10 total counts")
        ax_dict[ax].set_ylabel("log10 number of genes detected")
        fig.colorbar(mappable=ax_dict[mappable_ax], ax=ax_dict[ax])
        ax += 1

    plt.tight_layout()
    return fig


def age_converter(age, age_range, verbose=False):
    """
    Convert age or age range to numeric value.

    Takes either age or age_range (format: "[number]-[number]") and returns
    numeric age value. If only age_range is provided, returns the mean.

    Parameters
    ----------
    age : float or str
        Age value (may be NaN)
    age_range : str or float
        Age range string (e.g., "20-30") or NaN
    verbose : bool
        Whether to print debug info (default: False)

    Returns
    -------
    float
        Numeric age value
    """
    if isinstance(age, float):
        if np.isnan(age):
            if verbose:
                print(age_range)
            if isinstance(age_range, float):
                if np.isnan(age_range):
                    if verbose:
                        print("no age or age range available")
                    return np.nan
            else:
                age_lower = float(age_range.split("-")[0])
                age_higher = float(age_range.split("-")[1])
                mean_age = np.mean([age_lower, age_higher])
            if verbose:
                print(mean_age)
            return mean_age
        else:
            return age
    else:
        return age


# =============================================================================
# Helper Functions
# =============================================================================


def _rreplace(s, old, new, occurrence, only_if_not_followed_by=None):
    """
    Replace occurrences of character from right to left.

    Parameters
    ----------
    s : str
        Input string
    old : str
        Character to replace
    new : str
        Replacement character
    occurrence : int
        Number of occurrences to replace (from end)
    only_if_not_followed_by : str, optional
        Only replace if not followed by this string

    Returns
    -------
    str
        String with replacements
    """
    if old not in s:
        return s
    elif only_if_not_followed_by is not None:
        last_inst_old = s.rfind(old)
        last_inst_oinfb = s.rfind(only_if_not_followed_by)
        if last_inst_oinfb > last_inst_old:
            return s
    li = s.rsplit(old, occurrence)
    return new.join(li)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocessing utilities for single-cell RNA-seq data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Show duplicate genes in a dataset
    python preprocessing.py --input data.h5ad --check-duplicates

    # Add QC annotations
    python preprocessing.py --input data.h5ad --output data_qc.h5ad --add-qc

Requirements:
    pip install scanpy anndata pandas numpy matplotlib
        """,
    )
    parser.add_argument("--input", "-i", help="Path to input h5ad file")
    parser.add_argument("--output", "-o", help="Path to save processed h5ad file")
    parser.add_argument(
        "--check-duplicates",
        action="store_true",
        help="Check and report duplicate gene names",
    )
    parser.add_argument(
        "--merge-duplicates",
        action="store_true",
        help="Merge duplicate gene names by summing counts",
    )
    parser.add_argument(
        "--add-qc", action="store_true", help="Add QC annotations to the dataset"
    )
    parser.add_argument(
        "--var-index",
        default="gene_symbols",
        choices=["gene_symbols", "gene_ids"],
        help="Type of gene naming in var.index (default: gene_symbols)",
    )

    args = parser.parse_args()

    if args.input:
        if not HAS_SCANPY:
            print("Error: scanpy is required. Install with: pip install scanpy")
            sys.exit(1)

        print(f"Loading: {args.input}")
        adata = sc.read_h5ad(args.input)
        print(f"Shape: {adata.n_obs} cells x {adata.n_vars} genes")

        if args.check_duplicates:
            duplicates = adata.var.index[adata.var.index.duplicated()].unique()
            print(f"\nDuplicate genes ({len(duplicates)}):")
            for gene in duplicates[:20]:
                print(f"  - {gene}")
            if len(duplicates) > 20:
                print(f"  ... and {len(duplicates) - 20} more")

        if args.merge_duplicates:
            print("\nMerging duplicate genes...")
            adata = add_up_duplicate_gene_name_columns(adata)
            print(f"New shape: {adata.n_obs} cells x {adata.n_vars} genes")

        if args.add_qc:
            print("\nAdding QC annotations...")
            adata = add_cell_annotations(adata, var_index=args.var_index)
            print("Added: total_counts, log10_total_counts, n_genes_detected,")
            print("       mito_frac, ribo_frac, compl, n_cells")

        if args.output:
            print(f"\nSaving: {args.output}")
            adata.write_h5ad(args.output)
            print("Done!")
    else:
        parser.print_help()
