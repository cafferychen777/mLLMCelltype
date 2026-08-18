#!/usr/bin/env python3
"""
Clean and Process Tabula Sapiens Data for Reference Construction

This script processes the Tabula Sapiens dataset to prepare it for use as
a reference in cell type annotation pipelines (e.g., popV, scArches).

Reference: Adapted from YosefLab/popv-reproducibility
https://github.com/YosefLab/popv-reproducibility

Tabula Sapiens: https://tabula-sapiens-portal.ds.czbiohub.org/

Usage:
    python clean_tabula_sapiens.py --input tabula_sapiens.h5ad --output cleaned_ts.h5ad
"""

import argparse
import os

import scanpy as sc


def load_cell_ontology(obo_file=None):
    """
    Load Cell Ontology and create name-to-ID mappings.

    Parameters
    ----------
    obo_file : str, optional
        Path to cl.obo file. If None, will try to download.

    Returns
    -------
    name2id : dict
        Cell type name to ontology ID mapping
    id2name : dict
        Ontology ID to cell type name mapping
    """
    try:
        import obonet
    except ImportError:
        print("Warning: obonet not installed. Skipping ontology mapping.")
        return {}, {}

    if obo_file is None or not os.path.exists(obo_file):
        print("Ontology file not found. Skipping ontology mapping.")
        return {}, {}

    print(f"Loading Cell Ontology from: {obo_file}")
    with open(obo_file, "r") as f:
        co = obonet.read_obo(f)

    id2name = {
        id_: data.get("name") for id_, data in co.nodes(data=True) if "CL:" in id_
    }
    id2name = {k: v for k, v in id2name.items() if v is not None}
    name2id = {v: k for k, v in id2name.items()}

    print(f"Loaded {len(name2id)} cell type mappings")
    return name2id, id2name


def clean_tabula_sapiens(
    input_file, output_file, min_cells_per_type=10, obo_file=None, cols_to_keep=None
):
    """
    Clean and process Tabula Sapiens data.

    Parameters
    ----------
    input_file : str
        Path to input Tabula Sapiens h5ad file
    output_file : str
        Path to save cleaned data
    min_cells_per_type : int
        Minimum cells per cell type-tissue combination (default: 10)
    obo_file : str, optional
        Path to Cell Ontology OBO file for validation
    cols_to_keep : list, optional
        Columns to keep in obs. If None, uses default set.
    """
    print(f"Loading data from: {input_file}")
    adata = sc.read_h5ad(input_file)
    print(f"Original shape: {adata.n_obs} cells x {adata.n_vars} genes")

    # Use raw counts if available
    if adata.raw is not None:
        print("Using raw counts from adata.raw")
        adata.X = adata.raw.X

    # Default columns to keep
    if cols_to_keep is None:
        cols_to_keep = [
            "donor_id",
            "donor",
            "tissue_in_publication",
            "tissue",
            "free_annotation",
            "cell_type",
            "compartment",
            "assay",
            "method",
            "cell_type_ontology_term_id",
            "cell_ontology_class",
            "sex",
            "age",
        ]

    # Keep only existing columns
    existing_cols = [c for c in cols_to_keep if c in adata.obs.columns]
    cols_to_drop = [c for c in adata.obs.columns if c not in existing_cols]

    if cols_to_drop:
        print(f"Dropping {len(cols_to_drop)} columns")
        adata.obs = adata.obs[existing_cols]

    print(f"Keeping columns: {existing_cols}")

    # Identify cell type and tissue columns
    celltype_col = None
    for col in ["cell_ontology_class", "cell_type", "free_annotation"]:
        if col in adata.obs.columns:
            celltype_col = col
            break

    tissue_col = None
    for col in ["tissue_in_publication", "tissue"]:
        if col in adata.obs.columns:
            tissue_col = col
            break

    if celltype_col is None or tissue_col is None:
        print("Warning: Could not identify cell type or tissue columns")
        print(f"Available columns: {list(adata.obs.columns)}")
    else:
        # Create combined cell type-tissue label
        print(f"Using '{celltype_col}' for cell types and '{tissue_col}' for tissues")

        adata.obs["cell_ontology_class_tissue"] = [
            f"{t}_{c}" for t, c in zip(adata.obs[tissue_col], adata.obs[celltype_col])
        ]

        # Count cells per cell type-tissue combination
        counts = adata.obs["cell_ontology_class_tissue"].value_counts()
        print(f"\nCell type-tissue combinations: {len(counts)}")
        print(f"Min cells: {counts.min()}, Max cells: {counts.max()}")

        # Filter by minimum cell count
        valid_combinations = counts[counts >= min_cells_per_type].index
        n_before = adata.n_obs
        adata = adata[
            adata.obs["cell_ontology_class_tissue"].isin(valid_combinations)
        ].copy()
        n_after = adata.n_obs

        print(f"\nFiltering (min {min_cells_per_type} cells per type-tissue):")
        print(f"  Before: {n_before} cells")
        print(f"  After: {n_after} cells")
        print(
            f"  Removed: {n_before - n_after} cells ({100 * (n_before - n_after) / n_before:.1f}%)"
        )
        print(f"  Kept combinations: {len(valid_combinations)}")

    # Validate against Cell Ontology if available
    if obo_file:
        name2id, _id2name = load_cell_ontology(obo_file)
        if name2id and celltype_col:
            cell_types = adata.obs[celltype_col].unique()
            matched = [ct for ct in cell_types if ct in name2id]
            unmatched = [ct for ct in cell_types if ct not in name2id]
            print("\nOntology validation:")
            print(f"  Matched: {len(matched)} / {len(cell_types)} cell types")
            if unmatched:
                print(f"  Unmatched: {unmatched[:5]}...")

    # Summary statistics
    print("\n=== Final Dataset Summary ===")
    print(f"Shape: {adata.n_obs} cells x {adata.n_vars} genes")

    if celltype_col in adata.obs.columns:
        print(f"\nCell types ({len(adata.obs[celltype_col].unique())}):")
        print(adata.obs[celltype_col].value_counts().head(10))

    if tissue_col in adata.obs.columns:
        print(f"\nTissues ({len(adata.obs[tissue_col].unique())}):")
        print(adata.obs[tissue_col].value_counts())

    # Save cleaned data
    print(f"\nSaving to: {output_file}")
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    adata.write_h5ad(output_file)
    print("Done!")

    return adata


def main():
    parser = argparse.ArgumentParser(
        description="Clean and process Tabula Sapiens data for reference construction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python clean_tabula_sapiens.py --input tabula_sapiens.h5ad --output cleaned_ts.h5ad

    # With ontology validation
    python clean_tabula_sapiens.py --input ts.h5ad --output cleaned.h5ad --obo cl.obo

    # Custom minimum cell count
    python clean_tabula_sapiens.py --input ts.h5ad --output cleaned.h5ad --min-cells 50

Data source:
    Tabula Sapiens: https://figshare.com/articles/dataset/27921984
        """,
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Path to input Tabula Sapiens h5ad file"
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Path to save cleaned data"
    )
    parser.add_argument(
        "--min-cells",
        type=int,
        default=10,
        help="Minimum cells per cell type-tissue combination (default: 10)",
    )
    parser.add_argument(
        "--obo",
        default=None,
        help="Path to Cell Ontology OBO file for validation (optional)",
    )

    args = parser.parse_args()

    clean_tabula_sapiens(
        input_file=args.input,
        output_file=args.output,
        min_cells_per_type=args.min_cells,
        obo_file=args.obo,
    )


if __name__ == "__main__":
    main()
