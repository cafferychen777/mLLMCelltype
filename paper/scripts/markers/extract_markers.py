#!/usr/bin/env python3
"""
Extract marker genes from GTEx supplementary table S3.

This script processes the GTEx supplementary table (science.abl4290_table_s3.xlsx)
to extract cell type marker genes for various tissues.

Usage:
    python extract_markers.py --input /path/to/table_s3.xlsx
"""

import argparse

import pandas as pd


def extract_markers(file_path):
    """Extract marker genes from GTEx supplementary table."""

    # Print summary information
    summary_df = pd.read_excel(file_path, sheet_name="Summary")
    print("\n=== Marker Gene Summary ===")
    print(summary_df.to_string(index=False))

    # Process each tissue
    tissues = [
        "Breast",
        "Esophagus",
        "Heart",
        "Lung",
        "Prostate",
        "Skeletal Muscle",
        "Skin",
    ]

    all_markers = {}

    for tissue in tissues:
        print(f"\n\n=== {tissue} Cell Types and Marker Genes ===")
        try:
            df = pd.read_excel(file_path, sheet_name=tissue)
            # Get unique cell types
            cell_types = df["Cell-Type"].unique()
            print("\nCell types found:")

            tissue_markers = {}
            for ct in cell_types:
                if pd.notna(ct):
                    genes = df[df["Cell-Type"] == ct]["Gene"].tolist()
                    print(f"\nCell type: {ct}")
                    print(f"Marker genes: {', '.join(genes)}")
                    tissue_markers[ct] = genes

            all_markers[tissue] = tissue_markers

        except Exception as e:  # noqa: BLE001 - continue with the remaining worksheets
            print(f"Error processing {tissue}: {e!s}")

    return all_markers


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract marker genes from GTEx supplementary table"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the GTEx supplementary table Excel file (science.abl4290_table_s3.xlsx)",
    )
    parser.add_argument(
        "--output",
        default="./markers_output",
        help="Output directory for extracted markers (default: ./markers_output)",
    )
    args = parser.parse_args()

    markers = extract_markers(args.input)

    # Optionally save to CSV files
    import os

    os.makedirs(args.output, exist_ok=True)

    for tissue, tissue_markers in markers.items():
        output_file = os.path.join(
            args.output, f"{tissue.replace(' ', '_')}_markers.csv"
        )
        with open(output_file, "w") as f:
            f.write("cell_type,gene\n")
            for cell_type, genes in tissue_markers.items():
                f.writelines(f"{cell_type},{gene}\n" for gene in genes)
        print(f"\nSaved: {output_file}")
