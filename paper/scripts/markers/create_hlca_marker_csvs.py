#!/usr/bin/env python3
"""
Create marker gene CSV files from HLCA (Human Lung Cell Atlas) supplementary data.

This script processes the HLCA supplementary Excel file to extract marker genes
for each cell type and saves them as individual CSV files.

Reference: Sikkema et al. (2023) Nature Medicine

Usage:
    python create_hlca_marker_csvs.py --input /path/to/supplementary.xlsx --output ./output
"""

import argparse
import os

import pandas as pd


def create_marker_csv(df, cell_type, output_file):
    """
    Create marker gene CSV file for a specified cell type.

    Parameters:
    df: DataFrame containing marker genes
    cell_type: Cell type name (without _marker suffix)
    output_file: Output file path
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Get marker genes for this cell type
    marker_col = f"{cell_type}_marker"
    genes = df[marker_col].dropna().tolist()

    # Create output data
    output_data = [{"cluster": cell_type, "gene": gene} for gene in genes]

    # Write to CSV file
    with open(output_file, "w") as f:
        f.write("cluster,gene\n")
        f.writelines(f"{item['cluster']},{item['gene']}\n" for item in output_data)
    print(f"Created {output_file}")


def process_hlca_markers(input_file, output_dir):
    """Process HLCA marker genes from supplementary Excel file."""

    os.makedirs(output_dir, exist_ok=True)

    # Read marker genes sheet
    df = pd.read_excel(input_file, sheet_name="6 - marker genes")

    # Get first row as column names
    header = df.iloc[0]
    df = df.iloc[1:]  # Remove first row
    df.columns = header  # Use first row as column names

    # Get all cell types
    cell_types = [
        col.replace("_marker", "")
        for col in df.columns
        if not pd.isna(col) and col.endswith("_marker")
    ]

    print(f"Found {len(cell_types)} cell types in HLCA data")

    # Create CSV file for each cell type
    for cell_type in cell_types:
        # Clean cell type name for filename
        clean_name = cell_type.replace(" ", "_").replace("(", "").replace(")", "")
        output_file = os.path.join(output_dir, f"HLCA_{clean_name}_markers.csv")
        create_marker_csv(df, cell_type, output_file)

    print(f"\nAll marker files saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create HLCA marker gene CSV files")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the HLCA supplementary Excel file (41591_2023_2327_MOESM3_ESM.xlsx)",
    )
    parser.add_argument(
        "--output",
        default="./hlca_markers_output",
        help="Output directory for marker CSV files (default: ./hlca_markers_output)",
    )
    args = parser.parse_args()

    process_hlca_markers(args.input, args.output)
