#!/usr/bin/env python3
"""
Extract marker genes from Mouse Cell Atlas (MCA) supplementary data.

This script processes the MCA supplementary Excel file (mmc4.xlsx) to extract
cell type marker genes with statistical information.

Reference: Han et al. (2018) Cell

Usage:
    python extract_mca_markers.py --input /path/to/mmc4.xlsx --output ./output
"""

import argparse
import os

import pandas as pd


def extract_mca_markers(excel_path, output_dir):
    """Extract marker genes from MCA supplementary data."""

    os.makedirs(output_dir, exist_ok=True)

    # First read the 98 Clusters sheet to get cluster annotations
    clusters_df = pd.read_excel(excel_path, sheet_name="98 Clusters")
    # Convert all columns to string type to avoid datetime issues
    clusters_df = clusters_df.astype(str)

    print("\nReading cluster annotations from 98 Clusters sheet...")
    print(f"Found {len(clusters_df)} clusters")
    print("\nSample of cluster annotations:")
    print(clusters_df[["Cluster", "Cell Type"]].head())

    # Create a mapping from cluster to cell type
    cluster_to_celltype = dict(zip(clusters_df["Cluster"], clusters_df["Cell Type"]))

    # Now read the MCA Markers sheet to get marker genes
    markers_df = pd.read_excel(excel_path, sheet_name="MCA Markers")
    # Convert specific columns to string type
    markers_df["cluster"] = markers_df["cluster"].astype(str)
    markers_df["gene"] = markers_df["gene"].astype(str)

    print("\nReading marker genes from MCA Markers sheet...")
    print(f"Initial shape: {markers_df.shape}")

    # Clean up the data
    markers_df = markers_df.dropna(subset=["cluster", "gene", "avg_logFC", "p_val"])
    print(f"Shape after cleaning: {markers_df.shape}")

    # Process markers for each cluster
    result_data = []
    for cluster in sorted(markers_df["cluster"].unique()):
        group = markers_df[markers_df["cluster"] == cluster]

        # Sort by absolute value of avg_logFC
        group = group.assign(abs_logFC=abs(group["avg_logFC"]))
        group = group.sort_values("abs_logFC", ascending=False)

        # Get significant genes
        significant_genes = group[
            (group["p_val"] < 0.05)  # statistically significant
            & (group["avg_logFC"].abs() > 1.0)  # biologically significant
        ]

        if len(significant_genes) > 0:
            # Get cell type annotation for this cluster
            cell_type = cluster_to_celltype.get(cluster, f"Cluster_{cluster}")

            # Only keep top 10 genes
            significant_genes = significant_genes.head(10)

            # Get gene information and ensure all values are strings
            genes = significant_genes["gene"].astype(str).tolist()
            logfc_values = significant_genes["avg_logFC"].astype(float).tolist()
            pvals = significant_genes["p_val"].astype(float).tolist()

            # Combine the information
            gene_info = list(zip(genes, logfc_values, pvals))

            result_data.append(
                {
                    "cluster_id": cluster,
                    "cell_type": str(cell_type),
                    "gene": ",".join(genes),
                    "num_markers": len(genes),
                    "genes_with_stats": "; ".join(
                        [f"{g}(logFC={fc:.2f},p={p:.2e})" for g, fc, p in gene_info]
                    ),
                }
            )

            print(f"\nCluster {cluster} ({cell_type}):")
            print(f"Number of significant markers: {len(genes)}")
            print("Markers with stats:")
            print(
                "\n".join([f"{g}(logFC={fc:.2f},p={p:.2e})" for g, fc, p in gene_info])
            )

    # Create final dataframe
    final_df = pd.DataFrame(result_data)

    # Sort by cell type name
    final_df = final_df.sort_values("cell_type")

    print("\nSummary statistics:")
    print(f"Total clusters with significant markers: {len(final_df)}")
    print("\nNumber of markers per cluster:")
    print(final_df["num_markers"].describe())

    # Save to CSV
    output_path = os.path.join(output_dir, "MCA_markers_L1.csv")

    # Write CSV without cluster IDs
    with open(output_path, "w") as f:
        f.write("cell_type,gene\n")
        f.writelines(
            f"{row['cell_type']!s},{row['gene']!s}\n" for _, row in final_df.iterrows()
        )

    print(f"\nProcessed markers saved to: {output_path}")

    # Save detailed statistics to a separate file
    stats_path = os.path.join(output_dir, "MCA_markers_L1_stats.csv")
    final_df.to_csv(stats_path, index=False)
    print(f"Detailed statistics saved to: {stats_path}")

    return final_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract marker genes from Mouse Cell Atlas data"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the MCA supplementary Excel file (mmc4.xlsx)",
    )
    parser.add_argument(
        "--output",
        default="./mca_markers_output",
        help="Output directory for extracted markers (default: ./mca_markers_output)",
    )
    args = parser.parse_args()

    extract_mca_markers(args.input, args.output)
