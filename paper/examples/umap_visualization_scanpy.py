#!/usr/bin/env python3
"""
Scanpy UMAP Visualization Tutorial Script

This script demonstrates various UMAP visualization techniques using Scanpy,
including cell type annotation display, marker gene expression, and customization options.

Reference: Scanpy documentation https://scanpy.readthedocs.io/
Tutorial: https://scanpy.readthedocs.io/en/stable/tutorials/plotting/core.html

Usage:
    python umap_visualization_scanpy.py [--input /path/to/adata.h5ad] [--output ./figures]
"""

import argparse
import os

import matplotlib.pyplot as plt
import scanpy as sc

# =============================================================================
# Configuration
# =============================================================================
# Set scanpy settings for better figures
sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.settings.set_figure_params(dpi=100, facecolor="white", frameon=False)


def load_or_create_example_data(input_path=None):
    """Load data from file or create example PBMC dataset."""

    if input_path and os.path.exists(input_path):
        print(f"Loading data from: {input_path}")
        adata = sc.read_h5ad(input_path)
    else:
        print("Loading example PBMC3k dataset...")
        # Download and preprocess PBMC3k dataset
        adata = sc.datasets.pbmc3k_processed()

    print(f"Data shape: {adata.n_obs} cells x {adata.n_vars} genes")
    return adata


def basic_umap(adata, output_dir):
    """Basic UMAP visualization."""
    print("\n=== Basic UMAP ===")

    # Simple UMAP colored by cluster
    sc.pl.umap(
        adata,
        color="louvain",
        title="Basic UMAP - Clusters",
        save="_basic_clusters.png",
    )

    # UMAP colored by gene expression
    sc.pl.umap(
        adata,
        color="CST3",  # Example marker gene
        title="Gene Expression: CST3",
        save="_gene_expression.png",
    )


def customized_cluster_umap(adata, output_dir):
    """UMAP with customized cluster visualization."""
    print("\n=== Customized Cluster UMAP ===")

    # Clusters with outline and legend on data
    sc.pl.umap(
        adata,
        color="louvain",
        add_outline=True,
        legend_loc="on data",
        legend_fontsize=12,
        legend_fontoutline=2,
        frameon=False,
        title="Clusters with Outline",
        palette="Set1",
        save="_clusters_outlined.png",
    )


def multi_panel_umap(adata, output_dir):
    """Multi-panel UMAP showing multiple variables."""
    print("\n=== Multi-panel UMAP ===")

    # Multiple marker genes in one figure
    marker_genes = ["CD3D", "CD79A", "CST3", "NKG7"]
    available_genes = [g for g in marker_genes if g in adata.var_names]

    if available_genes:
        sc.pl.umap(
            adata,
            color=available_genes,
            ncols=2,
            s=30,  # dot size
            frameon=False,
            vmax="p99",  # cap at 99th percentile for better visualization
            save="_multi_markers.png",
        )


def cell_type_annotation_umap(adata, output_dir):
    """UMAP for displaying cell type annotations."""
    print("\n=== Cell Type Annotation UMAP ===")

    # Check for cell type column
    celltype_col = None
    for col in ["cell_type", "celltype", "cell_ontology_class", "louvain"]:
        if col in adata.obs.columns:
            celltype_col = col
            break

    if celltype_col:
        # Publication-quality cell type UMAP
        sc.pl.umap(
            adata,
            color=celltype_col,
            legend_loc="on data",
            frameon=False,
            legend_fontsize=10,
            legend_fontoutline=2,
            title="Cell Type Annotations",
            save="_cell_types.png",
        )

        # Alternative with legend on side
        sc.pl.umap(
            adata,
            color=celltype_col,
            legend_loc="right margin",
            frameon=False,
            title="Cell Types (legend on side)",
            save="_cell_types_legend_side.png",
        )


def expression_gradient_umap(adata, output_dir):
    """UMAP with expression gradient visualization."""
    print("\n=== Expression Gradient UMAP ===")

    # Using centered colormap for expression
    if "CST3" in adata.var_names:
        sc.pl.umap(
            adata,
            color="CST3",
            vcenter=0,  # center at 0 for diverging colormap
            cmap="RdBu_r",
            title="CST3 (centered colormap)",
            save="_expression_centered.png",
        )


def custom_palette_umap(adata, output_dir):
    """UMAP with custom color palette."""
    print("\n=== Custom Palette UMAP ===")

    # Define custom colors for categories
    if "louvain" in adata.obs.columns:
        # Using a qualitative palette
        sc.pl.umap(
            adata,
            color="louvain",
            palette="tab20",  # good for many categories
            title="Custom Palette (tab20)",
            save="_custom_palette.png",
        )


def high_quality_export(adata, output_dir):
    """Export high-quality figures for publication."""
    print("\n=== High Quality Export ===")

    # Create figure with specific size
    fig, ax = plt.subplots(figsize=(8, 8))

    sc.pl.umap(
        adata,
        color="louvain",
        ax=ax,
        show=False,
        legend_loc="on data",
        legend_fontsize=10,
        legend_fontoutline=2,
        frameon=False,
        title="",
    )

    # Save as high-resolution PNG and PDF
    fig.savefig(
        os.path.join(output_dir, "umap_highres.png"),
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        os.path.join(output_dir, "umap_highres.pdf"),
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    print(f"High-resolution figures saved to: {output_dir}")


def qc_metrics_umap(adata, output_dir):
    """UMAP showing QC metrics."""
    print("\n=== QC Metrics UMAP ===")

    # Check for QC columns
    qc_cols = []
    for col in [
        "n_genes",
        "n_genes_by_counts",
        "total_counts",
        "pct_counts_mt",
        "percent_mito",
    ]:
        if col in adata.obs.columns:
            qc_cols.append(col)

    if qc_cols:
        sc.pl.umap(
            adata,
            color=qc_cols[:3],  # max 3 columns
            ncols=3,
            s=20,
            frameon=False,
            wspace=0.4,
            save="_qc_metrics.png",
        )


def main(input_path=None, output_dir="./figures"):
    """Main function to run all UMAP visualizations."""

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Set scanpy figure directory
    sc.settings.figdir = output_dir

    # Load data
    adata = load_or_create_example_data(input_path)

    # Ensure UMAP is computed
    if "X_umap" not in adata.obsm:
        print("Computing UMAP embedding...")
        if "neighbors" not in adata.uns:
            sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
        sc.tl.umap(adata)

    # Run all visualization examples
    basic_umap(adata, output_dir)
    customized_cluster_umap(adata, output_dir)
    multi_panel_umap(adata, output_dir)
    cell_type_annotation_umap(adata, output_dir)
    expression_gradient_umap(adata, output_dir)
    custom_palette_umap(adata, output_dir)
    qc_metrics_umap(adata, output_dir)
    high_quality_export(adata, output_dir)

    print(f"\n=== All figures saved to: {output_dir} ===")
    print("\nGenerated files:")
    for f in sorted(os.listdir(output_dir)):
        if f.endswith((".png", ".pdf")):
            print(f"  - {f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scanpy UMAP Visualization Tutorial",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Use built-in PBMC3k dataset
    python umap_visualization_scanpy.py

    # Use your own data
    python umap_visualization_scanpy.py --input my_data.h5ad --output ./my_figures

Reference:
    Scanpy plotting tutorial: https://scanpy.readthedocs.io/en/stable/tutorials/plotting/core.html
        """,
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to input AnnData file (.h5ad). If not provided, uses PBMC3k example dataset.",
    )
    parser.add_argument(
        "--output",
        default="./figures",
        help="Output directory for figures (default: ./figures)",
    )
    args = parser.parse_args()

    main(args.input, args.output)
