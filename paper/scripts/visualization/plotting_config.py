#!/usr/bin/env python3
"""
Scientific Visualization Configuration for Single-Cell Analysis

This module provides publication-quality matplotlib/seaborn settings optimized
for single-cell RNA-seq visualizations with scanpy.

Reference: Adapted from YosefLab/popv-reproducibility
https://github.com/YosefLab/popv-reproducibility

Usage:
    # Import at the beginning of your script
    import plotting_config  # This will apply all settings

    # Or import specific settings
    from plotting_config import setup_plotting, DPI
"""

import seaborn as sns
from matplotlib import pyplot as plt

# Try to import scanpy, but don't fail if not available
try:
    import scanpy as sc

    HAS_SCANPY = True
except ImportError:
    HAS_SCANPY = False

# Default DPI for saved figures
DPI = 300


def setup_plotting(dpi=300, use_scanpy=True):
    """
    Configure matplotlib/seaborn for publication-quality figures.

    Parameters
    ----------
    dpi : int
        Resolution for saved figures (default: 300)
    use_scanpy : bool
        Whether to configure scanpy settings (default: True)
    """
    global DPI
    DPI = dpi

    # Reset seaborn to original matplotlib defaults
    sns.reset_orig()

    # Configure scanpy if available
    if HAS_SCANPY and use_scanpy:
        sc.settings._vector_friendly = True
        sc.settings.set_figure_params(
            dpi_save=dpi, frameon=False, format="pdf", transparent=True
        )

    # ==========================================================================
    # Font Configuration
    # ==========================================================================
    plt.rcParams["svg.fonttype"] = "none"  # Keep text as text in SVG
    plt.rcParams["pdf.fonttype"] = 42  # TrueType fonts in PDF
    plt.rcParams["savefig.transparent"] = True

    plt.rcParams["font.size"] = 11
    plt.rcParams["font.sans-serif"] = [
        "Helvetica",
        "Arial",
        "Computer Modern Sans Serif",
        "DejaVU Sans",
    ]
    plt.rcParams["font.weight"] = 500

    # ==========================================================================
    # Axes Configuration
    # ==========================================================================
    plt.rcParams["axes.titlesize"] = 15
    plt.rcParams["axes.titleweight"] = 500
    plt.rcParams["axes.titlepad"] = 8.0
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["axes.labelweight"] = 500
    plt.rcParams["axes.linewidth"] = 1.2
    plt.rcParams["axes.labelpad"] = 6.0

    # Remove top and right spines for cleaner look
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False

    # ==========================================================================
    # Tick Configuration
    # ==========================================================================
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["xtick.minor.size"] = 1.375
    plt.rcParams["xtick.major.size"] = 2.75
    plt.rcParams["xtick.major.pad"] = 2
    plt.rcParams["xtick.minor.pad"] = 2

    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["ytick.minor.size"] = 1.375
    plt.rcParams["ytick.major.size"] = 2.75
    plt.rcParams["ytick.major.pad"] = 2
    plt.rcParams["ytick.minor.pad"] = 2

    # ==========================================================================
    # Legend Configuration
    # ==========================================================================
    plt.rcParams["legend.fontsize"] = 12
    plt.rcParams["legend.handlelength"] = 1.4
    plt.rcParams["legend.numpoints"] = 1
    plt.rcParams["legend.scatterpoints"] = 3

    # ==========================================================================
    # Line Configuration
    # ==========================================================================
    plt.rcParams["lines.linewidth"] = 1.7


def setup_dark_background():
    """Configure settings for dark background plots."""
    plt.style.use("dark_background")
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False


def reset_plotting():
    """Reset matplotlib to default settings."""
    plt.rcdefaults()
    sns.reset_orig()


# Apply settings on import
setup_plotting()

if __name__ == "__main__":
    # Demo plot
    import numpy as np

    print("Generating demo plot with publication settings...")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Demo scatter plot
    np.random.seed(42)
    x = np.random.randn(100)
    y = np.random.randn(100)
    colors = np.random.rand(100)

    axes[0].scatter(x, y, c=colors, cmap="viridis", alpha=0.7)
    axes[0].set_xlabel("UMAP 1")
    axes[0].set_ylabel("UMAP 2")
    axes[0].set_title("Scatter Plot Demo")

    # Demo bar plot
    categories = ["A", "B", "C", "D"]
    values = [23, 45, 56, 78]
    axes[1].bar(categories, values, color="steelblue")
    axes[1].set_xlabel("Category")
    axes[1].set_ylabel("Value")
    axes[1].set_title("Bar Plot Demo")

    plt.tight_layout()
    plt.savefig("plotting_demo.png", dpi=DPI, bbox_inches="tight")
    plt.savefig("plotting_demo.pdf", bbox_inches="tight")
    print("Saved: plotting_demo.png, plotting_demo.pdf")
    plt.show()
