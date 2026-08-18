#!/usr/bin/env python3
"""
Preprocess MCA (Mouse Cell Atlas) h5ad + cellinfo for SingleR/scType.

The MCA h5ad file has no obs columns. Cell type labels are derived from
the louvain cluster assignments in the cellinfo CSV, mapped to cell type
names via the marker stats file.

Usage:
    python prepare_mca_data.py \
        --h5ad /path/to/MCA_BatchRemoved_Merge_dge.h5ad \
        --cellinfo /path/to/MCA_BatchRemoved_Merge_dge_cellinfo.csv \
        --marker_stats /path/to/MCA_markers_L1_stats.csv \
        --output_dir /path/to/prepared \
        [--max_cells 100000]
"""

import argparse
import logging
import os
import sys
from collections import Counter

import anndata as ad
import numpy as np
import pandas as pd
from scipy import io as sio
from scipy.sparse import csc_matrix, issparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Prepare MCA data for SingleR/scType")
    parser.add_argument("--h5ad", required=True, help="Path to MCA h5ad file")
    parser.add_argument("--cellinfo", required=True, help="Path to MCA cellinfo CSV")
    parser.add_argument("--marker_stats", required=True, help="Path to MCA marker stats CSV")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--max_cells", type=int, default=100000, help="Max cells to keep")
    args = parser.parse_args()

    ds_dir = os.path.join(args.output_dir, "MCA")
    if os.path.exists(os.path.join(ds_dir, "expression.mtx")):
        log.info("Already processed, skipping")
        return

    os.makedirs(ds_dir, exist_ok=True)

    # Read marker stats: cluster_id -> cell_type
    log.info("Reading marker stats...")
    marker_stats = pd.read_csv(args.marker_stats)
    cluster_to_celltype = dict(zip(marker_stats["cluster_id"], marker_stats["cell_type"]))
    log.info(f"  {len(cluster_to_celltype)} cluster->celltype mappings")

    # Read cellinfo: index, louvain, tissue, stage
    log.info("Reading cellinfo...")
    cellinfo = pd.read_csv(args.cellinfo, index_col=0)
    log.info(f"  {len(cellinfo)} cells in cellinfo")

    # Map louvain to cell type
    cellinfo["cell_type"] = cellinfo["louvain"].map(cluster_to_celltype)
    n_mapped = cellinfo["cell_type"].notna().sum()
    log.info(f"  Mapped {n_mapped}/{len(cellinfo)} cells to cell types")

    # Keep only mapped cells
    cellinfo = cellinfo[cellinfo["cell_type"].notna()].copy()

    # Read h5ad
    log.info(f"Reading h5ad: {args.h5ad}")
    adata = ad.read_h5ad(args.h5ad)
    log.info(f"  Shape: {adata.shape}")

    # Match cells between h5ad and cellinfo
    common_cells = list(set(adata.obs_names) & set(cellinfo["index"]))
    log.info(f"  Common cells: {len(common_cells)}")

    if len(common_cells) == 0:
        log.error("No common cells found! Check barcode formats.")
        log.info(f"  h5ad obs_names[:3]: {list(adata.obs_names[:3])}")
        log.info(f"  cellinfo index[:3]: {list(cellinfo['index'][:3])}")
        return

    # Subset adata to common cells
    adata = adata[common_cells].copy()

    # Add cell type labels
    cellinfo_indexed = cellinfo.set_index("index")
    adata.obs["cell_type"] = cellinfo_indexed.loc[common_cells, "cell_type"].values

    # Subsample if needed
    if adata.shape[0] > args.max_cells:
        log.info(f"  Subsampling from {adata.shape[0]} to {args.max_cells}...")
        labels = adata.obs["cell_type"].values
        unique_labels = np.unique(labels)
        indices = []
        rng = np.random.RandomState(42)

        for label in unique_labels:
            label_idx = np.where(labels == label)[0]
            n_sample = max(1, int(len(label_idx) * args.max_cells / adata.shape[0]))
            n_sample = min(n_sample, len(label_idx))
            sampled = rng.choice(label_idx, n_sample, replace=False)
            indices.extend(sampled)

        indices = sorted(indices[:args.max_cells])
        adata = adata[indices].copy()
        log.info(f"  Subsampled to {adata.shape[0]} cells")

    # Get expression matrix (genes x cells)
    X = adata.X
    if issparse(X):
        X = csc_matrix(X.T)
    else:
        X = csc_matrix(np.array(X).T)

    # Save
    log.info(f"Saving: {X.shape[0]} genes x {X.shape[1]} cells")
    sio.mmwrite(os.path.join(ds_dir, "expression.mtx"), X)

    pd.DataFrame({"gene": adata.var_names.tolist()}).to_csv(
        os.path.join(ds_dir, "genes.csv"), index=False
    )

    pd.DataFrame({"barcode": adata.obs_names.tolist()}).to_csv(
        os.path.join(ds_dir, "barcodes.csv"), index=False
    )

    metadata = pd.DataFrame({
        "barcode": adata.obs_names.tolist(),
        "label": adata.obs["cell_type"].values,
    })
    metadata.to_csv(os.path.join(ds_dir, "metadata.csv"), index=False)

    with open(os.path.join(ds_dir, "config.txt"), "w") as f:
        f.write("name=MCA\n")
        f.write("tissue=Immune system\n")
        f.write("organism=mouse\n")
        f.write(f"n_cells={adata.shape[0]}\n")
        f.write(f"n_genes={adata.shape[1]}\n")
        f.write(f"n_celltypes={adata.obs['cell_type'].nunique()}\n")

    ct_counts = adata.obs["cell_type"].value_counts()
    ct_counts.to_csv(os.path.join(ds_dir, "celltype_counts.csv"))

    log.info(f"Done: {adata.shape[0]} cells, {adata.obs['cell_type'].nunique()} cell types")


if __name__ == "__main__":
    main()
