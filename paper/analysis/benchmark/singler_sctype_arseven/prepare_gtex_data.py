#!/usr/bin/env python3
"""
Pre-process GTEx snRNA-seq h5ad into per-tissue R-friendly formats for SingleR/scType.

The GTEx atlas (Eraslan et al. 2022, Science) contains 8 tissues in a single h5ad file.
This script subsets by tissue and creates the same format as prepare_data.py:
  expression.mtx, genes.csv, barcodes.csv, metadata.csv, config.txt

Usage:
    python prepare_gtex_data.py \
        --h5ad /path/to/GTEx_8_tissues_snRNAseq_atlas_071421.public_obs.h5ad \
        --output_dir /path/to/prepared \
        [--tissue breast]
"""

import argparse
import logging
import os
import sys

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

# GTEx tissue -> (output_dir_name, display_name, sctype_tissue)
# Using Granular cell type as label column for all
GTEX_TISSUES = {
    "breast": ("GTEx_breast", "GTEx Breast", "Immune system"),
    "esophagusmucosa": ("GTEx_esophagus_mucosa", "GTEx Esophagus Mucosa", "Stomach"),
    "esophagusmuscularis": ("GTEx_esophagus_muscularis", "GTEx Esophagus Muscularis", "Muscle"),
    "heart": ("GTEx_heart", "GTEx Heart", "Heart"),
    "lung": ("GTEx_lung", "GTEx Lung", "Lung"),
    "prostate": ("GTEx_prostate", "GTEx Prostate", "Immune system"),
    "skeletalmuscle": ("GTEx_skeletal_muscle", "GTEx Skeletal Muscle", "Muscle"),
    "skin": ("GTEx_skin", "GTEx Skin", "Immune system"),
}

LABEL_COL = "Granular cell type"


def process_tissue(adata_full, tissue_key, output_dir, dir_name, display_name, sctype_tissue):
    """Extract a single tissue from the GTEx atlas and save in R-friendly format."""
    ds_dir = os.path.join(output_dir, dir_name)

    if os.path.exists(os.path.join(ds_dir, "expression.mtx")):
        log.info(f"  Skipping {display_name}: already processed")
        return True

    os.makedirs(ds_dir, exist_ok=True)

    try:
        # Subset to this tissue
        mask = adata_full.obs["tissue"] == tissue_key
        n_cells = mask.sum()
        log.info(f"  Subsetting {tissue_key}: {n_cells} cells")

        if n_cells == 0:
            log.error(f"  No cells found for tissue '{tissue_key}'")
            return False

        adata = adata_full[mask].copy()

        # Check label column
        if LABEL_COL not in adata.obs.columns:
            log.error(f"  Label column '{LABEL_COL}' not found. Available: {adata.obs.columns.tolist()}")
            return False

        labels = adata.obs[LABEL_COL].astype(str)
        n_clusters = labels.nunique()
        log.info(f"  {n_clusters} cell types, {len(adata)} cells")

        # Expression matrix: genes x cells for R
        X = adata.X
        if issparse(X):
            X = csc_matrix(X.T)
        else:
            X = csc_matrix(np.array(X).T)

        sio.mmwrite(os.path.join(ds_dir, "expression.mtx"), X)

        # Gene names
        pd.DataFrame({"gene": adata.var_names.tolist()}).to_csv(
            os.path.join(ds_dir, "genes.csv"), index=False
        )

        # Cell barcodes
        pd.DataFrame({"barcode": adata.obs_names.tolist()}).to_csv(
            os.path.join(ds_dir, "barcodes.csv"), index=False
        )

        # Metadata
        meta = pd.DataFrame({
            "barcode": adata.obs_names.tolist(),
            "label": labels.tolist(),
        })
        meta.to_csv(os.path.join(ds_dir, "metadata.csv"), index=False)

        # Config
        with open(os.path.join(ds_dir, "config.txt"), "w") as f:
            f.write(f"name={display_name}\n")
            f.write(f"tissue={sctype_tissue}\n")
            f.write(f"n_cells={len(adata)}\n")
            f.write(f"n_genes={adata.shape[1]}\n")
            f.write(f"label_col={LABEL_COL}\n")
            f.write(f"n_clusters={n_clusters}\n")

        log.info(f"  Saved to {ds_dir}")

        # Print cell type distribution
        for ct, count in labels.value_counts().items():
            log.info(f"    {ct}: {count}")

        del adata
        return True

    except Exception as e:
        log.error(f"  Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Prepare GTEx h5ad for SingleR/scType")
    parser.add_argument("--h5ad", required=True, help="Path to GTEx h5ad file")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--tissue", default="all", help="Specific tissue or 'all'")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    log.info(f"Loading GTEx atlas: {args.h5ad}")
    adata = ad.read_h5ad(args.h5ad)
    log.info(f"Full atlas shape: {adata.shape}")
    log.info(f"Tissues: {sorted(adata.obs['tissue'].unique())}")

    if args.tissue == "all":
        tissues_to_process = GTEX_TISSUES
    else:
        if args.tissue in GTEX_TISSUES:
            tissues_to_process = {args.tissue: GTEX_TISSUES[args.tissue]}
        else:
            log.error(f"Unknown tissue: {args.tissue}")
            log.info(f"Available: {list(GTEX_TISSUES.keys())}")
            sys.exit(1)

    success = 0
    failed = 0
    for tissue_key, (dir_name, display_name, sctype_tissue) in tissues_to_process.items():
        log.info(f"\n===== Processing: {display_name} ({tissue_key}) =====")
        if process_tissue(adata, tissue_key, args.output_dir, dir_name, display_name, sctype_tissue):
            success += 1
        else:
            failed += 1

    log.info(f"\nDone. Success: {success}, Failed: {failed}")
    del adata


if __name__ == "__main__":
    main()
