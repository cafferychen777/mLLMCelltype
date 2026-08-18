#!/usr/bin/env python3
"""
Pre-process h5ad files into R-friendly formats for SingleR/scType.
Converts each h5ad into: expression.mtx, genes.csv, barcodes.csv, metadata.csv

Usage:
    python prepare_data.py --data_dir /path/to/h5ad --output_dir /path/to/prepared
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

# Dataset configurations: filename -> (display_name, label_column, sctype_tissue)
DATASETS = {
    # Tabula Sapiens
    "TS_Bladder_filtered.h5ad": ("Bladder(TS)", "cell_ontology_class", "Immune system"),
    "TS_Blood_filtered.h5ad": ("Blood(TS)", "cell_ontology_class", "Immune system"),
    "TS_Bone_Marrow_filtered.h5ad": ("Bone Marrow(TS)", "cell_ontology_class", "Immune system"),
    "TS_Eye_filtered.h5ad": ("Eye(TS)", "cell_ontology_class", "Eye"),
    "TS_Fat_filtered.h5ad": ("Fat(TS)", "cell_ontology_class", "Immune system"),
    "TS_Heart_filtered.h5ad": ("Heart(TS)", "cell_ontology_class", "Heart"),
    "TS_Large_Intestine_filtered.h5ad": ("Large Intestine(TS)", "cell_ontology_class", "Intestine"),
    "TS_Liver_filtered.h5ad": ("Liver(TS)", "cell_ontology_class", "Liver"),
    "TS_Lung_filtered.h5ad": ("Lung(TS)", "cell_ontology_class", "Lung"),
    "TS_Lymph_Node_filtered.h5ad": ("Lymph Node(TS)", "cell_ontology_class", "Immune system"),
    "TS_Mammary_filtered.h5ad": ("Mammary(TS)", "cell_ontology_class", "Immune system"),
    "TS_Muscle_filtered.h5ad": ("Muscle(TS)", "cell_ontology_class", "Immune system"),
    "TS_Pancreas_filtered.h5ad": ("Pancreas(TS)", "cell_ontology_class", "Pancreas"),
    "TS_Prostate_filtered.h5ad": ("Prostate(TS)", "cell_ontology_class", "Immune system"),
    "TS_Salivary_Gland_filtered.h5ad": ("Salivary Gland(TS)", "cell_ontology_class", "Immune system"),
    "TS_Skin_filtered.h5ad": ("Skin(TS)", "cell_ontology_class", "Immune system"),
    "TS_Small_Intestine_filtered.h5ad": ("Small Intestine(TS)", "cell_ontology_class", "Intestine"),
    "TS_Spleen_filtered.h5ad": ("Spleen(TS)", "cell_ontology_class", "Spleen"),
    "TS_Thymus_filtered.h5ad": ("Thymus(TS)", "cell_ontology_class", "Thymus"),
    "TS_Tongue_filtered.h5ad": ("Tongue(TS)", "cell_ontology_class", "Immune system"),
    "TS_Trachea_filtered.h5ad": ("Trachea(TS)", "cell_ontology_class", "Lung"),
    "TS_Uterus_filtered.h5ad": ("Uterus(TS)", "cell_ontology_class", "Immune system"),
    "TS_Vasculature_filtered.h5ad": ("Vasculature(TS)", "cell_ontology_class", "Immune system"),
    # Additional datasets
    "BCL.h5ad": ("BCL", "refined_clusters", "Immune system"),
    "HGA.h5ad": ("HGA", "cell_type", "Brain"),
    "HGA_smartseq.h5ad": ("HGA_smartseq", "cell_type", "Brain"),
    "HLCA_Core.h5ad": ("HLCA", "cell_type", "Lung"),
    "InfantGut.h5ad": ("InfantGut", "cell_name", "Intestine"),
    "LCA.h5ad": ("LCA", "cell_type", "Lung"),
    "M1C.h5ad": ("M1C", "cell_type", "Brain"),
    "MTG.h5ad": ("MTG", "cell_type", "Brain"),
    "MTG_sampled.h5ad": ("MTG_sampled", "cell_type", "Brain"),
    "Schiller_2020.h5ad": ("Schiller_2020", "cell_type", "Lung"),
    "Sun_2020.h5ad": ("Sun_2020", "cell_type", "Lung"),
    "Thymus.h5ad": ("Thymus_dev", "cell_type", "Thymus"),
}


def process_dataset(h5ad_path, output_dir, name, label_col, tissue):
    """Convert a single h5ad file to R-friendly format."""
    ds_dir = os.path.join(output_dir, name.replace(" ", "_").replace("(", "").replace(")", ""))
    if os.path.exists(os.path.join(ds_dir, "expression.mtx")):
        log.info(f"  Skipping {name}: already processed")
        return True

    os.makedirs(ds_dir, exist_ok=True)

    try:
        log.info(f"  Loading {h5ad_path}...")
        adata = ad.read_h5ad(h5ad_path)
        log.info(f"  Shape: {adata.shape[0]} cells x {adata.shape[1]} genes")

        # Check label column exists
        if label_col not in adata.obs.columns:
            available = adata.obs.columns.tolist()
            log.error(f"  Label column '{label_col}' not found. Available: {available}")
            return False

        # Get expression matrix (genes x cells for R)
        X = adata.X
        if issparse(X):
            X = csc_matrix(X.T)  # transpose: genes x cells, CSC format
        else:
            X = csc_matrix(np.array(X).T)

        # Save as Matrix Market format
        sio.mmwrite(os.path.join(ds_dir, "expression.mtx"), X)

        # Save gene names
        pd.DataFrame({"gene": adata.var_names.tolist()}).to_csv(
            os.path.join(ds_dir, "genes.csv"), index=False
        )

        # Save cell barcodes
        pd.DataFrame({"barcode": adata.obs_names.tolist()}).to_csv(
            os.path.join(ds_dir, "barcodes.csv"), index=False
        )

        # Save metadata
        meta = pd.DataFrame({
            "barcode": adata.obs_names.tolist(),
            "label": adata.obs[label_col].astype(str).tolist(),
        })
        meta.to_csv(os.path.join(ds_dir, "metadata.csv"), index=False)

        # Save config
        with open(os.path.join(ds_dir, "config.txt"), "w") as f:
            f.write(f"name={name}\n")
            f.write(f"tissue={tissue}\n")
            f.write(f"n_cells={adata.shape[0]}\n")
            f.write(f"n_genes={adata.shape[1]}\n")
            f.write(f"label_col={label_col}\n")
            f.write(f"n_clusters={adata.obs[label_col].nunique()}\n")

        log.info(f"  Saved to {ds_dir} ({adata.shape[0]} cells, {adata.obs[label_col].nunique()} clusters)")
        del adata
        return True

    except Exception as e:
        log.error(f"  Failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Prepare h5ad files for SingleR/scType")
    parser.add_argument("--data_dir", required=True, help="Directory containing h5ad files")
    parser.add_argument("--output_dir", required=True, help="Output directory for prepared data")
    parser.add_argument("--dataset", default="all", help="Specific dataset filename or 'all'")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.dataset == "all":
        datasets_to_process = DATASETS
    else:
        if args.dataset in DATASETS:
            datasets_to_process = {args.dataset: DATASETS[args.dataset]}
        else:
            log.error(f"Unknown dataset: {args.dataset}")
            log.info(f"Available: {list(DATASETS.keys())}")
            sys.exit(1)

    success = 0
    failed = 0
    for filename, (name, label_col, tissue) in datasets_to_process.items():
        h5ad_path = os.path.join(args.data_dir, filename)
        if not os.path.exists(h5ad_path):
            log.warning(f"SKIP {name}: file not found ({h5ad_path})")
            continue

        log.info(f"===== Processing: {name} =====")
        if process_dataset(h5ad_path, args.output_dir, name, label_col, tissue):
            success += 1
        else:
            failed += 1

    log.info(f"\nDone. Success: {success}, Failed: {failed}")


if __name__ == "__main__":
    main()
