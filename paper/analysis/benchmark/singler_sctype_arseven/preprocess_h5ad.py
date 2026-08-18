#!/usr/bin/env python3
"""Preprocess h5ad files for SingleR/scType analysis on arseven.

Reads each h5ad, extracts expression matrix + metadata, saves as:
  - {name}_counts.mtx.gz  (genes x cells sparse matrix)
  - {name}_genes.csv      (gene names)
  - {name}_barcodes.csv   (cell barcodes)
  - {name}_metadata.csv   (cell metadata including labels)

Usage:
    python preprocess_h5ad.py --data_dir /path/to/h5ad --output_dir /path/to/output
    python preprocess_h5ad.py --data_dir /path/to/h5ad --output_dir /path/to/output --dataset TS_Blood
"""

import argparse
import logging
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset configuration: h5ad filename -> (display_name, label_column, tissue)
# ---------------------------------------------------------------------------
DATASETS = {
    # Tabula Sapiens (23 datasets)
    "TS_Bladder_filtered": ("Bladder(TS)", "cell_ontology_class", "Immune system"),
    "TS_Blood_filtered": ("Blood(TS)", "cell_ontology_class", "Immune system"),
    "TS_Bone_Marrow_filtered": ("Bone Marrow(TS)", "cell_ontology_class", "Immune system"),
    "TS_Eye_filtered": ("Eye(TS)", "cell_ontology_class", "Eye"),
    "TS_Fat_filtered": ("Fat(TS)", "cell_ontology_class", "Immune system"),
    "TS_Heart_filtered": ("Heart(TS)", "cell_ontology_class", "Heart"),
    "TS_Large_Intestine_filtered": ("Large Intestine(TS)", "cell_ontology_class", "Intestine"),
    "TS_Liver_filtered": ("Liver(TS)", "cell_ontology_class", "Liver"),
    "TS_Lung_filtered": ("Lung(TS)", "cell_ontology_class", "Lung"),
    "TS_Lymph_Node_filtered": ("Lymph Node(TS)", "cell_ontology_class", "Immune system"),
    "TS_Mammary_filtered": ("Mammary(TS)", "cell_ontology_class", "Immune system"),
    "TS_Muscle_filtered": ("Muscle(TS)", "cell_ontology_class", "Muscle"),
    "TS_Pancreas_filtered": ("Pancreas(TS)", "cell_ontology_class", "Pancreas"),
    "TS_Prostate_filtered": ("Prostate(TS)", "cell_ontology_class", "Immune system"),
    "TS_Salivary_Gland_filtered": ("Salivary Gland(TS)", "cell_ontology_class", "Immune system"),
    "TS_Skin_filtered": ("Skin(TS)", "cell_ontology_class", "Immune system"),
    "TS_Small_Intestine_filtered": ("Small Intestine(TS)", "cell_ontology_class", "Intestine"),
    "TS_Spleen_filtered": ("Spleen(TS)", "cell_ontology_class", "Spleen"),
    "TS_Thymus_filtered": ("Thymus(TS)", "cell_ontology_class", "Thymus"),
    "TS_Tongue_filtered": ("Tongue(TS)", "cell_ontology_class", "Immune system"),
    "TS_Trachea_filtered": ("Trachea(TS)", "cell_ontology_class", "Lung"),
    "TS_Uterus_filtered": ("Uterus(TS)", "cell_ontology_class", "Immune system"),
    "TS_Vasculature_filtered": ("Vasculature(TS)", "cell_ontology_class", "Immune system"),
    # Other human datasets
    "BCL": ("BCL", "cell_type", "Immune system"),
    "HGA": ("HGA", "cell_type", "Brain"),
    "HGA_smartseq": ("HGA_smartseq", "cell_type", "Brain"),
    "HLCA_Core": ("HLCA", "cell_type", "Lung"),
    "InfantGut": ("InfantGut", "cell_type", "Intestine"),
    "LCA": ("LCA", "cell_type", "Lung"),
    "M1C": ("M1C", "cell_type", "Brain"),
    "MTG": ("MTG", "cell_type", "Brain"),
    "MTG_sampled": ("MTG_sampled", "cell_type", "Brain"),
    "Schiller_2020": ("Schiller_2020", "cell_type", "Lung"),
    "Sun_2020": ("Sun_2020", "cell_type", "Lung"),
    "Thymus": ("Thymus", "cell_type", "Thymus"),
}


def get_counts(adata):
    """Extract count matrix, preferring raw counts."""
    if "counts" in adata.layers:
        X = adata.layers["counts"]
    elif adata.raw is not None:
        X = adata.raw.X
    else:
        X = adata.X

    if sp.issparse(X):
        X = X.tocsc()
    else:
        X = sp.csc_matrix(X)
    return X


def preprocess_dataset(h5ad_path, ds_key, output_dir):
    """Read h5ad and export intermediate files for R."""
    display_name, label_col, tissue = DATASETS[ds_key]
    log.info(f"Processing {display_name} from {h5ad_path}")

    adata = ad.read_h5ad(h5ad_path)
    log.info(f"  Shape: {adata.shape[0]} cells x {adata.shape[1]} genes")

    # Check label column
    label_candidates = [label_col, "cell_type", "cell_ontology_class", "celltype"]
    found_label = None
    for lc in label_candidates:
        if lc in adata.obs.columns:
            found_label = lc
            break
    if found_label is None:
        log.warning(f"  No label column found. Available: {list(adata.obs.columns)}")
        return False
    if found_label != label_col:
        log.info(f"  Using label column '{found_label}' instead of '{label_col}'")

    labels = adata.obs[found_label].astype(str)

    # Get counts (genes x cells for R)
    counts = get_counts(adata)  # cells x genes
    counts_t = counts.T.tocsc()  # genes x cells

    # Save
    prefix = os.path.join(output_dir, ds_key)
    scipy.io.mmwrite(prefix + "_counts", counts_t)
    os.system(f"gzip -f {prefix}_counts.mtx")

    pd.DataFrame({"gene": adata.var_names}).to_csv(
        prefix + "_genes.csv", index=False
    )
    pd.DataFrame({"barcode": adata.obs_names}).to_csv(
        prefix + "_barcodes.csv", index=False
    )
    pd.DataFrame({
        "barcode": adata.obs_names,
        "label": labels.values,
        "tissue": tissue,
    }).to_csv(prefix + "_metadata.csv", index=False)

    log.info(f"  Saved: {ds_key}_counts.mtx.gz, genes/barcodes/metadata CSVs")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset", default=None, help="Process single dataset (key name)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.dataset:
        datasets_to_run = [args.dataset]
    else:
        datasets_to_run = list(DATASETS.keys())

    success = 0
    for ds_key in datasets_to_run:
        if ds_key not in DATASETS:
            log.warning(f"Unknown dataset: {ds_key}")
            continue
        h5ad_path = os.path.join(args.data_dir, ds_key + ".h5ad")
        if not os.path.exists(h5ad_path):
            log.warning(f"SKIP {ds_key}: {h5ad_path} not found")
            continue
        try:
            if preprocess_dataset(h5ad_path, ds_key, args.output_dir):
                success += 1
        except Exception as e:
            log.error(f"FAILED {ds_key}: {e}")

    log.info(f"Preprocessed {success}/{len(datasets_to_run)} datasets")


if __name__ == "__main__":
    main()
