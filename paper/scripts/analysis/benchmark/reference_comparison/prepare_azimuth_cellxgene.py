#!/usr/bin/env python3
"""
Preprocess CELLxGENE h5ad files for Azimuth datasets into R-friendly format.

For each dataset:
1. Read h5ad from CELLxGENE
2. Find the obs column matching our marker file cell type labels
3. Subsample large datasets (>100K cells) to manageable size
4. Output: expression.mtx, genes.csv, barcodes.csv, metadata.csv, config.txt

Usage:
    python prepare_azimuth_cellxgene.py \
        --data_dir /path/to/downloaded \
        --marker_dir /path/to/markers \
        --output_dir /path/to/prepared \
        [--dataset adipose]
"""

import argparse
import logging
import os
import sys
from collections import Counter
from difflib import SequenceMatcher

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

# Dataset configurations:
# key -> (h5ad_filename, marker_filename, display_name, sctype_tissue, max_cells)
AZIMUTH_DATASETS = {
    "adipose": {
        "h5ad": "azimuth_adipose.h5ad",
        "marker_file": "adipose_markers_L1.csv",
        "display_name": "Azimuth Adipose",
        "dir_name": "Azimuth_adipose",
        "sctype_tissue": "Immune system",
        "max_cells": 100000,
        # Known column mapping: CELLxGENE cell_type_ontology_term_id or cell_type
        "preferred_columns": ["cell_type", "cell_type_ontology_term_id"],
    },
    "heart": {
        "h5ad": "azimuth_heart.h5ad",
        "marker_file": "heart_markers_L1.csv",
        "display_name": "Azimuth Heart",
        "dir_name": "Azimuth_heart",
        "sctype_tissue": "Heart",
        "max_cells": 100000,
        "preferred_columns": ["cell_type"],
    },
    "kidney": {
        "h5ad": "azimuth_kidney.h5ad",
        "marker_file": "kidney_markers_L1.csv",
        "display_name": "Azimuth Kidney",
        "dir_name": "Azimuth_kidney",
        "sctype_tissue": "Kidney",
        "max_cells": 100000,
        "preferred_columns": ["cell_type"],
    },
    "fetal": {
        "h5ad": "azimuth_fetal.h5ad",
        "marker_file": "fetal_development_markers_L1.csv",
        "display_name": "Azimuth Fetal Development",
        "dir_name": "Azimuth_fetal",
        "sctype_tissue": "Immune system",
        "max_cells": 100000,
        "preferred_columns": ["cell_type"],
    },
    "bone_marrow": {
        "h5ad": "azimuth_bone_marrow.h5ad",
        "marker_file": "bone_marrow_markers_L2.csv",
        "display_name": "Azimuth Bone Marrow",
        "dir_name": "Azimuth_bone_marrow",
        "sctype_tissue": "Immune system",
        "max_cells": 100000,
        "preferred_columns": ["cell_type"],
    },
    "lung": {
        "h5ad": "HLCA_Core.h5ad",
        "marker_file": "lung_v2_markers_L2.csv",
        "display_name": "Azimuth Lung",
        "dir_name": "Azimuth_lung",
        "sctype_tissue": "Lung",
        "max_cells": 100000,
        "preferred_columns": ["ann_level_2", "cell_type"],
    },
    "liver": {
        "h5ad": "azimuth_liver.h5ad",
        "marker_file": "liver_markers_L2.csv",
        "display_name": "Azimuth Liver",
        "dir_name": "Azimuth_liver",
        "sctype_tissue": "Liver",
        "max_cells": 100000,
        "preferred_columns": ["cell_type"],
    },
    "pancreas": {
        "h5ad": "azimuth_pancreas.h5ad",
        "marker_file": "pancreas_markers_L1.csv",
        "display_name": "Azimuth Pancreas",
        "dir_name": "Azimuth_pancreas",
        "sctype_tissue": "Pancreas",
        "max_cells": 100000,
        "preferred_columns": ["cell_type"],
    },
}


def read_marker_celltypes(marker_path):
    """Read cluster/cell type names from marker CSV file."""
    celltypes = []
    with open(marker_path, "r") as f:
        header = f.readline().strip()
        for line in f:
            parts = line.strip().split(",")
            if parts:
                celltypes.append(parts[0])
    return celltypes


def fuzzy_match_score(s1, s2):
    """Compute fuzzy match score between two strings."""
    s1_lower = s1.lower().strip()
    s2_lower = s2.lower().strip()
    if s1_lower == s2_lower:
        return 1.0
    return SequenceMatcher(None, s1_lower, s2_lower).ratio()


def find_best_column(adata, marker_celltypes, preferred_columns=None):
    """Find the obs column that best matches marker file cell type labels."""
    marker_set = set(ct.lower().strip() for ct in marker_celltypes)
    best_col = None
    best_score = -1
    best_mapping = None

    # Try preferred columns first
    candidates = []
    if preferred_columns:
        for col in preferred_columns:
            if col in adata.obs.columns:
                candidates.append(col)

    # Add other categorical/string columns
    for col in adata.obs.columns:
        if col in candidates:
            continue
        n_unique = adata.obs[col].nunique()
        if 2 <= n_unique <= 200:
            # Likely a cell type column
            dtype = adata.obs[col].dtype
            if dtype == "category" or dtype == object or dtype.name.startswith("str"):
                candidates.append(col)

    log.info(f"  Candidate columns for matching: {candidates[:10]}")

    for col in candidates:
        col_values = set(str(v).lower().strip() for v in adata.obs[col].unique() if pd.notna(v))

        # Exact match score
        exact_matches = marker_set & col_values
        exact_ratio = len(exact_matches) / len(marker_set) if marker_set else 0

        # Fuzzy match: for each marker type, find best match in column values
        mapping = {}
        total_score = 0
        for mt in marker_celltypes:
            mt_lower = mt.lower().strip()
            if mt_lower in col_values:
                mapping[mt] = mt
                total_score += 1.0
            else:
                # Find best fuzzy match
                best_match = None
                best_match_score = 0
                for cv in col_values:
                    score = fuzzy_match_score(mt, cv)
                    if score > best_match_score:
                        best_match_score = score
                        best_match = cv
                if best_match_score > 0.6:
                    # Find the original (non-lowered) value
                    for v in adata.obs[col].unique():
                        if str(v).lower().strip() == best_match:
                            mapping[mt] = str(v)
                            break
                    total_score += best_match_score

        avg_score = total_score / len(marker_celltypes) if marker_celltypes else 0

        log.info(
            f"  Column '{col}': {len(col_values)} unique values, "
            f"exact={len(exact_matches)}/{len(marker_set)}, avg_fuzzy={avg_score:.3f}"
        )

        if avg_score > best_score:
            best_score = avg_score
            best_col = col
            best_mapping = mapping

    if best_col is None:
        log.warning("  No matching column found!")
        return None, None, None

    log.info(f"  Best column: '{best_col}' (score={best_score:.3f})")
    return best_col, best_mapping, best_score


def subsample_adata(adata, max_cells, label_col):
    """Subsample adata to max_cells, preserving proportional representation."""
    n_cells = adata.shape[0]
    if n_cells <= max_cells:
        return adata

    log.info(f"  Subsampling from {n_cells} to {max_cells} cells...")
    labels = adata.obs[label_col].values
    unique_labels = np.unique(labels)

    # Calculate proportional sample sizes
    label_counts = Counter(labels)
    indices = []
    rng = np.random.RandomState(42)

    for label in unique_labels:
        label_idx = np.where(labels == label)[0]
        n_sample = max(1, int(len(label_idx) * max_cells / n_cells))
        n_sample = min(n_sample, len(label_idx))
        sampled = rng.choice(label_idx, n_sample, replace=False)
        indices.extend(sampled)

    indices = sorted(indices)
    if len(indices) > max_cells:
        indices = sorted(rng.choice(indices, max_cells, replace=False))

    log.info(f"  Subsampled to {len(indices)} cells ({len(unique_labels)} cell types)")
    return adata[indices].copy()


def process_dataset(data_dir, marker_dir, output_dir, key, config):
    """Process a single Azimuth dataset."""
    h5ad_path = os.path.join(data_dir, config["h5ad"])
    # Also check in the main data dir (for HLCA_Core etc.)
    if not os.path.exists(h5ad_path):
        alt_path = os.path.join(os.path.dirname(data_dir), config["h5ad"])
        if os.path.exists(alt_path):
            h5ad_path = alt_path
        else:
            log.warning(f"  h5ad not found: {h5ad_path}")
            return False

    marker_path = os.path.join(marker_dir, config["marker_file"])
    if not os.path.exists(marker_path):
        log.warning(f"  Marker file not found: {marker_path}")
        return False

    ds_dir = os.path.join(output_dir, config["dir_name"])
    if os.path.exists(os.path.join(ds_dir, "expression.mtx")) and not config.get("force", False):
        log.info(f"  Already processed: {config['display_name']}")
        return True

    os.makedirs(ds_dir, exist_ok=True)

    # Read marker cell types
    marker_celltypes = read_marker_celltypes(marker_path)
    log.info(f"  Marker cell types ({len(marker_celltypes)}): {marker_celltypes[:5]}...")

    # Read h5ad
    log.info(f"  Loading {h5ad_path}...")
    adata = ad.read_h5ad(h5ad_path)
    log.info(f"  Shape: {adata.shape[0]} cells x {adata.shape[1]} genes")
    log.info(f"  Obs columns: {list(adata.obs.columns)}")

    # Find best matching column
    label_col, mapping, score = find_best_column(
        adata, marker_celltypes, config.get("preferred_columns")
    )

    if label_col is None or score < 0.3:
        log.error(f"  Could not find matching column (best score={score})")
        # Save obs column info for debugging
        with open(os.path.join(ds_dir, "column_info.txt"), "w") as f:
            f.write(f"Obs columns: {list(adata.obs.columns)}\n\n")
            for col in adata.obs.columns:
                n_unique = adata.obs[col].nunique()
                if n_unique < 100:
                    f.write(f"\n{col} ({n_unique} unique):\n")
                    for v in sorted(adata.obs[col].unique()):
                        f.write(f"  {v}\n")
        return False

    # Create mapped labels
    if mapping:
        # Reverse mapping: original_value -> marker_celltype
        reverse_map = {}
        for marker_ct, orig_val in mapping.items():
            # Find all obs values that match
            for obs_val in adata.obs[label_col].unique():
                if str(obs_val).lower().strip() == str(orig_val).lower().strip():
                    reverse_map[obs_val] = marker_ct

        # Apply mapping
        mapped_labels = adata.obs[label_col].map(
            lambda x: reverse_map.get(x, None)
        )
        # Keep only cells that map to a known cell type
        valid_mask = mapped_labels.notna()
        n_valid = valid_mask.sum()
        log.info(
            f"  Mapped {n_valid}/{adata.shape[0]} cells "
            f"({n_valid/adata.shape[0]*100:.1f}%) to {len(set(mapped_labels.dropna()))} cell types"
        )

        if n_valid < adata.shape[0] * 0.3:
            log.warning(f"  Low mapping rate ({n_valid/adata.shape[0]*100:.1f}%)")
            # Save debugging info
            with open(os.path.join(ds_dir, "mapping_debug.txt"), "w") as f:
                f.write(f"Label column: {label_col}\n")
                f.write(f"Mapping: {mapping}\n")
                f.write(f"Reverse mapping: {reverse_map}\n")
                f.write(f"\nObs values:\n")
                for v in sorted(adata.obs[label_col].unique()):
                    f.write(f"  '{v}' -> {reverse_map.get(v, 'UNMAPPED')}\n")

        adata = adata[valid_mask].copy()
        adata.obs["_mapped_label"] = mapped_labels[valid_mask].values
        label_col = "_mapped_label"

    # Subsample if needed
    adata = subsample_adata(adata, config.get("max_cells", 100000), label_col)

    # Get expression matrix
    X = adata.X
    if issparse(X):
        X = csc_matrix(X.T)  # transpose: genes x cells
    else:
        X = csc_matrix(np.array(X).T)

    # Save expression matrix
    log.info(f"  Saving expression matrix ({X.shape[0]} genes x {X.shape[1]} cells)...")
    sio.mmwrite(os.path.join(ds_dir, "expression.mtx"), X)

    # Save gene names (use feature_name/gene symbols if available, not Ensembl IDs)
    if "feature_name" in adata.var.columns:
        gene_names = adata.var["feature_name"].tolist()
        log.info(f"  Using 'feature_name' column for gene symbols (e.g. {gene_names[:3]})")
        # Handle duplicates by appending suffix
        seen = {}
        unique_genes = []
        for g in gene_names:
            g_str = str(g)
            if g_str in seen:
                seen[g_str] += 1
                unique_genes.append(f"{g_str}.{seen[g_str]}")
            else:
                seen[g_str] = 0
                unique_genes.append(g_str)
        gene_names = unique_genes
    else:
        gene_names = adata.var_names.tolist()
        log.info(f"  Using var_names for genes (e.g. {gene_names[:3]})")
    pd.DataFrame({"gene": gene_names}).to_csv(
        os.path.join(ds_dir, "genes.csv"), index=False
    )

    # Save barcodes
    pd.DataFrame({"barcode": adata.obs_names.tolist()}).to_csv(
        os.path.join(ds_dir, "barcodes.csv"), index=False
    )

    # Save metadata with label column
    metadata = pd.DataFrame(
        {
            "barcode": adata.obs_names.tolist(),
            "label": adata.obs[label_col].values,
        }
    )
    metadata.to_csv(os.path.join(ds_dir, "metadata.csv"), index=False)

    # Save config
    with open(os.path.join(ds_dir, "config.txt"), "w") as f:
        f.write(f"name={config['display_name']}\n")
        f.write(f"tissue={config['sctype_tissue']}\n")
        f.write(f"label_col={label_col}\n")
        f.write(f"n_cells={adata.shape[0]}\n")
        f.write(f"n_genes={adata.shape[1]}\n")
        f.write(f"n_celltypes={adata.obs[label_col].nunique()}\n")

    log.info(
        f"  Saved: {adata.shape[0]} cells, {adata.shape[1]} genes, "
        f"{adata.obs[label_col].nunique()} cell types"
    )

    # Save cell type summary
    ct_counts = adata.obs[label_col].value_counts()
    ct_counts.to_csv(os.path.join(ds_dir, "celltype_counts.csv"))

    return True


def main():
    parser = argparse.ArgumentParser(description="Prepare Azimuth CELLxGENE data for SingleR/scType")
    parser.add_argument("--data_dir", required=True, help="Directory with downloaded h5ad files")
    parser.add_argument("--marker_dir", required=True, help="Directory with marker CSV files")
    parser.add_argument("--output_dir", required=True, help="Output directory for prepared data")
    parser.add_argument("--dataset", default="all", help="Specific dataset to process (or 'all')")
    args = parser.parse_args()

    if args.dataset == "all":
        datasets = AZIMUTH_DATASETS
    elif args.dataset in AZIMUTH_DATASETS:
        datasets = {args.dataset: AZIMUTH_DATASETS[args.dataset]}
    else:
        log.error(f"Unknown dataset: {args.dataset}. Available: {list(AZIMUTH_DATASETS.keys())}")
        sys.exit(1)

    results = {}
    for key, config in datasets.items():
        log.info(f"\n===== Processing: {config['display_name']} =====")
        success = process_dataset(args.data_dir, args.marker_dir, args.output_dir, key, config)
        results[key] = success

    log.info("\n===== Summary =====")
    for key, success in results.items():
        status = "OK" if success else "FAILED"
        log.info(f"  {AZIMUTH_DATASETS[key]['display_name']}: {status}")


if __name__ == "__main__":
    main()
