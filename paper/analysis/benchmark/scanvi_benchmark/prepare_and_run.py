#!/usr/bin/env python3
"""
scANVI Benchmark: Data Preparation + Training Pipeline
========================================================
Downloads, prepares, and runs scANVI on 6 datasets matching the popV comparison.

Datasets (subset of Extended Data Fig. 8, excluding Kidney due to missing TS ref):
  1. Thymus:           TS_Thymus (ref)   → Developmental Thymus (query)
  2. LCA:              TS_Lung (ref)     → Lung Cell Atlas (query)
  3. GTEx Lung:        TS_Lung (ref)     → GTEx snRNA-seq Lung (query)
  4. GTEx Heart:       TS_Heart (ref)    → GTEx snRNA-seq Heart (query)
  5. Azimuth Pancreas: TS_Pancreas (ref) → Azimuth Pancreas ref (query)
  6. Azimuth Liver:    TS_Liver (ref)    → Azimuth Liver ref (query)

Usage:
  python prepare_and_run.py download   # Download missing data
  python prepare_and_run.py prepare    # Fix gene names, convert formats
  python prepare_and_run.py run <dataset>  # Run scANVI on one dataset
  python prepare_and_run.py run all    # Run scANVI on all 6 datasets
"""

import argparse
import getpass
import logging
import os
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRATCH_ROOT = os.environ.get(
    "MLLMCELLTYPE_SCRATCH_ROOT", os.path.join("/scratch/user", getpass.getuser())
)
BASE_DIR = os.environ.get(
    "MLLMCELLTYPE_SCANVI_ROOT", os.path.join(SCRATCH_ROOT, "scanvi_benchmark")
)
DATA_DIR = os.path.join(BASE_DIR, "data")
REF_DIR = os.path.join(DATA_DIR, "ref")
QUERY_DIR = os.path.join(DATA_DIR, "query")
DOWNLOAD_DIR = os.path.join(DATA_DIR, "downloads")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# ---------------------------------------------------------------------------
# Download URLs
# ---------------------------------------------------------------------------
DOWNLOADS = {
    "gtex": {
        "url": "https://storage.googleapis.com/adult-gtex/single-cell/v9/snrna-seq-data/GTEx_8_tissues_snRNAseq_atlas_071421.public_obs.h5ad",
        "filename": "GTEx_8_tissues_snRNAseq_atlas.h5ad",
        "description": "GTEx 8-tissue snRNA-seq atlas (Eraslan et al. 2022)",
    },
    "azimuth_pancreas": {
        "url": "https://zenodo.org/records/4546926/files/ref.Rds?download=1",
        "filename": "azimuth_pancreas_ref.Rds",
        "description": "Azimuth Human Pancreas reference",
    },
    "azimuth_liver": {
        "url": "https://zenodo.org/records/7770308/files/ref.Rds?download=1",
        "filename": "azimuth_liver_ref.Rds",
        "description": "Azimuth Human Liver reference",
    },
}

# ---------------------------------------------------------------------------
# Dataset configuration (7 datasets matching popV comparison)
# ---------------------------------------------------------------------------
DATASETS = {
    "thymus": {
        "ref": os.path.join(REF_DIR, "TS_Thymus_filtered.h5ad"),
        "query": os.path.join(QUERY_DIR, "Thymus_prepared.h5ad"),
        "query_raw": os.path.join(QUERY_DIR, "Thymus.h5ad"),
        "ref_label": "cell_ontology_class",
        "query_label": "cell_type",
        "ref_batch": "donor_id",
        "query_batch": "donor_id",
    },
    "lca": {
        "ref": os.path.join(REF_DIR, "TS_Lung_filtered.h5ad"),
        "query": os.path.join(QUERY_DIR, "LCA_prepared.h5ad"),
        "query_raw": os.path.join(QUERY_DIR, "LCA_with_umap.h5ad"),
        "ref_label": "cell_ontology_class",
        "query_label": "cell_type",
        "ref_batch": "donor_id",
        "query_batch": "donor_id",
    },
    "gtex_lung": {
        "ref": os.path.join(REF_DIR, "TS_Lung_filtered.h5ad"),
        "query": os.path.join(QUERY_DIR, "GTEx_Lung_prepared.h5ad"),
        "ref_label": "cell_ontology_class",
        "query_label": "Granular cell type",
        "ref_batch": "donor_id",
        "query_batch": "channel",
    },
    "gtex_heart": {
        "ref": os.path.join(REF_DIR, "TS_Heart_filtered.h5ad"),
        "query": os.path.join(QUERY_DIR, "GTEx_Heart_prepared.h5ad"),
        "ref_label": "cell_ontology_class",
        "query_label": "Granular cell type",
        "ref_batch": "donor_id",
        "query_batch": "channel",
    },
    "azimuth_pancreas": {
        "ref": os.path.join(REF_DIR, "TS_Pancreas_filtered.h5ad"),
        "query": os.path.join(QUERY_DIR, "Pancreas_CELLxGENE_prepared.h5ad"),
        "ref_label": "cell_ontology_class",
        "query_label": "cell_type",
        "ref_batch": "donor_id",
        "query_batch": "dataset_id",
    },
    "azimuth_liver": {
        "ref": os.path.join(REF_DIR, "TS_Liver_filtered.h5ad"),
        "query": os.path.join(QUERY_DIR, "Liver_CELLxGENE_prepared.h5ad"),
        "ref_label": "cell_ontology_class",
        "query_label": "cell_type",
        "ref_batch": "donor_id",
        "query_batch": "dataset_id",
    },
    "azimuth_kidney": {
        "ref": os.path.join(REF_DIR, "TS_Kidney_filtered.h5ad"),
        "query": os.path.join(QUERY_DIR, "Kidney_CELLxGENE_prepared.h5ad"),
        "ref_label": "cell_ontology_class",
        "query_label": "cell_type",
        "ref_batch": "donor_id",
        "query_batch": "dataset_id",
    },
}


# ===================================================================
# STEP 1: Download
# ===================================================================
def cmd_download():
    """Download all missing datasets."""
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    for key, info in DOWNLOADS.items():
        dest = os.path.join(DOWNLOAD_DIR, info["filename"])
        if os.path.exists(dest):
            log.info("[SKIP] %s already exists: %s", key, dest)
            continue

        log.info("[DOWNLOAD] %s: %s", key, info["description"])
        log.info("  URL: %s", info["url"])
        log.info("  Dest: %s", dest)

        ret = subprocess.run(
            ["wget", "-q", "--show-progress", "-O", dest, info["url"]],
            check=False,
        )
        if ret.returncode != 0:
            log.error("Download failed for %s", key)
            if os.path.exists(dest):
                os.remove(dest)
        else:
            log.info("[OK] %s downloaded", key)


# ===================================================================
# STEP 2: Prepare (gene name harmonization + format conversion)
# ===================================================================
def harmonize_query_genes(query_path, output_path):
    """Convert Ensembl ID var_names to gene symbols using feature_name column.

    If the data has been HVG-subsetted (fewer genes in .X than .raw),
    restore the full gene set from .raw first to ensure sufficient overlap.
    """
    import anndata as ad

    if os.path.exists(output_path):
        log.info("[SKIP] %s already exists", output_path)
        return

    log.info("Harmonizing gene names: %s -> %s", query_path, output_path)
    adata = ad.read_h5ad(query_path)

    # If .raw has more genes than .X, use .raw to restore full gene set
    if adata.raw is not None and adata.raw.shape[1] > adata.shape[1]:
        log.info("Restoring full gene set from .raw (%d -> %d genes)",
                 adata.n_vars, adata.raw.shape[1])
        raw_adata = adata.raw.to_adata()
        raw_adata.obs = adata.obs.copy()
        adata = raw_adata

    if "feature_name" in adata.var.columns:
        # Use feature_name (gene symbols) as var_names
        adata.var["ensembl_id"] = adata.var_names.copy()
        adata.var_names = adata.var["feature_name"].values
        adata.var_names_make_unique()
        log.info("Converted var_names to gene symbols (%d genes)", adata.n_vars)
    else:
        log.info("No feature_name column; var_names unchanged")

    adata.write_h5ad(output_path)
    log.info("[OK] Saved: %s", output_path)


def extract_gtex_tissue(tissue_name, output_path):
    """Extract a single tissue from the GTEx 8-tissue atlas."""
    import anndata as ad

    if os.path.exists(output_path):
        log.info("[SKIP] %s already exists", output_path)
        return

    gtex_path = os.path.join(DOWNLOAD_DIR, DOWNLOADS["gtex"]["filename"])
    if not os.path.exists(gtex_path):
        log.error("GTEx atlas not found at %s. Run 'download' first.", gtex_path)
        return

    log.info("Extracting %s from GTEx atlas...", tissue_name)
    adata = ad.read_h5ad(gtex_path)
    log.info("GTEx atlas: %s, obs columns: %s", adata.shape, list(adata.obs.columns[:10]))

    # Find the tissue column
    tissue_col = None
    for col in ["tissue", "Tissue", "tissue_detailed", "Broad cell type"]:
        if col in adata.obs.columns:
            tissues = adata.obs[col].unique()
            log.info("Column '%s' has values: %s", col, list(tissues)[:10])
            if tissue_name in tissues:
                tissue_col = col
                break

    if tissue_col is None:
        # Try partial match
        for col in adata.obs.columns:
            if adata.obs[col].dtype == "object" or adata.obs[col].dtype.name == "category":
                vals = adata.obs[col].astype(str).unique()
                matches = [v for v in vals if tissue_name.lower() in v.lower()]
                if matches:
                    log.info("Found '%s' match in column '%s': %s", tissue_name, col, matches)
                    tissue_col = col
                    tissue_name = matches[0]
                    break

    if tissue_col is None:
        log.error("Could not find tissue '%s' in GTEx atlas", tissue_name)
        log.info("Available obs columns: %s", list(adata.obs.columns))
        return

    mask = adata.obs[tissue_col] == tissue_name
    tissue_adata = adata[mask].copy()
    log.info("Extracted %s: %d cells", tissue_name, tissue_adata.n_obs)

    # Ensure gene symbols in var_names
    if "feature_name" in tissue_adata.var.columns:
        tissue_adata.var["ensembl_id"] = tissue_adata.var_names.copy()
        tissue_adata.var_names = tissue_adata.var["feature_name"].values
        tissue_adata.var_names_make_unique()

    tissue_adata.write_h5ad(output_path)
    log.info("[OK] Saved: %s", output_path)


def convert_azimuth_rds(rds_key, output_path):
    """Convert Azimuth RDS reference to h5ad using an R subprocess."""
    if os.path.exists(output_path):
        log.info("[SKIP] %s already exists", output_path)
        return

    rds_path = os.path.join(DOWNLOAD_DIR, DOWNLOADS[rds_key]["filename"])
    if not os.path.exists(rds_path):
        log.error("RDS file not found at %s. Run 'download' first.", rds_path)
        return

    log.info("Converting %s -> %s", rds_path, output_path)

    # Write a temporary R script for conversion
    # Uses Seurat + Matrix for robust RDS → sparse matrix export
    r_script = f"""
.libPaths(c("~/R/4.4", .libPaths()))
library(Seurat)
library(Matrix)

cat("Loading RDS...\\n")
ref <- readRDS("{rds_path}")

# Extract the Seurat object (may be nested in Azimuth reference list)
obj <- NULL
if (inherits(ref, "Seurat")) {{
    obj <- ref
}} else if (is.list(ref)) {{
    for (nm in c("map", "reference", names(ref))) {{
        if (!is.null(ref[[nm]]) && inherits(ref[[nm]], "Seurat")) {{
            obj <- ref[[nm]]
            cat("Found Seurat object in slot:", nm, "\\n")
            break
        }}
    }}
}}

if (is.null(obj)) {{
    cat("ERROR: Could not find Seurat object. Class:", class(ref), "\\n")
    if (is.list(ref)) cat("Names:", paste(names(ref), collapse=", "), "\\n")
    quit(status = 1)
}}

cat("Seurat object:", ncol(obj), "cells,", nrow(obj), "genes\\n")
cat("Metadata columns:", paste(colnames(obj@meta.data), collapse=", "), "\\n")

# Get counts matrix - handle both v3 and v5 Assay formats
assay_name <- DefaultAssay(obj)
assay_obj <- obj[[assay_name]]

# Try to get counts from available layers
available_layers <- tryCatch(Layers(assay_obj), error = function(e) NULL)
cat("Available layers:", paste(available_layers, collapse=", "), "\\n")

counts <- NULL
for (lyr in c("counts", "data")) {{
    if (lyr %in% available_layers) {{
        counts <- tryCatch(
            LayerData(obj, assay = assay_name, layer = lyr),
            error = function(e) NULL
        )
        if (!is.null(counts) && (nrow(counts) > 0 && ncol(counts) > 0)) {{
            cat("Using layer:", lyr, "\\n")
            break
        }}
    }}
}}

# Fallback: try direct slot access for older format
if (is.null(counts) || nrow(counts) == 0 || ncol(counts) == 0) {{
    counts <- tryCatch(assay_obj@data, error = function(e) NULL)
    if (!is.null(counts) && nrow(counts) > 0) {{
        cat("Using @data slot directly\\n")
    }}
}}

if (is.null(counts) || nrow(counts) == 0) {{
    cat("ERROR: Could not extract count matrix\\n")
    quit(status = 1)
}}

# Ensure sparse format
if (!inherits(counts, "dgCMatrix")) {{
    counts <- as(counts, "dgCMatrix")
}}

cat("Matrix dimensions:", nrow(counts), "x", ncol(counts), "\\n")
cat("Matrix sum:", sum(counts@x), "\\n")

# Save as sparse matrix + metadata for Python to read
writeMM(counts, "{output_path}.mtx")
write.csv(obj@meta.data, "{output_path}.meta.csv")
write.csv(data.frame(gene = rownames(counts)), "{output_path}.genes.csv", row.names = FALSE)
cat("Saved intermediate files successfully.\\n")
"""
    r_script_path = output_path + ".convert.R"
    with open(r_script_path, "w") as f:
        f.write(r_script)

    ret = subprocess.run(
        ["Rscript", r_script_path],
        capture_output=True,
        text=True,
    )
    log.info("R stdout: %s", ret.stdout[-500:] if ret.stdout else "")
    if ret.returncode != 0:
        log.error("R stderr: %s", ret.stderr[-500:] if ret.stderr else "")
        return

    # Now read intermediate files in Python and create h5ad
    import anndata as ad
    import numpy as np
    import pandas as pd
    from scipy.io import mmread

    mtx_path = f"{output_path}.mtx"
    meta_path = f"{output_path}.meta.csv"
    genes_path = f"{output_path}.genes.csv"

    if not os.path.exists(mtx_path):
        log.error("Intermediate MTX file not created")
        return

    log.info("Reading intermediate files...")
    counts = mmread(mtx_path).T.tocsr()  # genes x cells -> cells x genes
    meta = pd.read_csv(meta_path, index_col=0)
    genes = pd.read_csv(genes_path)

    adata = ad.AnnData(
        X=counts,
        obs=meta,
        var=pd.DataFrame(index=genes["gene"].values),
    )
    adata.var_names_make_unique()
    log.info("Created AnnData: %s", adata.shape)
    log.info("Obs columns: %s", list(adata.obs.columns[:10]))

    adata.write_h5ad(output_path)
    log.info("[OK] Saved: %s", output_path)

    # Cleanup intermediate files
    for p in [mtx_path, meta_path, genes_path, r_script_path]:
        if os.path.exists(p):
            os.remove(p)


def cmd_prepare():
    """Prepare all query datasets: harmonize gene names, convert formats."""
    os.makedirs(QUERY_DIR, exist_ok=True)

    # 1. Thymus: fix gene names (Ensembl -> symbols)
    harmonize_query_genes(
        os.path.join(QUERY_DIR, "Thymus.h5ad"),
        os.path.join(QUERY_DIR, "Thymus_prepared.h5ad"),
    )

    # 2. LCA: fix gene names
    harmonize_query_genes(
        os.path.join(QUERY_DIR, "LCA_with_umap.h5ad"),
        os.path.join(QUERY_DIR, "LCA_prepared.h5ad"),
    )

    # 3-4. GTEx Lung and Heart: extract from atlas
    extract_gtex_tissue("Lung", os.path.join(QUERY_DIR, "GTEx_Lung_prepared.h5ad"))
    extract_gtex_tissue("Heart", os.path.join(QUERY_DIR, "GTEx_Heart_prepared.h5ad"))

    # 5-6. Azimuth references: convert RDS -> h5ad
    convert_azimuth_rds("azimuth_pancreas", os.path.join(QUERY_DIR, "Azimuth_Pancreas_prepared.h5ad"))
    convert_azimuth_rds("azimuth_liver", os.path.join(QUERY_DIR, "Azimuth_Liver_prepared.h5ad"))

    log.info("=== Preparation complete ===")


# ===================================================================
# STEP 3: Run scANVI
# ===================================================================
def run_scanvi_single(dataset_name):
    """Run scANVI on a single dataset."""
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scanpy as sc
    import scvi

    if dataset_name not in DATASETS:
        log.error("Unknown dataset: %s. Available: %s", dataset_name, list(DATASETS.keys()))
        return

    config = DATASETS[dataset_name]
    out_dir = os.path.join(RESULTS_DIR, dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    out_csv = os.path.join(out_dir, f"{dataset_name}_scanvi_predictions.csv")
    if os.path.exists(out_csv):
        log.info("[SKIP] Results already exist: %s", out_csv)
        return

    # Check files exist
    if not os.path.exists(config["ref"]):
        log.error("Reference not found: %s", config["ref"])
        return
    if not os.path.exists(config["query"]):
        log.error("Query not found: %s", config["query"])
        return

    # Load data
    log.info("=== %s ===", dataset_name.upper())
    log.info("Loading reference: %s", config["ref"])
    ref = ad.read_h5ad(config["ref"])
    log.info("Reference: %s", ref.shape)

    log.info("Loading query: %s", config["query"])
    query = ad.read_h5ad(config["query"])
    log.info("Query: %s", query.shape)

    # Validate label columns
    ref_label = config["ref_label"]
    query_label = config["query_label"]

    if ref_label not in ref.obs.columns:
        log.error("Label '%s' not in ref.obs: %s", ref_label, list(ref.obs.columns))
        return
    if query_label not in query.obs.columns:
        log.error("Label '%s' not in query.obs: %s", query_label, list(query.obs.columns))
        return

    # Save ground truth
    query_true = query.obs[query_label].astype(str).copy()

    # Ensure raw counts
    for adata, name in [(ref, "ref"), (query, "query")]:
        if "counts" in adata.layers:
            adata.X = adata.layers["counts"].copy()
            log.info("%s: using .layers['counts']", name)
        elif adata.raw is not None:
            raw_ad = adata.raw.to_adata()
            if raw_ad.shape[1] != adata.shape[1]:
                # raw has different genes; we'll handle after gene intersection
                log.info("%s: .raw has different gene set (%d vs %d), using .X", name, raw_ad.shape[1], adata.shape[1])
            else:
                adata.X = adata.raw.X.copy()
                log.info("%s: using .raw.X", name)
        else:
            log.info("%s: using .X as-is", name)

    # Gene intersection
    common_genes = sorted(set(ref.var_names) & set(query.var_names))
    log.info("Common genes: %d (ref=%d, query=%d)", len(common_genes), ref.n_vars, query.n_vars)

    if len(common_genes) < 1000:
        log.error("Too few common genes (%d). Aborting.", len(common_genes))
        return

    ref = ref[:, common_genes].copy()
    query = query[:, common_genes].copy()

    # Set up labels and batch
    label_key = "_scanvi_label"
    batch_key = "_scanvi_batch"

    ref.obs[label_key] = ref.obs[ref_label].astype(str)
    query.obs[label_key] = "unlabeled"

    ref_batch = config.get("ref_batch")
    if ref_batch and ref_batch in ref.obs.columns:
        ref.obs[batch_key] = ref.obs[ref_batch].astype(str)
    else:
        ref.obs[batch_key] = "ref"

    query_batch = config.get("query_batch")
    if query_batch and query_batch in query.obs.columns:
        query.obs[batch_key] = query.obs[query_batch].astype(str)
    else:
        query.obs[batch_key] = "query"

    # Concatenate
    combined = ad.concat([ref, query], keys=["ref", "query"], label="_dataset")
    combined.obs_names_make_unique()
    log.info("Combined: %s", combined.shape)

    # HVG selection
    n_hvg = min(4000, combined.n_vars)
    sc.pp.highly_variable_genes(
        combined, n_top_genes=n_hvg, flavor="seurat_v3", batch_key=batch_key, subset=True
    )
    log.info("After HVG: %s", combined.shape)

    # Train scVI
    scvi.model.SCVI.setup_anndata(combined, batch_key=batch_key, labels_key=label_key)
    scvi_model = scvi.model.SCVI(
        combined,
        n_latent=20,
        n_layers=3,
        dropout_rate=0.05,
        gene_likelihood="nb",
        use_batch_norm="none",
        use_layer_norm="both",
        encode_covariates=True,
    )
    log.info("Training scVI...")
    scvi_model.train(
        max_epochs=400,
        batch_size=512,
        early_stopping=True,
        early_stopping_patience=30,
        train_size=0.9,
    )

    # Save scVI model in case scANVI crashes
    scvi_save_path = os.path.join(RESULTS_DIR, ds_name, "scvi_model")
    os.makedirs(os.path.dirname(scvi_save_path), exist_ok=True)
    scvi_model.save(scvi_save_path, overwrite=True)
    log.info("Saved scVI model to %s", scvi_save_path)

    # Convert to scANVI
    scanvi_model = scvi.model.SCANVI.from_scvi_model(
        scvi_model,
        unlabeled_category="unlabeled",
        labels_key=label_key,
    )
    log.info("Training scANVI...")
    scanvi_model.train(
        max_epochs=200,
        batch_size=512,
        n_samples_per_label=20,
        early_stopping=True,
        early_stopping_patience=20,
        early_stopping_monitor="validation_accuracy",
    )

    # Predict
    predictions = scanvi_model.predict(combined)
    probs = scanvi_model.predict(combined, soft=True)

    # Extract query results
    query_mask = combined.obs["_dataset"] == "query"
    result = pd.DataFrame({
        "barcode": combined.obs_names[query_mask],
        "scanvi_prediction": predictions[query_mask].values if hasattr(predictions[query_mask], "values") else predictions[query_mask],
        "scanvi_probability": probs[query_mask].max(axis=1).values if hasattr(probs[query_mask].max(axis=1), "values") else probs[query_mask].max(axis=1),
        "true_label": query_true.loc[combined.obs_names[query_mask]].values,
    })

    result.to_csv(out_csv, index=False)
    log.info("Saved predictions: %s (%d cells)", out_csv, len(result))

    # Quick accuracy
    exact = (result["scanvi_prediction"] == result["true_label"]).mean()
    log.info("Exact match accuracy: %.3f", exact)


def cmd_run(dataset):
    """Run scANVI on specified dataset(s)."""
    if dataset == "all":
        for ds in DATASETS:
            try:
                run_scanvi_single(ds)
            except Exception as e:
                log.error("Failed on %s: %s", ds, e, exc_info=True)
    else:
        run_scanvi_single(dataset)


# ===================================================================
# CLI
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description="scANVI benchmark pipeline")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("download", help="Download missing datasets")
    subparsers.add_parser("prepare", help="Prepare query datasets (gene name fix, format conversion)")

    run_parser = subparsers.add_parser("run", help="Run scANVI benchmark")
    run_parser.add_argument("dataset", help="Dataset name or 'all'")

    subparsers.add_parser("status", help="Check data availability")

    args = parser.parse_args()

    if args.command == "download":
        cmd_download()
    elif args.command == "prepare":
        cmd_prepare()
    elif args.command == "run":
        cmd_run(args.dataset)
    elif args.command == "status":
        cmd_status()
    else:
        parser.print_help()


def cmd_status():
    """Print status of all data files."""
    log.info("=== Data Status ===")
    for ds, cfg in DATASETS.items():
        ref_ok = "✓" if os.path.exists(cfg["ref"]) else "✗"
        query_ok = "✓" if os.path.exists(cfg["query"]) else "✗"
        result_path = os.path.join(RESULTS_DIR, ds, f"{ds}_scanvi_predictions.csv")
        result_ok = "✓" if os.path.exists(result_path) else "✗"
        log.info("  %-20s  ref=%s  query=%s  result=%s", ds, ref_ok, query_ok, result_ok)

    log.info("\n=== Downloads ===")
    for key, info in DOWNLOADS.items():
        path = os.path.join(DOWNLOAD_DIR, info["filename"])
        ok = "✓" if os.path.exists(path) else "✗"
        log.info("  %-20s  %s  %s", key, ok, path)


if __name__ == "__main__":
    main()
