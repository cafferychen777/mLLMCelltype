#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scANVI Benchmark Pipeline
=========================
Train scANVI on a labeled reference atlas, then predict cell types on query data.

Uses the modern scvi-tools API (v1.0+):
    SCVI.setup_anndata -> SCVI -> SCANVI.from_scvi_model -> predict

Model hyper-parameters follow the popV defaults:
    n_latent=20, n_layers=3, dropout_rate=0.05, gene_likelihood='nb'

Usage
-----
    python run_scanvi_benchmark.py \
        --dataset Thymus \
        --reference /path/to/reference.h5ad \
        --query /path/to/query.h5ad \
        --output_dir /path/to/output \
        --ref_label_key cell_type \
        --ref_batch_key donor
"""

import argparse
import logging
import os
import sys
import time
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scvi

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants: popV-derived hyper-parameters
# ---------------------------------------------------------------------------
SCVI_MODEL_KWARGS = {
    "n_latent": 20,
    "n_layers": 3,
    "dropout_rate": 0.05,
    "gene_likelihood": "nb",
    "use_batch_norm": "none",
    "use_layer_norm": "both",
    "encode_covariates": True,
}

SCANVI_CLASSIFIER_KWARGS = {
    "n_layers": 3,
    "dropout_rate": 0.1,
}

SCVI_TRAIN_KWARGS = {
    "max_epochs": 20,
    "batch_size": 512,
    "plan_kwargs": {"n_epochs_kl_warmup": 20},
}

SCANVI_TRAIN_KWARGS = {
    "max_epochs": 20,
    "batch_size": 512,
    "n_samples_per_label": 20,
    "plan_kwargs": {"n_epochs_kl_warmup": 20},
}

UNLABELED_CATEGORY = "unlabeled"
N_TOP_GENES = 4000


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def ensure_raw_counts(adata: ad.AnnData) -> ad.AnnData:
    """Return an AnnData with raw integer counts in .X.

    Strategy (in priority order):
      1. Use .layers['counts'] if present.
      2. Use .raw.X if present.
      3. Fall back to .X (assumed to already be raw counts).
    """
    if "counts" in adata.layers:
        logger.info("Using .layers['counts'] as raw counts.")
        adata.X = adata.layers["counts"].copy()
    elif adata.raw is not None:
        logger.info("Using .raw.X as raw counts.")
        adata = adata.raw.to_adata()
    else:
        logger.info("Assuming .X already contains raw counts.")
    return adata


def harmonize_genes(
    ref: ad.AnnData, query: ad.AnnData
) -> tuple[ad.AnnData, ad.AnnData]:
    """Subset both AnnData objects to the intersection of their gene sets."""
    shared_genes = sorted(set(ref.var_names) & set(query.var_names))
    n_ref = ref.n_vars
    n_query = query.n_vars
    n_shared = len(shared_genes)
    logger.info(
        "Gene harmonization: ref=%d, query=%d, shared=%d",
        n_ref,
        n_query,
        n_shared,
    )
    if n_shared == 0:
        raise ValueError(
            "No shared genes between reference and query. "
            "Check that gene names use the same convention (e.g. symbols vs Ensembl IDs)."
        )
    ref = ref[:, shared_genes].copy()
    query = query[:, shared_genes].copy()
    return ref, query


def select_hvgs(adata: ad.AnnData, n_top_genes: int = N_TOP_GENES) -> ad.AnnData:
    """Select highly variable genes using Seurat v3 flavour (expects raw counts)."""
    logger.info("Selecting %d highly variable genes ...", n_top_genes)
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top_genes,
        flavor="seurat_v3",
        subset=False,
    )
    n_hvg = adata.var["highly_variable"].sum()
    logger.info("HVGs selected: %d", n_hvg)
    return adata


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_scanvi_benchmark(
    dataset: str,
    reference_path: str,
    query_path: str,
    output_dir: str,
    ref_label_key: str = "cell_type",
    ref_batch_key: str | None = None,
    query_batch_key: str | None = None,
    n_top_genes: int = N_TOP_GENES,
    save_model: bool = False,
) -> pd.DataFrame:
    """Run the full scANVI benchmark pipeline.

    Parameters
    ----------
    dataset
        Name of the dataset (used for logging / output naming).
    reference_path
        Path to the reference .h5ad file (with cell-type labels).
    query_path
        Path to the query .h5ad file.
    output_dir
        Directory to write output CSV (and optionally the model).
    ref_label_key
        Column in reference .obs holding cell-type labels.
    ref_batch_key
        Column in reference .obs for batch correction. If None, a dummy
        constant batch is assigned.
    query_batch_key
        Column in query .obs for batch key. If None, a constant "query" batch
        is assigned.
    n_top_genes
        Number of highly variable genes to select.
    save_model
        Whether to persist the trained scANVI model to disk.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['barcode', 'scanvi_prediction', 'scanvi_probability'].
    """
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    logger.info("Loading reference: %s", reference_path)
    ref = sc.read_h5ad(reference_path)
    logger.info("Reference shape: %s", ref.shape)

    logger.info("Loading query: %s", query_path)
    query = sc.read_h5ad(query_path)
    logger.info("Query shape: %s", query.shape)

    # ------------------------------------------------------------------
    # 2. Validate label key
    # ------------------------------------------------------------------
    if ref_label_key not in ref.obs.columns:
        raise KeyError(
            f"Label key '{ref_label_key}' not found in reference .obs. "
            f"Available keys: {list(ref.obs.columns)}"
        )
    n_labels = ref.obs[ref_label_key].nunique()
    logger.info(
        "Reference labels ('%s'): %d unique cell types", ref_label_key, n_labels
    )

    # ------------------------------------------------------------------
    # 3. Ensure raw counts
    # ------------------------------------------------------------------
    ref = ensure_raw_counts(ref)
    query = ensure_raw_counts(query)

    # ------------------------------------------------------------------
    # 4. Harmonize gene sets
    # ------------------------------------------------------------------
    ref, query = harmonize_genes(ref, query)

    # ------------------------------------------------------------------
    # 5. Assign batch keys
    # ------------------------------------------------------------------
    batch_key = "_scanvi_batch"
    if ref_batch_key and ref_batch_key in ref.obs.columns:
        ref.obs[batch_key] = ref.obs[ref_batch_key].astype(str)
    else:
        ref.obs[batch_key] = "reference"
        logger.info("No batch key in reference; using constant 'reference' batch.")

    if query_batch_key and query_batch_key in query.obs.columns:
        query.obs[batch_key] = query.obs[query_batch_key].astype(str)
    else:
        query.obs[batch_key] = "query"
        logger.info("No batch key in query; using constant 'query' batch.")

    # ------------------------------------------------------------------
    # 6. Create combined AnnData
    # ------------------------------------------------------------------
    label_key = "_scanvi_labels"
    ref.obs[label_key] = ref.obs[ref_label_key].astype(str)
    query.obs[label_key] = UNLABELED_CATEGORY

    combined = ad.concat([ref, query], join="inner", label="_dataset", keys=["reference", "query"])
    combined.obs_names_make_unique()
    logger.info("Combined shape: %s", combined.shape)

    # ------------------------------------------------------------------
    # 7. HVG selection on combined data
    # ------------------------------------------------------------------
    combined = select_hvgs(combined, n_top_genes=n_top_genes)
    combined = combined[:, combined.var["highly_variable"]].copy()
    logger.info("After HVG subset: %s", combined.shape)

    # ------------------------------------------------------------------
    # 8. Setup and train SCVI
    # ------------------------------------------------------------------
    logger.info("Setting up SCVI model ...")
    scvi.model.SCVI.setup_anndata(
        combined,
        batch_key=batch_key,
        labels_key=label_key,
    )

    scvi_model = scvi.model.SCVI(combined, **SCVI_MODEL_KWARGS)
    logger.info("Training SCVI (unsupervised) ...")
    t0 = time.time()
    scvi_model.train(**SCVI_TRAIN_KWARGS)
    logger.info("SCVI training completed in %.1f s", time.time() - t0)

    # ------------------------------------------------------------------
    # 9. Convert to SCANVI and train
    # ------------------------------------------------------------------
    logger.info("Converting to SCANVI model ...")
    scanvi_model = scvi.model.SCANVI.from_scvi_model(
        scvi_model,
        unlabeled_category=UNLABELED_CATEGORY,
        classifier_parameters=SCANVI_CLASSIFIER_KWARGS,
    )

    logger.info("Training SCANVI (semi-supervised) ...")
    t0 = time.time()
    scanvi_model.train(**SCANVI_TRAIN_KWARGS)
    logger.info("SCANVI training completed in %.1f s", time.time() - t0)

    # ------------------------------------------------------------------
    # 10. Predict cell types for query cells
    # ------------------------------------------------------------------
    logger.info("Predicting cell types ...")
    predictions = scanvi_model.predict(combined)
    soft_predictions = scanvi_model.predict(combined, soft=True)

    # Extract query cell indices
    query_mask = combined.obs["_dataset"] == "query"
    query_barcodes = combined.obs_names[query_mask]
    query_preds = predictions[query_mask]
    query_probs = soft_predictions[query_mask].max(axis=1)

    results_df = pd.DataFrame(
        {
            "barcode": query_barcodes,
            "scanvi_prediction": query_preds.values
            if hasattr(query_preds, "values")
            else query_preds,
            "scanvi_probability": query_probs.values
            if hasattr(query_probs, "values")
            else query_probs,
        }
    )

    # ------------------------------------------------------------------
    # 11. Save results
    # ------------------------------------------------------------------
    out_csv = os.path.join(output_dir, f"{dataset}_scanvi_predictions.csv")
    results_df.to_csv(out_csv, index=False)
    logger.info("Predictions saved to %s", out_csv)

    if save_model:
        model_dir = os.path.join(output_dir, f"{dataset}_scanvi_model")
        scanvi_model.save(model_dir, overwrite=True)
        logger.info("Model saved to %s", model_dir)

    # ------------------------------------------------------------------
    # 12. Summary
    # ------------------------------------------------------------------
    logger.info("--- Prediction summary ---")
    pred_counts = results_df["scanvi_prediction"].value_counts()
    logger.info("Unique predicted types: %d", len(pred_counts))
    logger.info("Top 10 predictions:\n%s", pred_counts.head(10).to_string())
    logger.info(
        "Mean prediction probability: %.3f",
        results_df["scanvi_probability"].mean(),
    )

    return results_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run scANVI cell-type prediction benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g. Thymus, PBMC, LCA).",
    )
    parser.add_argument(
        "--reference",
        type=str,
        required=True,
        help="Path to reference .h5ad file with cell-type labels.",
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Path to query .h5ad file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for predictions and model.",
    )
    parser.add_argument(
        "--ref_label_key",
        type=str,
        default="cell_type",
        help="Column in reference .obs for cell-type labels (default: cell_type).",
    )
    parser.add_argument(
        "--ref_batch_key",
        type=str,
        default=None,
        help="Column in reference .obs for batch correction (default: None).",
    )
    parser.add_argument(
        "--query_batch_key",
        type=str,
        default=None,
        help="Column in query .obs for batch key (default: None).",
    )
    parser.add_argument(
        "--n_top_genes",
        type=int,
        default=N_TOP_GENES,
        help=f"Number of HVGs to select (default: {N_TOP_GENES}).",
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        help="Save the trained SCANVI model to disk.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("scANVI Benchmark: %s", args.dataset)
    logger.info("=" * 60)
    logger.info("scvi-tools version: %s", scvi.__version__)
    logger.info("Reference: %s", args.reference)
    logger.info("Query: %s", args.query)
    logger.info("Output: %s", args.output_dir)
    logger.info("Label key: %s", args.ref_label_key)
    logger.info("Batch key (ref): %s", args.ref_batch_key)
    logger.info("Batch key (query): %s", args.query_batch_key)
    logger.info("N HVGs: %d", args.n_top_genes)

    t_start = time.time()

    try:
        results = run_scanvi_benchmark(
            dataset=args.dataset,
            reference_path=args.reference,
            query_path=args.query,
            output_dir=args.output_dir,
            ref_label_key=args.ref_label_key,
            ref_batch_key=args.ref_batch_key,
            query_batch_key=args.query_batch_key,
            n_top_genes=args.n_top_genes,
            save_model=args.save_model,
        )
        logger.info(
            "Pipeline completed successfully in %.1f s", time.time() - t_start
        )
    except Exception:
        logger.exception("Pipeline failed for dataset '%s'", args.dataset)
        sys.exit(1)


if __name__ == "__main__":
    main()
