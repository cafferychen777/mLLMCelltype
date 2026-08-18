"""
scANVI benchmark: train on Tabula Sapiens tissues, predict on independent datasets.

Usage:
    python run_all_scanvi.py --dataset thymus --data_dir /path/to/data --output_dir /path/to/output
    python run_all_scanvi.py --dataset all --data_dir /path/to/data --output_dir /path/to/output

Datasets:
    thymus: TS_Thymus (ref) → Thymus developmental atlas (query)
    lung:   TS_Lung (ref) → LCA (query)
    blood:  TS_Blood (ref, leave-one-donor-out cross-validation)
    heart:  TS_Heart (ref, leave-one-donor-out cross-validation)
    liver:  TS_Liver (ref, leave-one-donor-out cross-validation)
    pancreas: TS_Pancreas (ref, leave-one-donor-out cross-validation)
"""

import argparse
import logging
import os
import sys
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------
DATASET_CONFIG = {
    "thymus": {
        "reference": "TS_Thymus_filtered.h5ad",
        "query": "Thymus.h5ad",
        "ref_label": "cell_ontology_class",
        "query_label": "cell_type",
        "ref_batch": "donor_id",
        "query_batch": "donor_id",
        "mode": "cross_dataset",
    },
    "lung": {
        "reference": "TS_Lung_filtered.h5ad",
        "query": "LCA.h5ad",
        "ref_label": "cell_ontology_class",
        "query_label": "cell_type",
        "ref_batch": "donor_id",
        "query_batch": "donor_id",
        "mode": "cross_dataset",
    },
    "blood": {
        "reference": "TS_Blood_filtered.h5ad",
        "ref_label": "cell_ontology_class",
        "ref_batch": "donor_id",
        "mode": "leave_one_donor_out",
    },
    "heart": {
        "reference": "TS_Heart_filtered.h5ad",
        "ref_label": "cell_ontology_class",
        "ref_batch": "donor_id",
        "mode": "leave_one_donor_out",
    },
    "liver": {
        "reference": "TS_Liver_filtered.h5ad",
        "ref_label": "cell_ontology_class",
        "ref_batch": "donor_id",
        "mode": "leave_one_donor_out",
    },
    "pancreas": {
        "reference": "TS_Pancreas_filtered.h5ad",
        "ref_label": "cell_ontology_class",
        "ref_batch": "donor_id",
        "mode": "leave_one_donor_out",
    },
}


def get_raw_counts_adata(adata):
    """Extract raw counts, returning (counts_matrix, possibly_expanded_adata).

    If .raw has a different gene set, expands adata to use .raw's full genes.
    """
    if "counts" in adata.layers:
        log.info("Using counts from adata.layers['counts']")
        return adata.layers["counts"].copy(), adata
    if adata.raw is not None:
        if adata.raw.X.shape[1] != adata.shape[1]:
            log.info(f"Expanding from .raw ({adata.raw.X.shape[1]} genes vs {adata.shape[1]})")
            raw_adata = adata.raw.to_adata()
            raw_adata.obs = adata.obs.copy()
            return raw_adata.X.copy(), raw_adata
        log.info("Using counts from adata.raw.X")
        return adata.raw.X.copy(), adata
    sample = adata.X[:100, :100]
    if hasattr(sample, "toarray"):
        sample = sample.toarray()
    if np.allclose(sample, sample.astype(int)):
        log.info("adata.X appears to contain integer counts, using directly")
        return adata.X.copy(), adata
    log.warning("No raw counts found; using adata.X (may be normalized)")
    return adata.X.copy(), adata


def prepare_adata_for_scvi(adata, label_col, batch_col):
    """Prepare an AnnData for scVI/scANVI: ensure raw counts, clean labels."""
    adata = adata.copy()
    counts, adata = get_raw_counts_adata(adata)
    adata.X = counts
    # Clean labels
    adata.obs["_label"] = adata.obs[label_col].astype(str)
    adata.obs["_batch"] = adata.obs[batch_col].astype(str) if batch_col in adata.obs else "batch_0"
    # Remove cells with missing labels
    mask = adata.obs["_label"].notna() & (adata.obs["_label"] != "nan") & (adata.obs["_label"] != "")
    if mask.sum() < len(mask):
        log.info(f"Removing {(~mask).sum()} cells with missing labels")
        adata = adata[mask].copy()
    adata.var_names_make_unique()
    log.info(f"Prepared: {adata.shape}, labels: {adata.obs['_label'].nunique()}, batches: {adata.obs['_batch'].nunique()}")
    return adata


def run_scanvi(ref_adata, query_adata, n_hvg=4000, max_epochs_scvi=100, max_epochs_scanvi=20):
    """Train scVI → scANVI on reference, predict on query."""
    import scvi

    log.info(f"Reference: {ref_adata.shape}, Query: {query_adata.shape}")

    # Find common genes
    common_genes = ref_adata.var_names.intersection(query_adata.var_names)
    log.info(f"Common genes: {len(common_genes)}")
    if len(common_genes) < 2000:
        log.warning(f"Only {len(common_genes)} common genes — results may be unreliable")

    ref_adata = ref_adata[:, common_genes].copy()
    query_adata = query_adata[:, common_genes].copy()

    # Mark query as unlabeled
    query_adata.obs["_label"] = "unlabeled"

    # Concatenate
    combined = ad.concat([ref_adata, query_adata], keys=["ref", "query"], label="_split")
    combined.obs_names_make_unique()

    # HVG selection
    sc.pp.highly_variable_genes(
        combined,
        n_top_genes=min(n_hvg, combined.shape[1]),
        flavor="seurat_v3",
        batch_key="_batch",
        subset=True,
    )
    log.info(f"After HVG selection: {combined.shape}")

    # Setup and train scVI
    scvi.model.SCVI.setup_anndata(combined, batch_key="_batch", layer=None)
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
    log.info(f"Training scVI for {max_epochs_scvi} epochs...")
    scvi_model.train(
        max_epochs=max_epochs_scvi,
        batch_size=512,
        early_stopping=True,
        early_stopping_patience=10,
        train_size=0.9,
    )

    # Convert to scANVI
    scanvi_model = scvi.model.SCANVI.from_scvi_model(
        scvi_model,
        unlabeled_category="unlabeled",
        labels_key="_label",
    )
    log.info(f"Training scANVI for {max_epochs_scanvi} epochs...")
    scanvi_model.train(
        max_epochs=max_epochs_scanvi,
        batch_size=512,
        n_samples_per_label=20,
    )

    # Predict
    predictions = scanvi_model.predict(combined)
    probs = scanvi_model.predict(combined, soft=True)
    max_probs = probs.max(axis=1)

    # Extract query predictions
    query_mask = combined.obs["_split"] == "query"
    result = pd.DataFrame({
        "barcode": combined.obs_names[query_mask],
        "scanvi_prediction": predictions[query_mask],
        "scanvi_probability": max_probs[query_mask],
    })
    return result


def run_cross_dataset(config, data_dir, output_dir):
    """Run scANVI with separate reference and query datasets."""
    ref_path = os.path.join(data_dir, config["reference"])
    query_path = os.path.join(data_dir, config["query"])

    log.info(f"Loading reference: {ref_path}")
    ref = ad.read_h5ad(ref_path)
    ref = prepare_adata_for_scvi(ref, config["ref_label"], config["ref_batch"])

    log.info(f"Loading query: {query_path}")
    query = ad.read_h5ad(query_path)
    query_true_labels = query.obs[config["query_label"]].copy()
    query = prepare_adata_for_scvi(query, config["query_label"], config["query_batch"])

    results = run_scanvi(ref, query)
    results["true_label"] = query_true_labels.loc[results["barcode"]].values

    return results


def run_leave_one_donor_out(config, data_dir, output_dir):
    """Run scANVI with leave-one-donor-out cross-validation on a single dataset."""
    ref_path = os.path.join(data_dir, config["reference"])
    log.info(f"Loading: {ref_path}")
    adata = ad.read_h5ad(ref_path)
    adata = prepare_adata_for_scvi(adata, config["ref_label"], config["ref_batch"])

    donors = adata.obs["_batch"].unique()
    log.info(f"Donors: {donors.tolist()}")

    all_results = []
    for held_out in donors:
        log.info(f"--- Holding out donor: {held_out} ---")
        ref_mask = adata.obs["_batch"] != held_out
        query_mask = adata.obs["_batch"] == held_out

        if query_mask.sum() < 10:
            log.warning(f"Skipping donor {held_out}: only {query_mask.sum()} cells")
            continue

        ref = adata[ref_mask].copy()
        query = adata[query_mask].copy()
        true_labels = query.obs["_label"].copy()

        results = run_scanvi(ref, query, max_epochs_scvi=50, max_epochs_scanvi=10)
        results["true_label"] = true_labels.loc[results["barcode"]].values
        results["held_out_donor"] = held_out
        all_results.append(results)

    return pd.concat(all_results, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="scANVI benchmark")
    parser.add_argument("--dataset", required=True, help="Dataset name or 'all'")
    parser.add_argument("--data_dir", required=True, help="Directory containing h5ad files")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    datasets = list(DATASET_CONFIG.keys()) if args.dataset == "all" else [args.dataset]

    for ds_name in datasets:
        if ds_name not in DATASET_CONFIG:
            log.error(f"Unknown dataset: {ds_name}")
            continue

        config = DATASET_CONFIG[ds_name]
        log.info(f"===== Running scANVI benchmark: {ds_name} =====")

        try:
            if config["mode"] == "cross_dataset":
                results = run_cross_dataset(config, args.data_dir, args.output_dir)
            else:
                results = run_leave_one_donor_out(config, args.data_dir, args.output_dir)

            out_path = os.path.join(args.output_dir, f"scanvi_{ds_name}_predictions.csv")
            results.to_csv(out_path, index=False)
            log.info(f"Saved: {out_path} ({len(results)} predictions)")

            # Quick accuracy summary
            if "true_label" in results.columns:
                exact = (results["scanvi_prediction"] == results["true_label"]).mean()
                log.info(f"Exact match accuracy: {exact:.3f}")

        except Exception as e:
            log.error(f"Failed on {ds_name}: {e}", exc_info=True)


if __name__ == "__main__":
    main()
