#!/usr/bin/env python3
"""
Weighted KNN Label Transfer for Cell Type Annotation

This module provides weighted K-nearest neighbors (KNN) based label transfer
for single-cell RNA-seq cell type annotation. It trains a KNN classifier on
reference data and transfers labels to query data.

Reference: Adapted from HLCA reproducibility pipeline
https://github.com/LungCellAtlas/HLCA

Usage:
    python label_transfer.py --ref reference.h5ad --query query.h5ad --output predictions.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.neighbors import KNeighborsTransformer


def weighted_knn_trainer(train_adata, train_adata_emb, label_key, n_neighbors=50):
    """
    Train a weighted KNN classifier on reference data.

    Parameters
    ----------
    train_adata : anndata.AnnData
        Annotated dataset to be used to train KNN classifier
    train_adata_emb : str
        Name of the obsm layer to be used for calculation of neighbors.
        If set to "X", anndata.X will be used
    label_key : str
        Name of the column to be used as target variable (e.g. cell_type)
    n_neighbors : int
        Number of nearest neighbors in KNN classifier (default: 50)

    Returns
    -------
    sklearn.neighbors.KNeighborsTransformer
        Trained KNN transformer
    """
    print(f"Training weighted KNN with n_neighbors = {n_neighbors}...", end=" ")

    k_neighbors_transformer = KNeighborsTransformer(
        n_neighbors=n_neighbors,
        mode="distance",
        algorithm="brute",
        metric="euclidean",
        n_jobs=-1,
    )

    if train_adata_emb == "X":
        train_emb = train_adata.X
    elif train_adata_emb in train_adata.obsm:
        train_emb = train_adata.obsm[train_adata_emb]
    else:
        raise TypeError(
            f"train_adata_emb should be 'X' or a valid obsm key. "
            f"Available keys: {list(train_adata.obsm.keys())}"
        )

    k_neighbors_transformer.fit(train_emb)
    print("Done!")
    return k_neighbors_transformer


def weighted_knn_transfer(
    query_adata,
    query_adata_emb,
    ref_adata_obs,
    label_keys,
    knn_model,
    threshold=1,
    pred_unknown=False,
    mode="package",
):
    """
    Transfer labels from reference to query using weighted KNN.

    Parameters
    ----------
    query_adata : anndata.AnnData
        Query dataset to annotate
    query_adata_emb : str
        Name of the obsm layer to be used for label transfer.
        If set to "X", query_adata.X will be used
    ref_adata_obs : pd.DataFrame
        obs DataFrame from reference AnnData
    label_keys : str
        Prefix of column names to be used as target variables
    knn_model : sklearn.neighbors.KNeighborsTransformer
        KNN model trained with weighted_knn_trainer
    threshold : float
        Threshold of uncertainty for "Unknown" annotation (default: 1).
        Cells with uncertainties higher than this value will be "Unknown".
        Set to 1 to keep all predictions.
    pred_unknown : bool
        Whether to annotate uncertain cells as "Unknown" (default: False)
    mode : str
        Uncertainty calculation mode ("package" or "paper")

    Returns
    -------
    pred_labels : pd.DataFrame
        Predicted labels for each cell
    uncertainties : pd.DataFrame
        Uncertainty scores for each prediction
    """
    if not isinstance(knn_model, KNeighborsTransformer):
        raise TypeError(
            "knn_model should be of type sklearn.neighbors.KNeighborsTransformer!"
        )

    if query_adata_emb == "X":
        query_emb = query_adata.X
    elif query_adata_emb in query_adata.obsm:
        query_emb = query_adata.obsm[query_adata_emb]
    else:
        raise ValueError(
            f"query_adata_emb should be 'X' or a valid obsm key. "
            f"Available keys: {list(query_adata.obsm.keys())}"
        )

    print("Transferring labels...", end=" ")
    top_k_distances, top_k_indices = knn_model.kneighbors(X=query_emb)

    # Calculate weights based on distances
    stds = np.std(top_k_distances, axis=1)
    stds = (2.0 / stds) ** 2
    stds = stds.reshape(-1, 1)

    top_k_distances_tilda = np.exp(-np.true_divide(top_k_distances, stds))

    weights = top_k_distances_tilda / np.sum(
        top_k_distances_tilda, axis=1, keepdims=True
    )

    # Get matching columns
    cols = ref_adata_obs.columns[ref_adata_obs.columns.str.startswith(label_keys)]
    uncertainties = pd.DataFrame(columns=cols, index=query_adata.obs_names)
    pred_labels = pd.DataFrame(columns=cols, index=query_adata.obs_names)

    for i in range(len(weights)):
        for j in cols:
            y_train_labels = ref_adata_obs[j].values
            unique_labels = np.unique(y_train_labels[top_k_indices[i]])
            best_label, best_prob = None, 0.0

            for candidate_label in unique_labels:
                candidate_prob = weights[
                    i, y_train_labels[top_k_indices[i]] == candidate_label
                ].sum()
                if best_prob < candidate_prob:
                    best_prob = candidate_prob
                    best_label = candidate_label

            if pred_unknown:
                if best_prob >= threshold:
                    pred_label = best_label
                else:
                    pred_label = "Unknown"
            else:
                pred_label = best_label

            if mode == "package":
                uncertainties.iloc[i][j] = max(1 - best_prob, 0)
            else:
                raise ValueError(f"Unknown mode: {mode}. Use 'package'.")

            pred_labels.iloc[i][j] = pred_label

    print("Done!")
    return pred_labels, uncertainties


def transfer_labels(
    ref_path,
    query_path,
    output_path,
    label_col="cell_type",
    embedding="X_pca",
    n_neighbors=50,
    threshold=1.0,
    pred_unknown=False,
):
    """
    Complete label transfer pipeline.

    Parameters
    ----------
    ref_path : str
        Path to reference h5ad file
    query_path : str
        Path to query h5ad file
    output_path : str
        Path to save predictions
    label_col : str
        Column name for cell type labels in reference
    embedding : str
        Embedding to use ("X" or obsm key like "X_pca", "X_scvi")
    n_neighbors : int
        Number of neighbors for KNN
    threshold : float
        Uncertainty threshold for "Unknown" predictions
    pred_unknown : bool
        Whether to predict "Unknown" for uncertain cells
    """
    print(f"Loading reference: {ref_path}")
    ref = sc.read_h5ad(ref_path)
    print(f"Reference shape: {ref.n_obs} cells x {ref.n_vars} genes")

    print(f"Loading query: {query_path}")
    query = sc.read_h5ad(query_path)
    print(f"Query shape: {query.n_obs} cells x {query.n_vars} genes")

    # Check embedding
    if embedding != "X" and embedding not in ref.obsm:
        raise ValueError(
            f"Embedding '{embedding}' not found in reference. "
            f"Available: {list(ref.obsm.keys())}"
        )

    # Train KNN
    knn_model = weighted_knn_trainer(
        train_adata=ref,
        train_adata_emb=embedding,
        label_key=label_col,
        n_neighbors=n_neighbors,
    )

    # Transfer labels
    labels, uncertainties = weighted_knn_transfer(
        query_adata=query,
        query_adata_emb=embedding,
        label_keys=label_col,
        knn_model=knn_model,
        ref_adata_obs=ref.obs,
        threshold=threshold,
        pred_unknown=pred_unknown,
    )

    # Save results
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    labels_path = output_path
    uncert_path = output_path.replace(".csv", "_uncertainty.csv")

    labels.to_csv(labels_path)
    uncertainties.to_csv(uncert_path)

    print(f"\nSaved predictions: {labels_path}")
    print(f"Saved uncertainties: {uncert_path}")

    # Summary
    print("\n=== Summary ===")
    for col in labels.columns:
        print(f"\n{col}:")
        print(labels[col].value_counts().head(10))

    return labels, uncertainties


def main():
    parser = argparse.ArgumentParser(
        description="Weighted KNN label transfer for cell type annotation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python label_transfer.py --ref reference.h5ad --query query.h5ad --output predictions.csv

    # With custom parameters
    python label_transfer.py --ref ref.h5ad --query query.h5ad --output pred.csv \\
        --label-col cell_ontology_class --embedding X_scvi --n-neighbors 30

    # Predict "Unknown" for uncertain cells
    python label_transfer.py --ref ref.h5ad --query query.h5ad --output pred.csv \\
        --pred-unknown --threshold 0.8

Requirements:
    pip install scanpy scikit-learn pandas numpy
        """,
    )
    parser.add_argument(
        "--ref",
        "-r",
        required=True,
        help="Path to reference h5ad file with cell type labels",
    )
    parser.add_argument(
        "--query", "-q", required=True, help="Path to query h5ad file to annotate"
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Path to save predictions (CSV)"
    )
    parser.add_argument(
        "--label-col",
        default="cell_type",
        help="Column name for cell type labels in reference (default: cell_type)",
    )
    parser.add_argument(
        "--embedding",
        default="X",
        help="Embedding to use: 'X' or obsm key like 'X_pca' (default: X)",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=50,
        help="Number of neighbors for KNN (default: 50)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Uncertainty threshold for 'Unknown' predictions (default: 1.0)",
    )
    parser.add_argument(
        "--pred-unknown",
        action="store_true",
        help="Predict 'Unknown' for cells above uncertainty threshold",
    )

    args = parser.parse_args()

    transfer_labels(
        ref_path=args.ref,
        query_path=args.query,
        output_path=args.output,
        label_col=args.label_col,
        embedding=args.embedding,
        n_neighbors=args.n_neighbors,
        threshold=args.threshold,
        pred_unknown=args.pred_unknown,
    )


if __name__ == "__main__":
    main()
