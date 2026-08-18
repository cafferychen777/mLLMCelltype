import popv
import anndata
import scanpy as sc
import os
import joblib
import numpy as np
import scipy.sparse as scp
import pandas as pd
from popv.algorithms import Support_Vector, XGboost, ONCLASS, CELLTYPIST

# Set output directory for retrain mode
output_dir = "results/benchmark/popv_comparison/2_evaluation"
os.makedirs(output_dir, exist_ok=True)

print("Loading processed data...")
query_adata = anndata.read_h5ad("data/processed/Thymus_processed.h5ad")

# Save original celltype if exists
if 'cell_type' in query_adata.obs:
    original_celltype = query_adata.obs['cell_type'].copy()
else:
    original_celltype = pd.Series('Unknown', index=query_adata.obs.index)

# Ensure data is non-negative integers
X = query_adata.X.copy()
if isinstance(X, scp.spmatrix):
    X.data = np.round(X.data).astype(int)
    X.data[X.data < 0] = 0
else:
    X = np.round(X).astype(int)
    X[X < 0] = 0

# Save raw count data
query_adata.layers["raw_counts"] = X

# Basic preprocessing
print("\nPerforming basic preprocessing...")
sc.pp.normalize_total(query_adata)
sc.pp.log1p(query_adata)

# Ensure Sample column exists
if 'Sample' not in query_adata.obs:
    query_adata.obs['Sample'] = 'sample1'

# Load minified reference data
print("\nLoading minified reference data...")
ref_adata = anndata.read_h5ad("data/thymus/minified_ref_adata.h5ad")

# Print reference data information
print("\nReference data information:")
print("Available columns in ref_adata.obs:", list(ref_adata.obs.columns))
print("Available keys in ref_adata.uns:", list(ref_adata.uns.keys()))

print("\nInitializing PopV processor with retrain mode...")
processor = popv.preprocessing.Process_Query(
    query_adata=query_adata,
    ref_adata=ref_adata,  # Use minified reference data
    ref_labels_key='cell_type',  # Use cell_type as label
    ref_batch_key='Sample',  # Use Sample as batch key
    prediction_mode='retrain',  # Retrain mode for complete model retraining
    pretrained_scvi_path='data/thymus/scvi',  # Path to pretrained model
    cl_obo_folder='data/thymus/ontology',  # Path to cell ontology folder
    save_path_trained_models='data/thymus',  # Directory for pretrained models
    query_batch_key='Sample',  # Use Sample as query batch key
    query_layer_key='raw_counts',  # Use raw_counts as input data
    hvg=None  # Disable highly variable gene selection
)

# Add necessary information to adata.uns
processor.adata.uns["unknown_celltype_label"] = "unknown"
processor.adata.uns["label_categories"] = np.array(list(ref_adata.obs['cell_type'].unique()) + ["unknown"])

print("\nRunning annotation...")
# Use all available algorithms for annotation
methods = [
    "Support_Vector",  # Support Vector Machine classifier
    "XGboost",  # XGBoost classifier
    "ONCLASS",  # OnClass classifier
    "CELLTYPIST"  # CellTypist classifier
]
popv.annotation.annotate_data(
    adata=processor.adata,
    methods=methods,  # Use specified algorithms
    save_path=output_dir
)

print("\nSaving results...")
# Save results
processor.adata.write_h5ad(os.path.join(output_dir, "Thymus_popv_retrain.h5ad"))

# Extract and save prediction results
predictions = processor.adata.obs[[col for col in processor.adata.obs.columns if 'prediction' in col]]
# Add original celltype
predictions['original_celltype'] = original_celltype
predictions.to_csv(os.path.join(output_dir, "Thymus_popv_retrain_results.csv"))

# Print basic statistics
print("\nAnnotation Statistics:")
print("\nPopV Majority Vote Prediction Distribution:")
print(processor.adata.obs['popv_majority_vote_prediction'].value_counts())

print("\nPrediction Score Statistics:")
print(processor.adata.obs['popv_majority_vote_score'].describe())

# Print original celltype distribution if exists
if 'cell_type' in query_adata.obs:
    print("\nOriginal Cell Type Distribution:")
    print(original_celltype.value_counts())

print(f"\nResults saved to: {output_dir}")
print("- Full annotated data: Thymus_popv_retrain.h5ad")
print("- Prediction results: Thymus_popv_retrain_results.csv")
