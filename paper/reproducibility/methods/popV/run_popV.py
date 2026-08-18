# popV Cell Type Annotation Script
# Reference: Yosef Lab, Nature Genetics 2024
# GitHub: https://github.com/YosefLab/popV
# Reproducibility: https://github.com/YosefLab/popv-reproducibility

# =============================================================================
# Installation
# =============================================================================
# pip install popv
# or: pip install git+https://github.com/YosefLab/PopV

# =============================================================================
# Imports
# =============================================================================
import popv
import numpy as np
import scanpy as sc
import os
import requests

# =============================================================================
# Configuration - Modify these parameters
# =============================================================================

# Input: Path to your query dataset (.h5ad file)
QUERY_DATA_PATH = "path/to/your/query_data.h5ad"  # <-- Modify this

# Output folder
OUTPUT_FOLDER = "path/to/output"  # <-- Modify this

# Tissue type for reference selection
# Available options (from Tabula Sapiens):
#   - "Lung"
#   - "Heart"
#   - "Kidney"
#   - "Liver"
#   - "Pancreas"
#   - "Spleen"
#   - "Thymus"
#   - "Bladder"
#   - "Blood"
#   - "Bone_Marrow"
#   - "Eye"
#   - "Fat"
#   - "Large_Intestine"
#   - "Lymph_Node"
#   - "Mammary"
#   - "Muscle"
#   - "Prostate"
#   - "Salivary_Gland"
#   - "Skin"
#   - "Small_Intestine"
#   - "Tongue"
#   - "Trachea"
#   - "Uterus"
#   - "Vasculature"

TISSUE = "Lung"  # <-- Modify this

# Batch key in query data (set to None if no batch info)
QUERY_BATCH_KEY = "batch"  # <-- Modify this

# Prediction mode: "retrain", "inference", or "fast"
#   - "retrain": Train all classifiers from scratch (~1 hour with GPU)
#   - "inference": Use pretrained models (~30 min with GPU)
#   - "fast": Only methods with pretrained classifiers (~5 min, no GPU needed)
PREDICTION_MODE = "fast"  # <-- Modify this

# Use GPU acceleration
USE_GPU = False  # <-- Modify this

# =============================================================================
# Create output directories
# =============================================================================
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(f"{OUTPUT_FOLDER}/pretrained_models", exist_ok=True)

# =============================================================================
# Load query data
# =============================================================================
print(f"Loading query data from: {QUERY_DATA_PATH}")
query_adata = sc.read_h5ad(QUERY_DATA_PATH)
query_adata.obs_names_make_unique()
query_adata.var_names = query_adata.var_names.str.upper()

print(f"Query data: {query_adata.n_obs} cells, {query_adata.n_vars} genes")

# =============================================================================
# Download reference data from Zenodo
# =============================================================================
print(f"Fetching reference data for tissue: {TISSUE}")

res = requests.get("https://zenodo.org/api/records/7580683")
tissue_download_path = {
    ind["key"][3:-14]: ind["links"]["self"] for ind in res.json()["files"]
}

if TISSUE not in tissue_download_path:
    raise ValueError(f"Tissue '{TISSUE}' not found. Available: {list(tissue_download_path.keys())}")

refdata_url = tissue_download_path[TISSUE]
ref_file = f"{OUTPUT_FOLDER}/TS_{TISSUE}.h5ad"

if not os.path.exists(ref_file):
    print(f"Downloading reference to: {ref_file}")
    response = requests.get(refdata_url)
    with open(ref_file, "wb") as f:
        f.write(response.content)
else:
    print(f"Reference file already exists: {ref_file}")

# Load reference
ref_adata = sc.read_h5ad(ref_file)
ref_labels_key = "cell_ontology_class"
ref_batch_key = "donor_assay"

print(f"Reference data: {ref_adata.n_obs} cells, {ref_adata.n_vars} genes")

# =============================================================================
# Process query data
# =============================================================================
print("Processing query data...")

from popv.preprocessing import Process_Query

min_celltype_size = np.min(ref_adata.obs.groupby(ref_labels_key).size())
n_samples_per_label = np.max((min_celltype_size, 500))

adata = Process_Query(
    query_adata,
    ref_adata,
    query_labels_key=None,
    query_batch_key=QUERY_BATCH_KEY,
    ref_labels_key=ref_labels_key,
    ref_batch_key=ref_batch_key,
    unknown_celltype_label="unknown",
    save_path_trained_models=f"{OUTPUT_FOLDER}/pretrained_models/",
    prediction_mode=PREDICTION_MODE,
    n_samples_per_label=n_samples_per_label,
    use_gpu=USE_GPU,
    compute_embedding=True,
    hvg=4000,
).adata

# =============================================================================
# Run annotation
# =============================================================================
print("Running popV annotation...")

from popv.annotation import annotate_data

annotate_data(adata, save_path=f"{OUTPUT_FOLDER}/popv_output")

# =============================================================================
# Results summary
# =============================================================================
print("\n=== Annotation Results ===")
print(adata.obs["popv_prediction"].value_counts())

print("\n=== Confidence Scores ===")
print(f"Mean score: {adata.obs['popv_prediction_score'].mean():.2f}")
print(f"Min score: {adata.obs['popv_prediction_score'].min():.2f}")
print(f"Max score: {adata.obs['popv_prediction_score'].max():.2f}")

# =============================================================================
# Save results
# =============================================================================
output_file = f"{OUTPUT_FOLDER}/query_annotated_popv.h5ad"
print(f"\nSaving annotated data to: {output_file}")
adata.write(output_file)

print("\nDone!")
