import scanpy as sc
import pandas as pd

# Read the h5ad file
print("Reading MTG.h5ad file...")
adata = sc.read_h5ad("data/raw/MTG.h5ad")

# Print basic information
print("\nBasic information:")
print(adata)

# Check available annotations in obs
print("\nAvailable annotations in obs:")
print(adata.obs.columns.tolist())

# Check if there are any celltype annotations
if 'celltype' in adata.obs.columns:
    print("\nUnique celltypes:")
    print(adata.obs['celltype'].value_counts())

# Check if there are cluster annotations
cluster_cols = [col for col in adata.obs.columns if 'cluster' in col.lower()]
if cluster_cols:
    print("\nCluster related columns:")
    for col in cluster_cols:
        print(f"\n{col} unique values:")
        print(adata.obs[col].value_counts())

# Check for marker genes in var annotations
print("\nVariable annotations available:")
print(adata.var.columns.tolist())

# Check if there are any marker gene related annotations
marker_cols = [col for col in adata.var.columns if any(term in col.lower() for term in ['marker', 'score', 'gene'])]
if marker_cols:
    print("\nMarker gene related columns:")
    for col in marker_cols:
        print(f"\n{col}")
        print(adata.var[col].head())

# Check if there are any marker genes stored in uns
print("\nUnstructured annotations (uns) keys:")
print(list(adata.uns.keys()))
