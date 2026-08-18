import scanpy as sc
import pandas as pd
import os
import numpy as np

# Read the data
print("Loading data...")
adata = sc.read_h5ad('data/raw/Thymus.h5ad')

# Check data quality
print("\nData quality check:")
print(f"Total cells: {adata.n_obs}")
print(f"Total genes: {adata.n_vars}")
print(f"Sparsity: {(1.0 - adata.X.nnz/(adata.n_obs*adata.n_vars))*100:.2f}%")

# Verify gene names
print("\nVerifying gene names...")
n_duplicated = adata.var_names.duplicated().sum()
if n_duplicated > 0:
    print(f"Found {n_duplicated} duplicated gene names. Making them unique...")
    adata.var_names_make_unique()

# Perform rank_genes_groups analysis
print("\nPerforming differential expression analysis...")
sc.tl.rank_genes_groups(adata, 'cell_type', method='wilcoxon')

# Create a dictionary to store results
markers_dict = {}

# Extract top 10 markers for each cell type
print("\nExtracting top markers...")
cell_types = adata.obs['cell_type'].unique()

print("\nProcessing each cell type:")
for cell_type in cell_types:
    try:
        # Get the names and scores for this group
        names = adata.uns['rank_genes_groups']['names'][cell_type][:10]
        scores = adata.uns['rank_genes_groups']['scores'][cell_type][:10]

        print(f"✓ {cell_type}:")
        for name, score in zip(names, scores):
            print(f"  - {name} (score: {score:.2f})")

        markers_dict[cell_type] = names.tolist()
    except Exception as e:
        print(f"✗ Error processing {cell_type}: {str(e)}")
        continue

# Convert to the required format
print("\nConverting to required format...")
rows = []
for cell_type, genes in markers_dict.items():
    rows.append({'cluster': cell_type, 'gene': ','.join(genes)})

# Create DataFrame
df = pd.DataFrame(rows)

# Ensure output directory exists
os.makedirs('data/reference', exist_ok=True)

# Save to CSV
output_path = 'data/reference/Thymus_markers.csv'
print(f"\nSaving to {output_path}...")
df.to_csv(output_path, index=False)

print("\nDone! Here's a preview of the markers:")
print(df.head())
