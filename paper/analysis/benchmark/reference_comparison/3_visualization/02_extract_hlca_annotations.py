import scanpy as sc
import pandas as pd
import os

# Set project root
project_root = "."

# Read the h5ad file
print("Reading h5ad file...")
adata = sc.read_h5ad(os.path.join(project_root, "data/raw/HLCA_Core.h5ad"))

# Create a DataFrame with leiden_4 clusters and their annotations
print("Creating annotation table...")

# First, get the most common annotation for each cluster at each level
cluster_annotations = {}
ann_columns = [col for col in adata.obs.columns if col.startswith('ann_')]

for col in ann_columns:
    # Group by leiden_4 and get the most common annotation
    cluster_annotations[col] = adata.obs.groupby('leiden_4')[col].agg(
        lambda x: pd.Series.mode(x)[0] if len(pd.Series.mode(x)) == 1 else 'Mixed'
    )

# Combine all annotations into one DataFrame
result_df = pd.DataFrame(cluster_annotations)

# Add cell count
cell_counts = adata.obs.groupby('leiden_4').size()
result_df['cell_count'] = cell_counts

# Add leiden_4 as a column from the index
result_df = result_df.reset_index()
result_df.columns = ['leiden_4'] + list(result_df.columns[1:])

# Reorder columns to put leiden_4 and cell_count first
cols = ['leiden_4', 'cell_count'] + [col for col in result_df.columns if col not in ['leiden_4', 'cell_count']]
result_df = result_df[cols]

# Create output directory if it doesn't exist
output_dir = os.path.join(project_root, "results/benchmark/popv_comparison/3_visualization")
os.makedirs(output_dir, exist_ok=True)

# Save to CSV
output_file = os.path.join(output_dir, "HLCA_leiden_4_all_level_annotations.csv")
result_df.to_csv(output_file, index=False)

print(f"\nResults saved to {output_file}")
print("\nPreview of the results (first 5 rows):")
pd.set_option('display.max_columns', None)
print(result_df.head())

print("\nDataFrame shape:", result_df.shape)
