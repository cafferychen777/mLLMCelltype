import scanpy as sc
import pandas as pd
import os

# Read the data
print("Loading the data...")
adata = sc.read_h5ad('data/raw/LCA.h5ad')

# Create a copy of the AnnData object for marker gene calculation
adata_markers = adata.copy()

# Clean cell type names by removing ", human" suffix
adata_markers.obs['cell_type'] = adata_markers.obs['cell_type'].str.replace(', human', '')

# Create a mapping from Ensembl ID to gene name
gene_name_dict = pd.Series(adata.var['feature_name'].values, index=adata.var_names).to_dict()

# Calculate marker genes for each cell type using original Ensembl IDs
print("\nCalculating marker genes...")
sc.tl.rank_genes_groups(adata_markers, 'cell_type', method='wilcoxon', n_genes=10)

# Create a list to store the formatted strings
formatted_lines = ['cluster,gene']
cell_types = adata_markers.obs['cell_type'].unique()

for cell_type in cell_types:
    # Get the top 10 marker genes for this cell type and convert to gene names
    genes = [gene_name_dict[gene] for gene in
             adata_markers.uns['rank_genes_groups']['names'][cell_type]]
    # Format the line as: cell_type,gene1,gene2,gene3,...
    formatted_line = f"{cell_type},{','.join(genes)}"
    formatted_lines.append(formatted_line)

# Create output directory if it doesn't exist
output_dir = 'results/benchmark/popv_comparison/1_preprocessing'
os.makedirs(output_dir, exist_ok=True)

# Save to CSV
output_file = os.path.join(output_dir, 'LCA_markers.csv')
with open(output_file, 'w') as f:
    f.write('\n'.join(formatted_lines))

print(f"\nMarker genes saved to: {output_file}")
print("\nFirst few entries of the marker genes:")
print('\n'.join(formatted_lines[:4]))

print(f"\nNumber of cell types: {len(cell_types)}")
print(f"Total number of marker genes: {len(cell_types) * 10}")
