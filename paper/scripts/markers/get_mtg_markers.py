import scanpy as sc
import pandas as pd
import numpy as np

# Read the data
print("Reading MTG.h5ad file...")
adata = sc.read_h5ad("data/raw/MTG.h5ad")

# Remove splatter and miscellaneous cells, and cells with few samples
exclude_clusters = [
    'Splatter',
    'Miscellaneous',
    'Medium spiny neuron',
    'Hippocampal dentate gyrus',
    'Hippocampal CA1-3',
    'Amygdala excitatory',
    'Upper rhombic lip'
]

print("\nRemoving cells from excluded clusters...")
adata = adata[~adata.obs['supercluster_term'].isin(exclude_clusters)]

# Print remaining clusters and their sizes
print("\nRemaining clusters and their sizes:")
print(adata.obs['supercluster_term'].value_counts())

# Preprocess the data
print("\nPreprocessing data...")
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Calculate marker genes for each supercluster
print("\nCalculating marker genes...")
# 使用t-test而不是wilcoxon，因为数据现在是连续的而不是计数的
sc.tl.rank_genes_groups(adata, 'supercluster_term', method='t-test', n_genes=50)

# Function to get top N genes for a group
def get_top_n_genes(adata, group, n=10):
    genes = pd.DataFrame(
        {
            'names': adata.uns['rank_genes_groups']['names'][group],
            'scores': adata.uns['rank_genes_groups']['scores'][group],
            'logfoldchanges': adata.uns['rank_genes_groups']['logfoldchanges'][group],
            'pvals_adj': adata.uns['rank_genes_groups']['pvals_adj'][group]
        }
    )
    # Filter significant genes (adjusted p-value < 0.05) with positive log fold change
    genes = genes[
        (genes['pvals_adj'] < 0.05) &
        (genes['logfoldchanges'] > 0)
    ]
    return genes.nlargest(n, 'scores')['names'].tolist()

# Get markers for each supercluster
print("\nExtracting top markers...")
markers_dict = {}
for cluster in adata.obs['supercluster_term'].unique():
    if cluster not in exclude_clusters:
        top_genes = get_top_n_genes(adata, cluster)
        if len(top_genes) < 10:  # 如果找不到足够的显著性基因
            print(f"Warning: Only found {len(top_genes)} significant marker genes for {cluster}")
            # 直接使用得分最高的前10个基因
            genes = pd.DataFrame(
                {
                    'names': adata.uns['rank_genes_groups']['names'][cluster],
                    'scores': adata.uns['rank_genes_groups']['scores'][cluster]
                }
            )
            top_genes = genes.nlargest(10, 'scores')['names'].tolist()
        markers_dict[cluster] = top_genes

# Convert gene IDs to symbols if available
print("\nConverting gene IDs to symbols...")
gene_symbols = pd.Series(adata.var['Gene'].values, index=adata.var_names)
for cluster in markers_dict:
    markers_dict[cluster] = [gene_symbols[gene] for gene in markers_dict[cluster]]

# Convert to desired format
output_data = []
for cluster, genes in markers_dict.items():
    output_data.append([cluster] + genes)

# Create DataFrame and save to CSV
df = pd.DataFrame(output_data)
df.to_csv("data/reference/MTG_markers.csv",
          header=False, index=False)

print("Markers saved to MTG_markers.csv")
