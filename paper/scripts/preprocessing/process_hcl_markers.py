import pandas as pd
import numpy as np

# Read the Excel file
excel_path = 'data/raw/HCL/cluster_markers_HCL&MCA1.1.xlsx'
df = pd.read_excel(excel_path, sheet_name=0)

# Remove any completely empty rows and columns
df = df.dropna(how='all', axis=0)
df = df.dropna(how='all', axis=1)

# Get cell types from the first row
cell_types = []
genes_dict = {}
genes_scores_dict = {}  # 存储基因得分

# Process each group of 4 columns
for i in range(1, len(df.columns), 4):
    if pd.notna(df.iloc[0, i]):  # Check if cell type exists
        cell_type = df.iloc[0, i]
        genes = []

        # Get genes and their scores from this group
        for j in range(2, len(df)):  # Start from row 2 (0-based index)
            if pd.notna(df.iloc[j, i]):  # Check if gene exists
                gene_name = df.iloc[j, i]
                score = float(df.iloc[j, i+2]) if pd.notna(df.iloc[j, i+2]) else 0
                genes.append((gene_name, score))

        # Sort genes by score and get top 10
        genes.sort(key=lambda x: x[1], reverse=True)
        top_genes = [g[0] for g in genes[:10]]
        top_genes_with_scores = genes[:10]  # 保存带分数的版本

        if len(top_genes) > 0:
            genes_dict[cell_type] = top_genes
            genes_scores_dict[cell_type] = top_genes_with_scores

# Create the output string
output_lines = ['cell_type,gene']
for cell_type, genes in genes_dict.items():
    genes_str = ','.join([cell_type] + genes)
    output_lines.append(genes_str)

# Write to CSV file
output_path = 'data/reference/HCL_markers.csv'
with open(output_path, 'w') as f:
    f.write('\n'.join(output_lines))

# Print summary with scores
print(f"Processed {len(genes_dict)} cell types")
print("\nFirst 10 cell types with their top 10 genes and scores:")
print("-" * 80)
for i, (cell_type, genes_scores) in enumerate(genes_scores_dict.items()):
    if i < 10:  # Show first 10 cell types
        print(f"\n{cell_type}:")
        for gene, score in genes_scores:
            print(f"{gene}: {score:.2f}")
        print("-" * 40)
