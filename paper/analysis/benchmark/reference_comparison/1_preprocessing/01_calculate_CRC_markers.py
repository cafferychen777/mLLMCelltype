import scanpy as sc
import pandas as pd
import os

# 读取原始表达矩阵以获取基因名称
print("Reading expression matrix for gene names...")
expr_matrix = pd.read_csv('data/processed/GSE132465_GEO_processed_CRC_10X_natural_log_TPM_matrix.txt',
                         sep='\t', usecols=['Index'])
gene_names = expr_matrix['Index'].values

# 读取AnnData对象
print("\nReading AnnData object...")
adata = sc.read_h5ad('data/raw/CRC_tumor.h5ad')

# 将基因名称添加到adata中
adata.var_names = gene_names
adata.var['gene_symbols'] = gene_names

# 计算marker genes
print("\nCalculating marker genes for each cell subtype...")
sc.tl.rank_genes_groups(adata,
                       groupby='Cell_subtype',
                       method='wilcoxon',
                       key_added="cellsubtype_markers",
                       n_genes=50)

# 创建一个函数来格式化marker genes为指定格式
def format_markers_for_csv(adata, key="cellsubtype_markers"):
    results = []
    cell_subtypes = adata.obs['Cell_subtype'].unique()

    # 添加表头
    results.append(['cluster', 'gene'])

    for cell_subtype in cell_subtypes:
        # 获取这个cell subtype的top genes
        genes = sc.get.rank_genes_groups_df(adata, group=cell_subtype, key=key)

        # 只保留显著的genes (adjusted p-value < 0.05)
        significant_genes = genes[genes['pvals_adj'] < 0.05]

        # 按logfoldchange排序
        significant_genes = significant_genes.sort_values('logfoldchanges', ascending=False)

        # 获取top 10 genes
        top_genes = significant_genes['names'].head(10).tolist()

        # 如果有显著的marker genes
        if top_genes:
            # 将cell subtype和genes组合成一行
            results.append([cell_subtype, ','.join(top_genes)])

    return results

# 格式化结果
print("\nFormatting results...")
marker_results = format_markers_for_csv(adata)

# 保存结果
output_file = 'data/reference/CRC_markers.csv'
print(f"\nSaving results to {output_file}")

# 写入CSV文件
with open(output_file, 'w') as f:
    for row in marker_results:
        f.write(f"{row[0]},{row[1]}\n")

print("\nResults preview:")
with open(output_file, 'r') as f:
    print(f.read())

print("\nAnalysis complete!")
