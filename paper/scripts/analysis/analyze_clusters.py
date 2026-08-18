import scanpy as sc
import pandas as pd
import numpy as np

# Read the h5ad file
print("Reading MTG.h5ad file...")
adata = sc.read_h5ad("data/raw/MTG.h5ad")

# 1. 分析每个supercluster中包含多少个cluster
print("\n1. 每个supercluster包含的cluster数量:")
cluster_in_super = adata.obs.groupby('supercluster_term')['cluster_id'].nunique().sort_values(ascending=False)
print(cluster_in_super)

# 2. 显示每个supercluster中最主要的几个cluster及其细胞数量
print("\n2. 每个supercluster中的主要clusters及其细胞数量:")
for super_cluster in adata.obs['supercluster_term'].unique():
    print(f"\n{super_cluster}:")
    cluster_counts = adata.obs[adata.obs['supercluster_term'] == super_cluster]['cluster_id'].value_counts().head(3)
    print(cluster_counts)

# 3. 计算每个cluster的组成（属于哪些supercluster）
print("\n3. 检查cluster的纯度（每个cluster属于哪些supercluster）:")
for cluster in adata.obs['cluster_id'].unique()[:5]:  # 只显示前5个cluster作为示例
    print(f"\nCluster {cluster}:")
    super_dist = adata.obs[adata.obs['cluster_id'] == cluster]['supercluster_term'].value_counts()
    total = super_dist.sum()
    percentages = (super_dist / total * 100).round(2)
    print(percentages)
