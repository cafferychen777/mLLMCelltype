#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
检查Schiller_2020.h5ad数据集的结构，特别是基因名称信息
"""

import scanpy as sc
import pandas as pd
import numpy as np

# 加载数据集
print("Loading Schiller 2020 dataset...")
data_path = "/Users/apple/Research/LLMCelltype/data/raw/Schiller_2020.h5ad"
adata = sc.read_h5ad(data_path)

print(f"Dataset shape: {adata.shape[0]} cells, {adata.shape[1]} genes")

# 检查var索引和列
print("\n=== Gene information ===")
print(f"var index type: {type(adata.var.index)}")
print(f"var index name: {adata.var.index.name}")
print(f"First 5 genes in var index: {list(adata.var.index[:5])}")

# 检查var中的列
print(f"\nColumns in var: {list(adata.var.columns)}")

# 如果有gene_symbols列，显示一些样本
if 'gene_symbols' in adata.var.columns:
    print(f"\nFirst 5 gene symbols: {list(adata.var['gene_symbols'][:5])}")
    
# 如果有symbol列，显示一些样本
if 'symbol' in adata.var.columns:
    print(f"\nFirst 5 symbols: {list(adata.var['symbol'][:5])}")

# 检查是否有其他可能包含基因名称的列
for col in adata.var.columns:
    if 'name' in col.lower() or 'symbol' in col.lower() or 'gene' in col.lower():
        print(f"\nPossible gene name column: {col}")
        print(f"First 5 values: {list(adata.var[col][:5])}")

# 检查obs中的列（聚类信息）
print("\n=== Cell clustering information ===")
print(f"Columns in obs: {list(adata.obs.columns)}")

# 检查是否有聚类信息
cluster_cols = [col for col in adata.obs.columns if 'cluster' in col.lower() or 'leiden' in col.lower() or 'louvain' in col.lower()]
if cluster_cols:
    for col in cluster_cols:
        print(f"\nCluster column: {col}")
        print(f"Unique values: {adata.obs[col].unique()[:10]}")

# 检查uns字典中的键
print("\n=== Keys in uns ===")
print(f"Keys in uns: {list(adata.uns.keys())}")

# 如果有rank_genes_groups，检查其结构
if 'rank_genes_groups' in adata.uns:
    print("\n=== Rank genes groups structure ===")
    print(f"Keys in rank_genes_groups: {list(adata.uns['rank_genes_groups'].keys())}")
    
    if 'names' in adata.uns['rank_genes_groups']:
        print(f"\nShape of names: {adata.uns['rank_genes_groups']['names'].shape}")
        print(f"dtype.names: {adata.uns['rank_genes_groups']['names'].dtype.names}")
        
        # 显示第一个组的前5个基因
        if adata.uns['rank_genes_groups']['names'].dtype.names:
            first_group = adata.uns['rank_genes_groups']['names'].dtype.names[0]
            print(f"\nTop 5 genes for group {first_group}: {adata.uns['rank_genes_groups']['names'][first_group][:5]}")
