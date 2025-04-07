#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Check the structure of the Schiller_2020.h5ad dataset, especially the gene name information
"""

import scanpy as sc
import pandas as pd
import numpy as np

# Load dataset
print("Loading Schiller 2020 dataset...")
data_path = "/Users/apple/Research/LLMCelltype/data/raw/Schiller_2020.h5ad"
adata = sc.read_h5ad(data_path)

print(f"Dataset shape: {adata.shape[0]} cells, {adata.shape[1]} genes")

# Check var index and columns
print("\n=== Gene information ===")
print(f"var index type: {type(adata.var.index)}")
print(f"var index name: {adata.var.index.name}")
print(f"First 5 genes in var index: {list(adata.var.index[:5])}")

# Check columns in var
print(f"\nColumns in var: {list(adata.var.columns)}")

# If there is a gene_symbols column, display some samples
if 'gene_symbols' in adata.var.columns:
    print(f"\nFirst 5 gene symbols: {list(adata.var['gene_symbols'][:5])}")
    
# If there is a symbol column, display some samples
if 'symbol' in adata.var.columns:
    print(f"\nFirst 5 symbols: {list(adata.var['symbol'][:5])}")

# Check if there are other columns that might contain gene names
for col in adata.var.columns:
    if 'name' in col.lower() or 'symbol' in col.lower() or 'gene' in col.lower():
        print(f"\nPossible gene name column: {col}")
        print(f"First 5 values: {list(adata.var[col][:5])}")

# Check columns in obs (clustering information)
print("\n=== Cell clustering information ===")
print(f"Columns in obs: {list(adata.obs.columns)}")

# Check if there is clustering information
cluster_cols = [col for col in adata.obs.columns if 'cluster' in col.lower() or 'leiden' in col.lower() or 'louvain' in col.lower()]
if cluster_cols:
    for col in cluster_cols:
        print(f"\nCluster column: {col}")
        print(f"Unique values: {adata.obs[col].unique()[:10]}")

# Check keys in the uns dictionary
print("\n=== Keys in uns ===")
print(f"Keys in uns: {list(adata.uns.keys())}")

# If there is rank_genes_groups, check its structure
if 'rank_genes_groups' in adata.uns:
    print("\n=== Rank genes groups structure ===")
    print(f"Keys in rank_genes_groups: {list(adata.uns['rank_genes_groups'].keys())}")
    
    if 'names' in adata.uns['rank_genes_groups']:
        print(f"\nShape of names: {adata.uns['rank_genes_groups']['names'].shape}")
        print(f"dtype.names: {adata.uns['rank_genes_groups']['names'].dtype.names}")
        
        # Display the top 5 genes for the first group
        if adata.uns['rank_genes_groups']['names'].dtype.names:
            first_group = adata.uns['rank_genes_groups']['names'].dtype.names[0]
            print(f"\nTop 5 genes for group {first_group}: {adata.uns['rank_genes_groups']['names'][first_group][:5]}")
