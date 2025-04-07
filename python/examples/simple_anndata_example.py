#!/usr/bin/env python3
"""
Simplified example using LLMCelltype with AnnData.
"""

import os
import sys
import pandas as pd
import numpy as np
import scanpy as sc
from dotenv import load_dotenv
import argparse

# Add parent directory to path for local development
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import llmcelltype as lct

# Load API keys from .env file
load_dotenv()

def examine_anndata():
    """Examine the structure of the AnnData file without full processing"""
    print("Loading AnnData file...")
    adata = sc.read_h5ad('/Users/apple/Research/LLMCelltype/data/raw/MTG.h5ad')
    
    print("\nAnnData summary:")
    print(f"  Shape: {adata.shape}")
    print(f"  Number of genes: {adata.var.shape[0]}")
    
    print("\nObservations (cells) info:")
    print(f"  Number of cells: {adata.n_obs}")
    print("\nColumns in adata.obs:")
    for col in adata.obs.columns:
        print(f"  - {col}")
        
        # Check if column has categorical values
        if pd.api.types.is_categorical_dtype(adata.obs[col]):
            categories = adata.obs[col].cat.categories
            counts = adata.obs[col].value_counts()
            print(f"    Categorical with {len(categories)} values")
            if len(categories) < 20:
                for cat in categories:
                    print(f"      - {cat}: {counts.get(cat, 0)} cells")
            else:
                print(f"      (Too many categories to display, showing first 5)")
                for cat in categories[:5]:
                    print(f"      - {cat}: {counts.get(cat, 0)} cells")
    
    # Check existing cluster information
    cluster_columns = [col for col in adata.obs.columns if 'cluster' in col.lower()]
    if cluster_columns:
        print("\nFound potential cluster columns:", cluster_columns)
        
        # Choose the first one for demonstration
        cluster_col = cluster_columns[0]
        print(f"Using '{cluster_col}' as cluster column")
        
        # Count clusters
        clusters = adata.obs[cluster_col].unique()
        print(f"Found {len(clusters)} clusters")
        
        # Sample cells for testing
        test_adata = sc.pp.subsample(adata, n_obs=5000, copy=True)
        print(f"Sampled {test_adata.n_obs} cells for testing")
        
        # Save sampled data for testing
        test_adata.write_h5ad('/Users/apple/Research/LLMCelltype/data/raw/MTG_sampled.h5ad')
        print("Sampled data saved to /Users/apple/Research/LLMCelltype/data/raw/MTG_sampled.h5ad")

def run_simple_test():
    """Run a simplified test using pre-sampled data"""
    print("Loading sampled AnnData file...")
    adata = sc.read_h5ad('/Users/apple/Research/LLMCelltype/data/raw/MTG_sampled.h5ad')
    
    print(f"Working with {adata.n_obs} cells")
    
    # Find cluster column
    cluster_columns = [col for col in adata.obs.columns if 'cluster' in col.lower()]
    if not cluster_columns:
        print("No cluster column found. Running leiden clustering...")
        # Basic preprocessing
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        adata = adata[:, adata.var.highly_variable]
        sc.pp.pca(adata, svd_solver='arpack')
        sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
        
        # Run Leiden clustering
        sc.tl.leiden(adata, resolution=0.8)
        cluster_col = 'leiden'
    else:
        # Use first cluster column found
        cluster_col = cluster_columns[0]
    
    print(f"Using cluster column: {cluster_col}")
    print(f"Number of clusters: {len(adata.obs[cluster_col].unique())}")
    
    # Setup logging
    lct.setup_logging(log_level='INFO')
    
    # Find marker genes for clusters
    if 'rank_genes_groups' not in adata.uns:
        print("Calculating marker genes...")
        sc.tl.rank_genes_groups(adata, cluster_col, method='wilcoxon')
    
    # Get marker genes for each cluster
    marker_genes = {}
    markers_df = pd.DataFrame()
    
    # Get all clusters
    clusters = adata.obs[cluster_col].unique()
    
    for cluster in clusters:
        try:
            # Get top 15 genes for this cluster
            genes = sc.get.rank_genes_groups_df(adata, group=cluster).head(15)
            
            # Add to markers dataframe
            cluster_markers = pd.DataFrame({
                'cluster': [cluster] * len(genes),
                'gene': genes['names'].values,
                'avg_log2FC': genes['logfoldchanges'].values,
                'p_val_adj': genes['pvals_adj'].values
            })
            
            markers_df = pd.concat([markers_df, cluster_markers])
            
            # Also store in dictionary format
            marker_genes[cluster] = genes['names'].tolist()
        except Exception as e:
            print(f"Error processing cluster {cluster}: {str(e)}")
    
    # Print marker genes for first few clusters
    print("\nTop marker genes for first 3 clusters:")
    for i, cluster in enumerate(clusters[:3]):
        if cluster in marker_genes:
            print(f"Cluster {cluster}: {', '.join(marker_genes[cluster][:5])}")
    
    # Try to load API keys
    api_keys = {}
    for provider in ['openai']:  # Start with just OpenAI for simplicity
        api_key = lct.load_api_key(provider)
        if api_key:
            api_keys[provider] = api_key
            print(f"Found API key for {provider}")
    
    if not api_keys:
        print("No API keys found. Please set at least one API key in your .env file.")
        return
    
    # Get cell types using OpenAI
    if 'openai' in api_keys:
        print("\nAnnotating cell types with OpenAI...")
        
        # Limit to first 10 clusters for quick testing
        first_clusters = dict(list(marker_genes.items())[:10])
        
        try:
            openai_results = lct.annotate_clusters(
                marker_genes=first_clusters,  # Use dictionary format for testing
                species="human",
                tissue="middle temporal gyrus",
                provider="openai",
                model="gpt-4o",
                api_key=api_keys['openai'],
                use_cache=True
            )
            
            print("\nOpenAI annotation results:")
            for cluster, cell_type in openai_results.items():
                print(f"Cluster {cluster}: {cell_type}")
            
            print("\nLLMCelltype Python implementation is working properly!")
            return openai_results
        except Exception as e:
            print(f"Error in annotation: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test LLMCelltype with AnnData")
    parser.add_argument("--examine", action="store_true", help="Only examine AnnData structure")
    parser.add_argument("--annotate", action="store_true", help="Run annotation test")
    args = parser.parse_args()
    
    if args.examine:
        examine_anndata()
    elif args.annotate:
        run_simple_test()
    else:
        print("Please specify --examine or --annotate")