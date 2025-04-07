#!/usr/bin/env python3
"""
Example using LLMCelltype with AnnData (scanpy) objects.
"""

import os
import sys
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Add parent directory to path for local development
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import llmcelltype as lct

# Load API keys from .env file
load_dotenv()

# Define the path to the AnnData file
FILE_PATH = '/Users/apple/Research/LLMCelltype/data/raw/MTG.h5ad'

def run_anndata_example():
    """Process an AnnData object with LLMCelltype"""
    print("Loading AnnData file...")
    adata = sc.read_h5ad(FILE_PATH)
    
    print("\nAnnData summary:")
    print(f"  Shape: {adata.shape}")
    print(f"  Number of genes: {adata.var.shape[0]}")
    
    # Print available columns in obs
    print("Available columns in adata.obs:")
    for col in adata.obs.columns:
        print(f"  - {col}")
    
    # Check if we need to perform clustering
    if 'leiden' not in adata.obs.columns:
        print("\nPerforming preprocessing and clustering...")
        # Basic preprocessing
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        sc.pp.pca(adata, svd_solver='arpack')
        sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
        
        # Run UMAP for visualization
        sc.tl.umap(adata)
        
        # Run Leiden clustering
        sc.tl.leiden(adata, resolution=0.8)
        print(f"Generated {len(adata.obs['leiden'].unique())} leiden clusters")
    
    # Setup logging
    lct.setup_logging(log_level='INFO')
    
    # Find marker genes for each cluster
    print("\nFinding marker genes for each cluster...")
    sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')
    
    # Get top markers for each cluster
    marker_genes = {}
    clusters = adata.obs['leiden'].unique()
    
    print(f"Processing {len(clusters)} clusters...")
    
    # Create marker genes DataFrame
    markers_df = pd.DataFrame()
    
    for cluster in clusters:
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
        
        # Also store in dictionary format for visualization
        marker_genes[cluster] = genes['names'].tolist()
    
    # Print marker genes for first few clusters
    print("\nTop marker genes for first 3 clusters:")
    for cluster in list(clusters)[:3]:
        print(f"Cluster {cluster}: {', '.join(marker_genes[cluster][:5])}")
    
    # Try to load API keys for multiple models
    api_keys = {}
    models_to_try = ['openai', 'anthropic']
    for provider in models_to_try:
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
        openai_results = lct.annotate_clusters(
            marker_genes=markers_df,
            species="human",
            tissue="middle temporal gyrus",
            provider="openai",
            model="gpt-4o",
            api_key=api_keys['openai'],
            use_cache=True
        )
        
        print("\nOpenAI annotation results:")
        for cluster in sorted(openai_results.keys(), key=lambda x: int(x) if x.isdigit() else float('inf')):
            print(f"Cluster {cluster}: {openai_results[cluster]}")
        
        # Store annotations in AnnData
        cell_types = adata.obs['leiden'].map(openai_results).astype('category')
        adata.obs['cell_type_openai'] = cell_types
    
    # Get cell types using Anthropic if available
    if 'anthropic' in api_keys:
        print("\nAnnotating cell types with Anthropic...")
        anthropic_results = lct.annotate_clusters(
            marker_genes=markers_df,
            species="human",
            tissue="middle temporal gyrus",
            provider="anthropic",
            model="claude-3-5-sonnet-20240620",
            api_key=api_keys['anthropic'],
            use_cache=True
        )
        
        print("\nAnthropic annotation results:")
        for cluster in sorted(anthropic_results.keys(), key=lambda x: int(x) if x.isdigit() else float('inf')):
            print(f"Cluster {cluster}: {anthropic_results[cluster]}")
        
        # Store annotations in AnnData
        cell_types = adata.obs['leiden'].map(anthropic_results).astype('category')
        adata.obs['cell_type_anthropic'] = cell_types
    
    # Compare results if both models were used
    if 'openai' in api_keys and 'anthropic' in api_keys:
        print("\nComparing results between OpenAI and Anthropic...")
        
        model_predictions = {
            "OpenAI (GPT-4o)": openai_results,
            "Anthropic (Claude)": anthropic_results
        }
        
        # Compare model predictions
        agreement_df, metrics = lct.compare_model_predictions(
            model_predictions=model_predictions,
            display_plot=False
        )
        
        print(f"Average agreement: {metrics['agreement_avg']:.2f}")
        
        # Run consensus annotation
        print("\nRunning consensus annotation...")
        consensus_result = lct.interactive_consensus_annotation(
            marker_genes=marker_genes,
            species='human',
            tissue='middle temporal gyrus',
            models=['gpt-4o', 'claude-3-5-sonnet-20240620'],
            api_keys=api_keys,
            consensus_threshold=0.6,
            use_cache=True,
            verbose=False
        )
        
        # Print consensus summary
        lct.print_consensus_summary(consensus_result)
        
        # Store consensus annotations
        consensus_annotations = consensus_result['consensus']
        
        # Map consensus annotations to cells
        consensus_cell_types = adata.obs['leiden'].map(consensus_annotations).astype('category')
        adata.obs['cell_type_consensus'] = consensus_cell_types
    
    # Visualize results
    print("\nVisualizing results...")
    sc.pl.umap(adata, color=['leiden'], title='Clusters')
    
    if 'openai' in api_keys:
        sc.pl.umap(adata, color=['cell_type_openai'], title='OpenAI Annotations')
    
    if 'anthropic' in api_keys:
        sc.pl.umap(adata, color=['cell_type_anthropic'], title='Anthropic Annotations')
    
    if 'openai' in api_keys and 'anthropic' in api_keys:
        sc.pl.umap(adata, color=['cell_type_consensus'], title='Consensus Annotations')
    
    # Save the annotated data
    adata.write_h5ad('/Users/apple/Research/LLMCelltype/data/processed/MTG_annotated.h5ad')
    print("\nAnnotated data saved to: /Users/apple/Research/LLMCelltype/data/processed/MTG_annotated.h5ad")
    
    return adata

if __name__ == "__main__":
    run_anndata_example()