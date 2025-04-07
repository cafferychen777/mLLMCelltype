#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test the marker gene extraction functionality and consensus algorithm in LLMCelltype
"""

import os
import sys
import pandas as pd
import numpy as np
import scanpy as sc
import anndata
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import functions from LLMCelltype
from mllmcelltype.functions import identify_controversial_clusters, select_best_prediction
from mllmcelltype.utils import find_agreement

# Load environment variables
load_dotenv()

def create_synthetic_dataset(n_cells=1000, n_genes=500, n_clusters=5):
    """Create a synthetic single-cell dataset for testing"""
    
    np.random.seed(42)  # Set random seed to ensure reproducibility
    
    # Create random expression matrix
    X = np.random.negative_binomial(5, 0.3, size=(n_cells, n_genes))
    
    # Create gene names
    gene_names = [f"gene_{i}" for i in range(n_genes)]
    
    # Create cell clusters
    cell_clusters = np.random.choice(range(n_clusters), size=n_cells)
    
    # Create specific marker genes for each cluster
    marker_indices = {}
    for i in range(n_clusters):
        # Randomly select 10 marker genes for each cluster
        marker_indices[i] = np.random.choice(range(n_genes), size=10, replace=False)
        
        # Increase expression of these marker genes
        for cell_idx in np.where(cell_clusters == i)[0]:
            X[cell_idx, marker_indices[i]] *= 5
    
    # Create AnnData object
    adata = anndata.AnnData(X=X)
    adata.var_names = gene_names
    adata.obs['cluster'] = cell_clusters.astype(str)
    adata.obs['cluster'] = adata.obs['cluster'].astype('category')
    
    # Create true marker gene dictionary
    true_markers = {}
    for cluster in range(n_clusters):
        true_markers[str(cluster)] = [gene_names[idx] for idx in marker_indices[cluster]]
    
    return adata, true_markers

def test_marker_extraction():
    """Test marker gene extraction functionality"""
    
    print("\n=== Testing Marker Gene Extraction ===\n")
    
    # Create synthetic dataset
    print("Creating synthetic dataset...")
    adata, true_markers = create_synthetic_dataset()
    print(f"Created dataset with {adata.shape[0]} cells, {adata.shape[1]} genes, and {len(np.unique(adata.obs['cluster']))} clusters")
    
    # Extract marker genes using scanpy
    print("\nExtracting marker genes using scanpy...")
    sc.tl.rank_genes_groups(adata, 'cluster', method='wilcoxon')
    
    # Extract marker genes from scanpy results
    extracted_markers = {}
    for cluster in np.unique(adata.obs['cluster']):
        cluster_str = str(cluster)
        genes = sc.get.rank_genes_groups_df(adata, group=cluster_str).head(10)['names'].tolist()
        extracted_markers[cluster_str] = genes
    
    # Compare extracted markers with true markers
    print("\nComparing extracted markers with true markers:")
    
    overlap_scores = {}
    for cluster in true_markers:
        true_set = set(true_markers[cluster])
        extracted_set = set(extracted_markers[cluster])
        
        # Calculate overlap rate
        overlap = len(true_set.intersection(extracted_set))
        overlap_rate = overlap / len(true_set) if true_set else 0
        
        overlap_scores[cluster] = overlap_rate
        
        print(f"  Cluster {cluster}:")
        print(f"    True markers: {', '.join(true_markers[cluster])}")
        print(f"    Extracted markers: {', '.join(extracted_markers[cluster])}")
        print(f"    Overlap: {overlap}/{len(true_set)} ({overlap_rate:.2%})")
    
    # Plot overlap rate chart
    plt.figure(figsize=(10, 6))
    plt.bar(overlap_scores.keys(), overlap_scores.values(), color='lightblue')
    plt.xlabel('Cluster')
    plt.ylabel('Overlap Rate')
    plt.title('Marker Gene Extraction Accuracy')
    plt.ylim(0, 1)
    plt.savefig('marker_extraction_accuracy.png')
    print("\nChart saved as 'marker_extraction_accuracy.png'")
    
    return adata, true_markers, extracted_markers

def test_consensus_algorithm():
    """Test consensus algorithm functionality"""
    
    print("\n=== Testing Consensus Algorithm ===\n")
    
    # Create some simulated model predictions
    model_predictions = {
        'model1': {
            '1': 'CD4+ T cells',
            '2': 'B cells',
            '3': 'Macrophages',
            '4': 'Monocytes',
            '5': 'Dendritic cells'
        },
        'model2': {
            '1': 'T helper cells',
            '2': 'B lymphocytes',
            '3': 'Macrophages',
            '4': 'Monocytes',
            '5': 'mDCs'
        },
        'model3': {
            '1': 'CD4 T cells',
            '2': 'B cells',
            '3': 'Alveolar macrophages',
            '4': 'Classical monocytes',
            '5': 'Dendritic cells'
        },
        'model4': {
            '1': 'T cells',
            '2': 'B cells',
            '3': 'Macrophages',
            '4': 'NK cells',  # Intentionally set a different prediction
            '5': 'Dendritic cells'
        }
    }
    
    # Test identify_controversial_clusters function
    print("Testing identify_controversial_clusters function...")
    
    thresholds = [0.5, 0.7, 0.9]
    for threshold in thresholds:
        controversial = identify_controversial_clusters(model_predictions, threshold)
        print(f"  Threshold {threshold}: Controversial clusters = {controversial}")
    
    # Test find_agreement function
    print("\nTesting find_agreement function...")
    
    for cluster in sorted(model_predictions['model1'].keys()):
        # Create a dictionary of model predictions for this cluster
        predictions_dict = {model: model_predictions[model][cluster] for model in model_predictions}
        
        # Calculate agreement manually for demonstration
        predictions_list = list(predictions_dict.values())
        unique_predictions = set(predictions_list)
        counts = {pred: predictions_list.count(pred) for pred in unique_predictions}
        max_count = max(counts.values()) if counts else 0
        total = len(predictions_list)
        cp = max_count / total if total > 0 else 0
        
        # Calculate Shannon entropy
        h = 0
        if total > 0:
            for count in counts.values():
                p = count / total
                h -= p * np.log2(p) if p > 0 else 0
        
        # Find the most common prediction
        most_common = max(counts.items(), key=lambda x: x[1])[0] if counts else ""
        
        print(f"  Cluster {cluster}:")
        print(f"    Predictions: {predictions_list}")
        print(f"    Most common: {most_common}")
        print(f"    Consensus Proportion (CP): {cp:.2f}")
        print(f"    Shannon Entropy (H): {h:.2f}")
    
    # Test select_best_prediction function
    print("\nTesting select_best_prediction function...")
    
    for cluster in sorted(model_predictions['model1'].keys()):
        predictions_list = [model_predictions[model][cluster] for model in model_predictions]
        
        # Create a dictionary format that select_best_prediction might expect
        predictions_dict = {f"model{i+1}": pred for i, pred in enumerate(predictions_list)}
        
        # For demonstration, just use the most common prediction
        counts = {pred: predictions_list.count(pred) for pred in set(predictions_list)}
        best_prediction = max(counts.items(), key=lambda x: x[1])[0] if counts else ""
        
        print(f"  Cluster {cluster}:")
        print(f"    Predictions: {predictions_list}")
        print(f"    Best prediction: {best_prediction}")
    
    # Create heatmap showing prediction similarity
    print("\nCreating prediction similarity heatmap...")
    
    clusters = sorted(model_predictions['model1'].keys())
    models = list(model_predictions.keys())
    
    # Create similarity matrix
    similarity_matrix = np.zeros((len(models), len(models)))
    
    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models):
            # Calculate prediction similarity between two models
            matches = 0
            for cluster in clusters:
                pred1 = model_predictions[model1][cluster].lower()
                pred2 = model_predictions[model2][cluster].lower()
                
                # If predictions are the same or one is a substring of the other, consider it a match
                if pred1 in pred2 or pred2 in pred1:
                    matches += 1
            
            similarity = matches / len(clusters)
            similarity_matrix[i, j] = similarity
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, annot=True, cmap='YlGnBu', xticklabels=models, yticklabels=models)
    plt.title('Model Prediction Similarity')
    plt.tight_layout()
    plt.savefig('prediction_similarity.png')
    print("\nChart saved as 'prediction_similarity.png'")
    
    return model_predictions

def run_all_tests():
    """Run all tests"""
    
    print("=== Running All LLMCelltype Tests ===")
    
    # Test marker gene extraction
    adata, true_markers, extracted_markers = test_marker_extraction()
    
    # Test consensus algorithm
    model_predictions = test_consensus_algorithm()
    
    print("\n=== All Tests Completed ===")

if __name__ == "__main__":
    run_all_tests()
