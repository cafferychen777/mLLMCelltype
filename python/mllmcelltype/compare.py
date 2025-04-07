"""
Module for comparing annotations from different LLM providers.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from itertools import combinations

from .logger import write_log
from .utils import clean_annotation
from .functions import identify_controversial_clusters

def compare_model_predictions(model_predictions: Dict[str, Dict[str, str]], 
                            display_plot: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Compare cell type annotations from different LLM models.
    
    Args:
        model_predictions: Dictionary mapping model names to dictionaries of cluster annotations
        display_plot: Whether to display plots
        
    Returns:
        Tuple of:
            - DataFrame containing pairwise agreement scores
            - Dictionary with additional metrics
    """
    if not model_predictions:
        write_log("Error: No model predictions provided", level='error')
        return pd.DataFrame(), {}
    
    # Get all model names
    models = list(model_predictions.keys())
    if len(models) < 2:
        write_log("Warning: Need at least 2 models to compare", level='warning')
        return pd.DataFrame({"model1": models, "model2": models, "agreement": [1.0]}), {"agreement_avg": 1.0}
    
    # Get all cluster IDs
    clusters = set()
    for model_results in model_predictions.values():
        clusters.update(model_results.keys())
    clusters = sorted(list(clusters))
    
    # Calculate pairwise agreement
    agreement_data = []
    
    for model1, model2 in combinations(models, 2):
        # Get predictions for both models
        preds1 = model_predictions[model1]
        preds2 = model_predictions[model2]
        
        # Count agreements
        agreement_count = 0
        valid_clusters = 0
        
        for cluster in clusters:
            if cluster in preds1 and cluster in preds2:
                # Clean up annotations before comparing
                anno1 = clean_annotation(preds1[cluster])
                anno2 = clean_annotation(preds2[cluster])
                
                # Simple exact match for now
                if anno1.lower() == anno2.lower():
                    agreement_count += 1
                
                valid_clusters += 1
        
        # Calculate agreement score
        if valid_clusters > 0:
            agreement_score = agreement_count / valid_clusters
        else:
            agreement_score = 0.0
        
        agreement_data.append({
            "model1": model1,
            "model2": model2,
            "agreement": agreement_score,
            "agreement_count": agreement_count,
            "total_clusters": valid_clusters
        })
    
    # Create agreement matrix dataframe
    agreement_df = pd.DataFrame(agreement_data)
    
    # Create heatmap if requested
    if display_plot:
        try:
            # Create a matrix for the heatmap
            model_matrix = pd.DataFrame(index=models, columns=models, dtype=float)
            
            # Fill the matrix
            for _, row in agreement_df.iterrows():
                model_matrix.loc[row['model1'], row['model2']] = row['agreement']
                model_matrix.loc[row['model2'], row['model1']] = row['agreement']
            
            # Fill diagonal with 1s
            for model in models:
                model_matrix.loc[model, model] = 1.0
                
            # Create heatmap
            plt.figure(figsize=(10, 8))
            ax = sns.heatmap(model_matrix, annot=True, cmap="YlGnBu", vmin=0, vmax=1, 
                           square=True, cbar_kws={'label': 'Agreement Score'})
            plt.title('Model Agreement Matrix')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            write_log(f"Warning: Could not create plot: {str(e)}", level='warning')
    
    # Calculate more metrics
    # Average agreement score
    avg_agreement = agreement_df['agreement'].mean() if not agreement_df.empty else 0.0
    
    # Find the most and least agreeing pairs
    if not agreement_df.empty:
        most_agreeing = agreement_df.loc[agreement_df['agreement'].idxmax()]
        least_agreeing = agreement_df.loc[agreement_df['agreement'].idxmin()]
        
        most_agreeing_pair = (most_agreeing['model1'], most_agreeing['model2'])
        most_agreeing_score = most_agreeing['agreement']
        
        least_agreeing_pair = (least_agreeing['model1'], least_agreeing['model2'])
        least_agreeing_score = least_agreeing['agreement']
    else:
        most_agreeing_pair = ("none", "none")
        most_agreeing_score = 0.0
        least_agreeing_pair = ("none", "none")
        least_agreeing_score = 0.0
    
    # Identify controversial clusters
    controversial = identify_controversial_clusters(model_predictions, threshold=0.6)
    
    metrics = {
        "agreement_avg": avg_agreement,
        "most_agreeing_pair": most_agreeing_pair,
        "most_agreeing_score": most_agreeing_score,
        "least_agreeing_pair": least_agreeing_pair,
        "least_agreeing_score": least_agreeing_score,
        "controversial_clusters": controversial,
        "controversial_count": len(controversial),
        "total_clusters": len(clusters)
    }
    
    return agreement_df, metrics

def create_comparison_table(model_predictions: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    """
    Create a table comparing cluster annotations from different models.
    
    Args:
        model_predictions: Dictionary mapping model names to dictionaries of cluster annotations
        
    Returns:
        pd.DataFrame: Table comparing annotations across models
    """
    if not model_predictions:
        return pd.DataFrame()
    
    # Get all model names
    models = list(model_predictions.keys())
    
    # Get all cluster IDs
    clusters = set()
    for model_results in model_predictions.values():
        clusters.update(model_results.keys())
    clusters = sorted(list(clusters))
    
    # Create table
    table_data = []
    
    for cluster in clusters:
        row = {"cluster": cluster}
        
        for model in models:
            if cluster in model_predictions[model]:
                row[model] = model_predictions[model][cluster]
            else:
                row[model] = "N/A"
        
        table_data.append(row)
        
    return pd.DataFrame(table_data)

def analyze_confusion_patterns(model_predictions: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    """
    Analyze patterns in disagreements between models.
    
    Args:
        model_predictions: Dictionary mapping model names to dictionaries of cluster annotations
        
    Returns:
        Dict[str, Any]: Dictionary with analysis results
    """
    if not model_predictions or len(model_predictions) < 2:
        return {"error": "Need at least 2 models to analyze confusion patterns"}
    
    # Get all model names
    models = list(model_predictions.keys())
    
    # Get all cluster IDs
    clusters = set()
    for model_results in model_predictions.values():
        clusters.update(model_results.keys())
    clusters = sorted(list(clusters))
    
    # Analyze clusters with disagreements
    disagreement_data = {}
    
    for cluster in clusters:
        # Get all annotations for this cluster
        cluster_annotations = {}
        
        for model in models:
            if cluster in model_predictions[model]:
                annotation = clean_annotation(model_predictions[model][cluster])
                cluster_annotations[model] = annotation
                
        # Count unique annotations
        unique_annotations = set(cluster_annotations.values())
        
        # If more than one unique annotation, there's disagreement
        if len(unique_annotations) > 1:
            disagreement_data[cluster] = {
                "annotations": cluster_annotations,
                "unique_count": len(unique_annotations),
                "unique_annotations": list(unique_annotations)
            }
    
    # Count common disagreement pairs
    disagreement_pairs = []
    
    for cluster, data in disagreement_data.items():
        # Get pairs of different annotations
        annotations = list(data["annotations"].values())
        for i, anno1 in enumerate(annotations):
            for anno2 in annotations[i+1:]:
                if anno1 != anno2:
                    # Sort to make pairs consistent
                    pair = tuple(sorted([anno1, anno2]))
                    disagreement_pairs.append(pair)
    
    # Count occurrences of each disagreement pair
    pair_counts = Counter(disagreement_pairs)
    most_common_pairs = pair_counts.most_common(10)
    
    # Identify models with most disagreements
    model_disagreements = {model: 0 for model in models}
    
    for cluster, data in disagreement_data.items():
        # For each model, count how many other models it disagrees with
        for model1 in models:
            anno1 = data["annotations"].get(model1)
            if anno1:
                for model2 in models:
                    if model1 != model2:
                        anno2 = data["annotations"].get(model2)
                        if anno2 and anno1 != anno2:
                            model_disagreements[model1] += 1
    
    # Sort models by disagreement count
    sorted_model_disagreements = sorted(model_disagreements.items(), key=lambda x: x[1], reverse=True)
    
    return {
        "clusters_analyzed": len(clusters),
        "disagreement_clusters": len(disagreement_data),
        "disagreement_rate": len(disagreement_data) / len(clusters) if clusters else 0,
        "common_disagreement_pairs": most_common_pairs,
        "model_disagreements": sorted_model_disagreements,
        "cluster_disagreements": disagreement_data
    }