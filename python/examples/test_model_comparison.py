#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test the performance and comparison of different models for cell type annotation
"""

import os
import sys
import pandas as pd
import numpy as np
import time
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from tabulate import tabulate

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# 导入LLMCelltype的函数
from llmcelltype import annotate_clusters, batch_annotate_clusters
from llmcelltype.utils import format_results

# 加载环境变量
load_dotenv()

def test_model_comparison():
    """Test the performance and comparison of different models for cell type annotation"""
    
    print("\n=== Testing Different Models for Cell Type Annotation ===\n")
    
    # Create a test dataset with known cell types
    test_data = {
        "1": {
            "markers": ["CD3D", "CD3E", "CD4", "IL7R", "CCR7"],
            "expected": "CD4+ T cells"
        },
        "2": {
            "markers": ["CD19", "MS4A1", "CD79A", "CD79B", "CD22"],
            "expected": "B cells"
        },
        "3": {
            "markers": ["CD14", "LYZ", "CSF1R", "MARCO", "CD68"],
            "expected": "Macrophages"
        },
        "4": {
            "markers": ["FCGR3A", "MS4A7", "CDKN1C", "CX3CR1", "CD14"],
            "expected": "Monocytes"
        },
        "5": {
            "markers": ["FCER1A", "CST3", "IRF8", "CD1C", "CD11C"],
            "expected": "Dendritic cells"
        },
        "6": {
            "markers": ["PPBP", "PF4", "ITGA2B", "GP1BA", "GP9"],
            "expected": "Platelets"
        },
        "7": {
            "markers": ["HBA1", "HBA2", "HBB", "ALAS2", "CA1"],
            "expected": "Erythroid cells"
        },
        "8": {
            "markers": ["TPSAB1", "CPA3", "MS4A2", "HDC", "GATA2"],
            "expected": "Mast cells"
        },
        "9": {
            "markers": ["S100A8", "S100A9", "CXCR1", "CXCR2", "CSF3R"],
            "expected": "Neutrophils"
        },
        "10": {
            "markers": ["KLRB1", "NCR1", "NCAM1", "CD3D-", "NKG7"],
            "expected": "NK cells"
        }
    }
    
    # Extract marker genes
    marker_genes = {cluster: data["markers"] for cluster, data in test_data.items()}
    expected_types = {cluster: data["expected"] for cluster, data in test_data.items()}
    
    # Get API keys
    api_keys = {
        'openai': os.getenv('OPENAI_API_KEY'),
        'anthropic': os.getenv('ANTHROPIC_API_KEY'),
        'gemini': os.getenv('GEMINI_API_KEY'),
        'qwen': os.getenv('QWEN_API_KEY')
    }
    
    # Models to test
    models = {
        'gpt-4o': 'openai',
        'claude-3-5-sonnet-20241022': 'anthropic',
        'gemini-1.5-pro': 'gemini',
        'qwen2.5-72b-instruct': 'qwen'
    }
    
    # Store results
    results = {}
    timing = {}
    accuracy = {}
    
    # Test each model
    for model_name, provider in models.items():
        print(f"\n--- Testing model: {model_name} ---")
        
        # Record start time
        start_time = time.time()
        
        # Use model for annotation
        api_key = api_keys[provider]
        result = annotate_clusters(
            marker_genes=marker_genes,
            species='human',
            tissue='blood',
            provider=provider,
            model=model_name,
            api_key=api_key,
            use_cache=True
        )
        
        # Record end time
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # The results are already formatted as dictionaries
        formatted_result = result
        
        # Save results
        results[model_name] = formatted_result
        timing[model_name] = elapsed_time
        
        # Calculate accuracy
        correct = 0
        total = len(expected_types)
        predictions = {}
        
        for cluster, prediction in formatted_result.items():
            expected = expected_types[cluster]
            predictions[cluster] = prediction
            
            # Check if prediction matches expected (simple substring matching)
            if expected.lower() in prediction.lower() or prediction.lower() in expected.lower():
                correct += 1
        
        accuracy[model_name] = correct / total if total > 0 else 0
        
        # Print results
        print(f"\nResults for {model_name}:")
        print(f"Time taken: {elapsed_time:.2f} seconds")
        print(f"Accuracy: {accuracy[model_name]:.2%} ({correct}/{total})")
        
        # Print comparison of predictions vs expected
        comparison = []
        for cluster in sorted(expected_types.keys(), key=int):
            expected = expected_types[cluster]
            prediction = predictions.get(cluster, "N/A")
            match = "✓" if (expected.lower() in prediction.lower() or prediction.lower() in expected.lower()) else "✗"
            comparison.append([cluster, expected, prediction, match])
        
        print("\nPrediction vs Expected:")
        print(tabulate(comparison, headers=["Cluster", "Expected", "Prediction", "Match"], tablefmt="grid"))
    
    # Compare all models
    print("\n=== Model Comparison Summary ===\n")
    
    # Accuracy comparison
    print("Accuracy comparison:")
    for model_name, acc in accuracy.items():
        print(f"  {model_name}: {acc:.2%}")
    
    # Time comparison
    print("\nTime comparison (seconds):")
    for model_name, time_taken in timing.items():
        print(f"  {model_name}: {time_taken:.2f}")
    
    # Plot accuracy chart
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.bar(list(accuracy.keys()), list(accuracy.values()), color='lightgreen')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    
    plt.subplot(1, 2, 2)
    plt.bar(list(timing.keys()), list(timing.values()), color='salmon')
    plt.xlabel('Model')
    plt.ylabel('Time (seconds)')
    plt.title('Model Response Time Comparison')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    print("\nChart saved as 'model_comparison.png'")
    
    return results, accuracy, timing

if __name__ == "__main__":
    test_model_comparison()
