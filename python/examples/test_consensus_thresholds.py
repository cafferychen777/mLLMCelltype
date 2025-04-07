#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test the impact of different consensus thresholds on discussion rounds
"""

import os
import sys
import pandas as pd
import numpy as np
import scanpy as sc
from dotenv import load_dotenv
import matplotlib.pyplot as plt

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# 导入LLMCelltype的函数
from llmcelltype import interactive_consensus_annotation, print_consensus_summary

# 加载环境变量
load_dotenv()

def test_consensus_thresholds():
    """Test the impact of different consensus thresholds on discussion rounds"""
    
    print("\n=== Testing Different Consensus Thresholds ===\n")
    
    # Create a simple test dataset
    marker_genes = {
        "1": ["CD3D", "CD3E", "CD4", "IL7R", "CCR7"],  # T cells
        "2": ["CD19", "MS4A1", "CD79A", "CD79B", "CD22"],  # B cells
        "3": ["CD14", "LYZ", "CSF1R", "MARCO", "CD68"],  # Macrophages
        "4": ["FCGR3A", "MS4A7", "CDKN1C", "CX3CR1", "CD14"],  # Monocytes
        "5": ["FCER1A", "CST3", "IRF8", "CD1C", "CD11c"]  # Dendritic cells
    }
    
    # Get API keys
    api_keys = {
        'openai': os.getenv('OPENAI_API_KEY'),
        'anthropic': os.getenv('ANTHROPIC_API_KEY'),
        'gemini': os.getenv('GEMINI_API_KEY'),
        'qwen': os.getenv('QWEN_API_KEY')
    }
    
    # Models to use
    models = ['gpt-4o', 'claude-3-5-sonnet-20241022', 'gemini-1.5-pro', 'qwen2.5-72b-instruct']
    print(f"Using models: {', '.join(models)}")
    
    # Test different consensus thresholds
    thresholds = [0.5, 0.7, 0.9, 0.95]
    results = {}
    
    for threshold in thresholds:
        print(f"\n--- Testing consensus threshold: {threshold} ---")
        
        # Run consensus annotation
        result = interactive_consensus_annotation(
            marker_genes=marker_genes,
            species='human',
            tissue='blood',
            models=models,
            api_keys=api_keys,
            consensus_threshold=threshold,
            max_discussion_rounds=5,  # 最多5轮讨论
            use_cache=True,
            verbose=True
        )
        
        # Save results
        results[threshold] = result
        
        # Print result summary
        print_consensus_summary(result)
        
        # Analyze discussion rounds
        if "discussion_logs" in result:
            rounds_per_cluster = {}
            for cluster, logs in result["discussion_logs"].items():
                logs_text = logs if isinstance(logs, str) else "\n".join(logs)
                rounds = logs_text.count("Round ")
                rounds_per_cluster[cluster] = rounds
            
            print(f"\nDiscussion rounds per cluster (threshold={threshold}):")
            for cluster, rounds in rounds_per_cluster.items():
                print(f"  Cluster {cluster}: {rounds} rounds")
    
    # Analyze results
    print("\n=== Analysis of Different Consensus Thresholds ===\n")
    
    # Calculate average discussion rounds for each threshold
    avg_rounds = {}
    for threshold, result in results.items():
        if "discussion_logs" in result:
            total_rounds = 0
            cluster_count = 0
            for cluster, logs in result["discussion_logs"].items():
                logs_text = logs if isinstance(logs, str) else "\n".join(logs)
                rounds = logs_text.count("Round ")
                total_rounds += rounds
                cluster_count += 1
            
            if cluster_count > 0:
                avg_rounds[threshold] = total_rounds / cluster_count
            else:
                avg_rounds[threshold] = 0
    
    print("Average discussion rounds per threshold:")
    for threshold, avg in avg_rounds.items():
        print(f"  Threshold {threshold}: {avg:.2f} rounds")
    
    # Plot chart
    plt.figure(figsize=(10, 6))
    plt.bar(list(avg_rounds.keys()), list(avg_rounds.values()), color='skyblue')
    plt.xlabel('Consensus Threshold')
    plt.ylabel('Average Discussion Rounds')
    plt.title('Impact of Consensus Threshold on Discussion Rounds')
    plt.xticks(list(avg_rounds.keys()))
    plt.savefig('consensus_threshold_impact.png')
    print("\nChart saved as 'consensus_threshold_impact.png'")
    
    return results

if __name__ == "__main__":
    test_consensus_thresholds()
