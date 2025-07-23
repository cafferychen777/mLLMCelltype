#!/usr/bin/env python3
"""
Example script demonstrating the new parameters in mLLMCelltype Python version:
- clusters_to_analyze: Analyze specific clusters only
- force_rerun: Force fresh analysis ignoring cache

This mirrors the functionality added to the R version in commit 8e5e308.
"""

import sys
import os

# Add the package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mllmcelltype'))

from mllmcelltype.consensus import interactive_consensus_annotation


def example_clusters_to_analyze():
    """Example: Analyze specific clusters only"""
    print("=" * 60)
    print("Example 1: clusters_to_analyze parameter")
    print("=" * 60)
    print("Use case: You have many clusters but only want to focus on specific ones")
    
    # Example dataset with 6 clusters
    marker_genes = {
        "cluster_0": ["CD3D", "CD3E", "CD3G", "IL7R", "TCF7"],        # T cells
        "cluster_1": ["CD79A", "CD79B", "MS4A1", "IGHM", "IGKC"],     # B cells  
        "cluster_2": ["CD14", "LYZ", "S100A8", "S100A9", "FCN1"],     # Monocytes
        "cluster_3": ["CD68", "CD163", "MSR1", "MRC1", "MARCO"],      # Macrophages
        "cluster_4": ["FCGR3A", "NCR1", "KLRD1", "GNLY", "PRF1"],     # NK cells
        "cluster_5": ["PPBP", "PF4", "TUBB1", "GP9", "ITGA2B"]        # Platelets
    }
    
    print("Full dataset contains clusters:", list(marker_genes.keys()))
    print("But we only want to analyze immune cells (clusters 0, 1, 4)")
    
    # Analyze only specific clusters of interest
    result = interactive_consensus_annotation(
        marker_genes=marker_genes,
        species="human",
        models=["gpt-4o"],
        clusters_to_analyze=["cluster_0", "cluster_1", "cluster_4"],  # Focus on immune cells
        verbose=True
    )
    
    print(f"\\nAnalyzed clusters: {list(result['consensus'].keys())}")
    print("Results:")
    for cluster_id, cell_type in result['consensus'].items():
        print(f"  {cluster_id}: {cell_type}")
    
    return result


def example_force_rerun():
    """Example: Force rerun to get fresh analysis"""
    print("\\n" + "=" * 60)
    print("Example 2: force_rerun parameter")
    print("=" * 60)
    print("Use case: You want fresh analysis ignoring cached results")
    
    marker_genes = {
        "cluster_0": ["CD3D", "CD3E", "CD8A", "PRF1", "GZMB"],  # Could be CD8+ T or NK cells
        "cluster_1": ["CD19", "MS4A1", "CD79A", "IGHM", "CD27"] # B cells with memory markers
    }
    
    print("These clusters have somewhat ambiguous markers...")
    print("First run (will cache results):")
    
    # First run - will create cache
    result1 = interactive_consensus_annotation(
        marker_genes=marker_genes,
        species="human",
        models=["gpt-4o"],
        force_rerun=False,  # Use cache if available
        verbose=False
    )
    
    print("First analysis results:")
    for cluster_id, cell_type in result1['consensus'].items():
        print(f"  {cluster_id}: {cell_type}")
    
    print("\\nSecond run with force_rerun=True (ignores cache):")
    
    # Second run - force fresh analysis
    result2 = interactive_consensus_annotation(
        marker_genes=marker_genes,
        species="human", 
        models=["gpt-4o"],
        force_rerun=True,  # Ignore cache, force fresh analysis
        verbose=False
    )
    
    print("Fresh analysis results:")
    for cluster_id, cell_type in result2['consensus'].items():
        print(f"  {cluster_id}: {cell_type}")
    
    # Compare results
    print("\\nComparison:")
    for cluster_id in result1['consensus']:
        cached = result1['consensus'][cluster_id]
        fresh = result2['consensus'][cluster_id]
        match = "✓" if cached == fresh else "✗"
        print(f"  {cluster_id}: {match} {cached} vs {fresh}")
    
    return result1, result2


def example_combined_parameters():
    """Example: Using both parameters together"""
    print("\\n" + "=" * 60)
    print("Example 3: Combined parameters")
    print("=" * 60)
    print("Use case: Analyze specific controversial clusters with fresh analysis")
    
    marker_genes = {
        "cluster_0": ["CD3D", "CD3E", "CD8A", "PRF1", "GZMB"],      # Cytotoxic cells
        "cluster_1": ["CD79A", "CD79B", "MS4A1", "IGHM", "IGKC"],   # B cells
        "cluster_2": ["CD14", "LYZ", "S100A8", "S100A9", "FCN1"],   # Monocytes
        "cluster_3": ["FCGR3A", "NCR1", "KLRD1", "GNLY", "PRF1"],   # NK cells
        "cluster_4": ["CD68", "CD163", "MSR1", "MRC1", "MARCO"]     # Macrophages
    }
    
    print("Scenario: Previous analysis of cluster_0 and cluster_3 was uncertain")
    print("We want to re-analyze just these two with fresh LLM calls")
    
    result = interactive_consensus_annotation(
        marker_genes=marker_genes,
        species="human",
        models=["gpt-4o"],
        clusters_to_analyze=["cluster_0", "cluster_3"],  # Focus on uncertain clusters
        force_rerun=True,                                # Get fresh analysis
        verbose=True
    )
    
    print(f"\\nRe-analyzed clusters: {list(result['consensus'].keys())}")
    print("Fresh results for uncertain clusters:")
    for cluster_id, cell_type in result['consensus'].items():
        print(f"  {cluster_id}: {cell_type}")
    
    return result


def main():
    """Run all examples"""
    print("mLLMCelltype Python - New Parameters Examples")
    print("Demonstrating clusters_to_analyze and force_rerun parameters")
    print("These features match the R version enhancements from commit 8e5e308")
    
    try:
        # Example 1: Cluster filtering
        example_clusters_to_analyze()
        
        # Example 2: Force rerun
        example_force_rerun()
        
        # Example 3: Combined usage
        example_combined_parameters()
        
        print("\\n" + "=" * 60)
        print("✅ All examples completed successfully!")
        print("The new parameters provide enhanced control over cell type annotation:")
        print("• clusters_to_analyze: Focus analysis on specific clusters")
        print("• force_rerun: Force fresh analysis bypassing cache")
        print("• Both can be used together for maximum flexibility")
        print("=" * 60)
        
    except Exception as e:
        print(f"\\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()