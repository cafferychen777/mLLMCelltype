#!/usr/bin/env python3
"""
Example demonstrating JSON-formatted responses from LLMs.
"""

import os
import sys
import pandas as pd
import json
from dotenv import load_dotenv

# Add parent directory to path for local development
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import mllmcelltype as lct

# Load API keys from .env file
load_dotenv()

# Sample data - common cell types in human PBMCs
marker_genes = {
    "1": ["CD3D", "CD3E", "CD3G", "CD2", "IL7R", "TCF7", "LCK", "CD8A"],
    "2": ["CD19", "MS4A1", "CD79A", "CD79B", "HLA-DRA", "CD74", "CD22"],
    "3": ["CD14", "LYZ", "CSF1R", "ITGAM", "CD68", "FCGR3A", "S100A8", "S100A9"],
    "4": ["FCGR3A", "NCAM1", "KLRB1", "GZMB", "NKG7", "CD160", "FCER1G", "KLRC1", "CD56"],
    "5": ["HBA1", "HBA2", "HBB", "ALAS2", "AHSP", "CA1", "HEMGN"]
}

def run_json_example():
    """Run the JSON formatting example"""
    print("\n=== JSON Response Example ===\n")
    
    # Setup logging
    lct.setup_logging(log_level='INFO')
    
    # Get API key for OpenAI (GPT-4 handles JSON formatting well)
    openai_key = lct.load_api_key('openai')
    if not openai_key:
        print("Error: OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")
        print("Using mock data for demonstration...")
        use_mock = True
    else:
        use_mock = False
    
    print("Example marker genes dictionary:")
    for cluster, genes in marker_genes.items():
        print(f"Cluster {cluster}: {', '.join(genes[:3])}{'...' if len(genes) > 3 else ''}")
    print()
    
    # Create a dataframe version for comparison
    markers_df = pd.DataFrame({
        'cluster': [k for k, v in marker_genes.items() for _ in v],
        'gene': [gene for genes in marker_genes.values() for gene in genes],
        'avg_log2FC': [2.0] * sum(len(v) for v in marker_genes.values())  # Dummy values
    })
    
    print("\n=== Using the new JSON prompt ===\n")
    
    # Generate JSON prompt
    json_prompt = lct.create_json_prompt(
        marker_genes=marker_genes,
        species='human',
        tissue='peripheral blood'
    )
    
    print("Generated JSON-formatted prompt...")
    
    if use_mock:
        # Mock response for demonstration
        mock_response = """```json
{
  "annotations": [
    {
      "cluster": "1",
      "cell_type": "CD8+ T cells",
      "confidence": "high",
      "key_markers": ["CD3D", "CD3E", "CD8A", "LCK"]
    },
    {
      "cluster": "2",
      "cell_type": "B cells",
      "confidence": "high",
      "key_markers": ["CD19", "MS4A1", "CD79A", "CD79B"]
    },
    {
      "cluster": "3",
      "cell_type": "Monocytes",
      "confidence": "high",
      "key_markers": ["CD14", "LYZ", "CSF1R", "CD68"]
    },
    {
      "cluster": "4",
      "cell_type": "NK cells",
      "confidence": "high",
      "key_markers": ["NCAM1", "KLRB1", "GZMB", "NKG7"]
    },
    {
      "cluster": "5",
      "cell_type": "Erythrocytes",
      "confidence": "high",
      "key_markers": ["HBA1", "HBA2", "HBB", "ALAS2"]
    }
  ]
}```"""
        response = mock_response
        print("Using mock response for demonstration...")
    else:
        # Get response from model
        try:
            response = lct.get_model_response(
                prompt=json_prompt,
                provider='openai',
                model='gpt-4o',
                api_key=openai_key,
                use_cache=True
            )
            print("Received response from model...")
        except Exception as e:
            print(f"Error getting model response: {str(e)}")
            return
    
    # Parse the JSON from the response
    try:
        # Extract JSON from the response (it might be wrapped in ```json ... ``` blocks)
        import re
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
        if json_match:
            json_str = json_match.group(1)
        else:
            # If no code blocks, try to find JSON object directly
            json_match = re.search(r'(\{[\s\S]*\})', response)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response
        
        # Parse JSON
        data = json.loads(json_str)
        
        # Print formatted result
        print("\nParsed JSON response:")
        print(json.dumps(data, indent=2))
        
        # Extract cell type annotations and confidence
        if "annotations" in data:
            print("\nCell Type Annotations with Metadata:")
            for anno in data["annotations"]:
                cluster = anno.get("cluster", "?")
                cell_type = anno.get("cell_type", "Unknown")
                confidence = anno.get("confidence", "unknown")
                key_markers = ", ".join(anno.get("key_markers", []))
                print(f"Cluster {cluster}: {cell_type} (Confidence: {confidence})")
                print(f"  Key markers: {key_markers}")
        else:
            print("\nNo 'annotations' field found in the JSON response")
            
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {str(e)}")
        print("\nRaw response:")
        print(response)
    
    # Now try with the regular annotation function
    print("\n=== Using annotate_clusters with format_json=True ===\n")
    
    try:
        if use_mock:
            # Mock example
            print("Using mock results for demonstration...")
            results = {
                "1": "CD8+ T cells",
                "2": "B cells",
                "3": "Monocytes", 
                "4": "NK cells",
                "5": "Erythrocytes"
            }
        else:
            # Use the annotate_clusters function
            results = lct.annotate_clusters(
                marker_genes=markers_df,
                species="human",
                tissue="peripheral blood",
                provider="openai",
                model="gpt-4o",
                format_json=True  # Request JSON format
            )
        
        print("Annotation results:")
        for cluster, cell_type in results.items():
            print(f"Cluster {cluster}: {cell_type}")
    
    except Exception as e:
        print(f"Error during annotation: {str(e)}")

if __name__ == "__main__":
    run_json_example()
