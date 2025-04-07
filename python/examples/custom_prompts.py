#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Custom prompts example for LLMCelltype.
This script demonstrates how to use custom prompt templates for cell type annotation.
"""

import os
import pandas as pd
from mllmcelltype import annotate_clusters, setup_logging, load_api_key

# Setup logging
setup_logging(log_level="INFO")

# Example marker genes data
marker_genes_data = {
    "cluster": [1, 1, 1, 2, 2, 2, 3, 3, 3],
    "gene": ["CD3D", "CD3E", "CD2", "CD19", "MS4A1", "CD79A", "FCGR3A", "CD14", "LYZ"],
    "avg_log2FC": [2.5, 2.3, 2.1, 3.0, 2.8, 2.7, 2.2, 2.0, 1.9],
    "pct.1": [0.9, 0.85, 0.8, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7],
    "pct.2": [0.1, 0.15, 0.2, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
    "p_val_adj": [1e-10, 1e-9, 1e-8, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7]
}

marker_genes_df = pd.DataFrame(marker_genes_data)

# Define custom prompt templates
CUSTOM_TEMPLATES = {
    "simple": """You are a cell type annotation expert. Below are marker genes for different cell clusters in {context}.

{clusters}

For each numbered cluster, provide only the cell type name in a new line, without any explanation.

Format: "Cluster X: Cell Type"
""",

    "detailed": """You are a cell type annotation expert. Below are marker genes for different cell clusters in {context}.

{clusters}

For each numbered cluster, provide the following information:
1. The most likely cell type name
2. A confidence level (high, medium, low)
3. A brief justification (1-2 sentences) explaining your reasoning

FORMAT YOUR RESPONSE AS:
Cluster X: [Cell Type] (Confidence: [Level])
Justification: [Brief explanation]
""",

    "json": """You are a cell type annotation expert. Below are marker genes for different cell clusters in {context}.

{clusters}

For each numbered cluster, identify the most likely cell type.

RESPONSE FORMAT:
Provide your answer in JSON format as follows:
```json
{{
  "annotations": [
    {{
      "cluster": "1",
      "cell_type": "Type name",
      "confidence": "high|medium|low",
      "key_markers": ["gene1", "gene2", "gene3"]
    }},
    ...
  ]
}}
```

Only include the JSON in your response, nothing else.
"""
}

def main():
    """Main function to demonstrate custom prompts."""
    print("LLMCelltype Custom Prompts Example")
    print("----------------------------------")
    
    # Load API key
    api_key = load_api_key("openai")
    
    if not api_key:
        print("Warning: OpenAI API key not found. Using demonstration mode.")
        print("In a real scenario, you would need a valid API key.")
        api_key = "demo-key"  # This won't work for actual API calls
    
    # Print marker genes
    print("\nMarker Genes:")
    print(marker_genes_df)
    
    # Try each custom template
    for template_name, template in CUSTOM_TEMPLATES.items():
        print(f"\n\nUsing {template_name.upper()} template:")
        print("-" * (len(template_name) + 13))
        
        try:
            # Annotate with custom template
            annotations = annotate_clusters(
                marker_genes=marker_genes_df,
                species="human",
                provider="openai",
                model="gpt-4o",
                tissue="blood",
                api_key=api_key,
                prompt_template=template
                # Note: dry_run parameter was removed as it's not supported
            )
            
            # Print annotations
            print("\nAnnotation Results:")
            if isinstance(annotations, dict):
                for cluster, annotation in annotations.items():
                    print(f"Cluster {cluster}: {annotation}")
            else:
                print(annotations)  # For JSON and other formats that might return raw text
                
        except Exception as e:
            print(f"Error with {template_name} template: {e}")
            print("Note: This example requires a valid OpenAI API key to run.")
    
    print("\nCustom prompts example completed.")

    # Bonus: Show how to create a completely custom prompt function
    print("\n\nBonus: Custom Prompt Function Example")
    print("------------------------------------")
    print("You can also define a completely custom prompt function:")
    
    def my_custom_prompt_function(marker_genes, species, tissue, **kwargs):
        """Custom prompt function example."""
        # Process marker genes to create a custom format
        clusters_text = ""
        for cluster in sorted(marker_genes["cluster"].unique()):
            cluster_genes = marker_genes[marker_genes["cluster"] == cluster]
            genes_list = [f"{row['gene']} (FC={row['avg_log2FC']:.2f})" 
                         for _, row in cluster_genes.iterrows()]
            clusters_text += f"Cluster {cluster}: {', '.join(genes_list)}\n"
        
        # Create custom prompt
        prompt = f"""Annotate these {species} {tissue} cell clusters:

{clusters_text}
Provide annotations in CSV format: cluster,cell_type
"""
        print("\nCustom Prompt Function Output:")
        print(prompt)
        return prompt
    
    print("\nYou would use it like this:")
    print("annotations = annotate_clusters(")
    print("    marker_genes=marker_genes_df,")
    print("    species='human',")
    print("    provider='openai',")
    print("    prompt_function=my_custom_prompt_function")
    print(")")

if __name__ == "__main__":
    main()
