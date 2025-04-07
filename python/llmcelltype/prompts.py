"""
Prompt generation module for LLMCellType.
"""

from typing import Dict, List, Optional, Union, Any
from .logger import write_log

# Default prompt template for single dataset annotation
DEFAULT_PROMPT_TEMPLATE = """You are an expert single-cell RNA-seq analyst specializing in cell type annotation.
I need you to identify cell types of {species} cells from {tissue}.
Below is a list of marker genes for each cluster. 
Please assign the most likely cell type to each cluster based on the marker genes.

Only provide the cell type name for each cluster. Be concise but specific. 
Do not show numbers before the name.
Some clusters can be a mixture of multiple cell types.

Here are the marker genes for each cluster:
{markers}
"""

# Original simpler template
SIMPLE_PROMPT_TEMPLATE = """You are a cell type annotation expert. Below are marker genes for different cell clusters in {context}.

{clusters}

For each numbered cluster, provide only the cell type name in a new line, without any explanation.

Please provide your annotations in the following format:
Cluster 1: [Cell Type]
Cluster 2: [Cell Type]
...and so on.

Only provide the cell type name, without any additional text or explanations.
"""

# Default prompt template for batch annotation
DEFAULT_BATCH_PROMPT_TEMPLATE = """You are an expert single-cell RNA-seq analyst specializing in cell type annotation.
I need you to identify cell types of {species} cells from {tissue}.
Below are lists of marker genes for clusters from multiple datasets. 
Please assign the most likely cell type to each cluster based on the marker genes.

Format your response as follows for each set:
Set 1:
Cluster 1: [cell type]
Cluster 2: [cell type]
...

Set 2:
...

Here are the marker genes for each cluster:
{markers}
"""

# Original simpler batch template
SIMPLE_BATCH_PROMPT_TEMPLATE = """You are a cell type annotation expert. Below are marker genes for different cell clusters in {context}.

{clusters}

For each set and its numbered clusters, provide only the cell type name in a new line, without any explanation.

Please provide your annotations in the following format:
Set 1:
Cluster 1: [Cell Type]
Cluster 2: [Cell Type]
...

Set 2:
Cluster 1: [Cell Type]
Cluster 2: [Cell Type]
...

Only provide the cell type name, without any additional text or explanations.
"""

# Default JSON format prompt template
DEFAULT_JSON_PROMPT_TEMPLATE = """You are an expert single-cell RNA-seq analyst specializing in cell type annotation.
I need you to identify cell types of {species} cells from {tissue}.
Below is a list of marker genes for each cluster. 
Please assign the most likely cell type to each cluster based on the marker genes.

Format your response as a valid JSON object as follows:
```json
{{
  "annotations": [
    {{
      "cluster": "1",
      "cell_type": "T cells",
      "confidence": "high",
      "key_markers": ["CD3D", "CD3G", "CD3E"]
    }},
    {{
      "cluster": "2",
      "cell_type": "B cells",
      "confidence": "high",
      "key_markers": ["CD19", "CD79A", "MS4A1"]
    }},
    ...
  ]
}}
```

For each cluster, provide:
1. The cluster ID
2. The cell type name (be concise but specific)
3. Your confidence level (high, medium, low)
4. A list of 2-4 key markers that support your annotation

Here are the marker genes for each cluster:
{markers}
"""

# Template for facilitating discussion for controversial clusters
DEFAULT_DISCUSSION_TEMPLATE = """You are an expert in single-cell RNA-seq cell type annotation tasked with resolving disagreements between model predictions.

Cluster ID: {cluster_id}
Species: {species}
Tissue: {tissue}

Marker genes for this cluster:
{marker_genes}

Different model predictions:
{model_votes}

Your task:
1. Analyze the marker genes for this cluster
2. Evaluate each model's prediction, considering tissue context and marker gene specificity
3. Consider which cell types are characterized by these markers
4. Determine which prediction is most accurate or propose a better cell type annotation
5. Assess the confidence in your determination using Consensus Proportion (CP) and Shannon Entropy (H)

Provide a well-reasoned analysis with evidence from literature or known marker-cell type associations.
End with a clear final decision on the correct cell type, including:
- Final cell type determination
- Key supporting marker genes
- Consensus Proportion (CP): How strongly the evidence supports this annotation (0-1)
- Shannon Entropy (H): The uncertainty in this determination (lower is better)
"""

# Template for checking consensus across models
DEFAULT_CONSENSUS_CHECK_TEMPLATE = """You are an expert in single-cell RNA-seq analysis, evaluating the consensus cell type annotations across different models.

Species: {species}
Tissue: {tissue}

Here are the model predictions for each cluster:

{predictions}

For each cluster, assess:
1. The level of agreement between models
2. Which annotation is most accurate based on consensus
3. Any clusters where annotations significantly differ, which require further investigation

Provide a final consensus annotation for each cluster and note any controversial clusters that need additional review.
"""

# Template for checking if consensus is reached after discussion
DEFAULT_DISCUSSION_CONSENSUS_CHECK_TEMPLATE = """You are an expert in single-cell RNA-seq analysis, evaluating whether a consensus has been reached after discussion about a controversial cluster annotation.

Cluster ID: {cluster_id}

Discussion summary:
{discussion}

Proposed cell type: {proposed_cell_type}

Your task:
1. Evaluate whether the discussion has led to a clear and well-supported cell type determination
2. Consider if the evidence presented is sufficient to confidently annotate this cluster
3. Determine if further discussion would be beneficial or if consensus has been reached

Respond with one of the following:
- "Consensus reached: [reason]" if the evidence is clear and sufficient
- "Further discussion needed: [specific points to address]" if important aspects remain unresolved

Be specific about why you believe consensus has or has not been reached.
"""

def create_prompt(
    marker_genes: Dict[str, List[str]],
    species: str,
    tissue: Optional[str] = None,
    additional_context: Optional[str] = None,
    prompt_template: Optional[str] = None
) -> str:
    """
    Create a prompt for cell type annotation.
    
    Args:
        marker_genes: Dictionary mapping cluster names to lists of marker genes
        species: Species name (e.g., 'human', 'mouse')
        tissue: Tissue name (e.g., 'brain', 'liver')
        additional_context: Additional context to include in the prompt
        prompt_template: Custom prompt template
        
    Returns:
        str: The generated prompt
    """
    write_log(f"Creating prompt for {len(marker_genes)} clusters")
    
    # Use default template if not provided
    if not prompt_template:
        prompt_template = DEFAULT_PROMPT_TEMPLATE
        
    # Check if using the new or old template format
    if "{context}" in prompt_template and "{clusters}" in prompt_template:
        # Using old template format
        return create_prompt_legacy(
            marker_genes=marker_genes,
            species=species,
            tissue=tissue,
            additional_context=additional_context,
            prompt_template=prompt_template
        )
    
    # Default tissue if none provided
    tissue_text = tissue if tissue else "unknown tissue"
    
    # Format marker genes text
    marker_text_lines = []
    for cluster, genes in marker_genes.items():
        marker_text_lines.append(f"Cluster {cluster}: {', '.join(genes)}")
    
    marker_text = "\n".join(marker_text_lines)
    
    # Add additional context if provided
    context_text = f"\nAdditional context: {additional_context}\n" if additional_context else ""
    
    # Fill in the template
    prompt = prompt_template.format(
        species=species,
        tissue=tissue_text,
        markers=marker_text
    )
    
    # Add context
    if context_text:
        sections = prompt.split("Here are the marker genes for each cluster:")
        if len(sections) == 2:
            prompt = f"{sections[0]}{context_text}Here are the marker genes for each cluster:{sections[1]}"
        else:
            prompt = f"{prompt}{context_text}"
    
    write_log(f"Generated prompt with {len(prompt)} characters")
    return prompt

def create_prompt_legacy(
    marker_genes: Dict[str, List[str]],
    species: str,
    tissue: Optional[str] = None,
    additional_context: Optional[str] = None,
    prompt_template: Optional[str] = None
) -> str:
    """
    Create a prompt for cell type annotation using the legacy template format.
    
    Args:
        marker_genes: Dictionary mapping cluster names to lists of marker genes
        species: Species name (e.g., 'human', 'mouse')
        tissue: Tissue name (e.g., 'brain', 'liver')
        additional_context: Additional context to include in the prompt
        prompt_template: Custom prompt template
        
    Returns:
        str: The generated prompt
    """
    # Use default template if not provided
    if not prompt_template:
        prompt_template = SIMPLE_PROMPT_TEMPLATE
    
    # Format species and tissue
    species_str = species.strip()
    tissue_str = f" {tissue.strip()}" if tissue else ""
    
    # Create context string
    context_parts = []
    if species_str:
        context_parts.append(species_str)
    if tissue_str:
        context_parts.append(tissue_str.strip())
    
    context = " ".join(context_parts)
    if additional_context:
        context += f"\nAdditional Context: {additional_context}"
    
    # Create clusters string
    clusters_str = ""
    for cluster, genes in marker_genes.items():
        genes_str = ", ".join(genes)
        clusters_str += f"Cluster {cluster}: {genes_str}\n"
    
    # Format prompt
    prompt = prompt_template.format(
        context=context,
        clusters=clusters_str
    )
    
    write_log(f"Generated legacy prompt with {len(prompt)} characters")
    return prompt

def create_batch_prompt(
    marker_genes_list: List[Dict[str, List[str]]],
    species: str,
    tissue: Optional[str] = None,
    additional_context: Optional[str] = None,
    prompt_template: Optional[str] = None
) -> str:
    """
    Create a batch prompt for multiple sets of clusters.
    
    Args:
        marker_genes_list: List of dictionaries mapping cluster names to lists of marker genes
        species: Species name (e.g., 'human', 'mouse')
        tissue: Tissue name (e.g., 'brain', 'liver')
        additional_context: Additional context to include in the prompt
        prompt_template: Custom prompt template
        
    Returns:
        str: The generated batch prompt
    """
    write_log(f"Creating batch prompt for {len(marker_genes_list)} sets of clusters")
    
    # Use default template if not provided
    if not prompt_template:
        prompt_template = DEFAULT_BATCH_PROMPT_TEMPLATE
        
    # Check if using the new or old template format
    if "{context}" in prompt_template and "{clusters}" in prompt_template:
        # Using old template format
        return create_batch_prompt_legacy(
            marker_genes_list=marker_genes_list,
            species=species,
            tissue=tissue,
            additional_context=additional_context,
            prompt_template=prompt_template
        )
    
    # Default tissue if none provided
    tissue_text = tissue if tissue else "unknown tissue"
    
    # Format marker genes text
    marker_text_lines = []
    
    for i, marker_genes in enumerate(marker_genes_list):
        marker_text_lines.append(f"\nSet {i+1}:")
        for cluster, genes in marker_genes.items():
            marker_text_lines.append(f"Cluster {cluster}: {', '.join(genes)}")
    
    marker_text = "\n".join(marker_text_lines)
    
    # Add additional context if provided
    context_text = f"\nAdditional context: {additional_context}\n" if additional_context else ""
    
    # Fill in the template
    prompt = prompt_template.format(
        species=species,
        tissue=tissue_text,
        markers=marker_text
    )
    
    # Add context
    if context_text:
        sections = prompt.split("Here are the marker genes for each cluster:")
        if len(sections) == 2:
            prompt = f"{sections[0]}{context_text}Here are the marker genes for each cluster:{sections[1]}"
        else:
            prompt = f"{prompt}{context_text}"
    
    write_log(f"Generated batch prompt with {len(prompt)} characters")
    return prompt

def create_batch_prompt_legacy(
    marker_genes_list: List[Dict[str, List[str]]],
    species: str,
    tissue: Optional[str] = None,
    additional_context: Optional[str] = None,
    prompt_template: Optional[str] = None
) -> str:
    """
    Create a batch prompt for multiple sets of clusters using the legacy template format.
    
    Args:
        marker_genes_list: List of dictionaries mapping cluster names to lists of marker genes
        species: Species name (e.g., 'human', 'mouse')
        tissue: Tissue name (e.g., 'brain', 'liver')
        additional_context: Additional context to include in the prompt
        prompt_template: Custom prompt template
        
    Returns:
        str: The generated batch prompt
    """
    # Use default legacy template if not provided
    if not prompt_template:
        prompt_template = SIMPLE_BATCH_PROMPT_TEMPLATE
    
    # Format species and tissue
    species_str = species.strip()
    tissue_str = f" {tissue.strip()}" if tissue else ""
    
    # Create context string
    context_parts = []
    if species_str:
        context_parts.append(species_str)
    if tissue_str:
        context_parts.append(tissue_str.strip())
    
    context = " ".join(context_parts)
    if additional_context:
        context += f"\nAdditional Context: {additional_context}"
    
    # Create clusters string
    clusters_str = ""
    for i, marker_genes in enumerate(marker_genes_list):
        clusters_str += f"Set {i+1}:\n"
        for cluster, genes in marker_genes.items():
            genes_str = ", ".join(genes)
            clusters_str += f"Cluster {cluster}: {genes_str}\n"
        clusters_str += "\n"
    
    # Format prompt
    prompt = prompt_template.format(
        context=context,
        clusters=clusters_str
    )
    
    write_log(f"Generated legacy batch prompt with {len(prompt)} characters")
    return prompt

def create_json_prompt(
    marker_genes: Dict[str, List[str]],
    species: str,
    tissue: Optional[str] = None,
    additional_context: Optional[str] = None
) -> str:
    """
    Create a prompt for cell type annotation with JSON output format.
    
    Args:
        marker_genes: Dictionary mapping cluster names to lists of marker genes
        species: Species name (e.g., 'human', 'mouse')
        tissue: Tissue name (e.g., 'brain', 'blood')
        additional_context: Additional context to include in the prompt
        
    Returns:
        str: The generated prompt
    """
    return create_prompt(
        marker_genes=marker_genes,
        species=species,
        tissue=tissue,
        additional_context=additional_context,
        prompt_template=DEFAULT_JSON_PROMPT_TEMPLATE
    )

def create_discussion_prompt(
    cluster_id: str,
    marker_genes: List[str],
    model_votes: Dict[str, str],
    species: str,
    tissue: Optional[str] = None,
    previous_discussion: Optional[str] = None,
    prompt_template: Optional[str] = None
) -> str:
    """
    Create a prompt for facilitating discussion about a controversial cluster.
    
    Args:
        cluster_id: ID of the cluster
        marker_genes: List of marker genes for the cluster
        model_votes: Dictionary mapping model names to cell type annotations
        species: Species name (e.g., 'human', 'mouse')
        tissue: Tissue name (e.g., 'brain', 'blood')
        previous_discussion: Optional previous discussion text for iterative rounds
        prompt_template: Custom prompt template
        
    Returns:
        str: The generated prompt
    """
    write_log(f"Creating discussion prompt for cluster {cluster_id}")
    
    # Use default template if none provided
    if not prompt_template:
        prompt_template = DEFAULT_DISCUSSION_TEMPLATE
    
    # Default tissue if none provided
    tissue_text = tissue if tissue else "unknown tissue"
    
    # Format marker genes text
    marker_genes_text = ", ".join(marker_genes)
    
    # Format model votes text
    model_votes_lines = []
    for model, vote in model_votes.items():
        model_votes_lines.append(f"- {model}: {vote}")
    
    model_votes_text = "\n".join(model_votes_lines)
    
    # Modify template for iterative discussion if previous discussion exists
    if previous_discussion:
        # Create a modified template that includes previous discussion
        iterative_template = prompt_template.replace(
            "Your task:", 
            "Previous discussion round:\n{previous_discussion}\n\nYour task:"
        )
        
        # Fill in the template with previous discussion
        prompt = iterative_template.format(
            cluster_id=cluster_id,
            species=species,
            tissue=tissue_text,
            marker_genes=marker_genes_text,
            model_votes=model_votes_text,
            previous_discussion=previous_discussion
        )
    else:
        # Fill in the template without previous discussion
        prompt = prompt_template.format(
            cluster_id=cluster_id,
            species=species,
            tissue=tissue_text,
            marker_genes=marker_genes_text,
            model_votes=model_votes_text
        )
    
    write_log(f"Generated discussion prompt with {len(prompt)} characters")
    return prompt

def create_consensus_check_prompt(
    predictions: Dict[str, Dict[str, str]],
    species: str,
    tissue: Optional[str] = None,
    prompt_template: Optional[str] = None
) -> str:
    """
    Create a prompt for checking consensus across model predictions.
    
    Args:
        predictions: Dictionary mapping model names to dictionaries of cluster annotations
        species: Species name (e.g., 'human', 'mouse')
        tissue: Tissue name (e.g., 'brain', 'blood')
        prompt_template: Custom prompt template
        
    Returns:
        str: The generated prompt
    """
    write_log(f"Creating consensus check prompt for {len(predictions)} models")
    
    # Use default template if none provided
    if not prompt_template:
        prompt_template = DEFAULT_CONSENSUS_CHECK_TEMPLATE
    
    # Default tissue if none provided
    tissue_text = tissue if tissue else "unknown tissue"
    
    # Get all model names
    models = list(predictions.keys())
    
    # Get all cluster IDs
    clusters = set()
    for model_results in predictions.values():
        clusters.update(model_results.keys())
    clusters = sorted(list(clusters))
    
    # Format predictions text
    predictions_lines = []
    
    for cluster in clusters:
        predictions_lines.append(f"Cluster {cluster}:")
        for model in models:
            if cluster in predictions[model]:
                predictions_lines.append(f"- {model}: {predictions[model][cluster]}")
        predictions_lines.append("")
    
    predictions_text = "\n".join(predictions_lines)
    
    # Fill in the template
    prompt = prompt_template.format(
        species=species,
        tissue=tissue_text,
        predictions=predictions_text
    )
    
    write_log(f"Generated consensus check prompt with {len(prompt)} characters")
    return prompt

def create_discussion_consensus_check_prompt(
    cluster_id: str,
    discussion: str,
    proposed_cell_type: str,
    prompt_template: Optional[str] = None
) -> str:
    """
    Create a prompt for checking if consensus has been reached after a discussion round.
    
    Args:
        cluster_id: ID of the cluster being discussed
        discussion: The discussion text from the current round
        proposed_cell_type: The proposed cell type from the current round
        prompt_template: Custom prompt template
        
    Returns:
        str: The generated prompt
    """
    write_log(f"Creating consensus check prompt for cluster {cluster_id}")
    
    # Use default template if none provided
    if not prompt_template:
        prompt_template = DEFAULT_DISCUSSION_CONSENSUS_CHECK_TEMPLATE
    
    # Fill in the template
    prompt = prompt_template.format(
        cluster_id=cluster_id,
        discussion=discussion,
        proposed_cell_type=proposed_cell_type if proposed_cell_type else "Unclear"
    )
    
    write_log(f"Generated discussion consensus check prompt with {len(prompt)} characters")
    return prompt

def create_initial_discussion_prompt(
    cluster_id: str,
    marker_genes: List[str],
    species: str,
    tissue: Optional[str] = None
) -> str:
    """
    Create a prompt for initial cell type discussion about a cluster.
    
    Args:
        cluster_id: ID of the cluster
        marker_genes: List of marker genes for the cluster
        species: Species name (e.g., 'human', 'mouse')
        tissue: Tissue name (e.g., 'brain', 'blood')
        
    Returns:
        str: The generated prompt
    """
    write_log(f"Creating initial discussion prompt for cluster {cluster_id}")
    
    # Default tissue if none provided
    tissue_text = tissue if tissue else "unknown tissue"
    
    # Format marker genes text
    marker_genes_text = ", ".join(marker_genes)
    
    # Template for initial discussion
    template = """You are an expert in single-cell RNA-seq analysis, assigned to identify the cell type for a specific cluster.

Cluster ID: {cluster_id}
Species: {species}
Tissue: {tissue}

Marker genes: {marker_genes}

Your task:
1. Analyze these marker genes and their expression patterns
2. Consider the cell types that might express this combination of genes
3. Provide a detailed reasoning process
4. Determine the most likely cell type for this cluster

Give a thorough analysis, explaining which genes are most informative and why.
End with a clear cell type determination.
"""
    
    # Fill in the template
    prompt = template.format(
        cluster_id=cluster_id,
        species=species,
        tissue=tissue_text,
        marker_genes=marker_genes_text
    )
    
    write_log(f"Generated initial discussion prompt with {len(prompt)} characters")
    return prompt
