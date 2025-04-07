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

IMPORTANT: Provide your answers in the EXACT format below, with one cluster per line:
Cluster 0: [cell type]
Cluster 1: [cell type]
...and so on, IN NUMERICAL ORDER.

Only provide the cell type name for each cluster. Be concise but specific.
Some clusters can be a mixture of multiple cell types.

Here are the marker genes for each cluster:
{markers}
"""

# Original simpler template
SIMPLE_PROMPT_TEMPLATE = """You are a cell type annotation expert. Below are marker genes for different cell clusters in {context}.

{clusters}

For each numbered cluster, provide ONLY the cell type name in a new line, without any explanation.

IMPORTANT: Provide your answers in the EXACT format below, with one cluster per line:
Cluster 0: [Cell Type]
Cluster 1: [Cell Type]
...and so on, IN NUMERICAL ORDER.

Only provide the cell type name, without any additional text or explanations.
"""

# Default prompt template for batch annotation
DEFAULT_BATCH_PROMPT_TEMPLATE = """You are an expert single-cell RNA-seq analyst specializing in cell type annotation.
I need you to identify cell types of {species} cells from {tissue}.
Below are lists of marker genes for clusters from multiple datasets.
Please assign the most likely cell type to each cluster based on the marker genes.

IMPORTANT: Format your response EXACTLY as follows for each set:
Set 1:
Cluster 0: [cell type]
Cluster 1: [cell type]
...and so on, IN NUMERICAL ORDER.

Set 2:
Cluster 0: [cell type]
Cluster 1: [cell type]
...and so on, IN NUMERICAL ORDER.

Only provide the cell type name for each cluster. Be concise but specific.
"""

def create_consensus_check_prompt(annotations: List[str]) -> str:
    """
    Create a prompt for checking consensus among different annotations.

    Args:
        annotations: List of cell type annotations from different models

    Returns:
        str: Formatted prompt for LLM to check consensus
    """
    prompt = """You are an expert in single-cell RNA-seq analysis and cell type annotation.

I need you to analyze the following cell type annotations from different models for the same cluster and determine if there is a consensus.

The annotations are:
{annotations}

Please analyze these annotations and determine:
1. If there is a consensus (1 for yes, 0 for no)
2. The consensus proportion (between 0 and 1)
3. An entropy value measuring the diversity of opinions (higher means more diverse)
4. The best consensus annotation

Respond with exactly 4 lines:
Line 1: 0 or 1 (consensus reached?)
Line 2: Consensus proportion (e.g., 0.75)
Line 3: Entropy value (e.g., 0.85)
Line 4: The consensus cell type (or most likely if no clear consensus)

Only output these 4 lines, nothing else."""

    # Format the annotations
    formatted_annotations = "\n".join([f"- {anno}" for anno in annotations])

    # Replace the placeholder
    prompt = prompt.replace("{annotations}", formatted_annotations)

    return prompt

# Original simpler batch template
SIMPLE_BATCH_PROMPT_TEMPLATE = """You are a cell type annotation expert. Below are marker genes for different cell clusters in {context}.

{clusters}

For each set and its numbered clusters, provide ONLY the cell type name in a new line, without any explanation.

IMPORTANT: Provide your answers in the EXACT format below, with one cluster per line:
Set 1:
Cluster 0: [Cell Type]
Cluster 1: [Cell Type]
...and so on, IN NUMERICAL ORDER.

Set 2:
Cluster 0: [Cell Type]
Cluster 1: [Cell Type]
...and so on, IN NUMERICAL ORDER.

Only provide the cell type name, without any additional text or explanations.
"""

# Default JSON format prompt template
DEFAULT_JSON_PROMPT_TEMPLATE = """You are an expert single-cell RNA-seq analyst specializing in cell type annotation.
I need you to identify cell types of {species} cells from {tissue}.
Below is a list of marker genes for each cluster.
Please assign the most likely cell type to each cluster based on the marker genes.

IMPORTANT: Format your response as a valid JSON object as follows, using the EXACT SAME cluster IDs as provided in the input, and maintaining NUMERICAL ORDER:
```json
{{
  "annotations": [
    {{
      "cluster": "0",
      "cell_type": "T cells",
      "confidence": "high",
      "key_markers": ["CD3D", "CD3G", "CD3E"]
    }},
    {{
      "cluster": "1",
      "cell_type": "B cells",
      "confidence": "high",
      "key_markers": ["CD19", "CD79A", "MS4A1"]
    }},
    ...
  ]
}}
```

For each cluster, provide:
1. The cluster ID (use the SAME ID as in the input)
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
5. Calculate and provide the following metrics to quantify the consensus:

   a) Consensus Proportion (CP):
      CP = Number of models supporting the majority prediction / Total number of models
      Example: If 3 out of 4 models predict the same cell type, CP = 3/4 = 0.75

   b) Shannon Entropy (H):
      H = -∑(p_i * log2(p_i)) for all unique predictions i
      where p_i is the proportion of models predicting cell type i
      Example: If 3 models predict 'T cell' and 1 predicts 'NK cell', then:
      p_T = 3/4 = 0.75, p_NK = 1/4 = 0.25
      H = -(0.75*log2(0.75) + 0.25*log2(0.25)) = 0.81
      H ranges from 0 (perfect consensus) to log2(C) where C is the number of unique predictions

Provide a well-reasoned analysis with evidence from literature or known marker-cell type associations.
End with a clear final decision on the correct cell type, including:
- Final cell type determination
- Key supporting marker genes
- Consensus Proportion (CP): Calculate and provide the exact value (0-1)
- Shannon Entropy (H): Calculate and provide the exact value (0 for perfect consensus)

You MUST provide numerical values for both CP and H, not just qualitative descriptions.
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
1. Analyze the discussion and determine if there is consensus on the cell type annotation
2. Normalize minor differences in terminology (e.g., 'NK cells' = 'Natural Killer cells')
3. Calculate the following metrics:
   - Consensus Proportion = Number of supporting opinions / Total number of opinions
   - Shannon Entropy = -sum(p_i * log2(p_i)) where p_i is the proportion of each unique opinion

Determine if consensus is reached (Consensus Proportion > 2/3 AND Entropy <= 1.0)

RESPONSE FORMAT:
Line 1: 1 if consensus is reached, 0 if not
Line 2: Consensus Proportion (a decimal between 0 and 1)
Line 3: Shannon Entropy (a decimal number)
Line 4: The majority cell type prediction

RESPOND WITH EXACTLY FOUR LINES AS SPECIFIED ABOVE.
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

def create_model_consensus_check_prompt(
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
