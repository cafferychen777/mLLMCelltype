"""Prompt generation module for LLMCellType."""

from __future__ import annotations

from typing import Optional

from .logger import write_log


def _format_marker_genes_for_prompt(
    marker_genes: dict[str, list[str]], cluster_format: str = "Cluster {}: {}"
) -> str:
    """Format marker genes consistently for prompts.

    Args:
        marker_genes: Dictionary mapping cluster names to marker gene lists
        cluster_format: Format string for cluster entries

    Returns:
        str: Formatted marker genes text
    """
    marker_lines = []
    for cluster, genes in marker_genes.items():
        genes_str = ", ".join(genes)
        marker_lines.append(cluster_format.format(cluster, genes_str))
    return "\n".join(marker_lines)


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


def create_consensus_check_prompt(annotations: list[str]) -> str:
    """Create a prompt for checking consensus among different annotations.

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
    return prompt.replace("{annotations}", formatted_annotations)


def create_prompt(
    marker_genes: dict[str, list[str]],
    species: str,
    tissue: Optional[str] = None,
    additional_context: Optional[str] = None,
    prompt_template: Optional[str] = None,
) -> str:
    """Create a prompt for cell type annotation.

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

    # Default tissue if none provided
    tissue_text = tissue if tissue else "unknown tissue"

    # Format marker genes text using helper function
    marker_text = _format_marker_genes_for_prompt(marker_genes)

    # Add additional context if provided
    context_text = f"\nAdditional context: {additional_context}\n" if additional_context else ""

    # Fill in the template
    prompt = prompt_template.format(species=species, tissue=tissue_text, markers=marker_text)

    # Add context
    if context_text:
        sections = prompt.split("Here are the marker genes for each cluster:")
        if len(sections) == 2:
            prompt = f"{sections[0]}{context_text}Here are the marker genes for each cluster:{sections[1]}"
        else:
            prompt = f"{prompt}{context_text}"

    write_log(f"Generated prompt with {len(prompt)} characters")
    return prompt


def create_initial_discussion_prompt(
    cluster_id: str,
    marker_genes: list[str],
    initial_predictions: dict[str, str],
    species: str,
    tissue: Optional[str] = None,
) -> str:
    """Create a prompt for the initial round of multi-model discussion.

    This prompt is used when multiple models participate in discussing
    a controversial cluster annotation.

    Args:
        cluster_id: ID of the cluster
        marker_genes: List of marker genes for the cluster
        initial_predictions: Dictionary mapping model names to their initial predictions
        species: Species name (e.g., 'human', 'mouse')
        tissue: Tissue name (e.g., 'brain', 'blood')

    Returns:
        str: The generated prompt for initial discussion round
    """
    write_log(f"Creating initial discussion prompt for cluster {cluster_id}")

    tissue_text = tissue if tissue else "unknown tissue"
    marker_genes_text = ", ".join(marker_genes)

    # Format initial predictions
    predictions_text = "\n".join(
        f"- {model}: {prediction}"
        for model, prediction in initial_predictions.items()
    )

    prompt = f"""We are analyzing cluster {cluster_id} with the following marker genes: {marker_genes_text}
Species: {species}
Tissue: {tissue_text}

Different models have made different predictions:
{predictions_text}

Please provide your cell type prediction using the Toulmin argumentation model:

1. CLAIM: State your clear cell type prediction
2. GROUNDS: Present specific marker genes that support your claim
3. WARRANT: Explain why these genes indicate this cell type
4. BACKING: Provide references or established knowledge
5. QUALIFIER: Indicate your certainty level (definite, probable, possible)
6. REBUTTAL: Address counter-arguments or other models' predictions

Format your response as:
CELL TYPE: [your predicted cell type]
GROUNDS: [specific marker genes supporting your claim]
WARRANT: [logical connection between evidence and claim]
BACKING: [additional support for your reasoning]
QUALIFIER: [degree of certainty]
REBUTTAL: [addressing counter-arguments]"""

    write_log(f"Generated initial discussion prompt with {len(prompt)} characters")
    return prompt


def create_discussion_prompt(
    cluster_id: str,
    marker_genes: list[str],
    previous_rounds: list[dict[str, str]],
    round_number: int,
    species: str,
    tissue: Optional[str] = None,
) -> str:
    """Create a prompt for subsequent rounds of multi-model discussion.

    This prompt includes the discussion history from all previous rounds,
    allowing each model to see and respond to other models' arguments.

    Args:
        cluster_id: ID of the cluster
        marker_genes: List of marker genes for the cluster
        previous_rounds: List of dicts, each containing model responses for a round
            Example: [{"gpt-5": "response1", "claude": "response2"}, ...]
        round_number: Current round number (2, 3, ...)
        species: Species name (e.g., 'human', 'mouse')
        tissue: Tissue name (e.g., 'brain', 'blood')

    Returns:
        str: The generated prompt for the discussion round
    """
    write_log(f"Creating discussion prompt for cluster {cluster_id}, round {round_number}")

    tissue_text = tissue if tissue else "unknown tissue"
    marker_genes_text = ", ".join(marker_genes)

    # Format previous discussion history
    discussion_history_parts = []
    for round_idx, round_responses in enumerate(previous_rounds, start=1):
        round_text = f"Round {round_idx}:\n"
        for model_name, response in round_responses.items():
            round_text += f"\n{model_name}:\n{response}\n"
        discussion_history_parts.append(round_text)

    discussion_history = "\n".join(discussion_history_parts)

    prompt = f"""We are continuing the discussion for cluster {cluster_id}.
Marker genes: {marker_genes_text}
Species: {species}
Tissue: {tissue_text}

Previous discussion:
{discussion_history}

This is round {round_number} of the discussion.

Using the Toulmin argumentation model, please structure your response:

1. CLAIM: State your clear cell type prediction
2. GROUNDS: Present specific marker genes that support your claim
3. WARRANT: Explain why these genes indicate this cell type
4. BACKING: Provide references or established knowledge
5. QUALIFIER: Indicate your certainty level
6. REBUTTAL: Address counter-arguments or other models' predictions

Based on previous discussion, also indicate:
- Whether you agree or disagree with any emerging consensus
- If you've revised your previous position, explain why

Format your response as:
CELL TYPE: [your current prediction]
GROUNDS: [specific marker genes supporting your claim]
WARRANT: [logical connection between evidence and claim]
BACKING: [additional support for your reasoning]
QUALIFIER: [degree of certainty]
REBUTTAL: [addressing counter-arguments]
CONSENSUS STATUS: [Agree/Disagree with emerging consensus]"""

    write_log(f"Generated discussion prompt with {len(prompt)} characters")
    return prompt
