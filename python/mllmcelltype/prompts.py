"""Prompt generation module for LLMCellType."""

from __future__ import annotations

import string

from .logger import write_log
from .utils import cluster_sort_key
from .validation import normalize_text


def _format_marker_genes_for_prompt(
    marker_genes: dict[str, list[str]], cluster_format: str = "Cluster {}: {}"
) -> str:
    """Format marker genes consistently for prompts.

    Clusters are sorted in natural order to match the order shown in the
    rendered prompt.

    Args:
        marker_genes: Dictionary mapping cluster names to marker gene lists
        cluster_format: Format string for cluster entries

    Returns:
        str: Formatted marker genes text
    """
    marker_lines = []
    for cluster in sorted(marker_genes.keys(), key=cluster_sort_key):
        genes_str = ", ".join(str(g) for g in marker_genes[cluster])
        marker_lines.append(cluster_format.format(cluster, genes_str))
    return "\n".join(marker_lines)


# Default prompt template for single dataset annotation
DEFAULT_PROMPT_TEMPLATE = """You are an expert single-cell RNA-seq analyst specializing in cell type annotation.
I need you to identify cell types of {species} cells from {tissue}.
Below is a list of marker genes for each cluster.
Please assign the most likely cell type to each cluster based on the marker genes.

Return one line per cluster, IN THE ORDER SHOWN BELOW, using this format:
Cluster <exact input cluster ID>: <cell type>
Copy each input cluster ID exactly. Never renumber clusters or replace their IDs.

Only provide the cell type name for each cluster. Be concise but specific.
Some clusters can be a mixture of multiple cell types.

{context}
Here are the marker genes for each cluster:
{markers}
"""

# Reasoning prompt template that asks for structured JSON output per cluster
REASONING_PROMPT_TEMPLATE = """You are an expert single-cell RNA-seq analyst specializing in cell type annotation.
I need you to identify cell types of {species} cells from {tissue}.
Below is a list of marker genes for each cluster.
Please assign the most likely cell type to each cluster based on the marker genes.

For each cluster, return a JSON object with the following fields:
- "cluster_id": the cluster ID as shown below
- "cell_type": the annotated cell type
- "marker_genes": the marker genes from the provided gene set that support this annotation, comma-separated, preserving the original case exactly as shown below
- "gene_expression": a concise natural-language description of where each provided differential gene is typically expressed across cell types

Return a single JSON array containing one object per cluster, in the same order as the clusters shown below. Do not include markdown formatting or any explanation outside the JSON array.

{context}
Here are the marker genes for each cluster:
{markers}
"""

SUPPORTED_PROMPT_PLACEHOLDERS = ("species", "tissue", "markers", "context")


def _get_prompt_template_fields(prompt_template: str) -> set[str]:
    """Parse and validate exact named fields used by a prompt template."""
    try:
        fields = {
            field_name
            for _, field_name, _, _ in string.Formatter().parse(prompt_template)
            if field_name is not None
        }
    except ValueError as error:
        raise ValueError(f"Invalid prompt_template format: {error!s}") from error

    unsupported = sorted(fields - set(SUPPORTED_PROMPT_PLACEHOLDERS))
    if unsupported:
        raise ValueError(
            "Invalid prompt_template placeholder "
            f"'{unsupported[0]}'. Supported placeholders: "
            f"{', '.join(SUPPORTED_PROMPT_PLACEHOLDERS)}"
        )
    return fields


def validate_prompt_template(prompt_template: str | None) -> str | None:
    """Normalize and validate an optional annotation prompt template."""
    if prompt_template is not None and not isinstance(prompt_template, str):
        raise ValueError(
            f"prompt_template must be a string or None, got {type(prompt_template).__name__}"
        )
    normalized_template = normalize_text(prompt_template, "prompt_template")
    if normalized_template is None:
        return None
    _get_prompt_template_fields(normalized_template)
    return normalized_template


def _render_prompt_template(
    *,
    prompt_template: str,
    species: str,
    tissue_text: str,
    marker_text: str,
    context_text: str,
) -> str:
    """Render a validated prompt template with clear formatting errors."""
    try:
        return prompt_template.format(
            species=species,
            tissue=tissue_text,
            markers=marker_text,
            context=context_text,
        )
    except (AttributeError, IndexError, KeyError, ValueError) as error:
        raise ValueError(f"Invalid prompt_template format: {error!s}") from error


def create_consensus_check_prompt(
    annotations: list[str],
    consensus_threshold: float = 0.7,
    entropy_threshold: float = 1.0,
) -> str:
    """Create a prompt for checking consensus among different annotations.

    Args:
        annotations: List of cell type annotations from different models
        consensus_threshold: Minimum majority proportion required for consensus
        entropy_threshold: Maximum Shannon entropy allowed for consensus

    Returns:
        str: Formatted prompt for LLM to check consensus

    """
    prompt = f"""You are an expert in single-cell RNA-seq analysis and cell type annotation.

I need you to analyze the following cell type annotations from different models for the same cluster and determine if there is a consensus.

The annotations are:
{{annotations}}

Group annotations only when they name the same biological entity at the same level of granularity:
1. Treat synonymous nomenclature, capitalization, and formatting differences as equivalent.
2. Preserve biologically meaningful subtype or state qualifiers such as activated, mature, memory, regulatory, CD4+, and CD8+; do not merge labels when doing so would lose specificity.
3. Treat Unknown or Unclear as separate groups.

Calculate:
1. Consensus proportion = size of the unique largest group / total annotations.
2. Shannon entropy = -sum(p_i * log2(p_i)) over the annotation groups.
3. Consensus is reached only when there is a unique largest group, the consensus proportion is at least {consensus_threshold:g}, and entropy is at most {entropy_threshold:g}.

Respond with exactly 4 lines:
Line 1: 0 or 1 (consensus reached?)
Line 2: Consensus proportion as a decimal between 0 and 1
Line 3: Shannon entropy as a non-negative decimal
Line 4: The majority cell type, or Unknown when there is no unique majority

Only output these 4 lines, nothing else."""

    # Format the annotations
    formatted_annotations = "\n".join([f"- {anno}" for anno in annotations])

    # Replace the placeholder
    return prompt.replace("{annotations}", formatted_annotations)


def create_prompt(
    marker_genes: dict[str, list[str]],
    species: str,
    tissue: str | None = None,
    additional_context: str | None = None,
    prompt_template: str | None = None,
    return_reasoning: bool = False,
) -> str:
    """Create a prompt for cell type annotation.

    Args:
        marker_genes: Dictionary mapping cluster names to lists of marker genes
        species: Species name (e.g., 'human', 'mouse')
        tissue: Tissue name (e.g., 'brain', 'liver')
        additional_context: Additional context to include in the prompt
        prompt_template: Custom prompt template
        return_reasoning: If True, use the reasoning prompt template that asks
            for structured JSON output with cell_type, marker_genes, and
            gene_expression fields per cluster.

    Returns:
        str: The generated prompt

    """
    write_log(f"Creating prompt for {len(marker_genes)} clusters")

    species_text = normalize_text(species, "species", required=True)
    tissue_text = normalize_text(tissue, "tissue") or "unknown tissue"
    additional_context_text = normalize_text(additional_context, "additional_context")

    if prompt_template is None:
        prompt_template = REASONING_PROMPT_TEMPLATE if return_reasoning else DEFAULT_PROMPT_TEMPLATE
    else:
        prompt_template = validate_prompt_template(prompt_template)

    # Format marker genes text using helper function
    marker_text = _format_marker_genes_for_prompt(marker_genes)

    context_text = (
        f"Additional context: {additional_context_text}\n" if additional_context_text else ""
    )
    template_fields = _get_prompt_template_fields(prompt_template)

    # Fill in the template with clear validation errors.
    prompt = _render_prompt_template(
        prompt_template=prompt_template,
        species=species_text,
        tissue_text=tissue_text,
        marker_text=marker_text,
        context_text=context_text,
    )

    if context_text and "context" not in template_fields:
        prompt = f"{prompt.rstrip()}\n\n{context_text}"

    write_log(f"Generated prompt with {len(prompt)} characters")
    return prompt


def create_initial_discussion_prompt(
    cluster_id: str,
    marker_genes: list[str],
    initial_predictions: dict[str, str],
    species: str,
    tissue: str | None = None,
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
    if initial_predictions:
        predictions_text = "\n".join(
            f"- {model}: {prediction}" for model, prediction in initial_predictions.items()
        )
    else:
        predictions_text = "- (no confident initial predictions available)"

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
    tissue: str | None = None,
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


def create_cell_type_extraction_prompt(text: str) -> str:
    """Create a prompt to extract a concise cell type label from free-text.

    When model responses contain natural-language sentences instead of concise
    cell type labels (e.g., "This cluster is likely CD4+ T cells"), this prompt
    asks an LLM to produce a clean label (e.g., "CD4+ T cells").

    Args:
        text: Free-text containing a cell type reference

    Returns:
        str: Prompt for LLM to extract the cell type label
    """
    return (
        "Extract ONLY the cell type name from the text below.\n"
        'Reply with just the cell type (e.g., "CD4+ T cells", "NK cells", '
        '"B lymphocytes").\n'
        "Do not include any explanation or extra words.\n\n"
        f"Text: {text}"
    )


def create_discussion_consensus_prompt(
    round_responses: dict[str, str],
    consensus_threshold: float = 2 / 3,
    entropy_threshold: float = 1.0,
) -> str:
    """Create a prompt for LLM to extract cell types and check consensus.

    Unlike create_consensus_check_prompt which takes pre-extracted annotations,
    this function takes raw discussion responses and asks the LLM to perform
    both cell-type extraction and consensus checking in a single call.

    This mirrors the R implementation (check_consensus.R:488) where the LLM
    receives full discussion responses and produces clean labels + metrics.

    Args:
        round_responses: Dictionary mapping model identifiers to raw responses
        consensus_threshold: Agreement threshold for consensus determination
        entropy_threshold: Entropy threshold for consensus determination

    Returns:
        str: Formatted prompt for LLM consensus check
    """
    # Use anonymous model IDs (Model 1, 2, ...) to prevent provider-name bias
    responses_text = "\n\n".join(
        f"Model {i + 1}:\n{response}" for i, (_, response) in enumerate(round_responses.items())
    )

    return (
        "You are a cell type annotation expert. Below are different models' "
        "discussion responses for the same cell cluster.\n\n"
        "Your task:\n"
        "1. Extract the predicted cell type from each model's response\n"
        "2. Determine if the predictions reach consensus\n\n"
        f"RESPONSES:\n{responses_text}\n\n"
        "IMPORTANT GUIDELINES:\n"
        "1. Extract ONLY the cell type name from each response "
        '(e.g., "CD4+ T cells", not full sentences)\n'
        "2. Responses may contain structured predictions (with "
        '"CELL TYPE:" prefix) or free-text predictions\n'
        "3. Consider predictions as matching only when they refer to the same "
        "biological entity, allowing synonymous nomenclature, formatting, "
        "and capitalization differences\n"
        "4. Preserve biologically meaningful subtype or state qualifiers such as "
        "activated, mature, memory, regulatory, CD4+, and CD8+; do not merge "
        "labels when doing so would lose specificity\n"
        "5. Group only predictions that remain biologically equivalent at the "
        "same level of granularity\n"
        '6. If any prediction is "Unknown" or "Unclear", treat it as '
        "a separate group\n\n"
        "CALCULATE:\n"
        "1. Consensus Proportion = Number of models supporting the majority "
        "prediction / Total number of models\n"
        "2. Shannon Entropy = -sum(p_i * log2(p_i)) where p_i is the "
        "proportion of each unique prediction\n"
        f"3. Consensus is reached if Consensus Proportion >= "
        f"{consensus_threshold:.2f} AND Entropy <= "
        f"{entropy_threshold:.2f}\n\n"
        "RESPOND WITH EXACTLY 4 LINES:\n"
        "Line 1: 1 if consensus is reached, 0 if not\n"
        "Line 2: Consensus Proportion (a decimal between 0 and 1)\n"
        "Line 3: Shannon Entropy (a decimal number)\n"
        "Line 4: The majority cell type (concise label only, "
        'e.g., "CD4+ T cells")\n\n'
        "RESPOND WITH EXACTLY FOUR LINES AS SPECIFIED ABOVE."
    )
