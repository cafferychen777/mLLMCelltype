# Cell Type Annotation with Multi-LLM Framework

A comprehensive function for automated cell type annotation using
multiple Large Language Models (LLMs). This function supports both
Seurat's differential gene expression results and custom gene lists as
input. It implements a sophisticated annotation pipeline that leverages
state-of-the-art LLMs to identify cell types based on marker gene
expression patterns.

- A data frame from Seurat's FindAllMarkers() function containing
  differential gene expression results (must have columns: 'cluster',
  'gene', and 'avg_log2FC'). The function will select the top genes
  based on avg_log2FC for each cluster.

- A list where each element has a 'genes' field containing marker genes
  for a cluster. This can be in one of these formats:

  - Named with cluster IDs: list("0" = list(genes = c(...)), "1" =
    list(genes = c(...)))

  - Named with cell type names: list(t_cells = list(genes = c(...)),
    b_cells = list(genes = c(...)))

  - Unnamed list: list(list(genes = c(...)), list(genes = c(...)))

- Cluster IDs are preserved as-is. The function does not modify or
  re-index cluster IDs. 'mouse brain'). This helps provide context for
  more accurate annotations.

- OpenAI: 'gpt-5.5', 'gpt-5.4', 'gpt-5.4-mini'

- Anthropic: 'claude-opus-4-7', 'claude-opus-4-6', 'claude-sonnet-4-6',
  'claude-haiku-4-5-20251001'

- DeepSeek: 'deepseek-v4-flash', 'deepseek-v4-pro'

- Google: 'gemini-3.1-pro-preview', 'gemini-3-flash-preview',
  'gemini-3.1-flash-lite'

- Alibaba: 'qwen3.6-max-preview', 'qwen3.6-plus', 'qwen3.6-flash'

- Stepfun: 'step-3.5-flash', 'step-3.5-flash-2603', 'step-3'

- Zhipu/Z.AI: 'glm-5.1', 'glm-5-turbo', 'glm-5'

- MiniMax: 'MiniMax-M2.7', 'MiniMax-M2.7-highspeed', 'MiniMax-M2.5'

- X.AI: 'grok-4.3', 'grok-4.3-latest', 'grok-latest'

- OpenRouter: Provides access to models from multiple providers through
  a single API. Format: 'provider/model-name'

  - OpenAI models: 'openai/gpt-5.5', 'openai/gpt-5.4-mini'

  - Anthropic models: 'anthropic/claude-opus-4.7',
    'anthropic/claude-sonnet-4.6'

  - Google models: 'google/gemini-3.1-pro-preview',
    'google/gemini-3-flash-preview'

  - X.AI models: 'x-ai/grok-4.3'

  - Stepfun models: 'stepfun/step-3.5-flash' Each provider requires a
    specific API key format and authentication method:

- OpenAI: "sk-..." (obtain from OpenAI platform)

- Anthropic: "sk-ant-..." (obtain from Anthropic console)

- Google: A Google API key for Gemini models (obtain from Google AI)

- DeepSeek: API key from DeepSeek platform

- Qwen: API key from Alibaba Cloud

- Stepfun: API key from Stepfun AI

- Zhipu: API key from Zhipu AI

- MiniMax: API key from MiniMax

- X.AI: API key for Grok models

- OpenRouter: "sk-or-..." (obtain from OpenRouter) OpenRouter provides
  access to multiple models through a single API key

The API key can be provided directly or stored in environment variables:

    # Direct API key
    result <- annotate_cell_types(input, tissue_name, model="gpt-5.5",
                                 api_key="sk-...")

    # Using environment variables
    Sys.setenv(OPENAI_API_KEY="sk-...")
    Sys.setenv(ANTHROPIC_API_KEY="sk-ant-...")
    Sys.setenv(OPENROUTER_API_KEY="sk-or-...")

    # Then use with environment variables
    result <- annotate_cell_types(input, tissue_name, model="claude-sonnet-4-6",
                                 api_key=Sys.getenv("ANTHROPIC_API_KEY"))

If NA, returns the generated prompt without making an API call, which is
useful for reviewing the prompt before sending it to the API. when input
is from Seurat's FindAllMarkers(). Default: 10

- A single character string: Applied to all providers (e.g.,
  "https://api.proxy.com/v1")

- A named list: Provider-specific URLs (e.g., list(openai =
  "https://openai-proxy.com/v1", anthropic =
  "https://anthropic-proxy.com/v1")). This is useful for:

  - Users accessing international APIs through proxies

  - Enterprise users with internal API gateways

  - Development/testing with local or alternative endpoints If NULL
    (default), uses official API endpoints for each provider.

## Usage

``` r
annotate_cell_types(
  input,
  tissue_name,
  model = "gpt-5.5",
  api_key = NA,
  top_gene_count = 10,
  debug = FALSE,
  base_urls = NULL,
  return_reasoning = FALSE
)
```

## Arguments

- input:

  Either a data frame from Seurat's FindAllMarkers() containing columns
  'cluster', 'gene', and 'avg_log2FC', or a list with 'genes' field for
  each cluster

- tissue_name:

  Required tissue context (e.g., 'human PBMC', 'mouse brain') for more
  accurate annotations

- model:

  Model name to use. Default: 'gpt-5.5'. See details for supported
  models

- api_key:

  API key for the selected model provider as a non-empty character
  scalar. If `NA`, returns prompt only.

- top_gene_count:

  Number of top genes to use per cluster when input is from Seurat.
  Default: 10

- debug:

  Logical indicating whether to enable debug output. Default: FALSE

- base_urls:

  Optional base URLs for API endpoints. Can be a string or named list
  for custom endpoints

- return_reasoning:

  Logical. If `TRUE`, returns a structured list per cluster containing
  `cell_type`, `marker_genes`, and `gene_expression` fields instead of
  plain labels. Default: `FALSE`.

## Value

When `api_key` is provided and `return_reasoning` is `FALSE`, the
provider response split by newline as a character vector. When
`return_reasoning` is `TRUE`, a named list where each element is a list
with `cell_type`, `marker_genes`, and `gene_expression`. When `api_key`
is `NA`, the generated prompt string.

## See also

- [`Seurat::FindAllMarkers()`](https://satijalab.org/seurat/reference/FindAllMarkers.html)

- [`get_provider()`](https://cafferychen777.github.io/mLLMCelltype/reference/get_provider.md)

## Examples

``` r
# Example 1: Using custom gene lists, returning prompt only (no API call)
annotate_cell_types(
  input = list(
    t_cells = list(genes = c('CD3D', 'CD3E', 'CD3G', 'CD28')),
    b_cells = list(genes = c('CD19', 'CD79A', 'CD79B', 'MS4A1')),
    monocytes = list(genes = c('CD14', 'CD68', 'CSF1R', 'FCGR3A'))
  ),
  tissue_name = 'human PBMC',
  model = 'gpt-5.5',
  api_key = NA  # Returns prompt only without making API call
)
#> [1] "You are an expert single-cell RNA-seq analyst specializing in cell type annotation. Infer the most likely biological cell type for every cell cluster in human PBMC from its marker genes.\n\nMarker genes:\nt_cells: CD3D, CD3E, CD3G, CD28\nb_cells: CD19, CD79A, CD79B, MS4A1\nmonocytes: CD14, CD68, CSF1R, FCGR3A\n\nThe marker genes above are input evidence, not answers. Return exactly 3 non-empty lines in the same order as the clusters shown above. Each line must contain only one inferred biological cell type name (for example, T cell or macrophage). Do not repeat or output marker genes. Do not include cluster IDs, numbering, explanations, headings, or markdown."

# Example 2: Using with Seurat pipeline and OpenAI model
if (FALSE) { # \dontrun{
library(Seurat)

# Load example data
data("pbmc_small")

# Find marker genes
all.markers <- FindAllMarkers(
  object = pbmc_small,
  only.pos = TRUE,
  min.pct = 0.25,
  logfc.threshold = 0.25
)

# Set API key in environment variable (recommended approach)
Sys.setenv(OPENAI_API_KEY = "your-openai-api-key")

# Get cell type annotations using OpenAI model
openai_annotations <- annotate_cell_types(
  input = all.markers,
  tissue_name = 'human PBMC',
  model = 'gpt-5.5',
  api_key = Sys.getenv("OPENAI_API_KEY"),
  top_gene_count = 15
)

# Example 3: Using Anthropic Claude model
Sys.setenv(ANTHROPIC_API_KEY = "your-anthropic-api-key")

claude_annotations <- annotate_cell_types(
  input = all.markers,
  tissue_name = 'human PBMC',
  model = 'claude-opus-4-7',
  api_key = Sys.getenv("ANTHROPIC_API_KEY"),
  top_gene_count = 15
)

# Example 4: Using OpenRouter to access multiple models
Sys.setenv(OPENROUTER_API_KEY = "your-openrouter-api-key")

# Access OpenAI models through OpenRouter
openrouter_gpt4_annotations <- annotate_cell_types(
  input = all.markers,
  tissue_name = 'human PBMC',
  model = 'openai/gpt-5.5',  # Note the provider/model format
  api_key = Sys.getenv("OPENROUTER_API_KEY"),
  top_gene_count = 15
)

# Access Anthropic models through OpenRouter
openrouter_claude_annotations <- annotate_cell_types(
  input = all.markers,
  tissue_name = 'human PBMC',
  model = 'anthropic/claude-opus-4.6',  # Note the provider/model format
  api_key = Sys.getenv("OPENROUTER_API_KEY"),
  top_gene_count = 15
)

# Example 5: Using with mouse brain data
mouse_annotations <- annotate_cell_types(
  input = mouse_markers,  # Your mouse marker genes
  tissue_name = 'mouse brain',  # Specify correct tissue for context
  model = 'gpt-5.5',
  api_key = Sys.getenv("OPENAI_API_KEY"),
  top_gene_count = 20,  # Use more genes for complex tissues
  debug = TRUE  # Enable debug output
)
} # }
```
