# Compare predictions from different models

This function runs the same input through multiple models and compares
their predictions. It provides both individual predictions and a
consensus analysis.

## Usage

``` r
compare_model_predictions(
  input,
  tissue_name,
  models = c("claude-opus-4-7", "gpt-5.5", "gemini-3.1-pro-preview", "deepseek-v4-flash",
    "qwen3.6-plus", "grok-4.3"),
  api_keys,
  top_gene_count = 10,
  consensus_threshold = 0.5,
  base_urls = NULL
)
```

## Arguments

- input:

  Either a data frame from Seurat's FindAllMarkers() containing columns
  'cluster', 'gene', and 'avg_log2FC', or a list with 'genes' field for
  each cluster

- tissue_name:

  Tissue context (e.g., 'human PBMC', 'mouse brain') for more accurate
  annotations

- models:

  Vector of model names to use for comparison. Default includes top
  models from each provider

- api_keys:

  Named list of API keys for the models, with provider or model names as
  keys. Every model in `models` must resolve to a non-NULL API key.

- top_gene_count:

  Number of top genes to use per cluster when input is from Seurat.
  Default: 10

- consensus_threshold:

  Minimum agreement threshold for consensus (0-1). Default: 0.5.
  Consensus is only evaluated when at least two non-missing model
  predictions are available for a cluster.

- base_urls:

  Optional base URLs for API endpoints. Can be a string or named list
  for provider-specific custom endpoints.

## Value

List containing individual model predictions and consensus analysis If a
cluster has fewer than two valid predictions after alignment/padding,
its consensus-related outputs are `NA`.

## Note

This function uses create_standardization_prompt from prompt_templates.R
Supported models:

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

  - Stepfun models: 'stepfun/step-3.5-flash'

1.  With provider names as keys:
    `list("openai" = "sk-...", "anthropic" = "sk-ant-...", "openrouter" = "sk-or-...")`

2.  With model names as keys:
    `list("gpt-5.5" = "sk-...", "claude-sonnet-4-6" = "sk-ant-...")`

The system first tries to find the API key using the provider name. If
not found, it then tries using the model name. Example:

    api_keys <- list(
      "openai" = Sys.getenv("OPENAI_API_KEY"),
      "anthropic" = Sys.getenv("ANTHROPIC_API_KEY"),
      "openrouter" = Sys.getenv("OPENROUTER_API_KEY"),
      "claude-opus-4-7" = "your-claude-opus-key"
    )

## Examples

``` r
if (FALSE) { # \dontrun{
# Compare predictions using different models
api_keys <- list(
  "claude-sonnet-4-6" = "your-anthropic-key",
  "deepseek-v4-pro" = "your-deepseek-key",
  "gemini-3.1-pro-preview" = "your-gemini-key",
  "qwen3.6-plus" = "your-qwen-key"
)

results <- compare_model_predictions(
  input = list(gs1=c('CD4','CD3D'), gs2='CD14'),
  tissue_name = 'PBMC',
  api_keys = api_keys
)
} # }
```
