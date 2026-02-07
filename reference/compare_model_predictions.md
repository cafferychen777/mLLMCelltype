# Compare predictions from different models

This function runs the same input through multiple models and compares
their predictions. It provides both individual predictions and a
consensus analysis.

## Usage

``` r
compare_model_predictions(
  input,
  tissue_name,
  models = c("claude-opus-4.6", "gpt-5.2", "gemini-3-pro", "deepseek-r1", "o3-pro",
    "grok-4.1"),
  api_keys,
  top_gene_count = 10,
  consensus_threshold = 0.5
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
  keys

- top_gene_count:

  Number of top genes to use per cluster when input is from Seurat.
  Default: 10

- consensus_threshold:

  Minimum agreement threshold for consensus (0-1). Default: 0.5

## Value

List containing individual model predictions and consensus analysis

## Note

This function uses create_standardization_prompt from prompt_templates.R
Supported models:

- OpenAI: 'gpt-5.2', 'gpt-5.1', 'gpt-5', 'gpt-4.1', 'gpt-4o', 'o3-pro',
  'o3', 'o4-mini', 'o1', 'o1-pro'

- Anthropic: 'claude-opus-4-6-20260205', 'claude-opus-4-5-20251101',
  'claude-sonnet-4-5-20250929', 'claude-haiku-4-5-20251001',
  'claude-opus-4-1-20250805', 'claude-sonnet-4-20250514',
  'claude-3-7-sonnet-20250219'

- DeepSeek: 'deepseek-chat', 'deepseek-reasoner', 'deepseek-r1'

- Google: 'gemini-3-pro', 'gemini-3-flash', 'gemini-2.5-pro',
  'gemini-2.5-flash', 'gemini-2.0-flash'

- Alibaba: 'qwen3-max', 'qwen-max-2025-01-25', 'qwen-plus'

- Stepfun: 'step-3', 'step-2-16k', 'step-2-mini'

- Zhipu: 'glm-4.7', 'glm-4-plus'

- MiniMax: 'minimax-m2.1', 'minimax-m2', 'MiniMax-Text-01'

- X.AI: 'grok-4', 'grok-4.1', 'grok-4-heavy', 'grok-3', 'grok-3-fast',
  'grok-3-mini'

- OpenRouter: Provides access to models from multiple providers through
  a single API. Format: 'provider/model-name'

  - OpenAI models: 'openai/gpt-5.2', 'openai/gpt-5', 'openai/o3-pro',
    'openai/o4-mini'

  - Anthropic models: 'anthropic/claude-opus-4.5',
    'anthropic/claude-sonnet-4.5', 'anthropic/claude-haiku-4.5'

  - Meta models: 'meta-llama/llama-4-maverick',
    'meta-llama/llama-4-scout', 'meta-llama/llama-3.3-70b-instruct'

  - Google models: 'google/gemini-3-pro', 'google/gemini-3-flash',
    'google/gemini-2.5-pro'

  - Mistral models: 'mistralai/mistral-large',
    'mistralai/magistral-medium-2506'

  - Other models: 'deepseek/deepseek-r1', 'deepseek/deepseek-chat-v3.1',
    'microsoft/mai-ds-r1'

1.  With provider names as keys:
    `list("openai" = "sk-...", "anthropic" = "sk-ant-...", "openrouter" = "sk-or-...")`

2.  With model names as keys:
    `list("gpt-5" = "sk-...", "claude-sonnet-4-5-20250929" = "sk-ant-...")`

The system first tries to find the API key using the provider name. If
not found, it then tries using the model name. Example:

    api_keys <- list(
      "openai" = Sys.getenv("OPENAI_API_KEY"),
      "anthropic" = Sys.getenv("ANTHROPIC_API_KEY"),
      "openrouter" = Sys.getenv("OPENROUTER_API_KEY"),
      "claude-opus-4-6-20260205" = "sk-ant-api03-specific-key-for-opus"
    )

## Examples

``` r
if (FALSE) { # \dontrun{
# Compare predictions using different models
api_keys <- list(
  "claude-sonnet-4-5-20250929" = "your-anthropic-key",
  "deepseek-reasoner" = "your-deepseek-key",
  "gemini-3-pro" = "your-gemini-key",
  "qwen3-max" = "your-qwen-key"
)

results <- compare_model_predictions(
  input = list(gs1=c('CD4','CD3D'), gs2='CD14'),
  tissue_name = 'PBMC',
  api_keys = api_keys
)
} # }
```
