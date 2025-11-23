# Compare predictions from different models

This function runs the same input through multiple models and compares
their predictions. It provides both individual predictions and a
consensus analysis.

## Usage

``` r
compare_model_predictions(
  input,
  tissue_name,
  models = c("claude-sonnet-4-5-20250929", "claude-opus-4-1-20250805", "gpt-5",
    "gemini-2.5-pro", "deepseek-r1", "o1", "grok-3-latest"),
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

- OpenAI: 'gpt-5', 'gpt-5-mini', 'gpt-5-nano', 'gpt-4o', 'gpt-4o-mini',
  'gpt-4.1', 'gpt-4.1-mini', 'gpt-4.1-nano', 'gpt-4-turbo',
  'gpt-3.5-turbo', 'o1', 'o1-mini', 'o1-preview', 'o1-pro'

- Anthropic: 'claude-opus-4-1-20250805', 'claude-sonnet-4-20250514',
  'claude-opus-4-20250514', 'claude-3-7-sonnet-20250219',
  'claude-3-5-sonnet-20241022', 'claude-3-5-haiku-20241022',
  'claude-3-opus-20240229'

- DeepSeek: 'deepseek-chat', 'deepseek-r1', 'deepseek-r1-zero',
  'deepseek-reasoner'

- Google: 'gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-2.0-flash',
  'gemini-2.0-flash-lite', 'gemini-1.5-pro-latest',
  'gemini-1.5-flash-latest', 'gemini-1.5-flash-8b'

- Alibaba: 'qwen-max-2025-01-25', 'qwen3-72b'

- Stepfun: 'step-2-16k', 'step-2-mini', 'step-1-8k'

- Zhipu: 'glm-4-plus', 'glm-3-turbo'

- MiniMax: 'minimax-text-01'

- X.AI: 'grok-3-latest', 'grok-3', 'grok-3-fast', 'grok-3-fast-latest',
  'grok-3-mini', 'grok-3-mini-latest', 'grok-3-mini-fast',
  'grok-3-mini-fast-latest'

- OpenRouter: Provides access to models from multiple providers through
  a single API. Format: 'provider/model-name'

  - OpenAI models: 'openai/gpt-5', 'openai/gpt-5-mini', 'openai/gpt-4o',
    'openai/gpt-4o-mini', 'openai/gpt-4-turbo', 'openai/gpt-4',
    'openai/gpt-3.5-turbo'

  - Anthropic models: 'anthropic/claude-opus-4.1',
    'anthropic/claude-sonnet-4', 'anthropic/claude-opus-4',
    'anthropic/claude-3.7-sonnet', 'anthropic/claude-3.5-sonnet',
    'anthropic/claude-3.5-haiku', 'anthropic/claude-3-opus'

  - Meta models: 'meta-llama/llama-3-70b-instruct',
    'meta-llama/llama-3-8b-instruct', 'meta-llama/llama-2-70b-chat'

  - Google models: 'google/gemini-2.5-pro', 'google/gemini-2.5-flash',
    'google/gemini-2.0-flash', 'google/gemini-1.5-pro-latest',
    'google/gemini-1.5-flash'

  - Mistral models: 'mistralai/mistral-large',
    'mistralai/mistral-medium', 'mistralai/mistral-small'

  - Other models: 'microsoft/mai-ds-r1', 'perplexity/sonar-small-chat',
    'cohere/command-r', 'deepseek/deepseek-chat', 'thudm/glm-z1-32b'

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
      "claude-3-opus" = "sk-ant-api03-specific-key-for-opus"
    )

## Examples

``` r
if (FALSE) { # \dontrun{
# Compare predictions using different models
api_keys <- list(
  "claude-sonnet-4-20250514" = "your-anthropic-key",
  "deepseek-reasoner" = "your-deepseek-key",
  "gemini-1.5-pro" = "your-gemini-key",
  "qwen-max-2025-01-25" = "your-qwen-key"
)

results <- compare_model_predictions(
  input = list(gs1=c('CD4','CD3D'), gs2='CD14'),
  tissue_name = 'PBMC',
  api_keys = api_keys
)
} # }
```
