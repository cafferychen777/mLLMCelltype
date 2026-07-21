# Determine provider from model name

This function determines the appropriate provider (e.g., OpenAI,
Anthropic, Google, OpenRouter) based on the model name. Uses
prefix-based matching for efficient and maintainable provider detection.
New models following existing naming conventions are automatically
supported.

## Usage

``` r
get_provider(model)
```

## Arguments

- model:

  Character string specifying the model name (e.g., "gpt-5.5",
  "claude-opus-4-7").

## Value

Character string of the provider name (e.g., "openai", "anthropic").

## Details

Supported providers and model prefixes:

- OpenAI: gpt-*, o1*, o3\*, o4\*, chatgpt-*, codex-* (e.g., 'gpt-5.5',
  'gpt-5.4-mini')

- Anthropic: claude-\* (e.g., 'claude-opus-4-7', 'claude-sonnet-4-6')

- DeepSeek: deepseek-\* (e.g., 'deepseek-v4-flash', 'deepseek-v4-pro')

- Google: gemini-\* (e.g., 'gemini-3.1-pro-preview',
  'gemini-3-flash-preview')

- Qwen: qwen\*, qwq-\* (e.g., 'qwen3.6-plus', 'qwen3.6-flash')

- Stepfun: step-\* (e.g., 'step-3.5-flash', 'step-3')

- Zhipu: glm-*, chatglm* (e.g., 'glm-5.1', 'glm-5-turbo')

- MiniMax: minimax-\* (e.g., 'MiniMax-M2.7', 'MiniMax-M2.5')

- Grok: grok-\* (e.g., 'grok-4.3', 'grok-4.3-latest')

- Kimi (Moonshot AI Open Platform): kimi-*, moonshot-* (e.g.,
  'kimi-k2.6', 'moonshot-v1-8k')

- OpenRouter: Any model with '/' in the name (e.g., 'openai/gpt-5.5',
  'anthropic/claude-opus-4.7')
