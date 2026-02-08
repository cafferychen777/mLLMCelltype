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

  Character string specifying the model name (e.g., "gpt-5.2",
  "claude-sonnet-4.5").

## Value

Character string of the provider name (e.g., "openai", "anthropic").

## Details

Supported providers and model prefixes:

- OpenAI: gpt-*, o1*, o3\*, o4\*, chatgpt-*, codex-* (e.g., 'gpt-5.2',
  'o3-pro', 'o4-mini')

- Anthropic: claude-\* (e.g., 'claude-opus-4.6', 'claude-sonnet-4.5')

- DeepSeek: deepseek-\* (e.g., 'deepseek-chat', 'deepseek-r1')

- Google: gemini-\* (e.g., 'gemini-3-pro', 'gemini-2.5-flash')

- Qwen: qwen\*, qwq-\* (e.g., 'qwen3-max', 'qwq-32b')

- Stepfun: step-\* (e.g., 'step-2-mini', 'step-2-16k')

- Zhipu: glm-*, chatglm* (e.g., 'glm-4.7', 'glm-4-plus')

- MiniMax: minimax-\* (e.g., 'minimax-m2.1', 'minimax-m1')

- Grok: grok-\* (e.g., 'grok-4', 'grok-4-heavy')

- OpenRouter: Any model with '/' in the name (e.g., 'openai/gpt-5.2',
  'anthropic/claude-sonnet-4.5')
