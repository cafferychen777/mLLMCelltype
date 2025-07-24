# Using Free OpenRouter Models with mLLMCelltype

Hi @mhbsiam,

I've investigated the OpenRouter API errors you're encountering. The issue is with the model IDs you're trying to use. Here's what's happening and how to fix it:

## The Problem

1. For free OpenRouter models, you need to add the `:free` suffix to the model ID.
2. The model IDs need to be in the correct format.

Your attempts:
- `microsoft/mai-ds-r1` → Should be `microsoft/mai-ds-r1:free`
- `qwen/qwq-32b-free` → Should be `qwen/qwq-32b:free` (note the colon, not hyphen)

## The Solution

Here's how to use free OpenRouter models correctly:

```python
consensus_results = interactive_consensus_annotation(
    marker_genes=marker_genes,
    species="human",
    models=[
        {"provider": "openrouter", "model": "meta-llama/llama-4-maverick:free"},
        {"provider": "openrouter", "model": "nvidia/llama-3.1-nemotron-ultra-253b-v1:free"}
    ],
    api_keys={
        "openrouter": "your-openrouter-api-key"
    },
    max_discussion_rounds=3
)
```

## Recommended Free Models

Here are some high-quality free models with large parameter counts that are currently available on OpenRouter (updated July 2025):

### Large Language Models (Sorted by Size/Features)
- `qwen/qwen3-coder:free` - 480B parameters (35B active), context: 262K - Code generation model
- `qwen/qwen3-235b-a22b-07-25:free` - 235B parameters (22B active), context: 262K - General purpose
- `moonshotai/kimi-k2:free` - 1T parameters (32B active), context: 65K - MoE model with tool use
- `tencent/hunyuan-a13b-instruct:free` - 80B parameters (13B active), context: 32K - MoE with reasoning
- `tngtech/deepseek-r1t2-chimera:free` - 671B parameters, context: 163K - Reasoning model
- `google/gemma-3n-e2b-it:free` - 6B parameters (2B effective), context: 8K - Multimodal
- `cognitivecomputations/dolphin-mistral-24b-venice-edition:free` - 24B parameters, context: 32K - Uncensored

### Latest Models (July 2025)
- `qwen/qwen3-coder:free` - Latest code generation model (July 2025)
- `qwen/qwen3-235b-a22b-07-25:free` - Latest general purpose model (July 2025)
- `moonshotai/kimi-k2:free` - Advanced MoE model with tool capabilities (July 2025)
- `tencent/hunyuan-a13b-instruct:free` - Reasoning-capable MoE model (July 2025)
- `tngtech/deepseek-r1t2-chimera:free` - Second-generation reasoning model (July 2025)
- `google/gemma-3n-e2b-it:free` - Latest multimodal Gemma model (July 2025)

## Checking Available Models

You can check the current list of available models with this code:

```python
import requests
import os

# Get your OpenRouter API key
api_key = "your-openrouter-api-key"

# Get available models
response = requests.get(
    "https://openrouter.ai/api/v1/models",
    headers={"Authorization": f"Bearer {api_key}"}
)

# Print all free models
models = response.json()["data"]
print("Available free OpenRouter models:")
for model in models:
    model_id = model["id"]
    is_free = model.get("pricing", {}).get("prompt") == 0 and model.get("pricing", {}).get("completion") == 0
    if is_free:
        print(f"  - {model_id}")
```

I've updated our documentation to make this clearer for future users. Let me know if you have any other questions!
